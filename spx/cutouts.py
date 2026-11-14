"""Throttled cutout download with a partial-stream fast path, FITS parsing, wavelength, cache.

A SPHEREx cutout is ~4 MB, but 97% of it is a PSF table (EPSF) we do not need. We stream
the response, parse FITS headers block by block and close the connection as soon as the
EPSF header appears. The wavelength table (WCS-WAVE, after EPSF) is identical for all
exposures of one detector (verified 2026-11-14), so we download it once per detector.
"""
from __future__ import annotations

import hashlib
import io
import json
import random
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.coordinates import SkyCoord
import astropy.units as u

from . import config
from .archive import Exposure

warnings.simplefilter("ignore", FITSFixedWarning)
from astropy import log as _astropy_log
_astropy_log.setLevel("WARNING")  # SPHEREx headers trigger a harmless SIP notice
BLOCK = 2880
_SEM = threading.BoundedSemaphore(config.MAX_CONCURRENT)
_WAVE_LOCK = threading.Lock()


class HeadParser:
    """Feed FITS bytes; stops at the header of HDU `stop` (EXTNAME). `prefix` then holds a
    complete, valid FITS file with all HDUs before it."""

    def __init__(self, stop: str = "EPSF"):
        self.stop = stop
        self.buf = bytearray()
        self.pos = 0           # start of the next HDU in buf
        self.done = False
        self.cut = None        # byte length of the useful prefix

    def feed(self, chunk: bytes) -> bool:
        if self.done:
            return True
        self.buf += chunk
        while True:
            # find END card within the header starting at self.pos
            hdr_end = None
            p = self.pos
            while p + BLOCK <= len(self.buf):
                block = self.buf[p:p + BLOCK]
                for i in range(0, BLOCK, 80):
                    if block[i:i + 8] == b"END     ":
                        hdr_end = p + BLOCK
                        break
                if hdr_end:
                    break
                p += BLOCK
            if not hdr_end:
                return False
            hdr = fits.Header.fromstring(bytes(self.buf[self.pos:hdr_end]))
            if hdr.get("EXTNAME", "").strip().upper() == self.stop:
                self.cut = self.pos
                self.done = True
                return True
            n = 1 if hdr.get("NAXIS", 0) else 0
            for k in range(1, hdr.get("NAXIS", 0) + 1):
                n *= hdr[f"NAXIS{k}"]
            size = abs(hdr["BITPIX"]) // 8 * hdr.get("GCOUNT", 1) * (n + hdr.get("PCOUNT", 0)) if n else 0
            size = -(-size // BLOCK) * BLOCK
            if hdr_end + size > len(self.buf):
                return False
            self.pos = hdr_end + size

    @property
    def prefix(self) -> bytes:
        return bytes(self.buf[: self.cut if self.cut is not None else self.pos])


def _get_with_retry(url: str, stream: bool, get=requests.get, sleep=time.sleep):
    last = None
    for attempt in range(config.RETRIES):
        wait = min(12, 1.5 * 2 ** attempt)
        try:
            r = get(url, stream=stream, timeout=config.REQUEST_TIMEOUT)
        except requests.RequestException as e:
            last = str(e)
        else:
            if r.status_code not in (429, 502, 503, 504):
                r.raise_for_status()
                return r
            last = f"HTTP {r.status_code}"
            try:  # IRSA answers 503 "Server busy" with Retry-After: 5
                wait = float(getattr(r, "headers", {}).get("retry-after", wait))
            except (TypeError, ValueError):
                pass
            r.close()
        sleep(wait * (0.7 + 0.6 * random.random()))
    raise RuntimeError(f"archive busy ({last})")


def fetch_prefix(url: str, get=requests.get, sleep=time.sleep) -> bytes:
    """Download only the HDUs before EPSF."""
    with _SEM:
        r = _get_with_retry(url, True, get, sleep)
        p = HeadParser()
        try:
            for chunk in r.iter_content(65536):
                if p.feed(chunk):
                    break
        finally:
            r.close()
        if not p.done:
            raise RuntimeError("cutout ended before EPSF (truncated response)")
        return p.prefix


def fetch_full(url: str) -> bytes:
    with _SEM:
        r = _get_with_retry(url, False)
        return r.content


def cutout_url(exp: Exposure, ra: float, dec: float, size_arcmin: float) -> str:
    return f"{exp.url}?center={ra:.6f},{dec:.6f}&size={size_arcmin:.2f}arcmin"


# --- wavelength table per detector ----------------------------------------------------
def _wave_path(det: int, version: str):
    major = version.split("-")[1] if version.count("-") >= 1 else "v"
    return config.cache_dir() / f"wcswave_D{det}_{major}.fits"


def wave_table(exp: Exposure, url: str) -> fits.BinTableHDU:
    path = _wave_path(exp.detector, exp.version)
    with _WAVE_LOCK:
        if not path.exists():
            hd = fits.open(io.BytesIO(fetch_full(url)))
            fits.HDUList([fits.PrimaryHDU(), hd["WCS-WAVE"].copy()]).writeto(path, overwrite=True)
    return fits.open(path)[1]


def has_wave_table(exp: Exposure) -> bool:
    return _wave_path(exp.detector, exp.version).exists()


def image_wcs(header: fits.Header) -> WCS:
    h = header.copy()
    if "A_ORDER" in h:
        for k in ("CTYPE1", "CTYPE2"):
            if not h[k].endswith("-SIP"):
                h[k] = h[k] + "-SIP"
    return WCS(h)


def wavelength_at(header: fits.Header, table: fits.BinTableHDU, x: float, y: float):
    hl = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(header=header), table])
    sw = WCS(header, fobj=hl, key="W")
    sw.sip = None
    wl, bw = sw.pixel_to_world(x, y)
    return float(wl.to_value(u.um)), float(bw.to_value(u.um))


# --- stamps -------------------------------------------------------------------------
@dataclass
class Stamp:
    band: str
    mjd: float
    mjd_end: float
    uri: str
    version: str
    image: np.ndarray       # MJy/sr
    flags: np.ndarray       # int32 bitmask
    variance: np.ndarray
    zodi: np.ndarray
    header: str             # IMAGE header text
    flag_bits: dict = field(default_factory=dict)  # name -> bitmask
    wavelength: float = float("nan")  # um at cutout centre
    bandwidth: float = float("nan")

    @property
    def wcs(self) -> WCS:
        return image_wcs(fits.Header.fromstring(self.header))

    def mask_bits(self, names) -> int:
        m = 0
        for n in names:
            m |= int(self.flag_bits.get(n, 0))
        return m

    def bad_mask(self, names=None) -> np.ndarray:
        m = self.mask_bits(names or config.BAD_FLAGS)
        return (self.flags.astype(np.int64) & m) != 0


def flag_bits_from_header(h: fits.Header) -> dict:
    out = {}
    for k in h:
        if k.startswith("MSKN"):
            out[str(h[k]).strip()] = int(h.get("MSKM" + k[4:], 1 << int(k[4:])))
    return out


def parse_stamp(raw: bytes, exp: Exposure, ra: float, dec: float, table=None) -> Stamp:
    hd = fits.open(io.BytesIO(raw))
    h = hd["IMAGE"].header
    st = Stamp(exp.band, exp.mjd, exp.mjd_end, exp.uri, exp.version,
               hd["IMAGE"].data.astype(np.float32), hd["FLAGS"].data.astype(np.int32),
               hd["VARIANCE"].data.astype(np.float32), hd["ZODI"].data.astype(np.float32),
               h.tostring(), flag_bits_from_header(hd["FLAGS"].header))
    if table is None and "WCS-WAVE" in [x.name for x in hd]:
        table = hd["WCS-WAVE"]
    if table is not None:
        x, y = image_wcs(h).world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        try:
            st.wavelength, st.bandwidth = wavelength_at(h, table, float(x), float(y))
        except Exception:  # wavelength is a nice-to-have for the blink
            pass
    return st


def _key(exp: Exposure, ra: float, dec: float, size: float) -> str:
    s = f"{config.CACHE_VERSION}|{exp.uri}|{ra:.6f}|{dec:.6f}|{size:.2f}"
    return hashlib.sha1(s.encode()).hexdigest()[:20]


def _save(path, st: Stamp):
    np.savez_compressed(path, image=st.image, flags=st.flags, variance=st.variance, zodi=st.zodi,
                        meta=np.array(json.dumps({
                            "band": st.band, "mjd": st.mjd, "mjd_end": st.mjd_end, "uri": st.uri,
                            "version": st.version, "header": st.header, "flag_bits": st.flag_bits,
                            "wavelength": st.wavelength, "bandwidth": st.bandwidth})))


def _load(path) -> Stamp:
    z = np.load(path)
    m = json.loads(str(z["meta"]))
    return Stamp(m["band"], m["mjd"], m["mjd_end"], m["uri"], m["version"], z["image"], z["flags"],
                 z["variance"], z["zodi"], m["header"], m["flag_bits"], m["wavelength"], m["bandwidth"])


def get_stamp(exp: Exposure, ra: float, dec: float, size_arcmin: float) -> Stamp:
    path = config.cache_dir() / f"st_{_key(exp, ra, dec, size_arcmin)}.npz"
    if path.exists():
        try:
            return _load(path)
        except Exception:
            path.unlink(missing_ok=True)
    url = cutout_url(exp, ra, dec, size_arcmin)
    if has_wave_table(exp):
        st = parse_stamp(fetch_prefix(url), exp, ra, dec, wave_table(exp, url))
    else:  # first exposure of this detector: one full download, keep its wavelength table
        raw = fetch_full(url)
        hd = fits.open(io.BytesIO(raw))
        with _WAVE_LOCK:
            wp = _wave_path(exp.detector, exp.version)
            if not wp.exists():
                fits.HDUList([fits.PrimaryHDU(), hd["WCS-WAVE"].copy()]).writeto(wp, overwrite=True)
        st = parse_stamp(raw, exp, ra, dec)
    _save(path, st)
    return st


def is_cached(exp: Exposure, ra: float, dec: float, size_arcmin: float) -> bool:
    return (config.cache_dir() / f"st_{_key(exp, ra, dec, size_arcmin)}.npz").exists()


def fetch_many(exps: list[Exposure], ra: float, dec: float, size_arcmin: float, progress=None):
    """Returns (stamps sorted by time, number of failures)."""
    stamps, fails, done = [], 0, 0
    exps = list(exps)
    # one exposure per detector without a cached wavelength table goes first, alone
    firsts = {}
    for e in exps:
        if not has_wave_table(e) and not is_cached(e, ra, dec, size_arcmin):
            firsts.setdefault(e.detector, e)
    for e in firsts.values():
        exps.remove(e)
        done += 1
        try:
            stamps.append(get_stamp(e, ra, dec, size_arcmin))
        except Exception:
            fails += 1
        if progress:
            progress(done, len(exps) + done)
    total = len(exps) + done
    with ThreadPoolExecutor(config.MAX_CONCURRENT) as ex:
        futs = {ex.submit(get_stamp, e, ra, dec, size_arcmin): e for e in exps}
        for f in as_completed(futs):
            done += 1
            try:
                stamps.append(f.result())
            except Exception:
                fails += 1
            if progress:
                progress(done, total)
    stamps.sort(key=lambda s: s.mjd)
    return stamps, fails

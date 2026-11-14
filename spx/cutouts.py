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
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

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
    m = re.search(r"-(v\d+)-", f"-{version}-")
    major = m.group(1) if m else "v"
    name = f"wcswave_D{det}_{major}.fits"
    shipped = Path(__file__).parent / "data" / name  # tables shipped with the app (8.6 kB each)
    return shipped if shipped.exists() else config.cache_dir() / name


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


# --- fast path: byte ranges from the AWS copy of the archive --------------------------
_S3_SEM = threading.BoundedSemaphore(config.S3_CONCURRENT)
_S3 = requests.Session()
_S3.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=config.S3_CONCURRENT + 8))
_LAYOUTS: dict = {}  # (version, detector, image header bytes) -> layout of the HDUs after IMAGE
_LAYOUT_LOCK = threading.Lock()
_POOL = ThreadPoolExecutor(config.S3_CONCURRENT)


def s3_url(exp: Exposure) -> str | None:
    part = exp.uri.split("spherex/", 1)
    return f"{config.S3_BASE}/{part[1]}" if len(part) == 2 else None


def _range(url: str, start: int, stop: int) -> bytes:
    """Bytes [start, stop) of a file."""
    last = None
    for attempt in range(4):
        try:
            with _S3_SEM:
                r = _S3.get(url, headers={"Range": f"bytes={start}-{stop - 1}"}, timeout=30)
            if r.status_code == 206:
                return r.content
            last = f"HTTP {r.status_code}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        except requests.RequestException as e:
            last = type(e).__name__
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"S3 read failed ({last})")


def _header_len(buf: bytes, start: int = 0):
    p = start
    while p + BLOCK <= len(buf):
        blk = buf[p:p + BLOCK]
        if any(blk[i:i + 8] == b"END     " for i in range(0, BLOCK, 80)):
            return p + BLOCK - start
        p += BLOCK
    return None


def _data_len(h: fits.Header) -> int:
    naxis = h.get("NAXIS", 0)
    if not naxis:
        return 0
    n = 1
    for k in range(1, naxis + 1):
        n *= h[f"NAXIS{k}"]
    size = abs(h["BITPIX"]) // 8 * h.get("GCOUNT", 1) * (n + h.get("PCOUNT", 0))
    return -(-size // BLOCK) * BLOCK


def _header_at(url: str, off: int, guess: int = BLOCK * 10):
    buf = _range(url, off, off + guess)
    n = _header_len(buf)
    while n is None:
        buf += _range(url, off + len(buf), off + len(buf) + guess)
        n = _header_len(buf)
    return fits.Header.fromstring(buf[:n].decode("ascii", "replace")), n


def _layout_chain(url: str, off: int) -> dict:
    """Walk FLAGS, VARIANCE, ZODI headers one by one (3 round trips)."""
    lay = {}
    for name in ("FLAGS", "VARIANCE", "ZODI"):
        h, n = _header_at(url, off)
        if h.get("EXTNAME", "").strip().upper() != name:
            raise RuntimeError(f"unexpected HDU {h.get('EXTNAME')}")
        lay[name] = (off, n, h)
        off += n + _data_len(h)
    return lay


def _rice_rows(url: str, tab_off: int, h: fits.Header, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """Rows y0..y1 of a tile-compressed image stored one row per tile: read the row
    descriptors, then the matching slice of the heap, then decompress each row."""
    from astropy.io.fits.hdu.compressed._codecs import Rice1
    if h.get("ZTILE2", 1) != 1 or h.get("ZTILE1") != h.get("ZNAXIS1"):
        raise RuntimeError("unsupported tiling")
    q = "Q" in str(h["TFORM1"]).upper()
    rowlen = h["NAXIS1"]
    raw = _range(url, tab_off + y0 * rowlen, tab_off + y1 * rowlen)
    desc = np.frombuffer(raw, ">i8" if q else ">i4").reshape(y1 - y0, rowlen // (8 if q else 4))[:, :2]
    heap = tab_off + h.get("THEAP", h["NAXIS1"] * h["NAXIS2"])
    lo = int(desc[:, 1].min())
    hi = int((desc[:, 1] + desc[:, 0]).max())
    blob = _range(url, heap + lo, heap + hi)
    par = {str(h.get(f"ZNAME{i}", "")).upper(): h.get(f"ZVAL{i}") for i in range(1, 6)}
    codec = Rice1(blocksize=int(par.get("BLOCKSIZE") or 32), bytepix=int(par.get("BYTEPIX") or 4),
                  tilesize=int(h["ZTILE1"]))
    out = np.empty((y1 - y0, x1 - x0), np.int32)
    for i, (n, o) in enumerate(desc):
        out[i] = codec.decode(np.frombuffer(blob, np.uint8, int(n), int(o) - lo))[x0:x1]
    return out


def _looks_like_header_end(block: bytes) -> bool:
    return b"END     " in block and all(32 <= c < 127 for c in block)


def fetch_s3_stamp(exp: Exposure, ra: float, dec: float, size_arcmin: float, table=None) -> Stamp:
    """Read just the rows around (ra, dec) of IMAGE, FLAGS, VARIANCE and ZODI from the full file."""
    url = s3_url(exp)
    if not url:
        raise RuntimeError("no S3 path")
    # round trip 1: primary + IMAGE header
    buf = _range(url, 0, BLOCK * 16)
    n0 = _header_len(buf)
    while n0 is None:
        buf += _range(url, len(buf), len(buf) + BLOCK * 16)
        n0 = _header_len(buf)
    off = n0 + _data_len(fits.Header.fromstring(buf[:n0].decode("ascii", "replace")))
    n1 = _header_len(buf, off) if off < len(buf) else None
    if n1 is None:
        hi, n1 = _header_at(url, off)
    else:
        hi = fits.Header.fromstring(buf[off:off + n1].decode("ascii", "replace"))
    img_off = off + n1
    ny, nx = hi["NAXIS2"], hi["NAXIS1"]
    after_img = img_off + _data_len(hi)
    # pixel box (same size as an IRSA cutout of size_arcmin)
    x, y = image_wcs(hi).world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    half = int(np.ceil(size_arcmin * 60 / config.PIXEL_SCALE_ARCSEC / 2))
    x0, x1 = max(0, int(round(float(x))) - half), min(nx, int(round(float(x))) + half + 1)
    y0, y1 = max(0, int(round(float(y))) - half), min(ny, int(round(float(y))) + half + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise RuntimeError("position off the detector")
    def rows(data_off, h):
        if h.get("ZIMAGE"):  # FLAGS: RICE_1 tile-compressed, one tile per row
            return _rice_rows(url, data_off, h, y0, y1, x0, x1)
        rb = nx * abs(h["BITPIX"]) // 8
        raw = _range(url, data_off + y0 * rb, data_off + y1 * rb)
        dt = np.dtype(">f4") if h["BITPIX"] == -32 else np.dtype(">i4")
        return np.frombuffer(raw, dt).reshape(y1 - y0, nx)[:, x0:x1].astype(dt.newbyteorder("="))

    futs = {"IMAGE": _POOL.submit(rows, img_off, hi)}  # starts now, in parallel with the rest
    # Header lengths are fixed per detector and version, but the compressed FLAGS block has a
    # different size in every file, so its header (which gives that size) is read every time.
    key = (exp.version, exp.detector)
    lens = _LAYOUTS.get(key)
    if lens is None:  # 3 extra round trips, once per detector and processing version
        lay = _layout_chain(url, after_img)
        lens = tuple(lay[nm][1] for nm in ("FLAGS", "VARIANCE", "ZODI"))
        with _LAYOUT_LOCK:
            _LAYOUTS[key] = lens
        hf = lay["FLAGS"][2]
    else:
        hf, nf = _header_at(url, after_img, guess=lens[0])
        if nf != lens[0] or hf.get("EXTNAME", "").strip().upper() != "FLAGS":
            with _LAYOUT_LOCK:
                _LAYOUTS.pop(key, None)
            raise RuntimeError("layout changed")
    f_off = after_img + lens[0]
    v_off = f_off + _data_len(hf) + lens[1]
    z_off = v_off + _data_len(hi) + lens[2]  # VARIANCE has the same shape and type as IMAGE
    plain = {"BITPIX": -32}
    futs["FLAGS"] = _POOL.submit(rows, f_off, hf)
    futs["VARIANCE"] = _POOL.submit(rows, v_off, plain)
    futs["ZODI"] = _POOL.submit(rows, z_off, plain)
    checks = [_POOL.submit(_range, url, o - BLOCK, o) for o in (v_off, z_off)]  # header ends where expected?
    if not all(_looks_like_header_end(c.result()) for c in checks):
        with _LAYOUT_LOCK:
            _LAYOUTS.pop(key, None)
        raise RuntimeError("layout changed")
    arr = {k: f.result() for k, f in futs.items()}
    h = hi.copy()
    h["NAXIS1"], h["NAXIS2"] = x1 - x0, y1 - y0
    for k, d in (("CRPIX1", x0), ("CRPIX2", y0), ("CRPIX1W", x0), ("CRPIX2W", y0)):
        if k in h:
            h[k] = h[k] - d
    st = Stamp(exp.band, exp.mjd, exp.mjd_end, exp.uri, exp.version, arr["IMAGE"].astype(np.float32),
               arr["FLAGS"].astype(np.int32), arr["VARIANCE"].astype(np.float32), arr["ZODI"].astype(np.float32),
               h.tostring(), flag_bits_from_header(hf))
    if table is not None:
        cx, cy = image_wcs(h).world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        try:
            st.wavelength, st.bandwidth = wavelength_at(h, table, float(cx), float(cy))
        except Exception:
            pass
    return st


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
    st = None
    if has_wave_table(exp) and config.USE_S3:
        try:
            st = fetch_s3_stamp(exp, ra, dec, size_arcmin, wave_table(exp, url))
        except Exception:
            st = None  # fall back to the IRSA cutout service
    if st is not None:
        pass
    elif has_wave_table(exp):
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
    workers = max(config.MAX_CONCURRENT, config.S3_CONCURRENT // 2) if config.USE_S3 else config.MAX_CONCURRENT
    with ThreadPoolExecutor(workers) as ex:
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

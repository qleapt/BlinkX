"""Throttled cutout download with a partial-stream fast path, FITS parsing, wavelength, cache.

A SPHEREx cutout is ~4 MB, but 97% of it is a PSF table (EPSF) we do not need. We stream
the response, parse FITS headers block by block and close the connection as soon as the
EPSF header appears. The wavelength table (WCS-WAVE, after EPSF) is identical for all
exposures of one detector (verified 2026-11-14), so we download it once per detector.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import random
import re
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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


# The header cards a celestial WCS reads (82 of the 276 in a SPHEREx IMAGE header). A WCS built from these alone
# is identical to one built from the whole header (checked on 304 cached photos: parameters, distortion terms and
# pixel <-> sky to the last bit) and takes 3.6 ms instead of 9.9 ms.
_WCS_CARDS = re.compile(
    r"(WCSAXES|NAXIS\d*|CTYPE\d+|CUNIT\d+|CRVAL\d+|CRPIX\d+|CDELT\d+|CROTA\d+|PC\d+_\d+|CD\d+_\d+|PV\d+_\d+|PS\d+_\d+"
    r"|CNAME\d+|CRDER\d+|CSYER\d+|CZPHS\d+|CPERI\d+|LONPOLE|LATPOLE|RESTFRQ|RESTFREQ|RESTWAV|RADESYS|RADECSYS|EQUINOX"
    r"|EPOCH|VELREF|SPECSYS|SSYSOBS|SSYSSRC|VELOSYS|ZSOURCE|VELANGL|WCSNAME|DATE-OBS|DATE-BEG|DATE-AVG|DATE-END"
    r"|DATEREF|DATE|MJD-OBS|MJD-BEG|MJD-AVG|MJD-END|MJDOBS|MJDREF|MJDREFI|MJDREFF|JDREF|JDREFI|JDREFF|TIMESYS"
    r"|TREFPOS|TREFDIR|PLEPHEM|TIMEUNIT|TIMEOFFS|TSTART|TSTOP|XPOSURE|TELAPSE|TIMSYER|TIMRDER|TIMEDEL|TIMEPIXR"
    r"|OBSGEO-[XYZLBH]|OBSORBIT|A_ORDER|B_ORDER|AP_ORDER|BP_ORDER|A_\d+_\d+|B_\d+_\d+|AP_\d+_\d+|BP_\d+_\d+|A_DMAX"
    r"|B_DMAX|CPDIS\d+|CQDIS\d+|DP\d+|DQ\d+|CPERR\d+|CQERR\d+|D2IM\w*|XTENSION|SIMPLE|BITPIX|EXTNAME|EXTVER|END)")


def build_wcs(text: str) -> WCS:
    """image_wcs() of a header given as FITS text (80-character cards), reading only the cards a WCS uses."""
    if len(text) % 80:
        return image_wcs(fits.Header.fromstring(text))
    cards = (text[i:i + 80] for i in range(0, len(text), 80))
    return image_wcs(fits.Header.fromstring("".join(c for c in cards if _WCS_CARDS.fullmatch(c[:8].rstrip()))))


_WCS_MEMO: OrderedDict = OrderedDict()   # header text (sha1) -> WCS; copies are handed out (0.05 ms each)
_WCS_LOCK = threading.Lock()


def header_wcs(text: str) -> WCS:
    """The WCS of a photo's header text, built once per header (a band seen again, a line-up redone with more
    photos, a measurement) and handed out as a copy, so threads never share one."""
    key = hashlib.sha1(text.encode("ascii", "replace")).digest()
    with _WCS_LOCK:
        w = _WCS_MEMO.get(key)
        if w is not None:
            _WCS_MEMO.move_to_end(key)
    if w is None:
        w = build_wcs(text)
        with _WCS_LOCK:
            _WCS_MEMO[key] = w
            while len(_WCS_MEMO) > 4096:
                _WCS_MEMO.popitem(last=False)
    return copy.deepcopy(w)


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
        """The image WCS, built once per stamp and kept (read-only use); see header_wcs()."""
        w = self.__dict__.get("_wcs")
        if w is None:
            w = self.__dict__["_wcs"] = header_wcs(self.header)
        return w

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
# (major version, detector) -> header lengths of FLAGS, VARIANCE, ZODI and the bytes from the ZODI header to the
# end of the file. Constant for all files of one detector and processing version (checked on every read), so
# VARIANCE and ZODI can be read right after the IMAGE header. Kept on disk too (s3_layout.json).
_LAYOUTS: dict = {}
_LAYOUT_LOCK = threading.Lock()
_LAYOUTS_READ = [False]
_POOL = ThreadPoolExecutor(config.S3_CONCURRENT)


@dataclass(frozen=True)
class _Lane:
    """Connections, request limit and row-reading threads. The page the user waits for has its own lane; photos
    loaded ahead for /RTP use a small separate one, so they never hold the user's connections."""
    session: requests.Session
    sem: threading.BoundedSemaphore
    pool: ThreadPoolExecutor


def _session(size: int) -> requests.Session:
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=size))
    return s


_FG = _Lane(_S3, _S3_SEM, _POOL)
_BG = _Lane(_session(config.BG_S3_CONCURRENT + 4), threading.BoundedSemaphore(config.BG_S3_CONCURRENT),
            ThreadPoolExecutor(config.BG_S3_CONCURRENT + 4, thread_name_prefix="spx-bg-rows"))


def s3_url(exp: Exposure) -> str | None:
    part = exp.uri.split("spherex/", 1)
    return f"{config.S3_BASE}/{part[1]}" if len(part) == 2 else None


def _file_size(content_range: str) -> int:
    """Whole-file length from a Content-Range header ('bytes 0-99/66882240'), 0 when unknown."""
    try:
        return int(content_range.rsplit("/", 1)[1])
    except (IndexError, ValueError):
        return 0


def _range(url: str, start: int, stop: int, size: list | None = None, lane: _Lane | None = None) -> bytes:
    """Bytes [start, stop) of a file. size: a list that receives the whole file's length."""
    lane = lane or _FG
    last = None
    for attempt in range(4):
        try:
            with lane.sem:
                r = lane.session.get(url, headers={"Range": f"bytes={start}-{stop - 1}"}, timeout=30)
            if r.status_code == 206:
                if size is not None:
                    size.append(_file_size(r.headers.get("Content-Range", "")))
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


def _header_at(url: str, off: int, guess: int = BLOCK * 10, lane: _Lane | None = None):
    buf = _range(url, off, off + guess, lane=lane)
    n = _header_len(buf)
    while n is None:
        buf += _range(url, off + len(buf), off + len(buf) + guess, lane=lane)
        n = _header_len(buf)
    return fits.Header.fromstring(buf[:n].decode("ascii", "replace")), n


def _layout_chain(url: str, off: int, lane: _Lane | None = None) -> dict:
    """Walk FLAGS, VARIANCE, ZODI headers one by one (3 round trips)."""
    lay = {}
    for name in ("FLAGS", "VARIANCE", "ZODI"):
        h, n = _header_at(url, off, lane=lane)
        if h.get("EXTNAME", "").strip().upper() != name:
            raise RuntimeError(f"unexpected HDU {h.get('EXTNAME')}")
        lay[name] = (off, n, h)
        off += n + _data_len(h)
    return lay


def _layout_key(exp: Exposure) -> str:
    m = re.search(r"-(v\d+)-", f"-{exp.version}-")
    return f"{m.group(1) if m else 'v'}|{exp.detector}"


def _layout_path():
    return config.cache_dir() / "s3_layout.json"


def _layout(key: str):
    with _LAYOUT_LOCK:
        if not _LAYOUTS_READ[0]:   # what earlier sessions learned
            _LAYOUTS_READ[0] = True
            try:
                for k, v in json.loads(_layout_path().read_text()).items():
                    _LAYOUTS.setdefault(k, tuple(int(x) for x in v))
            except (OSError, ValueError, TypeError, AttributeError):
                pass
        return _LAYOUTS.get(key)


def _learned(key: str, lay: tuple | None) -> None:
    """Remember (or, with None, forget) the layout of one detector and version, here and on disk."""
    with _LAYOUT_LOCK:
        if _LAYOUTS.get(key) == lay:
            return
        if lay is None:
            _LAYOUTS.pop(key, None)
        else:
            _LAYOUTS[key] = lay
        try:
            _layout_path().write_text(json.dumps({k: list(v) for k, v in _LAYOUTS.items()}))
        except OSError:
            pass


def _rice_rows(url: str, tab_off: int, h: fits.Header, y0: int, y1: int, x0: int, x1: int,
               lane: _Lane | None = None, table: bytes | None = None) -> np.ndarray:
    """Rows y0..y1 of a tile-compressed image stored one row per tile: read the row
    descriptors (unless `table`, the bytes from the start of the row table, already holds them),
    then the matching slice of the heap, then decompress each row."""
    from astropy.io.fits.hdu.compressed._codecs import Rice1
    if h.get("ZTILE2", 1) != 1 or h.get("ZTILE1") != h.get("ZNAXIS1"):
        raise RuntimeError("unsupported tiling")
    q = "Q" in str(h["TFORM1"]).upper()
    rowlen = h["NAXIS1"]
    if table is not None and len(table) >= y1 * rowlen:
        raw = table[y0 * rowlen:y1 * rowlen]
    else:
        raw = _range(url, tab_off + y0 * rowlen, tab_off + y1 * rowlen, lane=lane)
    desc = np.frombuffer(raw, ">i8" if q else ">i4").reshape(y1 - y0, rowlen // (8 if q else 4))[:, :2]
    heap = tab_off + h.get("THEAP", h["NAXIS1"] * h["NAXIS2"])
    lo = int(desc[:, 1].min())
    hi = int((desc[:, 1] + desc[:, 0]).max())
    blob = _range(url, heap + lo, heap + hi, lane=lane)
    par = {str(h.get(f"ZNAME{i}", "")).upper(): h.get(f"ZVAL{i}") for i in range(1, 6)}
    codec = Rice1(blocksize=int(par.get("BLOCKSIZE") or 32), bytepix=int(par.get("BYTEPIX") or 4),
                  tilesize=int(h["ZTILE1"]))
    out = np.empty((y1 - y0, x1 - x0), np.int32)
    for i, (n, o) in enumerate(desc):
        out[i] = codec.decode(np.frombuffer(blob, np.uint8, int(n), int(o) - lo))[x0:x1]
    return out


def _looks_like_header_end(block: bytes) -> bool:
    return b"END     " in block and all(32 <= c < 127 for c in block)


class _LayoutChanged(RuntimeError):
    """A file is not laid out as learned (e.g. a new processing version): the layout is forgotten."""


def fetch_s3_stamp(exp: Exposure, ra: float, dec: float, size_arcmin: float, table=None,
                   lane: _Lane | None = None) -> Stamp:
    """Read just the rows around (ra, dec) of IMAGE, FLAGS, VARIANCE and ZODI from the full file, in three
    round trips: (1) the IMAGE header and the file's length; (2) the rows of IMAGE, VARIANCE and ZODI and the
    FLAGS header with its row table, all at once; (3) the compressed FLAGS rows. VARIANCE and ZODI can start
    in (2) because they end a fixed distance before the end of the file (learned once per detector and
    version); the FLAGS header read in (2) checks that place, so a different file is still read correctly.
    A file laid out differently than learned is read again at once, learning its layout from its headers."""
    try:
        return _read_s3(exp, ra, dec, size_arcmin, table, lane or _FG)
    except _LayoutChanged:
        return _read_s3(exp, ra, dec, size_arcmin, table, lane or _FG)


def _read_s3(exp: Exposure, ra: float, dec: float, size_arcmin: float, table, lane: _Lane) -> Stamp:
    url = s3_url(exp)
    if not url:
        raise RuntimeError("no S3 path")
    # round trip 1: primary + IMAGE header
    total = []
    buf = _range(url, 0, BLOCK * 16, total, lane=lane)
    n0 = _header_len(buf)
    while n0 is None:
        buf += _range(url, len(buf), len(buf) + BLOCK * 16, lane=lane)
        n0 = _header_len(buf)
    off = n0 + _data_len(fits.Header.fromstring(buf[:n0].decode("ascii", "replace")))
    n1 = _header_len(buf, off) if off < len(buf) else None
    if n1 is None:
        hi, n1 = _header_at(url, off, lane=lane)
        hi_text = hi.tostring()
    else:
        hi_text = buf[off:off + n1].decode("ascii", "replace")
        hi = fits.Header.fromstring(hi_text)
    img_off = off + n1
    ny, nx = hi["NAXIS2"], hi["NAXIS1"]
    after_img = img_off + _data_len(hi)
    # pixel box (same size as an IRSA cutout of size_arcmin)
    x, y = build_wcs(hi_text).world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    half = int(np.ceil(size_arcmin * 60 / config.PIXEL_SCALE_ARCSEC / 2))
    x0, x1 = max(0, int(round(float(x))) - half), min(nx, int(round(float(x))) + half + 1)
    y0, y1 = max(0, int(round(float(y))) - half), min(ny, int(round(float(y))) + half + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise RuntimeError("position off the detector")

    def rows(data_off, h):  # uncompressed rows y0..y1, cut to columns x0..x1
        rb = nx * abs(h["BITPIX"]) // 8
        raw = _range(url, data_off + y0 * rb, data_off + y1 * rb, lane=lane)
        dt = np.dtype(">f4") if h["BITPIX"] == -32 else np.dtype(">i4")
        return np.frombuffer(raw, dt).reshape(y1 - y0, nx)[:, x0:x1].astype(dt.newbyteorder("="))

    pool, plain, size = lane.pool, {"BITPIX": -32}, (total[0] if total else 0)
    futs = {"IMAGE": pool.submit(rows, img_off, hi)}  # starts now, in parallel with the rest
    key = _layout_key(exp)
    lay = _layout(key)
    if lay is None or not size or not lay[3]:  # 3 extra round trips, once per detector and processing version
        chain = _layout_chain(url, after_img, lane=lane)
        lay = tuple(chain[nm][1] for nm in ("FLAGS", "VARIANCE", "ZODI")) + (
            size - chain["ZODI"][0] if size else 0,)
        _learned(key, lay)
    lf, lv, lz, tail = lay
    # round trip 2: the FLAGS header with its row table (up to row y1; descriptors are at most 16 bytes) ...
    head = pool.submit(_range, url, after_img, after_img + lf + 16 * y1, lane=lane)
    v_off = z_off = None
    if size and tail:  # ... and VARIANCE and ZODI: VARIANCE (same shape as IMAGE) ends where ZODI's header starts
        z_head = size - tail
        v_off, z_off = z_head - _data_len(hi), z_head + lz
        futs["VARIANCE"] = pool.submit(rows, v_off, plain)
        futs["ZODI"] = pool.submit(rows, z_off, plain)
    checks = [pool.submit(_range, url, o - BLOCK, o, lane=lane) for o in (v_off, z_off) if o]
    hb = head.result()
    hf = fits.Header.fromstring(hb[:lf].decode("ascii", "replace"))
    if _header_len(hb[:lf]) != lf or hf.get("EXTNAME", "").strip().upper() != "FLAGS":
        _learned(key, None)
        raise _LayoutChanged("layout changed")
    v_true = after_img + lf + _data_len(hf) + lv   # where VARIANCE starts according to the FLAGS header
    if v_true != v_off:  # this file ends differently: read VARIANCE and ZODI at the right place, remember it
        v_off, z_off = v_true, v_true + _data_len(hi) + lz
        futs["VARIANCE"] = pool.submit(rows, v_off, plain)
        futs["ZODI"] = pool.submit(rows, z_off, plain)
        checks = [pool.submit(_range, url, o - BLOCK, o, lane=lane) for o in (v_off, z_off)]
        if size:
            _learned(key, (lf, lv, lz, size - (v_off + _data_len(hi))))
    # round trip 3: the compressed FLAGS rows
    flags = _rice_rows(url, after_img + lf, hf, y0, y1, x0, x1, lane=lane, table=hb[lf:])
    if not all(_looks_like_header_end(c.result()) for c in checks):  # header ends where expected?
        _learned(key, None)
        raise _LayoutChanged("layout changed")
    arr = {k: f.result() for k, f in futs.items()}
    h = hi.copy()
    h["NAXIS1"], h["NAXIS2"] = x1 - x0, y1 - y0
    for k, d in (("CRPIX1", x0), ("CRPIX2", y0), ("CRPIX1W", x0), ("CRPIX2W", y0)):
        if k in h:
            h[k] = h[k] - d
    st = Stamp(exp.band, exp.mjd, exp.mjd_end, exp.uri, exp.version, arr["IMAGE"].astype(np.float32),
               flags.astype(np.int32), arr["VARIANCE"].astype(np.float32), arr["ZODI"].astype(np.float32),
               h.tostring(), flag_bits_from_header(hf))
    if table is not None:
        cx, cy = st.wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))   # the WCS the stamp keeps
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
        x, y = st.wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))   # the WCS the stamp keeps
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


_HEDGE_POOL = ThreadPoolExecutor(48)


class Route:
    """Remembers which copy answered first lately (AWS or IRSA), so slow links stop waiting."""

    def __init__(self):
        self.prefer_irsa = False


_ROUTE = Route()


def race(s3, irsa, hedge_after: float = config.HEDGE_AFTER, route: Route = _ROUTE):
    """Read from AWS; if it has not answered within hedge_after seconds, ask IRSA as well and keep
    whichever arrives first. Once IRSA has won, ask both straight away until AWS wins again."""
    fs = _HEDGE_POOL.submit(s3)
    if not route.prefer_irsa:
        wait([fs], timeout=hedge_after)
        if fs.done() and fs.exception() is None:
            return fs.result()
    fi = _HEDGE_POOL.submit(irsa)
    pending, last = {fs, fi}, None
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for f in sorted(done, key=lambda f: f is fi):
            if f.exception() is None:
                route.prefer_irsa = f is fi
                return f.result()
            last = f.exception()
    raise last


_INFLIGHT: dict = {}   # cache path -> Future of a download in progress
_INFLIGHT_LOCK = threading.Lock()


def get_stamp(exp: Exposure, ra: float, dec: float, size_arcmin: float, quiet: bool = False) -> Stamp:
    """One cutout: from this computer's cache, else downloaded and cached. The same photo asked for twice at the
    same time (a /RTP spot loaded ahead, then used) is downloaded once: the second caller waits for the first.
    quiet: loaded ahead in the background (AWS only, on the small background lane)."""
    path = config.cache_dir() / f"st_{_key(exp, ra, dec, size_arcmin)}.npz"
    while True:
        if path.exists():
            try:
                return _load(path)
            except Exception:
                path.unlink(missing_ok=True)
        with _INFLIGHT_LOCK:
            other = _INFLIGHT.get(path)
            if other is None:
                mine = _INFLIGHT[path] = Future()
        if other is None:
            break
        try:
            other.result()   # on its way already: then it is in the cache (or it failed: then try here)
        except Exception:
            pass
    try:
        st = _download(exp, ra, dec, size_arcmin, quiet)
        _save(path, st)
    except BaseException as e:
        with _INFLIGHT_LOCK:
            _INFLIGHT.pop(path, None)
        mine.set_exception(e)
        raise
    with _INFLIGHT_LOCK:
        _INFLIGHT.pop(path, None)
    mine.set_result(None)
    return st


def _download(exp: Exposure, ra: float, dec: float, size_arcmin: float, quiet: bool = False) -> Stamp:
    url = cutout_url(exp, ra, dec, size_arcmin)
    if has_wave_table(exp):
        table = wave_table(exp, url)

        def irsa():
            return parse_stamp(fetch_prefix(url), exp, ra, dec, table)
        if quiet:  # in the background: AWS only, few connections, never IRSA's few slots
            if not config.USE_S3:
                raise RuntimeError("background loading needs the AWS copy")
            return fetch_s3_stamp(exp, ra, dec, size_arcmin, table, lane=_BG)
        if config.USE_S3:
            # The AWS copy is fast on a good connection but reads whole detector rows (~10x the
            # bytes); the IRSA cutout is small. race() hedges between them and remembers the winner.
            return race(lambda: fetch_s3_stamp(exp, ra, dec, size_arcmin, table), irsa)
        return irsa()
    if quiet:
        raise RuntimeError("the first photo of a detector is left for the foreground")
    # first exposure of this detector: one full download, keep its wavelength table
    raw = fetch_full(url)
    hd = fits.open(io.BytesIO(raw))
    with _WAVE_LOCK:
        wp = _wave_path(exp.detector, exp.version)
        if not wp.exists():
            fits.HDUList([fits.PrimaryHDU(), hd["WCS-WAVE"].copy()]).writeto(wp, overwrite=True)
    return parse_stamp(raw, exp, ra, dec)


def is_cached(exp: Exposure, ra: float, dec: float, size_arcmin: float) -> bool:
    return (config.cache_dir() / f"st_{_key(exp, ra, dec, size_arcmin)}.npz").exists()


_FETCH_POOL = ThreadPoolExecutor(max(config.MAX_CONCURRENT, config.S3_CONCURRENT // 2))
_BG_FETCH_POOL = ThreadPoolExecutor(config.BG_STAMPS, thread_name_prefix="spx-bg")
_FG_BUSY = [0]                  # foreground downloads running (a page the user is waiting for)
_FG_IDLE = threading.Condition()
_WARM_POOL = ThreadPoolExecutor(config.S3_CONCURRENT, thread_name_prefix="spx-warm")


class Stopped(Exception):
    """A background download that was called off (its /RTP spot was used, or the view settings changed)."""


def warm(n: int = config.S3_CONCURRENT) -> None:
    """Open n connections to the AWS copy now, in the background, so their set-up (TLS, ~0.5 s) overlaps the
    archive search instead of delaying the first photos. Tiny requests; returns at once."""
    for _ in range(n):
        _WARM_POOL.submit(_touch)


def _touch():
    try:
        _S3.head(f"{config.S3_BASE}/index.html", timeout=10)
    except Exception:
        pass


def _quiet_one(e: Exposure, ra: float, dec: float, size_arcmin: float, stop=None) -> Stamp:
    """One photo loaded ahead: it waits while a download the user is waiting for runs."""
    with _FG_IDLE:
        while _FG_BUSY[0]:
            _FG_IDLE.wait(0.5)
    if stop is not None and stop():
        raise Stopped()
    return get_stamp(e, ra, dec, size_arcmin, quiet=True)


def fetch_many(exps: list[Exposure], ra: float, dec: float, size_arcmin: float, progress=None,
               deadline: float | None = None, enough=None, stats: dict | None = None,
               background: bool = False, stop=None, first=None):
    """Returns (stamps sorted by time, number of failures).

    deadline / enough: after `deadline` seconds, return as soon as enough(stamps) is True. The
    downloads still running carry on in the background and land in the cache for the next view,
    so a slow connection never keeps the page waiting for the last few images. `stats['skipped']`
    receives how many were still downloading.

    first: uris loaded before the others (one photo per date), with the whole connection to themselves, so
    `enough` is reached as early as possible; the rest start as soon as they are in.

    background: photos loaded ahead (for /RTP) on a small separate lane, paused while a foreground download
    runs; stop() returning True calls the rest off. Photos that need IRSA are left for the foreground."""
    if background:
        todo = [e for e in exps if has_wave_table(e) or is_cached(e, ra, dec, size_arcmin)]
        futs = [_BG_FETCH_POOL.submit(_quiet_one, e, ra, dec, size_arcmin, stop) for e in todo]
        got = []
        for f in futs:
            try:
                got.append(f.result())
            except Exception:
                pass
        return sorted(got, key=lambda s: s.mjd), len(exps) - len(got)
    with _FG_IDLE:
        _FG_BUSY[0] += 1
    pending = ()
    try:
        stamps, fails, pending = _fetch_many(exps, ra, dec, size_arcmin, progress, deadline, enough, stats, first)
        return stamps, fails
    finally:
        _idle_after(pending)   # photos still arriving for the page shown early keep the background paused


def _idle_after(futs) -> None:
    """Count this foreground load as finished once all of futs are (now, when there are none)."""
    def release():
        with _FG_IDLE:
            _FG_BUSY[0] -= 1
            _FG_IDLE.notify_all()
    if not futs:
        release()
        return
    left, lock = [len(futs)], threading.Lock()

    def one(_):
        with lock:
            left[0] -= 1
            last = left[0] == 0
        if last:
            release()
    for f in list(futs):
        f.add_done_callback(one)


def _fetch_many(exps, ra, dec, size_arcmin, progress, deadline, enough, stats, first=None):
    t0 = time.time()
    stamps, fails, done = [], 0, 0
    exps = list(exps)
    total = len(exps)

    def drain(pending, may_stop):
        """Collect finished downloads; report progress every 0.5 s even when nothing arrived (heartbeat).
        may_stop: return right at the deadline once enough has arrived (the rest keep downloading into the cache)."""
        nonlocal done, fails
        while pending:
            soft = may_stop and deadline is not None
            if soft and time.time() - t0 > deadline and stamps and (enough is None or enough(stamps)):
                break
            left = t0 + deadline - time.time() if soft else 0.5
            finished, pending = wait(pending, timeout=min(0.5, left) if left > 0.02 else 0.5,
                                     return_when=FIRST_COMPLETED)
            for f in finished:
                done += 1
                if f.exception() is None:
                    stamps.append(f.result())
                else:
                    fails += 1
            if progress:
                progress(done, total)
        return pending

    # one exposure per detector without a cached wavelength table goes first (it brings the table along);
    # different detectors in parallel, then one photo per date (enough to show the view), then everything else
    firsts = {}
    for e in exps:
        if not has_wave_table(e) and not is_cached(e, ra, dec, size_arcmin):
            firsts.setdefault(e.detector, e)
    for e in firsts.values():
        exps.remove(e)
    drain({_FETCH_POOL.submit(get_stamp, e, ra, dec, size_arcmin) for e in firsts.values()}, False)

    def submit(es):   # started in this order (photos at hand are read in milliseconds, never behind a download)
        return {_FETCH_POOL.submit(get_stamp, e, ra, dec, size_arcmin) for e in es}
    at_hand = [e for e in exps if is_cached(e, ra, dec, size_arcmin)]
    lead = [e for e in exps if first and e.uri in first and e not in at_hand]
    later = [e for e in exps if e not in at_hand and e not in lead]
    # the photos at hand are always all read (milliseconds) before the page may be shown: only downloads are left
    # to arrive later. Returning while some were still being read showed part of a view that was all there, and
    # counted the rest as 'on their way', each costing the page another redraw.
    pending = set()
    if at_hand or lead:
        drain(submit(at_hand + lead), False)
    if later:
        pending = drain(submit(later), True)
    stamps.sort(key=lambda s: s.mjd)
    if stats is not None:
        stats["skipped"] = len(pending)
    return stamps, fails, pending

"""Motion engine: every mark of one object fitted together, the model chosen by the data, the object
classified from the evidence, and its path predicted into the future.

Positions are turned into offsets in arcseconds (east, north) from a reference point, and time into years
from the weighted mean date. Four models are tried, each a weighted least-squares fit to all marks at once:

* static            the object stands still                      x = x0
* linear            steady drift (proper motion)                 x = x0 + mu * t
* linear+parallax   drift plus the yearly wobble seen from Earth x = x0 + mu * t + plx * P(t)
* quadratic         a curved path (an orbit seen from Earth)     x = x0 + mu * t + acc * t^2 / 2

A richer model is kept only when it fits clearly better (a chi-square test scaled by the scatter that is
left, so marks that disagree with each other cannot fake a detection). Every mark is also checked against
the path drawn by the *other* marks (leave-one-out), and every pair of marks is compared directly.

classify() weighs the evidence (speed, time between the marks, consistency, parallax, curvature, ecliptic
latitude, catalogue match, what is left at the old position, single-exposure confirmation) and returns a
probability for each kind of object with the reasons, so the verdict can always be checked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import warnings

import numpy as np

try:   # dates years ahead have no leap-second table yet: harmless here
    from erfa import ErfaWarning
    warnings.filterwarnings("ignore", category=ErfaWarning)
except ImportError:  # pragma: no cover
    pass

YEAR = 365.25
FASTEST_STAR = 10.4      # "/yr, Barnard's Star
SYS_FLUX = 0.20          # brightness scatter between photos of one band that is not a real change
STAR_LIMIT = 20.0       # "/yr: faster than this is not a star (allows for an unknown nearby brown dwarf)
CLASSES = ("static", "pm_star", "solar_system", "mismatch")
CLASS_NAME = {
    "static": "Standing still: a background star or galaxy",
    "pm_star": "Moving star: a nearby star or brown dwarf with a steady proper motion",
    "solar_system": "Solar System object: an asteroid or comet",
    "mismatch": "Not one object: the marks do not belong together, or an artefact",
}
CLASS_SHORT = {"static": "Standing still", "pm_star": "Moving star", "solar_system": "Asteroid / comet",
               "mismatch": "Not one object"}


@dataclass(frozen=True)
class Obs:
    mjd: float
    ra: float        # deg
    dec: float       # deg
    err: float       # arcsec, 1 sigma per axis
    label: str = ""
    flux: float = float("nan")
    flux_err: float = float("nan")
    kind: str = "mark"   # mark | found (measured at the predicted spot) | catalogue


def parallax_factors(ra: float, dec: float, mjds):
    """Parallax shift (east, north) in arcsec per arcsec of parallax on each date, from Earth's position
    around the Sun (astropy's planetary ephemeris). SPHEREx orbits the Earth, so Earth's position is used."""
    from astropy.coordinates import get_body_barycentric
    from astropy.time import Time
    import astropy.units as u
    mjds = np.atleast_1d(np.asarray(mjds, float))
    xyz = get_body_barycentric("earth", Time(mjds, format="mjd")).xyz.to_value(u.au)
    x, y, z = (np.atleast_1d(v) for v in xyz)
    a, d = np.radians(ra), np.radians(dec)
    fe = x * np.sin(a) - y * np.cos(a)
    fn = x * np.cos(a) * np.sin(d) + y * np.sin(a) * np.sin(d) - z * np.cos(d)
    return fe, fn


def ecliptic_latitude(ra: float, dec: float) -> float:
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    return float(SkyCoord(ra * u.deg, dec * u.deg).barycentrictrueecliptic.lat.deg)


PARAMS = {"static": ["x0", "y0"], "linear": ["x0", "y0", "mu_e", "mu_n"],
          "linear+parallax": ["x0", "y0", "mu_e", "mu_n", "plx"],
          "quadratic": ["x0", "y0", "mu_e", "mu_n", "acc_e", "acc_n"]}


def _design(model: str, tau, fe, fn):
    """Rows for the east offsets, then rows for the north offsets."""
    n = len(tau)
    cols = []
    one, zero = np.ones(n), np.zeros(n)
    for p in PARAMS[model]:
        e, nn = {"x0": (one, zero), "y0": (zero, one), "mu_e": (tau, zero), "mu_n": (zero, tau),
                 "plx": (fe, fn), "acc_e": (0.5 * tau ** 2, zero), "acc_n": (zero, 0.5 * tau ** 2)}[p]
        cols.append(np.concatenate([e, nn]))
    return np.array(cols).T


def _lsq(model, tau, fe, fn, oe, on, err):
    A = _design(model, tau, fe, fn)
    y = np.concatenate([oe, on])
    w = 1 / np.concatenate([err, err]) ** 2
    AtWA = A.T @ (A * w[:, None])
    if np.linalg.cond(AtWA) > 1e12:
        return None
    cov = np.linalg.inv(AtWA)
    p = cov @ A.T @ (w * y)
    r = y - A @ p
    chi2 = float(np.sum(w * r ** 2))
    dof = len(y) - len(p)
    return dict(model=model, p=p, cov=cov, chi2=chi2, dof=dof, rchi2=chi2 / dof if dof > 0 else float("nan"))


@dataclass
class Fit:
    model: str
    n: int
    n_dates: int
    span_days: float
    t0: float
    ref_ra: float
    ref_dec: float
    p: np.ndarray
    cov: np.ndarray        # already scaled up by the reduced chi-square when the scatter is larger than the errors
    chi2: float
    dof: int
    rchi2: float
    mu_e: float
    mu_n: float
    mu: float
    mu_err: float
    mu_sig: float          # how many sigma the motion is (0 for a static object)
    mu_limit: float        # 3-sigma upper limit on the speed ("/yr) when no motion is found
    plx: float = float("nan")
    plx_err: float = float("nan")
    acc: tuple = (float("nan"), float("nan"))
    oe: np.ndarray = None  # the marks as offsets (") from the reference point
    on: np.ndarray = None
    err: np.ndarray = None
    mjd: np.ndarray = None
    resid: list = field(default_factory=list)   # per mark: east, north residual (") and sigma
    loo: list = field(default_factory=list)     # per mark: miss from the path of the other marks
    tried: dict = field(default_factory=dict)   # model -> chi2, dof, gain (sigma-like), kept
    pmodel: str = ""      # the model used to predict: a static object still gets its (uncertain) drift
    pp: np.ndarray = None
    pcov: np.ndarray = None

    def param(self, name, default=0.0):
        names = PARAMS[self.model]
        return float(self.p[names.index(name)]) if name in names else default

    def predict_offsets(self, mjds):
        """Predicted offsets (east, north) from the reference point, and their 1-sigma uncertainties."""
        mjds = np.atleast_1d(np.asarray(mjds, float))
        tau = (mjds - self.t0) / YEAR
        model = self.pmodel or self.model
        p, cov = (self.pp, self.pcov) if self.pp is not None else (self.p, self.cov)
        fe, fn = parallax_factors(self.ref_ra, self.ref_dec, mjds) if "plx" in PARAMS[model] else (
            np.zeros_like(tau), np.zeros_like(tau))
        A = _design(model, tau, fe, fn)
        v = A @ p
        var = np.einsum("ij,jk,ik->i", A, cov, A)
        k = len(tau)
        return v[:k], v[k:], np.sqrt(var[:k]), np.sqrt(var[k:])

    def to_radec(self, e, n):
        dec = self.ref_dec + np.asarray(n) / 3600
        ra = (self.ref_ra + np.asarray(e) / 3600 / np.cos(np.radians(self.ref_dec))) % 360
        return ra, dec

    def predict(self, mjds):
        """RA, Dec (deg) on each date and the 1-sigma uncertainty east and north (")."""
        e, n, se, sn = self.predict_offsets(mjds)
        ra, dec = self.to_radec(e, n)
        return ra, dec, se, sn

    def as_target(self, name: str, source: str = "your marks (fitted path)"):
        """The fitted path as a targets.Target (straight-line part; the viewer's ring follows it)."""
        from .targets import Target
        ra, dec = self.to_radec(self.param("x0"), self.param("y0"))
        plx = self.plx if np.isfinite(self.plx) else 0.0
        return Target(name, float(ra), float(dec), self.t0, self.mu_e * 1000, self.mu_n * 1000, plx * 1000, source)


def _offsets(obs):
    ra = np.array([o.ra for o in obs], float)
    dec = np.array([o.dec for o in obs], float)
    err = np.array([max(o.err, 1e-3) for o in obs], float)
    w = 1 / err ** 2
    ra_u = ra[0] + (ra - ra[0] + 180) % 360 - 180          # unwrap around RA = 0
    ref_dec = float(np.sum(w * dec) / w.sum())
    ref_ra = float(np.sum(w * ra_u) / w.sum()) % 360
    oe = ((ra - ref_ra + 180) % 360 - 180) * np.cos(np.radians(ref_dec)) * 3600
    on = (dec - ref_dec) * 3600
    return ref_ra, ref_dec, oe, on, err


def _gain(base, better):
    """How much better a richer model fits, in a chi-square scaled by the scatter the richer model leaves."""
    if base is None or better is None:
        return 0.0
    scale = max(1.0, better["rchi2"]) if better["dof"] > 0 else 1.0
    return max(0.0, (base["chi2"] - better["chi2"]) / scale)


def _choose(tau, fe, fn, oe, on, err, n_dates, span_yr):
    """Fit every model the marks allow; keep the simplest one the data do not clearly reject."""
    n = len(tau)
    fits = {"static": _lsq("static", tau, fe, fn, oe, on, err)}
    if span_yr * YEAR * 86400 >= 1 and n_dates >= 2:      # single exposures can be minutes apart
        fits["linear"] = _lsq("linear", tau, fe, fn, oe, on, err)
    if n_dates >= 3 and span_yr >= 0.25 and 2 * n - 5 >= 1:
        fits["linear+parallax"] = _lsq("linear+parallax", tau, fe, fn, oe, on, err)
    if n_dates >= 4 and 2 * n - 6 >= 2:
        fits["quadratic"] = _lsq("quadratic", tau, fe, fn, oe, on, err)
    fits = {k: v for k, v in fits.items() if v is not None}
    tried = {}
    best = fits["static"]
    g_lin = _gain(fits["static"], fits.get("linear"))
    tried["static"] = dict(chi2=fits["static"]["chi2"], dof=fits["static"]["dof"], gain=0.0, kept=True)
    if "linear" in fits:
        moves = g_lin > 11.83                       # 3 sigma for two parameters
        tried["linear"] = dict(chi2=fits["linear"]["chi2"], dof=fits["linear"]["dof"], gain=g_lin, kept=moves)
        if moves:
            best = fits["linear"]
    if best["model"] == "linear":
        cands = [best]
        f = fits.get("linear+parallax")
        if f is not None:
            g = _gain(best, f)
            plx, sp = f["p"][4], np.sqrt(f["cov"][4, 4] * max(1.0, f["rchi2"]))
            ok = g > 9 and 0 < plx <= 1.0 and plx > 3 * sp   # nearest star: 0.77"; anything bigger is not parallax
            tried["linear+parallax"] = dict(chi2=f["chi2"], dof=f["dof"], gain=g, kept=ok)
            if ok:
                cands.append(f)
        f = fits.get("quadratic")
        if f is not None:
            g = _gain(best, f)
            ok = g > 16
            tried["quadratic"] = dict(chi2=f["chi2"], dof=f["dof"], gain=g, kept=ok)
            if ok:
                cands.append(f)
        best = min(cands, key=lambda c: c["chi2"] + len(c["p"]) * np.log(2 * n))
    for k, v in tried.items():
        v["kept"] = k == best["model"]
    return best, fits, tried, g_lin


def fit(obs: list[Obs]) -> Fit:
    """All marks of one object fitted together. Needs at least two marks."""
    if len(obs) < 2:
        raise ValueError("at least two marks are needed")
    obs = sorted(obs, key=lambda o: o.mjd)
    ref_ra, ref_dec, oe, on, err = _offsets(obs)
    mjd = np.array([o.mjd for o in obs], float)
    w = 1 / err ** 2
    t0 = float(np.sum(w * mjd) / w.sum())
    tau = (mjd - t0) / YEAR
    fe, fn = parallax_factors(ref_ra, ref_dec, mjd)
    n_dates = _n_photos(mjd)
    span_yr = float(np.ptp(mjd)) / YEAR
    best, fits, tried, g_lin = _choose(tau, fe, fn, oe, on, err, n_dates, span_yr)
    scale = max(1.0, best["rchi2"]) if best["dof"] > 0 else 1.0
    cov = best["cov"] * scale
    names = PARAMS[best["model"]]
    lin = fits.get("linear")
    if "mu_e" in names:
        src_p, src_cov, src_names = best["p"], cov, names
    elif lin is not None:
        ls = max(1.0, lin["rchi2"]) if lin["dof"] > 0 else 1.0
        src_p, src_cov, src_names = lin["p"], lin["cov"] * ls, PARAMS["linear"]
    else:
        src_p = src_cov = src_names = None
    if src_p is not None:
        ie, inn = src_names.index("mu_e"), src_names.index("mu_n")
        mu_e, mu_n = float(src_p[ie]), float(src_p[inn])
        C = src_cov[np.ix_([ie, inn], [ie, inn])]
        mu = float(np.hypot(mu_e, mu_n))
        u = np.array([mu_e, mu_n]) / mu if mu > 0 else np.array([np.sqrt(0.5), np.sqrt(0.5)])
        mu_err = float(np.sqrt(u @ C @ u))
        mu_lim = float(3 * np.sqrt(np.trace(C) / 2))
    else:
        mu_e = mu_n = mu = mu_err = mu_lim = float("nan")
    moving = best["model"] != "static"
    f = Fit(model=best["model"], n=len(obs), n_dates=n_dates, span_days=float(np.ptp(mjd)), t0=t0,
            ref_ra=ref_ra, ref_dec=ref_dec, p=best["p"], cov=cov, chi2=best["chi2"], dof=best["dof"],
            rchi2=best["rchi2"], mu_e=mu_e if moving else 0.0, mu_n=mu_n if moving else 0.0,
            mu=mu if moving else 0.0, mu_err=mu_err, mu_sig=float(np.sqrt(g_lin)) if moving else 0.0,
            mu_limit=mu_lim, oe=oe, on=on, err=err, mjd=mjd, tried=tried)
    if "plx" in names:
        i = names.index("plx")
        f.plx, f.plx_err = float(best["p"][i]), float(np.sqrt(cov[i, i]))
    if "acc_e" in names:
        f.acc = (float(best["p"][names.index("acc_e")]), float(best["p"][names.index("acc_n")]))
    if not moving and lin is not None:   # predict with the drift that cannot be ruled out, and its uncertainty
        f.pmodel, f.pp, f.pcov = "linear", lin["p"], lin["cov"] * (max(1.0, lin["rchi2"]) if lin["dof"] > 0 else 1.0)
    pe, pn, _, _ = f.predict_offsets(mjd)
    f.resid = [dict(de=float(a - b), dn=float(c - d), nsig=float(np.hypot(a - b, c - d) / (s * np.sqrt(2))))
               for a, b, c, d, s in zip(oe, pe, on, pn, err)]
    f.loo = _leave_one_out(f.model, tau, fe, fn, oe, on, err)
    return f


def _leave_one_out(model, tau, fe, fn, oe, on, err):
    """Each mark against the path the other marks draw: a mark that is far off is probably another object."""
    n = len(tau)
    out = []
    if n < 3:
        return out
    order = ["quadratic", "linear+parallax", "linear", "static"]
    for i in range(n):
        keep = np.arange(n) != i
        mdl, r = model, None
        while True:   # fall back to a simpler model when the other marks cannot carry this one
            enough = 2 * keep.sum() - len(PARAMS[mdl]) >= (0 if mdl in ("linear", "static") else 1)
            dates = _n_photos(tau[keep] * YEAR) >= (1 if mdl == "static" else 2)
            if enough and dates:
                r = _lsq(mdl, tau[keep], fe[keep], fn[keep], oe[keep], on[keep], err[keep])
                if r is not None:
                    break
            if mdl == "static":
                break
            mdl = order[order.index(mdl) + 1]
        if r is None:
            out.append(dict(miss=float("nan"), sig=float("nan"), nsig=0.0, model=mdl))
            continue
        cov = r["cov"] * (max(1.0, r["rchi2"]) if r["dof"] > 0 else 1.0)
        A = _design(mdl, tau[i:i + 1], fe[i:i + 1], fn[i:i + 1])
        v = A @ r["p"]
        var = np.einsum("ij,jk,ik->i", A, cov, A)
        miss = float(np.hypot(oe[i] - v[0], on[i] - v[1]))
        sig = float(np.sqrt(0.5 * (var[0] + var[1]) + err[i] ** 2))
        out.append(dict(miss=miss, sig=sig, nsig=miss / (sig * np.sqrt(2)), model=mdl,
                        pred_e=float(v[0]), pred_n=float(v[1])))
    return out


def pairs(obs: list[Obs]) -> list[dict]:
    """Every pair of marks compared directly: distance, direction, speed and how sure the move is."""
    from .detect import COMPASS
    obs = sorted(obs, key=lambda o: o.mjd)
    out = []
    for i, j in combinations(range(len(obs)), 2):
        a, b = obs[i], obs[j]
        de = ((b.ra - a.ra + 180) % 360 - 180) * np.cos(np.radians(0.5 * (a.dec + b.dec))) * 3600
        dn = (b.dec - a.dec) * 3600
        dist, err = float(np.hypot(de, dn)), float(np.hypot(a.err, b.err))
        dt = b.mjd - a.mjd
        pa = float(np.degrees(np.arctan2(de, dn)) % 360)
        out.append(dict(i=i, j=j, de=float(de), dn=float(dn), dist=dist, err=err, days=float(dt),
                        speed_yr=dist / (dt / YEAR) if dt > 0 else float("nan"),
                        speed_hr=dist / (dt * 24) if dt > 0 else float("nan"),
                        pa=pa, way=COMPASS[int(round(pa / 45)) % 8], sig=dist / err if err > 0 else float("inf")))
    return out


@dataclass
class Classification:
    best: str
    prob: dict
    reasons: list           # dicts: cls, w (+ for, - against), text
    variable: str = ""      # a note when the brightness changed a lot
    horizon_days: float = 3652.5   # how far ahead a prediction still means something


def _n_photos(mjd) -> int:
    """How many different photos (dates to the second: single exposures can be a few minutes apart)."""
    return len(np.unique(np.round(np.asarray(mjd, float) * 86400)))


def _speed_text(f: Fit) -> str:
    if f.span_days < 1 / 24:
        return f"{f.mu / YEAR / 1440:.2f}″ per minute"
    if f.span_days < 5:
        return f"{f.mu / YEAR / 24:.2f}″ per hour"
    return f"{f.mu:.2f}″ per year"


def classify(obs, f: Fit, ecl_lat: float | None = None, gaia=None, left_behind=None, confirmed=None) -> Classification:
    """Weigh the evidence for each kind of object. gaia: None (not searched), False (no star at the marks) or a
    dict(g, pm (mas/yr), sep ("); left_behind: list of (i, j, ratio) = brightness still at mark i's position on
    the photo of mark j, relative to the object; confirmed: list of (i, fraction of single exposures)."""
    score = {"static": 0.5, "pm_star": 0.0, "solar_system": 0.0, "mismatch": -0.5}
    reasons = []

    def say(cls, w, text):
        score[cls] += w
        reasons.append(dict(cls=cls, w=w, text=text))

    obs = sorted(obs, key=lambda o: o.mjd)
    days = f.span_days
    when = (f"{days * 1440:.1f} minutes" if days < 1 / 24 else f"{days * 24:.1f} hours" if days < 2 else f"{days:.0f} days" if days < 2 * YEAR
            else f"{days / YEAR:.1f} years")
    consistent = f.dof > 0 and np.isfinite(f.rchi2) and f.rchi2 < 4
    scattered = f.dof > 0 and np.isfinite(f.rchi2) and f.rchi2 > 9
    if f.model == "static":
        lim = f.mu_limit if np.isfinite(f.mu_limit) else float("inf")
        if days < 5:   # photos hours or minutes apart: only fast movers could show; say it per hour / minute
            per = (f"{lim / YEAR / 1440:.2f}″ per minute" if days < 1 / 24 else f"{lim / YEAR / 24:.2f}″ per hour")
            say("static", 1.0, f"No motion between the positions ({when} apart): they agree within the measuring "
                               f"error (any motion is below {per} at 3σ). Over so short a time only an asteroid, "
                               "a satellite or a meteor could move visibly; every star stands still.")
        else:
            strength = 3.0 if lim < 1 else 1.5 if lim < 5 else 0.5   # a weak limit proves little
            say("static", strength, f"No motion between the positions ({when} apart): they agree within the "
                                    f"measuring error (any motion is below {lim:.2f}″ per year at 3σ).")
        if lim >= 1 and days >= 5:
            hint = float(np.sqrt(f.tried.get("linear", {}).get("gain", 0.0)))
            say("pm_star", 0.5 + (1.0 if hint >= 2 else 0.0),
                f"But the dates are too close together (or the errors too big) to rule out a drift of up to "
                f"{lim:.1f}″ per year, which a nearby star could have"
                + (f"; there is even a hint of motion ({hint:.1f}σ)" if hint >= 2 else "")
                + ". More dates, or the older 2MASS / WISE photos (sidebar), settle it.")
        if scattered:
            say("mismatch", 3.0 + min(3.0, np.log10(f.rchi2)),
                f"But the positions scatter {np.sqrt(f.rchi2):.0f}× more than the measuring error, and not along one "
                "path: they are probably on different objects.")
            score["static"] -= 2.0
    else:
        say("static", -3.0, f"It moved: {f.mu_sig:.0f}σ, far more than the measuring error.")
        if days < 5:
            say("solar_system", 4.0, f"It moved {_speed_text(f)} within {when}. No star moves measurably in days "
                                     f"(the fastest, Barnard's Star, drifts {FASTEST_STAR / YEAR:.3f}″ per day).")
            score["pm_star"] -= 4.0
        elif f.mu > STAR_LIMIT:
            say("pm_star", -4.0, f"{f.mu:.0f}″ per year is faster than any star (the fastest, Barnard's Star, "
                                 f"moves {FASTEST_STAR}″ per year).")
            score["solar_system"] += 1.5
            if days > 30:
                say("mismatch", 2.0, "An asteroid crosses a field this size in hours to days, so marks "
                                     f"{when} apart are probably on two different objects.")
        else:
            say("pm_star", 3.0, f"Steady drift of {f.mu:.2f}″ per year: typical of a nearby star or brown dwarf.")
            if days > 30:
                say("solar_system", -3.0, f"An asteroid would have left this patch of sky within days, yet it is "
                                          f"still here after {when}.")
        if f.model == "linear+parallax":
            dist = 1 / f.plx
            say("pm_star", 2.0, f"It shows the yearly parallax wobble (π = {f.plx:.3f} ± {f.plx_err:.3f}″), so it "
                                f"is close: about {dist:.1f} parsecs ({dist * 3.2616:.1f} light-years).")
        if f.model == "quadratic":
            if days < 60:
                say("solar_system", 2.0, "The path curves within weeks, as an orbit seen from the moving Earth does.")
            else:
                say("mismatch", 1.0, "The path bends, which a star's steady drift does not do.")
        if scattered:
            say("mismatch", 2.0 + min(3.0, np.log10(f.rchi2)),
                f"The positions do not lie on one path: the scatter is {np.sqrt(f.rchi2):.0f}× the measuring error.")
    if consistent:
        lead = "solar_system" if (f.model != "static" and days < 5) else "pm_star" if f.model != "static" else "static"
        say(lead, 0.5, f"All {f.n} positions lie on one path within the errors (χ²ν = {f.rchi2:.1f}).")
    if f.loo:
        k = int(np.argmax([x["nsig"] for x in f.loo]))
        if f.loo[k]["nsig"] > 5:
            say("mismatch", 1.5, f"Mark {obs[k].label or '#' + str(k + 1)} is {f.loo[k]['miss']:.1f}″ "
                                 f"({f.loo[k]['nsig']:.0f}σ) from where the other marks put it: leave it out and "
                                 "try again, or check it on its photo.")
    if ecl_lat is None:
        ecl_lat = ecliptic_latitude(float(np.mean([o.ra for o in obs])), float(np.mean([o.dec for o in obs])))
    if abs(ecl_lat) < 15:
        say("solar_system", 0.5, f"It is near the ecliptic (latitude {ecl_lat:+.0f}°), where most asteroids are.")
    elif abs(ecl_lat) > 40:
        say("solar_system", -1.0, f"It is far from the ecliptic (latitude {ecl_lat:+.0f}°), where asteroids are rare.")
    if gaia is False:
        say("solar_system", 0.5, "No Gaia star at the marks: not a catalogued star (very cold brown dwarfs, "
                                 "asteroids and faint galaxies are missing from Gaia).")
    elif isinstance(gaia, dict):
        pm = float(np.hypot(*gaia.get("pm", (0.0, 0.0))))
        who = gaia.get("who", "Gaia")
        txt = gaia.get("text") or f"A Gaia DR3 star (G = {gaia.get('g', float('nan')):.1f}) sits at the marks"
        say("solar_system", -2.0, txt + ": a catalogued star, not an asteroid.")
        if pm >= 50:
            say("pm_star", 2.5, f"{who} gives its proper motion as {pm / 1000:.2f}″ per year.")
        else:
            say("static", 1.5, f"{who} gives it almost no motion ({pm / 1000:.3f}″ per year).")
    if left_behind and f.model != "static":
        stays = [x for x in left_behind if x[2] > 0.5]
        if stays:
            i, j, r = stays[0]
            say("mismatch", 2.0, f"On the photo of mark {obs[j].label or j + 1}, something {r * 100:.0f}% as bright "
                                 f"still sits where mark {obs[i].label or i + 1} was: the marks may be on two "
                                 "different objects.")
        elif all(x[2] < 0.25 for x in left_behind):
            lead = "solar_system" if days < 5 else "pm_star"
            say(lead, 0.5, "The old spots are empty on the later photos: the object really left them.")
    for i, frac in confirmed or []:
        if frac < 0.5:
            say("mismatch", 1.0, f"Mark {obs[i].label or i + 1} is seen in only {frac * 100:.0f}% of the single "
                                 "exposures behind its photo: a cosmic-ray hit or hot pixel, or something that "
                                 "moved during the exposures.")
    variable = ""
    fl = np.array([o.flux for o in obs], float)
    fe_ = np.array([o.flux_err for o in obs], float)
    ok = np.isfinite(fl) & np.isfinite(fe_) & (fe_ > 0)
    if ok.sum() >= 2 and np.nanmedian(fl[ok]) > 0:
        lo, hi = int(np.argmin(np.where(ok, fl, np.inf))), int(np.argmax(np.where(ok, fl, -np.inf)))
        # each photo averages a different mix of wavelengths inside the band: 20% of the flux is normal scatter
        e_hi, e_lo = np.hypot(fe_[hi], SYS_FLUX * fl[hi]), np.hypot(fe_[lo], SYS_FLUX * max(fl[lo], 0.0))
        sig = (fl[hi] - fl[lo]) / np.hypot(e_hi, e_lo)
        if sig > 3 and fl[hi] > 1.5 * max(fl[lo], 1e-12):
            variable = (f"Its brightness changed: {fl[hi] / max(fl[lo], 1e-12):.1f}× brighter on "
                        f"{obs[hi].label or hi + 1} than on {obs[lo].label or lo + 1} ({sig:.0f}σ, allowing 20% for "
                        "the different colours each photo averages). A variable star, a flare, or a different object.")
    m = max(score.values())
    ex = {k: np.exp(v - m) for k, v in score.items()}
    tot = sum(ex.values())
    prob = {k: float(ex[k] / tot) for k in CLASSES}
    best = max(prob, key=prob.get)
    horizon = 3 if best == "solar_system" else 3652.5 if best in ("pm_star", "static") else 365.25
    return Classification(best=best, prob=prob, reasons=reasons, variable=variable, horizon_days=horizon)

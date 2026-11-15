"""The motion engine (spx/motion.py): fit every mark together, test models, classify, predict the future."""
import numpy as np
import pytest

from spx import motion, targets

RA0, DEC0 = 150.0, 20.0
T0 = 61041.0  # 2026-01-01


def make_obs(mjds, mu_e=0.0, mu_n=0.0, plx=0.0, err=0.3, ra=RA0, dec=DEC0, jitter=None, acc=(0.0, 0.0)):
    """Observations of a synthetic object: offsets (arcsec) turned into RA/Dec."""
    mjds = np.asarray(mjds, float)
    fe, fn = motion.parallax_factors(ra, dec, mjds)
    tau = (mjds - T0) / 365.25
    de = mu_e * tau + plx * fe + 0.5 * acc[0] * tau ** 2
    dn = mu_n * tau + plx * fn + 0.5 * acc[1] * tau ** 2
    if jitter is not None:
        de, dn = de + jitter[0], dn + jitter[1]
    return [motion.Obs(mjd=float(m), ra=ra + e / 3600 / np.cos(np.radians(dec)), dec=dec + n / 3600, err=err,
                       label=f"#{i + 1}") for i, (m, e, n) in enumerate(zip(mjds, de, dn))]


def test_parallax_factors_match_the_target_prediction():
    t = targets.Target("x", RA0, DEC0, T0, 0.0, 0.0, 1000.0)  # parallax 1" = 1000 mas
    mjds = [T0, T0 + 91, T0 + 182, T0 + 273]
    with_plx = targets.predicted_offsets(t, mjds, T0)
    fe, fn = motion.parallax_factors(RA0, DEC0, mjds)
    assert np.allclose(with_plx[:, 0] - with_plx[0, 0], fe - fe[0], atol=1e-3)
    assert np.allclose(with_plx[:, 1] - with_plx[0, 1], fn - fn[0], atol=1e-3)


def test_three_marks_are_fitted_together_and_recover_the_motion():
    obs = make_obs([T0, T0 + 200, T0 + 560], mu_e=1.0, mu_n=-2.0)
    f = motion.fit(obs)
    assert f.model in ("linear", "linear+parallax")
    assert f.mu_e == pytest.approx(1.0, abs=0.05) and f.mu_n == pytest.approx(-2.0, abs=0.05)
    assert f.n == 3


def test_a_still_object_is_called_static_with_an_upper_limit():
    rng = np.random.default_rng(1)
    obs = make_obs([T0, T0 + 180, T0 + 370, T0 + 550], err=0.5, jitter=rng.normal(0, 0.5, (2, 4)))
    f = motion.fit(obs)
    assert f.model == "static"
    assert 0 < f.mu_limit < 2.0


def test_parallax_is_found_for_a_nearby_star():
    mjds = T0 + np.array([0, 60, 180, 240, 365, 425, 545, 605])
    rng = np.random.default_rng(2)
    obs = make_obs(mjds, mu_e=-0.8, mu_n=10.36, plx=0.547, err=0.15, jitter=rng.normal(0, 0.15, (2, len(mjds))))
    f = motion.fit(obs)
    assert f.model == "linear+parallax"
    assert f.plx == pytest.approx(0.547, abs=4 * f.plx_err)
    assert f.mu_n == pytest.approx(10.36, abs=0.2)


def test_every_pair_of_marks_is_compared():
    obs = make_obs([T0, T0 + 200, T0 + 560], mu_e=3.0)
    ps = motion.pairs(obs)
    assert {(p["i"], p["j"]) for p in ps} == {(0, 1), (0, 2), (1, 2)}
    assert all(p["speed_yr"] == pytest.approx(3.0, abs=0.05) for p in ps)


def test_leave_one_out_points_at_the_mark_that_does_not_fit():
    obs = make_obs([T0, T0 + 180, T0 + 360, T0 + 540, T0 + 720], mu_e=2.0, err=0.3)
    bad = obs[2]
    obs[2] = motion.Obs(mjd=bad.mjd, ra=bad.ra + 20 / 3600, dec=bad.dec, err=bad.err, label=bad.label)
    f = motion.fit(obs)
    worst = max(range(len(obs)), key=lambda i: f.loo[i]["nsig"])
    assert worst == 2 and f.loo[2]["nsig"] > 10


def test_the_future_is_predicted_with_growing_uncertainty():
    obs = make_obs([T0, T0 + 200, T0 + 560], mu_e=1.0, mu_n=-2.0, err=0.5)
    f = motion.fit(obs)
    mjds = np.array([T0 + 280, T0 + 280 + 3652.5])
    ra, dec, se, sn = f.predict(mjds)
    assert np.hypot(se[1], sn[1]) > 3 * np.hypot(se[0], sn[0])
    de = (ra[1] - ra[0]) * np.cos(np.radians(DEC0)) * 3600
    assert de == pytest.approx(10.0, abs=0.3)


def test_fast_motion_within_hours_is_a_solar_system_object():
    t = T0 + np.array([0, 1.5, 3.0]) / 24
    obs = make_obs(t, mu_e=30 * 24 * 365.25, mu_n=10 * 24 * 365.25, ra=180.0, dec=2.0)  # 30"/h east, 10"/h north
    f = motion.fit(obs)
    c = motion.classify(obs, f)
    assert c.best == "solar_system"
    assert c.prob["pm_star"] < 0.1


def test_slow_steady_motion_over_years_is_a_moving_star():
    obs = make_obs(T0 + np.array([0, 180, 365, 545]), mu_e=1.5, mu_n=0.8, err=0.3, dec=60.0)
    c = motion.classify(obs, motion.fit(obs))
    assert c.best == "pm_star"
    assert c.reasons  # every verdict comes with its evidence


def test_marks_that_do_not_line_up_are_not_one_object():
    rng = np.random.default_rng(3)
    obs = make_obs(T0 + np.array([0, 180, 365, 545, 730]), err=0.3,
                   jitter=rng.normal(0, 15, (2, 5)))  # 15" scatter: five different stars
    c = motion.classify(obs, motion.fit(obs))
    assert c.best == "mismatch"


def test_a_still_object_is_classified_static():
    rng = np.random.default_rng(4)
    obs = make_obs(T0 + np.array([0, 180, 365]), err=0.4, jitter=rng.normal(0, 0.4, (2, 3)))
    c = motion.classify(obs, motion.fit(obs))
    assert c.best == "static"


def test_no_detection_with_a_weak_limit_is_not_called_standing_still_with_confidence():
    # Barnard's Star seen on three epochs a year apart with big errors: the drift is only ~2 sigma
    rng = np.random.default_rng(6)
    obs = make_obs(T0 + np.array([0, 180, 350]), mu_e=-0.8, mu_n=10.4, err=3.5, jitter=rng.normal(0, 1.0, (2, 3)))
    f = motion.fit(obs)
    c = motion.classify(obs, f)
    assert f.mu_limit > 5
    assert c.prob["static"] < 0.6
    assert any("rule out" in r["text"] for r in c.reasons)


def test_position_at_accepts_many_dates():
    t = targets.PRESETS["Barnard's Star"]
    ra, dec = targets.position_at(t, np.array([T0, T0 + 365.25]))
    assert (dec[1] - dec[0]) * 3600 == pytest.approx(10.36, abs=0.05)


def test_too_fast_for_any_star_over_months_is_not_a_star():
    obs = make_obs([T0, T0 + 180], mu_e=600.0)
    c = motion.classify(obs, motion.fit(obs))
    assert c.prob["pm_star"] < 0.1


def test_seen_fraction_counts_the_single_exposures_that_show_the_object():
    from spx import detect
    rng = np.random.default_rng(5)
    yy, xx = np.indices((40, 40))
    star = 50 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / (2 * 1.5 ** 2))
    singles = [rng.normal(0, 1, (40, 40)) + (star if k != 3 else 0) for k in range(4)]
    assert detect.seen_fraction(singles, 20, 20) == pytest.approx(0.75)
    assert detect.seen_fraction(None, 20, 20) is None


def test_a_mark_left_behind_on_a_later_photo_points_to_two_objects():
    obs = make_obs([T0, T0 + 365], mu_e=8.0, dec=60.0)
    c_ok = motion.classify(obs, motion.fit(obs), left_behind=[(0, 1, 0.05)])
    c_bad = motion.classify(obs, motion.fit(obs), left_behind=[(0, 1, 0.9)])
    assert c_bad.prob["mismatch"] > c_ok.prob["mismatch"]


def test_a_big_brightness_change_is_reported():
    obs = [motion.Obs(T0, RA0, DEC0, 0.3, "#1", flux=10.0, flux_err=0.5),
           motion.Obs(T0 + 180, RA0, DEC0, 0.3, "#2", flux=30.0, flux_err=0.5)]
    c = motion.classify(obs, motion.fit(obs))
    assert "brighter" in c.variable


def test_a_modest_change_with_tiny_noise_errors_is_not_called_variable():
    # each epoch averages different wavelengths in a band: tens of percent is normal, whatever the pixel noise says
    obs = [motion.Obs(T0, RA0, DEC0, 0.3, "#1", flux=100.0, flux_err=0.01),
           motion.Obs(T0 + 180, RA0, DEC0, 0.3, "#2", flux=145.0, flux_err=0.01)]
    assert motion.classify(obs, motion.fit(obs)).variable == ""


def test_marks_minutes_apart_on_two_exposures_are_two_dates_and_a_move():
    # Every exposure: a fast mover marked on two single photos taken 3 minutes apart
    obs = [motion.Obs(T0, RA0, DEC0, 1.0, "#1"),
           motion.Obs(T0 + 3 / 1440, RA0 + 20 / 3600 / np.cos(np.radians(DEC0)), DEC0, 1.0, "#2")]
    f = motion.fit(obs)
    assert f.n_dates == 2 and f.model == "linear"
    assert f.mu / motion.YEAR / 24 / 60 == pytest.approx(20 / 3, rel=0.01)   # "/minute
    c = motion.classify(obs, f)
    assert c.best == "solar_system"
    assert any("minute" in r["text"] for r in c.reasons)


def test_no_motion_within_hours_is_said_per_hour_without_star_advice():
    obs = [motion.Obs(T0, RA0, DEC0, 3.0, "#1"), motion.Obs(T0 + 0.2, RA0, DEC0 + 1 / 3600, 3.0, "#2")]
    c = motion.classify(obs, motion.fit(obs))
    text = " ".join(r["text"] for r in c.reasons)
    assert "per hour" in text and "per year" not in text and "nearby star could have" not in text


def test_viewer_frames_carry_a_unique_photo_id_even_on_the_same_date():
    from spx import stack, viewer
    from types import SimpleNamespace
    img = np.zeros((8, 8))
    frames = [stack.Frame(img, f"#{i} 2026-01-01", "D1", T0 + i * 0.002, T0 + i * 0.002 + 0.001, 1, kind="single",
                          stamps=[SimpleNamespace(uri=f"exp_{i}.fits")]) for i in range(3)]
    ids = [f["uid"] for f in viewer.payload(frames)["frames"]]
    assert len(set(ids)) == 3 and ids[0] == viewer.photo_uid(frames[0])


def test_fit_as_target_draws_the_same_path_in_the_viewer():
    obs = make_obs([T0, T0 + 200, T0 + 560], mu_e=1.0, mu_n=-2.0)
    f = motion.fit(obs)
    t = f.as_target("my find")
    ra, dec = targets.position_at(t, T0 + 900)
    pra, pdec, _, _ = f.predict([T0 + 900])
    assert abs((ra - pra[0]) * np.cos(np.radians(DEC0)) * 3600) < 0.05 and abs((dec - pdec[0]) * 3600) < 0.05

"""Random teleport (/RTP) and the loading time estimate."""
import numpy as np
import pytest

from spx import eta, targets


def test_rtp_command_is_recognised_in_any_case_and_nothing_else():
    assert targets.is_teleport("/RTP") and targets.is_teleport("  /rtp ") and targets.is_teleport("/Rtp")
    assert not targets.is_teleport("RTP Star") and not targets.is_teleport("269.4 4.7") and not targets.is_teleport("")


def test_random_teleport_covers_the_sky_evenly():
    rng = np.random.default_rng(7)
    pts = np.array([targets.random_position(rng) for _ in range(4000)])
    assert pts[:, 0].min() >= 0 and pts[:, 0].max() < 360
    assert pts[:, 1].min() >= -90 and pts[:, 1].max() <= 90
    # even on the sphere: half of the sky lies within 30 degrees of the equator
    assert np.mean(np.abs(pts[:, 1]) < 30) == pytest.approx(0.5, abs=0.03)


def test_random_teleport_text_is_read_back_as_the_same_position():
    ra, dec = targets.random_position(np.random.default_rng(1))
    t = targets.parse_coords(targets.teleport_text(ra, dec))
    assert t is not None and abs(t.ra - ra) < 1e-4 and abs(t.dec - dec) < 1e-4


def test_estimate_before_any_photo_uses_what_was_learned(tmp_path):
    e = eta.Estimator(tmp_path / "t.json")
    e.learn(new=20, seconds=10.0)                       # 0.5 s per new photo on this computer
    left = e.start(new=40, cached=10)
    assert left == pytest.approx(40 * 0.5 + 10 * eta.CACHED_SECONDS, rel=0.3)


def test_estimate_follows_the_real_speed_once_photos_arrive(tmp_path):
    e = eta.Estimator(tmp_path / "t.json")
    e.start(new=100, cached=0)
    left = e.update(done=50, total=100, elapsed=50.0)   # really 1 s per photo
    assert left == pytest.approx(50, rel=0.25)
    assert e.update(done=100, total=100, elapsed=100.0) == 0


def test_no_promise_when_nothing_arrived_and_the_expected_time_is_over(tmp_path):
    e = eta.Estimator(tmp_path / "t.json")
    e.start(new=10, cached=0)                           # expects ~6 s
    assert e.update(done=0, total=10, elapsed=3.0) > 0
    assert e.update(done=0, total=10, elapsed=20.0) is None
    assert eta.words(None) == "taking longer than usual"


def test_what_was_learned_is_remembered_for_the_next_visit(tmp_path):
    eta.Estimator(tmp_path / "t.json").learn(new=10, seconds=30.0)
    assert eta.Estimator(tmp_path / "t.json").per_photo == pytest.approx(3.0, rel=0.5)


def test_how_long_a_step_usually_takes_is_learned_and_remembered(tmp_path):
    e = eta.Estimator(tmp_path / "t.json")
    assert e.step_left("search", 0.0) == pytest.approx(eta.DEFAULT_STEP["search"])
    e.learn_step("search", 8.0)
    e2 = eta.Estimator(tmp_path / "t.json")
    assert e2.step_left("search", 3.0) == pytest.approx(5.0, abs=0.01)
    assert e2.step_left("search", 30.0) is None          # longer than usual: no promise


def test_estimate_is_said_in_plain_words():
    assert eta.words(3) == "almost done"
    assert eta.words(25) == "about 25 s left"
    assert eta.words(130) == "about 2 min left"

"""Effect-size helpers against known values.

    python -m pytest -q
"""
import numpy as np
import pytest

from stats_utils import cohens_d, cramers_v, eta_squared, pct_effect, rank_biserial


def test_cohens_d_known_value():
    a = np.array([1, 2, 3, 4, 5], float)
    assert cohens_d(a, a + 1.5811388) == pytest.approx(1.0, abs=1e-6)   # SD of a = 1.5811


def test_rank_biserial_extremes():
    assert rank_biserial([1, 2, 3], [4, 5, 6]) == pytest.approx(1.0)
    assert rank_biserial([4, 5, 6], [1, 2, 3]) == pytest.approx(-1.0)


def test_cramers_v_bounds():
    assert cramers_v([[50, 0], [0, 50]]) == pytest.approx(1.0)
    assert cramers_v([[25, 25], [25, 25]]) == pytest.approx(0.0)


def test_eta_squared():
    assert eta_squared([np.array([1.0, 1, 1]), np.array([5.0, 5, 5])]) == pytest.approx(1.0)


def test_pct_effect():
    assert pct_effect(np.log(2)) == pytest.approx(100.0)

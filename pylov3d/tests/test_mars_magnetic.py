import pytest

from pylov3d.mars_magnetic import (
    INSIGHT_BENCHMARKS,
    ORBITAL_BENCHMARKS,
    abundance_margin,
    paleofield_scaled_required_magnetite,
)


def test_bultel_reference_values_are_pinned():
    assert [b.magnetite_wt_percent for b in INSIGHT_BENCHMARKS] == [0.91, 0.49, 0.37, 1.10]
    assert [b.magnetite_wt_percent for b in ORBITAL_BENCHMARKS] == [14.0, 7.1, 3.7, 2.0]


def test_stronger_paleofield_reduces_required_magnetite_linearly():
    b = INSIGHT_BENCHMARKS[1]
    assert paleofield_scaled_required_magnetite(b, 100.0) == pytest.approx(0.245)
    assert paleofield_scaled_required_magnetite(b, 25.0) == pytest.approx(0.98)


def test_abundance_margin_unity_at_reference_requirement():
    b = ORBITAL_BENCHMARKS[2]
    assert abundance_margin(b.magnetite_wt_percent, b) == pytest.approx(1.0)


def test_invalid_paleofield_rejected():
    with pytest.raises(ValueError):
        paleofield_scaled_required_magnetite(INSIGHT_BENCHMARKS[0], 0.0)

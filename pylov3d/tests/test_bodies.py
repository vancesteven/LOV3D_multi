"""Tests for pylov3d.bodies — planetary body catalog."""

import math

import pytest

from pylov3d.bodies import get_body, list_bodies
from pylov3d.constants import G


class TestGetBody:

    def test_io_basic(self):
        io = get_body(51)
        assert io["name"] == "Io"
        assert io["r"] == pytest.approx(1821.46e3)
        assert io["Mass"] == pytest.approx(8.9297e22)

    def test_io_derived(self):
        io = get_body(51)
        # Surface gravity
        expected_grav = G * io["Mass"] / io["r"] ** 2
        assert io["grav"] == pytest.approx(expected_grav)
        # Mean motion
        expected_n = math.sqrt(G * io["Mp"] / io["a"] ** 3)
        assert io["n"] == pytest.approx(expected_n)

    def test_europa(self):
        eu = get_body(52)
        assert eu["name"] == "Europa"
        assert eu["nu"] == pytest.approx(0.499)
        assert eu["hlid"] == pytest.approx(15e3)

    def test_enceladus(self):
        en = get_body(62)
        assert en["name"] == "Enceladus"
        assert en["r"] == pytest.approx(252.10e3)

    def test_density_derived_mass(self):
        """Bodies without explicit Mass compute it from _density."""
        tethys = get_body(63)
        expected = (4 / 3) * math.pi * 0.98e3 * tethys["r"] ** 3
        assert tethys["Mass"] == pytest.approx(expected)

    def test_core_radius(self):
        io = get_body(51)
        assert io["rc"] == pytest.approx(io["r"] - io["hlid"] - io["hocean"])

    def test_unknown_id(self):
        with pytest.raises(KeyError):
            get_body(99)

    def test_tidal_amplitudes(self):
        io = get_body(51)
        assert "Ae" in io
        assert "Aobli" in io
        assert io["Ae"] > 0


class TestListBodies:

    def test_returns_list(self):
        bodies = list_bodies()
        assert len(bodies) >= 15
        ids = [b[0] for b in bodies]
        assert 51 in ids  # Io
        assert 62 in ids  # Enceladus

    def test_sorted(self):
        bodies = list_bodies()
        ids = [b[0] for b in bodies]
        assert ids == sorted(ids)

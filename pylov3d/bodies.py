"""Planetary body catalog.

Translated from Select_Moon.m.  Each body is a dict of orbital and physical
parameters; ``get_body()`` returns the dict for a given moon ID and computes
derived quantities (surface gravity, mean motion, orbital period, tidal
amplitudes).
"""

from __future__ import annotations

import math
from typing import Any

from .constants import G

# ── Raw catalog ──────────────────────────────────────────────────────────

_CATALOG: dict[int, dict[str, Any]] = {
    # Earth ────────────────────────────────────────────────────────────────
    31: dict(
        name="Moon", abbr="Moon",
        Mp=59.736e23, r=1737.4e3, Mass=7.3458e22,
        a=20 * 6370e3,  # close-orbit value used in MATLAB
        e=0.0554, i=5.16, obli=6.68,
        raan=125.08, peri=318.15,
        hlid=50e3, hocean=50e3,
        E=170e9, nu=0.33, rho_o=2800,
    ),
    # Jupiter ──────────────────────────────────────────────────────────────
    51: dict(
        name="Io", abbr="Io",
        Mp=1.8986e27, r=1821.46e3, Mass=8.9297e22,
        a=421800000, e=0.0041, i=0.036, obli=0.0021,
        raan=43.977, peri=84.129,
        hlid=30e3, hocean=50e3,
        E=170e9, nu=0.33, rho_o=2800,
    ),
    52: dict(
        name="Europa", abbr="Eu",
        Mp=1.8986e27, r=1560.8e3, Mass=4.7998e22,
        a=671100000, e=0.0094, i=0.466, obli=0.053,
        raan=219.106, peri=88.970,
        C22=131.5e-6, Im=0.3477,
        hlid=15e3, hocean=123e3,
        E=2 * 2e9 * (1 + 0.499), nu=0.499, rho_o=1000,
    ),
    53: dict(
        name="Ganymede", abbr="Ga",
        Mp=1.8986e27, r=2631.2e3, Mass=14.8e22,
        a=1070400000, e=0.0013, i=0.177, obli=0.0033,
        raan=63.552, peri=192.417,
        hlid=125e3, hocean=150e3,
        E=8.8e9, nu=0.33, rho_o=1000,
    ),
    54: dict(
        name="Callisto", abbr="Ca",
        Mp=1.8986e27, r=2410.0e3, Mass=10.8e22,
        a=1882700000, e=0.0074, i=0.192, obli=0.24,
        raan=298.848, peri=52.643,
        hlid=115e3, hocean=80e3,
        E=8.8e9, nu=0.33, rho_o=1000,
    ),
    # Saturn ───────────────────────────────────────────────────────────────
    61: dict(
        name="Mimas", abbr="Mi",
        Mp=568.46e24, r=198.2e3, Mass=3.7493e19,
        a=185539000, e=0.0196, i=1.574, obli=0.041,
        raan=173.026, peri=332.478,
        hlid=27.5e3, hocean=71.6e3 - 27.5e3,
        E=8.8e9, nu=0.33, rho_o=1000,
        mum=17e9 / (2 * (1 + 0.33)),
    ),
    62: dict(
        name="Enceladus", abbr="En",
        Mp=568.46e24, r=252.10e3, Mass=1.0799e20,
        a=238037000, e=0.0047, i=0.003, obli=4.5e-4,
        raan=343.266, peri=188.319,
        hlid=23e3, hocean=38e3,
        E=2 * 2e9 * (1 + 0.49), nu=0.49, rho_o=1000,
        mum=17e9 / (2 * (1 + 0.33)),
    ),
    63: dict(
        name="Tethys", abbr="Te",
        Mp=568.46e24, r=531.10e3,
        a=294672000, e=0.0001, i=1.091, obli=0.039,
        raan=259.845, peri=44.843,
        hlid=220e3, hocean=50e3,
        E=8.8e9, nu=0.33, rho_o=1000,
        _density=0.98e3,  # Mass computed from density
    ),
    64: dict(
        name="Dione", abbr="Di",
        Mp=568.46e24, r=561.7e3, Mass=1.0955e21,
        a=377415000, e=0.0022, i=0.028, obli=0.002,
        raan=290.615, peri=284.111,
        hlid=99e3, hocean=65e3,
        E=8.8e9, nu=0.33, rho_o=1000,
        mum=17e9 / (2 * (1 + 0.33)),
    ),
    65: dict(
        name="Rhea", abbr="Rh",
        Mp=568.46e24, r=764.4e3,
        a=527068000, e=0.001, i=0.333, obli=0.03,
        raan=351.018, peri=213.663,
        hlid=401e3, hocean=16e3,
        E=8.8e9, nu=0.33, rho_o=1000,
        _density=1.24e3,
    ),
    66: dict(
        name="Titan", abbr="Ti",
        Mp=568.46e24, r=2574.73e3, Mass=1.3452e23,
        a=1221865000, e=0.0288, i=0.306, obli=0.32,
        raan=28.758, peri=179.920,
        hlid=100e3, hocean=255e3,
        E=8.8e9, nu=0.33, rho_o=1000,
    ),
    # Uranus ───────────────────────────────────────────────────────────────
    71: dict(
        name="Miranda", abbr="Mr",
        Mp=86.832e24, r=235.8e3,
        a=129900000, e=0.0014, i=4.338, obli=0.021,
        raan=326.438, peri=68.312,
        hlid=115e3, hocean=10e3,
        E=8.8e9, nu=0.33, rho_o=1000,
        _density=1.18e3,
    ),
    72: dict(
        name="Ariel", abbr="Ar",
        Mp=86.832e24, r=578.9e3,
        a=190900000, e=0.0012, i=0.041, obli=0.0005,
        raan=22.394, peri=115.349,
        hlid=190e3, hocean=19e3,
        E=8.8e9, nu=0.33, rho_o=1000,
        _density=1.54e3,
    ),
    73: dict(
        name="Umbriel", abbr="Um",
        Mp=86.832e24, r=584.7e3,
        a=266000e3, e=0.0039, i=0.128, obli=0.0026,
        raan=33.485, peri=84.709,
        hlid=190e3, hocean=10e3,
        E=8.8e9, nu=0.33, rho_o=1000,
        _density=1.52e3,
    ),
    74: dict(
        name="Titania", abbr="Tt",
        Mp=86.832e24, r=788.9e3,
        a=436300000, e=0.0012, i=0.079, obli=0.014,
        raan=99.771, peri=284.4,
        hlid=230e3, hocean=20e3,
        E=8.8e9, nu=0.33, rho_o=1000,
        _density=1.65e3,
    ),
    75: dict(
        name="Oberon", abbr="Ob",
        Mp=86.832e24, r=761.4e3,
        a=583500000, e=0.0014, i=0.068, obli=0.075,
        raan=279.771, peri=104.4,
        hlid=230e3, hocean=20e3,
        E=8.8e9, nu=0.33, rho_o=1000,
        _density=1.66e3,
    ),
    # Neptune ──────────────────────────────────────────────────────────────
    81: dict(
        name="Triton", abbr="Tri",
        Mp=102.43e24, r=1353.0e3, Mass=2.1390e22,
        a=354759000, e=0.000016, i=156.865, obli=0.35,
        raan=177.608, peri=66.142,
        hlid=150e3, hocean=150e3,
        E=8.8e9, nu=0.33, rho_o=1000,
    ),
    # Pluto ────────────────────────────────────────────────────────────────
    91: dict(
        name="Charon", abbr="Ch",
        Mp=1.32e22, r=603.6e3, Mass=1.5327e21,
        a=17536000, e=0.0022, i=0.001,
        raan=85.187, peri=71.255,
    ),
    92: dict(
        name="Pluto", abbr="Pl",
        Mp=1.32e22, r=1183.3e3, Mass=1.3e22,
        a=17536000, e=0.0022, i=0.001,
        raan=85.187, peri=71.255,
    ),
}


# ── Public API ───────────────────────────────────────────────────────────

def get_body(moon_id: int) -> dict[str, Any]:
    """Return orbital/physical parameters for *moon_id*.

    Moon-ID convention: first digit = planet (3=Earth, 5=Jupiter, 6=Saturn,
    7=Uranus, 8=Neptune, 9=Pluto), second digit = satellite index.

    Derived quantities added to the returned dict:
      grav   — surface gravity [m/s^2]
      n      — mean motion [rad/s]
      Period — orbital period [s]
      density — bulk density [kg/m^3]
      Ae     — eccentricity tidal amplitude [m]
      Aobli  — obliquity tidal amplitude [m]
      rc     — core radius [m]
      rho_c  — core density [kg/m^3]
    """
    if moon_id not in _CATALOG:
        raise KeyError(f"Moon ID {moon_id} not in catalog")

    body = dict(_CATALOG[moon_id])  # shallow copy

    r = body["r"]

    # Compute mass from density if not given directly
    if "Mass" not in body and "_density" in body:
        body["Mass"] = (4 / 3) * math.pi * body["_density"] * r**3
    if "_density" in body:
        del body["_density"]

    Mass = body["Mass"]

    # Derived quantities
    body["grav"] = G * Mass / r**2
    body["n"] = math.sqrt(G * body["Mp"] / body["a"] ** 3)
    body["Period"] = 2 * math.pi / body["n"]
    body["density"] = Mass / ((4 / 3) * math.pi * r**3)

    grav = body["grav"]
    n = body["n"]

    if "e" in body and "obli" in body:
        body["Ae"] = 3.1963 * (n * r) ** 2 * body["e"] / grav
        body["Aobli"] = (
            1.4616
            * (n * r) ** 2
            * math.sin(math.radians(body["obli"]))
            / grav
        )

    # Ice shell / ocean derived quantities
    if all(k in body for k in ("hlid", "hocean", "rho_o")):
        body["rc"] = r - body["hlid"] - body["hocean"]
        rho_c = (r / body["rc"]) ** 3 * (
            body["density"] - body["rho_o"] * (1 - (body["rc"] / r) ** 3)
        )
        body["rho_c"] = rho_c
        body["gravc"] = G * (4 / 3) * math.pi * rho_c * body["rc"]

    return body


def list_bodies() -> list[tuple[int, str]]:
    """Return ``[(moon_id, name), ...]`` for all catalog entries."""
    return [(mid, d.get("name", "?")) for mid, d in sorted(_CATALOG.items())]

"""Direct text loaders for spherical-harmonic gravity and shape models."""

from __future__ import annotations

import csv
import gzip
import operator
from pathlib import Path

import numpy as np


def _open_text(path):
    """Open a plain or extension-identified gzip text file."""
    path = Path(path)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, mode="rt", encoding="ascii")
    return path.open(mode="rt", encoding="ascii")


def _validate_table(table, columns, label, lmax=None):
    """Validate coefficient rows and return integer degree/order arrays."""
    if table.ndim != 2 or table.shape[1] != columns or table.shape[0] == 0:
        raise ValueError(f"{label} must contain nonempty {columns}-column rows")
    if not np.all(np.isfinite(table)):
        raise ValueError(f"{label} contains non-finite values")

    degree_values = table[:, 0]
    order_values = table[:, 1]
    if not (np.all(degree_values == np.floor(degree_values))
            and np.all(order_values == np.floor(order_values))):
        raise ValueError(f"{label} degree and order values must be integers")
    degrees = degree_values.astype(np.int64)
    orders = order_values.astype(np.int64)
    if np.any(degrees < 0) or np.any(orders < 0) or np.any(orders > degrees):
        raise ValueError(f"{label} contains invalid degree/order indices")
    if lmax is not None and np.any(degrees > lmax):
        raise ValueError(f"{label} contains a degree above declared lmax")

    width = int(degrees.max()) + 1 if lmax is None else lmax + 1
    flat_indices = degrees * width + orders
    if np.unique(flat_indices).size != flat_indices.size:
        raise ValueError(f"{label} contains duplicate degree/order rows")
    return degrees, orders


def load_shadr(path) -> dict:
    """Load a normalized PDS SHADR gravity field into NumPy arrays."""
    with _open_text(path) as stream:
        header = next(csv.reader([stream.readline()]), None)
        if header is None or len(header) != 8:
            raise ValueError("SHADR header must contain exactly 8 values")
        try:
            header_values = [float(value) for value in header]
        except ValueError as exc:
            raise ValueError("SHADR header contains a non-numeric value") from exc
        table = np.loadtxt(stream, delimiter=",", ndmin=2, dtype=np.float64)

    if not np.all(np.isfinite(header_values)):
        raise ValueError("SHADR header contains non-finite values")
    lmax = int(header_values[3])
    if (lmax < 0 or header_values[3] != lmax or header_values[4] != lmax):
        raise ValueError("SHADR header has invalid or inconsistent lmax values")
    if header_values[5] != 1.0:
        raise ValueError("SHADR coefficients must be normalized")

    degrees, orders = _validate_table(table, 6, "SHADR", lmax=lmax)
    if int(degrees.max()) != lmax:
        raise ValueError("SHADR data does not reach its declared lmax")
    if np.any(table[:, 4:] < 0):
        raise ValueError("SHADR uncertainties must be nonnegative")

    shape = (lmax + 1, lmax + 1)
    clm = np.zeros(shape, dtype=np.float64)
    slm = np.zeros(shape, dtype=np.float64)
    sigma_clm = np.zeros(shape, dtype=np.float64)
    sigma_slm = np.zeros(shape, dtype=np.float64)
    clm[degrees, orders] = table[:, 2]
    slm[degrees, orders] = table[:, 3]
    sigma_clm[degrees, orders] = table[:, 4]
    sigma_slm[degrees, orders] = table[:, 5]

    # PDS SHADR files conventionally omit the trivial (l=0, m=0) row (its
    # coefficient is normalized to exactly 1.0 by definition -- GM already
    # carries the degree-0 term). Only fill that convention in when the row
    # is absent; a file that *does* carry an explicit l=0 row is
    # unexpected for this format and must not be silently clobbered with the
    # convention value.
    if np.any(degrees == 0):
        raise ValueError(
            "SHADR data unexpectedly contains an explicit degree-0 (l=0) "
            "row; this loader's C00=1.0 normalization convention assumes "
            "l=0 is always omitted (its value is implicit in GM) -- refusing "
            "to silently overwrite an explicit row with that convention."
        )
    clm[0, 0] = 1.0

    return {
        "r0_m": float(header_values[0] * 1_000.0),
        "gm": float(header_values[1] * 1_000_000_000.0),
        "gm_sigma": float(header_values[2] * 1_000_000_000.0),
        "lmax": lmax,
        "clm": clm,
        "slm": slm,
        "sigma_clm": sigma_clm,
        "sigma_slm": sigma_slm,
    }


def load_shape(path) -> dict:
    """Load a plain or gzipped Wieczorek shape expansion in meters."""
    with _open_text(path) as stream:
        table = np.loadtxt(stream, delimiter=",", ndmin=2, dtype=np.float64)
    degrees, orders = _validate_table(table, 4, "shape")
    lmax = int(degrees.max())

    # Completeness check mirroring load_shadr's (that function checks its
    # header-declared lmax is actually reached; this format has no header,
    # so lmax is instead inferred from the data itself -- the analogous,
    # non-tautological check here is that every (n, m) pair with 0 <= m <= n
    # up to that inferred lmax is actually present, i.e. the file is a full
    # triangle, not merely that it reaches some maximum degree).
    expected_rows = (lmax + 1) * (lmax + 2) // 2
    if table.shape[0] != expected_rows:
        raise ValueError(
            f"shape data is incomplete: {table.shape[0]} rows present but "
            f"{expected_rows} expected for a full (n, m) triangle "
            f"(0 <= m <= n for each n) up to the inferred lmax={lmax}"
        )

    shape = (lmax + 1, lmax + 1)
    clm = np.zeros(shape, dtype=np.float64)
    slm = np.zeros(shape, dtype=np.float64)
    clm[degrees, orders] = table[:, 2]
    slm[degrees, orders] = table[:, 3]
    if not np.any((degrees == 0) & (orders == 0)):
        raise ValueError("shape data is missing C00")
    return {"lmax": lmax, "clm": clm, "slm": slm}


def truncate(coeffs_dict, lmax_new) -> dict:
    """Return an independent copy of a coefficient dictionary at lower lmax."""
    try:
        lmax_new = operator.index(lmax_new)
    except TypeError as exc:
        raise TypeError("lmax_new must be an integer") from exc
    if "lmax" not in coeffs_dict:
        raise KeyError("coefficient dictionary is missing lmax")
    lmax_old = operator.index(coeffs_dict["lmax"])
    if lmax_new < 0 or lmax_new > lmax_old:
        raise ValueError("lmax_new must be between zero and the current lmax")

    size = lmax_new + 1
    result = {}
    for key, value in coeffs_dict.items():
        if isinstance(value, np.ndarray):
            if value.ndim != 2 or min(value.shape) < size:
                raise ValueError(f"coefficient array {key!r} has invalid shape")
            result[key] = value[:size, :size].copy()
        else:
            result[key] = value
    result["lmax"] = lmax_new
    return result

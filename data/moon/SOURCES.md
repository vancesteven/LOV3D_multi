# Moon field data — sources and integrity

Fetched 2026-08-08. Do not modify these files; loaders live in
`pylov3d/sh_data.py` (same two functions used for Mars, `load_shadr` /
`load_shape` -- no Moon-specific loader code exists yet).

## grgm900c_120_sha.tab — GRGM900C GRAIL lunar gravity field, truncated to degree/order 120

- Product: GRGM900C (Goddard Recovery and Interior Laboratory Model 900,
  revision C), spherical harmonic coefficients of the lunar gravitational
  potential, PDS SHADR (SHAdr Data Record) plain-text format. Native product
  is degree/order 900; **this file is a local truncation to lmax=120** (see
  below) -- there is no PDS-published lmax=120 GRGM900C product, unlike
  Mars's GMM-3 which ships natively at lmax=120.
- Citation: Lemoine, F.G., Goossens, S., Sabaka, T.J., Nicholas, J.B.,
  Mazarico, E., Rowlands, D.D., Loomis, B.D., Chinn, D.S., Neumann, G.A.,
  Smith, D.E., Zuber, M.T. (2014), "GRGM900C: A degree 900 lunar gravity
  model from GRAIL primary and extended mission data," Geophysical Research
  Letters, 41, 3382-3389, doi:10.1002/2014GL060027.
- Original (untruncated, lmax=900) retrieved from the PDS Geosciences node:
  https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_0900c_sha.tab
  - Size: 49,574,944 bytes.
  - sha256: dab6ab06e0d3d7cbc594ea4bd03151a65534ed5fdf4f147ae38662428c04454e
  - This hash was cross-checked against pyshtools' own pinned hash for this
    exact URL (`pyshtools.datasets.Moon.GRGM900C`, in
    `pyshtools/datasets/Moon/__init__.py`) -- exact match, independent
    confirmation the download is bit-identical to what pyshtools ships.
- Truncation procedure (reproducible, no coefficient values altered): the
  file has no explicit degree-0 row (implicit C00=1.0, same PDS SHADR
  convention as Mars's GMM-3) and rows are already stored in ascending
  (degree, then order 0..degree) block order, so truncation is a straight
  prefix cut. Kept the header line with its two lmax fields (columns 4 and
  5) rewritten from 900 to 120; kept the first 7,380 data rows verbatim
  (degree 1 through 120, the exact full triangle size for that range,
  `(121*122)//2 - 1 = 7380`, verified by row count before truncating). r0
  and GM in the header are global model properties and are carried through
  unchanged -- truncating an SH gravity series to a lower lmax does not
  change the lower-degree coefficients or GM/r0, it only drops the
  higher-degree information.
  - Reproduce: `head -n 1 gggrx_0900c_sha.tab` with cols 4-5 changed
    `900 -> 120`, followed by `sed -n '2,7381p' gggrx_0900c_sha.tab`.
- SHADR header (row 1) of the truncated file: r0 = 1738.0 km,
  GM = 4902.799967088640 km^3/s^2, GM_sigma = 1.4178101463045039e-05
  km^3/s^2, lmax = 120 (both header lmax fields), coefficients
  4pi-normalized (normalization flag = 1).
- Quick check (loaded via `pylov3d.sh_data.load_shadr`): C20_bar =
  -9.0886616361343905e-05 (implies J2 = -C20*sqrt(5) = 2.0327e-4, matching
  the published GRAIL-era lunar J2 of ~2.0331e-4 to 4 sig figs). GM/G =
  7.3459e22 kg (G = 6.6743e-11), matching `pylov3d.bodies` catalog id 31
  "Moon" `Mass=7.3458e22` to 5 significant figures.
- sha256 (of grgm900c_120_sha.tab, as stored): 602bc8b672905283a9a666b4a37729c2c8067877f579baf485e5c98c17f29794

## MoonTopo719.shape.gz — LOLA shape model, degree/order 719 (reformatted + truncated from MoonTopo2600p)

- Product: spherical harmonic model of the shape (planetary radius) of the
  Moon, principal-axis coordinate system, derived from LOLA laser altimetry.
  Text rows "l, m, Clm, Slm" in meters, 4pi-normalized real harmonics,
  excluding the Condon-Shortley phase -- same convention as Mars's
  MarsTopo719.shape.
- Citations:
  - Wieczorek, M.A. (2015), "Spherical harmonic model of the shape of
    Earth's Moon: MoonTopo2600p," Zenodo, doi:10.5281/zenodo.3870924
    (publication_date 2015-04-20; single author, Mark A. Wieczorek).
  - Smith, D.E., Zuber, M.T., Jackson, G.B., et al. (2010), "The Lunar
    Orbiter Laser Altimeter Investigation on the Lunar Reconnaissance
    Orbiter Mission," Space Science Reviews, 150, 209-241,
    doi:10.1007/s11214-009-9512-y (LOLA instrument/data description).
- **No lmax=719 companion product exists in ASCII "l,m,Clm,Slm" .shape
  format for the Moon** (unlike Mars, where MarsTopo719.shape.gz is itself
  a separately-published Zenodo record, doi:10.5281/zenodo.6475460). The
  only lower-degree Moon shape products available are in pyshtools' newer
  archive (doi:10.5281/zenodo.10820774 / zenodo.11533784), and those ship
  in `.bshc` binary format, which `pylov3d.sh_data.load_shape` cannot
  parse -- so per the task's instruction to prefer a same-format smaller
  companion when one exists, and to fall back to a documented local
  truncation otherwise, this file was built locally from the only
  ASCII/.shape-format Moon archive that exists (the full lmax=2600
  MoonTopo2600p.shape.gz), truncated to lmax=719 to mirror Mars's
  MarsTopo719 in both name and rough file size.
- Original (untruncated, lmax=2600) retrieved from:
  https://zenodo.org/api/records/3870924/files/MoonTopo2600p.shape.gz/content
  - Size: 74,742,688 bytes.
  - md5: 646bbc12d6c440d6b9ab8f888c40deaf (matches Zenodo's published
    checksum for this file exactly).
  - sha256: 193146df894e2fef796df9d6142c78fae6fa5c183fd79d3f79eeb356602af69a
- **Format deviation a loader-test author must know**: the raw
  MoonTopo2600p.shape file (after `gunzip`) is **whitespace-delimited**
  (fixed-width columns, e.g. `   0  0  1737151.19826508  0.000000000000000`),
  *not* comma-delimited like MarsTopo719.shape. `pylov3d.sh_data.load_shape`
  calls `np.loadtxt(..., delimiter=",")` and will fail (or silently
  misparse into 1 column) on the raw file. MoonTopo719.shape.gz, as stored
  here, has already been **reformatted to comma-delimited** (`l, m, Clm,
  Slm`) during truncation -- the numeric values are verbatim (byte-for-byte
  copies of the original number tokens, including the original file's mixed
  plain-decimal / uppercase-`E`-exponent notation for different rows; numpy
  parses both forms fine), only whitespace runs became `", "`. No value was
  reformatted or rounded.
- Truncation procedure (reproducible): kept every row with degree l <= 719
  (259,560 rows total, the exact full triangle size for lmax=719,
  `(719+1)*(719+2)//2 = 259560`, verified against the row count actually
  written). Row order in the source file is already ascending by degree
  then order, so this is a straight filter, not a re-sort.
- Quick check (loaded via `pylov3d.sh_data.load_shape`): degree-0 term =
  1737151.19826508 m (mean radius 1737.1512 km) -- matches
  `pylov3d.moon.MOON["R"]` = 1737.151 km (`docs/MOON_MODEL.md`) to 6
  significant figures, an independent cross-check against a value that
  predates and shares no code with this fetch.
- sha256 (of MoonTopo719.shape.gz, as stored): 5f88c13f9f38b148b14bd6c8833a358fffdd210fffa18e0608e70ef88f40cdc4

## .gitignore

`data/moon` was added to the same whitelist-exception pattern as `data/mars`
(`data/*` is ignored by default; `!/data/moon` re-includes this directory).

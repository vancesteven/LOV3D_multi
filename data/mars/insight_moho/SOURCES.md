# InSight-calibrated Mars crustal thickness models — sources and integrity

Fetched 2026-08-11. These are **non-Airy** crust-mantle interface models,
committed to let the lateral-variation stage be re-run against a crustal
model derived independently of the Airy assumption. TASK-027 established
that crustal-model uncertainty exceeds truncation uncertainty by roughly
sevenfold, so this substitution addresses the dominant error term.

## Product

Spherical-harmonic coefficients of the **radius of the crust-mantle
interface** (not the thickness directly): crustal thickness at a point is
the MOLA surface radius minus the Moho radius. Files are
`n, m, Clm, Slm`, comma-delimited, in metres, 4pi-normalized without the
Condon-Shortley phase -- the same convention as `MarsTopo719.shape` and
readable by `pylov3d.sh_data.load_shape` unmodified. lmax = 90.

## Citation

Wieczorek, M. A., et al. (2022), "InSight constraints on the global
character of the Martian crust," *Journal of Geophysical Research:
Planets*, 127, e2022JE007298, doi:10.1029/2022JE007298. Supplemental
data archive: doi:10.5281/zenodo.6477509, licensed CC-BY-4.0. The paper
has 26 authors; retrieve the full list from Crossref if one is needed.

**Correction of record.** An earlier version of this file gave a partial
author list written from memory rather than retrieved. It contained a
name that does not appear on the paper ("Kim, J." -- the paper has one
Kim, and the entry appears to be a corruption of King, S. D.) and
silently skipped two authors, so it was not the paper's order either. It
is replaced above by the short form plus the paper's own DOI, both
verifiable. This is precisely the memory-written-citation failure mode
this project has ruled against, and it is recorded here rather than
quietly deleted.

## Retrieval

Source archive: https://zenodo.org/records/6477509 --
`InSight-Crustal-Thickness-Archive.tar.gz` (2.7 GB, md5
c051f38aef217e59c03b66c96c931026 as published by Zenodo). Only the
`constant/` spherical-harmonic files were extracted (streamed, never
stored whole); the bulk of the archive is precomputed rasters under
`grids/` that this project does not need.

## Model selection, and why these five

Filenames are `Moho-Mars-MODEL-THICK-RHO.sh`: MODEL is the assumed
mantle/core interior model, THICK the assumed seismic crustal thickness
at the InSight landing site [km], RHO the crustal density [kg/m^3].

All five committed models use **RHO = 2900**, matching this project's
`MARS["crust_density"]`, and were selected as those whose **mean crustal
thickness falls within 1 km of the 50 km reference** used by
`pylov3d.mars` -- so a substitution changes the lateral *pattern* while
holding mean thickness and crust density fixed. That makes the comparison
against the Airy field like-for-like rather than confounded by a
different mean.

The five span distinct interior models, so the spread across them is an
**interior-model sensitivity**, not a single alternative.

| File | InSight thickness [km] | mean crustal thickness [km] |
|---|---|---|
| `Moho-Mars-DWAK-33-2900.sh` | 33 | 49.81 |
| `Moho-Mars-DWThotCrust1-36-2900.sh` | 36 | 49.59 |
| `Moho-Mars-EH45Tcold-36-2900.sh` | 36 | 50.35 |
| `Moho-Mars-Khan2022-33-2900.sh` | 33 | 50.47 |
| `Moho-Mars-EH45TcoldCrust1r-36-2900.sh` | 36 | 50.48 |

`Moho-Mars-DWThotCrust1r-36-2900.sh` was fetched but **deliberately not
committed**: it is byte-identical to the non-`r` variant (same sha256),
so including it would have double-weighted that model in the spread.

Mean thicknesses above were computed on machine A as
`MarsTopo719 C00 - Moho C00`, both loaded through
`pylov3d.sh_data.load_shape`.

## sha256 of the committed files

- `Moho-Mars-DWAK-33-2900.sh` : `a0e03ca7575e854523879155a62ea40c5718dd2d5da8a762ca348a3441ca2d19`
- `Moho-Mars-DWThotCrust1-36-2900.sh` : `1284d37bde2e4479cb3244782bc79422ee08750232d676e03c5be7692f84d4c4`
- `Moho-Mars-EH45Tcold-36-2900.sh` : `0251a8b7a9e1112faedf4ecc3f8ec551d4841511e012c4ee197dfa37b5ec345c`
- `Moho-Mars-EH45TcoldCrust1r-36-2900.sh` : `0198f26eb7ef6dde607b969cc82fb194904745460052e682c3c00db1ee28c34d`
- `Moho-Mars-Khan2022-33-2900.sh` : `3b2bd2921086c3829f2efe91f0bc2409aea167a6515eaf158180c32fd58870d3`
- `ARCHIVE_README.txt` : `244ed44498a6320f8b22a78183cd97f95f950a7f182bace76a8cbe0a460a98cd`

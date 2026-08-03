# Mars field data — sources and integrity

Fetched 2026-08-04 by Machine A (the primary PDS Geosciences node was down;
mirrors used are noted). Do not modify these files; loaders live in
`pylov3d/sh_data.py` (TASK-013).

## gmm3_120_sha.tab — GMM-3 Mars gravity field, degree/order 120

- Product: GMM-3 (Goddard Mars Model 3), spherical harmonic coefficients,
  PDS SHADR (SHAdr Data Record) plain-text format.
- Citation: Genova, A., Goossens, S., Lemoine, F.G., Mazarico, E.,
  Neumann, G.A., Smith, D.E., Zuber, M.T. (2016), "Seasonal and static
  gravity field of Mars from MGS, Mars Odyssey and MRO radio science,"
  Icarus, 272, 228-245, doi:10.1016/j.icarus.2016.02.050.
- Retrieved from (mirror of PDS mrors_1xxx archive, via pyshtools'
  dataset registry):
  https://pds-geosciences.wustl.edu/mro/mro-m-rss-5-sdp-v1/mrors_1xxx/data/shadr/gmm3_120_sha.tab
- SHADR header (row 1): r0 = 3396.0 km, GM = 42828.372854187757 km^3/s^2,
  lmax = 120, coefficients 4pi-normalized.
- Quick check: C20_bar = -8.750211323545289e-4
  (= -J2/sqrt(5) with J2 = 1.9566e-3).
- sha256: eb4913b1afb6682406e6a9dad5be7918a162fa8462473c9a2e7aae258d4c2c9c

NOTE: the 1D bulk-constraint model (docs/MARS_MODEL.md) uses GM from
Konopliv et al. (2016) MRO120D (42828.375 km^3/s^2); GMM-3's GM differs by
~2e-6 relative — irrelevant at our tolerances, but the two products should
not be silently mixed for the same quantity.

## MarsTopo719.shape.gz — MOLA shape model, degree/order 719

- Product: MarsTopo719, spherical harmonic model of the shape (planetary
  radius) of Mars, derived from MOLA; text rows "l, m, Clm, Slm" in meters,
  4pi-normalized real harmonics.
- Citation: Wieczorek, M.A. (2022), "Spherical harmonic model of the shape
  of Mars: MarsTopo719," Zenodo, doi:10.5281/zenodo.6475460 (subsampled
  from MarsTopo2600, Wieczorek 2015, doi:10.5281/zenodo.3870922).
- Retrieved from:
  https://zenodo.org/api/records/6475460/files/MarsTopo719.shape.gz/content
- Quick check: degree-0 term = 3.38950012207057e6 m (mean radius 3389.5 km).
- sha256 (of the .gz as stored): 37a98efae5eab7c85260f4b43315fe9fcf44247a61581bed1b6f7f10f79adea0

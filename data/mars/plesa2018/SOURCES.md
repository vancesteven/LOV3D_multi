# Plesa et al. (2018) Mars thermal field

TASK-043 uses Data Set S1 from:

- A.-C. Plesa et al. (2018), *The thermal state and interior structure of
  Mars*, Geophysical Research Letters, DOI
  [10.1029/2018GL080728](https://doi.org/10.1029/2018GL080728).
- TU Berlin DepositOnce item
  [8fe0416f-1ad5-4953-8027-d0374e07e42d](https://depositonce.tu-berlin.de/items/8fe0416f-1ad5-4953-8027-d0374e07e42d),
  archive DOI [10.14279/depositonce-9562](https://doi.org/10.14279/depositonce-9562).
- ORIGINAL-bundle bitstream UUID:
  `8f4cb632-ac48-4534-87d7-e0a1974b02bb`.
- Stored filename: `grl58258-sup-0002-data`.
- Size: 7,841,309 bytes.
- Archive MD5: `47bab533418619fa8da74c99e9a4e6d1`.
- Independently computed SHA-256:
  `88c80be18a4a4bef411c18218ead2f8019c8bf33e96ec71e1a42a5788b9ed1ee`.

Run `python scripts/fetch_mars_thermal_data.py` from the repository root to
download and verify the table. The raw file is intentionally ignored rather
than redistributed: the repository record says open access, but also records
AGU copyright and an in-copyright RightsStatements URI, so downstream file
redistribution has not been assumed.

## Data contract

Data Set S1 is a whitespace-delimited 64,800-by-8 table after eight comment
lines. Its columns are:

1. longitude [degrees east]
2. latitude [degrees]
3. crustal thickness [km]
4. surface heat flow [mW/m2]
5. elastic thickness at strain rate 1e-14 1/s [km]
6. elastic thickness at strain rate 1e-17 1/s [km]
7. temperature at 150 km depth [K]
8. depth to the 1370 K isotherm [km]

The registered grid is global at one-degree spacing: longitude 0 through 359
degrees east and latitude 90 through -89 degrees. Rows at the geographic pole
repeat the same physical point at every longitude, as expected for this
rectangular storage convention.

The source table does **not** contain a lateral temperature slice at 400 km.
TASK-043 therefore uses the archived 150 km field and records this as a
source-driven change from the provisional depth in TASK-042. The pilot treats
the resulting horizontal template as constant through the model's upper-
mantle layer; that is a test hypothesis, not a claim that the source field is
radially uniform.

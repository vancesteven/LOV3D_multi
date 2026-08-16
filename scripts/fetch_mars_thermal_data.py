#!/usr/bin/env python3
"""Fetch and verify Plesa et al. (2018) Mars Data Set S1.

The raw table is intentionally not committed; see
``data/mars/plesa2018/SOURCES.md`` for provenance and rights notes.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


BITSTREAM_UUID = "8f4cb632-ac48-4534-87d7-e0a1974b02bb"
URL = (
    "https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/"
    f"{BITSTREAM_UUID}/content"
)
EXPECTED_SIZE = 7_841_309
EXPECTED_MD5 = "47bab533418619fa8da74c99e9a4e6d1"
EXPECTED_SHA256 = (
    "88c80be18a4a4bef411c18218ead2f8019c8bf33e96ec71e1a42a5788b9ed1ee"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "mars"
    / "plesa2018"
    / "grl58258-sup-0002-data"
)


def checksums(path: Path) -> tuple[int, str, str]:
    """Return ``(size, md5, sha256)`` while streaming *path* once."""
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return size, md5.hexdigest(), sha256.hexdigest()


def verify(path: Path) -> None:
    """Raise ``ValueError`` unless *path* matches the registered artifact."""
    size, md5, sha256 = checksums(path)
    failures = []
    if size != EXPECTED_SIZE:
        failures.append(f"size {size} != {EXPECTED_SIZE}")
    if md5 != EXPECTED_MD5:
        failures.append(f"MD5 {md5} != {EXPECTED_MD5}")
    if sha256 != EXPECTED_SHA256:
        failures.append(f"SHA-256 {sha256} != {EXPECTED_SHA256}")
    if failures:
        raise ValueError("; ".join(failures))


def fetch(output: Path, *, force: bool = False) -> Path:
    """Download atomically to *output*, then verify both registered hashes."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        verify(output)
        return output

    partial = output.with_name(output.name + ".partial")
    try:
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as sink:
            while chunk := response.read(1024 * 1024):
                sink.write(chunk)
        verify(partial)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="replace an existing file")
    parser.add_argument(
        "--verify-only", action="store_true", help="verify an existing file without network access"
    )
    args = parser.parse_args(argv)

    try:
        if args.verify_only:
            verify(args.output)
            path = args.output
        else:
            path = fetch(args.output, force=args.force)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    size, md5, sha256 = checksums(path)
    print(f"verified {path}")
    print(f"size={size} md5={md5} sha256={sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

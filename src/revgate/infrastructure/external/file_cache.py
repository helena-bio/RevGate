# FileCache -- SHA-256 content-addressed local cache for downloaded data sources.
# All data downloads go through this cache to avoid repeated network requests.

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".revgate" / "cache"


class FileCache:
    """SHA-256 content-addressed local file cache.

    Directory layout:
        ~/.revgate/cache/
            depmap/
                CRISPRGeneEffect.csv
                Model.csv
            tcga/
                SKCM_expression.parquet
                SKCM_clinical.parquet
            string/
                9606.protein.links.v12.0.txt
            gnomad/
                gnomad.v4.constraint.tsv
            checksums.json

    Checksums are stored in checksums.json for validation.
    Cache invalidation is manual (delete cache directory) or
    version-based (pass expected_sha256 to get()).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._checksum_file = self._cache_dir / "checksums.json"
        self._checksums: dict[str, str] = self._load_checksums()

    def get(self, key: str, expected_sha256: str | None = None) -> Path | None:
        """Return cached file path if it exists and passes checksum.

        Args:
            key:             Cache key (e.g. 'depmap/CRISPRGeneEffect.csv').
            expected_sha256: If provided, validate against stored checksum.

        Returns:
            Path to cached file, or None if not cached or checksum mismatch.
        """
        path = self._cache_dir / key

        if not path.exists():
            return None

        if expected_sha256 is not None:
            stored = self._checksums.get(key)
            if stored != expected_sha256:
                return None

        return path

    def put(self, key: str, data: bytes) -> Path:
        """Write data to cache and record its SHA-256 checksum.

        Args:
            key:  Cache key (e.g. 'depmap/CRISPRGeneEffect.csv').
            data: Raw bytes to cache.

        Returns:
            Path to the written cache file.
        """
        path = self._cache_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        sha256 = self._compute_sha256(data)
        self._checksums[key] = sha256
        self._save_checksums()

        return path

    def exists(self, key: str) -> bool:
        """Return True if a cache entry exists for the given key."""
        return (self._cache_dir / key).exists()

    def invalidate(self, key: str) -> None:
        """Remove a cache entry and its checksum record."""
        path = self._cache_dir / key
        if path.exists():
            path.unlink()
        self._checksums.pop(key, None)
        self._save_checksums()

    def path_for(self, key: str) -> Path:
        """Return the expected path for a cache key (may not exist yet)."""
        return self._cache_dir / key

    def _compute_sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _load_checksums(self) -> dict[str, str]:
        if self._checksum_file.exists():
            try:
                return json.loads(self._checksum_file.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_checksums(self) -> None:
        self._checksum_file.write_text(
            json.dumps(self._checksums, indent=2, sort_keys=True)
        )

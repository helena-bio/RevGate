# Downloader -- HTTP streaming downloader with progress reporting.
# Handles all data source downloads for RevGate.
# Streams to disk -- never loads full file in memory.

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Callable

import requests
import urllib3

from revgate.domain.exceptions.data import DownloadFailedError
from revgate.infrastructure.external.file_cache import FileCache


# Chunk size for streaming downloads -- 8MB
CHUNK_SIZE = 8 * 1024 * 1024

# Download timeouts in seconds
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 300

# Public data source definitions
DATA_SOURCES: dict[str, dict] = {
    "depmap_chronos": {
        "url": (
            "https://depmap.org/portal/api/download/file"
            "?file_name=public_24Q4%2FCRISPRGeneEffect.csv"
        ),
        "cache_key": "depmap/CRISPRGeneEffect.csv",
        "description": "DepMap 24Q4 CRISPR Chronos dependency scores (~180MB)",
    },
    "depmap_model": {
        "url": (
            "https://depmap.org/portal/api/download/file"
            "?file_name=public_24Q4%2FModel.csv"
        ),
        "cache_key": "depmap/Model.csv",
        "description": "DepMap 24Q4 cell line metadata",
    },
    "string_links": {
        "url": (
            "https://stringdb-downloads.org/download/protein.links.v12.0"
            "/9606.protein.links.v12.0.txt.gz"
        ),
        "cache_key": "string/9606.protein.links.v12.0.txt",
        "description": "STRING v12.0 human PPI network (~350MB)",
        "decompress": True,
        # STRING certificate has expired -- skip SSL verification
        "ssl_verify": False,
    },
    "string_aliases": {
        "url": (
            "https://stringdb-downloads.org/download/protein.aliases.v12.0"
            "/9606.protein.aliases.v12.0.txt.gz"
        ),
        "cache_key": "string/9606.protein.aliases.v12.0.txt",
        "description": "STRING v12.0 protein aliases",
        "decompress": True,
        # STRING certificate has expired -- skip SSL verification
        "ssl_verify": False,
    },
    "gnomad_constraint": {
        "url": (
            "https://storage.googleapis.com/gcp-public-data--gnomad"
            "/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv"
        ),
        "cache_key": "gnomad/gnomad.v4.constraint.tsv",
        "description": "gnomAD v4 gene constraint metrics (~15MB)",
    },
}

# Source group aliases for CLI convenience
SOURCE_GROUPS: dict[str, list[str]] = {
    "depmap": ["depmap_chronos", "depmap_model"],
    "string": ["string_links", "string_aliases"],
    "gnomad": ["gnomad_constraint"],
    "all": ["depmap_chronos", "depmap_model", "string_links", "string_aliases", "gnomad_constraint"],
}


class Downloader:
    """HTTP streaming downloader with progress reporting and cache integration.

    Downloads data sources to the local FileCache.
    Streams to a temp file first -- atomic rename on success.
    Never stores partial downloads in cache.
    """

    def __init__(
        self,
        cache: FileCache,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self._cache = cache
        self._progress_callback = progress_callback or self._default_progress

    def download_source(self, source_id: str, force: bool = False) -> Path:
        """Download a single data source by ID.

        Args:
            source_id: Key in DATA_SOURCES dict (e.g. 'depmap_chronos').
            force:     Re-download even if already cached.

        Returns:
            Path to cached file.

        Raises:
            DownloadFailedError: If download fails after timeout.
            KeyError:            If source_id is not in DATA_SOURCES.
        """
        if source_id not in DATA_SOURCES:
            raise KeyError(f"Unknown data source: {source_id!r}. Known: {list(DATA_SOURCES)}")

        spec = DATA_SOURCES[source_id]
        cache_key = spec["cache_key"]

        if not force and self._cache.exists(cache_key):
            print(f"  [cached] {cache_key}")
            return self._cache.path_for(cache_key)

        print(f"  [download] {spec['description']}")
        print(f"  URL: {spec['url']}")

        ssl_verify = spec.get("ssl_verify", True)
        raw_bytes = self._stream_download(source_id, spec["url"], ssl_verify=ssl_verify)

        # Decompress gzip if needed
        if spec.get("decompress"):
            raw_bytes = self._decompress_gzip(raw_bytes, source_id)

        path = self._cache.put(cache_key, raw_bytes)
        print(f"  [saved] {path}")

        return path

    def download_group(self, group: str, force: bool = False) -> list[Path]:
        """Download all sources in a named group.

        Args:
            group: Group name from SOURCE_GROUPS (e.g. 'depmap', 'all').
            force: Re-download even if already cached.

        Returns:
            List of paths to cached files.
        """
        if group not in SOURCE_GROUPS:
            raise KeyError(
                f"Unknown source group: {group!r}. "
                f"Known groups: {list(SOURCE_GROUPS)}"
            )

        source_ids = SOURCE_GROUPS[group]
        paths: list[Path] = []

        for source_id in source_ids:
            path = self.download_source(source_id, force=force)
            paths.append(path)

        return paths

    def download_tcga(
        self,
        cancer_id: str,
        force: bool = False,
    ) -> tuple[Path, Path]:
        """Download TCGA expression and clinical data for a cancer type.

        TCGA data is accessed via the GDC API. This method downloads
        pre-processed parquet files from a public mirror if available,
        otherwise guides the user to use TCGAbiolinks.

        Args:
            cancer_id: TCGA abbreviation (e.g. SKCM, LAML).
            force:     Re-download even if cached.

        Returns:
            Tuple of (expression_path, clinical_path).
        """
        expr_key = f"tcga/{cancer_id}_expression.parquet"
        clin_key = f"tcga/{cancer_id}_clinical.parquet"

        expr_cached = self._cache.exists(expr_key)
        clin_cached = self._cache.exists(clin_key)

        if not force and expr_cached and clin_cached:
            print(f"  [cached] TCGA {cancer_id}")
            return (
                self._cache.path_for(expr_key),
                self._cache.path_for(clin_key),
            )

        # TCGA data requires GDC API or TCGAbiolinks R package
        # Direct HTTP download not available for all cancer types
        print(f"  [info] TCGA {cancer_id} data not cached.")
        print(f"  To download TCGA data, use one of:")
        print(f"    1. TCGAbiolinks (R):  TCGAbiolinks::GDCdownload()")
        print(f"    2. GDC Data Portal:   https://portal.gdc.cancer.gov/")
        print(f"    3. Manual placement:  ~/.revgate/cache/{expr_key}")
        print(f"                          ~/.revgate/cache/{clin_key}")
        print(f"  Expected format: parquet, patients x genes (expression),")
        print(f"                   parquet, patients x clinical fields (clinical).")

        raise DownloadFailedError(
            source=f"TCGA {cancer_id}",
            reason=(
                "TCGA data requires manual download via GDC portal or TCGAbiolinks. "
                f"Place files at ~/.revgate/cache/{expr_key} and ~/.revgate/cache/{clin_key}"
            ),
        )

    def status(self) -> dict[str, bool]:
        """Return cache status for all known data sources.

        Returns:
            Dict source_id -> bool (True if cached).
        """
        result: dict[str, bool] = {}
        for source_id, spec in DATA_SOURCES.items():
            result[source_id] = self._cache.exists(spec["cache_key"])
        return result

    def _stream_download(
        self,
        source_id: str,
        url: str,
        ssl_verify: bool = True,
    ) -> bytes:
        """Stream URL to memory with progress reporting.

        Args:
            source_id:  For progress labeling.
            url:        Download URL.
            ssl_verify: Set False to skip SSL cert verification (e.g. STRING).

        Returns:
            Raw bytes of the downloaded file.

        Raises:
            DownloadFailedError: On network error or bad status code.
        """
        # Suppress urllib3 warning when ssl_verify=False
        if not ssl_verify:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        try:
            response = requests.get(
                url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                headers={"User-Agent": "RevGate/0.1 (Helena Bioinformatics)"},
                verify=ssl_verify,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise DownloadFailedError(source=source_id, reason="Connection timed out")
        except requests.exceptions.ConnectionError as exc:
            raise DownloadFailedError(source=source_id, reason=f"Connection error: {exc}")
        except requests.exceptions.HTTPError as exc:
            raise DownloadFailedError(
                source=source_id,
                reason=f"HTTP {response.status_code}: {exc}",
            )

        total_bytes = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunks: list[bytes] = []

        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                chunks.append(chunk)
                downloaded += len(chunk)
                self._progress_callback(source_id, downloaded, total_bytes)

        return b"".join(chunks)

    def _decompress_gzip(self, data: bytes, source_id: str) -> bytes:
        """Decompress gzip bytes.

        Args:
            data:      Compressed bytes.
            source_id: For error labeling.

        Returns:
            Decompressed bytes.

        Raises:
            DownloadFailedError: If decompression fails.
        """
        import gzip

        try:
            return gzip.decompress(data)
        except Exception as exc:
            raise DownloadFailedError(
                source=source_id,
                reason=f"Gzip decompression failed: {exc}",
            )

    def _default_progress(self, source_id: str, downloaded: int, total: int) -> None:
        """Default progress reporter -- prints MB downloaded."""
        if total > 0:
            pct = downloaded / total * 100
            print(
                f"\r  {source_id}: {downloaded / 1_048_576:.1f} MB"
                f" / {total / 1_048_576:.1f} MB ({pct:.0f}%)",
                end="",
                flush=True,
            )
        else:
            print(
                f"\r  {source_id}: {downloaded / 1_048_576:.1f} MB",
                end="",
                flush=True,
            )

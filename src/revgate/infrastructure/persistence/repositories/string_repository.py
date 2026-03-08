# STRINGRepository -- concrete implementation of NetworkRepository port.
# Downloads and caches STRING v12.0 PPI network.

from __future__ import annotations

import networkx as nx
import pandas as pd

from revgate.domain.entities.gene import Gene
from revgate.domain.exceptions.data import DataNotAvailableError
from revgate.infrastructure.external.file_cache import FileCache


STRING_URL = (
    "https://stringdb-downloads.org/download/protein.links.v12.0"
    "/9606.protein.links.v12.0.txt.gz"
)
STRING_ALIASES_URL = (
    "https://stringdb-downloads.org/download/protein.aliases.v12.0"
    "/9606.protein.aliases.v12.0.txt.gz"
)

STRING_CACHE_KEY = "string/9606.protein.links.v12.0.txt"
STRING_ALIASES_CACHE_KEY = "string/9606.protein.aliases.v12.0.txt"

# Default minimum combined score per HLD spec
DEFAULT_MIN_SCORE = 700


class STRINGRepository:
    """Concrete implementation of NetworkRepository.

    Builds a weighted networkx Graph from STRING v12.0 human PPI data.
    Node identifiers are gene symbols (mapped via aliases file).
    """

    def __init__(self, cache: FileCache) -> None:
        self._cache = cache
        self._graph: nx.Graph | None = None
        self._current_min_score: int | None = None

    def get_ppi_graph(self, min_score: int = DEFAULT_MIN_SCORE) -> nx.Graph:
        """Return STRING PPI graph filtered by combined score.

        Args:
            min_score: Minimum STRING combined score (default 700).

        Returns:
            networkx.Graph with gene symbols as nodes,
            combined_score as edge weight.
        """
        if self._graph is None or self._current_min_score != min_score:
            self._graph = self._build_graph(min_score)
            self._current_min_score = min_score

        return self._graph

    def get_degree_centrality(self, gene: Gene) -> int:
        """Return raw degree of a gene in the PPI network.

        Args:
            gene: Gene entity to look up.

        Returns:
            Degree (number of interaction partners).
            Returns 0 if gene is not in the network.
        """
        graph = self.get_ppi_graph()
        if gene.symbol in graph:
            return graph.degree(gene.symbol)
        return 0

    def _build_graph(self, min_score: int) -> nx.Graph:
        """Build networkx Graph from STRING links file."""
        links_path = self._cache.get(STRING_CACHE_KEY)

        if links_path is None:
            raise DataNotAvailableError(
                "STRING v12.0 PPI data not cached. "
                "Run: revgate download --source string"
            )

        df = pd.read_csv(links_path, sep=" ")
        df = df[df["combined_score"] >= min_score]

        # Load protein -> gene symbol alias mapping
        aliases = self._load_aliases()

        graph = nx.Graph()

        for _, row in df.iterrows():
            p1 = aliases.get(str(row["protein1"]), str(row["protein1"]))
            p2 = aliases.get(str(row["protein2"]), str(row["protein2"]))
            score = int(row["combined_score"])
            graph.add_edge(p1, p2, weight=score)

        return graph

    def _load_aliases(self) -> dict[str, str]:
        """Load STRING protein ID -> gene symbol mapping."""
        aliases_path = self._cache.get(STRING_ALIASES_CACHE_KEY)

        if aliases_path is None:
            return {}

        df = pd.read_csv(aliases_path, sep="\t", header=0)

        # Keep only BLAST_UniProt_GN_Name aliases (gene symbol column)
        if "source" in df.columns:
            df = df[df["source"] == "Ensembl_HGNC_symbol"]

        mapping: dict[str, str] = {}
        for _, row in df.iterrows():
            protein_id = str(row.get("#string_protein_id", ""))
            alias = str(row.get("alias", ""))
            if protein_id and alias:
                mapping[protein_id] = alias

        return mapping

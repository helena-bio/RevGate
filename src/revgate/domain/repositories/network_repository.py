# Port interface for STRING PPI network data access.
# Infrastructure layer implements this protocol.

from typing import Protocol, runtime_checkable

import networkx as nx

from revgate.domain.entities.gene import Gene


@runtime_checkable
class NetworkRepository(Protocol):
    """Abstract interface for STRING v12.0 protein-protein interaction network.

    The concrete implementation downloads and caches the human
    STRING PPI file (9606.protein.links.v12.0.txt) and builds
    a weighted networkx graph.

    Methods:
        get_ppi_graph        -- full STRING graph filtered by min_score
        get_degree_centrality -- degree centrality for a single gene
    """

    def get_ppi_graph(self, min_score: int = 700) -> nx.Graph:
        """Return the STRING PPI graph filtered by combined score.

        Args:
            min_score: minimum STRING combined score to include an edge.
                       Default 700 per HLD specification.

        Returns:
            networkx.Graph with protein symbols as nodes,
            combined_score as edge weight.
        """
        ...

    def get_degree_centrality(self, gene: Gene) -> int:
        """Return the degree centrality (raw degree) for a gene.

        Args:
            gene: Gene entity to look up.

        Returns:
            Number of interaction partners in the STRING network
            at the configured score threshold.
            Returns 0 if gene is not found in the network.
        """
        ...

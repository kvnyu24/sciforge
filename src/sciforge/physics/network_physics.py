"""
Network and Percolation Physics module

This module implements:
- Network models (random graphs, small-world, scale-free)
- Network analysis (centrality, communities)
- Percolation theory (site, bond, thresholds)
"""

import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Callable, Dict, List, Tuple, Set, Union
from dataclasses import dataclass
from collections import deque
import heapq


class RandomGraph:
    """
    Erdős-Rényi random graph model.

    G(n, p): n nodes, each edge exists with probability p
    G(n, m): n nodes, exactly m edges

    Args:
        n_nodes: Number of nodes
        p: Edge probability (for G(n,p) model)
        n_edges: Number of edges (for G(n,m) model)
    """

    def __init__(
        self,
        n_nodes: int,
        p: Optional[float] = None,
        n_edges: Optional[int] = None
    ):
        self.n = n_nodes
        self.p = p
        self.m = n_edges

        if p is None and n_edges is None:
            raise ValueError("Must specify either p or n_edges")

        # Adjacency list representation
        self.adjacency: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
        self._generate()

    def _generate(self):
        """Generate random graph."""
        if self.p is not None:
            # G(n, p) model
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if np.random.random() < self.p:
                        self.adjacency[i].add(j)
                        self.adjacency[j].add(i)
        else:
            # G(n, m) model
            edges_added = 0
            max_edges = self.n * (self.n - 1) // 2

            if self.m > max_edges:
                raise ValueError(f"Cannot have more than {max_edges} edges")

            while edges_added < self.m:
                i = np.random.randint(0, self.n)
                j = np.random.randint(0, self.n)
                if i != j and j not in self.adjacency[i]:
                    self.adjacency[i].add(j)
                    self.adjacency[j].add(i)
                    edges_added += 1

    def degree(self, node: int) -> int:
        """Get degree of a node."""
        return len(self.adjacency[node])

    def degree_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute degree distribution.

        Returns:
            Tuple of (degrees, frequencies)
        """
        degrees = [self.degree(i) for i in range(self.n)]
        unique, counts = np.unique(degrees, return_counts=True)
        return unique, counts / self.n

    def average_degree(self) -> float:
        """Compute average degree."""
        return sum(self.degree(i) for i in range(self.n)) / self.n

    def clustering_coefficient(self, node: Optional[int] = None) -> float:
        """
        Compute clustering coefficient.

        Args:
            node: Specific node (None = global average)

        Returns:
            Clustering coefficient
        """
        if node is not None:
            neighbors = list(self.adjacency[node])
            k = len(neighbors)
            if k < 2:
                return 0.0

            triangles = 0
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if n2 in self.adjacency[n1]:
                        triangles += 1

            return 2 * triangles / (k * (k - 1))
        else:
            # Global average
            return np.mean([self.clustering_coefficient(i) for i in range(self.n)])

    def largest_component_size(self) -> int:
        """Find size of largest connected component."""
        visited = set()
        max_size = 0

        for start in range(self.n):
            if start in visited:
                continue

            # BFS
            component_size = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component_size += 1
                for neighbor in self.adjacency[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            max_size = max(max_size, component_size)

        return max_size

    def is_connected(self) -> bool:
        """Check if graph is connected."""
        return self.largest_component_size() == self.n

    @staticmethod
    def critical_probability(n: int) -> float:
        """
        Critical probability for giant component emergence.

        p_c = 1/n

        Args:
            n: Number of nodes

        Returns:
            Critical probability
        """
        return 1.0 / n


class SmallWorldNetwork:
    """
    Watts-Strogatz small-world network.

    Starts with ring lattice and rewires edges with probability p.

    Args:
        n_nodes: Number of nodes
        k_neighbors: Number of neighbors in ring (must be even)
        rewire_prob: Probability of rewiring each edge
    """

    def __init__(self, n_nodes: int, k_neighbors: int, rewire_prob: float):
        if k_neighbors % 2 != 0:
            raise ValueError("k_neighbors must be even")
        if k_neighbors >= n_nodes:
            raise ValueError("k_neighbors must be less than n_nodes")

        self.n = n_nodes
        self.k = k_neighbors
        self.p = rewire_prob

        self.adjacency: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
        self._generate()

    def _generate(self):
        """Generate small-world network."""
        # Start with ring lattice
        for i in range(self.n):
            for j in range(1, self.k // 2 + 1):
                neighbor = (i + j) % self.n
                self.adjacency[i].add(neighbor)
                self.adjacency[neighbor].add(i)

        # Rewire edges
        for i in range(self.n):
            for j in range(1, self.k // 2 + 1):
                if np.random.random() < self.p:
                    neighbor = (i + j) % self.n
                    if neighbor in self.adjacency[i]:
                        # Remove old edge
                        self.adjacency[i].remove(neighbor)
                        self.adjacency[neighbor].remove(i)

                        # Add new random edge
                        new_neighbor = np.random.randint(0, self.n)
                        while new_neighbor == i or new_neighbor in self.adjacency[i]:
                            new_neighbor = np.random.randint(0, self.n)

                        self.adjacency[i].add(new_neighbor)
                        self.adjacency[new_neighbor].add(i)

    def average_path_length(self, n_samples: int = 1000) -> float:
        """
        Estimate average shortest path length.

        Args:
            n_samples: Number of node pairs to sample

        Returns:
            Average path length
        """
        total_length = 0
        count = 0

        for _ in range(n_samples):
            i = np.random.randint(0, self.n)
            j = np.random.randint(0, self.n)
            if i != j:
                length = self._shortest_path_length(i, j)
                if length is not None:
                    total_length += length
                    count += 1

        return total_length / count if count > 0 else float('inf')

    def _shortest_path_length(self, start: int, end: int) -> Optional[int]:
        """BFS shortest path."""
        if start == end:
            return 0

        visited = {start}
        queue = deque([(start, 0)])

        while queue:
            node, dist = queue.popleft()
            for neighbor in self.adjacency[node]:
                if neighbor == end:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        return None  # No path exists

    def clustering_coefficient(self) -> float:
        """Compute global clustering coefficient."""
        cc_sum = 0.0
        for node in range(self.n):
            neighbors = list(self.adjacency[node])
            k = len(neighbors)
            if k < 2:
                continue

            triangles = 0
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if n2 in self.adjacency[n1]:
                        triangles += 1

            cc_sum += 2 * triangles / (k * (k - 1))

        return cc_sum / self.n

    def small_world_coefficient(self, n_random: int = 10) -> float:
        """
        Compute small-world coefficient σ = (C/C_r) / (L/L_r).

        σ > 1 indicates small-world behavior.

        Args:
            n_random: Number of random graphs to average

        Returns:
            Small-world coefficient
        """
        # This network's properties
        C = self.clustering_coefficient()
        L = self.average_path_length()

        # Average over random graphs with same n, m
        n_edges = sum(len(self.adjacency[i]) for i in range(self.n)) // 2
        C_r_sum = 0.0
        L_r_sum = 0.0

        for _ in range(n_random):
            rg = RandomGraph(self.n, n_edges=n_edges)
            C_r_sum += rg.clustering_coefficient()
            # Simple path length estimate
            L_r_sum += np.log(self.n) / np.log(2 * n_edges / self.n)

        C_r = C_r_sum / n_random
        L_r = L_r_sum / n_random

        if C_r < 1e-10 or L < 1e-10:
            return 0.0

        return (C / C_r) / (L / L_r)


class ScaleFreeNetwork:
    """
    Barabási-Albert scale-free network.

    Uses preferential attachment: new nodes connect preferentially
    to high-degree nodes.

    Args:
        n_nodes: Final number of nodes
        m_edges: Edges to add per new node
        m0: Initial complete graph size
    """

    def __init__(self, n_nodes: int, m_edges: int, m0: Optional[int] = None):
        self.n = n_nodes
        self.m = m_edges

        if m0 is None:
            m0 = m_edges + 1
        self.m0 = m0

        if m_edges > m0:
            raise ValueError("m_edges must be <= m0")

        self.adjacency: Dict[int, Set[int]] = {i: set() for i in range(n_nodes)}
        self._generate()

    def _generate(self):
        """Generate scale-free network via preferential attachment."""
        # Start with complete graph on m0 nodes
        for i in range(self.m0):
            for j in range(i + 1, self.m0):
                self.adjacency[i].add(j)
                self.adjacency[j].add(i)

        # Degree list for preferential attachment
        degrees = [self.m0 - 1] * self.m0 + [0] * (self.n - self.m0)

        # Add remaining nodes
        for new_node in range(self.m0, self.n):
            # Choose m targets by preferential attachment
            total_degree = sum(degrees[:new_node])
            if total_degree == 0:
                targets = list(range(min(self.m, new_node)))
            else:
                probs = np.array(degrees[:new_node]) / total_degree
                targets = np.random.choice(
                    new_node, size=self.m, replace=False, p=probs
                )

            for target in targets:
                self.adjacency[new_node].add(target)
                self.adjacency[target].add(new_node)
                degrees[new_node] += 1
                degrees[target] += 1

    def degree_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute degree distribution.

        For BA networks: P(k) ~ k^{-γ} with γ ≈ 3

        Returns:
            Tuple of (degrees, frequencies)
        """
        degrees = [len(self.adjacency[i]) for i in range(self.n)]
        unique, counts = np.unique(degrees, return_counts=True)
        return unique, counts / self.n

    def fit_power_law(self) -> Tuple[float, float]:
        """
        Fit power law to degree distribution.

        P(k) ~ k^{-γ}

        Returns:
            Tuple of (exponent γ, fit error)
        """
        k, pk = self.degree_distribution()

        # Filter out zero frequencies
        mask = pk > 0
        k = k[mask]
        pk = pk[mask]

        # Log-log fit
        log_k = np.log(k)
        log_pk = np.log(pk)

        coeffs = np.polyfit(log_k, log_pk, 1)
        gamma = -coeffs[0]

        # Fit error
        fit = coeffs[0] * log_k + coeffs[1]
        error = np.sqrt(np.mean((log_pk - fit)**2))

        return gamma, error

    def hubs(self, n_hubs: int = 5) -> List[int]:
        """
        Find hub nodes (highest degree).

        Args:
            n_hubs: Number of hubs to return

        Returns:
            List of hub node indices
        """
        degrees = [(len(self.adjacency[i]), i) for i in range(self.n)]
        degrees.sort(reverse=True)
        return [node for _, node in degrees[:n_hubs]]


class NetworkCentrality:
    """
    Network centrality measures.

    Args:
        adjacency: Adjacency dictionary
    """

    def __init__(self, adjacency: Dict[int, Set[int]]):
        self.adjacency = adjacency
        self.n = len(adjacency)

    def degree_centrality(self) -> np.ndarray:
        """
        Compute degree centrality.

        C_D(i) = k_i / (n-1)

        Returns:
            Array of centrality values
        """
        centrality = np.zeros(self.n)
        for i in range(self.n):
            centrality[i] = len(self.adjacency[i]) / (self.n - 1)
        return centrality

    def betweenness_centrality(self) -> np.ndarray:
        """
        Compute betweenness centrality.

        C_B(v) = Σ_{s≠v≠t} σ_{st}(v) / σ_{st}

        Returns:
            Array of centrality values
        """
        centrality = np.zeros(self.n)

        for s in range(self.n):
            # BFS from s
            distances = {s: 0}
            paths = {s: 1}
            predecessors: Dict[int, List[int]] = {s: []}
            queue = deque([s])
            order = []

            while queue:
                v = queue.popleft()
                order.append(v)
                for w in self.adjacency[v]:
                    if w not in distances:
                        distances[w] = distances[v] + 1
                        queue.append(w)
                    if distances[w] == distances[v] + 1:
                        paths[w] = paths.get(w, 0) + paths[v]
                        if w not in predecessors:
                            predecessors[w] = []
                        predecessors[w].append(v)

            # Accumulate dependencies
            delta = {v: 0.0 for v in range(self.n)}
            for w in reversed(order):
                for v in predecessors.get(w, []):
                    delta[v] += (paths[v] / paths[w]) * (1 + delta[w])
                if w != s:
                    centrality[w] += delta[w]

        # Normalize
        norm = 2.0 / ((self.n - 1) * (self.n - 2))
        return centrality * norm

    def closeness_centrality(self) -> np.ndarray:
        """
        Compute closeness centrality.

        C_C(i) = (n-1) / Σ_j d(i,j)

        Returns:
            Array of centrality values
        """
        centrality = np.zeros(self.n)

        for i in range(self.n):
            # BFS to find all distances
            distances = self._bfs_distances(i)
            total_dist = sum(distances.values())
            if total_dist > 0:
                centrality[i] = (len(distances) - 1) / total_dist

        return centrality

    def _bfs_distances(self, start: int) -> Dict[int, int]:
        """Get all distances from start node."""
        distances = {start: 0}
        queue = deque([start])

        while queue:
            node = queue.popleft()
            for neighbor in self.adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)

        return distances

    def eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """
        Compute eigenvector centrality.

        Central nodes are connected to other central nodes.

        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Array of centrality values
        """
        # Power iteration
        x = np.ones(self.n) / self.n

        for _ in range(max_iter):
            x_new = np.zeros(self.n)
            for i in range(self.n):
                for j in self.adjacency[i]:
                    x_new[i] += x[j]

            x_new /= np.linalg.norm(x_new)

            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new

        return x

    def pagerank(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Compute PageRank centrality.

        Args:
            damping: Damping factor
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Array of PageRank values
        """
        x = np.ones(self.n) / self.n

        for _ in range(max_iter):
            x_new = np.zeros(self.n)

            for i in range(self.n):
                for j in self.adjacency[i]:
                    out_degree = len(self.adjacency[j])
                    if out_degree > 0:
                        x_new[i] += x[j] / out_degree

            x_new = damping * x_new + (1 - damping) / self.n

            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new

        return x


class CommunityDetection:
    """
    Community detection algorithms.

    Args:
        adjacency: Adjacency dictionary
    """

    def __init__(self, adjacency: Dict[int, Set[int]]):
        self.adjacency = adjacency
        self.n = len(adjacency)

    def modularity(self, communities: Dict[int, int]) -> float:
        """
        Compute modularity Q.

        Q = (1/2m) Σ_{ij} [A_{ij} - k_i k_j / 2m] δ(c_i, c_j)

        Args:
            communities: Node -> community mapping

        Returns:
            Modularity value
        """
        m = sum(len(self.adjacency[i]) for i in range(self.n)) / 2

        Q = 0.0
        for i in range(self.n):
            k_i = len(self.adjacency[i])
            for j in range(self.n):
                if communities[i] == communities[j]:
                    k_j = len(self.adjacency[j])
                    A_ij = 1 if j in self.adjacency[i] else 0
                    Q += A_ij - k_i * k_j / (2 * m)

        return Q / (2 * m)

    def label_propagation(self, max_iter: int = 100) -> Dict[int, int]:
        """
        Label propagation community detection.

        Args:
            max_iter: Maximum iterations

        Returns:
            Node -> community mapping
        """
        # Initialize: each node is its own community
        labels = {i: i for i in range(self.n)}

        for _ in range(max_iter):
            order = np.random.permutation(self.n)
            changed = False

            for node in order:
                if not self.adjacency[node]:
                    continue

                # Count neighbor labels
                label_counts: Dict[int, int] = {}
                for neighbor in self.adjacency[node]:
                    label = labels[neighbor]
                    label_counts[label] = label_counts.get(label, 0) + 1

                # Choose most common label
                max_count = max(label_counts.values())
                max_labels = [l for l, c in label_counts.items() if c == max_count]
                new_label = np.random.choice(max_labels)

                if new_label != labels[node]:
                    labels[node] = new_label
                    changed = True

            if not changed:
                break

        return labels

    def louvain(self) -> Dict[int, int]:
        """
        Louvain algorithm for community detection.

        Returns:
            Node -> community mapping
        """
        # Start with each node in own community
        communities = {i: i for i in range(self.n)}
        m = sum(len(self.adjacency[i]) for i in range(self.n)) / 2

        improved = True
        while improved:
            improved = False
            order = np.random.permutation(self.n)

            for node in order:
                current_comm = communities[node]
                k_i = len(self.adjacency[node])

                # Calculate modularity change for each neighbor community
                neighbor_comms = set()
                for neighbor in self.adjacency[node]:
                    neighbor_comms.add(communities[neighbor])

                best_comm = current_comm
                best_delta = 0.0

                for comm in neighbor_comms:
                    if comm == current_comm:
                        continue

                    # Modularity change for moving to this community
                    delta_Q = self._modularity_change(
                        node, current_comm, comm, communities, m
                    )

                    if delta_Q > best_delta:
                        best_delta = delta_Q
                        best_comm = comm

                if best_comm != current_comm:
                    communities[node] = best_comm
                    improved = True

        return communities

    def _modularity_change(
        self,
        node: int,
        old_comm: int,
        new_comm: int,
        communities: Dict[int, int],
        m: float
    ) -> float:
        """Calculate modularity change for moving node."""
        k_i = len(self.adjacency[node])

        # Sum of edges to old community
        sigma_old = sum(1 for n in self.adjacency[node] if communities[n] == old_comm)
        # Sum of degrees in old community
        sum_tot_old = sum(len(self.adjacency[n]) for n in range(self.n)
                         if communities[n] == old_comm)

        # Sum of edges to new community
        sigma_new = sum(1 for n in self.adjacency[node] if communities[n] == new_comm)
        # Sum of degrees in new community
        sum_tot_new = sum(len(self.adjacency[n]) for n in range(self.n)
                         if communities[n] == new_comm)

        # Modularity change
        delta_Q = (sigma_new - sigma_old) / m
        delta_Q -= k_i * (sum_tot_new - sum_tot_old + k_i) / (2 * m**2)

        return delta_Q


class SitePercolation:
    """
    Site percolation on a lattice.

    Each site is occupied with probability p.

    Args:
        size: Linear size of lattice
        dimension: Lattice dimension
        p: Occupation probability
    """

    def __init__(self, size: int, dimension: int = 2, p: float = 0.5):
        self.L = size
        self.d = dimension
        self.p = p

        # Generate configuration
        shape = tuple([size] * dimension)
        self.occupied = np.random.random(shape) < p

        self._cluster_labels = None

    def find_clusters(self) -> np.ndarray:
        """
        Find connected clusters using union-find.

        Returns:
            Array of cluster labels
        """
        labels = -np.ones(self.occupied.shape, dtype=int)
        current_label = 0

        # Iterate over all sites
        it = np.nditer(self.occupied, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            if self.occupied[idx] and labels[idx] == -1:
                # BFS to label cluster
                self._bfs_label(idx, labels, current_label)
                current_label += 1
            it.iternext()

        self._cluster_labels = labels
        return labels

    def _bfs_label(self, start: tuple, labels: np.ndarray, label: int):
        """BFS to label a cluster."""
        queue = deque([start])
        labels[start] = label

        while queue:
            idx = queue.popleft()

            # Check neighbors
            for dim in range(self.d):
                for delta in [-1, 1]:
                    neighbor = list(idx)
                    neighbor[dim] += delta

                    # Boundary conditions
                    if 0 <= neighbor[dim] < self.L:
                        neighbor = tuple(neighbor)
                        if self.occupied[neighbor] and labels[neighbor] == -1:
                            labels[neighbor] = label
                            queue.append(neighbor)

    def cluster_sizes(self) -> np.ndarray:
        """
        Get sizes of all clusters.

        Returns:
            Array of cluster sizes
        """
        if self._cluster_labels is None:
            self.find_clusters()

        unique, counts = np.unique(self._cluster_labels, return_counts=True)
        # Remove background (-1)
        mask = unique >= 0
        return counts[mask]

    def largest_cluster_size(self) -> int:
        """Get size of largest cluster."""
        sizes = self.cluster_sizes()
        return sizes.max() if len(sizes) > 0 else 0

    def percolates(self) -> bool:
        """
        Check if system percolates (spanning cluster exists).

        Returns:
            True if percolating
        """
        if self._cluster_labels is None:
            self.find_clusters()

        # Check if any cluster spans the system
        for dim in range(self.d):
            # Labels on opposite faces
            slices_low = [slice(None)] * self.d
            slices_high = [slice(None)] * self.d
            slices_low[dim] = 0
            slices_high[dim] = self.L - 1

            labels_low = set(self._cluster_labels[tuple(slices_low)].flatten())
            labels_high = set(self._cluster_labels[tuple(slices_high)].flatten())

            # Remove -1 (unoccupied)
            labels_low.discard(-1)
            labels_high.discard(-1)

            if labels_low & labels_high:
                return True

        return False

    def order_parameter(self) -> float:
        """
        Compute percolation order parameter P_∞.

        P_∞ = size of largest cluster / total sites

        Returns:
            Order parameter value
        """
        return self.largest_cluster_size() / self.L**self.d


class BondPercolation:
    """
    Bond percolation on a lattice.

    Each bond is present with probability p.

    Args:
        size: Linear size of lattice
        dimension: Lattice dimension
        p: Bond probability
    """

    def __init__(self, size: int, dimension: int = 2, p: float = 0.5):
        self.L = size
        self.d = dimension
        self.p = p

        # Generate bonds for each direction
        self.bonds: List[np.ndarray] = []
        for dim in range(dimension):
            shape = [size] * dimension
            shape[dim] = size - 1  # One less bond than sites
            self.bonds.append(np.random.random(shape) < p)

        self._cluster_labels = None

    def find_clusters(self) -> np.ndarray:
        """Find connected clusters via bonds."""
        shape = tuple([self.L] * self.d)
        labels = -np.ones(shape, dtype=int)
        current_label = 0

        it = np.nditer(np.zeros(shape), flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            if labels[idx] == -1:
                self._bfs_label_bonds(idx, labels, current_label)
                current_label += 1
            it.iternext()

        self._cluster_labels = labels
        return labels

    def _bfs_label_bonds(self, start: tuple, labels: np.ndarray, label: int):
        """BFS to label cluster following bonds."""
        queue = deque([start])
        labels[start] = label

        while queue:
            idx = queue.popleft()

            for dim in range(self.d):
                # Forward bond
                if idx[dim] < self.L - 1:
                    bond_idx = list(idx)
                    if self.bonds[dim][tuple(bond_idx)]:
                        neighbor = list(idx)
                        neighbor[dim] += 1
                        neighbor = tuple(neighbor)
                        if labels[neighbor] == -1:
                            labels[neighbor] = label
                            queue.append(neighbor)

                # Backward bond
                if idx[dim] > 0:
                    bond_idx = list(idx)
                    bond_idx[dim] -= 1
                    if self.bonds[dim][tuple(bond_idx)]:
                        neighbor = list(idx)
                        neighbor[dim] -= 1
                        neighbor = tuple(neighbor)
                        if labels[neighbor] == -1:
                            labels[neighbor] = label
                            queue.append(neighbor)

    def percolates(self) -> bool:
        """Check if system percolates."""
        if self._cluster_labels is None:
            self.find_clusters()

        for dim in range(self.d):
            slices_low = [slice(None)] * self.d
            slices_high = [slice(None)] * self.d
            slices_low[dim] = 0
            slices_high[dim] = self.L - 1

            labels_low = set(self._cluster_labels[tuple(slices_low)].flatten())
            labels_high = set(self._cluster_labels[tuple(slices_high)].flatten())

            if labels_low & labels_high:
                return True

        return False


class PercolationThreshold:
    """
    Percolation threshold estimation.
    """

    @staticmethod
    def site_square() -> float:
        """Site percolation threshold for 2D square lattice."""
        return 0.592746

    @staticmethod
    def bond_square() -> float:
        """Bond percolation threshold for 2D square lattice."""
        return 0.5

    @staticmethod
    def site_triangular() -> float:
        """Site percolation threshold for 2D triangular lattice."""
        return 0.5

    @staticmethod
    def bond_triangular() -> float:
        """Bond percolation threshold for 2D triangular lattice."""
        return 2 * np.sin(np.pi / 18)  # ≈ 0.3473

    @staticmethod
    def site_cubic() -> float:
        """Site percolation threshold for 3D cubic lattice."""
        return 0.3116

    @staticmethod
    def bond_cubic() -> float:
        """Bond percolation threshold for 3D cubic lattice."""
        return 0.2488

    @staticmethod
    def bethe_lattice(z: int) -> float:
        """
        Exact threshold for Bethe lattice (Cayley tree).

        p_c = 1/(z-1)

        Args:
            z: Coordination number

        Returns:
            Threshold probability
        """
        return 1.0 / (z - 1)

    @staticmethod
    def estimate_threshold(
        lattice_class,
        size: int,
        dimension: int = 2,
        n_samples: int = 100,
        p_range: Tuple[float, float] = (0.3, 0.8),
        n_points: int = 20
    ) -> float:
        """
        Estimate percolation threshold numerically.

        Args:
            lattice_class: SitePercolation or BondPercolation
            size: Lattice size
            dimension: Lattice dimension
            n_samples: Samples per probability
            p_range: Range of p values to test
            n_points: Number of p values

        Returns:
            Estimated threshold
        """
        p_vals = np.linspace(p_range[0], p_range[1], n_points)
        percolation_prob = np.zeros(n_points)

        for i, p in enumerate(p_vals):
            n_percolating = 0
            for _ in range(n_samples):
                lattice = lattice_class(size, dimension, p)
                if lattice.percolates():
                    n_percolating += 1
            percolation_prob[i] = n_percolating / n_samples

        # Find p where P(percolation) ≈ 0.5
        idx = np.argmin(np.abs(percolation_prob - 0.5))
        return p_vals[idx]


class ClusterStatistics:
    """
    Cluster statistics analysis for percolation.
    """

    def __init__(self, percolation_system):
        self.system = percolation_system

    def size_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cluster size distribution n_s.

        Returns:
            Tuple of (sizes, frequencies)
        """
        sizes = self.system.cluster_sizes()
        unique, counts = np.unique(sizes, return_counts=True)
        n_s = counts / np.prod([self.system.L] * self.system.d)
        return unique, n_s

    def mean_cluster_size(self, exclude_largest: bool = True) -> float:
        """
        Compute mean cluster size S = Σ s² n_s / Σ s n_s.

        Args:
            exclude_largest: Whether to exclude spanning cluster

        Returns:
            Mean cluster size
        """
        sizes = self.system.cluster_sizes()

        if exclude_largest and len(sizes) > 1:
            largest = sizes.max()
            sizes = sizes[sizes < largest]

        if len(sizes) == 0:
            return 0.0

        return np.sum(sizes**2) / np.sum(sizes)

    def correlation_length(self) -> float:
        """
        Estimate correlation length from cluster sizes.

        ξ² ~ Σ R_g² s n_s / Σ s n_s

        Returns:
            Correlation length estimate
        """
        # Simplified: use characteristic cluster size
        S = self.mean_cluster_size()
        return np.sqrt(S) if S > 0 else 0.0


class CorrelationLengthPerc:
    """
    Correlation length in percolation.

    Near criticality: ξ ~ |p - p_c|^{-ν}
    """

    @staticmethod
    def critical_exponent_nu(dimension: int) -> float:
        """
        Critical exponent ν for correlation length.

        Args:
            dimension: Spatial dimension

        Returns:
            Exponent ν
        """
        if dimension == 2:
            return 4 / 3
        elif dimension == 3:
            return 0.88
        elif dimension == 4:
            return 0.68
        elif dimension >= 6:
            return 0.5  # Mean field
        else:
            # Interpolate
            return 0.88 + (0.68 - 0.88) * (dimension - 3)

    @staticmethod
    def scaling_form(p: float, p_c: float, nu: float, xi_0: float = 1.0) -> float:
        """
        Correlation length scaling form.

        ξ = ξ_0 |p - p_c|^{-ν}

        Args:
            p: Occupation probability
            p_c: Critical probability
            nu: Critical exponent
            xi_0: Microscopic length scale

        Returns:
            Correlation length
        """
        if abs(p - p_c) < 1e-10:
            return float('inf')
        return xi_0 * abs(p - p_c)**(-nu)


__all__ = [
    'RandomGraph',
    'SmallWorldNetwork',
    'ScaleFreeNetwork',
    'NetworkCentrality',
    'CommunityDetection',
    'SitePercolation',
    'BondPercolation',
    'PercolationThreshold',
    'ClusterStatistics',
    'CorrelationLengthPerc',
]

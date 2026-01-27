import networkx as nx
import numpy as np
from itertools import combinations


class Problem:
    """Problem class for the Gold Collector optimization problem."""
    
    def __init__(self, num_cities, *, alpha=0.1, beta=1.5, density=0.1, seed=42):
        """
        Initialize the problem instance.
        
        Args:
            num_cities: Number of cities in the problem
            alpha: Load scaling factor
            beta: Load penalty exponent
            density: Graph edge density
            seed: Random seed for reproducibility
        """
        rng = np.random.default_rng(seed)
        self.alpha, self.beta = alpha, beta
        self.num_cities = num_cities
        
        # 1. Generate Coordinates and Gold amounts
        cities = rng.random(size=(num_cities, 2))
        cities[0, 0] = cities[0, 1] = 0.5  # Base is at the center
        
        self.graph = nx.Graph()
        self.graph.add_node(0, pos=(cities[0, 0], cities[0, 1]), gold=0)
        for c in range(1, num_cities):
            gold_amount = 1 + 999 * rng.random()
            self.graph.add_node(
                c,
                pos=(cities[c, 0], cities[c, 1]),
                gold=gold_amount
            )

        # 2. Build Topology (Sparse graph for performance)
        tmp = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
        d = np.sqrt(np.sum(np.square(tmp), axis=-1))
        for c1, c2 in combinations(range(num_cities), 2):
            if rng.random() < density or c2 == c1 + 1:
                self.graph.add_edge(c1, c2, dist=d[c1, c2])
        
        # Ensure graph is connected
        if not nx.is_connected(self.graph):
            components = list(nx.connected_components(self.graph))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                if not self.graph.has_edge(node1, node2):
                    self.graph.add_edge(node1, node2, dist=d[node1, node2])
        
        # 3. Precompute shortest path distances for efficiency
        print(" > Precomputing shortest path distances...")
        self.shortest_distances = dict(
            nx.all_pairs_dijkstra_path_length(self.graph, weight='dist')
        )

    def cost(self, path, weight):
        """
        Calculate cost for a path segment with given gold weight.
        
        Args:
            path: Either [u, v] for direct segment or a full path
            weight: Current gold weight being carried
            
        Returns:
            Cost of traversing the path with the given weight
        """
        if len(path) == 2:
            # Direct segment - use precomputed shortest distance
            u, v = path
            try:
                dist = self.shortest_distances[u][v]
            except KeyError:
                return float('inf')
        else:
            # Full path provided
            try:
                dist = nx.path_weight(self.graph, path, weight='dist')
            except nx.NetworkXNoPath:
                return float('inf')
        
        return dist + (self.alpha * dist * weight) ** self.beta
    
    def baseline(self):
        """
        Calculate baseline cost: go to each city and back separately.
        
        Returns:
            Total cost of the baseline solution
        """
        total_cost = 0
        paths = nx.single_source_dijkstra_path(
            self.graph, source=0, weight='dist'
        )
        
        for dest, path in paths.items():
            if dest == 0:
                continue
            
            cost = 0
            # Cost going TO the city (empty)
            for c1, c2 in zip(path, path[1:]):
                cost += self.cost([c1, c2], 0)
            
            # Cost coming BACK (with gold from dest)
            gold_weight = self.graph.nodes[dest]['gold']
            for c1, c2 in zip(path, path[1:]):
                cost += self.cost([c1, c2], gold_weight)
            
            total_cost += cost
        
        return total_cost

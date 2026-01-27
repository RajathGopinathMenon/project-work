import logging
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class Problem:
    """Problem class for the Gold Collector optimization problem."""
    
    _graph: nx.Graph
    _alpha: float
    _beta: float
    
    def __init__(
        self,
        num_cities: int,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        density: float = 0.5,
        seed: int = 42,
    ):
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
        self._alpha = alpha
        self._beta = beta
        self.num_cities = num_cities  # Add this for compatibility with your solvers
        
        # 1. Generate Coordinates and Gold amounts
        cities = rng.random(size=(num_cities, 2))
        cities[0, 0] = cities[0, 1] = 0.5  # Base is at the center
        
        self._graph = nx.Graph()
        self._graph.add_node(0, pos=(cities[0, 0], cities[0, 1]), gold=0)
        for c in range(1, num_cities):
            gold_amount = 1 + 999 * rng.random()
            self._graph.add_node(
                c,
                pos=(cities[c, 0], cities[c, 1]),
                gold=gold_amount
            )

        # 2. Build Topology (Sparse graph for performance)
        tmp = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
        d = np.sqrt(np.sum(np.square(tmp), axis=-1))
        for c1, c2 in combinations(range(num_cities), 2):
            if rng.random() < density or c2 == c1 + 1:
                self._graph.add_edge(c1, c2, dist=d[c1, c2])
        
        # Ensure graph is connected - use assert or manual connection
        if not nx.is_connected(self._graph):
            # Option 1: Uncomment to use assert (will fail if not connected)
            # assert nx.is_connected(self._graph)
            
            # Option 2: Manually connect components (your original approach)
            components = list(nx.connected_components(self._graph))
            for i in range(len(components) - 1):
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                if not self._graph.has_edge(node1, node2):
                    self._graph.add_edge(node1, node2, dist=d[node1, node2])
        
        # 3. Precompute shortest path distances for efficiency
        # This is CRITICAL for your GA solvers - keep this!
        print(" > Precomputing shortest path distances...")
        self.shortest_distances = dict(
            nx.all_pairs_dijkstra_path_length(self._graph, weight='dist')
        )

    @property
    def graph(self) -> nx.Graph:
        """Return a copy of the graph to prevent external modifications"""
        return nx.Graph(self._graph)
    
    @property
    def alpha(self):
        """Load scaling factor"""
        return self._alpha
    
    @property
    def beta(self):
        """Load penalty exponent"""
        return self._beta

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
            # Direct segment - use precomputed shortest distance if available
            u, v = path
            try:
                if hasattr(self, 'shortest_distances'):
                    dist = self.shortest_distances[u][v]
                else:
                    dist = nx.shortest_path_length(self._graph, u, v, weight='dist')
            except (KeyError, nx.NetworkXNoPath):
                return float('inf')
        else:
            # Full path provided
            try:
                dist = nx.path_weight(self._graph, path, weight='dist')
            except nx.NetworkXNoPath:
                return float('inf')
        
        return dist + (self._alpha * dist * weight) ** self._beta
    
    def baseline(self):
        """
        Calculate baseline cost: go to each city and back separately.
        
        Returns:
            Total cost of the baseline solution
        """
        total_cost = 0
        paths = nx.single_source_dijkstra_path(
            self._graph, source=0, weight='dist'
        )
        
        for dest, path in paths.items():
            if dest == 0:
                continue
            
            cost = 0
            # Cost going TO the city (empty)
            for c1, c2 in zip(path, path[1:]):
                cost += self.cost([c1, c2], 0)
            
            # Cost coming BACK (with gold from dest)
            gold_weight = self._graph.nodes[dest]['gold']
            for c1, c2 in zip(path, path[1:]):
                cost += self.cost([c1, c2], gold_weight)
            
            logging.debug(
                f"baseline: go to {dest} ({' > '.join(str(n) for n in path)}) (cost: {cost:.2f})"
            )
            total_cost += cost
        
        return total_cost
    
    def plot(self):
        """Plot the graph with nodes sized by gold amount"""
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self._graph, 'pos')
        size = [100] + [self._graph.nodes[n]['gold'] for n in range(1, len(self._graph))]
        color = ['red'] + ['lightblue'] * (len(self._graph) - 1)
        return nx.draw(self._graph, pos, with_labels=True, node_color=color, node_size=size)
    
    def plot_solution(self, solution_path, figsize=(14, 10), save_path=None):
        """
        Plot the graph with the solution routes overlaid.
        
        Args:
            solution_path: List of tuples [(c1, g1), (c2, g2), ..., (0, 0)]
                          This is the output from solution() function
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        
        Returns:
            fig, ax: Matplotlib figure and axes objects
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get node positions
        pos = nx.get_node_attributes(self._graph, 'pos')
        
        # Draw the base graph (edges in light gray)
        nx.draw_networkx_edges(self._graph, pos, edge_color='lightgray', 
                              width=0.5, alpha=0.3, ax=ax)
        
        # Parse solution into trips
        trips = []
        current_trip = []
        for city, gold in solution_path:
            if city == 0:
                if current_trip:
                    trips.append(current_trip)
                    current_trip = []
            else:
                current_trip.append(city)
        
        # Color palette for different trips
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(trips), 1)))
        
        # Draw each trip with different color
        for trip_idx, trip in enumerate(trips):
            color = colors[trip_idx % len(colors)]
            
            # Create full path: depot -> cities -> depot
            full_path = [0] + trip + [0]
            
            # Draw edges for this trip
            for i in range(len(full_path) - 1):
                u, v = full_path[i], full_path[i + 1]
                
                # Get shortest path between u and v
                try:
                    shortest_path = nx.shortest_path(self._graph, u, v, weight='dist')
                    
                    # Draw each edge in the shortest path
                    path_edges = [(shortest_path[j], shortest_path[j+1]) 
                                 for j in range(len(shortest_path)-1)]
                    
                    nx.draw_networkx_edges(
                        self._graph, pos, 
                        edgelist=path_edges,
                        edge_color=[color],
                        width=2.5,
                        alpha=0.7,
                        ax=ax,
                        arrows=True,
                        arrowsize=15,
                        arrowstyle='->'
                    )
                except nx.NetworkXNoPath:
                    print(f"Warning: No path found between {u} and {v}")
                    continue
        
        # Draw nodes
        node_sizes = [300] + [100 + self._graph.nodes[n]['gold']/3 
                             for n in range(1, len(self._graph))]
        node_colors = ['red'] + ['lightblue'] * (len(self._graph) - 1)
        
        nx.draw_networkx_nodes(self._graph, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              edgecolors='black',
                              linewidths=1.5,
                              ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(self._graph, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='red', marker='o', linestyle='None',
                      markersize=10, label='Depot (Base)'),
            plt.Line2D([0], [0], color='lightblue', marker='o', linestyle='None',
                      markersize=10, label='Cities (size ∝ gold)'),
        ]
        
        # Add trip colors to legend (first 10 trips)
        for trip_idx in range(min(len(trips), 10)):
            color = colors[trip_idx % len(colors)]
            legend_elements.append(
                plt.Line2D([0], [0], color=color, linewidth=3,
                          label=f'Trip {trip_idx + 1}')
            )
        
        if len(trips) > 10:
            legend_elements.append(
                plt.Line2D([0], [0], color='gray', linewidth=3,
                          label=f'... +{len(trips) - 10} more')
            )
        
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
        
        # Title with statistics
        total_trips = len(trips)
        total_cities = sum(len(trip) for trip in trips)
        ax.set_title(f'Gold Collector Solution\n{total_trips} trips, {total_cities} cities visited',
                    fontsize=14, fontweight='bold')
        
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig, ax
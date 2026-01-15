import math
import random
import numpy as np
import networkx as nx
from copy import deepcopy
from functools import lru_cache

class GeneticSolver:
    def __init__(self, problem):
        self.problem = problem
        self.n = self.problem.num_cities
        print(f"  Initializing Solver for {self.n} cities...")
        print(f"  Beta: {self.problem.beta}, Alpha: {self.problem.alpha}")
        
        # Gold values
        self.gold_values = np.array([self.problem.graph.nodes[i]['gold'] for i in range(self.n)])
        self.customers = list(range(1, self.n))
        
        # GA parameters
        self.pop_size = 40  # Slightly smaller population for speed with 1000 cities
        self.generations = 100
        self.elite_size = 3
        self.tournament_size = 5
        self.mutation_rate = 0.3
        self.patience = 15
        
        # Baseline
        self.initial_cost = self.problem.baseline()
        print(f"  Baseline cost: {self.initial_cost:.2f}")
        
        # Run GA
        print(f"  Running Genetic Algorithm (Lazy Pathfinding)...")
        best_routes, best_cost = self.run_ga()
        
        self.state = best_routes
        self.cost = best_cost
        self.best_state = deepcopy(best_routes)
        self.best_cost = best_cost
        
        self.iterations = 0
        self.accepted_moves = 0
        self.improvement_count = 0

    @lru_cache(maxsize=50000)
    def get_cached_path_edges(self, u, v):
        """Lazy Dijkstra: Only compute shortest paths when the GA requests them."""
        try:
            path = nx.dijkstra_path(self.problem.graph, u, v, weight='dist')
            return [(path[k], path[k+1]) for k in range(len(path)-1)]
        except nx.NetworkXNoPath:
            return None

    def calculate_edge_cost(self, u, v, current_gold):
        edges = self.get_cached_path_edges(u, v)
        if edges is None:
            return float('inf')
        
        total_cost = 0.0
        for edge_u, edge_v in edges:
            segment_cost = self.problem.cost([edge_u, edge_v], current_gold)
            total_cost += segment_cost
        return total_cost

    def split_algorithm(self, permutation):
        """DP to split a permutation into optimal trips with a search window."""
        n = len(permutation)
        dp = np.full(n + 1, np.inf)
        dp[0] = 0
        predecessor = np.zeros(n + 1, dtype=int)
        
        # Human Heuristic: A thief won't carry gold for more than 20-30 cities 
        # before the (alpha * weight)^beta cost becomes astronomical.
        max_trip_size = 25 
        
        for i in range(n):
            if np.isinf(dp[i]): continue
            
            current_gold = 0.0
            trip_dist_cost = 0.0
            current_node = 0 
            
            # Search only within the feasible 'max_trip_size' window
            for j in range(i + 1, min(i + max_trip_size + 1, n + 1)):
                next_customer = permutation[j - 1]
                
                edge_cost = self.calculate_edge_cost(current_node, next_customer, current_gold)
                if np.isinf(edge_cost): break
                
                trip_dist_cost += edge_cost
                current_gold += self.gold_values[next_customer]
                current_node = next_customer
                
                return_cost = self.calculate_edge_cost(current_node, 0, current_gold)
                total_cost = dp[i] + trip_dist_cost + return_cost
                
                if total_cost < dp[j]:
                    dp[j] = total_cost
                    predecessor[j] = i
        
        # Reconstruct
        routes = []
        curr = n
        while curr > 0:
            prev = predecessor[curr]
            routes.append(list(permutation[prev:curr]))
            curr = prev
        routes.reverse()
        return dp[n], routes

    # --- GA Engine ---
    def initial_population(self):
        return [random.sample(self.customers, len(self.customers)) for _ in range(self.pop_size)]

    def evaluate_population(self, population):
        return [self.split_algorithm(ind)[0] for ind in population]

    def tournament_selection(self, population, scores):
        selected = []
        for _ in range(len(population)):
            indices = random.sample(range(len(population)), self.tournament_size)
            winner = min(indices, key=lambda i: scores[i])
            selected.append(population[winner][:])
        return selected

    def ordered_crossover(self, p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [None]*size
        child[a:b] = p1[a:b]
        p2_fill = [item for item in p2 if item not in child[a:b]]
        idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = p2_fill[idx]
                idx += 1
        return child

    def run_ga(self):
        pop = self.initial_population()
        best_cost = float('inf')
        best_routes = []
        no_improve = 0
        
        for gen in range(self.generations):
            scores = self.evaluate_population(pop)
            min_score = min(scores)
            
            if min_score < best_cost:
                best_cost = min_score
                best_routes = self.split_algorithm(pop[scores.index(min_score)])[1]
                no_improve = 0
                print(f"    Gen {gen}: Cost {best_cost:.2f}")
            else:
                no_improve += 1
            
            if no_improve >= self.patience: break
            
            # Elitism
            sorted_idx = np.argsort(scores)
            new_pop = [pop[i][:] for i in sorted_idx[:self.elite_size]]
            
            # Selection & Crossover
            parents = self.tournament_selection(pop, scores)
            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(parents, 2)
                child = self.ordered_crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                new_pop.append(child)
            pop = new_pop
            
        return best_routes, best_cost

    def _evaluate(self, solution):
        """Used for local search refinement."""
        total = 0
        for trip in solution:
            curr, gold = 0, 0
            for city in trip:
                total += self.calculate_edge_cost(curr, city, gold)
                gold += self.gold_values[city]
                curr = city
            total += self.calculate_edge_cost(curr, 0, gold)
        return total

    def step(self):
        """Basic local search step."""
        if not self.state: return self.best_cost
        new_state = deepcopy(self.state)
        # Randomly move a city between trips
        t1, t2 = random.sample(range(len(new_state)), 2)
        if new_state[t1]:
            city = new_state[t1].pop(random.randrange(len(new_state[t1])))
            new_state[t2].insert(random.randint(0, len(new_state[t2])), city)
            
        new_cost = self._evaluate(new_state)
        if new_cost < self.cost:
            self.state, self.cost = new_state, new_cost
            if new_cost < self.best_cost:
                self.best_cost, self.best_state = new_cost, deepcopy(new_state)
        return self.best_cost

    def get_statistics(self):
        return {'best_cost': self.best_cost, 'initial_cost': self.initial_cost,
                'improvement_pct': ((self.initial_cost - self.best_cost)/self.initial_cost)*100,
                'num_trips': len(self.best_state)}
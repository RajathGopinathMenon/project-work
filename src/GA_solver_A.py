import math
import random
import numpy as np
from copy import deepcopy

class GeneticSolver:
    def __init__(self, problem):
        self.problem = problem
        
        print(f"  Beta: {self.problem.beta}, Alpha: {self.problem.alpha}")
        
        # Precompute shortest paths as edge sequences for CORRECT evaluation
        n = self.problem.num_cities
        print(f"  Precomputing path structures...")
        import networkx as nx
        
        self.path_edges = {}
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.path_edges[(i, j)] = []
                else:
                    try:
                        path = nx.dijkstra_path(self.problem.graph, i, j, weight='dist')
                        edges = [(path[k], path[k+1]) for k in range(len(path)-1)]
                        self.path_edges[(i, j)] = edges
                    except nx.NetworkXNoPath:
                        self.path_edges[(i, j)] = None
        
        print(f"  ✓ Paths cached")
        
        # Gold values
        self.gold_values = np.array([self.problem.graph.nodes[i]['gold'] for i in range(n)])
        
        # Customers to visit (all non-depot cities)
        self.customers = list(range(1, n))
        
        # GA parameters
        self.pop_size = 50
        self.generations = 100
        self.elite_size = 5
        self.tournament_size = 5
        self.mutation_rate = 0.3
        self.patience = 20
        
        # Baseline
        baseline_cost = self.problem.baseline()
        print(f"  Baseline cost: {baseline_cost:.2f}")
        
        # Run GA
        print(f"  Running Genetic Algorithm...")
        best_routes, best_cost = self.run_ga()
        
        self.state = best_routes
        self.cost = best_cost
        self.best_state = deepcopy(best_routes)
        self.best_cost = best_cost
        self.initial_cost = baseline_cost
        
        improvement = ((baseline_cost - best_cost) / baseline_cost * 100)
        print(f"  GA Result: {best_cost:.2f}, Trips: {len(best_routes)}, Improvement: {improvement:.2f}%")
        
        self.iterations = 0
        self.accepted_moves = 0
        self.improvement_count = 0

    def calculate_edge_cost(self, u, v, current_gold):
        """Calculate cost of traveling from u to v with given gold load
        CRITICAL: Must traverse each edge in shortest path separately"""
        edges = self.path_edges.get((u, v))
        if edges is None:
            return float('inf')
        
        total_cost = 0.0
        for edge_u, edge_v in edges:
            segment_cost = self.problem.cost([edge_u, edge_v], current_gold)
            if np.isinf(segment_cost):
                return float('inf')
            total_cost += segment_cost
        
        return total_cost

    def split_algorithm(self, permutation):
        """Dynamic programming to split a permutation into optimal trips"""
        n = len(permutation)
        
        # dp[i] = minimum cost to service first i customers
        dp = np.full(n + 1, np.inf)
        dp[0] = 0
        predecessor = np.zeros(n + 1, dtype=int)
        
        for i in range(n):
            if np.isinf(dp[i]):
                continue
            
            current_gold = 0.0
            trip_cost = 0.0
            current_node = 0  # depot
            
            # Try extending trip to customer j
            for j in range(i + 1, n + 1):
                next_customer = permutation[j - 1]
                
                # Cost to go to next customer
                edge_cost = self.calculate_edge_cost(current_node, next_customer, current_gold)
                if np.isinf(edge_cost):
                    break
                
                trip_cost += edge_cost
                current_gold += self.gold_values[next_customer]
                current_node = next_customer
                
                # Cost to return to depot
                return_cost = self.calculate_edge_cost(current_node, 0, current_gold)
                if np.isinf(return_cost):
                    continue
                
                total_cost = dp[i] + trip_cost + return_cost
                
                if total_cost < dp[j]:
                    dp[j] = total_cost
                    predecessor[j] = i
        
        # Reconstruct routes
        routes = []
        curr = n
        while curr > 0:
            prev = predecessor[curr]
            route_cities = permutation[prev:curr]
            routes.append(list(route_cities))
            curr = prev
        
        routes.reverse()
        return dp[n], routes

    def initial_population(self):
        """Generate initial population of random permutations"""
        population = []
        for _ in range(self.pop_size):
            perm = list(self.customers)
            random.shuffle(perm)
            population.append(perm)
        return population

    def evaluate_population(self, population):
        """Evaluate fitness of all individuals"""
        scores = []
        for individual in population:
            cost, _ = self.split_algorithm(individual)
            scores.append(cost)
        return scores

    def tournament_selection(self, population, scores):
        """Tournament selection"""
        selected = []
        for _ in range(len(population)):
            tournament_idx = random.sample(range(len(population)), self.tournament_size)
            winner_idx = min(tournament_idx, key=lambda i: scores[i])
            selected.append(population[winner_idx][:])
        return selected

    def ordered_crossover(self, parent1, parent2):
        """Ordered Crossover (OX)"""
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [None] * size
        child[start:end] = parent1[start:end]
        child_set = set(parent1[start:end])
        
        pos = end
        for gene in parent2:
            if gene not in child_set:
                if pos >= size:
                    pos = 0
                while child[pos] is not None:
                    pos += 1
                    if pos >= size:
                        pos = 0
                child[pos] = gene
                pos += 1
        
        return child

    def mutate(self, individual):
        """Swap mutation"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def run_ga(self):
        """Run the genetic algorithm"""
        population = self.initial_population()
        best_cost = float('inf')
        best_routes = []
        generations_without_improvement = 0
        
        for gen in range(self.generations):
            # Evaluate
            scores = self.evaluate_population(population)
            
            # Track best
            min_cost = min(scores)
            if min_cost < best_cost:
                best_cost = min_cost
                best_idx = scores.index(min_cost)
                _, best_routes = self.split_algorithm(population[best_idx])
                generations_without_improvement = 0
                if gen % 10 == 0:
                    print(f"    Gen {gen}: {best_cost:.2f}")
            else:
                generations_without_improvement += 1
            
            # Early stopping
            if generations_without_improvement >= self.patience:
                print(f"    Early stop at generation {gen}")
                break
            
            # Elitism - keep best individuals
            sorted_idx = np.argsort(scores)
            new_population = [population[i][:] for i in sorted_idx[:self.elite_size]]
            
            # Selection
            parents = self.tournament_selection(population, scores)
            
            # Crossover and mutation
            while len(new_population) < self.pop_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                child = self.ordered_crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return best_routes, best_cost

    def _evaluate(self, solution):
        """Evaluate a solution (list of trips) - CORRECT version matching baseline"""
        total_cost = 0
        for trip in solution:
            if not trip:
                continue
            
            current_pos = 0
            current_gold = 0
            
            # Visit each city
            for city in trip:
                edges = self.path_edges.get((current_pos, city))
                if edges is None:
                    return float('inf')
                
                # Traverse each edge in shortest path
                for u, v in edges:
                    cost = self.problem.cost([u, v], current_gold)
                    if np.isinf(cost):
                        return float('inf')
                    total_cost += cost
                
                current_gold += self.gold_values[city]
                current_pos = city
            
            # Return to depot
            edges = self.path_edges.get((current_pos, 0))
            if edges is None:
                return float('inf')
            
            for u, v in edges:
                cost = self.problem.cost([u, v], current_gold)
                if np.isinf(cost):
                    return float('inf')
                total_cost += cost
        
        return total_cost

    def step(self):
        """Single SA step for refinement after GA"""
        self.iterations += 1
        
        if self.iterations > 5000:  # Limited refinement
            return self.best_cost
        
        new_state = [list(trip) for trip in self.state]
        
        # Simple local search operators
        op = random.random()
        
        if op < 0.4 and len(new_state) >= 2:  # Merge
            i, j = random.sample(range(len(new_state)), 2)
            t1 = new_state.pop(max(i, j))
            t2 = new_state.pop(min(i, j))
            new_state.append(t1 + t2)
        
        elif op < 0.7 and new_state:  # Relocate
            valid = [i for i, t in enumerate(new_state) if len(t) > 0]
            if valid:
                src = random.choice(valid)
                city = new_state[src].pop(random.randrange(len(new_state[src])))
                if not new_state[src]:
                    new_state.pop(src)
                if new_state:
                    dst = random.randrange(len(new_state))
                    pos = random.randint(0, len(new_state[dst]))
                    new_state[dst].insert(pos, city)
                else:
                    new_state.append([city])
        
        else:  # Split
            valid = [i for i, t in enumerate(new_state) if len(t) >= 2]
            if valid:
                idx = random.choice(valid)
                trip = new_state.pop(idx)
                split = random.randint(1, len(trip) - 1)
                new_state.append(trip[:split])
                new_state.append(trip[split:])
        
        new_state = [t for t in new_state if t]
        if not new_state:
            return self.best_cost
        
        new_cost = self._evaluate(new_state)
        
        if new_cost == float('inf'):
            return self.best_cost
        
        # Simple hill climbing (only accept improvements after GA)
        if new_cost < self.cost:
            self.state = new_state
            self.cost = new_cost
            self.accepted_moves += 1
            
            if new_cost < self.best_cost:
                self.best_cost = new_cost
                self.best_state = deepcopy(new_state)
                self.improvement_count += 1
        
        return self.best_cost

    def get_statistics(self):
        return {
            'best_cost': self.best_cost,
            'initial_cost': self.initial_cost,
            'improvement_pct': ((self.initial_cost - self.best_cost) / self.initial_cost) * 100,
            'num_trips': len(self.best_state),
            'iterations': self.iterations,
            'acceptance_rate': (self.accepted_moves / max(self.iterations, 1)) * 100,
            'improvements': self.improvement_count
        }


# Backwards compatibility
HybridSolver = GeneticSolver
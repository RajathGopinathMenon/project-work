"""
This file contains the main solution() function that solves the
Gold Collector problem using a Genetic Algorithm approach.

The solver selection is based on problem size:
- GA_solver_A: For problems with < 1000 cities (more thorough, slower)
- GA_solver: For problems with >= 1000 cities (optimized for large scale)
"""

from Problem import Problem


def solution(p: Problem):
    """
    Solve the Gold Collector problem.
    
    Args:
        p: An instance of the Problem class
        
    Returns:
        A list of tuples [(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]
        where ci is the city index and gi is the gold collected at that city
    """
    # Select appropriate solver based on problem size
    if p.num_cities >= 1000:
        from src.GA_solver import GeneticSolver
    else:
        from src.GA_solver_A import GeneticSolver
    
    # Run the solver
    solver = GeneticSolver(p)
    
    # Get the best solution (list of trips)
    best_routes = solver.best_state
    
    # Build output path in required format
    path = []
    for route in best_routes:
        for city in route:
            gold = p.graph.nodes[city]['gold']
            path.append((city, gold))
        path.append((0, 0))
    
    return path

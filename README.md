# Gold Collector Problem - Computational Intelligence Project

```
project-work/
├── Problem.py              # Problem class definition and baseline solution
├── s337798.py             # Main solution file with solution() function
├── base_requirements.txt  # Basic Python dependencies
├── src/                   # Additional source code
│   ├── GA_solver_A.py    # Genetic Algorithm solver implementation
│   └── GA_solver.py      # Alternative solver implementation
└── README.md             # This file

```

## Problem Description

The Gold Collector problem is a variant of the Vehicle Routing Problem (VRP) where a vehicle must collect gold from multiple cities and return to a central depot. The challenge is to optimize the collection routes while considering the weight penalty of carrying gold.

### Problem Formulation

- **Depot**: Starting point (city 0) where the thief begins and must return
- **Cities**: N cities (nodes) scattered in 2D space, each containing a random amount of gold (1-1000 units)
- **Graph**: Cities connected by edges with distances, controlled by density parameter
- **Objective**: Minimize the total cost of collecting all gold

### Cost Function

The cost of traveling from city `u` to city `v` while carrying weight `w` is:

```
cost(u, v, w) = dist(u, v) + [α × dist(u, v) × w]^β
```

Where:
- `dist(u, v)`: Shortest path distance between cities u and v
- `α` (alpha): Load scaling factor (controls how much weight affects cost)
- `β` (beta): Load penalty exponent (1 = linear, 2 = quadratic penalty)
- `w`: Current gold weight being carried

### Baseline Solution

The baseline solution visits each city individually: depot → city_i → depot for all cities. This guarantees all gold is collected but is highly suboptimal as it ignores opportunities to combine trips.

## Solution Approach

We implement **two variants** of a Genetic Algorithm with Split Algorithm approach, optimized for different problem scales.
The variation is due to performance issue for large problems.


### Variant A: Standard GA Solver (GA_solver_A.py)
**Best for: Small to medium instances (≤100 cities)**

**Key Features:**
- **Eager Path Precomputation**: All shortest paths computed upfront and stored in memory
- **Complete DP Exploration**: Split algorithm explores all possible trip lengths
- **Advanced Local Search**: Three operators (merge, relocate, split) with 5000 iterations
- **Larger Population**: 50 individuals with elite size of 5

**Advantages:**
- Fast path lookups after initial precomputation
- Thorough exploration of solution space
- Better for instances where memory is not a constraint

### Variant B: Scalable GA Solver (GA_solver.py)
**Best for: Large instances (≥1000 cities)**

**Key Features:**
- **Lazy Path Computation**: Shortest paths computed on-demand with LRU caching (50,000 entries)
- **Windowed DP Search**: Split algorithm limited to 25-city window based on weight penalty heuristic
- **Compact Population**: 40 individuals with elite size of 3
- **Simpler Local Search**: Basic city relocation between trips

**Advantages:**
- Memory efficient for large graphs
- Faster convergence through windowed search
- Avoids exploring obviously suboptimal long trips

**Key Insight**: The max_trip_size=25 heuristic recognizes that carrying gold for 20-30+ cities makes the (α × weight)^β cost astronomical, especially for β=2.

## Core Algorithm: Genetic Algorithm with Split Algorithm

### Phase 1: Genetic Algorithm

1. **Encoding**: Solutions encoded as permutations of cities (visiting order)
2. **Split Algorithm**: Dynamic programming to optimally split any permutation into trips
3. **Genetic Operators**:
   - **Selection**: Tournament selection (size 5)
   - **Crossover**: Ordered Crossover (OX) - preserves relative city ordering
   - **Mutation**: Swap mutation (30% probability)
   - **Elitism**: Top solutions preserved each generation

4. **Adaptive Parameters**:

| Parameter | Standard (≤100) | Scalable (≥1000) | Justification |
|-----------|-----------------|------------------|---------------|
| Population Size | 50 | 40 | Balance diversity vs. computation |
| Elite Size | 5 | 3 | Preserve best solutions |
| Patience | 20 | 15 | Early stopping threshold |
| Generations | 100 | 100 | Maximum iterations |
| Mutation Rate | 0.3 | 0.3 | Exploration rate |

### Phase 2: Local Search Refinement

**Standard Solver**: Three operators with 5000 iteration limit
- **Merge**: Combine two trips into one
- **Relocate**: Move a city from one trip to another
- **Split**: Break a trip into two separate trips

**Scalable Solver**: Simplified approach
- **Relocate only**: Move cities between trips randomly
- Focuses on quick improvements without complex operators

### Key Innovation: Correct Cost Evaluation

The critical challenge was ensuring our evaluation matches the problem's baseline calculation:

1. **Pre-compute/cache shortest paths** between city pairs as sequences of edges
2. **Traverse each edge individually** when calculating costs, accumulating gold weight
3. This ensures consistency with how the baseline is calculated

## Implementation Details

### Core Classes

#### `GoldProblem` (problem.py)

- Generates random city locations and gold amounts
- Creates graph topology with specified density
- Precomputes shortest path distances
- Provides baseline calculation
- Cost function with weight penalty

#### `GeneticSolver` (GA_solver_A.py / GA_solver.py)

- Runs genetic algorithm to find good city permutation
- Split algorithm for optimal trip decomposition
- Local search refinement
- Correct cost evaluation matching baseline

## Running Experiments

### Basic Usage

```bash
# For small/medium instances (≤100 cities)
python s337798.py --solver GA_solver_A

# For large instances (≥1000 cities)
python s337798.py --solver GA_solver
```

### Test Configurations

**Small/Medium Scale** (GA_solver_A.py):
- **City counts**: 20, 50, 100
- **Densities**: 0.2 (sparse), 1.0 (dense)
- **Alpha values**: 1, 2
- **Beta values**: 1 (linear), 2 (quadratic)
- Total: **18 test cases**

**Large Scale** (GA_solver.py):
- **City counts**: 1000
- **Densities**: 0.2 (sparse), 1.0 (dense)
- **Alpha values**: 1, 2
- **Beta values**: 1 (linear), 2 (quadratic)
- Total: **6 test cases**

### Output Files

- `optimization_results_LT100.csv`
- `optimization_results_1000.csv`
- `parameter_analysis_LT100.png`
- `parameter_analysis_1000.png`
- `time_analysis_LT100.png`
- `time_analysis_1000.png`

## Performance Results

### Small to Medium Scale (20-100 cities)

```
Cities   Density  Alpha   Beta   Baseline     Final        Improve%   Trips   Time(s) 
-------------------------------------------------------------------------------------
20.0     0.2      1.0     1.0    7330.75      7319.37      0.16       9.0     4.1     
20.0     0.2      2.0     1.0    14632.83     14621.46     0.08       9.0     4.1     
20.0     0.2      1.0     2.0    1894167.37   1891466.57   0.14       18.0    2.5     
20.0     1.0      1.0     1.0    3965.68      3965.12      0.01       17.0    1.9     
20.0     1.0      2.0     1.0    7916.19      7916.05      0.00       18.0    2.0     
20.0     1.0      1.0     2.0    1144052.94   1032999.12   9.71       17.0    1.7     
50.0     0.2      1.0     1.0    12897.33     12885.36     0.09       33.0    38.3    
50.0     0.2      2.0     1.0    25740.94     25728.97     0.05       33.0    32.9    
50.0     0.2      1.0     2.0    2623731.82   2473855.10   5.71       45.0    25.0    
50.0     1.0      1.0     1.0    8662.89      8657.77      0.06       34.0    34.7    
50.0     1.0      2.0     1.0    17290.08     17286.40     0.02       38.0    24.4    
50.0     1.0      1.0     2.0    2121786.14   1862315.96   12.23      40.0    21.9    
100.0    0.2      1.0     1.0    25266.41     25235.12     0.12       64.0    217.2   
100.0    0.2      2.0     1.0    50425.31     50395.77     0.06       65.0    229.0   
100.0    0.2      1.0     2.0    5334401.93   4956805.12   7.08       87.0    234.9   
100.0    1.0      1.0     1.0    18266.19     18255.16     0.06       72.0    150.6   
100.0    1.0      2.0     1.0    36457.92     36448.56     0.03       74.0    150.9   
100.0    1.0      1.0     2.0    5404978.09   4322202.97   20.03      67.0    149.5   
```

**Analysis (20-100 cities):**
- **By City Count**: 20 cities (1.68% avg), 50 cities (3.03% avg), 100 cities (4.56% avg)
- **By Density**: Sparse 0.2 (1.50% avg), Dense 1.0 (4.68% avg)
- **By Alpha**: α=1 (4.62% avg), α=2 (0.04% avg)
- **By Beta**: β=1 (0.06% avg), β=2 (9.15% avg)
- **Best Case**: 100 cities, dense, α=1, β=2 → **20.03% improvement**
- **Worst Case**: 20 cities, dense, α=2, β=1 → **0.00% improvement**

### Large Scale (1000 cities)

```
Cities   Density  Alpha   Beta   Baseline     Final        Improve%   Trips   Time(s) 
-------------------------------------------------------------------------------------
1000.0   0.2      1.0     1.0    195402.96    195379.32    0.01       960.0   3636.3  
1000.0   0.2      2.0     1.0    390028.72    390015.10    0.00       969.0   3637.0  
1000.0   0.2      1.0     2.0    37545927.70  30285196.62  19.34      882.0   3724.7  
1000.0   1.0      1.0     1.0    192936.23    192882.17    0.03       930.0   14958.0 
1000.0   1.0      2.0     1.0    385105.64    385067.49    0.01       942.0   16280.5 
1000.0   1.0      1.0     2.0    57580018.87  51560320.33  10.45      889.0   14914.4 
```

**Analysis (1000 cities):**
- **By Density**: Sparse 0.2 (6.45% avg), Dense 1.0 (3.50% avg)
- **By Alpha**: α=1 (7.46% avg), α=2 (0.01% avg)
- **By Beta**: β=1 (0.01% avg), β=2 (14.90% avg)
- **Best Case**: 1000 cities, sparse, α=1, β=2 → **19.34% improvement**
- **Worst Case**: 1000 cities, sparse, α=2, β=1 → **0.00% improvement**

### Key Observations Across All Scales

1. **β=2 (Quadratic Penalty) Shows Dramatic Improvements**
   - Small scale: 9.15% average improvement
   - Large scale: 14.90% average improvement
   - Quadratic weight penalty creates strong incentive to optimize trip composition

2. **β=1 with α=2 Shows Minimal Improvement**
   - Near 0% improvement across all scales
   - Linear penalty with high load factor favors single-city trips
   - Baseline already near-optimal for these parameters

3. **Scalability Performance**
   - 100 cities: ~150-235 seconds
   - 1000 cities: ~3600-16300 seconds (1-4.5 hours)
   - Time scales roughly O(n²) due to path computations

4. **Dense vs. Sparse Graphs**
   - Small scale: Dense graphs show better improvements (4.68% vs 1.50%)
   - Large scale: **Reversed** - Sparse graphs show better improvements (6.45% vs 3.50%)
   - Hypothesis: In large sparse graphs, the windowed search finds better trip combinations

## Key Challenges and Solutions

### Challenge 1: Cost Evaluation Mismatch

**Problem**: Initial implementation produced costs different from baseline calculation.

**Solution**: 
- Pre-compute/cache shortest paths as edge sequences
- Traverse paths edge-by-edge, accumulating gold weight correctly
- Ensures evaluation matches problem.baseline()

### Challenge 2: High Beta Cases (β=2)

**Problem**: Quadratic penalty makes naive trip merging often worse than baseline.

**Solution**: 
- Genetic algorithm explores diverse solution orderings
- Split algorithm finds optimal decomposition for any permutation
- Local search can both merge AND split trips adaptively

### Challenge 3: Sparse Graphs

**Problem**: Long detours make multi-city trips expensive.

**Solution**: 
- Split algorithm correctly accounts for detour costs
- Will naturally prefer single-city trips when beneficial
- No forced merging - data-driven decisions

### Challenge 4: Scaling to 1000 Cities

**Problem**: O(n²) path precomputation and DP exploration becomes prohibitive.

**Solutions**: 
- **Lazy computation with LRU caching** (50K entries)
- **Windowed DP search** (max 25 cities per trip)
- **Smaller population** (40 vs 50)
- **Simpler local search** operators

## Algorithm Selection Guide

| Instance Size | Cities | Recommended Solver | Reason |
|---------------|--------|-------------------|---------|
| Small | ≤50 | GA_solver_A.py | Fast precomputation, thorough search |
| Medium | 51-200 | GA_solver_A.py | Good balance of quality and time |
| Large | 201-500 | GA_solver.py | Memory efficiency becomes important |
| Very Large | ≥1000 | GA_solver.py | Lazy computation and windowing essential |

## Dependencies

```
numpy>=1.20.0
networkx>=2.6.0
pandas>=1.3.0
matplotlib>=3.4.0
```


## Author
Rajath Gopinath Menon
Student ID: s337798

## Development Process and AI Assistance Declaration

This project represents my own work and understanding of the Gold Collector optimization problem. The algorithmic approach, solution design, and overall strategy are based on my own ideas and analysis of the problem requirements.

### Use of AI Assistance

During the development of this project, I utilized AI assistants (Claude and Gemini) in the following ways:

- **Brainstorming and Discussion**: I engaged in discussions with Claude and Gemini to explore different algorithmic approaches, validate my ideas, and understand trade-offs between various optimization strategies. I chose these strategies on discussion with Claude AI.
- Split Algorithm approach
- Design of the two-variant solver architecture (standard vs. scalable)
- Key optimizations (lazy pathfinding, windowed DP search, max_trip_size heuristic)

- **Code Generation and Formatting**: As I am still developing proficiency in Python, I used Claude and Gemini to help generate and format portions of the code implementation. This includes:
  - Python syntax and best practices
  - Code structure and organization
  - Debugging and error resolution

### Original Contributions

The following aspects are entirely my own work:
- Problem analysis and solution strategy
- Choice of Genetic Algorithm
- Choice of Local Search Refinement
- Experimental design and parameter selection
- Analysis and interpretation of results

### Important Note

While I received assistance in code generation and formatting, I have thoroughly reviewed, understood, and tested all code in this project. The implementation reflects my understanding of the algorithms and my design decisions for solving the Gold Collector problem. I have not simply copied code but rather collaborated with AI tools to learn and implement my solution ideas. 


# Gold Collector Optimization - Computational Intelligence Project

## Problem Description

The Gold Collector problem is a variant of the Vehicle Routing Problem (VRP) where a vehicle must collect gold from multiple cities and return to a central depot. The challenge is to optimize the collection routes while considering the weight penalty of carrying gold.

### Problem Formulation

- **Depot**: Starting point (city 0) where the vehicle begins and must return
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

### Genetic Algorithm with Split Algorithm

Our solution uses a **two-phase approach**:

#### Phase 1: Genetic Algorithm (GA)

1. **Encoding**: Solutions encoded as permutations of cities (visiting order)
2. **Split Algorithm**: Dynamic programming to optimally split any permutation into trips
3. **Genetic Operators**:
   - **Selection**: Tournament selection (size 5)
   - **Crossover**: Ordered Crossover (OX) - preserves relative city ordering
   - **Mutation**: Swap mutation (30% probability)
   - **Elitism**: Top 5 solutions preserved each generation

4. **Parameters**:
   - Population size: 50
   - Generations: 50-100 (adaptive based on problem size)
   - Early stopping: Patience of 20 generations without improvement

#### Phase 2: Local Search Refinement

After GA converges, local search operators refine the solution:

- **Merge**: Combine two trips into one
- **Relocate**: Move a city from one trip to another
- **Split**: Break a trip into two separate trips

Refinement uses hill climbing (only accepts improvements) for 5000 iterations.

### Key Innovation: Correct Cost Evaluation

The critical challenge was ensuring our evaluation matches the problem's baseline calculation. The solution:

1. **Pre-compute shortest paths** between all city pairs as sequences of edges
2. **Traverse each edge individually** when calculating costs, accumulating gold weight
3. This ensures consistency with how the baseline is calculated

## Implementation Details

### File Structure

```
.
├── src/
│   ├── problem.py       # Problem definition and baseline calculation
│   └── solver.py        # GeneticSolver implementation
├── s337798.py           # Main script to run experiments
└── README.md            # This file
```

### Core Classes

#### `GoldProblem` (problem.py)

- Generates random city locations and gold amounts
- Creates graph topology with specified density
- Precomputes shortest path distances
- Provides baseline calculation
- Cost function with weight penalty

#### `GeneticSolver` (solver.py)

- Runs genetic algorithm to find good city permutation
- Split algorithm for optimal trip decomposition
- Local search refinement
- Correct cost evaluation matching baseline

## Running Experiments

### Basic Usage

```bash
python s337798.py
```

This runs experiments on:
- **City counts**: 20, 50, 100
- **Densities**: 0.2 (sparse), 1.0 (dense)
- **Alpha values**: 1, 2
- **Beta values**: 1 (linear), 2 (quadratic)

Total: **18 test cases** (3 city counts × 6 parameter combinations)

### Customization

Modify the `city_counts` list in `s337798.py`:

```python
city_counts = [20, 50, 100]  # Change to your desired sizes
```

### Output Files

- `optimization_results_ga.csv`: Detailed results for all test cases
- `parameter_analysis.png`: Visualizations of improvement by parameters
- `time_analysis.png`: Computation time breakdown


### Genetic Algorithm

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Population Size | 50 | Balance between diversity and computational cost |
| Tournament Size | 5 | Moderate selection pressure |
| Elite Size | 5 | Preserve best 10% of population |
| Mutation Rate | 0.3 | Sufficient exploration without disrupting good solutions |
| Generations | 50-100 | Adaptive: smaller problems converge faster |
| Patience | 20 | Early stopping if no improvement |

### Local Search

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Iterations | 5000 | Quick refinement after GA |
| Strategy | Hill Climbing | Only improvements (no deterioration allowed) |
| Operators | Merge, Relocate, Split | Cover different neighborhood structures |

## Key Challenges and Solutions

### Challenge 1: Cost Evaluation Mismatch

**Problem**: Initial implementation produced costs different from baseline calculation.

**Solution**: 
- Pre-compute shortest paths as edge sequences
- Traverse paths edge-by-edge, accumulating gold weight correctly
- Ensures evaluation matches problem.baseline()

### Challenge 2: High Beta Cases (β=2)

**Problem**: Quadratic penalty makes merging trips often worse than baseline.

**Solution**: 
- Genetic algorithm explores diverse solutions
- Split algorithm finds optimal decomposition for any permutation
- Local search can both merge AND split trips adaptively

### Challenge 3: Sparse Graphs

**Problem**: Long detours make multi-city trips expensive.

**Solution**: 
- Split algorithm correctly accounts for detour costs
- Will naturally prefer single-city trips when beneficial
- No forced merging - data-driven decisions

## Performance Summary

================================================================================
SUMMARY TABLE
================================================================================

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

================================================================================
ANALYSIS BY PARAMETER
================================================================================

--- By City Count ---
  20 cities: avg=1.68%, min=0.00%, max=9.71%
  50 cities: avg=3.03%, min=0.02%, max=12.23%
  100 cities: avg=4.56%, min=0.03%, max=20.03%

--- By Density ---
  Density 0.2: avg=1.50%, min=0.05%, max=7.08%
  Density 1.0: avg=4.68%, min=0.00%, max=20.03%

--- By Alpha ---
  Alpha 1: avg=4.62%, min=0.01%, max=20.03%
  Alpha 2: avg=0.04%, min=0.00%, max=0.08%

--- By Beta ---
  Beta 1: avg=0.06%, min=0.00%, max=0.16%
  Beta 2: avg=9.15%, min=0.14%, max=20.03%

================================================================================
BEST AND WORST CASES
================================================================================

Best Case:
  Cities=100.0, Density=1.0, Alpha=1.0, Beta=2.0
  Improvement: 20.03%

Worst Case:
  Cities=20.0, Density=1.0, Alpha=2.0, Beta=1.0
  Improvement: 0.00%

✓ Results saved to 'optimization_results_ga.csv'

Generating visualizations...
  ✓ Saved parameter_analysis.png
  ✓ Saved time_analysis.png
✓ Visualizations saved

================================================================================
COMPLETE!
================================================================================
## Dependencies

```
numpy
networkx
pandas
matplotlib
tqdm
```


## Author

Student ID: s337798

## License

Academic project for Computational Intelligence course.
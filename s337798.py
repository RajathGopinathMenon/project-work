from src.problem import GoldProblem
from src.GA_solver_A import GeneticSolver
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import time

def run_case(num_cities, density, alpha, beta, ga_generations=100):
    """Run optimization for a single case"""
    print(f"\n{'='*80}")
    print(f"Running: cities={num_cities}, density={density}, alpha={alpha}, beta={beta}")
    print(f"{'='*80}")
    
    # Create problem
    problem = GoldProblem(num_cities=num_cities, alpha=alpha, beta=beta, 
                         density=density, seed=42)
    
    # Initialize solver (runs GA automatically)
    start_time = time.time()
    solver = GeneticSolver(problem)
    ga_time = time.time() - start_time
    
    print(f"\nGA completed in {ga_time:.1f}s")
    print(f"Starting local search refinement...")
    
    # Run local search refinement
    refinement_iterations = 5000
    last_report = solver.best_cost
    
    for i in tqdm(range(refinement_iterations), desc="Refining"):
        solver.step()
        
        if i > 0 and i % 1000 == 0:
            if solver.best_cost < last_report:
                improvement = ((solver.initial_cost - solver.best_cost) / solver.initial_cost) * 100
                print(f"\n    Iter {i}: Cost={solver.best_cost:.2f}, Improvement={improvement:.2f}%")
                last_report = solver.best_cost
    
    elapsed = time.time() - start_time
    
    # Final results
    stats = solver.get_statistics()
    
    print(f"\nResults:")
    print(f"  Final cost: {stats['best_cost']:.2f}")
    print(f"  Final improvement: {stats['improvement_pct']:.2f}%")
    print(f"  Number of trips: {stats['num_trips']}")
    print(f"  Total time: {elapsed:.1f}s (GA: {ga_time:.1f}s, Refinement: {elapsed-ga_time:.1f}s)")
    
    return {
        'num_cities': num_cities,
        'density': density,
        'alpha': alpha,
        'beta': beta,
        'baseline_cost': stats['initial_cost'],
        'final_cost': stats['best_cost'],
        'final_improvement': stats['improvement_pct'],
        'num_trips': stats['num_trips'],
        'time_seconds': elapsed,
        'ga_time': ga_time,
        'refinement_time': elapsed - ga_time
    }

def create_visualizations(df, results):
    """Create comprehensive visualizations"""
    
    # Get unique city counts
    city_counts = sorted(df['num_cities'].unique())
    
    # Figure 1: Improvement by parameters
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # By city count
    ax1 = axes[0, 0]
    x_labels = []
    city_data = {}
    
    # Get one representative set of parameters for x-axis labels
    first_city_count = city_counts[0]
    params = df[df['num_cities'] == first_city_count][['density', 'alpha', 'beta']].values
    x_labels = [f"D={d}\nα={a}\nβ={b}" for d, a, b in params]
    
    # Collect data for each city count
    for city_count in city_counts:
        city_data[city_count] = df[df['num_cities'] == city_count]['final_improvement'].values
    
    x = range(len(x_labels))
    width = 0.8 / len(city_counts)
    
    colors = ['coral', 'steelblue', 'lightgreen', 'gold', 'mediumpurple']
    for idx, (city_count, values) in enumerate(city_data.items()):
        offset = (idx - len(city_counts)/2 + 0.5) * width
        ax1.bar([i + offset for i in x], values, width, 
                label=f'{city_count} cities', alpha=0.7, color=colors[idx % len(colors)])
    
    ax1.set_ylabel('Improvement (%)', fontweight='bold')
    ax1.set_title('Improvement by Problem Size', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # By density
    ax2 = axes[0, 1]
    density_data = df.groupby('density')['final_improvement'].mean()
    density_labels = [f'Density {d}' for d in density_data.index]
    bars = ax2.bar(density_labels, density_data.values, color=['coral', 'steelblue'])
    ax2.set_ylabel('Average Improvement (%)', fontweight='bold')
    ax2.set_title('Impact of Graph Density', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # By alpha
    ax3 = axes[1, 0]
    alpha_data = df.groupby('alpha')['final_improvement'].mean()
    bars = ax3.bar([f'α={a}' for a in alpha_data.index], alpha_data.values, 
                   color=['lightgreen', 'lightcoral'])
    ax3.set_ylabel('Average Improvement (%)', fontweight='bold')
    ax3.set_title('Impact of Alpha (Load Scaling)', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # By beta
    ax4 = axes[1, 1]
    beta_data = df.groupby('beta')['final_improvement'].mean()
    bars = ax4.bar(['β=1 (Linear)', 'β=2 (Quadratic)'], beta_data.values,
                   color=['lightyellow', 'lightblue'])
    ax4.set_ylabel('Average Improvement (%)', fontweight='bold')
    ax4.set_title('Impact of Beta (Load Penalty)', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('parameter_analysis.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved parameter_analysis.png")
    
    # Figure 2: Time breakdown
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time by problem size
    time_by_city = df.groupby('num_cities')['time_seconds'].mean()
    bars = ax1.bar([f'{n} cities' for n in time_by_city.index], time_by_city.values, 
                   color=colors[:len(time_by_city)])
    ax1.set_ylabel('Average Time (seconds)', fontweight='bold')
    ax1.set_title('Computation Time by Problem Size', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # GA vs Refinement time
    avg_ga = df['ga_time'].mean()
    avg_ref = df['refinement_time'].mean()
    ax2.bar(['GA', 'Refinement'], [avg_ga, avg_ref], color=['lightblue', 'lightcoral'])
    ax2.set_ylabel('Average Time (seconds)', fontweight='bold')
    ax2.set_title('Time Distribution: GA vs Refinement', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, (label, val) in enumerate([('GA', avg_ga), ('Refinement', avg_ref)]):
        ax2.text(i, val, f'{val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('time_analysis.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved time_analysis.png")

def main():
    print("="*80)
    print("GOLD COLLECTOR - GENETIC ALGORITHM + LOCAL SEARCH")
    print("="*80)
    
    # Define city counts to test
    city_counts = [20, 50, 100, 1000]
    
    # Define parameter combinations
    param_combinations = [
        (0.2, 1, 1),
        (0.2, 2, 1),
        (0.2, 1, 2),
        (1.0, 1, 1),
        (1.0, 2, 1),
        (1.0, 1, 2),
    ]
    
    # Build test cases
    test_cases = []
    for num_cities in city_counts:
        for density, alpha, beta in param_combinations:
            test_cases.append((num_cities, density, alpha, beta))
    
    print(f"\nRunning {len(test_cases)} test cases:")
    print(f"  City counts: {city_counts}")
    print(f"  Parameter combinations: {len(param_combinations)}")
    
    # Run all cases
    results = []
    for num_cities, density, alpha, beta in test_cases:
        # Adjust GA generations based on problem size
        if num_cities <= 20:
            ga_gens = 50
        elif num_cities <= 50:
            ga_gens = 75
        else:
            ga_gens = 100
        
        result = run_case(num_cities, density, alpha, beta, ga_gens)
        results.append(result)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"\n{'Cities':<8} {'Density':<8} {'Alpha':<7} {'Beta':<6} "
          f"{'Baseline':<12} {'Final':<12} {'Improve%':<10} {'Trips':<7} {'Time(s)':<8}")
    print("-"*85)
    
    for _, row in df.iterrows():
        print(f"{row['num_cities']:<8} {row['density']:<8} {row['alpha']:<7} {row['beta']:<6} "
              f"{row['baseline_cost']:<12.2f} {row['final_cost']:<12.2f} "
              f"{row['final_improvement']:<10.2f} {row['num_trips']:<7} {row['time_seconds']:<8.1f}")
    
    # Analysis by parameter
    print("\n" + "="*80)
    print("ANALYSIS BY PARAMETER")
    print("="*80)
    
    print("\n--- By City Count ---")
    city_groups = df.groupby('num_cities')['final_improvement'].agg(['mean', 'min', 'max'])
    for cities, row in city_groups.iterrows():
        print(f"  {cities} cities: avg={row['mean']:.2f}%, min={row['min']:.2f}%, max={row['max']:.2f}%")
    
    print("\n--- By Density ---")
    density_groups = df.groupby('density')['final_improvement'].agg(['mean', 'min', 'max'])
    for dens, row in density_groups.iterrows():
        print(f"  Density {dens}: avg={row['mean']:.2f}%, min={row['min']:.2f}%, max={row['max']:.2f}%")
    
    print("\n--- By Alpha ---")
    alpha_groups = df.groupby('alpha')['final_improvement'].agg(['mean', 'min', 'max'])
    for alph, row in alpha_groups.iterrows():
        print(f"  Alpha {alph}: avg={row['mean']:.2f}%, min={row['min']:.2f}%, max={row['max']:.2f}%")
    
    print("\n--- By Beta ---")
    beta_groups = df.groupby('beta')['final_improvement'].agg(['mean', 'min', 'max'])
    for bet, row in beta_groups.iterrows():
        print(f"  Beta {bet}: avg={row['mean']:.2f}%, min={row['min']:.2f}%, max={row['max']:.2f}%")
    
    # Best and worst cases
    print("\n" + "="*80)
    print("BEST AND WORST CASES")
    print("="*80)
    
    best_idx = df['final_improvement'].idxmax()
    worst_idx = df['final_improvement'].idxmin()
    
    print("\nBest Case:")
    best = df.iloc[best_idx]
    print(f"  Cities={best['num_cities']}, Density={best['density']}, Alpha={best['alpha']}, Beta={best['beta']}")
    print(f"  Improvement: {best['final_improvement']:.2f}%")
    
    print("\nWorst Case:")
    worst = df.iloc[worst_idx]
    print(f"  Cities={worst['num_cities']}, Density={worst['density']}, Alpha={worst['alpha']}, Beta={worst['beta']}")
    print(f"  Improvement: {worst['final_improvement']:.2f}%")
    
    # Save results
    df.to_csv('optimization_results.csv', index=False)
    print("\n✓ Results saved to 'optimization_results.csv'")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(df, results)
    print("✓ Visualizations saved")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
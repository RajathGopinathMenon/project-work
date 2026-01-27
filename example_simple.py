"""
Simple example demonstrating the solution() function.
"""
from Problem import Problem
from s337798 import solution
import matplotlib.pyplot as plt


# Example 1: Small problem
print("Example 1: Small problem")
print("-" * 40)
p = Problem(num_cities=20, alpha=1, beta=2, density=1, seed=42)

path = solution(p)

print(f"Output: {path}")  # Show first 10 entries
print(f"\nFormat: [(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]")


# Example 2: Verify format
print("\n" + "="*40)
print("Example 2: Format verification")
print("-" * 40)
p2 = Problem(num_cities=20, alpha=1, beta=2, density=1, seed=123)

path2 = solution(p2)

print("First trip:")
for i, (city, gold) in enumerate(path2):
    if city == 0:
        print(f"  {i}. (0, 0) ← Return to base")
        break
    else:
        print(f"  {i}. ({city}, {gold:.1f}) ← Visit city {city}, collect {gold:.1f} gold")


from Problem import Problem
from s337798 import solution


p.plot_solution(path, save_path='my_solution.png')
plt.show()

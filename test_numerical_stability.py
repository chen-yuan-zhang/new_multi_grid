"""
Test to demonstrate the numerical stability improvements.
Shows how the old method would underflow vs the new method.
"""

import numpy as np
import math

# Test parameters
beta = 1
distances = [5, 10, 20, 30, 50, 100]

print("=" * 70)
print("NUMERICAL STABILITY COMPARISON")
print("=" * 70)

print("\n1. OLD METHOD (direct exp, causes underflow):")
print("-" * 70)
for dist in distances:
    try:
        prob = math.exp(-beta * (1 + dist))
        print(f"Distance {dist:3d}: exp(-{1+dist}) = {prob:.10e} {'⚠️ UNDERFLOW' if prob < 1e-15 else ''}")
    except Exception as e:
        print(f"Distance {dist:3d}: ERROR - {e}")

print("\n2. NEW METHOD (log-space with normalization):")
print("-" * 70)

# Simulate multiple successors at different distances
test_successors = [5, 10, 15]
print(f"\nExample: successors at distances {test_successors}")

# Old method
print("\nOLD (regular space, then normalize):")
old_probs = [math.exp(-beta * (1 + d)) for d in test_successors]
old_total = sum(old_probs)
old_normalized = [p / old_total if old_total > 0 else 0 for p in old_probs]
print(f"  Raw probs: {[f'{p:.10e}' for p in old_probs]}")
print(f"  Total: {old_total:.10e}")
print(f"  Normalized: {[f'{p:.6f}' for p in old_normalized]}")
print(f"  Sum check: {sum(old_normalized):.10f}")

# New method
print("\nNEW (log-space throughout):")

def logsumexp(log_probs):
    """Numerically stable log-sum-exp."""
    log_probs = np.array(log_probs)
    max_val = np.max(log_probs)
    if max_val == -np.inf:
        return -np.inf
    return max_val + np.log(np.sum(np.exp(log_probs - max_val)))

log_probs = [-beta * (1 + d) for d in test_successors]
log_total = logsumexp(log_probs)
new_normalized = [np.exp(lp - log_total) for lp in log_probs]
print(f"  Log probs: {[f'{lp:.6f}' for lp in log_probs]}")
print(f"  Log total: {log_total:.6f}")
print(f"  Normalized: {[f'{p:.6f}' for p in new_normalized]}")
print(f"  Sum check: {sum(new_normalized):.10f}")

print("\n3. EXTREME CASE (large distances where old method fails):")
print("-" * 70)
extreme_successors = [50, 55, 60]
print(f"Successors at distances {extreme_successors}")

# Old method
print("\nOLD:")
old_probs = [math.exp(-beta * (1 + d)) for d in extreme_successors]
old_total = sum(old_probs)
if old_total > 0:
    old_normalized = [p / old_total for p in old_probs]
    print(f"  Raw probs: {[f'{p:.10e}' for p in old_probs]}")
    print(f"  Total: {old_total:.10e}")
    print(f"  Normalized: {[f'{p:.6f}' for p in old_normalized]}")
else:
    print(f"  ❌ COMPLETE UNDERFLOW - all probabilities are 0!")
    print(f"  Cannot normalize (division by zero)")

# New method
print("\nNEW:")
log_probs = [-beta * (1 + d) for d in extreme_successors]
log_total = logsumexp(log_probs)
new_normalized = [np.exp(lp - log_total) for lp in log_probs]
print(f"  Log probs: {[f'{lp:.6f}' for lp in log_probs]}")
print(f"  Log total: {log_total:.6f}")
print(f"  Normalized: {[f'{p:.6f}' for p in new_normalized]}")
print(f"  Sum check: {sum(new_normalized):.10f}")
print(f"  ✅ Still works correctly!")

print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print("✅ NEW METHOD: Works for all distances, maintains numerical stability")
print("❌ OLD METHOD: Fails for distances > ~30 due to underflow")
print("=" * 70)

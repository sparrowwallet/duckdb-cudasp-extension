#!/usr/bin/env python3
"""
Compute expected lower 64 bits for gECC test case 0.
"""

import sys
sys.path.insert(0, '.')

import field

# secp256k1 field
fq = field.Fq_SECP256K1

# From gECC test output - result_x[0] in Montgomery form
result_x_mont = 0xbce9d493b5ebeeff5a5f128ce1405d9ee2df5318eaaa70a863ef06c805eb176a

# Convert from Montgomery to normal form
result_x_normal = fq.from_mont(result_x_mont)

print(f"Result X (Montgomery): {result_x_mont:064x}")
print(f"Result X (Normal):     {result_x_normal:064x}")

# Extract lower 64 bits
lower_64_bits = result_x_normal & 0xFFFFFFFFFFFFFFFF
print(f"\nLower 64 bits:")
print(f"  Hex: {lower_64_bits:016x}")
print(f"  Dec (unsigned): {lower_64_bits}")
print(f"  Dec (signed int64): {lower_64_bits if lower_64_bits < (1 << 63) else lower_64_bits - (1 << 64)}")

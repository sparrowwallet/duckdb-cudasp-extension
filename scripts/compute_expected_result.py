#!/usr/bin/env python3
"""
Compute the expected result for our test scalar multiplication.
"""

import sys
sys.path.insert(0, '.')

import ec
import field

# secp256k1 curve
curve = ec.G1_SECP256K1
fq = field.Fq_SECP256K1

# Our test scalar (repeating pattern)
s = 0xb8b9babbbcbdbebfb0b1b2b3b4b5b6b7a8a9aaabacadaeafa0a1a2a3a4a5a6a7

# Generator point G for secp256k1
G_x = 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
G_y = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8

print("="*70)
print("Computing Expected Result for EC Multiplication")
print("="*70)
print(f"\nScalar s = {s:064x}")
print(f"Point G_x = {G_x:064x}")
print(f"Point G_y = {G_y:064x}")

# Compute s * G
point_affine = (G_x, G_y)
point_jac = curve.to_jacobian(point_affine)
result_jac = curve.multiply_jacobian(point_jac, s)
result_x, result_y = curve.get_xy(result_jac)

print(f"\nExpected Result (normal form):")
print(f"  result_x = {result_x:064x}")
print(f"  result_y = {result_y:064x}")

# Extract lower 64 bits
lower_64_bits = result_x & 0xFFFFFFFFFFFFFFFF
print(f"\nLower 64 bits of result_x:")
print(f"  Hex: {lower_64_bits:016x}")
print(f"  Dec: {lower_64_bits}")
print(f"  As int64: {lower_64_bits if lower_64_bits < (1 << 63) else lower_64_bits - (1 << 64)}")

# Also compute in Montgomery form to double-check
mont_x = fq.to_mont(result_x)
print(f"\nExpected Result (Montgomery form):")
print(f"  result_x_mont = {mont_x:064x}")

# Split into u32 limbs (little-endian)
limbs = []
for i in range(8):
    limb = (result_x >> (32 * i)) & 0xFFFFFFFF
    limbs.append(limb)

print(f"\nResult X in u32 limbs (little-endian order, digits[0] to digits[7]):")
for i, limb in enumerate(limbs):
    print(f"  digits[{i}] = 0x{limb:08x} ({limb})")

print(f"\nResult X in u32 limbs (big-endian display order, as hex string):")
hex_str = ''.join([f'{limbs[7-i]:08x}' for i in range(8)])
print(f"  {hex_str}")

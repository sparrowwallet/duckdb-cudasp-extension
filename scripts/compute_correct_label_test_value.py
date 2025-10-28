#!/usr/bin/env python3
"""
Compute the correct output values for the height=400 label test.
We need the label case to produce a value that's NOT in the base case.
"""

import sys
import os
import hashlib

# Add gECC scripts to path
script_dir = os.path.dirname(os.path.abspath(__file__))
gecc_scripts_dir = os.path.join(script_dir, '..', 'gECC', 'scripts')
sys.path.insert(0, gecc_scripts_dir)

from constants import SECP256K1_q, SECP256K1_n

p = SECP256K1_q
n = SECP256K1_n

def decompress_point(compressed_hex):
    """Decompress a compressed SEC1 point to (x, y) integers."""
    prefix = int(compressed_hex[0:2], 16)
    x_hex = compressed_hex[2:]
    x = int(x_hex, 16)

    y_squared = (pow(x, 3, p) + 7) % p
    y = pow(y_squared, (p + 1) // 4, p)

    if (y % 2) != (prefix % 2):
        y = p - y

    return x, y

def point_add(x1, y1, x2, y2):
    """Add two EC points on secp256k1."""
    if x1 == 0 and y1 == 0:
        return x2, y2
    if x2 == 0 and y2 == 0:
        return x1, y1

    if x1 == x2:
        if y1 == y2:
            s = (3 * x1 * x1 * pow(2 * y1, -1, p)) % p
        else:
            return 0, 0
    else:
        s = ((y2 - y1) * pow(x2 - x1, -1, p)) % p

    x3 = (s * s - x1 - x2) % p
    y3 = (s * (x1 - x3) - y1) % p

    return x3, y3

def scalar_mult(k, x, y):
    """Scalar multiplication k * (x, y)."""
    if k == 0:
        return 0, 0

    result_x, result_y = 0, 0
    addend_x, addend_y = x, y

    while k > 0:
        if k & 1:
            result_x, result_y = point_add(result_x, result_y, addend_x, addend_y)
        addend_x, addend_y = point_add(addend_x, addend_y, addend_x, addend_y)
        k >>= 1

    return result_x, result_y

def bip340_tagged_hash(tag, data):
    """BIP-340 tagged hash."""
    tag_hash = hashlib.sha256(tag.encode()).digest()
    return hashlib.sha256(tag_hash + tag_hash + data).digest()

# Input values
scan_private_key_hex = "0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c"
spend_public_key_compressed = "025cc9856d6f8375350e123978daac200c260cb5b5ae83106cab90484dcd8fcf36"
tweak_key_compressed = "0314bec14463d6c0181083d607fecfba67bb83f95915f6f247975ec566d5642ee8"
label_key_compressed = "034e52d154b56ffe17964bd72e1dc4478c956f3fa29e1ea7e8bdee2d2a21f963cd"

print("=== Computing Correct Label Test Values ===\n")
print("Keys:")
print(f"  scan_private_key: {scan_private_key_hex}")
print(f"  spend_public_key: {spend_public_key_compressed}")
print(f"  tweak_key: {tweak_key_compressed}")
print(f"  label_key: {label_key_compressed}")
print()

# Parse keys
scan_private_key = int(scan_private_key_hex, 16)
spend_x, spend_y = decompress_point(spend_public_key_compressed)
tweak_x, tweak_y = decompress_point(tweak_key_compressed)
label_x, label_y = decompress_point(label_key_compressed)

# Compute output_point following BIP-352 pipeline
# 1. shared_secret = scan_private_key * tweak_key
shared_x, shared_y = scalar_mult(scan_private_key, tweak_x, tweak_y)

# 2. serialized = compressed_sec1(shared_secret) || 0x00000000
prefix = 0x02 if (shared_y % 2 == 0) else 0x03
serialized_shared_secret = bytes([prefix]) + shared_x.to_bytes(32, 'big') + b'\x00\x00\x00\x00'

# 3. t = tagged_hash("BIP0352/SharedSecret", serialized)
t_bytes = bip340_tagged_hash("BIP0352/SharedSecret", serialized_shared_secret)
t = int.from_bytes(t_bytes, 'big') % n

# 4. output_point = t × G
G_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
G_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
output_x, output_y = scalar_mult(t, G_x, G_y)

# Base case
base_x, base_y = point_add(output_x, output_y, spend_x, spend_y)
base_x_hex = f"{base_x:064x}"
base_upper_64 = base_x_hex[:16]
base_value_unsigned = int(base_upper_64, 16)
base_value = base_value_unsigned if base_value_unsigned < 2**63 else base_value_unsigned - 2**64

print("BASE CASE (output_point + spend_public_key):")
print(f"  Full X: {base_x_hex}")
print(f"  Upper 64 bits (hex): {base_upper_64}")
print(f"  Upper 64 bits (signed int64): {base_value}")
print()

# Label case
label_result_x, label_result_y = point_add(output_x, output_y, label_x, label_y)
label_x_hex = f"{label_result_x:064x}"
label_upper_64 = label_x_hex[:16]
label_value_unsigned = int(label_upper_64, 16)
label_value = label_value_unsigned if label_value_unsigned < 2**63 else label_value_unsigned - 2**64

print("LABEL CASE (output_point + label_key):")
print(f"  Full X: {label_x_hex}")
print(f"  Upper 64 bits (hex): {label_upper_64}")
print(f"  Upper 64 bits (signed int64): {label_value}")
print(f"  Negated (signed int64): {-label_value}")
print()

print("=== CORRECT TEST DATA ===")
print()
print("For the test to work correctly, use ONE of these output values:")
print()
print("Option 1: Use the LABEL case value (positive):")
print(f"  output: {label_value}")
print(f"  This will match when label_key is provided")
print(f"  Will NOT match without label_key (base case produces {base_value})")
print()
print("Option 2: Use the LABEL case value (negated):")
print(f"  output: {-label_value}")
print(f"  This will match when label_key is provided (after negation)")
print(f"  Will NOT match without label_key")
print()
print("CURRENT test is using: -1006811617310360495")
print("This value doesn't match either case, so the test data is incorrect.")

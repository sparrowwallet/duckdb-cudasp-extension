#!/usr/bin/env python3
"""
Verify which case produces the output value: base or label.
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
expected_x_coord = "f207162b1a7abc51c42017bef055e9ec1efc3d3567cb720357e2b84325db33ac"
expected_output = -1006811617310360495

print("=== Verifying Which Case Produces the Output ===\n")

# Parse keys
scan_private_key = int(scan_private_key_hex, 16)
spend_x, spend_y = decompress_point(spend_public_key_compressed)
tweak_x, tweak_y = decompress_point(tweak_key_compressed)
label_x, label_y = decompress_point(label_key_compressed)

print("Expected x-coordinate (from test vectors):")
print(f"  {expected_x_coord}")
print(f"  Upper 64 bits: {expected_x_coord[:16]} = {expected_output}")
print()

# Compute shared secret and output_point
shared_x, shared_y = scalar_mult(scan_private_key, tweak_x, tweak_y)
prefix = 0x02 if (shared_y % 2 == 0) else 0x03
serialized_shared_secret = bytes([prefix]) + shared_x.to_bytes(32, 'big')

input_hash = bip340_tagged_hash("BIP0352/Inputs", b'\x00' * 36)
shared_secret_hash = bip340_tagged_hash("BIP0352/SharedSecret", serialized_shared_secret)
t_bytes = bip340_tagged_hash("BIP0352/SharedSecret", input_hash + shared_secret_hash)
t = int.from_bytes(t_bytes, 'big') % n

G_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
G_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
output_x, output_y = scalar_mult(t, G_x, G_y)

print("output_point (from BIP-352 derivation):")
print(f"  X: {output_x:064x}")
print(f"  Y: {output_y:064x}")
print()

# Base case: output_point + spend_public_key
print("=== BASE CASE: output_point + spend_public_key ===")
base_x, base_y = point_add(output_x, output_y, spend_x, spend_y)
base_x_hex = f"{base_x:064x}"
base_upper_64 = base_x_hex[:16]
base_value_unsigned = int(base_upper_64, 16)
base_value = base_value_unsigned if base_value_unsigned < 2**63 else base_value_unsigned - 2**64

print(f"Result X: {base_x_hex}")
print(f"Upper 64 bits: {base_upper_64}")
print(f"As signed int64: {base_value}")
print(f"MATCHES expected? {base_x_hex == expected_x_coord}")
print()

# Label case: output_point + label_key
print("=== LABEL CASE: output_point + label_key ===")
label_result_x, label_result_y = point_add(output_x, output_y, label_x, label_y)
label_x_hex = f"{label_result_x:064x}"
label_upper_64 = label_x_hex[:16]
label_value_unsigned = int(label_upper_64, 16)
label_value = label_value_unsigned if label_value_unsigned < 2**63 else label_value_unsigned - 2**64

print(f"Result X: {label_x_hex}")
print(f"Upper 64 bits: {label_upper_64}")
print(f"As signed int64: {label_value}")
print(f"As negated int64: {-label_value}")
print(f"MATCHES expected x-coord? {label_x_hex == expected_x_coord}")
print(f"MATCHES expected output? {label_value == expected_output}")
print(f"NEGATED matches expected output? {-label_value == expected_output}")
print()

print("=== CONCLUSION ===")
if base_x_hex == expected_x_coord:
    print("❌ PROBLEM: The expected x-coordinate comes from the BASE case (spend_public_key)!")
    print("   This means the test is WRONG - we WILL get a match without labels.")
    print("   The test expects 0 matches without labels, but we correctly return 1.")
elif label_x_hex == expected_x_coord:
    print("✓ CORRECT: The expected x-coordinate comes from the LABEL case.")
    print("  The test should work as expected - no match without labels.")
elif label_value == expected_output or -label_value == expected_output:
    print("✓ CORRECT: The expected OUTPUT VALUE comes from the LABEL case (via upper 64 bits).")
    print("  The test should work as expected - no match without labels.")
else:
    print("❓ UNKNOWN: Neither base nor label case produces the expected value!")
    print("  There may be an error in the test data.")

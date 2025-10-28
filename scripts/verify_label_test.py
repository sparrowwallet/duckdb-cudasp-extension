#!/usr/bin/env python3
"""
Verify the label checking test expectations.
This script computes what values we should get for both:
1. Base case: output_point + spend_public_key
2. Label case: output_point + label_key (and its negation)
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

    # Decompress: compute y from x
    y_squared = (pow(x, 3, p) + 7) % p
    y = pow(y_squared, (p + 1) // 4, p)

    # Choose the correct y based on parity
    if (y % 2) != (prefix % 2):
        y = p - y

    return x, y

def point_add(x1, y1, x2, y2):
    """Add two EC points on secp256k1."""
    if x1 == 0 and y1 == 0:
        return x2, y2
    if x2 == 0 and y2 == 0:
        return x1, y1

    # Compute slope
    if x1 == x2:
        if y1 == y2:
            # Point doubling
            s = (3 * x1 * x1 * pow(2 * y1, -1, p)) % p
        else:
            # Points are inverses
            return 0, 0
    else:
        # Point addition
        s = ((y2 - y1) * pow(x2 - x1, -1, p)) % p

    # Compute result
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

print("=== Label Test Verification (Height 400) ===\n")

# Input values
scan_private_key_hex = "0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c"
spend_public_key_compressed = "025cc9856d6f8375350e123978daac200c260cb5b5ae83106cab90484dcd8fcf36"
tweak_key_compressed = "0314bec14463d6c0181083d607fecfba67bb83f95915f6f247975ec566d5642ee8"
label_key_compressed = "034e52d154b56ffe17964bd72e1dc4478c956f3fa29e1ea7e8bdee2d2a21f963cd"
expected_output = -1006811617310360495

print("Input values:")
print(f"  scan_private_key: {scan_private_key_hex}")
print(f"  spend_public_key: {spend_public_key_compressed}")
print(f"  tweak_key: {tweak_key_compressed}")
print(f"  label_key: {label_key_compressed}")
print(f"  expected_output: {expected_output}")
print()

# Parse keys
scan_private_key = int(scan_private_key_hex, 16)
spend_x, spend_y = decompress_point(spend_public_key_compressed)
tweak_x, tweak_y = decompress_point(tweak_key_compressed)
label_x, label_y = decompress_point(label_key_compressed)

print("Decompressed points:")
print(f"  spend_public_key: ({spend_x:064x}, {spend_y:064x})")
print(f"  tweak_key: ({tweak_x:064x}, {tweak_y:064x})")
print(f"  label_key: ({label_x:064x}, {label_y:064x})")
print()

# Step 1: Compute shared secret = scan_private_key * tweak_key
print("Step 1: Compute shared secret")
shared_x, shared_y = scalar_mult(scan_private_key, tweak_x, tweak_y)
print(f"  shared_secret = scan_private_key * tweak_key")
print(f"  shared_secret: ({shared_x:064x}, {shared_y:064x})")
print()

# Step 2: Serialize shared secret (compressed form with even y)
print("Step 2: Serialize shared secret")
prefix = 0x02 if (shared_y % 2 == 0) else 0x03
serialized_shared_secret = bytes([prefix]) + shared_x.to_bytes(32, 'big')
print(f"  Compressed (SEC1): {serialized_shared_secret.hex()}")
print()

# Step 3: Compute tagged hash
print("Step 3: Compute BIP340 tagged hash")
input_hash = bip340_tagged_hash("BIP0352/Inputs", b'\x00' * 36)  # Simplified
shared_secret_hash = bip340_tagged_hash("BIP0352/SharedSecret", serialized_shared_secret)
t_bytes = bip340_tagged_hash("BIP0352/SharedSecret", input_hash + shared_secret_hash)
t = int.from_bytes(t_bytes, 'big') % n
print(f"  t = {t:064x}")
print()

# Step 4: Compute output_point = t * G
print("Step 4: Compute output_point = t * G")
# G is the secp256k1 generator
G_x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
G_y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
output_x, output_y = scalar_mult(t, G_x, G_y)
print(f"  output_point: ({output_x:064x}, {output_y:064x})")
print()

# Step 5: Base case - Add spend_public_key to output_point
print("Step 5: Base case - output_point + spend_public_key")
base_x, base_y = point_add(output_x, output_y, spend_x, spend_y)
print(f"  result: ({base_x:064x}, {base_y:064x})")

# Extract upper 64 bits
base_x_hex = f"{base_x:064x}"
upper_64_hex = base_x_hex[:16]
upper_64_unsigned = int(upper_64_hex, 16)
if upper_64_unsigned >= 2**63:
    base_value = upper_64_unsigned - 2**64
else:
    base_value = upper_64_unsigned

print(f"  Upper 64 bits (hex): {upper_64_hex}")
print(f"  Upper 64 bits (signed): {base_value}")
print(f"  Matches expected output? {base_value == expected_output}")
print()

# Step 6: Label case - Add label_key to output_point
print("Step 6: Label case - output_point + label_key")
label_result_x, label_result_y = point_add(output_x, output_y, label_x, label_y)
print(f"  result: ({label_result_x:064x}, {label_result_y:064x})")

# Extract upper 64 bits
label_x_hex = f"{label_result_x:064x}"
upper_64_hex_label = label_x_hex[:16]
upper_64_unsigned_label = int(upper_64_hex_label, 16)
if upper_64_unsigned_label >= 2**63:
    label_value = upper_64_unsigned_label - 2**64
else:
    label_value = upper_64_unsigned_label

print(f"  Upper 64 bits (hex): {upper_64_hex_label}")
print(f"  Upper 64 bits (signed): {label_value}")
print(f"  Matches expected output? {label_value == expected_output}")

# Check negation
label_value_neg = -label_value
print(f"  Negated value: {label_value_neg}")
print(f"  Negated matches expected output? {label_value_neg == expected_output}")
print()

print("=== Summary ===")
print(f"Expected output: {expected_output}")
print(f"Base case (spend_public_key): {base_value} - {'MATCH' if base_value == expected_output else 'NO MATCH'}")
print(f"Label case (label_key): {label_value} - {'MATCH' if label_value == expected_output else 'NO MATCH'}")
print(f"Label case negated: {label_value_neg} - {'MATCH' if label_value_neg == expected_output else 'NO MATCH'}")
print()

if base_value == expected_output:
    print("WARNING: Base case matches! This test won't properly test label checking.")
    print("The test expects the base case NOT to match, but it does.")
elif label_value == expected_output or label_value_neg == expected_output:
    print("OK: Only label case matches. Test is correctly designed.")
else:
    print("ERROR: Neither base case nor label case matches!")

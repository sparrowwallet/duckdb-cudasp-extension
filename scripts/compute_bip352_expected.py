#!/usr/bin/env python3
"""
Compute expected lower 64 bits for BIP-352 Silent Payment pipeline.

Pipeline:
1. First EC multiply: tweak_key * scan_private_key → shared_secret
2. Serialize shared_secret to compressed SEC1 (33 bytes) + 4 zero bytes
3. Compute tagged hash: SHA256(SHA256(tag) || SHA256(tag) || serialized)
4. Second EC multiply: hash × G → output_point
5. Extract lower 64 bits of output_point.x
"""

import hashlib
import sys
import os

# Add gECC scripts to path
script_dir = os.path.dirname(os.path.abspath(__file__))
gecc_scripts_dir = os.path.join(script_dir, '..', 'gECC', 'scripts')
sys.path.insert(0, gecc_scripts_dir)

# Import field and EC operations from gECC scripts
import field
from constants import SECP256K1_q, SECP256K1_n, SECP256K1_g1_generator

# secp256k1 parameters
fq = field.Fq_SECP256K1
p = SECP256K1_q
n = SECP256K1_n
Gx, Gy = SECP256K1_g1_generator

def point_add(P, Q):
    """Add two EC points on secp256k1."""
    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    if x1 == x2:
        if y1 == y2:
            return point_double(P)
        else:
            return None  # Point at infinity

    # s = (y2 - y1) / (x2 - x1)
    s = (y2 - y1) * pow(x2 - x1, -1, p) % p
    x3 = (s * s - x1 - x2) % p
    y3 = (s * (x1 - x3) - y1) % p

    return (x3, y3)

def point_double(P):
    """Double an EC point on secp256k1."""
    if P is None:
        return None

    x, y = P

    # s = (3 * x^2) / (2 * y)  [since a=0 for secp256k1]
    s = (3 * x * x) * pow(2 * y, -1, p) % p
    x3 = (s * s - 2 * x) % p
    y3 = (s * (x - x3) - y) % p

    return (x3, y3)

def point_multiply(k, P):
    """Scalar multiplication k * P using double-and-add."""
    if k == 0:
        return None

    result = None
    addend = P

    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_double(addend)
        k >>= 1

    return result

def compress_point(x, y):
    """Convert EC point to compressed SEC1 format (33 bytes)."""
    prefix = 0x02 if (y & 1) == 0 else 0x03
    return bytes([prefix]) + x.to_bytes(32, 'big')

def tagged_hash(tag, msg):
    """Compute BIP-340 style tagged hash: SHA256(SHA256(tag) || SHA256(tag) || msg)."""
    tag_hash = hashlib.sha256(tag).digest()
    return hashlib.sha256(tag_hash + tag_hash + msg).digest()

# Test case 0 from gECC correctness test
print("=== BIP-352 Silent Payment Expected Value Calculation ===\n")

# Input: scalar and tweak key from test
scan_private_key = 0x0278927476e92caa3912937a7f003e45c741ddc47d80d70ae8f35c0c7f3c78fd
tweak_x = 0xef8ef523cd9e1a96dc497886b69cfc28474207c5679252541288869af65ee7f9
tweak_y = 0xf59a57a32f25c0b0963dc44e5a268c1e258a118cfaecda3dadd2394b3e4bacc8

print(f"Inputs:")
print(f"  scan_private_key = {scan_private_key:064x}")
print(f"  tweak_key.x = {tweak_x:064x}")
print(f"  tweak_key.y = {tweak_y:064x}")
print()

# Step 1: First EC multiply - tweak_key * scan_private_key
print("Step 1: First EC multiply (tweak_key * scan_private_key)")
shared_secret = point_multiply(scan_private_key, (tweak_x, tweak_y))
print(f"  shared_secret.x = {shared_secret[0]:064x}")
print(f"  shared_secret.y = {shared_secret[1]:064x}")
print()

# Step 2: Serialize to compressed SEC1 + 4 zero bytes
print("Step 2: Serialize to compressed SEC1 + 4 zero bytes")
compressed = compress_point(shared_secret[0], shared_secret[1])
serialized = compressed + b'\x00\x00\x00\x00'
print(f"  compressed SEC1 = {compressed.hex()}")
print(f"  serialized (37 bytes) = {serialized.hex()}")
print()

# Step 3: Compute BIP-352 tagged hash
print("Step 3: Compute BIP-352 tagged hash")
tag = b"BIP0352/SharedSecret"
hash_result = tagged_hash(tag, serialized)
hash_int = int.from_bytes(hash_result, 'big')
print(f"  tag = {tag.decode()}")
print(f"  hash = {hash_result.hex()}")
print(f"  hash (as integer) = {hash_int:064x}")
print()

# Step 4: Second EC multiply - hash × G
print("Step 4: Second EC multiply (hash × G)")
G = (Gx, Gy)
output_point = point_multiply(hash_int, G)
print(f"  output_point.x = {output_point[0]:064x}")
print(f"  output_point.y = {output_point[1]:064x}")
print()

# Step 5: Extract lower 64 bits
print("Step 5: Extract lower 64 bits of output_point.x")
lower_64_bits = output_point[0] & 0xFFFFFFFFFFFFFFFF
print(f"  Lower 64 bits (hex): {lower_64_bits:016x}")
print(f"  Lower 64 bits (unsigned): {lower_64_bits}")
print(f"  Lower 64 bits (signed int64): {lower_64_bits if lower_64_bits < (1 << 63) else lower_64_bits - (1 << 64)}")
print()

print("=== Expected value for test ===")
print(f"Row 0 expected output: {lower_64_bits if lower_64_bits < (1 << 63) else lower_64_bits - (1 << 64)}")

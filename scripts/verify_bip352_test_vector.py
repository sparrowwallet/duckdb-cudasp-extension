#!/usr/bin/env python3
"""
Verify BIP-352 test vector by computing the complete pipeline step by step.
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

# BIP-352 test vector inputs
print("=== BIP-352 Test Vector Verification ===\n")

scan_private_key_hex = "0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c"
spend_public_key_hex = "36cf8fcd4d4890ab6c1083aeb5b50c260c20acda7839120e3575836f6d85c95ce0d705e31ff9fdcce67a8f3598871c6dfbe6bcde8a51cb7b48b0f95be0ea94de"
# tweak_key from decompressed point (little-endian x||y, 64 bytes)
tweak_key_hex = "040096db612390ee6cef521e784c897c446a26cea8e28819962e5316c253c24a501e53f71071162afab559954064f0ccb7a6779c23b305597b6335829cc1f5b7"
expected_output = 4512552348537027144

print("Inputs (as provided):")
print(f"  scan_private_key (big-endian): {scan_private_key_hex}")
print(f"  spend_public_key (little-endian x||y): {spend_public_key_hex}")
print(f"  tweak_key (little-endian x||y, 64 bytes): {tweak_key_hex}")
print(f"  expected_output: {expected_output}")
print()

# Parse scan_private_key (big-endian scalar)
scan_private_key = int.from_bytes(bytes.fromhex(scan_private_key_hex), 'big')
print(f"Parsed scan_private_key (scalar): {scan_private_key:064x}")
print()

# Parse spend_public_key (little-endian x||y, 64 bytes total)
spend_pubkey_bytes = bytes.fromhex(spend_public_key_hex)
spend_x_le = spend_pubkey_bytes[:32]
spend_y_le = spend_pubkey_bytes[32:]
spend_x = int.from_bytes(spend_x_le, 'little')
spend_y = int.from_bytes(spend_y_le, 'little')
print(f"Parsed spend_public_key:")
print(f"  x (from little-endian): {spend_x:064x}")
print(f"  y (from little-endian): {spend_y:064x}")
print()

# Parse tweak_key (little-endian x||y, 64 bytes total)
tweak_key_bytes = bytes.fromhex(tweak_key_hex)
tweak_x_le = tweak_key_bytes[:32]
tweak_y_le = tweak_key_bytes[32:]
tweak_x = int.from_bytes(tweak_x_le, 'little')
tweak_y = int.from_bytes(tweak_y_le, 'little')
print(f"Parsed tweak_key (little-endian x||y):")
print(f"  x (from little-endian): {tweak_x:064x}")
print(f"  y (from little-endian): {tweak_y:064x}")
print()

# Verify points are on the curve: y^2 = x^3 + 7 (mod p)
def verify_point(x, y, name):
    lhs = (y * y) % p
    rhs = (x * x * x + 7) % p
    if lhs == rhs:
        print(f"✓ {name} is on the curve")
        return True
    else:
        print(f"✗ {name} is NOT on the curve!")
        print(f"  y^2 mod p = {lhs:064x}")
        print(f"  x^3 + 7 mod p = {rhs:064x}")
        return False

print("Point verification:")
verify_point(spend_x, spend_y, "spend_public_key")
verify_point(tweak_x, tweak_y, "tweak_key")
print()

# Pipeline
print("=== BIP-352 Pipeline ===\n")

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

# Step 5: Add spend_public_key to output_point
print("Step 5: Add spend_public_key to output_point")
final_point = point_add(output_point, (spend_x, spend_y))
print(f"  final_point.x = {final_point[0]:064x}")
print(f"  final_point.y = {final_point[1]:064x}")
print()

# Step 6: Extract 64-bit values
print("Step 6: Extract 64-bit values from final_point.x")
print(f"  Full x-coordinate: {final_point[0]:064x}")

# Lower 64 bits (least significant)
lower_64_bits = final_point[0] & 0xFFFFFFFFFFFFFFFF
print(f"  Lower 64 bits (hex): {lower_64_bits:016x}")
print(f"  Lower 64 bits (unsigned): {lower_64_bits}")

# Upper 64 bits (most significant)
upper_64_bits = (final_point[0] >> 192) & 0xFFFFFFFFFFFFFFFF
print(f"  Upper 64 bits (hex): {upper_64_bits:016x}")
print(f"  Upper 64 bits (unsigned): {upper_64_bits}")
print()

print("=== Comparison ===")
print(f"Expected output: {expected_output} (hex: {expected_output:016x})")
print(f"Computed lower 64 bits: {lower_64_bits} → Match: {'✓ YES' if lower_64_bits == expected_output else '✗ NO'}")
print(f"Computed upper 64 bits: {upper_64_bits} → Match: {'✓ YES' if upper_64_bits == expected_output else '✗ NO'}")

#!/usr/bin/env python3
"""
Verify BIP-352 test vector using EXACT blob values from the test file.
"""

import hashlib
import sys
import os

# Add gECC scripts to path
script_dir = os.path.dirname(os.path.abspath(__file__))
gecc_scripts_dir = os.path.join(script_dir, '..', 'gECC', 'scripts')
sys.path.insert(0, gecc_scripts_dir)

import field
from constants import SECP256K1_q, SECP256K1_n, SECP256K1_g1_generator

p = SECP256K1_q
n = SECP256K1_n
Gx, Gy = SECP256K1_g1_generator

def point_add(P, Q):
    if P is None: return Q
    if Q is None: return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2:
        if y1 == y2: return point_double(P)
        else: return None
    s = (y2 - y1) * pow(x2 - x1, -1, p) % p
    x3 = (s * s - x1 - x2) % p
    y3 = (s * (x1 - x3) - y1) % p
    return (x3, y3)

def point_double(P):
    if P is None: return None
    x, y = P
    s = (3 * x * x) * pow(2 * y, -1, p) % p
    x3 = (s * s - 2 * x) % p
    y3 = (s * (x - x3) - y) % p
    return (x3, y3)

def point_multiply(k, P):
    if k == 0: return None
    result = None
    addend = P
    while k:
        if k & 1: result = point_add(result, addend)
        addend = point_double(addend)
        k >>= 1
    return result

def compress_point(x, y):
    prefix = 0x02 if (y & 1) == 0 else 0x03
    return bytes([prefix]) + x.to_bytes(32, 'big')

def tagged_hash(tag, msg):
    tag_hash = hashlib.sha256(tag).digest()
    return hashlib.sha256(tag_hash + tag_hash + msg).digest()

print("=== BIP-352 Test Vector from Test File ===\n")

# EXACT blob values from test file
scan_private_key_blob = bytes.fromhex("2c1f0cb94db3946522cc1487256535dd33a1f911946baff817a72880064e690f")
spend_public_key_blob = bytes.fromhex("36cf8fcd4d4890ab6c1083aeb5b50c260c20acda7839120e3575836f6d85c95ce0d705e31ff9fdcce67a8f3598871c6dfbe6bcde8a51cb7b48b0f95be0ea94de")
tweak_key_blob = bytes.fromhex("040096db612390ee6cef521e784c897c446a26cea8e28819962e5316c253c24a501e53f71071162afab559954064f0ccb7a6779c23b305597b6335829cc1f5b7")

print(f"scan_private_key_blob (32 bytes, little-endian): {scan_private_key_blob.hex()}")
print(f"spend_public_key_blob (64 bytes, little-endian x||y): {spend_public_key_blob.hex()}")
print(f"tweak_key_blob (64 bytes, little-endian x||y): {tweak_key_blob.hex()}")
print()

# Parse as little-endian
scan_private_key = int.from_bytes(scan_private_key_blob, 'little')
spend_x = int.from_bytes(spend_public_key_blob[:32], 'little')
spend_y = int.from_bytes(spend_public_key_blob[32:], 'little')
tweak_x = int.from_bytes(tweak_key_blob[:32], 'little')
tweak_y = int.from_bytes(tweak_key_blob[32:], 'little')

print(f"Parsed (little-endian):")
print(f"  scan_private_key: {scan_private_key:064x}")
print(f"  spend_public_key.x: {spend_x:064x}")
print(f"  spend_public_key.y: {spend_y:064x}")
print(f"  tweak_key.x: {tweak_x:064x}")
print(f"  tweak_key.y: {tweak_y:064x}")
print()

# Verify points on curve
def verify_point(x, y, name):
    lhs = (y * y) % p
    rhs = (x * x * x + 7) % p
    if lhs == rhs:
        print(f"✓ {name} is on curve")
        return True
    else:
        print(f"✗ {name} NOT on curve!")
        return False

verify_point(spend_x, spend_y, "spend_public_key")
verify_point(tweak_x, tweak_y, "tweak_key")
print()

# Pipeline
print("=== Pipeline ===\n")

print("Step 1: shared_secret = tweak_key * scan_private_key")
shared_secret = point_multiply(scan_private_key, (tweak_x, tweak_y))
print(f"  x = {shared_secret[0]:064x}")
print(f"  y = {shared_secret[1]:064x}")
print()

print("Step 2: Serialize + 4 zeros")
compressed = compress_point(shared_secret[0], shared_secret[1])
serialized = compressed + b'\x00\x00\x00\x00'
print(f"  serialized = {serialized.hex()}")
print()

print("Step 3: Tagged hash")
tag = b"BIP0352/SharedSecret"
hash_result = tagged_hash(tag, serialized)
hash_int = int.from_bytes(hash_result, 'big')
print(f"  hash = {hash_result.hex()}")
print()

print("Step 4: output_point = hash × G")
output_point = point_multiply(hash_int, (Gx, Gy))
print(f"  x = {output_point[0]:064x}")
print(f"  y = {output_point[1]:064x}")
print()

print("Step 5: final_point = output_point + spend_public_key")
final_point = point_add(output_point, (spend_x, spend_y))
print(f"  x = {final_point[0]:064x}")
print(f"  y = {final_point[1]:064x}")
print()

print("Step 6: Extract upper 64 bits")
upper_64 = (final_point[0] >> 192) & 0xFFFFFFFFFFFFFFFF
print(f"  Upper 64 bits: {upper_64:016x} = {upper_64}")
print()

print(f"Expected in test: 4512552348537027144")
print(f"Match: {'✓ YES' if upper_64 == 4512552348537027144 else '✗ NO'}")

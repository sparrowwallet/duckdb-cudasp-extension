#!/usr/bin/env python3
"""
Decompress the tweak_key from compressed SEC1 format to uncompressed little-endian x||y format.
"""

import sys
import os

# Add gECC scripts to path
script_dir = os.path.dirname(os.path.abspath(__file__))
gecc_scripts_dir = os.path.join(script_dir, '..', 'gECC', 'scripts')
sys.path.insert(0, gecc_scripts_dir)

from constants import SECP256K1_q

p = SECP256K1_q

# Compressed tweak_key
compressed_hex = "024ac253c216532e961988e2a8ce266a447c894c781e52ef6cee902361db960004"

print("=== Decompressing tweak_key ===\n")
print(f"Compressed form: {compressed_hex}")

# Parse prefix and x-coordinate
prefix = int(compressed_hex[0:2], 16)
x_hex = compressed_hex[2:]
x = int(x_hex, 16)

print(f"Prefix: 0x{prefix:02x} ({'even y' if prefix == 0x02 else 'odd y'})")
print(f"X (big-endian): {x:064x}")

# Decompress: compute y from x
# y^2 = x^3 + 7 (mod p)
y_squared = (pow(x, 3, p) + 7) % p

# Compute square root using Tonelli-Shanks (or use pow for p ≡ 3 mod 4)
# For secp256k1, p ≡ 3 mod 4, so we can use: y = y_squared^((p+1)/4) mod p
y = pow(y_squared, (p + 1) // 4, p)

# Choose the correct y based on parity
if (y % 2) == (prefix % 2):
    # y has correct parity
    pass
else:
    # Negate y
    y = p - y

print(f"Y (big-endian): {y:064x}")

# Verify the point is on the curve
y_squared_check = (y * y) % p
x_cubed_plus_7 = (pow(x, 3, p) + 7) % p
if y_squared_check == x_cubed_plus_7:
    print("✓ Point is on the curve")
else:
    print("✗ ERROR: Point is NOT on the curve!")
    sys.exit(1)

print()

# Convert to little-endian format for our test
x_le_bytes = x.to_bytes(32, 'big')[::-1]  # Reverse to little-endian
y_le_bytes = y.to_bytes(32, 'big')[::-1]  # Reverse to little-endian
uncompressed_le = x_le_bytes + y_le_bytes

print("=== Uncompressed little-endian x||y format ===")
print(f"X (little-endian): {x_le_bytes.hex()}")
print(f"Y (little-endian): {y_le_bytes.hex()}")
print()
print(f"Full uncompressed (64 bytes, little-endian x||y):")
print(f"  {uncompressed_le.hex()}")
print()
print(f"BLOB format:")
print(f"  BLOB '\\x{uncompressed_le.hex()}'")

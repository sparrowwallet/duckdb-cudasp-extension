#!/usr/bin/env python3
"""
Prepare test data for label checking test (test 2 - with different label key).
"""

import sys
import os

# Add gECC scripts to path
script_dir = os.path.dirname(os.path.abspath(__file__))
gecc_scripts_dir = os.path.join(script_dir, '..', 'gECC', 'scripts')
sys.path.insert(0, gecc_scripts_dir)

from constants import SECP256K1_q

p = SECP256K1_q

def decompress_point(compressed_hex):
    """Decompress a compressed SEC1 point to uncompressed little-endian x||y format."""
    # Parse prefix and x-coordinate
    prefix = int(compressed_hex[0:2], 16)
    x_hex = compressed_hex[2:]
    x = int(x_hex, 16)

    print(f"  Compressed: {compressed_hex}")
    print(f"  Prefix: 0x{prefix:02x} ({'even y' if prefix == 0x02 else 'odd y'})")
    print(f"  X (big-endian): {x:064x}")

    # Decompress: compute y from x
    # y^2 = x^3 + 7 (mod p)
    y_squared = (pow(x, 3, p) + 7) % p

    # Compute square root (p ≡ 3 mod 4 for secp256k1)
    y = pow(y_squared, (p + 1) // 4, p)

    # Choose the correct y based on parity
    if (y % 2) == (prefix % 2):
        # y has correct parity
        pass
    else:
        # Negate y
        y = p - y

    print(f"  Y (big-endian): {y:064x}")

    # Verify the point is on the curve
    y_squared_check = (y * y) % p
    x_cubed_plus_7 = (pow(x, 3, p) + 7) % p
    if y_squared_check == x_cubed_plus_7:
        print(f"  ✓ Point is on the curve")
    else:
        print(f"  ✗ ERROR: Point is NOT on the curve!")
        sys.exit(1)

    # Convert to little-endian format
    x_le_bytes = x.to_bytes(32, 'big')[::-1]  # Reverse to little-endian
    y_le_bytes = y.to_bytes(32, 'big')[::-1]  # Reverse to little-endian
    uncompressed_le = x_le_bytes + y_le_bytes

    print(f"  Uncompressed (64 bytes, little-endian x||y): {uncompressed_le.hex()}")
    print()

    return uncompressed_le

print("=== Label Checking Test Data Preparation (Test 2) ===\n")

# Input values
scan_private_key_hex = "0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c"
spend_public_key_compressed = "025cc9856d6f8375350e123978daac200c260cb5b5ae83106cab90484dcd8fcf36"
tweak_key_compressed = "0314bec14463d6c0181083d607fecfba67bb83f95915f6f247975ec566d5642ee8"
label_key_compressed = "034e52d154b56ffe17964bd72e1dc4478c956f3fa29e1ea7e8bdee2d2a21f963cd"
output_value = -1006811617310360495

print(f"Input values:")
print(f"  scan_private_key: {scan_private_key_hex}")
print(f"  spend_public_key: {spend_public_key_compressed}")
print(f"  tweak_key: {tweak_key_compressed}")
print(f"  label_key: {label_key_compressed}")
print(f"  output: {output_value}")
print()

# Convert scan_private_key to little-endian BLOB
scan_private_key_bytes = bytes.fromhex(scan_private_key_hex)
# Reverse to little-endian
scan_private_key_le = scan_private_key_bytes[::-1]
print(f"scan_private_key (little-endian): {scan_private_key_le.hex()}")
print()

# Decompress spend_public_key
print("Decompressing spend_public_key:")
spend_public_key_le = decompress_point(spend_public_key_compressed)

# Decompress tweak_key
print("Decompressing tweak_key:")
tweak_key_le = decompress_point(tweak_key_compressed)

# Decompress label_key
print("Decompressing label_key:")
label_key_le = decompress_point(label_key_compressed)

# Generate SQL BLOB literals
print("=== SQL BLOB Literals ===\n")

def to_blob_literal(data):
    return "BLOB '" + ''.join(f'\\x{b:02x}' for b in data) + "'"

print(f"scan_private_key BLOB:")
print(f"  {to_blob_literal(scan_private_key_le)}")
print()

print(f"spend_public_key BLOB:")
print(f"  {to_blob_literal(spend_public_key_le)}")
print()

print(f"tweak_key BLOB:")
print(f"  {to_blob_literal(tweak_key_le)}")
print()

print(f"label_key BLOB:")
print(f"  {to_blob_literal(label_key_le)}")
print()

print(f"output value: {output_value}")
print()

print("=== Test Case SQL ===\n")
print(f"INSERT INTO test_data VALUES")
print(f"    (BLOB '\\x00\\x01\\x02\\x06', 400, {to_blob_literal(tweak_key_le)}, [{output_value}]);")
print()
print(f"# Test with different label key - should match using label")
print(f"query I")
print(f"SELECT COUNT(*) FROM cudasp_scan((SELECT txid, height, tweak_key, outputs FROM test_data WHERE height = 400), {to_blob_literal(scan_private_key_le)}, {to_blob_literal(spend_public_key_le)}, [{to_blob_literal(label_key_le)}]);")
print(f"----")
print(f"1")

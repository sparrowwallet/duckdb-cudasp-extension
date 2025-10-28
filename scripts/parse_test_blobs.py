#!/usr/bin/env python3
"""
Parse the actual BLOB values from the test file to see what they represent.
"""

import sys

# From test file line 110 (spend_public_key BLOB)
spend_blob_hex = "36cf8fcd4d4890ab6c1083aeb5b50c260c20acda78391230e3575836f6d85c95ce0d705e31ff9fdcce67a8f3598871c6dfbe6bcde8a51cb7b48b0f95be0ea94de"

print("=== Parsing Test BLOBs ===\n")

print("spend_public_key BLOB (from test file line 110):")
print(f"  Hex: {spend_blob_hex}")
print(f"  Length: {len(spend_blob_hex) // 2} bytes")

# Parse as little-endian x||y (32 bytes each)
spend_x_le_hex = spend_blob_hex[:64]  # First 32 bytes
spend_y_le_hex = spend_blob_hex[64:128]  # Next 32 bytes

print(f"\n  X (little-endian): {spend_x_le_hex}")
print(f"  Y (little-endian): {spend_y_le_hex}")

# Convert to big-endian
spend_x_be_hex = ''.join(reversed([spend_x_le_hex[i:i+2] for i in range(0, len(spend_x_le_hex), 2)]))
spend_y_be_hex = ''.join(reversed([spend_y_le_hex[i:i+2] for i in range(0, len(spend_y_le_hex), 2)]))

print(f"\n  X (big-endian): {spend_x_be_hex}")
print(f"  Y (big-endian): {spend_y_be_hex}")

# Verify this matches the compressed form: 025cc9856d6f8375350e123978daac200c260cb5b5ae83106cab90484dcd8fcf36
expected_x = "5cc9856d6f8375350e123978daac200c260cb5b5ae83106cab90484dcd8fcf36"
print(f"\n  Expected X from compressed: {expected_x}")
print(f"  Match: {spend_x_be_hex == expected_x}")

# Now check the scan_private_key from test file line 98
scan_priv_blob_hex = "2c1f0cb94db3946522cc1487256535dd33a1f919946baffb17a728800646e690f"

print("\n\nscan_private_key BLOB (from test file line 98):")
print(f"  Hex: {scan_priv_blob_hex}")
print(f"  Length: {len(scan_priv_blob_hex) // 2} bytes")

# Convert to big-endian
scan_priv_be_hex = ''.join(reversed([scan_priv_blob_hex[i:i+2] for i in range(0, len(scan_priv_blob_hex), 2)]))
print(f"\n  Big-endian: {scan_priv_be_hex}")

# Expected: 0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c
expected_scan = "0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c"
print(f"  Expected: {expected_scan}")
print(f"  Match: {scan_priv_be_hex == expected_scan}")

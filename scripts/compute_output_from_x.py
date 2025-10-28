#!/usr/bin/env python3
"""
Compute the output value (upper 64 bits as int64_t) from an x-coordinate.
Usage: ./compute_output_from_x.py <x_coordinate_hex>
"""

import sys

if len(sys.argv) != 2:
    print("Usage: ./compute_output_from_x.py <x_coordinate_hex>")
    print("Example: ./compute_output_from_x.py be368e28979d950245d742891ae6064020ba548c1e2e65a639a8bb0675d95cff")
    sys.exit(1)

# X-coordinate in hex (big-endian)
x_hex = sys.argv[1].strip().lower()

# Remove '0x' prefix if present
if x_hex.startswith('0x'):
    x_hex = x_hex[2:]

print(f"X-coordinate (hex, big-endian): {x_hex}")
print()

# Convert to integer
x_int = int(x_hex, 16)
print(f"X-coordinate (integer): {x_int}")
print()

# Extract upper 64 bits (most significant 8 bytes)
# In big-endian hex representation, the first 16 hex chars are the upper 64 bits
upper_64_hex = x_hex[:16]
print(f"Upper 64 bits (hex): {upper_64_hex}")

# Convert to unsigned 64-bit integer
upper_64_unsigned = int(upper_64_hex, 16)
print(f"Upper 64 bits (unsigned): {upper_64_unsigned}")

# Convert to signed 64-bit integer (int64_t)
# If MSB is set, it's negative in two's complement
if upper_64_unsigned >= 2**63:
    upper_64_signed = upper_64_unsigned - 2**64
else:
    upper_64_signed = upper_64_unsigned

print(f"Upper 64 bits (signed int64_t): {upper_64_signed}")
print()

# Verify by converting back
print("Verification:")
if upper_64_signed < 0:
    back_to_unsigned = (2**64 + upper_64_signed)
else:
    back_to_unsigned = upper_64_signed
print(f"  Back to unsigned: {back_to_unsigned} (hex: {back_to_unsigned:016x})")
print(f"  Matches original: {back_to_unsigned == upper_64_unsigned}")

#!/usr/bin/env python3
"""
Decode the actual data from height=400 INSERT statement.
"""

# tweak_key from INSERT statement (line 94)
tweak = b'\xe8\x2e\x64\xd5\x66\xc5\x5e\x97\x47\xf2\xf6\x15\x59\xf9\x83\xbb\x67\xba\xcf\xfe\x07\xd6\x83\x10\x18\xc0\xd6\x63\x44\xc1\xbe\x14\xc3\x80\x32\xa4\x8f\x5b\x3c\x56\xb5\xb6\x28\x6a\x06\xc0\x27\x08\x46\xb7\xb8\x52\xcd\x31\x8d\x9a\x13\x71\x73\xa5\xb4\x1c\x2f\x84'

output_value = -1006811617310360495

print("=== Height 400 Data ===\n")

print(f"tweak_key ({len(tweak)} bytes):")
print(f"  Little-endian hex: {tweak.hex()}")

# Parse X and Y (32 bytes each)
tweak_x_le = tweak[:32]
tweak_y_le = tweak[32:64]
tweak_x_be = tweak_x_le[::-1]
tweak_y_be = tweak_y_le[::-1]

print(f"  X (big-endian): {tweak_x_be.hex()}")
print(f"  Y (big-endian): {tweak_y_be.hex()}")
print(f"  Expected X: 14bec14463d6c0181083d607fecfba67bb83f95915f6f247975ec566d5642ee8")
print(f"  Match: {tweak_x_be.hex() == '14bec14463d6c0181083d607fecfba67bb83f95915f6f247975ec566d5642ee8'}")
print()

print(f"output_value: {output_value}")
print(f"  As unsigned (if it were): {output_value if output_value >= 0 else (2**64 + output_value)}")
print(f"  As hex: {(output_value if output_value >= 0 else (2**64 + output_value)):016x}")

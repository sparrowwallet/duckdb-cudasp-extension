#!/usr/bin/env python3
"""
Convert BIP-352 test vectors to DuckDB BLOB format.
- scan_private_key: big-endian -> reverse to little-endian
- spend_public_key: already little-endian x||y
- tweak_key: already little-endian x||y (strip 04 prefix if present)
"""

# Input values from BIP-352 test vectors
scan_private_key_hex = "0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c"
spend_public_key_hex = "36cf8fcd4d4890ab6c1083aeb5b50c260c20acda7839120e3575836f6d85c95ce0d705e31ff9fdcce67a8f3598871c6dfbe6bcde8a51cb7b48b0f95be0ea94de"
tweak_key_hex = "040096db612390ee6cef521e784c897c446a26cea8e28819962e5316c253c24a501e53f71071162afab559954064f0ccb7a6779c23b305597b6335829cc1f5b7"
expected_output = 4512552348537027144

print("=== BIP-352 Test Vector Conversion ===\n")

# Convert scan_private_key (32 bytes, big-endian hex -> little-endian BLOB)
scan_key_bytes = bytes.fromhex(scan_private_key_hex)
scan_key_le = scan_key_bytes[::-1]  # Reverse for little-endian
print(f"scan_private_key (32 bytes, big-endian -> little-endian):")
print(f"  BLOB '\\x{scan_key_le.hex()}'")
print()

# spend_public_key is already little-endian x||y (64 bytes)
spend_pubkey_bytes = bytes.fromhex(spend_public_key_hex)
print(f"spend_public_key (64 bytes, already little-endian x||y):")
print(f"  BLOB '\\x{spend_pubkey_bytes.hex()}'")
print()

# tweak_key: strip 04 prefix if present, already little-endian x||y (64 bytes)
if tweak_key_hex.startswith("04"):
    tweak_key_hex = tweak_key_hex[2:]  # Strip uncompressed point prefix
tweak_key_bytes = bytes.fromhex(tweak_key_hex)
print(f"tweak_key (64 bytes, already little-endian x||y, stripped 04 prefix):")
print(f"  BLOB '\\x{tweak_key_bytes.hex()}'")
print()

print(f"expected_output: {expected_output}\n")

print("=== SQL INSERT statement ===")
print(f"INSERT INTO test_data VALUES")
print(f"    (BLOB '\\x00\\x01\\x02\\x04', 200, BLOB '\\x{tweak_key_bytes.hex()}', [{expected_output}, 99999]);")

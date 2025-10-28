#!/usr/bin/env python3
"""
Decode the BLOB escape sequences from the test file.
"""

# scan_private_key from test file (line 98, 104, 110)
scan_priv = b'\x2c\x1f\x0c\xb9\x4d\xb3\x94\x65\x22\xcc\x14\x87\x25\x65\x35\xdd\x33\xa1\xf9\x11\x94\x6b\xaf\xf8\x17\xa7\x28\x80\x06\x4e\x69\x0f'

# spend_public_key from test file (line 98, 104, 110)
spend_pub = b'\x36\xcf\x8f\xcd\x4d\x48\x90\xab\x6c\x10\x83\xae\xb5\xb5\x0c\x26\x0c\x20\xac\xda\x78\x39\x12\x0e\x35\x75\x83\x6f\x6d\x85\xc9\x5c\xe0\xd7\x05\xe3\x1f\xf9\xfd\xcc\xe6\x7a\x8f\x35\x98\x87\x1c\x6d\xfb\xe6\xbc\xde\x8a\x51\xcb\x7b\x48\xb0\xf9\x5b\xe0\xea\x94\xde'

# label_key from test file (line 98)
label = b'\xcd\x63\xf9\x21\x2a\x2d\xee\xbd\xe8\xa7\x1e\x9e\xa2\x3f\x6f\x95\x8c\x47\xc4\x1d\x2e\xd7\x4b\x96\x17\xfe\x6f\xb5\x54\xd1\x52\x4e\x29\x2f\xab\xdd\xbd\xcb\xb6\x43\xea\xfc\x32\x88\x75\xc4\x6d\x75\xa1\xd6\x97\xb2\xb3\x1c\x42\xd3\x8a\xa9\x3f\x85\xea\xb3\x4b\xc1'

print("=== Decoding Test BLOBs ===\n")

print(f"scan_private_key ({len(scan_priv)} bytes):")
print(f"  Little-endian hex: {scan_priv.hex()}")
# Convert to big-endian
scan_priv_be = scan_priv[::-1]
print(f"  Big-endian hex: {scan_priv_be.hex()}")
print(f"  Expected: 0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c")
print(f"  Match: {scan_priv_be.hex() == '0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c'}")
print()

print(f"spend_public_key ({len(spend_pub)} bytes):")
print(f"  Little-endian hex: {spend_pub.hex()}")
# Parse X and Y (32 bytes each)
spend_x_le = spend_pub[:32]
spend_y_le = spend_pub[32:64]
spend_x_be = spend_x_le[::-1]
spend_y_be = spend_y_le[::-1]
print(f"  X (big-endian): {spend_x_be.hex()}")
print(f"  Y (big-endian): {spend_y_be.hex()}")
print(f"  Expected X: 5cc9856d6f8375350e123978daac200c260cb5b5ae83106cab90484dcd8fcf36")
print(f"  Match: {spend_x_be.hex() == '5cc9856d6f8375350e123978daac200c260cb5b5ae83106cab90484dcd8fcf36'}")
print()

print(f"label_key ({len(label)} bytes):")
print(f"  Little-endian hex: {label.hex()}")
# Parse X and Y (32 bytes each)
label_x_le = label[:32]
label_y_le = label[32:64]
label_x_be = label_x_le[::-1]
label_y_be = label_y_le[::-1]
print(f"  X (big-endian): {label_x_be.hex()}")
print(f"  Y (big-endian): {label_y_be.hex()}")
print(f"  Expected X: 4e52d154b56ffe17964bd72e1dc4478c956f3fa29e1ea7e8bdee2d2a21f963cd")
print(f"  Match: {label_x_be.hex() == '4e52d154b56ffe17964bd72e1dc4478c956f3fa29e1ea7e8bdee2d2a21f963cd'}")

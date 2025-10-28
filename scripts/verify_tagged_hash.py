#!/usr/bin/env python3
"""
Verify the tagged hash calculation step by step.
"""

import hashlib

# Input from GPU debug output
serialized_hex = "02fdccb55f22a9a4806c7f20e1493296c4872ef996f898c9ee2b8fcab3265db42300000000"
serialized = bytes.fromhex(serialized_hex)

print(f"Serialized input: {serialized_hex}")
print(f"Serialized length: {len(serialized)} bytes\n")

# Tag
tag = b"BIP0352/SharedSecret"
print(f"Tag: {tag.decode()}")
print(f"Tag length: {len(tag)} bytes")
print(f"Tag bytes: {tag.hex()}\n")

# Step 1: Compute SHA256(tag)
tag_hash = hashlib.sha256(tag).digest()
print(f"SHA256(tag): {tag_hash.hex()}\n")

# Step 2: Compute SHA256(tag_hash || tag_hash || serialized)
message = tag_hash + tag_hash + serialized
print(f"Message for final hash: {message.hex()}")
print(f"Message length: {len(message)} bytes\n")

# Step 3: Compute final hash
final_hash = hashlib.sha256(message).digest()
print(f"Final tagged hash: {final_hash.hex()}")
print(f"\nExpected: f426e65257691bfc75b5e09f9aec83557544f1aa766bb48f3424dc5c040e0e15")
print(f"Match: {final_hash.hex() == 'f426e65257691bfc75b5e09f9aec83557544f1aa766bb48f3424dc5c040e0e15'}")

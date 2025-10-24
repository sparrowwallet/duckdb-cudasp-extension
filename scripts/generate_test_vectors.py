#!/usr/bin/env python3
"""
Generate test vectors for cudasp_scan testing.
Computes EC multiplication on secp256k1 and extracts int64 values.
"""

import hashlib
import secrets

def int_to_bytes_le(n, length):
    """Convert integer to little-endian bytes"""
    return n.to_bytes(length, byteorder='little')

def bytes_le_to_int(b):
    """Convert little-endian bytes to integer"""
    return int.from_bytes(b, byteorder='little')

# secp256k1 parameters
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
A = 0
B = 7

def modinv(a, m):
    """Modular inverse using extended Euclidean algorithm"""
    if a < 0:
        a = (a % m + m) % m
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise Exception('Modular inverse does not exist')
    return x % m

def extended_gcd(a, b):
    """Extended Euclidean algorithm"""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def point_add(p1, p2):
    """Add two points on secp256k1"""
    if p1 is None:
        return p2
    if p2 is None:
        return p1

    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2:
        if y1 == y2:
            # Point doubling
            s = (3 * x1 * x1 * modinv(2 * y1, P)) % P
        else:
            # Points are inverses
            return None
    else:
        # Point addition
        s = ((y2 - y1) * modinv(x2 - x1, P)) % P

    x3 = (s * s - x1 - x2) % P
    y3 = (s * (x1 - x3) - y1) % P

    return (x3, y3)

def point_multiply(k, point):
    """Multiply point by scalar k using double-and-add"""
    if k == 0:
        return None
    if k == 1:
        return point

    result = None
    addend = point

    while k:
        if k & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        k >>= 1

    return result

def extract_int64(point):
    """Extract lower 64 bits from x-coordinate of point"""
    if point is None:
        return 0
    x, _ = point
    # Extract lower 64 bits
    return x & 0xFFFFFFFFFFFFFFFF

def generate_test_vector(scan_key, tweak_point, outputs_values):
    """
    Generate a test vector.

    Args:
        scan_key: 32-byte scalar (int)
        tweak_point: (x, y) tuple for EC point
        outputs_values: list of int64 values to include in outputs

    Returns:
        dict with test data
    """
    # Compute result = scan_key * tweak_point
    result_point = point_multiply(scan_key, tweak_point)

    # Extract int64 from result
    computed_int64 = extract_int64(result_point)

    # Convert to bytes for DuckDB
    scan_key_bytes = int_to_bytes_le(scan_key, 32)
    tweak_x_bytes = int_to_bytes_le(tweak_point[0], 32)
    tweak_y_bytes = int_to_bytes_le(tweak_point[1], 32)
    tweak_key_bytes = tweak_x_bytes + tweak_y_bytes

    # Check if computed value should match (is in outputs)
    should_match = computed_int64 in outputs_values

    return {
        'scan_key_hex': scan_key_bytes.hex(),
        'tweak_key_hex': tweak_key_bytes.hex(),
        'tweak_x': tweak_point[0],
        'tweak_y': tweak_point[1],
        'outputs': outputs_values,
        'computed_int64': computed_int64,
        'should_match': should_match,
        'result_point': result_point
    }

def main():
    print("Generating test vectors for cudasp_scan...")
    print()

    # Fixed scan key for all tests
    scan_key = 0xa0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebf

    # Test 1: Point that should match (computed value in outputs)
    # Use G as tweak point for simplicity
    tweak1 = (Gx, Gy)
    vec1 = generate_test_vector(scan_key, tweak1, [])

    # Add the computed value to outputs so it matches
    vec1_match = generate_test_vector(scan_key, tweak1, [vec1['computed_int64'], 67890])

    print("Test Vector 1 (should match):")
    print(f"  scan_key: {vec1_match['scan_key_hex']}")
    print(f"  tweak_key: {vec1_match['tweak_key_hex']}")
    print(f"  outputs: {vec1_match['outputs']}")
    print(f"  computed_int64: {vec1_match['computed_int64']}")
    print(f"  should_match: {vec1_match['should_match']}")
    print()

    # Test 2: Point that should NOT match
    # Use 2*G as tweak point
    tweak2 = point_multiply(2, (Gx, Gy))
    vec2 = generate_test_vector(scan_key, tweak2, [11111, 22222])

    print("Test Vector 2 (should NOT match):")
    print(f"  scan_key: {vec2['scan_key_hex']}")
    print(f"  tweak_key: {vec2['tweak_key_hex']}")
    print(f"  outputs: {vec2['outputs']}")
    print(f"  computed_int64: {vec2['computed_int64']}")
    print(f"  should_match: {vec2['should_match']}")
    print()

    # Generate SQL for test file
    print("=" * 80)
    print("SQL for test file:")
    print("=" * 80)
    print()

    print("# Test with real secp256k1 points")
    print("statement ok")
    print("CREATE TABLE real_test_data(")
    print("    txid BLOB,")
    print("    height INTEGER,")
    print("    tweak_key BLOB,")
    print("    outputs BIGINT[]")
    print(");")
    print()

    print("statement ok")
    print("INSERT INTO real_test_data VALUES")

    # Vector 1 (matches)
    txid1 = "00010203"
    outputs1_sql = "[" + ", ".join(str(x) for x in vec1_match['outputs']) + "]"
    print(f"    (BLOB '\\x{txid1}', 100, BLOB '\\x{vec1_match['tweak_key_hex']}', {outputs1_sql}),")

    # Vector 2 (no match)
    txid2 = "10111213"
    outputs2_sql = "[" + ", ".join(str(x) for x in vec2['outputs']) + "]"
    print(f"    (BLOB '\\x{txid2}', 101, BLOB '\\x{vec2['tweak_key_hex']}', {outputs2_sql});")
    print()

    print("# Should return only the matching row")
    print("query I")
    print(f"SELECT COUNT(*) FROM cudasp_scan((SELECT txid, height, tweak_key, outputs FROM real_test_data), BLOB '\\x{vec1_match['scan_key_hex']}');")
    print("----")
    print("1")
    print()

    print("# Verify it's the correct row")
    print("query I")
    print(f"SELECT height FROM cudasp_scan((SELECT txid, height, tweak_key, outputs FROM real_test_data), BLOB '\\x{vec1_match['scan_key_hex']}');")
    print("----")
    print("100")

if __name__ == '__main__':
    main()

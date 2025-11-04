# DuckDB CUDA Silent Payments Extension

A high-performance DuckDB extension that provides GPU-accelerated Bitcoin Silent Payments (BIP-352) scanning using NVIDIA CUDA. This extension enables efficient scanning of large transaction datasets by leveraging GPU parallel processing for elliptic curve cryptography operations.

## Features

- **GPU Acceleration**: Utilizes NVIDIA CUDA for parallel elliptic curve multiplication
- **Multi-GPU Support**: Automatically distributes workload across multiple GPUs
- **High Throughput**: Processes millions of transactions per second
- **Optimized Batching**: Configurable batch sizes for optimal GPU utilization
- **Thread-Safe**: Concurrent multi-user access supported
- **Memory Efficient**: Handles databases with 100M+ rows

## Building the Extension

### Prerequisites

- CMake 3.18 or higher
- C++ compiler with C++17 support
- NVIDIA GPU with compute capability 8.0+ (Ampere, Ada Lovelace, or Hopper)
- CUDA Toolkit 12.8 or 13.0
- Python 3 (for gECC constant generation)
- Git

**Supported GPUs:**
- NVIDIA A100 (compute capability 80)
- NVIDIA RTX 30xx series (compute capability 86)
- NVIDIA RTX 40xx/50xx series (compute capability 89)
- NVIDIA H100/H200 (compute capability 90)

### Build Steps

1. Clone the repository:
```bash
git clone --recursive https://github.com/sparrowwallet/duckdb-cudasp-extension.git
cd duckdb-cudasp-extension
```

2. Set CUDA environment variables (if necessary):
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

3. Build the extension:
```bash
make clean
make
```

4. Run tests:
```bash
make test
```

The compiled extension will be available at `build/release/extension/cudasp/cudasp.duckdb_extension`.
Running the compiled DuckDB binary at `build/release/duckdb` will run DuckDB with the extension already loaded.

### Loading the Extension

```sql
LOAD 'path/to/cudasp.duckdb_extension';
```

## Functions

### Primary Function

#### `cudasp_scan(input_table, scan_private_key, spend_public_key, label_keys, batch_size := 300000)`

Scans a table of Bitcoin transactions for Silent Payments (BIP-352) matches using GPU acceleration. This function implements the complete Silent Payments scanning algorithm with optimized elliptic curve operations.

**Parameters:**
- `input_table` (TABLE): Input table with columns:
  - `txid` (BLOB): 32-byte transaction ID
  - `height` (INTEGER): Block height
  - `tweak_key` (BLOB): 64-byte uncompressed EC point (32-byte x || 32-byte y, little-endian)
  - `outputs` (BIGINT[]): Array of output values (first 8 bytes of x-coordinates as big-endian integers)
- `scan_private_key` (BLOB): 32-byte scan private key (little-endian)
- `spend_public_key` (BLOB): 64-byte uncompressed spend public key (32-byte x || 32-byte y, little-endian)
- `label_keys` (LIST[BLOB]): Array of 64-byte uncompressed label public keys (can be empty)
- `batch_size` (INTEGER, optional): Number of rows to process per GPU batch (default: 300000)

**Returns:** TABLE with columns:
- `txid` (BLOB): Transaction ID of matching transaction
- `height` (INTEGER): Block height of matching transaction
- `tweak_key` (BLOB): Tweak key that produced the match

**Algorithm:**
1. **Batch Processing**: Groups input rows into batches for efficient GPU processing
2. **EC Multiplication**: Computes `tweak_key × scan_private_key` for each row
3. **Shared Secret**: Hashes the result using BIP-352 tagged hash (SHA256)
4. **Fixed-Point Multiplication**: Computes `shared_secret × G` using GPU-optimized fixed-point multiplication
5. **Point Addition**: Adds spend public key to create candidate output keys
6. **Label Checking**: Tests both base output and label-tweaked variants
7. **Match Detection**: Compares x-coordinates against output list
8. **Result Aggregation**: Returns all matching transactions

**Example:**
```sql
-- Create a table of transactions to scan
CREATE TABLE tweak AS
SELECT
    txid,
    height,
    tweak_key,
    outputs
FROM read_parquet('bitcoin_transactions.parquet');

-- Scan for silent payments
SELECT hex(txid), height
FROM cudasp_scan(
    (SELECT txid, height, tweak_key, outputs FROM tweak),
    from_hex('0f694e068028a717f8af6b9411f9a133dd3565258714cc226594b34db90c1f2c'),  -- scan_private_key
    from_hex('36cf8fcd4d4890ab6c1083aeb5b50c260c20acda7839120e3575836f6d85c95ce0d705e31ff9fdcce67a8f3598871c6dfbe6bcde8a51cb7b48b0f95be0ea94de'),  -- spend_public_key
    [from_hex('cd63f9212a2deebde8a71e9ea23f6f958c47c41d2ed74b9617fe6fb554d1524e292fabddbdcbb643eafc328875c46d75a1d697b2b31c42d38aa93f85eab34bc1')],  -- label_keys
    batch_size := 300000
);
```

## Performance Characteristics

### Throughput Benchmarks

Measured on dual RTX 5090 GPUs with batch_size = 300000:

| Dataset Size | Processing Time | Throughput (tx/sec) |
|--------------|-----------------|---------------------|
| 1 week (1M rows) | 575ms | 1,989,401 |
| 2 weeks (2.3M rows) | 1.04s | 2,265,266 |
| 4 weeks (5M rows) | 2.28s | 2,198,706 |
| 8 weeks (9.4M rows) | 3.64s | 2,596,475 |
| 32 weeks (32.7M rows) | 12.5s | 2,622,216 |

### Multi-GPU Scaling

- **Single GPU**: ~7.2 seconds for 1M rows
- **Dual GPU**: ~6.1 seconds for 1M rows (~1.17× speedup)
- Speedup limited by serial table scan overhead

## Multi-GPU Support

The extension automatically detects and utilizes multiple GPUs:

```sql
SELECT * FROM cudasp_scan(...);
-- Both GPUs will process batches concurrently
```

**GPU Assignment:**
- Round-robin thread assignment to GPUs
- Independent CUDA streams per thread
- Thread-safe per-device initialization

## Monitoring GPU Usage

```bash
# Real-time GPU monitoring (recommended)
nvtop

# Or use nvidia-smi
nvidia-smi -l 0.5
```

## Technical Details

### Dependencies

- [gECC](https://github.com/sparrowwallet/gECC): Fork of GPU elliptic curve cryptography library
- NVIDIA CUDA Runtime (statically linked)
- DuckDB 1.4.1

### CUDA Optimizations

- **Column-major memory layout**: Optimized for coalesced GPU memory access
- **Fixed-point multiplication**: Precomputed base point multiples
- **Batch inversion**: Efficient modular inverse using Montgomery's trick
- **Persistent L2 cache**: Pinned frequently accessed data
- **Concurrent kernel execution**: Multiple batches processed simultaneously on multi-GPU

## Error Handling

The function handles errors gracefully:
- Returns empty result set if no matches found
- Throws exception for invalid input formats
- Validates BLOB sizes (32 bytes for scalars, 64 bytes for points)
- Reports CUDA errors with detailed messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [gECC](https://github.com/CGCL-codes/gECC) for GPU elliptic curve operations
- [BIP-352](https://github.com/bitcoin/bips/blob/master/bip-0352.mediawiki) Silent Payments specification

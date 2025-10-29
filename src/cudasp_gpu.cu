#include <cuda_runtime.h>
#include <cstdint>

#include "gecc.h"
#include "gecc/arith.h"
#include "gecc/ecdsa.h"
#include "gecc/hash/sha256.h"

using namespace gecc;
using namespace arith;
using namespace ecdsa;
using namespace hash;

// Define field and EC types for secp256k1 (matching gECC test naming)
// secp256k1 has a=0, so use DBL_FLAG=1 for the a=0 optimized doubling formula
DEFINE_SECP256K1_FP(Fq_SECP256K1_1, FqSECP256K1, u32, 32, LayoutT<1>, 8, gecc::arith::MONTFLAG::SOS, gecc::arith::CURVEFLAG::DEFAULT);
DEFINE_FP(Fq_SECP256K1_n, FqSECP256K1_n, u32, 32, LayoutT<1>, 8);
DEFINE_EC(G1_1, G1SECP256K1, Fq_SECP256K1_1, SECP256K1_CURVE, 1);
DEFINE_ECDSA(ECDSA_Solver, G1_1_G1SECP256K1, Fq_SECP256K1_1, Fq_SECP256K1_n);

using Field = Fq_SECP256K1_1;
using Order = Fq_SECP256K1_n;
using ECPoint = G1_1_G1SECP256K1;  // DEFINE_EC creates TYPE_NAME_CURVE_NAME
using Solver = ECDSA_Solver;

// Device function to check if a value matches any output
__device__ bool CheckValueMatch(
    int64_t computed_value,
    const int64_t *outputs,
    uint32_t offset,
    uint32_t length) {

    for (uint32_t j = 0; j < length; ++j) {
        if (outputs[offset + j] == computed_value) {
            return true;
        }
    }
    return false;
}

// Device function to add two EC points and return X coordinate (in normal form)
__device__ Field AddPointsAndGetX(
    const Field &px_mont, const Field &py_mont,
    const Field &qx_mont, const Field &qy_mont) {

    // Create affine points
    typename ECPoint::Affine p_affine;
    p_affine.x = px_mont;
    p_affine.y = py_mont;

    typename ECPoint::Affine q_affine;
    q_affine.x = qx_mont;
    q_affine.y = qy_mont;

    // Convert to Jacobian and add
    ECPoint p_jac = p_affine.to_nonzero_jacobian();
    ECPoint result = p_jac + q_affine;

    // Convert back to affine and return X in normal form
    typename ECPoint::Affine result_affine = result.to_affine();
    return result_affine.x.from_montgomery();
}

// Device function to extract upper 64 bits from field element
__device__ int64_t ExtractUpper64(const Field &x_normal) {
    uint64_t computed_value_u64 = static_cast<uint64_t>(x_normal.digits[6]) |
                                  (static_cast<uint64_t>(x_normal.digits[7]) << 32);
    return static_cast<int64_t>(computed_value_u64);
}

// Kernel to check computed results with label checking and negation
__global__ void CheckMatchesWithLabelsKernel(
    const ECPoint::Base *fpm_results,    // FPM results (output_point before adding spend_pubkey)
    const uint32_t *spend_pubkey_x,      // Spend public key X (8 u32 limbs, normal form)
    const uint32_t *spend_pubkey_y,      // Spend public key Y (8 u32 limbs, normal form)
    const uint32_t *label_keys_x,        // Label keys X (label_count * 8 u32 limbs, normal form)
    const uint32_t *label_keys_y,        // Label keys Y (label_count * 8 u32 limbs, normal form)
    uint32_t label_count,                // Number of label keys
    const int64_t *outputs,              // Flattened outputs array
    const uint32_t *output_offsets,      // Offset into outputs for each row
    const uint32_t *output_lengths,      // Length of outputs list for each row
    uint8_t *match_flags,                // Output: 1 if match found, 0 otherwise
    uint32_t count,                      // Number of points to process
    uint64_t batch_id) {                 // Batch ID for debugging

    constexpr u32 field_limbs = 8;

    // Grid-stride loop: each thread processes multiple elements
    for (u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
         instance < count;
         instance += gridDim.x * blockDim.x) {

        // Load FPM result (output_point in Montgomery form, column-major)
        Field output_x_mont, output_y_mont;
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
            for (u32 j = 0; j < field_limbs; j++) {
                output_x_mont.digits[j] = fpm_results[j * count + instance];
                output_y_mont.digits[j] = fpm_results[(field_limbs + j) * count + instance];
            }
        #else
            for (u32 j = 0; j < field_limbs; j++) {
                output_x_mont.digits[j] = fpm_results[instance * ECPoint::Affine::LIMBS + j];
                output_y_mont.digits[j] = fpm_results[instance * ECPoint::Affine::LIMBS + field_limbs + j];
            }
        #endif

        // Load spend_public_key and convert to Montgomery form
        Field spend_x, spend_y;
        for (u32 j = 0; j < field_limbs; j++) {
            spend_x.digits[j] = spend_pubkey_x[j];
            spend_y.digits[j] = spend_pubkey_y[j];
        }
        spend_x.inplace_to_montgomery();
        spend_y.inplace_to_montgomery();

        // Get output list metadata
        uint32_t offset = output_offsets[instance];
        uint32_t length = output_lengths[instance];

        bool found_match = false;

        // Base case: output_point + spend_public_key
        Field final_x_normal = AddPointsAndGetX(output_x_mont, output_y_mont, spend_x, spend_y);

        int64_t base_value = ExtractUpper64(final_x_normal);

        if (CheckValueMatch(base_value, outputs, offset, length)) {
            found_match = true;
        }

        // Try each label key if no match yet
        if (!found_match) {
            // For label checking, we need final_point (output_point + spend_public_key) in Montgomery form
            // Recompute the full point addition to get both X and Y coordinates
            typename ECPoint::Affine output_affine;
            output_affine.x = output_x_mont;
            output_affine.y = output_y_mont;

            typename ECPoint::Affine spend_affine;
            spend_affine.x = spend_x;
            spend_affine.y = spend_y;

            ECPoint output_jac = output_affine.to_nonzero_jacobian();
            ECPoint final_jac = output_jac + spend_affine;
            typename ECPoint::Affine final_affine = final_jac.to_affine();

            // final_affine.x and final_affine.y are already in Montgomery form
            Field final_x_mont = final_affine.x;
            Field final_y_mont = final_affine.y;

            for (uint32_t label_idx = 0; label_idx < label_count; label_idx++) {
                // Load label key and convert to Montgomery form
                Field label_x, label_y;
                for (u32 j = 0; j < field_limbs; j++) {
                    label_x.digits[j] = label_keys_x[label_idx * field_limbs + j];
                    label_y.digits[j] = label_keys_y[label_idx * field_limbs + j];
                }
                label_x.inplace_to_montgomery();
                label_y.inplace_to_montgomery();

                // Compute: final_point + label_key (NOT output_point + label_key!)
                Field labeled_x_normal = AddPointsAndGetX(final_x_mont, final_y_mont, label_x, label_y);
                int64_t labeled_value = ExtractUpper64(labeled_x_normal);

                // Check value
                if (CheckValueMatch(labeled_value, outputs, offset, length)) {
                    found_match = true;
                    break;  // Exit label loop
                }
            }
        }

        // Set match flag
        match_flags[instance] = found_match ? 1 : 0;
    }  // end grid-stride loop
}

// Kernel to check computed results against outputs and set match flags
// Results are extracted from solver.R0 which has interleaved X,Y format (not Affine point format)
__global__ void CheckMatchesKernel(
    const uint32_t *result_x_coords,     // Extracted X coordinates (in Montgomery form, raw u32 limbs)
    const int64_t *outputs,              // Flattened outputs array
    const uint32_t *output_offsets,      // Offset into outputs for each row
    const uint32_t *output_lengths,      // Length of outputs list for each row
    uint8_t *match_flags,                // Output: 1 if match found, 0 otherwise
    uint32_t count) {                    // Number of points to process

    const u32 instance = blockIdx.x * blockDim.x + threadIdx.x;

    if (instance >= count) return;

    constexpr u32 field_limbs = 8;  // 8 u32 limbs for 256-bit field

    // Load X coordinate (raw u32 limbs in Montgomery form)
    // result_x_coords is a flat array: point i's X is at [i * field_limbs]
    const uint32_t *x_montgomery = result_x_coords + instance * field_limbs;

    // Convert from Montgomery form to normal form
    // Load into Field type for Montgomery conversion
    Field x_mont;
    for (u32 i = 0; i < field_limbs; ++i) {
        x_mont.digits[i] = x_montgomery[i];
    }

    Field x_normal = x_mont.from_montgomery();

    // Extract upper 64 bits (last 2 u32 limbs in little-endian storage = most significant bits)
    // For 256-bit field with 8 u32 limbs: digits[6] and digits[7] are the most significant
    uint64_t computed_value_u64 = static_cast<uint64_t>(x_normal.digits[6]) |
                                  (static_cast<uint64_t>(x_normal.digits[7]) << 32);
    int64_t computed_value = static_cast<int64_t>(computed_value_u64);

    // Get output list metadata
    uint32_t offset = output_offsets[instance];
    uint32_t length = output_lengths[instance];

    // Check if computed_value is in the outputs list for this row
    bool found = false;

    for (uint32_t j = 0; j < length; ++j) {
        if (outputs[offset + j] == computed_value) {
            found = true;
            break;
        }
    }

    // Write match flag
    match_flags[instance] = found ? 1 : 0;
}

// Kernel to serialize EC points from R0 to compressed SEC1 format + 4 zero bytes
// R0 format: Column-major with X at [j * count + i] for point i, limb j
// Output: 33-byte compressed SEC1 (prefix || X) + 4 zero bytes = 37 bytes per point
__global__ void SerializeToCompressedSEC1Kernel(
    const uint32_t *R0,             // Input: EC points in column-major format
    uint8_t *serialized,            // Output: serialized points (37 bytes each)
    uint32_t count,                 // Number of points
    uint64_t batch_id) {            // Batch ID for debugging

    constexpr u32 field_limbs = 8;  // 8 u32 limbs for 256-bit field

    // Grid-stride loop: each thread processes multiple elements
    for (u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
         instance < count;
         instance += gridDim.x * blockDim.x) {

        // Load X and Y coordinates from column-major R0 (in Montgomery form)
        Field x_mont, y_mont;
        for (u32 j = 0; j < field_limbs; ++j) {
            x_mont.digits[j] = R0[j * count + instance];                    // X coord
            y_mont.digits[j] = R0[(field_limbs + j) * count + instance];    // Y coord
        }

        // Convert from Montgomery form to normal form
        Field x_normal = x_mont.from_montgomery();
        Field y_normal = y_mont.from_montgomery();

        // Compute Y parity (even = 0x02, odd = 0x03)
        uint8_t prefix = 0x02 + (y_normal.digits[0] & 1);

        // Output pointer for this point (37 bytes)
        uint8_t *output = serialized + instance * 37;

        // Write prefix
        output[0] = prefix;

        // Write X coordinate (32 bytes, little-endian limbs to big-endian bytes)
        for (u32 i = 0; i < 8; ++i) {
            uint32_t limb = x_normal.digits[7 - i];  // Reverse limb order for big-endian
            output[1 + i * 4 + 0] = (limb >> 24) & 0xFF;
            output[1 + i * 4 + 1] = (limb >> 16) & 0xFF;
            output[1 + i * 4 + 2] = (limb >> 8) & 0xFF;
            output[1 + i * 4 + 3] = (limb >> 0) & 0xFF;
        }

        // Append 4 zero bytes
        output[33] = 0x00;
        output[34] = 0x00;
        output[35] = 0x00;
        output[36] = 0x00;
    }  // end grid-stride loop
}

// Kernel to compute BIP0352 tagged hashes on serialized EC points
// Each thread computes: SHA256(SHA256(tag) || SHA256(tag) || serialized_point)
__global__ void ComputeTaggedHashesKernel(
    const uint8_t *serialized,      // Input: serialized points (37 bytes each)
    uint8_t *hashes,                // Output: SHA256 hashes (32 bytes each)
    uint32_t count,                 // Number of points
    uint64_t batch_id) {            // Batch ID for debugging

    // Tag for BIP-352 Silent Payments
    const char tag_str[] = "BIP0352/SharedSecret";
    const uint8_t *tag = reinterpret_cast<const uint8_t*>(tag_str);
    const uint64_t tag_len = 20;  // Length of "BIP0352/SharedSecret"

    // Grid-stride loop: each thread processes multiple elements
    for (u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
         instance < count;
         instance += gridDim.x * blockDim.x) {

        // Input message: 37 bytes (33-byte compressed point + 4 zero bytes)
        const uint8_t *msg = serialized + instance * 37;
        const uint64_t msg_len = 37;

        // Output hash
        uint8_t *hash = hashes + instance * 32;

        // Compute tagged hash
        tagged_hash(tag, tag_len, msg, msg_len, hash);
    }
}

// Kernel for fixed-point multiplication: Computes hash × G for each hash
// This uses the precomputed table from ECDSACONST.d_mul_table[]
__global__ void FixedPointMultiplyKernel(
    u32 count,
    Order::Base *scalars,           // Input: scalar values (hashes converted to scalars)
    ECPoint::Base *results,         // Output: EC points (affine coordinates)
    uint64_t batch_id) {            // Batch ID for debugging

    // Grid-stride loop: each thread processes multiple elements
    for (u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
         instance < count;
         instance += gridDim.x * blockDim.x) {

        // Load scalar
        Order s;
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
            s.load_arbitrary(scalars, count, instance, 0);
        #else
            s.load(scalars + instance * Order::LIMBS, 0, 0, 0);
        #endif

        // Compute s × G using fixed-point multiplication
        // This reads from precomputed table in device constant memory
        ECPoint p = ECPoint::zero();
        Solver::fixed_point_mult(p, s, true);

        // Convert Jacobian to affine coordinates
        typename ECPoint::Affine result = p.to_affine();

        // Store result
        #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
            result.x.store_arbitrary(results, count, instance, 0);
            result.y.store_arbitrary(results + count * Field::LIMBS, count, instance, 0);
        #else
            result.x.store(results + instance * ECPoint::Affine::LIMBS, 0, 0, 0);
            result.y.store(results + instance * ECPoint::Affine::LIMBS + Field::LIMBS, 0, 0, 0);
        #endif
    }
}

// Kernel for EC point addition: Adds spend_public_key to each output point
// Input points are in affine coordinates (column-major, Montgomery form)
// Output: Updated points in place
__global__ void AddSpendPublicKeyKernel(
    u32 count,
    const uint32_t *spend_pubkey_x,  // Spend public key X (8 u32 limbs, little-endian, normal form)
    const uint32_t *spend_pubkey_y,  // Spend public key Y (8 u32 limbs, little-endian, normal form)
    ECPoint::Base *points) {         // Input/Output: EC points (affine coordinates)

    u32 instance = blockIdx.x * blockDim.x + threadIdx.x;
    if (instance >= count) return;

    constexpr u32 field_limbs = 8;

    // Load input point (in Montgomery form, column-major)
    Field px_mont, py_mont;
    #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        for (u32 j = 0; j < field_limbs; j++) {
            px_mont.digits[j] = points[j * count + instance];
            py_mont.digits[j] = points[(field_limbs + j) * count + instance];
        }
    #else
        for (u32 j = 0; j < field_limbs; j++) {
            px_mont.digits[j] = points[instance * ECPoint::Affine::LIMBS + j];
            py_mont.digits[j] = points[instance * ECPoint::Affine::LIMBS + field_limbs + j];
        }
    #endif

    // Load spend_public_key (in normal form, little-endian u32 limbs)
    Field spend_x, spend_y;
    for (u32 j = 0; j < field_limbs; j++) {
        spend_x.digits[j] = spend_pubkey_x[j];
        spend_y.digits[j] = spend_pubkey_y[j];
    }

    // Convert spend_public_key to Montgomery form
    spend_x.inplace_to_montgomery();
    spend_y.inplace_to_montgomery();

    // Create Affine point for input (already in Montgomery form)
    typename ECPoint::Affine p_affine;
    p_affine.x = px_mont;
    p_affine.y = py_mont;

    // Convert to Jacobian for addition
    ECPoint p1 = p_affine.to_nonzero_jacobian();

    // Create Affine point for spend_public_key (now in Montgomery form)
    typename ECPoint::Affine spend_affine;
    spend_affine.x = spend_x;
    spend_affine.y = spend_y;

    // Perform EC point addition using mixed addition: p1 + spend_affine
    ECPoint result = p1 + spend_affine;

    // Convert result back to affine coordinates
    typename ECPoint::Affine result_affine = result.to_affine();

    // Store result (in Montgomery form, column-major)
    #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        result_affine.x.store_arbitrary(points, count, instance, 0);
        result_affine.y.store_arbitrary(points + count * Field::LIMBS, count, instance, 0);
    #else
        result_affine.x.store(points + instance * ECPoint::Affine::LIMBS, 0, 0, 0);
        result_affine.y.store(points + instance * ECPoint::Affine::LIMBS + Field::LIMBS, 0, 0, 0);
    #endif
}

// Per-batch state to hold auxiliary data between LaunchBatchScan and RunBatchScanKernels
struct BatchScanState {
    int64_t *d_outputs;
    uint32_t *d_output_offsets;
    uint32_t *d_output_lengths;
    uint32_t *d_spend_pubkey_x;  // Device memory for spend public key X (8 u32 limbs)
    uint32_t *d_spend_pubkey_y;  // Device memory for spend public key Y (8 u32 limbs)
    uint32_t *d_label_keys_x;    // Device memory for label keys X (label_count * 8 u32 limbs)
    uint32_t *d_label_keys_y;    // Device memory for label keys Y (label_count * 8 u32 limbs)
    uint32_t *d_fpm_results_backup;  // Backup of FPM results before adding spend_pubkey (for label checking)
    uint32_t label_count;        // Number of label keys
    Solver *solver;              // ECDSA solver instance
    uint32_t count;
    uint64_t batch_id;           // Unique batch identifier for debugging
    cudaStream_t stream;         // CUDA stream for concurrent batch execution
};

// Host function to initialize solver and prepare for EC multiplication
// This follows the pattern from gECC's ec_pmul_init, but with our specific data
// Returns an opaque handle to BatchScanState (cast to void*) for thread-safe operation
extern "C" void* LaunchBatchScan(
    uint32_t **managed_points_x,      // Will allocate managed memory for input points
    uint32_t **managed_points_y,      // Will allocate managed memory for input points
    const uint32_t *h_scalar,         // Host scalar (8 u32 limbs)
    const uint32_t *h_spend_pubkey_x, // Host spend public key X (8 u32 limbs)
    const uint32_t *h_spend_pubkey_y, // Host spend public key Y (8 u32 limbs)
    const uint32_t *h_label_keys_x,   // Host label keys X (label_count * 8 u32 limbs, flattened)
    const uint32_t *h_label_keys_y,   // Host label keys Y (label_count * 8 u32 limbs, flattened)
    uint32_t label_count,             // Number of label keys
    const int64_t *h_outputs,         // Host outputs array
    const uint32_t *h_output_offsets, // Host output offsets
    const uint32_t *h_output_lengths, // Host output lengths
    uint8_t **managed_match_flags,    // Will allocate managed memory for match results
    uint32_t count,
    size_t outputs_size) {

    // Initialize field and solver once per program (not per batch)
    // Use C++11 static initialization guarantee for thread safety
    static bool initialized = []() {
        Solver::initialize();
        return true;
    }();

    // Allocate per-batch state (thread-safe)
    BatchScanState *state = new BatchScanState();
    state->d_outputs = nullptr;
    state->d_output_offsets = nullptr;
    state->d_output_lengths = nullptr;
    state->d_spend_pubkey_x = nullptr;
    state->d_spend_pubkey_y = nullptr;
    state->d_label_keys_x = nullptr;
    state->d_label_keys_y = nullptr;
    state->d_fpm_results_backup = nullptr;
    state->label_count = label_count;
    state->solver = nullptr;
    state->count = count;

    // Generate unique batch ID for debugging (use pointer address as unique ID)
    state->batch_id = reinterpret_cast<uint64_t>(state);

    // Create CUDA stream for this batch to enable concurrent execution
    cudaError_t err;
    err = cudaStreamCreate(&state->stream);
    if (err != cudaSuccess) {
        printf("cudaStreamCreate error: %s\n", cudaGetErrorString(err));
        delete state;
        return nullptr;
    }

    // Allocate managed memory for point coordinates (caller will fill these)
    err = cudaMallocManaged(managed_points_x, Field::SIZE * count);
    if (err != cudaSuccess) {
        delete state;
        return nullptr;
    }

    err = cudaMallocManaged(managed_points_y, Field::SIZE * count);
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        delete state;
        return nullptr;
    }

    // Allocate outputs metadata as DEVICE memory (not unified)
    // CRITICAL: Use cudaMalloc instead of cudaMallocManaged to avoid coherency issues
    // in concurrent batch processing when combined with cudaMemcpyHostToDevice
    err = cudaMalloc(&state->d_outputs, outputs_size * sizeof(int64_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        delete state;
        return nullptr;
    }

    err = cudaMalloc(&state->d_output_offsets, count * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(state->d_outputs);
        delete state;
        return nullptr;
    }
    // Zero out the memory to ensure no stale data
    cudaMemset(state->d_output_offsets, 0, count * sizeof(uint32_t));

    err = cudaMalloc(&state->d_output_lengths, count * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(state->d_outputs);
        cudaFree(state->d_output_offsets);
        delete state;
        return nullptr;
    }
    // Zero out the memory to ensure no stale data
    cudaMemset(state->d_output_lengths, 0, count * sizeof(uint32_t));

    err = cudaMallocManaged(managed_match_flags, count * sizeof(uint8_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(state->d_outputs);
        cudaFree(state->d_output_offsets);
        cudaFree(state->d_output_lengths);
        delete state;
        return nullptr;
    }

    // Initialize match_flags to 0
    memset(*managed_match_flags, 0, count * sizeof(uint8_t));

    // Allocate device memory for spend public key (8 u32 limbs each for x and y)
    constexpr u32 field_limbs = 8;
    err = cudaMalloc(&state->d_spend_pubkey_x, field_limbs * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(state->d_outputs);
        cudaFree(state->d_output_offsets);
        cudaFree(state->d_output_lengths);
        cudaFree(*managed_match_flags);
        delete state;
        return nullptr;
    }

    err = cudaMalloc(&state->d_spend_pubkey_y, field_limbs * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(state->d_outputs);
        cudaFree(state->d_output_offsets);
        cudaFree(state->d_output_lengths);
        cudaFree(*managed_match_flags);
        cudaFree(state->d_spend_pubkey_x);
        delete state;
        return nullptr;
    }

    // Copy outputs metadata using cudaMemcpyAsync with stream for concurrent execution
    // This allows multiple batches to copy data concurrently without blocking
    err = cudaMemcpyAsync(state->d_outputs, h_outputs, outputs_size * sizeof(int64_t), cudaMemcpyHostToDevice, state->stream);
    if (err != cudaSuccess) {
        printf("cudaMemcpyAsync d_outputs error: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpyAsync(state->d_output_offsets, h_output_offsets, count * sizeof(uint32_t), cudaMemcpyHostToDevice, state->stream);
    if (err != cudaSuccess) {
        printf("cudaMemcpyAsync d_output_offsets error: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpyAsync(state->d_output_lengths, h_output_lengths, count * sizeof(uint32_t), cudaMemcpyHostToDevice, state->stream);
    if (err != cudaSuccess) {
        printf("cudaMemcpyAsync d_output_lengths error: %s\n", cudaGetErrorString(err));
    }

    // Copy spend public key
    cudaMemcpyAsync(state->d_spend_pubkey_x, h_spend_pubkey_x, field_limbs * sizeof(uint32_t), cudaMemcpyHostToDevice, state->stream);
    cudaMemcpyAsync(state->d_spend_pubkey_y, h_spend_pubkey_y, field_limbs * sizeof(uint32_t), cudaMemcpyHostToDevice, state->stream);

    // Allocate and copy label keys (if any)
    if (label_count > 0) {
        err = cudaMalloc(&state->d_label_keys_x, label_count * field_limbs * sizeof(uint32_t));
        if (err != cudaSuccess) {
            cudaFree(*managed_points_x);
            cudaFree(*managed_points_y);
            cudaFree(state->d_outputs);
            cudaFree(state->d_output_offsets);
            cudaFree(state->d_output_lengths);
            cudaFree(*managed_match_flags);
            cudaFree(state->d_spend_pubkey_x);
            cudaFree(state->d_spend_pubkey_y);
            delete state;
            return nullptr;
        }

        err = cudaMalloc(&state->d_label_keys_y, label_count * field_limbs * sizeof(uint32_t));
        if (err != cudaSuccess) {
            cudaFree(*managed_points_x);
            cudaFree(*managed_points_y);
            cudaFree(state->d_outputs);
            cudaFree(state->d_output_offsets);
            cudaFree(state->d_output_lengths);
            cudaFree(*managed_match_flags);
            cudaFree(state->d_spend_pubkey_x);
            cudaFree(state->d_spend_pubkey_y);
            cudaFree(state->d_label_keys_x);
            delete state;
            return nullptr;
        }

        cudaMemcpyAsync(state->d_label_keys_x, h_label_keys_x, label_count * field_limbs * sizeof(uint32_t), cudaMemcpyHostToDevice, state->stream);
        cudaMemcpyAsync(state->d_label_keys_y, h_label_keys_y, label_count * field_limbs * sizeof(uint32_t), cudaMemcpyHostToDevice, state->stream);
    }

    // Create fresh solver for this batch
    state->solver = new Solver();

    return static_cast<void*>(state);
}

// Run the GPU kernels after caller has filled managed memory
// This follows the pattern from ecdsa_ec_unknown_pmul.cu test
extern "C" int RunBatchScanKernels(
    void *state_handle,              // Opaque handle from LaunchBatchScan
    uint32_t *managed_points_x,
    uint32_t *managed_points_y,
    const uint32_t *h_scalar,        // Scalar for all multiplications
    const uint32_t *h_spend_pubkey_x, // Host spend public key X (8 u32 limbs)
    const uint32_t *h_spend_pubkey_y, // Host spend public key Y (8 u32 limbs)
    const uint32_t *h_label_keys_x,   // Host label keys X (label_count * 8 u32 limbs)
    const uint32_t *h_label_keys_y,   // Host label keys Y (label_count * 8 u32 limbs)
    uint32_t label_count,             // Number of label keys
    uint8_t *managed_match_flags,
    uint32_t count) {

    BatchScanState *state = static_cast<BatchScanState*>(state_handle);
    if (!state || !state->solver) {
        printf("Error: Invalid state or solver not initialized\n");
        return -1;
    }

    Solver *solver = state->solver;

    // Prepare data in the format expected by ec_pmul_init
    // MAX_LIMBS is defined in gECC as 64 (maximum array size)
    // For secp256k1 (256-bit), we use 4 u64 limbs, but arrays must be size MAX_LIMBS
    constexpr u32 MAX_LIMBS = 64;
    constexpr u32 USED_LIMBS = 4;  // 256 bits = 4 u64 limbs

    // Allocate host arrays in the format ec_pmul_init expects
    u64 (*h_scalars)[MAX_LIMBS] = new u64[count][MAX_LIMBS];
    u64 (*h_keys_x)[MAX_LIMBS] = new u64[count][MAX_LIMBS];
    u64 (*h_keys_y)[MAX_LIMBS] = new u64[count][MAX_LIMBS];

    // Zero-initialize arrays
    memset(h_scalars, 0, count * MAX_LIMBS * sizeof(u64));
    memset(h_keys_x, 0, count * MAX_LIMBS * sizeof(u64));
    memset(h_keys_y, 0, count * MAX_LIMBS * sizeof(u64));

    // Fill scalar array (same scalar for all)
    for (u32 i = 0; i < count; i++) {
        // Convert from 8 u32 limbs to 4 u64 limbs
        // h_scalar array is in little-endian order: h_scalar[0] is LEAST significant u32
        // We need to pack sequentially: h_scalars[0] should be least significant u64
        //
        // IMPORTANT: gECC's ec_pmul_init uses reinterpret_cast<Base*>(u64_array)
        // which interprets each u64 as two u32s. We pack two sequential u32s into each u64.
        for (u32 j = 0; j < USED_LIMBS; j++) {
            // Pack two sequential u32s: low_u32 | (high_u32 << 32)
            u32 idx = j * 2;
            h_scalars[i][j] = static_cast<u64>(h_scalar[idx]) |
                              (static_cast<u64>(h_scalar[idx + 1]) << 32);
        }
    }

    // Fill point arrays from column-major format
    #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        for (u32 i = 0; i < count; i++) {
            // Extract u32 limbs for this point from column-major layout
            u32 x_limbs[8], y_limbs[8];
            for (u32 j = 0; j < 8; j++) {
                x_limbs[j] = managed_points_x[j * count + i];
                y_limbs[j] = managed_points_y[j * count + i];
            }
            // Convert to u64: Pack as low_u32 | (high_u32 << 32) for correct little-endian layout
            for (u32 j = 0; j < USED_LIMBS; j++) {
                h_keys_x[i][j] = static_cast<u64>(x_limbs[j * 2]) |
                                 (static_cast<u64>(x_limbs[j * 2 + 1]) << 32);
                h_keys_y[i][j] = static_cast<u64>(y_limbs[j * 2]) |
                                 (static_cast<u64>(y_limbs[j * 2 + 1]) << 32);
            }
        }
    #else
        for (u32 i = 0; i < count; i++) {
            // Row-major layout: Pack as low_u32 | (high_u32 << 32) for correct little-endian layout
            for (u32 j = 0; j < USED_LIMBS; j++) {
                h_keys_x[i][j] = static_cast<u64>(managed_points_x[i * 8 + j * 2]) |
                                 (static_cast<u64>(managed_points_x[i * 8 + j * 2 + 1]) << 32);
                h_keys_y[i][j] = static_cast<u64>(managed_points_y[i * 8 + j * 2]) |
                                 (static_cast<u64>(managed_points_y[i * 8 + j * 2 + 1]) << 32);
            }
        }
    #endif

    // Call ec_pmul_init with our specific data and stream
    solver->ec_pmul_init(h_scalars, h_keys_x, h_keys_y, count, state->stream);

    // Free host arrays
    delete[] h_scalars;
    delete[] h_keys_x;
    delete[] h_keys_y;

    // Check for initialization errors
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ec_pmul_init error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Run EC multiplication (matching ecdsa_ec_pmul call)
    // Use MAX_SM_NUMS blocks (like gECC tests) to ensure proper work distribution
    // The kernels use grid-stride loops, so they can handle any count with any block_num
    u32 max_thread_per_block = 256;
    u32 block_num = MAX_SM_NUMS;  // Use SM count like gECC test for optimal occupancy

    solver->ecdsa_ec_pmul(block_num, max_thread_per_block, true, state->stream);  // true = unknown points

    // Check for multiplication errors
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ecdsa_ec_pmul (first) error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // ecdsa_ec_pmul() already synchronizes internally, no need to sync again

    // === BIP-352 Silent Payment Pipeline ===
    // Step 1: Serialize shared secrets to compressed SEC1 format + 4 zero bytes (37 bytes each)
    uint8_t *d_serialized;
    err = cudaMallocManaged(&d_serialized, count * 37);
    if (err != cudaSuccess) {
        printf("cudaMallocManaged for d_serialized error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    int threads_per_block = 256;
    // Use fixed block count with grid-stride loops to handle any batch size
    // Kernels will process multiple elements per thread when count > blocks * threads
    int num_blocks = MAX_SM_NUMS;  // Use SM count for good occupancy

    SerializeToCompressedSEC1Kernel<<<num_blocks, threads_per_block, 0, state->stream>>>(
        solver->R0, d_serialized, count, state->batch_id
    );

    err = cudaStreamSynchronize(state->stream);
    if (err != cudaSuccess) {
        printf("SerializeToCompressedSEC1Kernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_serialized);
        return -1;
    }

    // Step 2: Compute BIP-352 tagged hashes (32 bytes each)
    uint8_t *d_hashes;
    err = cudaMallocManaged(&d_hashes, count * 32);
    if (err != cudaSuccess) {
        printf("cudaMallocManaged for d_hashes error: %s\n", cudaGetErrorString(err));
        cudaFree(d_serialized);
        return -1;
    }

    ComputeTaggedHashesKernel<<<num_blocks, threads_per_block, 0, state->stream>>>(
        d_serialized, d_hashes, count, state->batch_id
    );

    // Step 3: Copy hashes to host and convert to Order scalars for fixed-point multiplication
    // Copy hashes to host memory first to avoid coherency issues
    // Use async memcpy with stream, then sync the stream
    uint8_t *h_hashes = new uint8_t[count * 32];
    err = cudaMemcpyAsync(h_hashes, d_hashes, count * 32, cudaMemcpyDeviceToHost, state->stream);
    if (err != cudaSuccess) {
        printf("cudaMemcpyAsync for h_hashes error: %s\n", cudaGetErrorString(err));
        cudaFree(d_serialized);
        cudaFree(d_hashes);
        delete[] h_hashes;
        return -1;
    }

    err = cudaStreamSynchronize(state->stream);
    if (err != cudaSuccess) {
        printf("cudaStreamSynchronize after memcpy error: %s\n", cudaGetErrorString(err));
        cudaFree(d_serialized);
        cudaFree(d_hashes);
        delete[] h_hashes;
        return -1;
    }

    // Free device memory after memcpy completes
    cudaFree(d_serialized);
    cudaFree(d_hashes);

    // Allocate host buffer for conversion
    Order::Base *h_hash_scalars = new Order::Base[Order::SIZE * count];

    // Convert 32-byte hashes (big-endian) to Order::Base (u32) limbs in column-major format
    #ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
        // Column-major: limb j of scalar i is at [j * count + i]
        for (u32 i = 0; i < count; i++) {
            uint8_t *hash = h_hashes + i * 32;
            // Hash is big-endian: hash[0] is most significant byte
            // Convert to u32 limbs in little-endian order
            for (u32 j = 0; j < Order::LIMBS; j++) {
                // Each u32 limb is 4 bytes, starting from least significant
                uint32_t limb = (static_cast<uint32_t>(hash[31 - j * 4 - 0]) << 0) |
                                (static_cast<uint32_t>(hash[31 - j * 4 - 1]) << 8) |
                                (static_cast<uint32_t>(hash[31 - j * 4 - 2]) << 16) |
                                (static_cast<uint32_t>(hash[31 - j * 4 - 3]) << 24);
                h_hash_scalars[j * count + i] = limb;
            }
        }
    #else
        // Row-major: limb j of scalar i is at [i * Order::LIMBS + j]
        for (u32 i = 0; i < count; i++) {
            uint8_t *hash = h_hashes + i * 32;
            for (u32 j = 0; j < Order::LIMBS; j++) {
                uint32_t limb = (static_cast<uint32_t>(hash[31 - j * 4 - 0]) << 0) |
                                (static_cast<uint32_t>(hash[31 - j * 4 - 1]) << 8) |
                                (static_cast<uint32_t>(hash[31 - j * 4 - 2]) << 16) |
                                (static_cast<uint32_t>(hash[31 - j * 4 - 3]) << 24);
                h_hash_scalars[i * Order::LIMBS + j] = limb;
            }
        }
    #endif

    delete[] h_hashes;  // No longer needed

    // Allocate device memory and copy
    Order::Base *d_hash_scalars;
    err = cudaMalloc(&d_hash_scalars, Order::SIZE * count);
    if (err != cudaSuccess) {
        printf("cudaMalloc for d_hash_scalars error: %s\n", cudaGetErrorString(err));
        delete[] h_hash_scalars;
        return -1;
    }

    err = cudaMemcpyAsync(d_hash_scalars, h_hash_scalars, Order::SIZE * count, cudaMemcpyHostToDevice, state->stream);
    if (err != cudaSuccess) {
        printf("cudaMemcpyAsync for d_hash_scalars error: %s\n", cudaGetErrorString(err));
        delete[] h_hash_scalars;
        cudaFree(d_hash_scalars);
        return -1;
    }

    delete[] h_hash_scalars;  // Can be freed after async copy is queued

    // Step 4: Allocate output buffer for fixed-point multiply results
    ECPoint::Base *d_fpm_results;
    err = cudaMallocManaged(&d_fpm_results, ECPoint::Affine::SIZE * count);
    if (err != cudaSuccess) {
        printf("cudaMallocManaged for d_fpm_results error: %s\n", cudaGetErrorString(err));
        cudaFree(d_hash_scalars);
        return -1;
    }

    // Step 5: Fixed-point multiply: hash × G using precomputed table
    FixedPointMultiplyKernel<<<num_blocks, threads_per_block, 0, state->stream>>>(
        count, d_hash_scalars, d_fpm_results, state->batch_id
    );

    err = cudaStreamSynchronize(state->stream);
    if (err != cudaSuccess) {
        printf("FixedPointMultiplyKernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_hash_scalars);
        cudaFree(d_fpm_results);
        return -1;
    }

    cudaFree(d_hash_scalars);  // No longer needed

    // Step 6: Check matches with label support
    // This will: (1) try base case: output_point + spend_pubkey
    //           (2) for each label: try output_point + label_key (and negated)

    CheckMatchesWithLabelsKernel<<<num_blocks, threads_per_block, 0, state->stream>>>(
        d_fpm_results,
        state->d_spend_pubkey_x,
        state->d_spend_pubkey_y,
        state->d_label_keys_x,
        state->d_label_keys_y,
        state->label_count,
        state->d_outputs,
        state->d_output_offsets,
        state->d_output_lengths,
        managed_match_flags,
        count,
        state->batch_id
    );

    err = cudaStreamSynchronize(state->stream);
    if (err != cudaSuccess) {
        printf("CheckMatchesWithLabelsKernel error: %s\n", cudaGetErrorString(err));
        cudaFree(d_fpm_results);
        return -1;
    }

    cudaFree(d_fpm_results);  // No longer needed

    return 0;
}

// Cleanup function to free batch state
extern "C" void FreeBatchScanState(void *state_handle) {
    if (!state_handle) return;

    BatchScanState *state = static_cast<BatchScanState*>(state_handle);

    // CRITICAL: Synchronize stream before cleanup to ensure all operations complete
    // Without this, destroying the stream or freeing memory can cause deadlocks
    if (state->stream) {
        cudaError_t sync_err = cudaStreamSynchronize(state->stream);
        if (sync_err != cudaSuccess) {
            printf("WARNING: cudaStreamSynchronize in FreeBatchScanState failed: %s\n", cudaGetErrorString(sync_err));
        }

        // Double-check stream is idle
        cudaError_t query_err = cudaStreamQuery(state->stream);
        if (query_err == cudaErrorNotReady) {
            printf("WARNING: Stream not ready after synchronize, forcing device sync\n");
            cudaDeviceSynchronize();
        } else if (query_err != cudaSuccess) {
            printf("WARNING: cudaStreamQuery failed: %s\n", cudaGetErrorString(query_err));
        }
    }

    // Free CUDA buffers
    if (state->d_outputs) cudaFree(state->d_outputs);
    if (state->d_output_offsets) cudaFree(state->d_output_offsets);
    if (state->d_output_lengths) cudaFree(state->d_output_lengths);
    if (state->d_spend_pubkey_x) cudaFree(state->d_spend_pubkey_x);
    if (state->d_spend_pubkey_y) cudaFree(state->d_spend_pubkey_y);
    if (state->d_label_keys_x) cudaFree(state->d_label_keys_x);
    if (state->d_label_keys_y) cudaFree(state->d_label_keys_y);
    if (state->d_fpm_results_backup) cudaFree(state->d_fpm_results_backup);

    // Clean up solver resources (frees managed memory allocated in ec_pmul_init)
    if (state->solver) {
        state->solver->ec_pmul_close();
        delete state->solver;
    }

    // Destroy CUDA stream (safe now after synchronization)
    if (state->stream) {
        cudaStreamDestroy(state->stream);
    }

    // Free state struct
    delete state;
}

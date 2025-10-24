#include <cuda_runtime.h>
#include <cstdint>

#include "gecc.h"
#include "gecc/arith.h"
#include "gecc/ecdsa.h"

using namespace gecc;
using namespace arith;
using namespace ecdsa;

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

    // Extract lower 64 bits (first 2 u32 limbs in little-endian)
    uint64_t computed_value_u64 = static_cast<uint64_t>(x_normal.digits[0]) |
                                  (static_cast<uint64_t>(x_normal.digits[1]) << 32);
    int64_t computed_value = static_cast<int64_t>(computed_value_u64);

    // Check if computed_value is in the outputs list for this row
    bool found = false;
    uint32_t offset = output_offsets[instance];
    uint32_t length = output_lengths[instance];

    for (uint32_t j = 0; j < length; ++j) {
        if (outputs[offset + j] == computed_value) {
            found = true;
            break;
        }
    }

    // Write match flag
    match_flags[instance] = found ? 1 : 0;
}

// Per-batch state to hold auxiliary data between LaunchBatchScan and RunBatchScanKernels
struct BatchScanState {
    int64_t *d_outputs;
    uint32_t *d_output_offsets;
    uint32_t *d_output_lengths;
    Solver *solver;  // ECDSA solver instance
    uint32_t count;
};

// Host function to initialize solver and prepare for EC multiplication
// This follows the pattern from gECC's ec_pmul_random_init, but with our specific data
// Returns an opaque handle to BatchScanState (cast to void*) for thread-safe operation
extern "C" void* LaunchBatchScan(
    uint32_t **managed_points_x,      // Will allocate managed memory for input points
    uint32_t **managed_points_y,      // Will allocate managed memory for input points
    const uint32_t *h_scalar,         // Host scalar (8 u32 limbs)
    const int64_t *h_outputs,         // Host outputs array
    const uint32_t *h_output_offsets, // Host output offsets
    const uint32_t *h_output_lengths, // Host output lengths
    uint8_t **managed_match_flags,    // Will allocate managed memory for match results
    uint32_t count,
    size_t outputs_size) {

    // Initialize field and solver once per program (not per batch)
    static bool initialized = false;
    if (!initialized) {
        Solver::initialize();
        initialized = true;
    }

    // Allocate per-batch state (thread-safe)
    BatchScanState *state = new BatchScanState();
    state->d_outputs = nullptr;
    state->d_output_offsets = nullptr;
    state->d_output_lengths = nullptr;
    state->solver = nullptr;
    state->count = count;

    // Allocate managed memory for point coordinates (caller will fill these)
    cudaError_t err;
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

    // Allocate outputs metadata
    err = cudaMallocManaged(&state->d_outputs, outputs_size * sizeof(int64_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        delete state;
        return nullptr;
    }

    err = cudaMallocManaged(&state->d_output_offsets, count * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(state->d_outputs);
        delete state;
        return nullptr;
    }

    err = cudaMallocManaged(&state->d_output_lengths, count * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(state->d_outputs);
        cudaFree(state->d_output_offsets);
        delete state;
        return nullptr;
    }

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

    // Copy outputs metadata
    memcpy(state->d_outputs, h_outputs, outputs_size * sizeof(int64_t));
    memcpy(state->d_output_offsets, h_output_offsets, count * sizeof(uint32_t));
    memcpy(state->d_output_lengths, h_output_lengths, count * sizeof(uint32_t));

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
    uint8_t *managed_match_flags,
    uint32_t count) {

    BatchScanState *state = static_cast<BatchScanState*>(state_handle);
    if (!state || !state->solver) {
        printf("Error: Invalid state or solver not initialized\n");
        return -1;
    }

    Solver *solver = state->solver;

    cudaDeviceSynchronize();

    // Prepare data in the format expected by ec_pmul_random_init
    // MAX_LIMBS is defined in gECC as 64 (maximum array size)
    // For secp256k1 (256-bit), we use 4 u64 limbs, but arrays must be size MAX_LIMBS
    constexpr u32 MAX_LIMBS = 64;
    constexpr u32 USED_LIMBS = 4;  // 256 bits = 4 u64 limbs

    // Allocate host arrays in the format ec_pmul_random_init expects
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
        // IMPORTANT: gECC's ec_pmul_random_init uses reinterpret_cast<Base*>(u64_array)
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

    // Call ec_pmul_random_init with our specific data
    solver->ec_pmul_random_init(h_scalars, h_keys_x, h_keys_y, count);

    // Free host arrays
    delete[] h_scalars;
    delete[] h_keys_x;
    delete[] h_keys_y;

    // Check for initialization errors
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ec_pmul_random_init error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Run EC multiplication (matching ecdsa_ec_pmul call)
    // gECC correctness test uses MAX_SM_NUMS blocks (SM count) with 256 threads
    // This seems to be required for the algorithm to work correctly
    u32 max_thread_per_block = 256;
    u32 block_num = MAX_SM_NUMS;  // Use SM count like gECC test

    solver->ecdsa_ec_pmul(block_num, max_thread_per_block, true);  // true = unknown points

    // Check for multiplication errors
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        printf("ecdsa_ec_pmul error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Extract X coordinates from interleaved R0 format
    // R0 layout: Point i has X at [i * field_limbs * 2], Y at [i * field_limbs * 2 + field_limbs]
    // See gECC/ECDSA_CORRECTNESS_TEST.md lines 99-111 for details
    constexpr u32 field_limbs = 8;  // 8 u32 limbs for 256-bit field

    uint32_t *result_x_coords;
    err = cudaMallocManaged(&result_x_coords, count * field_limbs * sizeof(uint32_t));
    if (err != cudaSuccess) {
        printf("cudaMallocManaged for result_x_coords error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Extract X coordinates from column-major format (when GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS is defined)
    // With store_arbitrary: X limb j of point i is at R0[j * count + i]
    for (u32 i = 0; i < count; i++) {
        for (u32 j = 0; j < field_limbs; j++) {
            result_x_coords[i * field_limbs + j] = solver->R0[j * count + i];
        }
    }

    cudaDeviceSynchronize();

    // Now check matches
    int threads_per_block = 256;
    int num_blocks = (count + threads_per_block - 1) / threads_per_block;

    CheckMatchesKernel<<<num_blocks, threads_per_block>>>(
        result_x_coords,  // Extracted X coordinates (in Montgomery form)
        state->d_outputs, state->d_output_offsets, state->d_output_lengths,
        managed_match_flags, count
    );

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CheckMatchesKernel error: %s\n", cudaGetErrorString(err));
        cudaFree(result_x_coords);
        return -1;
    }

    // Free extracted coordinates
    cudaFree(result_x_coords);

    return 0;
}

// Cleanup function to free batch state
extern "C" void FreeBatchScanState(void *state_handle) {
    if (!state_handle) return;

    BatchScanState *state = static_cast<BatchScanState*>(state_handle);

    // Free CUDA buffers
    if (state->d_outputs) cudaFree(state->d_outputs);
    if (state->d_output_offsets) cudaFree(state->d_output_offsets);
    if (state->d_output_lengths) cudaFree(state->d_output_lengths);

    // Delete solver
    if (state->solver) delete state->solver;

    // Free state struct
    delete state;
}

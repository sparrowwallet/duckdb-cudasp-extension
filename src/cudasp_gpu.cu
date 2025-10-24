#include <cuda_runtime.h>
#include <cstdint>

#include "gecc.h"
#include "gecc/arith.h"

using namespace gecc;
using namespace arith;

// Define field and EC types for secp256k1
// secp256k1 has a=0, so use DBL_FLAG=1 for the a=0 optimized doubling formula
DEFINE_SECP256K1_FP(Fq_SECP256K1, FqSECP256K1, u32, 32, LayoutT<1>, 8, gecc::arith::MONTFLAG::SOS, gecc::arith::CURVEFLAG::DEFAULT);
DEFINE_EC(G1_EC, G1SECP256K1, Fq_SECP256K1, SECP256K1_CURVE, 1);

using Field = Fq_SECP256K1;
using ECPoint = G1_EC_G1SECP256K1;  // DEFINE_EC creates TYPE_NAME_CURVE_NAME

// Kernel to convert points to Montgomery form (similar to processScalarPoint in gECC)
template <typename EC, typename Field>
__global__ void ProcessPointsKernel(
    typename Field::Base *points_x,
    typename Field::Base *points_y,
    uint32_t count) {

    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const u32 slot_idx = LayoutT<1>::global_slot_idx();
    const u32 lane_idx = LayoutT<1>::lane_idx();

    // Load point coordinates
    Field px, py;
    px.load_arbitrary(points_x, count, slot_idx, lane_idx);
    py.load_arbitrary(points_y, count, slot_idx, lane_idx);

    // Convert to Montgomery form
    px.inplace_to_montgomery();
    py.inplace_to_montgomery();

    // Store back
    px.store_arbitrary(points_x, count, slot_idx, lane_idx);
    py.store_arbitrary(points_y, count, slot_idx, lane_idx);
}

// CUDA kernel: Batch scalar multiplication followed by int64 extraction and list comparison
// Each thread computes: result_int64 = extract_int64(scalar * point[i]), then checks if result_int64 is in outputs[i]
__global__ void BatchScanKernel(
    const uint32_t *input_points_x,      // X coordinates of input tweak keys (column-major if GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS)
    const uint32_t *input_points_y,      // Y coordinates of input tweak keys
    const uint32_t *scalar,              // Shared scalar (scan_private_key) - 8 u32 limbs
    const int64_t *outputs,              // Flattened outputs array
    const uint32_t *output_offsets,      // Offset into outputs for each row
    const uint32_t *output_lengths,      // Length of outputs list for each row
    uint8_t *match_flags,                // Output: 1 if match found, 0 otherwise
    uint32_t count) {                    // Number of points to process

    const u32 slot_idx = LayoutT<1>::global_slot_idx();
    const u32 lane_idx = LayoutT<1>::lane_idx();

    if (slot_idx >= count) return;

    // Load input point coordinates
    Field px, py;
    px.load_arbitrary(input_points_x, count, slot_idx, lane_idx);
    py.load_arbitrary(input_points_y, count, slot_idx, lane_idx);

    // Load scalar (shared across all threads, but each thread loads it)
    Field scalar_field;
    // For shared scalar, we broadcast it to all threads
    // Since scalar is not in column-major format, we need to load it specially
    #pragma unroll
    for (u32 i = 0; i < 8; ++i) {
        scalar_field.digits[i] = scalar[i];
    }

    // Convert to Jacobian coordinates (Z=1 in Montgomery form)
    ECPoint base_jac;
    base_jac.x = px;
    base_jac.y = py;
    base_jac.z = Field::mont_one();

    // Perform scalar multiplication using double-and-add
    ECPoint result_jac = ECPoint::zero();

    for (int bit = Field::BITS - 1; bit >= 0; --bit) {
        result_jac = result_jac.dbl();

        // Check if bit is set in scalar
        u32 limb_idx = bit / 32;
        u32 bit_pos = bit % 32;
        bool bit_set = (scalar_field.digits[limb_idx] >> bit_pos) & 1;

        if (bit_set) {
            result_jac = result_jac + base_jac;
        }
    }

    // Convert result back to affine coordinates
    Field result_z_inv = result_jac.z.inverse();
    Field result_z_inv_sq = result_z_inv * result_z_inv;
    Field result_x = result_jac.x * result_z_inv_sq;

    // Extract int64 from result point (lower 64 bits of x-coordinate)
    // The x-coordinate is in Montgomery form, so convert to normal form first
    Field result_x_normal = result_x.from_montgomery();

    // Extract lower 64 bits (first 2 u32 limbs in little-endian)
    int64_t computed_value = static_cast<int64_t>(result_x_normal.digits[0]) |
                             (static_cast<int64_t>(result_x_normal.digits[1]) << 32);

    // Check if computed_value is in the outputs list for this row
    bool found = false;
    uint32_t offset = output_offsets[slot_idx];
    uint32_t length = output_lengths[slot_idx];

    for (uint32_t j = 0; j < length; ++j) {
        if (outputs[offset + j] == computed_value) {
            found = true;
            break;
        }
    }

    // Write match flag
    match_flags[slot_idx] = found ? 1 : 0;
}

// Host function to setup managed memory and launch kernels
// This uses cudaMallocManaged and optimizations similar to gECC's ec_pmul_random_init
extern "C" int LaunchBatchScan(
    uint32_t **managed_points_x,      // Will allocate managed memory
    uint32_t **managed_points_y,      // Will allocate managed memory
    const uint32_t *h_scalar,         // Host scalar (8 u32 limbs)
    const int64_t *h_outputs,         // Host outputs array
    const uint32_t *h_output_offsets, // Host output offsets
    const uint32_t *h_output_lengths, // Host output lengths
    uint8_t **managed_match_flags,    // Will allocate managed memory
    uint32_t count,
    size_t outputs_size) {

    // Initialize field parameters (must be called before using field operations)
    Fq_SECP256K1::initialize();

    // Allocate managed memory (accessible from both CPU and GPU)
    cudaError_t err;
    err = cudaMallocManaged(managed_points_x, Field::SIZE * count);
    if (err != cudaSuccess) return -1;

    err = cudaMallocManaged(managed_points_y, Field::SIZE * count);
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        return -1;
    }

    uint32_t *d_scalar;
    err = cudaMallocManaged(&d_scalar, Field::SIZE);
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        return -1;
    }

    int64_t *d_outputs;
    err = cudaMallocManaged(&d_outputs, outputs_size * sizeof(int64_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(d_scalar);
        return -1;
    }

    uint32_t *d_output_offsets, *d_output_lengths;
    err = cudaMallocManaged(&d_output_offsets, count * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(d_scalar);
        cudaFree(d_outputs);
        return -1;
    }

    err = cudaMallocManaged(&d_output_lengths, count * sizeof(uint32_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(d_scalar);
        cudaFree(d_outputs);
        cudaFree(d_output_offsets);
        return -1;
    }

    err = cudaMallocManaged(managed_match_flags, count * sizeof(uint8_t));
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(d_scalar);
        cudaFree(d_outputs);
        cudaFree(d_output_offsets);
        cudaFree(d_output_lengths);
        return -1;
    }

    // Copy scalar and auxiliary data (host->managed can be done directly)
    memcpy(d_scalar, h_scalar, Field::SIZE);
    memcpy(d_outputs, h_outputs, outputs_size * sizeof(int64_t));
    memcpy(d_output_offsets, h_output_offsets, count * sizeof(uint32_t));
    memcpy(d_output_lengths, h_output_lengths, count * sizeof(uint32_t));

    // Points data should already be copied by caller to *managed_points_x and *managed_points_y

    cudaDeviceSynchronize();

    // Process points: convert to Montgomery form on GPU
    int threads_per_block = 256;
    int num_blocks = (count + threads_per_block - 1) / threads_per_block;
    ProcessPointsKernel<ECPoint, Field><<<num_blocks, threads_per_block>>>(*managed_points_x, *managed_points_y, count);
    cudaDeviceSynchronize();

#ifdef PERSISTENT_L2_CACHE
    // Optional: Set up persistent L2 cache for better performance (CUDA 11.0+)
    // This helps keep frequently accessed data in L2 cache
    #if CUDART_VERSION >= 11000
        cudaDeviceProp device_prop;
        int current_device = 0;
        cudaGetDevice(&current_device);
        cudaGetDeviceProperties(&device_prop, current_device);
        size_t accessPolicyMaxWindowSize = device_prop.accessPolicyMaxWindowSize;

        if (accessPolicyMaxWindowSize > 0) {
            size_t needed_bytes_pers_l2_cache = count * Field::SIZE;
            size_t setted_pers_l2_cache = std::max(needed_bytes_pers_l2_cache,
                                                     std::min(needed_bytes_pers_l2_cache, accessPolicyMaxWindowSize));

            cudaStreamAttrValue stream_attribute;
            stream_attribute.accessPolicyWindow.base_ptr = reinterpret_cast<void*>(*managed_points_x);
            stream_attribute.accessPolicyWindow.num_bytes = setted_pers_l2_cache;
            stream_attribute.accessPolicyWindow.hitRatio = 1.0;
            stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
        }
    #endif
#endif

    // Launch batch scan kernel
    BatchScanKernel<<<num_blocks, threads_per_block>>>(
        *managed_points_x, *managed_points_y, d_scalar,
        d_outputs, d_output_offsets, d_output_lengths,
        *managed_match_flags, count
    );

    cudaDeviceSynchronize();

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(*managed_points_x);
        cudaFree(*managed_points_y);
        cudaFree(d_scalar);
        cudaFree(d_outputs);
        cudaFree(d_output_offsets);
        cudaFree(d_output_lengths);
        cudaFree(*managed_match_flags);
        return -1;
    }

    // Free auxiliary memory (keep points and match_flags for caller)
    cudaFree(d_scalar);
    cudaFree(d_outputs);
    cudaFree(d_output_offsets);
    cudaFree(d_output_lengths);

    return 0;
}

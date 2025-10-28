#define DUCKDB_EXTENSION_MAIN

#include "cudasp_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

// CUDA runtime for GPU operations
#include <cuda_runtime.h>

// Declare CUDA functions from cudasp_gpu.cu
extern "C" {
	void* LaunchBatchScan(
	    uint32_t **managed_points_x,
	    uint32_t **managed_points_y,
	    const uint32_t *h_scalar,
	    const uint32_t *h_spend_pubkey_x,
	    const uint32_t *h_spend_pubkey_y,
	    const uint32_t *h_label_keys_x,
	    const uint32_t *h_label_keys_y,
	    uint32_t label_count,
	    const int64_t *h_outputs,
	    const uint32_t *h_output_offsets,
	    const uint32_t *h_output_lengths,
	    uint8_t **managed_match_flags,
	    uint32_t count,
	    size_t outputs_size);

	int RunBatchScanKernels(
	    void *state_handle,
	    uint32_t *managed_points_x,
	    uint32_t *managed_points_y,
	    const uint32_t *h_scalar,
	    const uint32_t *h_spend_pubkey_x,
	    const uint32_t *h_spend_pubkey_y,
	    const uint32_t *h_label_keys_x,
	    const uint32_t *h_label_keys_y,
	    uint32_t label_count,
	    uint8_t *managed_match_flags,
	    uint32_t count);

	void FreeBatchScanState(void *state_handle);
}

namespace duckdb {

// Helper function to convert little-endian 64-byte BLOB (x||y) to uint32_t array
// Input: 64 bytes = 32-byte x (little-endian) || 32-byte y (little-endian)
// Output: 16 uint32_t values (8 for x, 8 for y) in little-endian order
static void ConvertTweakKeyToU32(const uint8_t* blob_data, uint32_t* out_x, uint32_t* out_y) {
	// Convert 32-byte x coordinate (little-endian) to 8 u32 limbs
	for (int i = 0; i < 8; i++) {
		out_x[i] = static_cast<uint32_t>(blob_data[i * 4]) |
		           (static_cast<uint32_t>(blob_data[i * 4 + 1]) << 8) |
		           (static_cast<uint32_t>(blob_data[i * 4 + 2]) << 16) |
		           (static_cast<uint32_t>(blob_data[i * 4 + 3]) << 24);
	}

	// Convert 32-byte y coordinate (little-endian) to 8 u32 limbs
	for (int i = 0; i < 8; i++) {
		out_y[i] = static_cast<uint32_t>(blob_data[32 + i * 4]) |
		           (static_cast<uint32_t>(blob_data[32 + i * 4 + 1]) << 8) |
		           (static_cast<uint32_t>(blob_data[32 + i * 4 + 2]) << 16) |
		           (static_cast<uint32_t>(blob_data[32 + i * 4 + 3]) << 24);
	}
}

// Helper function to convert 32-byte scalar BLOB to uint32_t array
static void ConvertScalarToU32(const uint8_t* blob_data, uint32_t* out_scalar) {
	for (int i = 0; i < 8; i++) {
		out_scalar[i] = static_cast<uint32_t>(blob_data[i * 4]) |
		                (static_cast<uint32_t>(blob_data[i * 4 + 1]) << 8) |
		                (static_cast<uint32_t>(blob_data[i * 4 + 2]) << 16) |
		                (static_cast<uint32_t>(blob_data[i * 4 + 3]) << 24);
	}
}

struct CudaspScanBindData : public TableFunctionData {
	CudaspScanBindData() : batch_size(10000) {
	}
	static constexpr idx_t TWEAK_KEY_SIZE = 64; // 64 bytes: uncompressed EC point (32-byte x || 32-byte y)
	static constexpr idx_t SCALAR_SIZE = 32; // 32 bytes: scalar for EC multiplication

	// Configurable batch size for GPU processing
	idx_t batch_size;

	// Scalar parameter for EC multiplication (shared across all rows) - owned copy
	std::string scan_private_key_data;

	// Spend public key for EC point addition (shared across all rows) - owned copy
	std::string spend_public_key_data;

	// Label keys for checking alternative outputs (shared across all rows) - owned copies
	std::vector<std::string> label_keys_data;
};

struct CudaspScanLocalState : public LocalTableFunctionState {
	CudaspScanLocalState() : finalized(false), output_position(0) {
	}
	bool finalized;

	// Per-thread accumulated input data
	vector<string_t> accumulated_txids;        // Transaction IDs (BLOB)
	vector<int32_t> accumulated_heights;       // Block heights (INTEGER)
	vector<string_t> accumulated_tweak_keys;   // 64-byte EC points (BLOB)
	vector<int64_t> accumulated_outputs;       // Flattened output values (BIGINT)
	vector<idx_t> accumulated_output_offsets;  // Offset into accumulated_outputs for each row
	vector<idx_t> accumulated_output_lengths;  // Length of each outputs list

	// Per-thread processed output data (only rows with matches)
	vector<string_t> output_txids;
	vector<int32_t> output_heights;
	vector<string_t> output_tweak_keys;
	idx_t output_position;
};

struct CudaspScanState : public GlobalTableFunctionState {
	CudaspScanState() : currently_adding(0) {
		finalize_lock = make_uniq<std::mutex>();
	}

	// Thread synchronization
	std::atomic_uint64_t currently_adding;
	unique_ptr<std::mutex> finalize_lock;
};

static void AccumulateInput(CudaspScanLocalState &local_state, DataChunk &input) {
	idx_t count = input.size();
	// Expected columns: txid (BLOB), height (INTEGER), tweak_key (BLOB), outputs (LIST[BIGINT])
	auto &txid_column = input.data[0];
	auto &height_column = input.data[1];
	auto &tweak_key_column = input.data[2];
	auto &outputs_column = input.data[3];

	// Get unified vector format for input columns
	UnifiedVectorFormat txid_data;
	UnifiedVectorFormat height_data;
	UnifiedVectorFormat tweak_key_data;
	UnifiedVectorFormat outputs_data;

	txid_column.ToUnifiedFormat(count, txid_data);
	height_column.ToUnifiedFormat(count, height_data);
	tweak_key_column.ToUnifiedFormat(count, tweak_key_data);
	outputs_column.ToUnifiedFormat(count, outputs_data);

	auto txid_ptr = UnifiedVectorFormat::GetData<string_t>(txid_data);
	auto height_ptr = UnifiedVectorFormat::GetData<int32_t>(height_data);
	auto tweak_key_ptr = UnifiedVectorFormat::GetData<string_t>(tweak_key_data);
	auto outputs_entries = UnifiedVectorFormat::GetData<list_entry_t>(outputs_data);

	// Get outputs list child data
	auto &outputs_child = ListVector::GetEntry(outputs_column);
	UnifiedVectorFormat outputs_child_data;
	outputs_child.ToUnifiedFormat(ListVector::GetListSize(outputs_column), outputs_child_data);
	auto outputs_child_ptr = UnifiedVectorFormat::GetData<int64_t>(outputs_child_data);

	// Accumulate the data
	for (idx_t i = 0; i < count; i++) {
		auto txid_idx = txid_data.sel->get_index(i);
		auto height_idx = height_data.sel->get_index(i);
		auto tweak_key_idx = tweak_key_data.sel->get_index(i);
		auto outputs_idx = outputs_data.sel->get_index(i);

		// Only process rows with valid txid, height, and tweak_key
		if (txid_data.validity.RowIsValid(txid_idx) &&
		    height_data.validity.RowIsValid(height_idx) &&
		    tweak_key_data.validity.RowIsValid(tweak_key_idx)) {

			local_state.accumulated_txids.push_back(txid_ptr[txid_idx]);
			local_state.accumulated_heights.push_back(height_ptr[height_idx]);
			local_state.accumulated_tweak_keys.push_back(tweak_key_ptr[tweak_key_idx]);

			// Store outputs list offset and length
			idx_t outputs_offset = local_state.accumulated_outputs.size();
			local_state.accumulated_output_offsets.push_back(outputs_offset);

			idx_t outputs_len = 0;
			if (outputs_data.validity.RowIsValid(outputs_idx)) {
				auto &outputs_entry = outputs_entries[outputs_idx];
				for (idx_t out_i = 0; out_i < outputs_entry.length; out_i++) {
					auto child_idx = outputs_child_data.sel->get_index(outputs_entry.offset + out_i);
					if (outputs_child_data.validity.RowIsValid(child_idx)) {
						local_state.accumulated_outputs.push_back(outputs_child_ptr[child_idx]);
						outputs_len++;
					}
				}
			}
			local_state.accumulated_output_lengths.push_back(outputs_len);
		}
	}
}

static void ProcessBatch(CudaspScanLocalState &local_state, const CudaspScanBindData &bind_data) {
	// Clear any previous output
	local_state.output_txids.clear();
	local_state.output_heights.clear();
	local_state.output_tweak_keys.clear();
	local_state.output_position = 0;

	idx_t batch_size = local_state.accumulated_txids.size();
	if (batch_size == 0) {
		return;
	}

	// Prepare host data for GPU
	const idx_t field_limbs = 8; // 8 u32 limbs for 256-bit field

	// Allocate managed memory for point coordinates (will be accessible from GPU)
	uint32_t *managed_points_x = nullptr;
	uint32_t *managed_points_y = nullptr;
	uint8_t *managed_match_flags = nullptr;

	// For now, allocate temporary host arrays for conversion
	std::vector<uint32_t> h_points_x(batch_size * field_limbs);
	std::vector<uint32_t> h_points_y(batch_size * field_limbs);

	// Convert tweak_keys from BLOB to u32 format (row-major first)
	for (idx_t i = 0; i < batch_size; i++) {
		const uint8_t* tweak_data = reinterpret_cast<const uint8_t*>(local_state.accumulated_tweak_keys[i].GetData());
		ConvertTweakKeyToU32(tweak_data,
		                     &h_points_x[i * field_limbs],
		                     &h_points_y[i * field_limbs]);
	}

#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
	// Convert to column-major layout for gECC optimization
	std::vector<uint32_t> h_points_x_col(batch_size * field_limbs);
	std::vector<uint32_t> h_points_y_col(batch_size * field_limbs);

	for (idx_t j = 0; j < field_limbs; j++) {
		for (idx_t i = 0; i < batch_size; i++) {
			h_points_x_col[j * batch_size + i] = h_points_x[i * field_limbs + j];
			h_points_y_col[j * batch_size + i] = h_points_y[i * field_limbs + j];
		}
	}

	// Use column-major data
	h_points_x = std::move(h_points_x_col);
	h_points_y = std::move(h_points_y_col);
#endif

	// Convert scalar from BLOB to u32 format
	uint32_t h_scalar[field_limbs];
	const uint8_t* scalar_data = reinterpret_cast<const uint8_t*>(bind_data.scan_private_key_data.data());
	ConvertScalarToU32(scalar_data, h_scalar);

	// Convert spend_public_key from BLOB to u32 format (x and y coordinates)
	uint32_t h_spend_pubkey_x[field_limbs];
	uint32_t h_spend_pubkey_y[field_limbs];
	const uint8_t* spend_pubkey_data = reinterpret_cast<const uint8_t*>(bind_data.spend_public_key_data.data());
	ConvertTweakKeyToU32(spend_pubkey_data, h_spend_pubkey_x, h_spend_pubkey_y);

	// Convert label keys from BLOB to u32 format (flattened array)
	idx_t label_count = bind_data.label_keys_data.size();
	std::vector<uint32_t> h_label_keys_x(label_count * field_limbs);
	std::vector<uint32_t> h_label_keys_y(label_count * field_limbs);
	for (idx_t i = 0; i < label_count; i++) {
		const uint8_t* label_key_data = reinterpret_cast<const uint8_t*>(bind_data.label_keys_data[i].data());
		ConvertTweakKeyToU32(label_key_data,
		                     &h_label_keys_x[i * field_limbs],
		                     &h_label_keys_y[i * field_limbs]);
	}

	// Prepare output offsets and lengths as uint32_t
	std::vector<uint32_t> h_output_offsets(batch_size);
	std::vector<uint32_t> h_output_lengths(batch_size);
	for (idx_t i = 0; i < batch_size; i++) {
		h_output_offsets[i] = static_cast<uint32_t>(local_state.accumulated_output_offsets[i]);
		h_output_lengths[i] = static_cast<uint32_t>(local_state.accumulated_output_lengths[i]);
	}

	// Allocate managed memory for GPU processing and create solver state
	void *state_handle = LaunchBatchScan(
	    &managed_points_x,  // Function allocates managed memory
	    &managed_points_y,  // Function allocates managed memory
	    h_scalar,
	    h_spend_pubkey_x,
	    h_spend_pubkey_y,
	    h_label_keys_x.data(),
	    h_label_keys_y.data(),
	    static_cast<uint32_t>(label_count),
	    local_state.accumulated_outputs.data(),
	    h_output_offsets.data(),
	    h_output_lengths.data(),
	    &managed_match_flags,  // Function allocates managed memory
	    static_cast<uint32_t>(batch_size),
	    local_state.accumulated_outputs.size()
	);

	if (state_handle) {
	    // Write points data directly to managed memory (exactly like gECC's ec_pmul_random_init)
#ifdef GECC_QAPW_OPT_COLUMN_MAJORED_INPUTS
	    // Write in column-major layout directly
	    for (idx_t j = 0; j < field_limbs; j++) {
	        for (idx_t i = 0; i < batch_size; i++) {
	            const uint8_t* tweak_data = reinterpret_cast<const uint8_t*>(local_state.accumulated_tweak_keys[i].GetData());
	            uint32_t x_limbs[8], y_limbs[8];
	            ConvertTweakKeyToU32(tweak_data, x_limbs, y_limbs);
	            managed_points_x[j * batch_size + i] = x_limbs[j];
	            managed_points_y[j * batch_size + i] = y_limbs[j];
	        }
	    }
#else
	    memcpy(managed_points_x, h_points_x.data(), batch_size * field_limbs * sizeof(uint32_t));
	    memcpy(managed_points_y, h_points_y.data(), batch_size * field_limbs * sizeof(uint32_t));
#endif

	    // Convert scalar to u32 array
	    uint32_t h_scalar_local[8];
	    ConvertScalarToU32(reinterpret_cast<const uint8_t*>(bind_data.scan_private_key_data.data()), h_scalar_local);

	    // Convert spend_public_key to u32 arrays
	    uint32_t h_spend_pubkey_x_local[8];
	    uint32_t h_spend_pubkey_y_local[8];
	    ConvertTweakKeyToU32(reinterpret_cast<const uint8_t*>(bind_data.spend_public_key_data.data()),
	                         h_spend_pubkey_x_local, h_spend_pubkey_y_local);

	    // Prepare label keys for kernel call
	    std::vector<uint32_t> h_label_keys_x_local(label_count * 8);
	    std::vector<uint32_t> h_label_keys_y_local(label_count * 8);
	    for (idx_t i = 0; i < label_count; i++) {
	        const uint8_t* label_key_data = reinterpret_cast<const uint8_t*>(bind_data.label_keys_data[i].data());
	        ConvertTweakKeyToU32(label_key_data,
	                             &h_label_keys_x_local[i * 8],
	                             &h_label_keys_y_local[i * 8]);
	    }

	    // Now run the GPU kernels
	    int kernel_result = RunBatchScanKernels(
	        state_handle,
	        managed_points_x,
	        managed_points_y,
	        h_scalar_local,
	        h_spend_pubkey_x_local,
	        h_spend_pubkey_y_local,
	        h_label_keys_x_local.data(),
	        h_label_keys_y_local.data(),
	        static_cast<uint32_t>(label_count),
	        managed_match_flags,
	        static_cast<uint32_t>(batch_size)
	    );

	    if (kernel_result == 0) {
	        // Build output for matching rows
	        for (idx_t i = 0; i < batch_size; i++) {
	            if (managed_match_flags[i]) {
	                local_state.output_txids.push_back(local_state.accumulated_txids[i]);
	                local_state.output_heights.push_back(local_state.accumulated_heights[i]);
	                local_state.output_tweak_keys.push_back(local_state.accumulated_tweak_keys[i]);
	            }
	        }
	    }

	    // Cleanup managed memory
	    cudaFree(managed_points_x);
	    cudaFree(managed_points_y);
	    cudaFree(managed_match_flags);

	    // Free batch state
	    FreeBatchScanState(state_handle);
	}

	// Clear accumulated input after processing
	local_state.accumulated_txids.clear();
	local_state.accumulated_heights.clear();
	local_state.accumulated_tweak_keys.clear();
	local_state.accumulated_outputs.clear();
	local_state.accumulated_output_offsets.clear();
	local_state.accumulated_output_lengths.clear();
}

static bool HasOutput(const CudaspScanLocalState &local_state) {
	return local_state.output_position < local_state.output_txids.size();
}

static bool ShouldProcessBatch(const CudaspScanLocalState &local_state, const CudaspScanBindData &bind_data) {
	return local_state.accumulated_txids.size() >= bind_data.batch_size;
}

static unique_ptr<FunctionData> CudaspScanBind(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<string> &names) {
	// Validate input: expects TABLE, scan_private_key BLOB, spend_public_key BLOB, label_keys LIST, and optional batch_size INTEGER
	if (input.inputs.size() != 4 && input.inputs.size() != 5) {
		throw InvalidInputException("cudasp_scan requires 4 or 5 arguments: TABLE, scan_private_key BLOB, spend_public_key BLOB, label_keys LIST[BLOB], and optional batch_size INTEGER");
	}

	// Validate scan_private_key parameter
	auto &scalar_value = input.inputs[1];
	if (scalar_value.type().id() != LogicalTypeId::BLOB) {
		throw InvalidInputException("Second argument must be a BLOB (32-byte scan_private_key)");
	}

	// Get the scan_private_key blob and copy it
	string_t scan_private_key = StringValue::Get(scalar_value);
	if (scan_private_key.GetSize() != CudaspScanBindData::SCALAR_SIZE) {
		throw InvalidInputException("scan_private_key must be exactly 32 bytes, got %llu bytes", scan_private_key.GetSize());
	}

	// Validate spend_public_key parameter
	auto &spend_pubkey_value = input.inputs[2];
	if (spend_pubkey_value.type().id() != LogicalTypeId::BLOB) {
		throw InvalidInputException("Third argument must be a BLOB (64-byte spend_public_key)");
	}

	// Get the spend_public_key blob and copy it
	string_t spend_public_key = StringValue::Get(spend_pubkey_value);
	if (spend_public_key.GetSize() != CudaspScanBindData::TWEAK_KEY_SIZE) {
		throw InvalidInputException("spend_public_key must be exactly 64 bytes, got %llu bytes", spend_public_key.GetSize());
	}

	// Validate label_keys parameter (LIST of BLOBs)
	auto &label_keys_value = input.inputs[3];
	if (label_keys_value.type().id() != LogicalTypeId::LIST) {
		throw InvalidInputException("Fourth argument must be a LIST[BLOB] (label keys)");
	}

	// Parse label_keys list
	std::vector<std::string> label_keys;
	auto &list_value = ListValue::GetChildren(label_keys_value);
	for (idx_t i = 0; i < list_value.size(); i++) {
		auto &label_key_value = list_value[i];
		if (label_key_value.type().id() != LogicalTypeId::BLOB) {
			throw InvalidInputException("All elements in label_keys must be BLOBs");
		}
		string_t label_key = StringValue::Get(label_key_value);
		if (label_key.GetSize() != CudaspScanBindData::TWEAK_KEY_SIZE) {
			throw InvalidInputException("Each label key must be exactly 64 bytes, got %llu bytes", label_key.GetSize());
		}
		label_keys.push_back(std::string(label_key.GetData(), label_key.GetSize()));
	}

	// Parse optional batch_size parameter (default: 10000)
	idx_t batch_size = 10000;
	if (input.inputs.size() == 5) {
		auto &batch_size_value = input.inputs[4];
		if (batch_size_value.type().id() != LogicalTypeId::INTEGER &&
		    batch_size_value.type().id() != LogicalTypeId::BIGINT) {
			throw InvalidInputException("Fifth argument (batch_size) must be an INTEGER");
		}
		int64_t batch_size_int = IntegerValue::Get(batch_size_value);
		if (batch_size_int <= 0) {
			throw InvalidInputException("batch_size must be positive, got %lld", batch_size_int);
		}
		if (batch_size_int > 10000000) {
			throw InvalidInputException("batch_size too large (max 10,000,000), got %lld", batch_size_int);
		}
		batch_size = static_cast<idx_t>(batch_size_int);
	}

	// Set return types: txid (BLOB), height (INTEGER), tweak_key (BLOB)
	return_types.push_back(LogicalType::BLOB);    // txid
	return_types.push_back(LogicalType::INTEGER); // height
	return_types.push_back(LogicalType::BLOB);    // tweak_key
	names.push_back("txid");
	names.push_back("height");
	names.push_back("tweak_key");

	auto bind_data = make_uniq<CudaspScanBindData>();
	// Copy the scan_private_key data into owned memory
	bind_data->scan_private_key_data = std::string(scan_private_key.GetData(), scan_private_key.GetSize());
	// Copy the spend_public_key data into owned memory
	bind_data->spend_public_key_data = std::string(spend_public_key.GetData(), spend_public_key.GetSize());
	// Copy label keys
	bind_data->label_keys_data = std::move(label_keys);
	// Set batch size
	bind_data->batch_size = batch_size;
	return bind_data;
}

static unique_ptr<GlobalTableFunctionState> CudaspScanInit(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<CudaspScanState>();
}

static unique_ptr<LocalTableFunctionState> CudaspScanLocalInit(ExecutionContext &context, TableFunctionInitInput &input,
                                                                      GlobalTableFunctionState *global_state) {
	auto &state = global_state->Cast<CudaspScanState>();
	state.currently_adding++;
	return make_uniq<CudaspScanLocalState>();
}

static OperatorResultType CudaspScanFunction(ExecutionContext &context, TableFunctionInput &data_p,
                                                    DataChunk &input, DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<CudaspScanBindData>();
	auto &local_state = data_p.local_state->Cast<CudaspScanLocalState>();

	// If we have pending output from a previous batch, return it first
	if (HasOutput(local_state)) {
		auto &txid_result = output.data[0];
		auto &height_result = output.data[1];
		auto &tweak_key_result = output.data[2];

		idx_t output_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE,
		                                       local_state.output_txids.size() - local_state.output_position);

		auto txid_data = FlatVector::GetData<string_t>(txid_result);
		auto height_data = FlatVector::GetData<int32_t>(height_result);
		auto tweak_key_data = FlatVector::GetData<string_t>(tweak_key_result);

		for (idx_t i = 0; i < output_count; i++) {
			txid_data[i] = StringVector::AddStringOrBlob(txid_result, local_state.output_txids[local_state.output_position + i]);
			height_data[i] = local_state.output_heights[local_state.output_position + i];
			tweak_key_data[i] = StringVector::AddStringOrBlob(tweak_key_result, local_state.output_tweak_keys[local_state.output_position + i]);
		}

		output.SetCardinality(output_count);
		local_state.output_position += output_count;

		// If we still have more output, signal that
		if (HasOutput(local_state)) {
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}

		// All output returned, clear buffers
		local_state.output_txids.clear();
		local_state.output_heights.clear();
		local_state.output_tweak_keys.clear();
		local_state.output_position = 0;

		// Otherwise keep accepting input
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// Process new input
	if (input.size() > 0) {
		AccumulateInput(local_state, input);

		// Process batch if we've accumulated enough data
		if (ShouldProcessBatch(local_state, bind_data)) {
			ProcessBatch(local_state, bind_data);
			local_state.output_position = 0;

			// Write output immediately
			auto &txid_result = output.data[0];
			auto &height_result = output.data[1];
			auto &tweak_key_result = output.data[2];

			idx_t output_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, local_state.output_txids.size());

			auto txid_data = FlatVector::GetData<string_t>(txid_result);
			auto height_data = FlatVector::GetData<int32_t>(height_result);
			auto tweak_key_data = FlatVector::GetData<string_t>(tweak_key_result);

			for (idx_t i = 0; i < output_count; i++) {
				txid_data[i] = StringVector::AddStringOrBlob(txid_result, local_state.output_txids[i]);
				height_data[i] = local_state.output_heights[i];
				tweak_key_data[i] = StringVector::AddStringOrBlob(tweak_key_result, local_state.output_tweak_keys[i]);
			}

			output.SetCardinality(output_count);
			local_state.output_position = output_count;

			// We just wrote data to output, so we MUST return HAVE_MORE_OUTPUT
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}

		// Keep accumulating
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// No more input - should not reach here, finalize handles remaining data
	return OperatorResultType::NEED_MORE_INPUT;
}

static OperatorFinalizeResultType CudaspScanFinalFunction(ExecutionContext &context, TableFunctionInput &data_p,
                                                                  DataChunk &output) {
	auto &bind_data = data_p.bind_data->Cast<CudaspScanBindData>();
	auto &state = data_p.global_state->Cast<CudaspScanState>();
	auto &local_state = data_p.local_state->Cast<CudaspScanLocalState>();

	// If we still have pending output from previous batch, return it
	if (HasOutput(local_state)) {
		auto &txid_result = output.data[0];
		auto &height_result = output.data[1];
		auto &tweak_key_result = output.data[2];

		idx_t output_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE,
		                                       local_state.output_txids.size() - local_state.output_position);

		auto txid_data = FlatVector::GetData<string_t>(txid_result);
		auto height_data = FlatVector::GetData<int32_t>(height_result);
		auto tweak_key_data = FlatVector::GetData<string_t>(tweak_key_result);

		for (idx_t i = 0; i < output_count; i++) {
			txid_data[i] = StringVector::AddStringOrBlob(txid_result, local_state.output_txids[local_state.output_position + i]);
			height_data[i] = local_state.output_heights[local_state.output_position + i];
			tweak_key_data[i] = StringVector::AddStringOrBlob(tweak_key_result, local_state.output_tweak_keys[local_state.output_position + i]);
		}

		output.SetCardinality(output_count);
		local_state.output_position += output_count;

		if (HasOutput(local_state)) {
			return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
		}

		// Clear output buffers after returning all data
		local_state.output_txids.clear();
		local_state.output_heights.clear();
		local_state.output_tweak_keys.clear();
		local_state.output_position = 0;
	}

	// Decrement thread counter only once per thread
	if (!local_state.finalized) {
		state.finalize_lock->lock();
		state.currently_adding--;
		local_state.finalized = true;
		state.finalize_lock->unlock();
	}

	// Process any remaining accumulated data for this thread
	if (!local_state.accumulated_txids.empty()) {
		ProcessBatch(local_state, bind_data);
		local_state.output_position = 0;

		auto &txid_result = output.data[0];
		auto &height_result = output.data[1];
		auto &tweak_key_result = output.data[2];

		idx_t output_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, local_state.output_txids.size());

		auto txid_data = FlatVector::GetData<string_t>(txid_result);
		auto height_data = FlatVector::GetData<int32_t>(height_result);
		auto tweak_key_data = FlatVector::GetData<string_t>(tweak_key_result);

		for (idx_t i = 0; i < output_count; i++) {
			txid_data[i] = StringVector::AddStringOrBlob(txid_result, local_state.output_txids[i]);
			height_data[i] = local_state.output_heights[i];
			tweak_key_data[i] = StringVector::AddStringOrBlob(tweak_key_result, local_state.output_tweak_keys[i]);
		}

		output.SetCardinality(output_count);
		local_state.output_position = output_count;

		if (HasOutput(local_state)) {
			return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
		}
	}

	return OperatorFinalizeResultType::FINISHED;
}

static void LoadInternal(ExtensionLoader &loader) {
	TableFunctionSet cudasp_scan("cudasp_scan");

	// Function with 4 parameters (default batch_size)
	TableFunction func4({LogicalType::TABLE, LogicalType::BLOB, LogicalType::BLOB, LogicalType::LIST(LogicalType::BLOB)}, nullptr, CudaspScanBind, CudaspScanInit, CudaspScanLocalInit);
	func4.in_out_function = CudaspScanFunction;
	func4.in_out_function_final = CudaspScanFinalFunction;
	cudasp_scan.AddFunction(func4);

	// Function with 5 parameters (custom batch_size)
	TableFunction func5({LogicalType::TABLE, LogicalType::BLOB, LogicalType::BLOB, LogicalType::LIST(LogicalType::BLOB), LogicalType::INTEGER}, nullptr, CudaspScanBind, CudaspScanInit, CudaspScanLocalInit);
	func5.in_out_function = CudaspScanFunction;
	func5.in_out_function_final = CudaspScanFinalFunction;
	cudasp_scan.AddFunction(func5);

	loader.RegisterFunction(cudasp_scan);
}

void CudaspExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string CudaspExtension::Name() {
	return "cudasp";
}

std::string CudaspExtension::Version() const {
#ifdef EXT_VERSION_CUDASP
	return EXT_VERSION_CUDASP;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(cudasp, loader) {
	duckdb::LoadInternal(loader);
}
}

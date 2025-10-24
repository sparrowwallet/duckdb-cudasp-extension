#define DUCKDB_EXTENSION_MAIN

#include "cudasp_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include <duckdb/parser/parsed_data/create_scalar_function_info.hpp>

// OpenSSL linked through vcpkg
#include <openssl/opensslv.h>

namespace duckdb {

struct CudaspScanBindData : public TableFunctionData {
	CudaspScanBindData() {
	}
	static constexpr idx_t BATCH_SIZE = 10000; // Accumulate 10K rows before processing
	static constexpr idx_t TWEAK_KEY_SIZE = 64; // 64 bytes: uncompressed EC point (32-byte x || 32-byte y)
	static constexpr idx_t SCALAR_SIZE = 32; // 32 bytes: scalar for EC multiplication

	// Scalar parameter for EC multiplication (shared across all rows)
	string_t scan_private_key;
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
	// TODO: GPU processing with gECC will happen here
	// Steps:
	// 1. Copy tweak_keys to GPU memory (column-major format for gECC)
	// 2. Copy scalar to GPU memory (shared across all rows)
	// 3. Perform batch EC point multiplication: result_point[i] = scalar * tweak_key[i]
	// 4. Extract int64 from each result_point
	// 5. Copy outputs to GPU memory
	// 6. Compare extracted int64 against outputs[i] list on GPU
	// 7. Copy matching row indices back to CPU
	// 8. Build output vectors for matching rows

	// Clear any previous output
	local_state.output_txids.clear();
	local_state.output_heights.clear();
	local_state.output_tweak_keys.clear();
	local_state.output_position = 0;

	idx_t batch_size = local_state.accumulated_txids.size();

	// CPU placeholder implementation (will be replaced with GPU code)
	for (idx_t i = 0; i < batch_size; i++) {
		// TODO: Replace with GPU EC multiplication and int64 extraction
		// For now, use a dummy value
		int64_t computed_value = 12345; // Placeholder for: extract_int64(scalar * tweak_key[i])

		// Check if computed value is in the outputs list
		bool found = false;
		idx_t outputs_offset = local_state.accumulated_output_offsets[i];
		idx_t outputs_length = local_state.accumulated_output_lengths[i];
		for (idx_t j = 0; j < outputs_length; j++) {
			if (local_state.accumulated_outputs[outputs_offset + j] == computed_value) {
				found = true;
				break;
			}
		}

		// Only output rows where a match was found
		if (found) {
			local_state.output_txids.push_back(local_state.accumulated_txids[i]);
			local_state.output_heights.push_back(local_state.accumulated_heights[i]);
			local_state.output_tweak_keys.push_back(local_state.accumulated_tweak_keys[i]);
		}
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

static bool ShouldProcessBatch(const CudaspScanLocalState &local_state) {
	return local_state.accumulated_txids.size() >= CudaspScanBindData::BATCH_SIZE;
}

static unique_ptr<FunctionData> CudaspScanBind(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<string> &names) {
	// Validate input: expects TABLE and scan_private_key BLOB
	if (input.inputs.size() != 2) {
		throw InvalidInputException("cudasp_scan requires 2 arguments: TABLE and scan_private_key BLOB");
	}

	// Validate scan_private_key parameter
	auto &scalar_value = input.inputs[1];
	if (scalar_value.type().id() != LogicalTypeId::BLOB) {
		throw InvalidInputException("Second argument must be a BLOB (32-byte scan_private_key)");
	}

	// Get the scan_private_key blob using GetValueUnsafe
	string_t scan_private_key = StringValue::Get(scalar_value);
	if (scan_private_key.GetSize() != CudaspScanBindData::SCALAR_SIZE) {
		throw InvalidInputException("scan_private_key must be exactly 32 bytes, got %llu bytes", scan_private_key.GetSize());
	}

	// Set return types: txid (BLOB), height (INTEGER), tweak_key (BLOB)
	return_types.push_back(LogicalType::BLOB);    // txid
	return_types.push_back(LogicalType::INTEGER); // height
	return_types.push_back(LogicalType::BLOB);    // tweak_key
	names.push_back("txid");
	names.push_back("height");
	names.push_back("tweak_key");

	auto bind_data = make_uniq<CudaspScanBindData>();
	bind_data->scan_private_key = scan_private_key;
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
		if (ShouldProcessBatch(local_state)) {
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

	TableFunction func({LogicalType::TABLE, LogicalType::BLOB}, nullptr, CudaspScanBind, CudaspScanInit, CudaspScanLocalInit);
	func.in_out_function = CudaspScanFunction;
	func.in_out_function_final = CudaspScanFinalFunction;

	cudasp_scan.AddFunction(func);
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

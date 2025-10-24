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
};

struct CudaspScanLocalState : public LocalTableFunctionState {
	CudaspScanLocalState() : finalized(false), output_position(0) {
	}
	bool finalized;

	// Per-thread accumulated input data
	vector<double> accumulated_values;
	vector<double> accumulated_list_values;  // Flattened list values
	vector<idx_t> accumulated_list_offsets;  // Offset into accumulated_list_values for each row
	vector<idx_t> accumulated_list_lengths;  // Length of each list

	// Per-thread processed output data
	vector<double> output_multiplied;
	vector<uint8_t> output_is_in_list;  // Use uint8_t instead of bool to avoid vector<bool> issues
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
	auto &value_column = input.data[0];
	auto &list_column = input.data[1];

	// Get unified vector format for input columns
	UnifiedVectorFormat value_data;
	UnifiedVectorFormat list_data;
	value_column.ToUnifiedFormat(count, value_data);
	list_column.ToUnifiedFormat(count, list_data);

	auto value_ptr = UnifiedVectorFormat::GetData<double>(value_data);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_data);

	// Get list child data
	auto &list_child = ListVector::GetEntry(list_column);
	UnifiedVectorFormat list_child_data;
	list_child.ToUnifiedFormat(ListVector::GetListSize(list_column), list_child_data);
	auto list_child_ptr = UnifiedVectorFormat::GetData<double>(list_child_data);

	// Accumulate the data
	for (idx_t i = 0; i < count; i++) {
		auto value_idx = value_data.sel->get_index(i);
		auto list_idx = list_data.sel->get_index(i);

		if (value_data.validity.RowIsValid(value_idx)) {
			local_state.accumulated_values.push_back(value_ptr[value_idx]);

			// Store list offset and length
			idx_t list_offset = local_state.accumulated_list_values.size();
			local_state.accumulated_list_offsets.push_back(list_offset);

			idx_t list_len = 0;
			if (list_data.validity.RowIsValid(list_idx)) {
				auto &list_entry = list_entries[list_idx];
				for (idx_t list_i = 0; list_i < list_entry.length; list_i++) {
					auto child_idx = list_child_data.sel->get_index(list_entry.offset + list_i);
					if (list_child_data.validity.RowIsValid(child_idx)) {
						local_state.accumulated_list_values.push_back(list_child_ptr[child_idx]);
						list_len++;
					}
				}
			}
			local_state.accumulated_list_lengths.push_back(list_len);
		}
	}
}

static void ProcessBatch(CudaspScanLocalState &local_state) {
	// This is where GPU processing would happen
	// For now, we process on CPU

	// Clear any previous output
	local_state.output_multiplied.clear();
	local_state.output_is_in_list.clear();
	local_state.output_position = 0;

	idx_t batch_size = local_state.accumulated_values.size();
	local_state.output_multiplied.reserve(batch_size);
	local_state.output_is_in_list.reserve(batch_size);

	for (idx_t i = 0; i < batch_size; i++) {
		double value = local_state.accumulated_values[i];
		double multiplied = value * 2.0;

		// Check if multiplied value is in the list
		bool found = false;
		idx_t list_offset = local_state.accumulated_list_offsets[i];
		idx_t list_length = local_state.accumulated_list_lengths[i];
		for (idx_t j = 0; j < list_length; j++) {
			if (local_state.accumulated_list_values[list_offset + j] == multiplied) {
				found = true;
				break;
			}
		}

		// Only output rows where the value was found in the list
		if (found) {
			local_state.output_multiplied.push_back(multiplied);
			local_state.output_is_in_list.push_back(1);  // Always true for returned rows
		}
	}

	// Clear accumulated input after processing
	local_state.accumulated_values.clear();
	local_state.accumulated_list_values.clear();
	local_state.accumulated_list_offsets.clear();
	local_state.accumulated_list_lengths.clear();
}

static bool HasOutput(const CudaspScanLocalState &local_state) {
	return local_state.output_position < local_state.output_multiplied.size();
}

static bool ShouldProcessBatch(const CudaspScanLocalState &local_state) {
	return local_state.accumulated_values.size() >= CudaspScanBindData::BATCH_SIZE;
}

static unique_ptr<FunctionData> CudaspScanBind(ClientContext &context, TableFunctionBindInput &input,
                                                      vector<LogicalType> &return_types, vector<string> &names) {
	return_types.push_back(LogicalType::DOUBLE);
	return_types.push_back(LogicalType::BOOLEAN);
	names.push_back("multiplied_value");
	names.push_back("is_in_list");

	return make_uniq<CudaspScanBindData>();
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
	auto &local_state = data_p.local_state->Cast<CudaspScanLocalState>();

	// If we have pending output from a previous batch, return it first
	if (HasOutput(local_state)) {
		auto &multiplied_result = output.data[0];
		auto &is_in_list_result = output.data[1];

		auto multiplied_data = FlatVector::GetData<double>(multiplied_result);
		auto is_in_list_data = FlatVector::GetData<bool>(is_in_list_result);

		idx_t output_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE,
		                                       local_state.output_multiplied.size() - local_state.output_position);

		for (idx_t i = 0; i < output_count; i++) {
			multiplied_data[i] = local_state.output_multiplied[local_state.output_position + i];
			is_in_list_data[i] = local_state.output_is_in_list[local_state.output_position + i];
		}

		output.SetCardinality(output_count);
		local_state.output_position += output_count;

		// If we still have more output, signal that
		if (HasOutput(local_state)) {
			return OperatorResultType::HAVE_MORE_OUTPUT;
		}

		// All output returned, clear buffers
		local_state.output_multiplied.clear();
		local_state.output_is_in_list.clear();
		local_state.output_position = 0;

		// Otherwise keep accepting input
		return OperatorResultType::NEED_MORE_INPUT;
	}

	// Process new input
	if (input.size() > 0) {
		AccumulateInput(local_state, input);

		// Process batch if we've accumulated enough data
		if (ShouldProcessBatch(local_state)) {
			ProcessBatch(local_state);
			local_state.output_position = 0;

			// Write output immediately
			auto &multiplied_result = output.data[0];
			auto &is_in_list_result = output.data[1];

			auto multiplied_data = FlatVector::GetData<double>(multiplied_result);
			auto is_in_list_data = FlatVector::GetData<bool>(is_in_list_result);

			idx_t output_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, local_state.output_multiplied.size());

			for (idx_t i = 0; i < output_count; i++) {
				multiplied_data[i] = local_state.output_multiplied[i];
				is_in_list_data[i] = local_state.output_is_in_list[i];
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
	auto &state = data_p.global_state->Cast<CudaspScanState>();
	auto &local_state = data_p.local_state->Cast<CudaspScanLocalState>();

	// If we still have pending output from previous batch, return it
	if (HasOutput(local_state)) {
		auto &multiplied_result = output.data[0];
		auto &is_in_list_result = output.data[1];

		auto multiplied_data = FlatVector::GetData<double>(multiplied_result);
		auto is_in_list_data = FlatVector::GetData<bool>(is_in_list_result);

		idx_t output_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE,
		                                       local_state.output_multiplied.size() - local_state.output_position);

		for (idx_t i = 0; i < output_count; i++) {
			multiplied_data[i] = local_state.output_multiplied[local_state.output_position + i];
			is_in_list_data[i] = local_state.output_is_in_list[local_state.output_position + i];
		}

		output.SetCardinality(output_count);
		local_state.output_position += output_count;

		if (HasOutput(local_state)) {
			return OperatorFinalizeResultType::HAVE_MORE_OUTPUT;
		}

		// Clear output buffers after returning all data
		local_state.output_multiplied.clear();
		local_state.output_is_in_list.clear();
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
	if (!local_state.accumulated_values.empty()) {
		ProcessBatch(local_state);
		local_state.output_position = 0;

		auto &multiplied_result = output.data[0];
		auto &is_in_list_result = output.data[1];

		auto multiplied_data = FlatVector::GetData<double>(multiplied_result);
		auto is_in_list_data = FlatVector::GetData<bool>(is_in_list_result);

		idx_t output_count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, local_state.output_multiplied.size());

		for (idx_t i = 0; i < output_count; i++) {
			multiplied_data[i] = local_state.output_multiplied[i];
			is_in_list_data[i] = local_state.output_is_in_list[i];
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

	TableFunction func({LogicalType::TABLE}, nullptr, CudaspScanBind, CudaspScanInit, CudaspScanLocalInit);
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

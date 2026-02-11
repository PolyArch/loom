//===-- FabricError.h - Centralized Fabric error codes ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Single source of truth for all COMP_ (compile-time) error code symbols.
// Other files that emit COMP_ errors should include this header and use these
// constants instead of inline string literals.
//
// CFG_ and RT_ (hardware runtime) error codes are defined in:
//   lib/loom/Hardware/SystemVerilog/Common/fabric_error.svh
//
// The normative specification for all error codes is:
//   docs/spec-fabric-error.md
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_HARDWARE_COMMON_FABRICERROR_H
#define LOOM_HARDWARE_COMMON_FABRICERROR_H

#include <string>

namespace loom {

/// Centralized compile-time error code constants.
///
/// These match the error code strings used in diagnostic messages. Each constant
/// is used as either:
///   - The `code` field of a ValidationError (ADGBuilderValidation.cpp)
///   - The bracketed prefix in emitOpError("[COMP_XXX] ...") (Fabric dialect)
///
/// Error codes are grouped by the module/subsystem they relate to.
namespace CompError {

// --- Switch Errors ---
inline constexpr const char *SWITCH_TABLE_SHAPE =
    "COMP_SWITCH_TABLE_SHAPE";
inline constexpr const char *SWITCH_ROW_EMPTY =
    "COMP_SWITCH_ROW_EMPTY";
inline constexpr const char *SWITCH_COL_EMPTY =
    "COMP_SWITCH_COL_EMPTY";
inline constexpr const char *SWITCH_PORT_ZERO =
    "COMP_SWITCH_PORT_ZERO";
inline constexpr const char *SWITCH_PORT_LIMIT =
    "COMP_SWITCH_PORT_LIMIT";
inline constexpr const char *SWITCH_ROUTE_LEN_MISMATCH =
    "COMP_SWITCH_ROUTE_LEN_MISMATCH";

// --- Temporal Switch Errors ---
inline constexpr const char *TEMPORAL_SW_TABLE_SHAPE =
    "COMP_TEMPORAL_SW_TABLE_SHAPE";
inline constexpr const char *TEMPORAL_SW_ROW_EMPTY =
    "COMP_TEMPORAL_SW_ROW_EMPTY";
inline constexpr const char *TEMPORAL_SW_COL_EMPTY =
    "COMP_TEMPORAL_SW_COL_EMPTY";
inline constexpr const char *TEMPORAL_SW_PORT_ZERO =
    "COMP_TEMPORAL_SW_PORT_ZERO";
inline constexpr const char *TEMPORAL_SW_PORT_LIMIT =
    "COMP_TEMPORAL_SW_PORT_LIMIT";
inline constexpr const char *TEMPORAL_SW_NUM_ROUTE_TABLE =
    "COMP_TEMPORAL_SW_NUM_ROUTE_TABLE";
inline constexpr const char *TEMPORAL_SW_INTERFACE_NOT_TAGGED =
    "COMP_TEMPORAL_SW_INTERFACE_NOT_TAGGED";
inline constexpr const char *TEMPORAL_SW_TOO_MANY_SLOTS =
    "COMP_TEMPORAL_SW_TOO_MANY_SLOTS";
inline constexpr const char *TEMPORAL_SW_ROUTE_ILLEGAL =
    "COMP_TEMPORAL_SW_ROUTE_ILLEGAL";
inline constexpr const char *TEMPORAL_SW_MIXED_FORMAT =
    "COMP_TEMPORAL_SW_MIXED_FORMAT";
inline constexpr const char *TEMPORAL_SW_SLOT_ORDER =
    "COMP_TEMPORAL_SW_SLOT_ORDER";
inline constexpr const char *TEMPORAL_SW_IMPLICIT_HOLE =
    "COMP_TEMPORAL_SW_IMPLICIT_HOLE";

// --- PE Errors ---
inline constexpr const char *PE_EMPTY_BODY =
    "COMP_PE_EMPTY_BODY";
inline constexpr const char *PE_MIXED_INTERFACE =
    "COMP_PE_MIXED_INTERFACE";
inline constexpr const char *PE_TAGGED_INTERFACE_NATIVE_PORTS =
    "COMP_PE_TAGGED_INTERFACE_NATIVE_PORTS";
inline constexpr const char *PE_NATIVE_INTERFACE_TAGGED_PORTS =
    "COMP_PE_NATIVE_INTERFACE_TAGGED_PORTS";
inline constexpr const char *PE_OUTPUT_TAG_NATIVE =
    "COMP_PE_OUTPUT_TAG_NATIVE";
inline constexpr const char *PE_OUTPUT_TAG_MISSING =
    "COMP_PE_OUTPUT_TAG_MISSING";
inline constexpr const char *PE_LOADSTORE_BODY =
    "COMP_PE_LOADSTORE_BODY";
inline constexpr const char *PE_LOADSTORE_TAG_MODE =
    "COMP_PE_LOADSTORE_TAG_MODE";
inline constexpr const char *PE_LOADSTORE_TAG_WIDTH =
    "COMP_PE_LOADSTORE_TAG_WIDTH";
inline constexpr const char *PE_CONSTANT_BODY =
    "COMP_PE_CONSTANT_BODY";
inline constexpr const char *PE_INSTANCE_ONLY_BODY =
    "COMP_PE_INSTANCE_ONLY_BODY";
inline constexpr const char *PE_INSTANCE_ILLEGAL_TARGET =
    "COMP_PE_INSTANCE_ILLEGAL_TARGET";
inline constexpr const char *PE_DATAFLOW_BODY =
    "COMP_PE_DATAFLOW_BODY";
inline constexpr const char *PE_MIXED_CONSUMPTION =
    "COMP_PE_MIXED_CONSUMPTION";

// --- Temporal PE Errors ---
inline constexpr const char *TEMPORAL_PE_INTERFACE_NOT_TAGGED =
    "COMP_TEMPORAL_PE_INTERFACE_NOT_TAGGED";
inline constexpr const char *TEMPORAL_PE_NUM_INSTRUCTION =
    "COMP_TEMPORAL_PE_NUM_INSTRUCTION";
inline constexpr const char *TEMPORAL_PE_REG_FIFO_DEPTH =
    "COMP_TEMPORAL_PE_REG_FIFO_DEPTH";
inline constexpr const char *TEMPORAL_PE_EMPTY_BODY =
    "COMP_TEMPORAL_PE_EMPTY_BODY";
inline constexpr const char *TEMPORAL_PE_FU_INVALID =
    "COMP_TEMPORAL_PE_FU_INVALID";
inline constexpr const char *TEMPORAL_PE_TAGGED_FU =
    "COMP_TEMPORAL_PE_TAGGED_FU";
inline constexpr const char *TEMPORAL_PE_FU_ARITY =
    "COMP_TEMPORAL_PE_FU_ARITY";
inline constexpr const char *TEMPORAL_PE_FU_WIDTH =
    "COMP_TEMPORAL_PE_FU_WIDTH";
inline constexpr const char *TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE =
    "COMP_TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE";
inline constexpr const char *TEMPORAL_PE_OPERAND_BUFFER_SIZE_MISSING =
    "COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_MISSING";
inline constexpr const char *TEMPORAL_PE_OPERAND_BUFFER_SIZE_RANGE =
    "COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_RANGE";
inline constexpr const char *TEMPORAL_PE_TAG_WIDTH =
    "COMP_TEMPORAL_PE_TAG_WIDTH";
inline constexpr const char *TEMPORAL_PE_TAGGED_PE =
    "COMP_TEMPORAL_PE_TAGGED_PE";
inline constexpr const char *TEMPORAL_PE_LOADSTORE =
    "COMP_TEMPORAL_PE_LOADSTORE";
inline constexpr const char *TEMPORAL_PE_DATAFLOW_INVALID =
    "COMP_TEMPORAL_PE_DATAFLOW_INVALID";
inline constexpr const char *TEMPORAL_PE_TOO_MANY_SLOTS =
    "COMP_TEMPORAL_PE_TOO_MANY_SLOTS";
inline constexpr const char *TEMPORAL_PE_MIXED_FORMAT =
    "COMP_TEMPORAL_PE_MIXED_FORMAT";
inline constexpr const char *TEMPORAL_PE_SLOT_ORDER =
    "COMP_TEMPORAL_PE_SLOT_ORDER";
inline constexpr const char *TEMPORAL_PE_IMPLICIT_HOLE =
    "COMP_TEMPORAL_PE_IMPLICIT_HOLE";
inline constexpr const char *TEMPORAL_PE_REG_DISABLED =
    "COMP_TEMPORAL_PE_REG_DISABLED";
inline constexpr const char *TEMPORAL_PE_DEST_COUNT =
    "COMP_TEMPORAL_PE_DEST_COUNT";
inline constexpr const char *TEMPORAL_PE_SRC_COUNT =
    "COMP_TEMPORAL_PE_SRC_COUNT";
inline constexpr const char *TEMPORAL_PE_SRC_MISMATCH =
    "COMP_TEMPORAL_PE_SRC_MISMATCH";

// --- Load/Store PE Errors ---
inline constexpr const char *LOADPE_TRANSPARENT_NATIVE =
    "COMP_LOADPE_TRANSPARENT_NATIVE";
inline constexpr const char *LOADPE_TRANSPARENT_QUEUE_DEPTH =
    "COMP_LOADPE_TRANSPARENT_QUEUE_DEPTH";
inline constexpr const char *STOREPE_TRANSPARENT_NATIVE =
    "COMP_STOREPE_TRANSPARENT_NATIVE";
inline constexpr const char *STOREPE_TRANSPARENT_QUEUE_DEPTH =
    "COMP_STOREPE_TRANSPARENT_QUEUE_DEPTH";

// --- Memory Errors ---
inline constexpr const char *MEMORY_PORTS_EMPTY =
    "COMP_MEMORY_PORTS_EMPTY";
inline constexpr const char *MEMORY_LSQ_MIN =
    "COMP_MEMORY_LSQ_MIN";
inline constexpr const char *MEMORY_LSQ_WITHOUT_STORE =
    "COMP_MEMORY_LSQ_WITHOUT_STORE";
inline constexpr const char *MEMORY_ADDR_TYPE =
    "COMP_MEMORY_ADDR_TYPE";
inline constexpr const char *MEMORY_DATA_TYPE =
    "COMP_MEMORY_DATA_TYPE";
inline constexpr const char *MEMORY_TAG_REQUIRED =
    "COMP_MEMORY_TAG_REQUIRED";
inline constexpr const char *MEMORY_TAG_FOR_SINGLE =
    "COMP_MEMORY_TAG_FOR_SINGLE";
inline constexpr const char *MEMORY_TAG_WIDTH =
    "COMP_MEMORY_TAG_WIDTH";
inline constexpr const char *MEMORY_STATIC_REQUIRED =
    "COMP_MEMORY_STATIC_REQUIRED";
inline constexpr const char *MEMORY_PRIVATE_OUTPUT =
    "COMP_MEMORY_PRIVATE_OUTPUT";
inline constexpr const char *MEMORY_EXTMEM_BINDING =
    "COMP_MEMORY_EXTMEM_BINDING";
inline constexpr const char *MEMORY_EXTMEM_PRIVATE =
    "COMP_MEMORY_EXTMEM_PRIVATE";

// --- Tag Errors ---
inline constexpr const char *TAG_WIDTH_RANGE =
    "COMP_TAG_WIDTH_RANGE";
inline constexpr const char *ADD_TAG_VALUE_TYPE_MISMATCH =
    "COMP_ADD_TAG_VALUE_TYPE_MISMATCH";
inline constexpr const char *ADD_TAG_VALUE_OVERFLOW =
    "COMP_ADD_TAG_VALUE_OVERFLOW";
inline constexpr const char *DEL_TAG_VALUE_TYPE_MISMATCH =
    "COMP_DEL_TAG_VALUE_TYPE_MISMATCH";
inline constexpr const char *MAP_TAG_VALUE_TYPE_MISMATCH =
    "COMP_MAP_TAG_VALUE_TYPE_MISMATCH";
inline constexpr const char *MAP_TAG_TABLE_SIZE =
    "COMP_MAP_TAG_TABLE_SIZE";
inline constexpr const char *MAP_TAG_TABLE_LENGTH =
    "COMP_MAP_TAG_TABLE_LENGTH";

// --- FIFO Errors ---
inline constexpr const char *FIFO_DEPTH_ZERO =
    "COMP_FIFO_DEPTH_ZERO";
inline constexpr const char *FIFO_TYPE_MISMATCH =
    "COMP_FIFO_TYPE_MISMATCH";
inline constexpr const char *FIFO_INVALID_TYPE =
    "COMP_FIFO_INVALID_TYPE";
inline constexpr const char *FIFO_BYPASSED_NOT_BYPASSABLE =
    "COMP_FIFO_BYPASSED_NOT_BYPASSABLE";
inline constexpr const char *FIFO_BYPASSED_MISSING =
    "COMP_FIFO_BYPASSED_MISSING";

// --- Module/Fabric Errors ---
inline constexpr const char *MODULE_PORT_ORDER =
    "COMP_MODULE_PORT_ORDER";
inline constexpr const char *MODULE_EMPTY_BODY =
    "COMP_MODULE_EMPTY_BODY";
inline constexpr const char *MODULE_MISSING_YIELD =
    "COMP_MODULE_MISSING_YIELD";
inline constexpr const char *FABRIC_TYPE_MISMATCH =
    "COMP_FABRIC_TYPE_MISMATCH";

// --- Instance Errors ---
inline constexpr const char *INSTANCE_UNRESOLVED =
    "COMP_INSTANCE_UNRESOLVED";
inline constexpr const char *INSTANCE_OPERAND_MISMATCH =
    "COMP_INSTANCE_OPERAND_MISMATCH";
inline constexpr const char *INSTANCE_RESULT_MISMATCH =
    "COMP_INSTANCE_RESULT_MISMATCH";
inline constexpr const char *INSTANCE_CYCLIC_REFERENCE =
    "COMP_INSTANCE_CYCLIC_REFERENCE";

// --- Connection Errors ---
inline constexpr const char *FANOUT_MODULE_INNER =
    "COMP_FANOUT_MODULE_INNER";
inline constexpr const char *FANOUT_MODULE_BOUNDARY =
    "COMP_FANOUT_MODULE_BOUNDARY";
inline constexpr const char *OUTPUT_UNCONNECTED =
    "COMP_OUTPUT_UNCONNECTED";
inline constexpr const char *OUTPUT_DANGLING =
    "COMP_OUTPUT_DANGLING";
inline constexpr const char *INPUT_UNCONNECTED =
    "COMP_INPUT_UNCONNECTED";
inline constexpr const char *MULTI_DRIVER =
    "COMP_MULTI_DRIVER";
inline constexpr const char *ADG_COMBINATIONAL_LOOP =
    "COMP_ADG_COMBINATIONAL_LOOP";

// --- Handshake Conversion Errors ---
inline constexpr const char *HANDSHAKE_CTRL_MULTI_MEM =
    "COMP_HANDSHAKE_CTRL_MULTI_MEM";

} // namespace CompError

/// Format a bracketed error prefix: "[COMP_FOO] msg"
inline std::string compErrMsg(const char *code, const char *msg) {
  return std::string("[") + code + "] " + msg;
}

/// Format a bracketed error code: "[COMP_FOO]"
inline std::string compErrCode(const char *code) {
  return std::string("[") + code + "]";
}

} // namespace loom

#endif // LOOM_HARDWARE_COMMON_FABRICERROR_H

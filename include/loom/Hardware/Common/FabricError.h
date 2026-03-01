//===-- FabricError.h - Centralized Fabric error codes ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Single source of truth for all CPL_ (compile-time) error code symbols.
// Other files that emit CPL_ errors should include this header and use these
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
///   - The bracketed prefix in emitOpError("[CPL_XXX] ...") (Fabric dialect)
///
/// Error codes are grouped by the module/subsystem they relate to.
namespace CplError {

// --- Switch Errors ---
inline constexpr const char *SWITCH_TABLE_SHAPE =
    "CPL_SWITCH_TABLE_SHAPE";
inline constexpr const char *SWITCH_ROW_EMPTY =
    "CPL_SWITCH_ROW_EMPTY";
inline constexpr const char *SWITCH_COL_EMPTY =
    "CPL_SWITCH_COL_EMPTY";
inline constexpr const char *SWITCH_PORT_ZERO =
    "CPL_SWITCH_PORT_ZERO";
inline constexpr const char *SWITCH_PORT_LIMIT =
    "CPL_SWITCH_PORT_LIMIT";
inline constexpr const char *SWITCH_ROUTE_LEN_MISMATCH =
    "CPL_SWITCH_ROUTE_LEN_MISMATCH";

// --- Temporal Switch Errors ---
inline constexpr const char *TEMPORAL_SW_TABLE_SHAPE =
    "CPL_TEMPORAL_SW_TABLE_SHAPE";
inline constexpr const char *TEMPORAL_SW_ROW_EMPTY =
    "CPL_TEMPORAL_SW_ROW_EMPTY";
inline constexpr const char *TEMPORAL_SW_COL_EMPTY =
    "CPL_TEMPORAL_SW_COL_EMPTY";
inline constexpr const char *TEMPORAL_SW_PORT_ZERO =
    "CPL_TEMPORAL_SW_PORT_ZERO";
inline constexpr const char *TEMPORAL_SW_PORT_LIMIT =
    "CPL_TEMPORAL_SW_PORT_LIMIT";
inline constexpr const char *TEMPORAL_SW_NUM_ROUTE_TABLE =
    "CPL_TEMPORAL_SW_NUM_ROUTE_TABLE";
inline constexpr const char *TEMPORAL_SW_INTERFACE_NOT_TAGGED =
    "CPL_TEMPORAL_SW_INTERFACE_NOT_TAGGED";
inline constexpr const char *TEMPORAL_SW_TOO_MANY_SLOTS =
    "CPL_TEMPORAL_SW_TOO_MANY_SLOTS";
inline constexpr const char *TEMPORAL_SW_ROUTE_ILLEGAL =
    "CPL_TEMPORAL_SW_ROUTE_ILLEGAL";
inline constexpr const char *TEMPORAL_SW_MIXED_FORMAT =
    "CPL_TEMPORAL_SW_MIXED_FORMAT";
inline constexpr const char *TEMPORAL_SW_SLOT_ORDER =
    "CPL_TEMPORAL_SW_SLOT_ORDER";
inline constexpr const char *TEMPORAL_SW_IMPLICIT_HOLE =
    "CPL_TEMPORAL_SW_IMPLICIT_HOLE";

// --- PE Errors ---
inline constexpr const char *PE_EMPTY_BODY =
    "CPL_PE_EMPTY_BODY";
inline constexpr const char *PE_MIXED_INTERFACE =
    "CPL_PE_MIXED_INTERFACE";
inline constexpr const char *PE_TAGGED_INTERFACE_NATIVE_PORTS =
    "CPL_PE_TAGGED_INTERFACE_NATIVE_PORTS";
inline constexpr const char *PE_NATIVE_INTERFACE_TAGGED_PORTS =
    "CPL_PE_NATIVE_INTERFACE_TAGGED_PORTS";
inline constexpr const char *PE_OUTPUT_TAG_NATIVE =
    "CPL_PE_OUTPUT_TAG_NATIVE";
inline constexpr const char *PE_OUTPUT_TAG_MISSING =
    "CPL_PE_OUTPUT_TAG_MISSING";
inline constexpr const char *PE_LOADSTORE_BODY =
    "CPL_PE_LOADSTORE_BODY";
inline constexpr const char *PE_LOADSTORE_TAG_MODE =
    "CPL_PE_LOADSTORE_TAG_MODE";
inline constexpr const char *PE_LOADSTORE_TAG_WIDTH =
    "CPL_PE_LOADSTORE_TAG_WIDTH";
inline constexpr const char *PE_CONSTANT_BODY =
    "CPL_PE_CONSTANT_BODY";
inline constexpr const char *PE_INSTANCE_ONLY_BODY =
    "CPL_PE_INSTANCE_ONLY_BODY";
inline constexpr const char *PE_INSTANCE_ILLEGAL_TARGET =
    "CPL_PE_INSTANCE_ILLEGAL_TARGET";
inline constexpr const char *PE_DATAFLOW_BODY =
    "CPL_PE_DATAFLOW_BODY";
inline constexpr const char *PE_MIXED_CONSUMPTION =
    "CPL_PE_MIXED_CONSUMPTION";

// --- Temporal PE Errors ---
inline constexpr const char *TEMPORAL_PE_INTERFACE_NOT_TAGGED =
    "CPL_TEMPORAL_PE_INTERFACE_NOT_TAGGED";
inline constexpr const char *TEMPORAL_PE_NUM_INSTRUCTION =
    "CPL_TEMPORAL_PE_NUM_INSTRUCTION";
inline constexpr const char *TEMPORAL_PE_REG_FIFO_DEPTH =
    "CPL_TEMPORAL_PE_REG_FIFO_DEPTH";
inline constexpr const char *TEMPORAL_PE_EMPTY_BODY =
    "CPL_TEMPORAL_PE_EMPTY_BODY";
inline constexpr const char *TEMPORAL_PE_FU_INVALID =
    "CPL_TEMPORAL_PE_FU_INVALID";
inline constexpr const char *TEMPORAL_PE_TAGGED_FU =
    "CPL_TEMPORAL_PE_TAGGED_FU";
inline constexpr const char *TEMPORAL_PE_FU_ARITY =
    "CPL_TEMPORAL_PE_FU_ARITY";
inline constexpr const char *TEMPORAL_PE_FU_WIDTH =
    "CPL_TEMPORAL_PE_FU_WIDTH";
inline constexpr const char *TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE =
    "CPL_TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE";
inline constexpr const char *TEMPORAL_PE_OPERAND_BUFFER_SIZE_MISSING =
    "CPL_TEMPORAL_PE_OPERAND_BUFFER_SIZE_MISSING";
inline constexpr const char *TEMPORAL_PE_OPERAND_BUFFER_SIZE_RANGE =
    "CPL_TEMPORAL_PE_OPERAND_BUFFER_SIZE_RANGE";
inline constexpr const char *TEMPORAL_PE_TAG_WIDTH =
    "CPL_TEMPORAL_PE_TAG_WIDTH";
inline constexpr const char *TEMPORAL_PE_TAGGED_PE =
    "CPL_TEMPORAL_PE_TAGGED_PE";
inline constexpr const char *TEMPORAL_PE_LOADSTORE =
    "CPL_TEMPORAL_PE_LOADSTORE";
inline constexpr const char *TEMPORAL_PE_DATAFLOW_INVALID =
    "CPL_TEMPORAL_PE_DATAFLOW_INVALID";
inline constexpr const char *TEMPORAL_PE_TOO_MANY_SLOTS =
    "CPL_TEMPORAL_PE_TOO_MANY_SLOTS";
inline constexpr const char *TEMPORAL_PE_MIXED_FORMAT =
    "CPL_TEMPORAL_PE_MIXED_FORMAT";
inline constexpr const char *TEMPORAL_PE_SLOT_ORDER =
    "CPL_TEMPORAL_PE_SLOT_ORDER";
inline constexpr const char *TEMPORAL_PE_IMPLICIT_HOLE =
    "CPL_TEMPORAL_PE_IMPLICIT_HOLE";
inline constexpr const char *TEMPORAL_PE_REG_DISABLED =
    "CPL_TEMPORAL_PE_REG_DISABLED";
inline constexpr const char *TEMPORAL_PE_DEST_COUNT =
    "CPL_TEMPORAL_PE_DEST_COUNT";
inline constexpr const char *TEMPORAL_PE_SRC_COUNT =
    "CPL_TEMPORAL_PE_SRC_COUNT";
inline constexpr const char *TEMPORAL_PE_SRC_MISMATCH =
    "CPL_TEMPORAL_PE_SRC_MISMATCH";

// --- Load/Store PE Errors ---
inline constexpr const char *LOADPE_TRANSPARENT_NATIVE =
    "CPL_LOADPE_TRANSPARENT_NATIVE";
inline constexpr const char *LOADPE_TRANSPARENT_QUEUE_DEPTH =
    "CPL_LOADPE_TRANSPARENT_QUEUE_DEPTH";
inline constexpr const char *STOREPE_TRANSPARENT_NATIVE =
    "CPL_STOREPE_TRANSPARENT_NATIVE";
inline constexpr const char *STOREPE_TRANSPARENT_QUEUE_DEPTH =
    "CPL_STOREPE_TRANSPARENT_QUEUE_DEPTH";

// --- Memory Errors ---
inline constexpr const char *MEMORY_PORTS_EMPTY =
    "CPL_MEMORY_PORTS_EMPTY";
inline constexpr const char *MEMORY_LSQ_MIN =
    "CPL_MEMORY_LSQ_MIN";
inline constexpr const char *MEMORY_LSQ_WITHOUT_STORE =
    "CPL_MEMORY_LSQ_WITHOUT_STORE";
inline constexpr const char *MEMORY_ADDR_TYPE =
    "CPL_MEMORY_ADDR_TYPE";
inline constexpr const char *MEMORY_DATA_TYPE =
    "CPL_MEMORY_DATA_TYPE";
inline constexpr const char *MEMORY_TAG_REQUIRED =
    "CPL_MEMORY_TAG_REQUIRED";
inline constexpr const char *MEMORY_TAG_FOR_SINGLE =
    "CPL_MEMORY_TAG_FOR_SINGLE";
inline constexpr const char *MEMORY_TAG_WIDTH =
    "CPL_MEMORY_TAG_WIDTH";
inline constexpr const char *MEMORY_STATIC_REQUIRED =
    "CPL_MEMORY_STATIC_REQUIRED";
inline constexpr const char *MEMORY_PRIVATE_OUTPUT =
    "CPL_MEMORY_PRIVATE_OUTPUT";
inline constexpr const char *MEMORY_EXTMEM_BINDING =
    "CPL_MEMORY_EXTMEM_BINDING";
inline constexpr const char *MEMORY_EXTMEM_PRIVATE =
    "CPL_MEMORY_EXTMEM_PRIVATE";
inline constexpr const char *MEMORY_INVALID_REGION =
    "CPL_MEMORY_INVALID_REGION";

// --- Tag Errors ---
inline constexpr const char *TAG_WIDTH_RANGE =
    "CPL_TAG_WIDTH_RANGE";
inline constexpr const char *ADD_TAG_VALUE_TYPE_MISMATCH =
    "CPL_ADD_TAG_VALUE_TYPE_MISMATCH";
inline constexpr const char *ADD_TAG_VALUE_OVERFLOW =
    "CPL_ADD_TAG_VALUE_OVERFLOW";
inline constexpr const char *DEL_TAG_VALUE_TYPE_MISMATCH =
    "CPL_DEL_TAG_VALUE_TYPE_MISMATCH";
inline constexpr const char *MAP_TAG_VALUE_TYPE_MISMATCH =
    "CPL_MAP_TAG_VALUE_TYPE_MISMATCH";
inline constexpr const char *MAP_TAG_TABLE_SIZE =
    "CPL_MAP_TAG_TABLE_SIZE";
inline constexpr const char *MAP_TAG_TABLE_LENGTH =
    "CPL_MAP_TAG_TABLE_LENGTH";

// --- Routing Payload Errors ---
inline constexpr const char *ROUTING_PAYLOAD_NOT_BITS =
    "CPL_ROUTING_PAYLOAD_NOT_BITS";

// --- FIFO Errors ---
inline constexpr const char *FIFO_DEPTH_ZERO =
    "CPL_FIFO_DEPTH_ZERO";
inline constexpr const char *FIFO_TYPE_MISMATCH =
    "CPL_FIFO_TYPE_MISMATCH";
inline constexpr const char *FIFO_INVALID_TYPE =
    "CPL_FIFO_INVALID_TYPE";
inline constexpr const char *FIFO_BYPASSED_NOT_BYPASSABLE =
    "CPL_FIFO_BYPASSED_NOT_BYPASSABLE";
inline constexpr const char *FIFO_BYPASSED_MISSING =
    "CPL_FIFO_BYPASSED_MISSING";

// --- Module/Fabric Errors ---
inline constexpr const char *MODULE_PORT_ORDER =
    "CPL_MODULE_PORT_ORDER";
inline constexpr const char *MODULE_EMPTY_BODY =
    "CPL_MODULE_EMPTY_BODY";
inline constexpr const char *MODULE_MISSING_YIELD =
    "CPL_MODULE_MISSING_YIELD";
inline constexpr const char *FABRIC_TYPE_MISMATCH =
    "CPL_FABRIC_TYPE_MISMATCH";

// --- Instance Errors ---
inline constexpr const char *INSTANCE_UNRESOLVED =
    "CPL_INSTANCE_UNRESOLVED";
inline constexpr const char *INSTANCE_OPERAND_MISMATCH =
    "CPL_INSTANCE_OPERAND_MISMATCH";
inline constexpr const char *INSTANCE_RESULT_MISMATCH =
    "CPL_INSTANCE_RESULT_MISMATCH";
inline constexpr const char *INSTANCE_CYCLIC_REFERENCE =
    "CPL_INSTANCE_CYCLIC_REFERENCE";

// --- Connection Errors ---
inline constexpr const char *FANOUT_MODULE_INNER =
    "CPL_FANOUT_MODULE_INNER";
inline constexpr const char *FANOUT_MODULE_BOUNDARY =
    "CPL_FANOUT_MODULE_BOUNDARY";
inline constexpr const char *OUTPUT_UNCONNECTED =
    "CPL_OUTPUT_UNCONNECTED";
inline constexpr const char *OUTPUT_DANGLING =
    "CPL_OUTPUT_DANGLING";
inline constexpr const char *INPUT_UNCONNECTED =
    "CPL_INPUT_UNCONNECTED";
inline constexpr const char *MULTI_DRIVER =
    "CPL_MULTI_DRIVER";
inline constexpr const char *ADG_COMBINATIONAL_LOOP =
    "CPL_ADG_COMBINATIONAL_LOOP";

// --- Handshake Conversion Errors ---
inline constexpr const char *HANDSHAKE_CTRL_MULTI_MEM =
    "CPL_HANDSHAKE_CTRL_MULTI_MEM";

} // namespace CplError

/// Format a bracketed error prefix: "[CPL_FOO] msg"
inline std::string cplErrMsg(const char *code, const char *msg) {
  return std::string("[") + code + "] " + msg;
}

/// Format a bracketed error code: "[CPL_FOO]"
inline std::string cplErrCode(const char *code) {
  return std::string("[") + code + "]";
}

} // namespace loom

#endif // LOOM_HARDWARE_COMMON_FABRICERROR_H

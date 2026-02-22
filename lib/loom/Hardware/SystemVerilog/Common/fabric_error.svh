//===-- fabric_error.svh - Hardware error code constants -------*- SV -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Single source of truth for all CFG_ and RT_ hardware error codes.
// Included by fabric_common.svh. Other SV modules should not include this
// file directly -- include fabric_common.svh instead.
//
// CPL_ (compile-time) error codes are defined in:
//   include/loom/Hardware/Common/FabricError.h
//
// The normative specification for all error codes is:
//   docs/spec-fabric-error.md
//
//===----------------------------------------------------------------------===//

`ifndef FABRIC_ERROR_SVH
`define FABRIC_ERROR_SVH

//===----------------------------------------------------------------------===//
// CFG_ Errors: Runtime Configuration Errors (Codes 0-255)
//===----------------------------------------------------------------------===//

localparam logic [15:0] FABRIC_OK = 16'd0;

// Switch: multiple inputs route to the same output (fan-in)
localparam logic [15:0] CFG_SWITCH_ROUTE_MIX_INPUTS_TO_SAME_OUTPUT = 16'd1;

// Code 2 reserved (was CFG_SWITCH_ROUTE_MULTI_IN, removed)

// Code 3 reserved (was CFG_TEMPORAL_SW_ROUTE_MULTI_OUT, removed)

// Temporal switch: per-slot fan-in (multiple inputs to same output in one slot)
localparam logic [15:0] CFG_TEMPORAL_SW_ROUTE_SAME_TAG_INPUTS_TO_SAME_OUTPUT = 16'd4;

// Temporal switch: duplicate tags in route_table
localparam logic [15:0] CFG_TEMPORAL_SW_DUP_TAG = 16'd5;

// Temporal PE: duplicate tags in instruction_mem
localparam logic [15:0] CFG_TEMPORAL_PE_DUP_TAG = 16'd6;

// Temporal PE: register index >= num_register
localparam logic [15:0] CFG_TEMPORAL_PE_ILLEGAL_REG = 16'd7;

// Temporal PE: res_tag != 0 when writing a register
localparam logic [15:0] CFG_TEMPORAL_PE_REG_TAG_NONZERO = 16'd8;

// Map tag: duplicate src_tag in table
localparam logic [15:0] CFG_MAP_TAG_DUP_TAG = 16'd9;

// PE stream: cont_cond_sel is not one-hot
localparam logic [15:0] CFG_PE_STREAM_CONT_COND_ONEHOT = 16'd10;

// PE cmpi: predicate value >= 10 (only 0-9 valid for integer compare)
localparam logic [15:0] CFG_PE_CMPI_PREDICATE_INVALID = 16'd11;

// Memory: overlapping tag ranges in addr_offset_table
localparam logic [15:0] CFG_MEMORY_OVERLAP_TAG_REGION = 16'd12;

// Memory: a region has start_tag > end_tag (empty range)
localparam logic [15:0] CFG_MEMORY_EMPTY_TAG_RANGE = 16'd13;

// ExtMemory: overlapping tag ranges in addr_offset_table
localparam logic [15:0] CFG_EXTMEMORY_OVERLAP_TAG_REGION = 16'd14;

// ExtMemory: a region has start_tag > end_tag (empty range)
localparam logic [15:0] CFG_EXTMEMORY_EMPTY_TAG_RANGE = 16'd15;

//===----------------------------------------------------------------------===//
// RT_ Errors: Runtime Execution Errors (Codes 256+)
//===----------------------------------------------------------------------===//

// Temporal PE: input tag matches no instruction
localparam logic [15:0] RT_TEMPORAL_PE_NO_MATCH = 16'd256;

// Temporal switch: input tag matches no route table slot
localparam logic [15:0] RT_TEMPORAL_SW_NO_MATCH = 16'd257;

// Map tag: no matching entry for input tag
localparam logic [15:0] RT_MAP_TAG_NO_MATCH = 16'd258;

// Dataflow stream: step = 0 at runtime
localparam logic [15:0] RT_DATAFLOW_STREAM_ZERO_STEP = 16'd259;

// Memory: tag >= count on a load/store request
localparam logic [15:0] RT_MEMORY_TAG_OOB = 16'd260;

// Memory: store request cannot pair addr+data within timeout
localparam logic [15:0] RT_MEMORY_STORE_DEADLOCK = 16'd261;

// Switch: valid input has no enabled route
localparam logic [15:0] RT_SWITCH_UNROUTED_INPUT = 16'd262;

// Temporal switch: matched slot does not route the input
localparam logic [15:0] RT_TEMPORAL_SW_UNROUTED_INPUT = 16'd263;

// Memory: load/store tag matches no region in addr_offset_table
localparam logic [15:0] RT_MEMORY_NO_MATCH = 16'd264;

// ExtMemory: load/store tag matches no region in addr_offset_table
localparam logic [15:0] RT_EXTMEMORY_NO_MATCH = 16'd265;

`endif // FABRIC_ERROR_SVH

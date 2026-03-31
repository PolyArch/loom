#ifndef FCC_DIALECT_FABRIC_FABRICPEOPS_TD
#define FCC_DIALECT_FABRIC_FABRICPEOPS_TD

include "fcc/Dialect/Fabric/FabricDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"


// fabric.function_unit - computation unit inside a PE
def Fabric_FunctionUnitOp : Fabric_Op<"function_unit", [
    SingleBlock, SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Configurable computation unit";
  let description = [{
    Defines a function unit (FU) that performs computation inside a PE.
    The body may contain multiple ops connected via fabric.mux
    to form a configurable hardware DAG. FU ports use native types.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<I64Attr>:$latency,
    OptionalAttr<I64Attr>:$interval
  );
  let regions = (region SizedRegion<1>:$body);
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// fabric.mux - configurable mux inside function_unit
def Fabric_MuxOp : Fabric_Op<"mux", []> {
  let summary = "Statically-configured runtime-programmable multiplexer";
  let description = [{
    A statically-configured runtime-programmable routing primitive inside
    fabric.function_unit.

    It supports:
    - N-input, 1-output mux
    - 1-input, N-output demux

    Control: sel (log2Ceil(N) bits) + disconnect (1 bit) + discard (1 bit).
    Can ONLY appear inside fabric.function_unit.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    I64Attr:$sel,
    DefaultValuedAttr<BoolAttr, "false">:$discard,
    DefaultValuedAttr<BoolAttr, "false">:$disconnect
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs) `->` type($results)
  }];
  let hasVerifier = 1;
}

// fabric.spatial_pe - spatial processing element (container for FUs)
def Fabric_SpatialPEOp : Fabric_Op<"spatial_pe", [
    SingleBlock, SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Spatial processing element containing function units";
  let description = [{
    A spatial PE is a container for function_unit instances. At runtime,
    exactly one FU is active (selected by opcode register). PE ports
    are !fabric.bits<N>. Internally, FUs use native types (i32, f32).

    Config registers:
    - Opcode: selects which FU fires
    - Input mux: routes PE inputs to FU inputs (with disconnect/discard)
    - Output demux: routes FU outputs to PE outputs
    - FU internal config: max(fu_cfg_bits) across all FUs
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// fabric.temporal_pe - time-multiplexed processing element
def Fabric_TemporalPEOp : Fabric_Op<"temporal_pe", [
    SingleBlock, SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Time-multiplexed processing element";
  let description = [{
    A temporal PE multiplexes multiple FU types across instruction slots.
    Tag-indexed instruction memory selects which FU fires per token.
    All ports must be !fabric.tagged. Body contains FU definitions.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    I64Attr:$num_register,
    I64Attr:$num_instruction,
    I64Attr:$reg_fifo_depth,
    DefaultValuedAttr<BoolAttr, "false">:$enable_share_operand_buffer,
    OptionalAttr<I64Attr>:$operand_buffer_size,
    OptionalAttr<ArrayAttr>:$instruction_mem
  );
  let regions = (region SizedRegion<1>:$body);
  let results = (outs Variadic<AnyType>:$outputs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif

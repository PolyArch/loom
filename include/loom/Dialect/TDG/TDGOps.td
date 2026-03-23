#ifndef LOOM_DIALECT_TDG_TDGOPS_TD
#define LOOM_DIALECT_TDG_TDGOPS_TD

include "loom/Dialect/TDG/TDGDialect.td"
include "loom/Dialect/TDG/TDGTypes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TDG_Op<string mnemonic, list<Trait> traits = []>
    : Op<TDG_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// tdg.graph - Top-level task dataflow graph container
//===----------------------------------------------------------------------===//
def TDG_GraphOp : TDG_Op<"graph", [
    Symbol,
    SymbolTable,
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator
  ]> {
  let summary = "Top-level task dataflow graph";
  let description = [{
    Container for a multi-kernel dataflow application. Contains kernel
    definitions, contract specifications, and optional buffer declarations.
    Kernel names within a graph must be unique.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$target_arch
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name (`target_arch` `=` $target_arch^)? attr-dict-with-keyword
    $body
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tdg.kernel - Kernel node within a graph
//===----------------------------------------------------------------------===//
def TDG_KernelOp : TDG_Op<"kernel", [
    Symbol,
    HasParent<"GraphOp">
  ]> {
  let summary = "Kernel node in a task dataflow graph";
  let description = [{
    Represents a computational kernel in the task graph. The kernel body
    contains the computation expressed in SCF/affine/linalg dialects.
    The kernel_type attribute classifies the computation pattern.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kernel_type,
    OptionalAttr<DenseI64ArrayAttr>:$tile_hint,
    OptionalAttr<StrAttr>:$target_core_type
  );

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $sym_name `type` $kernel_type
    (`tile_hint` `=` $tile_hint^)?
    (`target_core_type` `=` $target_core_type^)?
    attr-dict-with-keyword $body
  }];
}

//===----------------------------------------------------------------------===//
// tdg.contract - First-class contract on a producer->consumer edge
//===----------------------------------------------------------------------===//
def TDG_ContractOp : TDG_Op<"contract", [
    HasParent<"GraphOp">
  ]> {
  let summary = "Contract on an inter-kernel communication edge";
  let description = [{
    Specifies the communication contract between a producer kernel and
    a consumer kernel. Includes ordering semantics, rate constraints,
    buffering parameters, memory visibility, and transformation permissions.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$producer,
    FlatSymbolRefAttr:$consumer,
    StrAttr:$ordering,
    TypeAttr:$data_type,
    OptionalAttr<AffineMapAttr>:$production_rate,
    OptionalAttr<AffineMapAttr>:$consumption_rate,
    OptionalAttr<DenseI64ArrayAttr>:$steady_state_ratio,
    OptionalAttr<DenseI64ArrayAttr>:$tile_shape,
    OptionalAttr<I64Attr>:$min_buffer_elements,
    OptionalAttr<I64Attr>:$max_buffer_elements,
    DefaultValuedStrAttr<StrAttr, "BLOCK">:$backpressure,
    DefaultValuedAttr<BoolAttr, "false">:$double_buffering,
    DefaultValuedStrAttr<StrAttr, "LOCAL_SPM">:$visibility,
    DefaultValuedStrAttr<StrAttr, "EAGER">:$producer_writeback,
    DefaultValuedStrAttr<StrAttr, "NONE">:$consumer_prefetch,
    DefaultValuedAttr<BoolAttr, "true">:$may_fuse,
    DefaultValuedAttr<BoolAttr, "true">:$may_replicate,
    DefaultValuedAttr<BoolAttr, "true">:$may_pipeline,
    DefaultValuedAttr<BoolAttr, "false">:$may_reorder,
    DefaultValuedAttr<BoolAttr, "true">:$may_retile
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tdg.buffer - Explicit intermediate buffer declaration
//===----------------------------------------------------------------------===//
def TDG_BufferOp : TDG_Op<"buffer", [
    HasParent<"GraphOp">
  ]> {
  let summary = "Explicit intermediate buffer";
  let description = [{
    Declares an explicit intermediate buffer with a given capacity
    and element type. Can be referenced by contracts for explicit
    buffer placement.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$capacity,
    TypeAttr:$element_type
  );

  let assemblyFormat = [{
    $sym_name `capacity` `=` $capacity `element_type` `=` $element_type
    attr-dict
  }];
}

#endif

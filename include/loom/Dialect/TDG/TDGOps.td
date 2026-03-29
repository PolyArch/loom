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
// tdg.contract - Edge contract (Ordering, Throughput, Placement, Shape)
//===----------------------------------------------------------------------===//
def TDG_ContractOp : TDG_Op<"contract", [
    HasParent<"GraphOp">
  ]> {
  let summary = "TDC edge contract on a producer->consumer communication edge";
  let description = [{
    Specifies the TDC edge contract between a producer kernel and a consumer
    kernel. The 3 required attributes identify the edge; the 4 optional
    attributes specify the TDC edge dimensions:
      - ordering: element delivery order (FIFO, UNORDERED, SYMBOLIC)
      - throughput: symbolic throughput expression
      - placement: memory placement (LOCAL_SPM, SHARED_L2, EXTERNAL, AUTO)
      - tile_shape: symbolic tile shape expression (e.g. "[128, hidden_dim]")

    Dimensions left unspecified are "unconstrained" and will be determined by
    the compiler.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$producer,
    FlatSymbolRefAttr:$consumer,
    TypeAttr:$data_type,
    OptionalAttr<StrAttr>:$ordering,
    OptionalAttr<StrAttr>:$throughput,
    OptionalAttr<StrAttr>:$placement,
    OptionalAttr<StrAttr>:$tile_shape
  );

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tdg.path_contract - Path contract (Latency across a chain of edges)
//===----------------------------------------------------------------------===//
def TDG_PathContractOp : TDG_Op<"path_contract", [
    HasParent<"GraphOp">
  ]> {
  let summary = "TDC path contract specifying latency across a chain of edges";
  let description = [{
    Specifies a latency constraint on a path from start_edge to end_edge.
    Both start_edge and end_edge reference tdg.contract operations within
    the same parent graph. The latency attribute is a symbolic expression
    for the upper-bound cycle count along the path.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$start_edge,
    FlatSymbolRefAttr:$end_edge,
    StrAttr:$latency
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

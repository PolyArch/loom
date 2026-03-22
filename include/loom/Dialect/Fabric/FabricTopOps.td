#ifndef LOOM_DIALECT_FABRIC_FABRICTOPOPS_TD
#define LOOM_DIALECT_FABRIC_FABRICTOPOPS_TD

include "loom/Dialect/Fabric/FabricDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"


// fabric.yield - region terminator
def Fabric_YieldOp : Fabric_Op<"yield", [Terminator]> {
  let summary = "Region terminator for fabric operations";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

// fabric.module - top-level accelerator module
def Fabric_ModuleOp : Fabric_Op<"module", [
    IsolatedFromAbove, SingleBlock,
    SingleBlockImplicitTerminator<"YieldOp">,
    RegionKindInterface, HasOnlyGraphRegion, Symbol
  ]> {
  let summary = "Top-level accelerator module";
  let description = [{
    Defines an accelerator module with typed ports. Contains the hardware
    graph (PEs, switches, FIFOs, memories) connected via SSA values.
    Port ordering: memref* -> bits* -> tagged*.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
  );
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// fabric.instance - instantiate a named fabric component
def Fabric_InstanceOp : Fabric_Op<"instance", []> {
  let summary = "Instantiate a named fabric component";
  let arguments = (ins
    FlatSymbolRefAttr:$module,
    OptionalAttr<StrAttr>:$sym_name,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $module `(` $operands `)` attr-dict `:` functional-type($operands, $results)
  }];
  let hasVerifier = 1;
}

#endif

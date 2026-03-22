#ifndef LOOM_DIALECT_FABRIC_FABRICMEMOPS_TD
#define LOOM_DIALECT_FABRIC_FABRICMEMOPS_TD

include "loom/Dialect/Fabric/FabricDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"


// fabric.extmemory - external memory interface
def Fabric_ExtMemoryOp : Fabric_Op<"extmemory", []> {
  let summary = "External memory interface";
  let description = [{
    Interface to external (off-chip/system) memory. First operand is
    a memref from the module input. For single-port memories the hardware
    interface is memref, ld_addr, st_addr, st_data with outputs
    ld_data, ld_done, st_done. When ldCount or stCount is greater than 1,
    the address/data/done ports become tagged aggregate ports rather than
    expanding positionally.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    TypeAttr:$memref_type,
    I64Attr:$ldCount,
    I64Attr:$stCount,
    DefaultValuedAttr<I64Attr, "0">:$lsqDepth,
    DefaultValuedAttr<I64Attr, "1">:$numRegion,
    OptionalAttr<DenseI64ArrayAttr>:$addrOffsetTable,
    TypeAttrOf<FunctionType>:$function_type
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// fabric.memory - on-chip scratchpad memory
def Fabric_MemoryOp : Fabric_Op<"memory", []> {
  let summary = "On-chip scratchpad memory";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    TypeAttr:$memref_type,
    I64Attr:$ldCount,
    I64Attr:$stCount,
    DefaultValuedAttr<I64Attr, "0">:$lsqDepth,
    DefaultValuedAttr<BoolAttr, "true">:$is_private,
    DefaultValuedAttr<I64Attr, "1">:$numRegion,
    OptionalAttr<DenseI64ArrayAttr>:$addrOffsetTable,
    TypeAttrOf<FunctionType>:$function_type
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif

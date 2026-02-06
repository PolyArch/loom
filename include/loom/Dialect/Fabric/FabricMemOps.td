//===- FabricMemOps.td - Fabric memory operations -----------*- tablegen -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_FABRIC_FABRICMEMOPS_TD
#define LOOM_DIALECT_FABRIC_FABRICMEMOPS_TD

//===----------------------------------------------------------------------===//
// fabric.memory
//
// Named form:
//   fabric.memory @name
//     [ldCount = N, stCount = N, lsqDepth = N, private = true]
//     : memref<shape>,
//       (input_types) -> (output_types)
//
// Inline form:
//   %outs = fabric.memory
//     [ldCount = N, stCount = N, lsqDepth = N, private = true]
//     (%ins) : memref<shape>, (input_types) -> (output_types)
//===----------------------------------------------------------------------===//

def Fabric_MemoryOp : Fabric_Op<"memory", []> {
  let summary = "On-chip scratchpad memory";
  let description = [{
    Defines an on-chip scratchpad memory with configurable load/store ports.
    Multi-port access uses tagged types for port discrimination.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    TypeAttr:$memref_type,
    I64Attr:$ldCount,
    I64Attr:$stCount,
    DefaultValuedAttr<I64Attr, "0">:$lsqDepth,
    DefaultValuedAttr<BoolAttr, "true">:$is_private,
    OptionalAttr<TypeAttrOf<FunctionType>>:$function_type
  );
  let results = (outs Variadic<AnyType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// fabric.extmemory
//
// Named form:
//   fabric.extmemory @name
//     [ldCount = N, stCount = N, lsqDepth = N]
//     : memref<?xT>,
//       (input_types) -> (output_types)
//
// Inline form:
//   %outs = fabric.extmemory
//     [ldCount = N, stCount = N, lsqDepth = N]
//     (%memref, %ins) : memref<?xT>, (input_types) -> (output_types)
//===----------------------------------------------------------------------===//

def Fabric_ExtMemoryOp : Fabric_Op<"extmemory", []> {
  let summary = "External memory interface";
  let description = [{
    Defines an external memory interface for off-chip or system memory access.
    The first operand must be a memref passed from the module input.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    TypeAttr:$memref_type,
    I64Attr:$ldCount,
    I64Attr:$stCount,
    DefaultValuedAttr<I64Attr, "0">:$lsqDepth,
    OptionalAttr<TypeAttrOf<FunctionType>>:$function_type
  );
  let results = (outs Variadic<AnyType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // LOOM_DIALECT_FABRIC_FABRICMEMOPS_TD

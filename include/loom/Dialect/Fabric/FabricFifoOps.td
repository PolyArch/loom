//===- FabricFifoOps.td - Fabric FIFO operations -------------*- tablegen -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_FABRIC_FABRICFIFOOPS_TD
#define LOOM_DIALECT_FABRIC_FABRICFIFOOPS_TD

//===----------------------------------------------------------------------===//
// fabric.fifo
//
// Named form:
//   fabric.fifo @name [depth = N] : (T) -> (T)
//   fabric.fifo @name [depth = N, bypassable] {bypassed = false} : (T) -> (T)
//
// Inline form:
//   %out = fabric.fifo [depth = N] %in : T
//   %out = fabric.fifo [depth = N, bypassable] {bypassed = false} %in : T
//===----------------------------------------------------------------------===//

def Fabric_FifoOp : Fabric_Op<"fifo", []> {
  let summary = "Pipeline register / FIFO buffer";
  let description = [{
    A single-input, single-output FIFO buffer that breaks combinational loops
    by introducing pipeline delay, and provides backpressure support.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    I64Attr:$depth,
    OptionalAttr<UnitAttr>:$bypassable,
    OptionalAttr<BoolAttr>:$bypassed,
    OptionalAttr<TypeAttrOf<FunctionType>>:$function_type
  );
  let results = (outs Variadic<AnyType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // LOOM_DIALECT_FABRIC_FABRICFIFOOPS_TD

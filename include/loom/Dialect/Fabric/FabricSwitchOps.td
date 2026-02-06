//===- FabricSwitchOps.td - Fabric switch operations ---------*- tablegen -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_FABRIC_FABRICSWITCHOPS_TD
#define LOOM_DIALECT_FABRIC_FABRICSWITCHOPS_TD

//===----------------------------------------------------------------------===//
// fabric.switch
//
// Named form:
//   fabric.switch @sw4x4
//     [connectivity_table = [...], route_table = [...]]
//     : (T, T, T, T) -> (T, T, T, T)
//
// Inline form:
//   %out0, %out1 = fabric.switch
//     [connectivity_table = [...], route_table = [...]]
//     %in0, %in1, %in2 : i32 -> i32, i32
//===----------------------------------------------------------------------===//

def Fabric_SwitchOp : Fabric_Op<"switch", []> {
  let summary = "Configurable routing switch";
  let description = [{
    A configurable routing switch that routes inputs to outputs based on a
    physical connectivity table and a runtime route table.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    OptionalAttr<DenseI8ArrayAttr>:$connectivity_table,
    OptionalAttr<DenseI8ArrayAttr>:$route_table,
    OptionalAttr<TypeAttrOf<FunctionType>>:$function_type
  );
  let results = (outs Variadic<AnyType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// fabric.temporal_sw
//
// Named form:
//   fabric.temporal_sw @tsw
//     [num_route_table = 4, connectivity_table = [...]]
//     {route_table = [...]}
//     : (!dataflow.tagged<T, iN>, ...) -> (!dataflow.tagged<T, iN>, ...)
//
// Inline form:
//   %out0, %out1 = fabric.temporal_sw
//     [num_route_table = 4, connectivity_table = [...]]
//     {route_table = [...]}
//     %in0, %in1, %in2 : !dataflow.tagged<T, iN> -> ..., ...
//===----------------------------------------------------------------------===//

def Fabric_TemporalSwOp : Fabric_Op<"temporal_sw", []> {
  let summary = "Tag-aware routing switch";
  let description = [{
    A tag-aware routing switch that selects a routing configuration based
    on the tag of the incoming token. All ports must be !dataflow.tagged
    with the same type.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    I64Attr:$num_route_table,
    OptionalAttr<DenseI8ArrayAttr>:$connectivity_table,
    OptionalAttr<ArrayAttr>:$route_table,
    OptionalAttr<TypeAttrOf<FunctionType>>:$function_type
  );
  let results = (outs Variadic<AnyType>:$outputs);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // LOOM_DIALECT_FABRIC_FABRICSWITCHOPS_TD

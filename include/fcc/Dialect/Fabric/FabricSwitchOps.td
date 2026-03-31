#ifndef FCC_DIALECT_FABRIC_FABRICSWITCHOPS_TD
#define FCC_DIALECT_FABRIC_FABRICSWITCHOPS_TD

include "fcc/Dialect/Fabric/FabricDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"


// fabric.spatial_sw - configurable routing switch (renamed from fabric.switch)
def Fabric_SpatialSwOp : Fabric_Op<"spatial_sw", []> {
  let summary = "Configurable routing switch with optional decomposition";
  let description = [{
    Configurable routing switch. decomposable_bits attribute controls
    sub-lane decomposition: -1 = non-decomposable, positive value D means
    every port is decomposed into W/D independent sub-lanes.

    Ports may be either all non-tagged or all tagged. When tagged, routing is
    still tag-agnostic: the tag bits are carried as payload and do not affect
    route selection.

    connectivity_table and route_table use binary strings, one per output
    (sub-)lane. For decomposable switches, tables operate at sub-lane
    granularity. discard_bit is per logical input port. Tagged spatial switches
    may not be decomposable.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    OptionalAttr<ArrayAttr>:$connectivity_table,
    OptionalAttr<ArrayAttr>:$route_table,
    DefaultValuedAttr<I64Attr, "-1">:$decomposable_bits,
    TypeAttrOf<FunctionType>:$function_type
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// fabric.temporal_sw - tag-aware routing switch
def Fabric_TemporalSwOp : Fabric_Op<"temporal_sw", []> {
  let summary = "Tag-aware routing switch with instruction memory";
  let description = [{
    Tag-indexed route tables. Each slot: [routes | tag | valid].
    All ports must be !fabric.tagged with the same type.
  }];
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    I64Attr:$num_route_table,
    OptionalAttr<ArrayAttr>:$connectivity_table,
    OptionalAttr<ArrayAttr>:$route_table,
    TypeAttrOf<FunctionType>:$function_type
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// fabric.fifo - pipeline buffer
def Fabric_FifoOp : Fabric_Op<"fifo", []> {
  let summary = "Pipeline register / FIFO buffer";
  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<SymbolNameAttr>:$sym_name,
    I64Attr:$depth,
    OptionalAttr<UnitAttr>:$bypassable,
    OptionalAttr<BoolAttr>:$bypassed,
    TypeAttrOf<FunctionType>:$function_type
  );
  let results = (outs Variadic<AnyType>:$outputs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif

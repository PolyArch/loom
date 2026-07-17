#ifndef LOOM_FRONTEND_IR_LOOMOPS_TD
#define LOOM_FRONTEND_IR_LOOMOPS_TD

include "Frontend/IR/LoomDialect.td"
include "Dataflow/IR/DataflowDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Loom_SpatialRegionOp : Loom_Op<"spatial_region", [
    IsolatedFromAbove,
    SingleBlock,
    AttrSizedOperandSegments,
    AttrSizedResultSegments,
    RecursiveMemoryEffects
]> {
  let summary = "Structured candidate for one SpatialCore graph";
  let description = [{
    Holds one structured candidate inside a `dataflow.thread`. Operands are
    normalized as value inputs, stream input channels, memory inputs, and
    stream output channels. Results are normalized as value outputs followed
    by memory outputs. Each stream input has one affine `source_map`.

    This operation is temporary compiler IR. Successful publication replaces
    it with one native-valid `dataflow.graph` and its matching launch.
  }];

  let arguments = (ins
      Variadic<AnyType>:$valueInputs,
      Variadic<Dataflow_ChannelType>:$streamInputs,
      Variadic<AnyType>:$memoryInputs,
      Variadic<Dataflow_ChannelType>:$streamOutputs,
      AffineMapArrayAttr:$source_maps,
      OptionalAttr<StrAttr>:$graph_name);

  let results = (outs
      Variadic<AnyType>:$valueResults,
      Variadic<AnyType>:$memoryResults);

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
        "::mlir::ValueRange":$valueInputs,
        "::mlir::ValueRange":$streamInputs,
        "::mlir::ValueRange":$memoryInputs,
        "::mlir::ValueRange":$streamOutputs,
        "::mlir::TypeRange":$valueResultTypes,
        "::mlir::TypeRange":$memoryResultTypes,
        "::mlir::ArrayAttr":$sourceMaps,
        CArg<"::mlir::StringAttr", "{}">:$graphName), [{
      $_state.addOperands(valueInputs);
      $_state.addOperands(streamInputs);
      $_state.addOperands(memoryInputs);
      $_state.addOperands(streamOutputs);
      $_state.addTypes(valueResultTypes);
      $_state.addTypes(memoryResultTypes);
      $_state.addAttribute("source_maps", sourceMaps);
      if (graphName)
        $_state.addAttribute("graph_name", graphName);
      auto &properties = $_state.getOrAddProperties<Properties>();
      properties.operandSegmentSizes = {
          static_cast<int32_t>(valueInputs.size()),
          static_cast<int32_t>(streamInputs.size()),
          static_cast<int32_t>(memoryInputs.size()),
          static_cast<int32_t>(streamOutputs.size())};
      properties.resultSegmentSizes = {
          static_cast<int32_t>(valueResultTypes.size()),
          static_cast<int32_t>(memoryResultTypes.size())};
      $_state.addRegion();
    }]>
  ];

  let hasVerifier = 1;
}

def Loom_SpatialYieldOp : Loom_Op<"spatial_yield", [
    Terminator,
    ParentOneOf<["::loom::SpatialRegionOp"]>,
    AttrSizedOperandSegments,
    Pure
]> {
  let summary = "Yield value and memory results from a spatial candidate";

  let arguments = (ins
      Variadic<AnyType>:$values,
      Variadic<AnyType>:$memories);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
        "::mlir::ValueRange":$values,
        "::mlir::ValueRange":$memories), [{
      $_state.addOperands(values);
      $_state.addOperands(memories);
      auto &properties = $_state.getOrAddProperties<Properties>();
      properties.operandSegmentSizes = {
          static_cast<int32_t>(values.size()),
          static_cast<int32_t>(memories.size())};
    }]>
  ];

  let hasVerifier = 1;
}

#endif // LOOM_FRONTEND_IR_LOOMOPS_TD

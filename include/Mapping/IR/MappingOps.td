#ifndef LOOM_MAPPING_IR_MAPPINGOPS_TD
#define LOOM_MAPPING_IR_MAPPINGOPS_TD

include "Mapping/IR/MappingAttrs.td"
include "Mapping/IR/MappingDialect.td"
include "Mapping/IR/MappingEnums.td"
include "mlir/IR/BuiltinAttributes.td"

def Mapping_TechOp : Mapping_Op<"tech", [IsolatedFromAbove, SingleBlock,
                                         NoTerminator]> {
  let summary = "complete loom.mapping 2.0 TechMapping root";
  let arguments = (ins
      Mapping_ArtifactIdentityAttr:$dataflow,
      Mapping_ArtifactIdentityAttr:$fabric,
      ArrayAttr:$covers);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Mapping_ComputeRealizationOp
    : Mapping_Op<"compute_realization", [HasParent<"TechOp">, SingleBlock,
                                         NoTerminator]> {
  let summary = "one complete FU capability realization";
  let arguments = (ins
      I64Attr:$entity_id,
      Mapping_FabricFuCapabilityTemplateRefAttr:$capability_template);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{$entity_id `capability` `(` $capability_template `)`
                         $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ComputeActorOp
    : Mapping_Op<"compute_actor", [HasParent<"ComputeRealizationOp">]> {
  let summary = "actor to FU operation and ordered port correspondence";
  let arguments = (ins
      Mapping_ActorRefAttr:$actor,
      Mapping_FabricFuTemplateNodeRefAttr:$fabric_op,
      DenseI32ArrayAttr:$operand_ports,
      DenseI32ArrayAttr:$result_ports);
  let assemblyFormat = [{`actor` `(` $actor `)` `op` `(` $fabric_op `)`
                         `operand_ports` `(` $operand_ports `)`
                         `result_ports` `(` $result_ports `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ComputeBoundaryOp
    : Mapping_Op<"compute_boundary", [HasParent<"ComputeRealizationOp">]> {
  let summary = "actor port to owner FU boundary port correspondence";
  let arguments = (ins
      Mapping_ActorRefAttr:$actor,
      Mapping_PortDirectionAttr:$direction,
      I64Attr:$port_ordinal,
      Mapping_FabricFuTemplatePortRefAttr:$fu_port);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // LOOM_MAPPING_IR_MAPPINGOPS_TD

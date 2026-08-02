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
      DenseI64ArrayAttr:$operand_ports,
      DenseI64ArrayAttr:$result_ports);
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

def Mapping_MemoryRealizationOp
    : Mapping_Op<"memory_realization", [HasParent<"TechOp">, SingleBlock,
                                         NoTerminator]> {
  let summary = "one complete memory operation-engine realization";
  let arguments = (ins
      I64Attr:$entity_id,
      Mapping_FabricMemoryEngineTemplateRefAttr:$engine);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{$entity_id `engine` `(` $engine `)` $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_MemoryActorOp
    : Mapping_Op<"memory_actor", [HasParent<"MemoryRealizationOp">]> {
  let summary = "memory actor to template operation-port correspondence";
  let arguments = (ins
      Mapping_ActorRefAttr:$actor,
      Mapping_FabricMemoryEngineTemplateOperationPortRefAttr:$operation_port,
      Mapping_FabricMemoryEngineTemplateCapabilityAlternativeRefAttr:$capability,
      ArrayAttr:$operand_ports,
      ArrayAttr:$result_ports);
  let assemblyFormat = [{`actor` `(` $actor `)`
                         `operation_port` `(` $operation_port `)`
                         `capability` `(` $capability `)`
                         `operand_ports` `(` $operand_ports `)`
                         `result_ports` `(` $result_ports `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_MemoryGraphBoundaryOp
    : Mapping_Op<"memory_graph_boundary",
                 [HasParent<"MemoryRealizationOp">]> {
  let summary = "graph token endpoint to memory engine endpoint correspondence";
  let arguments = (ins
      Mapping_GraphEndpointRef:$terminal,
      Mapping_FabricMemoryEngineTemplateEndpointRefAttr:$endpoint);
  let assemblyFormat = [{`terminal` `(` $terminal `)`
                         `endpoint` `(` $endpoint `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_MemoryInternalEdgeOp
    : Mapping_Op<"memory_internal_edge",
                 [HasParent<"MemoryRealizationOp">]> {
  let summary = "canonical software edge to memory engine internal connection";
  let arguments = (ins
      Mapping_GraphProducerEndpointRefAttr:$producer,
      Mapping_GraphConsumerEndpointRefAttr:$consumer,
      Mapping_FabricMemoryEngineTemplateInternalConnectionRefAttr:$connection);
  let assemblyFormat = [{`producer` `(` $producer `)`
                         `consumer` `(` $consumer `)`
                         `connection` `(` $connection `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ConstraintsSpatialOp
    : Mapping_Op<"constraints.spatial", [IsolatedFromAbove, SingleBlock,
                                          NoTerminator]> {
  let summary = "complete loom.mapping_constraints 1.0 Spatial root";
  let arguments = (ins
      Mapping_ArtifactIdentityAttr:$dataflow,
      Mapping_ArtifactIdentityAttr:$tech_mapping,
      Mapping_ArtifactIdentityAttr:$fabric);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Mapping_ConstraintDomainRestrictionOp
    : Mapping_Op<"constraint.domain_restriction",
                 [HasParent<"ConstraintsSpatialOp">]> {
  let summary = "one typed projected-domain restriction";
  let arguments = (ins
      Mapping_SpatialConstraintProjectionAttr:$projection,
      AnyAttr:$subject,
      ArrayAttr:$admissible_domain);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Mapping_ConstraintEqualOp
    : Mapping_Op<"constraint.equal", [HasParent<"ConstraintsSpatialOp">]> {
  let summary = "one canonical variadic projected equality";
  let arguments = (ins Mapping_SpatialConstraintProjectionAttr:$projection,
                       ArrayAttr:$subjects);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Mapping_ConstraintDisjointOp
    : Mapping_Op<"constraint.disjoint", [HasParent<"ConstraintsSpatialOp">]> {
  let summary = "one canonical variadic projected disjointness relation";
  let arguments = (ins Mapping_SpatialConstraintProjectionAttr:$projection,
                       ArrayAttr:$subjects);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // LOOM_MAPPING_IR_MAPPINGOPS_TD

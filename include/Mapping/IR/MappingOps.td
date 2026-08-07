#ifndef LOOM_MAPPING_IR_MAPPINGOPS_TD
#define LOOM_MAPPING_IR_MAPPINGOPS_TD

include "Mapping/IR/MappingAttrs.td"
include "Mapping/IR/MappingDialect.td"
include "Mapping/IR/MappingEnums.td"
include "mlir/IR/BuiltinAttributes.td"

def Mapping_TechOp
    : Mapping_Op<"tech", [IsolatedFromAbove, SingleBlock, NoTerminator]> {
  let summary = "complete loom.mapping 2.0 TechMapping root";
  let arguments = (ins Mapping_ArtifactIdentityAttr:$dataflow,
      Mapping_ArtifactIdentityAttr:$fabric, ArrayAttr:$covers);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Mapping_ComputeRealizationOp
    : Mapping_Op<"compute_realization", [HasParent<"TechOp">, SingleBlock,
                                         NoTerminator]> {
  let summary = "one complete FU capability realization";
  let arguments = (ins I64Attr:$entity_id,
      Mapping_FabricFuCapabilityTemplateRefAttr:$capability_template);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{$entity_id `capability` `(` $capability_template `)`
                         $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ComputeActorOp
    : Mapping_Op<"compute_actor", [HasParent<"ComputeRealizationOp">]> {
  let summary = "actor to FU operation and ordered port correspondence";
  let arguments = (ins Mapping_ActorRefAttr:$actor,
      Mapping_FabricFuTemplateNodeRefAttr:$fabric_op,
      DenseI64ArrayAttr:$operand_ports, DenseI64ArrayAttr:$result_ports);
  let assemblyFormat = [{`actor` `(` $actor `)` `op` `(` $fabric_op `)`
                         `operand_ports` `(` $operand_ports `)`
                         `result_ports` `(` $result_ports `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ComputeBoundaryOp
    : Mapping_Op<"compute_boundary", [HasParent<"ComputeRealizationOp">]> {
  let summary = "actor port to owner FU boundary port correspondence";
  let arguments = (ins Mapping_ActorRefAttr:$actor,
      Mapping_PortDirectionAttr:$direction, I64Attr:$port_ordinal,
      Mapping_FabricFuTemplatePortRefAttr:$fu_port);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Mapping_MemoryRealizationOp
    : Mapping_Op<"memory_realization", [HasParent<"TechOp">, SingleBlock,
                                        NoTerminator]> {
  let summary = "one complete memory operation-engine realization";
  let arguments = (ins I64Attr:$entity_id,
      Mapping_FabricMemoryEngineTemplateRefAttr:$engine);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{$entity_id `engine` `(` $engine `)` $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_MemoryActorOp
    : Mapping_Op<"memory_actor", [HasParent<"MemoryRealizationOp">]> {
  let summary = "memory actor to template operation-port correspondence";
  let arguments = (ins Mapping_ActorRefAttr:$actor,
      Mapping_FabricMemoryEngineTemplateOperationPortRefAttr:$operation_port,
      Mapping_FabricMemoryEngineTemplateCapabilityAlternativeRefAttr:$capability,
      ArrayAttr:$operand_ports, ArrayAttr:$result_ports);
  let assemblyFormat = [{`actor` `(` $actor `)`
                         `operation_port` `(` $operation_port `)`
                         `capability` `(` $capability `)`
                         `operand_ports` `(` $operand_ports `)`
                         `result_ports` `(` $result_ports `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_MemoryGraphBoundaryOp
    : Mapping_Op<"memory_graph_boundary", [HasParent<"MemoryRealizationOp">]> {
  let summary = "graph token endpoint to memory engine endpoint correspondence";
  let arguments = (ins Mapping_GraphEndpointRef:$terminal,
      Mapping_FabricMemoryEngineTemplateEndpointRefAttr:$endpoint);
  let assemblyFormat = [{`terminal` `(` $terminal `)`
                         `endpoint` `(` $endpoint `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_MemoryInternalEdgeOp
    : Mapping_Op<"memory_internal_edge", [HasParent<"MemoryRealizationOp">]> {
  let summary = "canonical software edge to memory engine internal connection";
  let arguments = (ins Mapping_GraphProducerEndpointRefAttr:$producer,
      Mapping_GraphConsumerEndpointRefAttr:$consumer,
      Mapping_FabricMemoryEngineTemplateInternalConnectionRefAttr:$connection);
  let assemblyFormat = [{`producer` `(` $producer `)`
                         `consumer` `(` $consumer `)`
                         `connection` `(` $connection `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_SpatialOp
    : Mapping_Op<"spatial", [IsolatedFromAbove, SingleBlock, NoTerminator]> {
  let summary = "complete loom.mapping 2.0 SpatialMapping root";
  let arguments = (ins Mapping_ArtifactIdentityAttr:$tech_mapping,
      Mapping_ArtifactIdentityAttr:$dataflow,
      Mapping_ArtifactIdentityAttr:$fabric);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Mapping_ComputeBindingOp
    : Mapping_Op<"compute_binding", [HasParent<"SpatialOp">]> {
  let summary = "one TechMapping compute realization placement";
  let arguments = (ins Mapping_ComputeRealizationRefAttr:$realization,
      Mapping_FabricFuOccurrenceRefAttr:$occurrence,
      Mapping_InstructionContextRefAttr:$context, ArrayAttr:$refinements);
  let assemblyFormat = [{`realization` `(` $realization `)`
                         `occurrence` `(` $occurrence `)`
                         `context` `(` $context `)`
                         `refinements` `(` $refinements `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_MemoryEngineBindingOp
    : Mapping_Op<"memory_engine_binding", [HasParent<"SpatialOp">, SingleBlock,
                                           NoTerminator]> {
  let summary = "one TechMapping memory realization placement";
  let arguments = (ins Mapping_MemoryRealizationRefAttr:$realization,
      Mapping_FabricMemoryOccurrenceRefAttr:$occurrence);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{`realization` `(` $realization `)`
                         `occurrence` `(` $occurrence `)` $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_AddressedMemoryOperationOp
    : Mapping_Op<
          "addressed_memory_operation", [HasParent<"MemoryEngineBindingOp">,
                                         SingleBlock, NoTerminator]> {
  let summary = "one addressed actor placement and its rooted uses";
  let arguments = (ins Mapping_ActorRefAttr:$actor,
      Mapping_MemoryOperationPlacementRef:$placement);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{`actor` `(` $actor `)`
                         `placement` `(` $placement `)`
                         $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_AddressedMemoryUseOp
    : Mapping_Op<
          "addressed_memory_use", [HasParent<"AddressedMemoryOperationOp">]> {
  let summary = "one rooted addressed-memory binding and dispatch";
  let arguments = (ins Mapping_RootedGraphLaunchRefAttr:$launch,
      Mapping_MemoryBindingRefAttr:$binding,
      Mapping_MemoryDispatchTarget:$dispatch_target);
  let assemblyFormat = [{`launch` `(` $launch `)`
                         `binding` `(` $binding `)`
                         `dispatch` `(` $dispatch_target `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_FenceMemoryOperationOp
    : Mapping_Op<"fence_memory_operation", [HasParent<"MemoryEngineBindingOp">,
                                            SingleBlock, NoTerminator]> {
  let summary = "one fence actor placement and its rooted uses";
  let arguments = (ins Mapping_ActorRefAttr:$actor,
      Mapping_MemoryOperationPlacementRef:$placement);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{`actor` `(` $actor `)`
                         `placement` `(` $placement `)`
                         $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_FenceMemoryUseOp
    : Mapping_Op<"fence_memory_use", [HasParent<"FenceMemoryOperationOp">]> {
  let summary = "one rooted fence consistency target";
  let arguments = (ins Mapping_RootedGraphLaunchRefAttr:$launch,
      Mapping_MemoryConsistencyTarget:$consistency_target);
  let assemblyFormat = [{`launch` `(` $launch `)`
                         `consistency` `(` $consistency_target `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_MemoryBindingOp
    : Mapping_Op<"memory_binding", [HasParent<"SpatialOp">, SingleBlock,
                                    NoTerminator]> {
  let summary = "one logical memory interval placement or boundary proxy";
  let arguments = (ins I64Attr:$entity_id,
      Mapping_LogicalMemoryRootOrViewRefAttr:$logical_memory,
      Mapping_LogicalMemoryInterval:$interval,
      Mapping_MemoryBindingTarget:$target);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{$entity_id `logical_memory` `(` $logical_memory `)`
                         `interval` `(` $interval `)`
                         `target` `(` $target `)` $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ExposureEntryOp
    : Mapping_Op<"exposure_entry", [HasParent<"MemoryBindingOp">]> {
  let summary = "one Dataflow memory exposure and selected dispatch";
  let arguments = (ins Mapping_MemoryExposureRefAttr:$exposure,
      Mapping_SubordinateEndpointRefAttr:$terminal,
      Mapping_MemoryDispatchTarget:$dispatch_target);
  let assemblyFormat = [{`exposure` `(` $exposure `)`
                         `terminal` `(` $terminal `)`
                         `dispatch` `(` $dispatch_target `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_RouteTreeOp
    : Mapping_Op<"route_tree", [HasParent<"SpatialOp">, SingleBlock,
                                NoTerminator]> {
  let summary = "one complete Spatial logical-net route tree";
  let arguments = (ins Mapping_GraphProducerEndpointRefAttr:$logical_net,
      Mapping_FabricTransportEndpointRefAttr:$root_endpoint);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{`logical_net` `(` $logical_net `)`
                         `root_endpoint` `(` $root_endpoint `)`
                         $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_RouteNodeOp : Mapping_Op<"route_node", [HasParent<"RouteTreeOp">]> {
  let summary = "one canonical node in a flat Spatial route tree";
  let arguments = (ins I64Attr:$node_ordinal,
      OptionalAttr<I64Attr>:$parent_node_ordinal,
      OptionalAttr<Mapping_FabricPhysicalTraversalRefAttr>:$incoming_traversal,
      ArrayAttr:$refinements);
  let assemblyFormat = [{`node` $node_ordinal
                         (`parent` $parent_node_ordinal^
                          `traversal` `(` $incoming_traversal `)`)?
                         `refinements` `(` $refinements `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_RouteSinkOp : Mapping_Op<"route_sink", [HasParent<"RouteTreeOp">]> {
  let summary = "one logical sink obligation attached to a route-tree node";
  let arguments = (ins Mapping_GraphConsumerEndpointRefAttr:$sink,
      I64Attr:$node_ordinal);
  let assemblyFormat = [{`sink` `(` $sink `)` `node` $node_ordinal attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ResourceUseOp
    : Mapping_Op<"resource_use", [HasParent<"SpatialOp">]> {
  let summary = "one event-relative Mapping selection of a Fabric UsePattern";
  let arguments = (ins Mapping_SpatialResourceOwnerRef:$owner,
      Mapping_FabricUsePatternRefAttr:$use_site,
      Mapping_SpatialRelativeActivationAttr:$activation, ArrayAttr:$parameters,
      ArrayAttr:$sharing_assignments);
  let assemblyFormat = [{`owner` `(` $owner `)`
                         `use_site` `(` $use_site `)`
                         `activation` `(` $activation `)`
                         `parameters` `(` $parameters `)`
                         `sharing` `(` $sharing_assignments `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_SystemOp
    : Mapping_Op<"system", [IsolatedFromAbove, SingleBlock, NoTerminator]> {
  let summary = "complete loom.mapping 2.0 SystemMapping root";
  let arguments = (ins Mapping_ArtifactIdentityAttr:$dataflow,
      Mapping_ArtifactIdentityAttr:$fabric,
      ArrayAttr:$spatial_mapping_imports,
      ArrayAttr:$root_thread_launches);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Mapping_ThreadExecutionBindingOp
    : Mapping_Op<"thread_execution_binding",
                 [HasParent<"SystemOp">, SingleBlock, NoTerminator]> {
  let summary = "one root-thread execution binding relation";
  let arguments = (ins Mapping_RootThreadLaunchRefAttr:$key,
      Mapping_SystemBindingRelationKindAttr:$relation_kind,
      OptionalAttr<Mapping_FabricAccCoreOccurrenceRefAttr>:$default_target);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{`key` `(` $key `)` `relation` $relation_kind
                         (`default` `(` $default_target^ `)`)?
                         $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ThreadPresburgerClauseOp
    : Mapping_Op<"thread_presburger_clause",
                 [HasParent<"ThreadExecutionBindingOp">]> {
  let summary = "one target and canonical integer-set union";
  let arguments = (ins ArrayAttr:$cells,
      Mapping_FabricAccCoreOccurrenceRefAttr:$target);
  let assemblyFormat = [{`cells` `(` $cells `)` `target` `(` $target `)`
                         attr-dict}];
  let hasVerifier = 1;
}

def Mapping_GraphExecutionBindingOp
    : Mapping_Op<"graph_execution_binding",
                 [HasParent<"SystemOp">, SingleBlock, NoTerminator]> {
  let summary = "one rooted graph-launch execution binding relation";
  let arguments = (ins Mapping_RootedGraphLaunchRefAttr:$key,
      Mapping_SystemBindingRelationKindAttr:$relation_kind,
      OptionalAttr<Mapping_SpatialMappingImportRefAttr>:$default_target);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{`key` `(` $key `)` `relation` $relation_kind
                         (`default` `(` $default_target^ `)`)?
                         $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_GraphPresburgerClauseOp
    : Mapping_Op<"graph_presburger_clause",
                 [HasParent<"GraphExecutionBindingOp">]> {
  let summary = "one SpatialMapping target and canonical integer-set union";
  let arguments = (ins ArrayAttr:$cells,
      Mapping_SpatialMappingImportRefAttr:$target);
  let assemblyFormat = [{`cells` `(` $cells `)` `target` `(` $target `)`
                         attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ServiceRealizationOp
    : Mapping_Op<"service_realization",
                 [HasParent<"SystemOp">, SingleBlock, NoTerminator]> {
  let summary = "one System service-obligation realization";
  let arguments = (ins Mapping_SystemServiceObligationKeyAttr:$key);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{`key` `(` $key `)` $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ServicePlanOp
    : Mapping_Op<"service_plan",
                 [HasParent<"ServiceRealizationOp">, SingleBlock,
                  NoTerminator]> {
  let summary = "one owner-local System service plan";
  let arguments = (ins I64Attr:$plan_ordinal);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{`plan` $plan_ordinal $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ServicePlanSelectionOp
    : Mapping_Op<"service_plan_selection",
                 [HasParent<"ServiceRealizationOp">, SingleBlock,
                  NoTerminator]> {
  let summary = "one contextual owner-local service-plan selection relation";
  let arguments = (ins Mapping_ServicePlanSelectionKeyAttr:$key,
      Mapping_SystemBindingRelationKindAttr:$relation_kind,
      OptionalAttr<I64Attr>:$default_plan_ordinal);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{`key` `(` $key `)` `relation` $relation_kind
                         (`default` $default_plan_ordinal^)?
                         $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ServicePlanPresburgerClauseOp
    : Mapping_Op<"service_plan_presburger_clause",
                 [HasParent<"ServicePlanSelectionOp">]> {
  let summary = "one plan ordinal and canonical integer-set union";
  let arguments = (ins ArrayAttr:$cells, I64Attr:$target_plan_ordinal);
  let assemblyFormat = [{`cells` `(` $cells `)`
                         `target` $target_plan_ordinal attr-dict}];
  let hasVerifier = 1;
}

def Mapping_TransferLegRealizationOp
    : Mapping_Op<"transfer_leg_realization",
                 [HasParent<"ServicePlanOp">, SingleBlock, NoTerminator]> {
  let summary = "one canonical service leg and its reduced flat route tree";
  let arguments = (ins Mapping_CanonicalServiceLegKeyAttr:$leg,
      Mapping_FabricTransportEndpointRefAttr:$root_endpoint);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{`leg` `(` $leg `)`
                         `root_endpoint` `(` $root_endpoint `)`
                         $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_SystemRouteNodeOp
    : Mapping_Op<"system_route_node",
                 [HasParent<"TransferLegRealizationOp">]> {
  let summary = "one non-root node in a reduced System route tree";
  let arguments = (ins I64Attr:$node_ordinal, I64Attr:$parent_node_ordinal,
      Mapping_FabricPhysicalTraversalRefAttr:$incoming_traversal);
  let assemblyFormat = [{`node` $node_ordinal `parent` $parent_node_ordinal
                         `traversal` `(` $incoming_traversal `)` attr-dict}];
  let hasVerifier = 1;
}

def Mapping_SystemRouteSinkOp
    : Mapping_Op<"system_route_sink",
                 [HasParent<"TransferLegRealizationOp">]> {
  let summary = "one System transfer sink attached to a route node";
  let arguments = (ins Mapping_SystemTransferTerminalKeyAttr:$terminal,
      I64Attr:$node_ordinal);
  let assemblyFormat = [{`terminal` `(` $terminal `)`
                         `node` $node_ordinal attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ConstraintsSpatialOp
    : Mapping_Op<"constraints.spatial", [IsolatedFromAbove, SingleBlock,
                                         NoTerminator]> {
  let summary = "complete loom.mapping_constraints 1.0 Spatial root";
  let arguments = (ins Mapping_ArtifactIdentityAttr:$dataflow,
      Mapping_ArtifactIdentityAttr:$tech_mapping,
      Mapping_ArtifactIdentityAttr:$fabric);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Mapping_ConstraintsSystemOp
    : Mapping_Op<"constraints.system", [IsolatedFromAbove, SingleBlock,
                                         NoTerminator]> {
  let summary = "complete loom.mapping_constraints 1.0 System root";
  let arguments = (ins Mapping_ArtifactIdentityAttr:$dataflow,
      Mapping_ArtifactIdentityAttr:$fabric,
      ArrayAttr:$root_thread_launches,
      ArrayAttr:$spatial_mapping_reference_table);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{`dataflow` `(` $dataflow `)`
                         `fabric` `(` $fabric `)`
                         `root_thread_launches` `(` $root_thread_launches `)`
                         `spatial_mapping_reference_table`
                         `(` $spatial_mapping_reference_table `)`
                         $body attr-dict}];
  let hasVerifier = 1;
}

def Mapping_ConstraintProjectionAttr
    : AnyAttrOf<[Mapping_SpatialConstraintProjectionKeyAttr,
                 Mapping_SystemConstraintProjectionKeyAttr],
                "closed Spatial or System constraint projection">;

def Mapping_ConstraintDomainRestrictionOp
    : Mapping_Op<"constraint.domain_restriction"> {
  let summary = "one typed projected-domain restriction";
  let arguments = (ins Mapping_ConstraintProjectionAttr:$projection,
      AnyAttr:$subject, ArrayAttr:$admissible_domain);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Mapping_ConstraintEqualOp
    : Mapping_Op<"constraint.equal"> {
  let summary = "one canonical variadic projected equality";
  let arguments = (ins Mapping_ConstraintProjectionAttr:$projection,
      ArrayAttr:$subjects);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Mapping_ConstraintDisjointOp
    : Mapping_Op<"constraint.disjoint"> {
  let summary = "one canonical variadic projected disjointness relation";
  let arguments = (ins Mapping_ConstraintProjectionAttr:$projection,
      ArrayAttr:$subjects);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // LOOM_MAPPING_IR_MAPPINGOPS_TD

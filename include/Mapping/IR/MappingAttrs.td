#ifndef LOOM_MAPPING_IR_MAPPINGATTRS_TD
#define LOOM_MAPPING_IR_MAPPINGATTRS_TD

include "Mapping/IR/MappingDialect.td"
include "mlir/IR/AttrTypeBase.td"

class Mapping_Attr<string name, string attrMnemonic>
    : AttrDef<Mapping_Dialect, name> {
  let mnemonic = attrMnemonic;
}

class Mapping_CanonicalBytesAttr<string name, string mnemonic,
                                 string summaryText>
    : Mapping_Attr<name, mnemonic> {
  let summary = summaryText;
  let parameters = (ins "::mlir::DenseI8ArrayAttr":$record);
  let assemblyFormat = "`<` $record `>`";
  let genVerifyDecl = 1;
}

def Mapping_ArtifactIdentityAttr
    : Mapping_CanonicalBytesAttr<"ArtifactIdentity", "artifact_identity",
                                 "exact Common ArtifactIdentity bytes">;

def Mapping_GraphRefAttr
    : Mapping_CanonicalBytesAttr<
          "GraphRef", "graph_ref",
          "Dataflow-owned GraphRef scoped by the root dataflow binding">;

def Mapping_ActorRefAttr
    : Mapping_CanonicalBytesAttr<
          "ActorRef", "actor_ref",
          "Dataflow-owned ActorRef scoped by the root dataflow binding">;

def Mapping_RootedGraphLaunchRefAttr
    : Mapping_CanonicalBytesAttr<
          "RootedGraphLaunchRef", "rooted_graph_launch_ref",
          "Dataflow-owned rooted graph launch reference">;

def Mapping_RootThreadLaunchRefAttr
    : Mapping_CanonicalBytesAttr<
          "RootThreadLaunchRef", "root_thread_launch_ref",
          "Dataflow-owned root thread launch reference">;

def Mapping_SystemServiceObligationKeyAttr
    : Mapping_CanonicalBytesAttr<
          "SystemServiceObligationKey", "system_service_obligation_key",
          "Mapping-owned System service-obligation key">;

def Mapping_CanonicalServiceLegKeyAttr
    : Mapping_CanonicalBytesAttr<
          "CanonicalServiceLegKey", "canonical_service_leg_key",
          "Mapping-owned canonical System service-leg key">;

def Mapping_SystemTransferTerminalKeyAttr
    : Mapping_CanonicalBytesAttr<
          "SystemTransferTerminalKey", "system_transfer_terminal_key",
          "Mapping-owned System service transfer-terminal key">;

def Mapping_SystemPresburgerCellAttr
    : Mapping_Attr<"SystemPresburgerCell", "system_presburger_cell"> {
  let summary = "one convex canonical integer-set disjunct";
  let parameters = (ins "uint32_t":$dimensionCount,
      "uint32_t":$symbolCount, "::mlir::ArrayAttr":$equalities,
      "::mlir::ArrayAttr":$inequalities);
  let assemblyFormat = "`<` $dimensionCount `,` $symbolCount `,` "
                       "$equalities `,` $inequalities `>`";
  let genVerifyDecl = 1;
}

def Mapping_SpatialMappingImportRefAttr
    : Mapping_Attr<"SpatialMappingImportRef", "spatial_mapping_import_ref"> {
  let summary = "SystemMapping-local exact SpatialMapping import ordinal";
  let parameters = (ins "uint64_t":$ordinal);
  let assemblyFormat = "`<` $ordinal `>`";
}

def Mapping_ArtifactRootReferenceAttr
    : Mapping_CanonicalBytesAttr<
          "ArtifactRootReference", "artifact_root_reference",
          "exact Common ArtifactRootReference framing bytes">;

def Mapping_LogicalMemoryRootRefAttr
    : Mapping_CanonicalBytesAttr<
          "LogicalMemoryRootRef", "logical_memory_root_ref",
          "Dataflow-owned LogicalMemoryRootRef scoped by the root binding">;

def Mapping_LogicalMemoryRootOrViewRefAttr
    : Mapping_CanonicalBytesAttr<
          "LogicalMemoryRootOrViewRef", "logical_memory_root_or_view_ref",
          "Dataflow-owned logical memory root-or-view reference">;

def Mapping_MemoryExposureRefAttr
    : Mapping_CanonicalBytesAttr<"MemoryExposureRef", "memory_exposure_ref",
                                 "Dataflow-owned memory exposure reference">;

def Mapping_ComputeRealizationRefAttr
    : Mapping_Attr<"ComputeRealizationRef", "compute_realization_ref"> {
  let summary = "TechMapping-owned ComputeRealizationRef";
  let parameters = (ins "uint64_t":$entity);
  let assemblyFormat = "`<` $entity `>`";
}

def Mapping_MemoryRealizationRefAttr
    : Mapping_Attr<"MemoryRealizationRef", "memory_realization_ref"> {
  let summary = "TechMapping-owned MemoryRealizationRef";
  let parameters = (ins "uint64_t":$entity);
  let assemblyFormat = "`<` $entity `>`";
}

def Mapping_GraphProducerEndpointRefAttr
    : Mapping_CanonicalBytesAttr<
          "GraphProducerEndpointRef", "graph_producer_endpoint_ref",
          "Dataflow-owned canonical graph producer endpoint reference">;

def Mapping_GraphConsumerEndpointRefAttr
    : Mapping_CanonicalBytesAttr<
          "GraphConsumerEndpointRef", "graph_consumer_endpoint_ref",
          "Dataflow-owned canonical graph consumer endpoint reference">;

def Mapping_GraphEndpointRef
    : AnyAttrOf<[Mapping_GraphProducerEndpointRefAttr,
                 Mapping_GraphConsumerEndpointRefAttr],
                "Dataflow-owned canonical graph producer or consumer endpoint "
                "reference">;

def Mapping_FabricFuCapabilityTemplateRefAttr
    : Mapping_CanonicalBytesAttr<
          "FabricFuCapabilityTemplateRef", "fabric_fu_capability_template_ref",
          "Fabric-owned FU capability template reference">;

def Mapping_FabricFuTemplateNodeRefAttr
    : Mapping_CanonicalBytesAttr<"FabricFuTemplateNodeRef",
                                 "fabric_fu_template_node_ref",
                                 "Fabric-owned FU template node reference">;

def Mapping_FabricFuTemplatePortRefAttr
    : Mapping_CanonicalBytesAttr<"FabricFuTemplatePortRef",
                                 "fabric_fu_template_port_ref",
                                 "Fabric-owned FU template port reference">;

def Mapping_FabricFuOccurrenceRefAttr
    : Mapping_CanonicalBytesAttr<"FabricFuOccurrenceRef",
                                 "fabric_fu_occurrence_ref",
                                 "Fabric-owned FU occurrence reference">;

def Mapping_FabricAccCoreOccurrenceRefAttr
    : Mapping_CanonicalBytesAttr<"FabricAccCoreOccurrenceRef",
                                 "fabric_acc_core_occurrence_ref",
                                 "Fabric-owned AccCore occurrence reference">;

def Mapping_FabricPeOccurrenceRefAttr
    : Mapping_CanonicalBytesAttr<"FabricPeOccurrenceRef",
                                 "fabric_pe_occurrence_ref",
                                 "Fabric-owned PE occurrence reference">;

def Mapping_InstructionContextRefAttr
    : Mapping_CanonicalBytesAttr<"InstructionContextRef",
                                 "instruction_context_ref",
                                 "Fabric-owned instruction context reference">;

def Mapping_FabricUsePatternRefAttr
    : Mapping_CanonicalBytesAttr<"FabricUsePatternRef",
                                 "fabric_use_pattern_ref",
                                 "Fabric-owned atomic use-pattern reference">;

def Mapping_FabricPhysicalRefinementDomainRefAttr
    : Mapping_CanonicalBytesAttr<
          "FabricPhysicalRefinementDomainRef",
          "fabric_physical_refinement_domain_ref",
          "Fabric-owned physical-refinement domain reference">;

def Mapping_FabricMemoryOccurrenceRefAttr
    : Mapping_CanonicalBytesAttr<"FabricMemoryOccurrenceRef",
                                 "fabric_memory_occurrence_ref",
                                 "Fabric-owned memory occurrence reference">;

def Mapping_FabricPhysicalTraversalRefAttr
    : Mapping_CanonicalBytesAttr<"FabricPhysicalTraversalRef",
                                 "fabric_physical_traversal_ref",
                                 "Fabric-owned physical traversal reference">;

def Mapping_FabricResourceStateRefAttr
    : Mapping_CanonicalBytesAttr<"FabricResourceStateRef",
                                 "fabric_resource_state_ref",
                                 "Fabric-owned resource state reference">;

def Mapping_FabricTransportEndpointRefAttr
    : Mapping_CanonicalBytesAttr<"FabricTransportEndpointRef",
                                 "fabric_transport_endpoint_ref",
                                 "Fabric-owned transport endpoint reference">;

def Mapping_FabricMemoryOperationPortRefAttr
    : Mapping_CanonicalBytesAttr<
          "FabricMemoryOperationPortRef", "fabric_memory_operation_port_ref",
          "Fabric-owned memory operation port reference">;

def Mapping_FabricMemoryOperationContextRefAttr
    : Mapping_CanonicalBytesAttr<
          "FabricMemoryOperationContextRef",
          "fabric_memory_operation_context_ref",
          "Fabric-owned memory operation context reference">;

def Mapping_FabricMemoryServiceRefAttr
    : Mapping_CanonicalBytesAttr<"FabricMemoryServiceRef",
                                 "fabric_memory_service_ref",
                                 "Fabric-owned memory service reference">;

def Mapping_FabricMemoryServiceRegionRefAttr
    : Mapping_CanonicalBytesAttr<
          "FabricMemoryServiceRegionRef", "fabric_memory_service_region_ref",
          "Fabric-owned memory service-region reference">;

def Mapping_LocalMemoryServiceRefAttr
    : Mapping_CanonicalBytesAttr<
          "LocalMemoryServiceRef", "local_memory_service_ref",
          "Fabric-owned local memory service refinement">;

def Mapping_ManagerEndpointRefAttr
    : Mapping_CanonicalBytesAttr<
          "ManagerEndpointRef", "manager_endpoint_ref",
          "Fabric-owned manager memory endpoint refinement">;

def Mapping_SubordinateEndpointRefAttr
    : Mapping_CanonicalBytesAttr<
          "SubordinateEndpointRef", "subordinate_endpoint_ref",
          "Fabric-owned subordinate memory endpoint refinement">;

def Mapping_MemoryConsistencyDomainRefAttr
    : Mapping_CanonicalBytesAttr<
          "MemoryConsistencyDomainRef", "memory_consistency_domain_ref",
          "Fabric-owned memory consistency-domain refinement">;

def Mapping_MemoryWholeIntervalAttr
    : Mapping_Attr<"MemoryWholeInterval", "memory_whole_interval"> {
  let summary = "the complete logical memory root or view";
  let assemblyFormat = "";
}

def Mapping_MemoryByteRangeAttr
    : Mapping_Attr<"MemoryByteRange", "memory_byte_range"> {
  let summary = "one non-empty logical byte interval";
  let parameters = (ins "uint64_t":$offset_bytes, "uint64_t":$size_bytes);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Mapping_LogicalMemoryInterval
    : AnyAttrOf<[Mapping_MemoryWholeIntervalAttr, Mapping_MemoryByteRangeAttr],
                "closed Whole or ByteRange logical memory interval">;

def Mapping_MemoryLocalRegionAttr
    : Mapping_Attr<"MemoryLocalRegion", "memory_local_region"> {
  let summary = "one exact local memory service region and byte offset";
  let parameters = (ins
      "::mapping::FabricMemoryServiceRegionRefAttr":$service_region,
      "uint64_t":$physical_offset_bytes);
  let assemblyFormat = "`<` struct(params) `>`";
}

def Mapping_MemoryBoundaryProxyAttr
    : Mapping_Attr<"MemoryBoundaryProxy", "memory_boundary_proxy"> {
  let summary = "one logical interval crossing the Spatial service boundary";
  let assemblyFormat = "";
}

def Mapping_MemoryBindingTarget
    : AnyAttrOf<[Mapping_MemoryLocalRegionAttr,
                 Mapping_MemoryBoundaryProxyAttr],
                "closed local-region or boundary-proxy memory target">;

def Mapping_MemoryOperationPlacementRef
    : AnyAttrOf<[Mapping_FabricMemoryOperationPortRefAttr,
                 Mapping_FabricMemoryOperationContextRefAttr],
                "closed Spatial or Temporal memory operation placement">;

def Mapping_MemoryDispatchTarget
    : AnyAttrOf<[Mapping_LocalMemoryServiceRefAttr,
                 Mapping_ManagerEndpointRefAttr],
                "closed local-service or manager-endpoint dispatch target">;

def Mapping_MemoryConsistencyTarget
    : AnyAttrOf<[Mapping_MemoryConsistencyDomainRefAttr,
                 Mapping_ManagerEndpointRefAttr],
                "closed local consistency-domain or manager-endpoint target">;

def Mapping_OwnerTypedValueAttr
    : Mapping_CanonicalBytesAttr<
          "OwnerTypedValue", "owner_typed_value",
          "canonical bytes decoded by the exact referenced owner codec">;

def Mapping_MemoryBindingRefAttr
    : Mapping_Attr<"MemoryBindingRef", "memory_binding_ref"> {
  let summary = "SpatialMapping-owned MemoryBindingRef";
  let parameters = (ins "uint64_t":$entity);
  let assemblyFormat = "`<` $entity `>`";
}

def Mapping_RouteTreeNodeRefAttr
    : Mapping_Attr<"RouteTreeNodeRef", "route_tree_node_ref"> {
  let summary = "Spatial logical-net key and canonical route-node ordinal";
  let parameters = (ins "::mapping::GraphProducerEndpointRefAttr":$logical_net,
      "uint64_t":$node_ordinal);
  let assemblyFormat = "`<` struct(params) `>`";
}

def Mapping_ActorTransitionEventAttr
    : Mapping_Attr<"ActorTransitionEvent", "actor_transition_event"> {
  let summary = "OperationSchema-owned actor handshake transition occurrence";
  let parameters = (ins "::mapping::ActorRefAttr":$actor,
      "uint32_t":$transition);
  let assemblyFormat = "`<` struct(params) `>`";
}

def Mapping_SpatialActivityEventRef
    : AnyAttrOf<[Mapping_ActorTransitionEventAttr,
                 Mapping_GraphProducerEndpointRefAttr,
                 Mapping_GraphConsumerEndpointRefAttr],
                "graph-local actor transition, produced terminal, or consumed "
                "terminal">;

def Mapping_SpatialEventPointAttr
    : Mapping_Attr<"SpatialEventPoint", "spatial_event_point"> {
  let summary = "graph-local activity event with an optional guaranteed offset";
  let parameters = (ins "::mlir::Attribute":$event,
      OptionalParameter<"::mapping::OwnerTypedValueAttr">:$guaranteed_offset);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Mapping_SpatialRelativeActivationAttr
    : Mapping_Attr<"SpatialRelativeActivation", "spatial_relative_activation"> {
  let summary = "one trigger and intrinsic or causal release";
  let parameters = (ins "::mapping::SpatialEventPointAttr":$trigger,
      OptionalParameter<"::mapping::SpatialEventPointAttr">:$release);
  let assemblyFormat = "`<` struct(params) `>`";
}

def Mapping_PhysicalRefinementAssignmentAttr
    : Mapping_Attr<"PhysicalRefinementAssignment",
                   "physical_refinement_assignment"> {
  let summary = "one Fabric-owned physical-refinement selection";
  let parameters = (ins
      "::mapping::FabricPhysicalRefinementDomainRefAttr":$domain,
      "::mapping::OwnerTypedValueAttr":$value);
  let assemblyFormat = "`<` struct(params) `>`";
}

def Mapping_SpatialResourceOwnerRef
    : AnyAttrOf<[Mapping_ComputeRealizationRefAttr,
                 Mapping_MemoryRealizationRefAttr, Mapping_MemoryBindingRefAttr,
                 Mapping_RouteTreeNodeRefAttr],
                "closed Spatial ResourceUse owner reference">;

def Mapping_SpatialTransferTerminalAttr
    : Mapping_Attr<"SpatialTransferTerminal", "spatial_transfer_terminal"> {
  let summary = "source or exact sink terminal of one Spatial logical net";
  let parameters = (ins "::mapping::GraphProducerEndpointRefAttr":$producer,
      OptionalParameter<"::mapping::GraphConsumerEndpointRefAttr">:$consumer);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Mapping_ConstraintFuContextAttr
    : Mapping_Attr<"ConstraintFuContext", "constraint_fu_context"> {
  let summary = "correlated FU occurrence and instruction context carrier";
  let parameters = (ins "::mapping::FabricFuOccurrenceRefAttr":$fu,
      "::mapping::InstructionContextRefAttr":$instruction_context);
  let assemblyFormat = "`<` struct(params) `>`";
}

def Mapping_ConstraintUnsignedIntervalAttr
    : Mapping_Attr<"ConstraintUnsignedInterval",
                   "constraint_unsigned_interval"> {
  let summary = "one non-empty unsigned half-open interval";
  let parameters = (ins "::mlir::IntegerAttr":$lower,
      "::mlir::IntegerAttr":$upper);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Mapping_ConstraintAddressRegionAttr
    : Mapping_Attr<"ConstraintAddressRegion", "constraint_address_region"> {
  let summary = "one memory service and its canonical address intervals";
  let parameters = (ins "::mapping::FabricMemoryServiceRefAttr":$service,
      "::mlir::ArrayAttr":$intervals);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Mapping_FabricMemoryEngineTemplateRefAttr
    : Mapping_CanonicalBytesAttr<
          "FabricMemoryEngineTemplateRef", "fabric_memory_engine_template_ref",
          "Fabric-owned memory engine template reference">;

def Mapping_FabricMemoryEngineTemplateOperationPortRefAttr
    : Mapping_CanonicalBytesAttr<
          "FabricMemoryEngineTemplateOperationPortRef",
          "fabric_memory_engine_template_operation_port_ref",
          "Fabric-owned memory engine template operation-port reference">;

def Mapping_FabricMemoryEngineTemplateCapabilityAlternativeRefAttr
    : Mapping_CanonicalBytesAttr<
          "FabricMemoryEngineTemplateCapabilityAlternativeRef",
          "fabric_memory_engine_template_capability_alternative_ref",
          "Fabric-owned memory engine template capability-alternative "
          "reference">;

def Mapping_FabricMemoryEngineTemplateEndpointRefAttr
    : Mapping_CanonicalBytesAttr<
          "FabricMemoryEngineTemplateEndpointRef",
          "fabric_memory_engine_template_endpoint_ref",
          "Fabric-owned memory engine template token-endpoint reference">;

def Mapping_FabricMemoryEngineTemplateInternalConnectionRefAttr
    : Mapping_CanonicalBytesAttr<
          "FabricMemoryEngineTemplateInternalConnectionRef",
          "fabric_memory_engine_template_internal_connection_ref",
          "Fabric-owned memory engine template internal-connection reference">;

#endif // LOOM_MAPPING_IR_MAPPINGATTRS_TD

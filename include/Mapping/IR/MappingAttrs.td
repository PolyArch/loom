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
    : Mapping_CanonicalBytesAttr<"GraphRef", "graph_ref",
        "Dataflow-owned GraphRef scoped by the root dataflow binding">;

def Mapping_ActorRefAttr
    : Mapping_CanonicalBytesAttr<"ActorRef", "actor_ref",
        "Dataflow-owned ActorRef scoped by the root dataflow binding">;

def Mapping_LogicalMemoryRootRefAttr
    : Mapping_CanonicalBytesAttr<"LogicalMemoryRootRef", "logical_memory_root_ref",
        "Dataflow-owned LogicalMemoryRootRef scoped by the root binding">;

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
    : Mapping_CanonicalBytesAttr<"GraphProducerEndpointRef",
        "graph_producer_endpoint_ref",
        "Dataflow-owned canonical graph producer endpoint reference">;

def Mapping_GraphConsumerEndpointRefAttr
    : Mapping_CanonicalBytesAttr<"GraphConsumerEndpointRef",
        "graph_consumer_endpoint_ref",
        "Dataflow-owned canonical graph consumer endpoint reference">;

def Mapping_GraphEndpointRef : AnyAttrOf<[
    Mapping_GraphProducerEndpointRefAttr,
    Mapping_GraphConsumerEndpointRefAttr],
    "Dataflow-owned canonical graph producer or consumer endpoint reference">;

def Mapping_FabricFuCapabilityTemplateRefAttr
    : Mapping_CanonicalBytesAttr<"FabricFuCapabilityTemplateRef",
        "fabric_fu_capability_template_ref",
        "Fabric-owned FU capability template reference">;

def Mapping_FabricFuTemplateNodeRefAttr
    : Mapping_CanonicalBytesAttr<"FabricFuTemplateNodeRef",
        "fabric_fu_template_node_ref", "Fabric-owned FU template node reference">;

def Mapping_FabricFuTemplatePortRefAttr
    : Mapping_CanonicalBytesAttr<"FabricFuTemplatePortRef",
        "fabric_fu_template_port_ref", "Fabric-owned FU template port reference">;

def Mapping_FabricFuOccurrenceRefAttr
    : Mapping_CanonicalBytesAttr<"FabricFuOccurrenceRef",
        "fabric_fu_occurrence_ref", "Fabric-owned FU occurrence reference">;

def Mapping_FabricPeOccurrenceRefAttr
    : Mapping_CanonicalBytesAttr<"FabricPeOccurrenceRef",
        "fabric_pe_occurrence_ref", "Fabric-owned PE occurrence reference">;

def Mapping_InstructionContextRefAttr
    : Mapping_CanonicalBytesAttr<"InstructionContextRef",
        "instruction_context_ref", "Fabric-owned instruction context reference">;

def Mapping_FabricMemoryOccurrenceRefAttr
    : Mapping_CanonicalBytesAttr<"FabricMemoryOccurrenceRef",
        "fabric_memory_occurrence_ref", "Fabric-owned memory occurrence reference">;

def Mapping_FabricPhysicalTraversalRefAttr
    : Mapping_CanonicalBytesAttr<"FabricPhysicalTraversalRef",
        "fabric_physical_traversal_ref", "Fabric-owned physical traversal reference">;

def Mapping_FabricResourceStateRefAttr
    : Mapping_CanonicalBytesAttr<"FabricResourceStateRef",
        "fabric_resource_state_ref", "Fabric-owned resource state reference">;

def Mapping_FabricTransportEndpointRefAttr
    : Mapping_CanonicalBytesAttr<"FabricTransportEndpointRef",
        "fabric_transport_endpoint_ref", "Fabric-owned transport endpoint reference">;

def Mapping_FabricMemoryOperationPortRefAttr
    : Mapping_CanonicalBytesAttr<"FabricMemoryOperationPortRef",
        "fabric_memory_operation_port_ref", "Fabric-owned memory operation port reference">;

def Mapping_FabricMemoryServiceRefAttr
    : Mapping_CanonicalBytesAttr<"FabricMemoryServiceRef",
        "fabric_memory_service_ref", "Fabric-owned memory service reference">;

def Mapping_SpatialTransferTerminalAttr
    : Mapping_Attr<"SpatialTransferTerminal", "spatial_transfer_terminal"> {
  let summary = "source or exact sink terminal of one Spatial logical net";
  let parameters = (ins
      "::mapping::GraphProducerEndpointRefAttr":$producer,
      OptionalParameter<"::mapping::GraphConsumerEndpointRefAttr">:$consumer);
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Mapping_ConstraintFuContextAttr
    : Mapping_Attr<"ConstraintFuContext", "constraint_fu_context"> {
  let summary = "correlated FU occurrence and instruction context carrier";
  let parameters = (ins
      "::mapping::FabricFuOccurrenceRefAttr":$fu,
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
    : Mapping_CanonicalBytesAttr<"FabricMemoryEngineTemplateRef",
        "fabric_memory_engine_template_ref",
        "Fabric-owned memory engine template reference">;

def Mapping_FabricMemoryEngineTemplateOperationPortRefAttr
    : Mapping_CanonicalBytesAttr<"FabricMemoryEngineTemplateOperationPortRef",
        "fabric_memory_engine_template_operation_port_ref",
        "Fabric-owned memory engine template operation-port reference">;

def Mapping_FabricMemoryEngineTemplateCapabilityAlternativeRefAttr
    : Mapping_CanonicalBytesAttr<
        "FabricMemoryEngineTemplateCapabilityAlternativeRef",
        "fabric_memory_engine_template_capability_alternative_ref",
        "Fabric-owned memory engine template capability-alternative reference">;

def Mapping_FabricMemoryEngineTemplateEndpointRefAttr
    : Mapping_CanonicalBytesAttr<"FabricMemoryEngineTemplateEndpointRef",
        "fabric_memory_engine_template_endpoint_ref",
        "Fabric-owned memory engine template token-endpoint reference">;

def Mapping_FabricMemoryEngineTemplateInternalConnectionRefAttr
    : Mapping_CanonicalBytesAttr<
        "FabricMemoryEngineTemplateInternalConnectionRef",
        "fabric_memory_engine_template_internal_connection_ref",
        "Fabric-owned memory engine template internal-connection reference">;

#endif // LOOM_MAPPING_IR_MAPPINGATTRS_TD

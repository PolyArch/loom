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

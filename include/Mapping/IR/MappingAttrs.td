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

#endif // LOOM_MAPPING_IR_MAPPINGATTRS_TD

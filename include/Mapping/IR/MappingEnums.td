#ifndef LOOM_MAPPING_IR_MAPPINGENUMS_TD
#define LOOM_MAPPING_IR_MAPPINGENUMS_TD

include "Mapping/IR/MappingDialect.td"
include "mlir/IR/EnumAttr.td"

def Mapping_PortDirection : I32EnumAttr<"PortDirection",
    "software actor port direction", [
  I32EnumAttrCase<"Input", 0, "input">,
  I32EnumAttrCase<"Output", 1, "output">
]> {
  let cppNamespace = "::mapping";
}

def Mapping_PortDirectionAttr
    : EnumAttr<Mapping_Dialect, Mapping_PortDirection, "port_direction">;

def Mapping_SystemBindingRelationKind : I32EnumAttr<
    "SystemBindingRelationKind",
    "closed System execution-binding relation shape", [
  I32EnumAttrCase<"PresburgerPartition", 0, "presburger_partition">,
  I32EnumAttrCase<"StableKeyLookup", 1, "stable_key_lookup">
]> {
  let cppNamespace = "::mapping";
}

def Mapping_SystemBindingRelationKindAttr
    : EnumAttr<Mapping_Dialect, Mapping_SystemBindingRelationKind,
               "system_binding_relation_kind">;

def Mapping_SpatialConstraintProjection : I32EnumAttr<
    "SpatialConstraintProjection",
    "closed Spatial MappingConstraintSet projection catalog", [
  I32EnumAttrCase<"ComputePlacement", 0, "compute_placement">,
  I32EnumAttrCase<"ComputeParentPe", 1, "compute_parent_pe">,
  I32EnumAttrCase<"ComputeInstructionContext", 2,
                  "compute_instruction_context">,
  I32EnumAttrCase<"ComputeFuContext", 3, "compute_fu_context">,
  I32EnumAttrCase<"MemoryPlacement", 4, "memory_placement">,
  I32EnumAttrCase<"NetAssignedTagValues", 5,
                  "net_assigned_tag_values">,
  I32EnumAttrCase<"NetSelectedPhysicalTraversals", 6,
                  "net_selected_physical_traversals">,
  I32EnumAttrCase<"NetTraversalResourceStates", 7,
                  "net_traversal_resource_states">,
  I32EnumAttrCase<"SpatialTransferAttachment", 8,
                  "spatial_transfer_attachment">,
  I32EnumAttrCase<"MemoryOperationPort", 9, "memory_operation_port">,
  I32EnumAttrCase<"MemoryBoundServices", 10, "memory_bound_services">,
  I32EnumAttrCase<"MemoryAddressRegion", 11, "memory_address_region">
]> {
  let cppNamespace = "::mapping";
}

def Mapping_SystemConstraintProjection : I32EnumAttr<
    "SystemConstraintProjection",
    "closed System MappingConstraintSet projection catalog", [
  I32EnumAttrCase<"ThreadTargetAccCore", 0, "thread_target_acc_core">,
  I32EnumAttrCase<"GraphSelectedSpatialMapping", 1,
                  "graph_selected_spatial_mapping">,
  I32EnumAttrCase<"GraphTargetSpatialCore", 2,
                  "graph_target_spatial_core">,
  I32EnumAttrCase<"ServiceTargetRegion", 3, "service_target_region">,
  I32EnumAttrCase<"TransferTerminalAttachment", 4,
                  "transfer_terminal_attachment">,
  I32EnumAttrCase<"TransferSelectedTraversals", 5,
                  "transfer_selected_traversals">,
  I32EnumAttrCase<"TransferResourceStates", 6,
                  "transfer_resource_states">,
  I32EnumAttrCase<"TransferAssignedTagValues", 7,
                  "transfer_assigned_tag_values">
]> {
  let cppNamespace = "::mapping";
}

#endif // LOOM_MAPPING_IR_MAPPINGENUMS_TD

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

def Mapping_SpatialConstraintProjectionAttr
    : EnumAttr<Mapping_Dialect, Mapping_SpatialConstraintProjection,
               "spatial_projection">;

#endif // LOOM_MAPPING_IR_MAPPINGENUMS_TD

#ifndef FABRIC_ENUMS_TD
#define FABRIC_ENUMS_TD

include "Fabric/IR/FabricDialect.td"
include "mlir/IR/EnumAttr.td"
include "Fabric/IR/ImplementationFamilyEnum.td.inc"

def Fabric_ScheduleAttr : I32EnumAttr<"Schedule",
    "fabric container schedule (spatial vs temporal time-share)",
    [
      I32EnumAttrCase<"Spatial",  0, "spatial">,
      I32EnumAttrCase<"Temporal", 1, "temporal">
    ]> {
  let cppNamespace = "::fabric";
}

def Fabric_BoundaryDirectionAttr : I32EnumAttr<"BoundaryDirection",
    "fabric.boundary direction (s2t / t2t / t2s)",
    [
      I32EnumAttrCase<"S2t", 0, "s2t">,
      I32EnumAttrCase<"T2t", 1, "t2t">,
      I32EnumAttrCase<"T2s", 2, "t2s">
    ]> {
  let cppNamespace = "::fabric";
}

def Fabric_OperandBufferMode : I32EnumAttr<"OperandBufferMode",
    "temporal PE operand-buffer physical storage organization",
    [
      I32EnumAttrCase<"PerInstruction", 0, "per_instruction">,
      I32EnumAttrCase<"PerInputPort",   1, "per_input_port">,
      I32EnumAttrCase<"AllFuShare",     2, "all_fu_share">
    ]> {
  let cppNamespace = "::fabric";
  let genSpecializedAttr = 0;
}

def Fabric_OperandBufferModeAttr
    : EnumAttr<Fabric_Dialect, Fabric_OperandBufferMode,
               "operand_buffer_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

def Fabric_FifoQueueDiscipline : I32EnumAttr<"FifoQueueDiscipline",
    "dequeue scheduling discipline of one physical FIFO occurrence",
    [
      I32EnumAttrCase<"StrictFifo", 0, "strict_fifo">,
      I32EnumAttrCase<"PerTagVirtualChannel", 1, "per_tag_virtual_channel">
    ]> {
  let cppNamespace = "::fabric";
  let genSpecializedAttr = 0;
}

def Fabric_FifoQueueDisciplineAttr
    : EnumAttr<Fabric_Dialect, Fabric_FifoQueueDiscipline,
               "fifo_queue_discipline"> {
  let assemblyFormat = "`<` $value `>`";
}

def Fabric_FuConfigMode : I32EnumAttr<"FuConfigMode",
    "temporal PE FU configuration storage organization",
    [
      I32EnumAttrCase<"PerInstructionFuConfig", 0,
                      "per_instruction_fu_config">,
      I32EnumAttrCase<"PerFuConfig", 1, "per_fu_config">
    ]> {
  let cppNamespace = "::fabric";
  let genSpecializedAttr = 0;
}

def Fabric_FuConfigModeAttr
    : EnumAttr<Fabric_Dialect, Fabric_FuConfigMode, "fu_config_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

def Fabric_InterconnectProtocolSchemaAttr
    : I32EnumAttr<"InterconnectProtocolSchema",
        "closed interconnect implementation protocol schema",
        [
          I32EnumAttrCase<"Gem5EventTransportV1", 0,
                          "gem5_event_transport_v1">
        ]> {
  let cppNamespace = "::fabric";
}

def Fabric_InterconnectEndpointDirectionAttr
    : I32EnumAttr<"InterconnectEndpointDirection",
        "direction of one protocol endpoint",
        [
          I32EnumAttrCase<"Ingress", 0, "ingress">,
          I32EnumAttrCase<"Egress", 1, "egress">
        ]> {
  let cppNamespace = "::fabric";
}

def Fabric_InterconnectRefinementKindAttr
    : I32EnumAttr<"InterconnectRefinementKind",
        "closed architecture-to-protocol refinement relation",
        [
          I32EnumAttrCase<"Endpoint", 0, "endpoint">,
          I32EnumAttrCase<"ResourceState", 1, "resource_state">,
          I32EnumAttrCase<"TransferPattern", 2, "transfer_pattern">,
          I32EnumAttrCase<"Configuration", 3, "configuration">
        ]> {
  let cppNamespace = "::fabric";
}

#endif // FABRIC_ENUMS_TD

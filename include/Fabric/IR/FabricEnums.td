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

#endif // FABRIC_ENUMS_TD

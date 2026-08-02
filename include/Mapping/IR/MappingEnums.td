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

#endif // LOOM_MAPPING_IR_MAPPINGENUMS_TD

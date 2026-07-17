#ifndef DATAFLOW_ENUMS_TD
#define DATAFLOW_ENUMS_TD

include "Dataflow/IR/DataflowDialect.td"
include "mlir/IR/EnumAttr.td"

def Dataflow_StreamStepAdd : I32EnumAttrCase<"Add", 0, "add">;
def Dataflow_StreamStepSub : I32EnumAttrCase<"Sub", 1, "sub">;
def Dataflow_StreamStepMul : I32EnumAttrCase<"Mul", 2, "mul">;
def Dataflow_StreamStepSDiv : I32EnumAttrCase<"SDiv", 3, "sdiv">;
def Dataflow_StreamStepUDiv : I32EnumAttrCase<"UDiv", 4, "udiv">;
def Dataflow_StreamStepShL : I32EnumAttrCase<"ShL", 5, "shl">;
def Dataflow_StreamStepAShr : I32EnumAttrCase<"AShr", 6, "ashr">;
def Dataflow_StreamStepLShr : I32EnumAttrCase<"LShr", 7, "lshr">;

def Dataflow_StreamStepKindAttr : I32EnumAttr<
    "StreamStepKind", "integer recurrence update operation",
    [Dataflow_StreamStepAdd, Dataflow_StreamStepSub,
     Dataflow_StreamStepMul, Dataflow_StreamStepSDiv,
     Dataflow_StreamStepUDiv, Dataflow_StreamStepShL,
     Dataflow_StreamStepAShr, Dataflow_StreamStepLShr]> {
  let cppNamespace = "::dataflow";
}

#endif // DATAFLOW_ENUMS_TD

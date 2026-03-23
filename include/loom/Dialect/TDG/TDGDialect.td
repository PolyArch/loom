#ifndef LOOM_DIALECT_TDG_TDGDIALECT_TD
#define LOOM_DIALECT_TDG_TDGDIALECT_TD

include "mlir/IR/OpBase.td"

def TDG_Dialect : Dialect {
  let name = "tdg";
  let cppNamespace = "::loom::tdg";
  let summary = "Task Dataflow Graph dialect for multi-kernel dataflow applications";
  let description = [{
    The TDG dialect represents multi-kernel dataflow applications as a
    task graph with explicit contracts on inter-kernel communication edges.
    Each kernel is a computational unit; contracts specify ordering,
    rate, buffering, visibility, and transformation permissions.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;
  let usePropertiesForAttributes = 0;
}

#endif

#ifndef LOOM_DIALECT_DATAFLOW_DATAFLOWDIALECT_TD
#define LOOM_DIALECT_DATAFLOW_DATAFLOWDIALECT_TD

include "mlir/IR/OpBase.td"

def Dataflow_Dialect : Dialect {
  let name = "dataflow";
  let cppNamespace = "::loom::dataflow";
  let summary = "Dataflow operations for loop-level transformations";
  let description = [{
    The dataflow dialect provides operations for expressing loop-level
    dataflow transformations: stream generation, gating, loop-carried
    dependencies, and loop-invariant replication.
  }];
  let useDefaultTypePrinterParser = 0;
  let useDefaultAttributePrinterParser = 0;
  let usePropertiesForAttributes = 0;
}

#endif

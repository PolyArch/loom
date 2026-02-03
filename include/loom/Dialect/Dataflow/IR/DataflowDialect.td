//===- DataflowDialect.td - Dataflow dialect definition -------*- tablegen -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_DATAFLOW_DATAFLOWDIALECT_TD
#define LOOM_DIALECT_DATAFLOW_DATAFLOWDIALECT_TD

include "mlir/IR/DialectBase.td"

// Dataflow dialect for loop-carried stream semantics.
def Dataflow_Dialect : Dialect {
  let name = "dataflow";
  let cppNamespace = "::loom::dataflow";
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

#endif // LOOM_DIALECT_DATAFLOW_DATAFLOWDIALECT_TD

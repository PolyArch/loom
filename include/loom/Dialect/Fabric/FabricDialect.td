//===- FabricDialect.td - Fabric dialect definition ----------*- tablegen -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_FABRIC_FABRICDIALECT_TD
#define LOOM_DIALECT_FABRIC_FABRICDIALECT_TD

include "mlir/IR/DialectBase.td"

def Fabric_Dialect : Dialect {
  let name = "fabric";
  let cppNamespace = "::loom::fabric";
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
  let usePropertiesForAttributes = 0;
  let dependentDialects = ["::loom::dataflow::DataflowDialect"];
}

#endif // LOOM_DIALECT_FABRIC_FABRICDIALECT_TD

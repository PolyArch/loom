#ifndef FCC_DIALECT_FABRIC_FABRICDIALECT_TD
#define FCC_DIALECT_FABRIC_FABRICDIALECT_TD

include "mlir/IR/OpBase.td"

def Fabric_Dialect : Dialect {
  let name = "fabric";
  let cppNamespace = "::fcc::fabric";
  let summary = "Fabric dialect for CGRA architecture description";
  let description = [{
    The fabric dialect describes CGRA hardware architecture: processing
    elements, switches, FIFOs, memory interfaces, and their connectivity.
    Types (!fabric.bits<N>, !fabric.tagged<T, iK>) live in this namespace.
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 0;
  let usePropertiesForAttributes = 0;
  let dependentDialects = [
    "::fcc::dataflow::DataflowDialect"
  ];
  let hasNonDefaultDestructor = 0;
}

// Base class for all Fabric ops
class Fabric_Op<string mnemonic, list<Trait> traits = []>
    : Op<Fabric_Dialect, mnemonic, traits>;

#endif

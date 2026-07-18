#ifndef FABRIC_DIALECT_TD
#define FABRIC_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def Fabric_Dialect : Dialect {
  let name = "fabric";
  let summary = "Fabric dialect for reconfigurable CGRA architecture description";
  let cppNamespace = "::fabric";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class Fabric_Op<string mnemonic, list<Trait> traits = []>
    : Op<Fabric_Dialect, mnemonic, traits>;

class Fabric_Type<string name, string typeMnemonic,
                  list<Trait> traits = []>
    : TypeDef<Fabric_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // FABRIC_DIALECT_TD

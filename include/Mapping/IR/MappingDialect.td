#ifndef LOOM_MAPPING_IR_MAPPINGDIALECT_TD
#define LOOM_MAPPING_IR_MAPPINGDIALECT_TD

include "mlir/IR/OpBase.td"

def Mapping_Dialect : Dialect {
  let name = "mapping";
  let cppNamespace = "::mapping";
  let summary = "persistent Dataflow-to-Fabric mapping artifacts";
  let description = [{
    The mapping dialect is the typed persistent wire for complete Tech,
    Spatial, and System Mapping artifacts and their exact constraint sets.
    Mutable search state and native hot indexes are not part of this dialect.
  }];
  let useDefaultAttributePrinterParser = 1;
}

class Mapping_Op<string mnemonic, list<Trait> traits = []>
    : Op<Mapping_Dialect, mnemonic, traits>;

#endif // LOOM_MAPPING_IR_MAPPINGDIALECT_TD

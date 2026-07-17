#ifndef LOOM_FRONTEND_IR_LOOMDIALECT_TD
#define LOOM_FRONTEND_IR_LOOMDIALECT_TD

include "mlir/IR/OpBase.td"

def Loom_Dialect : Dialect {
  let name = "loom";
  let summary = "Temporary structured compiler boundaries";
  let description = [{
    The Loom frontend dialect owns transient structured candidates that must
    be eliminated before a canonical Dataflow Program is published.
  }];
  let cppNamespace = "::loom";
}

class Loom_Op<string mnemonic, list<Trait> traits = []>
    : Op<Loom_Dialect, mnemonic, traits>;

#endif // LOOM_FRONTEND_IR_LOOMDIALECT_TD

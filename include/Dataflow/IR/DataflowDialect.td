#ifndef DATAFLOW_DIALECT_TD
#define DATAFLOW_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def Dataflow_Dialect : Dialect {
  let name = "dataflow";
  let summary =
      "Dataflow dialect for software-level control+data flow semantics";
  let description = [{
    The dataflow dialect describes producer/consumer relations between
    values that conceptually carry a *stream* of data. Inside a
    dataflow graph region, values are not SSA in the classical sense:
    feedback / backward references are allowed.
  }];
  let cppNamespace = "::dataflow";
}

class Dataflow_Op<string mnemonic, list<Trait> traits = []>
    : Op<Dataflow_Dialect, mnemonic, traits>;

#endif // DATAFLOW_DIALECT_TD

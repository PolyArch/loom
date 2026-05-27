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
  let useDefaultTypePrinterParser = 1;
}

class Dataflow_Op<string mnemonic, list<Trait> traits = []>
    : Op<Dataflow_Dialect, mnemonic, traits>;

class Dataflow_Type<string name, string typeMnemonic,
                    list<Trait> traits = []>
    : TypeDef<Dataflow_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Dataflow_ThreadTokenType
    : Dataflow_Type<"ThreadToken", "thread_token"> {
  let summary = "one-shot async completion handle for a dataflow.thread.launch";
  let description = [{
    A one-shot completion handle returned by `dataflow.thread.launch`
    and consumed by `dataflow.thread.wait` or
    `dataflow.thread.fence`. Equivalent to `!async.token` but local to
    the Loom dataflow dialect.
  }];
  let assemblyFormat = "";
}

#endif // DATAFLOW_DIALECT_TD

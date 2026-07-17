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
  let extraClassDeclaration = [{
    static bool isMemoryCapabilityType(::mlir::Type type);
    static bool containsMemoryCapability(::mlir::Type type);
    static bool containsChannelOrThreadToken(::mlir::Type type);
  }];
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
    and consumed by launch dependencies or `dataflow.thread.wait`.
    It represents collective retirement of all dynamic instances from
    one launch. Equivalent to `!async.token` but local to the Loom
    dataflow dialect.
  }];
  let assemblyFormat = "";
}

def Dataflow_ChannelType : Dataflow_Type<"Channel", "channel"> {
  let summary = "typed handle for ordered thread-level message transport";
  let description = [{
    A thread-level channel handle carrying ordered messages of one exact
    element type. Memory capabilities, channel handles, and thread completion
    tokens are distinct carriers and cannot be used as channel message
    payloads.
  }];
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let genVerifyDecl = 1;
}

#endif // DATAFLOW_DIALECT_TD

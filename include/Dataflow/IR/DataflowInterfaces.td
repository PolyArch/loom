#ifndef DATAFLOW_INTERFACES_TD
#define DATAFLOW_INTERFACES_TD

include "mlir/IR/OpBase.td"

def CanonicalDataflowActor : OpInterface<"CanonicalDataflowActorOpInterface"> {
  let cppNamespace = "::dataflow";
  let description = [{
    Marks an operation with complete typed per-firing semantics that may appear
    as an actor in a finalized canonical dataflow graph.
  }];
}

#endif // DATAFLOW_INTERFACES_TD

#ifndef DATAFLOW_INTERFACES_TD
#define DATAFLOW_INTERFACES_TD

include "mlir/IR/OpBase.td"

def CanonicalDataflowActor : OpInterface<"CanonicalDataflowActorOpInterface"> {
  let cppNamespace = "::dataflow";
  let description = [{
    The typed projection an operation admitted as a Canonical Dataflow actor
    presents to the rest of the stack. The Dataflow dialect's own actors
    implement it directly; every other registered operation implements it
    through an external model.

    The interface projects only the Loom contract: the registered
    `OperationSchemaId`, the actor kind, the closed semantic attribute
    projection, the instance verifier, and the transition descriptor identity.
    The operation's native definition remains the owner of its source
    semantics; this interface never copies an attribute dictionary.

    `include/Dataflow/IR/OperationSchemas.td` remains the one registration and
    classification authority. Implementing this interface is how an operation
    projects, not a second admission list: the default implementations answer
    from that registry, so an operation cannot project a schema the registry
    does not declare for it.
  }];

  let methods = [
    InterfaceMethod<[{
        The registered schema of this actor instance.
      }],
      "::dataflow::OperationSchemaId", "getOperationSchema", (ins), "", [{
        return ::dataflow::requireOperationSchema($_op.getOperation());
      }]>,
    InterfaceMethod<[{
        The canonical actor kind of this actor instance.
      }],
      "::dataflow::CanonicalDataflowActorKind", "getCanonicalActorKind",
      (ins), "", [{
        return ::dataflow::actorKind(
            ::dataflow::requireOperationSchema($_op.getOperation()));
      }]>,
    InterfaceMethod<[{
        The complete identity-critical typed projection of this instance: the
        registered schema, the exact function type, and the one closed typed
        semantic payload the schema's declared case owns.

        Returns failure when the instance does not carry the exact typed state
        its declared semantic case requires. A projection never falls back to
        an empty payload.
      }],
      "::llvm::Expected<::dataflow::CanonicalActorSemantics>",
      "projectCanonicalActorSemantics", (ins), "", [{
        return ::dataflow::projectRegisteredActorSemantics(
            $_op.getOperation());
      }]>,
    InterfaceMethod<[{
        Verifies that this instance carries exactly the typed state its
        registered schema and semantic case declare. It fails closed.
      }],
      "::mlir::LogicalResult", "verifyCanonicalActorInstance", (ins), "", [{
        return ::dataflow::verifyRegisteredActorInstance($_op.getOperation());
      }]>,
    InterfaceMethod<[{
        The transition descriptor identity of this actor instance. Two
        instances share a descriptor exactly when their typed projections are
        equal, so a consumer may key one transition implementation on it.
      }],
      "::llvm::Expected<::dataflow::TransitionDescriptorIdentity>",
      "getTransitionDescriptorIdentity", (ins), "", [{
        return ::dataflow::deriveTransitionDescriptorIdentity(
            $_op.getOperation());
      }]>,
  ];
}

#endif // DATAFLOW_INTERFACES_TD

//===- OperationSchemaBase.td - Registry declaration classes --------------===//
//
// The declaration classes of Loom's canonical operation authority. They are
// separated from the registry records so the generator's own contract can be
// anchored against a small declared registry without copying a second set of
// classes.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_OPERATION_SCHEMA_BASE_TD
#define LOOM_OPERATION_SCHEMA_BASE_TD

//===----------------------------------------------------------------------===//
// Declaration classes
//===----------------------------------------------------------------------===//

// One member of the closed typed semantic vocabulary. The case selects which
// typed reader produces an actor's identity-critical semantic payload; no
// semantic value is ever reached through an attribute name.
class SemanticsCase<int id, int wireTag> {
  int caseId = id;
  int stableWireTag = wireTag;
}

// One closed selector for an instance carried by an operation class. Most
// schemas select the whole class. A generic carrier may instead select one
// source-owner identity, but selectors for one class must remain disjoint.
class OperationInstanceSelector<int id, string kind, string value> {
  int selectorId = id;
  string selectorKind = kind;
  string selectorValue = value;
}

def WholeOperationClass : OperationInstanceSelector<
    0, "WholeOperationClass", "not_intrinsic">;

class LLVMIntrinsicSelector<string intrinsic> : OperationInstanceSelector<
    1, "LLVMIntrinsic", intrinsic>;

// One registered canonical actor operation schema. `cls` is the operation
// class that carries the source semantics. `selector` identifies the exact
// source-owned instance when that class is a generic carrier.
class ActorSchema<int id, int wireTag, string cls, string kind,
                  SemanticsCase case,
                  OperationInstanceSelector selector = WholeOperationClass,
                  bit elementwiseDecomposable = 0> {
  int schemaId = id;
  int stableWireTag = wireTag;
  string opClass = cls;
  string actorKind = kind;
  SemanticsCase semantics = case;
  OperationInstanceSelector instanceSelector = selector;
  bit supportsElementwiseVectorDecomposition = elementwiseDecomposable;
}

// A pure software actor whose fixed-vector instance is exactly the pointwise
// lift of the same scalar operation. This is a software semantic fact, not an
// implementation-family admission rule.
class ElementwiseActorSchema<
    int id, int wireTag, string cls, string kind, SemanticsCase case,
    OperationInstanceSelector selector = WholeOperationClass>
    : ActorSchema<id, wireTag, cls, kind, case, selector, 1>;

// One closed typed `hw_params` record schema an implementation family selects.
// Every schema names one concrete C++ record; there is no generic record or
// property bag.
class CapabilityParamsSchema<int id> {
  int paramsId = id;
}

// One typed admission rule. A family descriptor selects a provider by this
// identity and the generated dispatch resolves it to the focused C++ rule;
// nothing chooses a rule by switching on a family.
class TypedAdmissionProvider<int id> {
  int providerId = id;
}

// One registered implementation family. `members` names schemas by record
// reference, so a dangling member cannot be written.
class ImplementationFamily<int id, list<ActorSchema> members,
                           CapabilityParamsSchema params,
                           TypedAdmissionProvider admission> {
  int familyId = id;
  list<ActorSchema> admittedSchemas = members;
  CapabilityParamsSchema capabilityParams = params;
  TypedAdmissionProvider typedAdmission = admission;
}

#endif // LOOM_OPERATION_SCHEMA_BASE_TD

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

// One registered canonical actor operation schema. `cls` is the operation
// class that owns the source semantics and the stable spelling.
class ActorSchema<int id, int wireTag, string cls, string kind,
                  SemanticsCase case> {
  int schemaId = id;
  int stableWireTag = wireTag;
  string opClass = cls;
  string actorKind = kind;
  SemanticsCase semantics = case;
}

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

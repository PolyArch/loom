// The same declared registry as `declaration_order_a.td`, written in a
// different textual order. TableGen requires a record to be defined before it
// is referenced, so the permutation reorders every independent declaration and
// each family relative to its own members.

include "Dataflow/IR/OperationSchemaBase.td"

def PlainAdmission : TypedAdmissionProvider<0>;
def PlainParams : CapabilityParamsSchema<0>;

def PredicateCase : SemanticsCase<1, 41>;
def PlainCase : SemanticsCase<0, 103>;

def GammaSchema : ActorSchema<2, 500, "::test::GammaOp", "Control", PlainCase>;

def LoneGamma : ImplementationFamily<1, [GammaSchema],
    PlainParams, PlainAdmission>;

def PredicateParams : CapabilityParamsSchema<1>;
def PredicateAdmission : TypedAdmissionProvider<1>;

def BetaSchema : ActorSchema<1, 73, "::test::BetaOp", "Compute", PredicateCase>;
def AlphaSchema : ActorSchema<0, 9001, "::test::AlphaOp", "Compute", PlainCase>;

def SharedAlphaBeta : ImplementationFamily<0, [AlphaSchema, BetaSchema],
    PredicateParams, PredicateAdmission>;

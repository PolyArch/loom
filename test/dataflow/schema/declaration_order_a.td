// One small declared registry. `declaration_order_b.td` declares exactly the
// same records in a different textual order; the generator must emit identical
// bytes for both, so declaration order is not an authority.

include "Dataflow/IR/OperationSchemaBase.td"

def PlainCase : SemanticsCase<0, 103>;
def PredicateCase : SemanticsCase<1, 41>;

def AlphaSchema : ActorSchema<0, 9001, "::test::AlphaOp", "Compute", PlainCase>;
def BetaSchema : ActorSchema<1, 73, "::test::BetaOp", "Compute", PredicateCase>;
def GammaSchema : ActorSchema<2, 500, "::test::GammaOp", "Control", PlainCase>;

def PlainParams : CapabilityParamsSchema<0>;
def PredicateParams : CapabilityParamsSchema<1>;

def PlainAdmission : TypedAdmissionProvider<0>;
def PredicateAdmission : TypedAdmissionProvider<1>;

def SharedAlphaBeta : ImplementationFamily<0, [AlphaSchema, BetaSchema],
    PredicateParams, PredicateAdmission>;
def LoneGamma : ImplementationFamily<1, [GammaSchema],
    PlainParams, PlainAdmission>;

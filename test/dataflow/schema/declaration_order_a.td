// One small declared registry. `declaration_order_b.td` declares exactly the
// same records in a different textual order; the generator must emit identical
// bytes for both, so declaration order is not an authority.

include "Dataflow/IR/OperationSchemaBase.td"

def PlainCase : SemanticsCase<0>;
def PredicateCase : SemanticsCase<1>;

def AlphaSchema : ActorSchema<0, "::test::AlphaOp", "Compute", PlainCase>;
def BetaSchema : ActorSchema<1, "::test::BetaOp", "Compute", PredicateCase>;
def GammaSchema : ActorSchema<2, "::test::GammaOp", "Control", PlainCase>;

def PlainParams : CapabilityParamsSchema<0>;
def PredicateParams : CapabilityParamsSchema<1>;

def PlainAdmission : TypedAdmissionProvider<0>;
def PredicateAdmission : TypedAdmissionProvider<1>;

def SharedAlphaBeta : ImplementationFamily<0, [AlphaSchema, BetaSchema],
    PredicateParams, PredicateAdmission>;
def LoneGamma : ImplementationFamily<1, [GammaSchema],
    PlainParams, PlainAdmission>;

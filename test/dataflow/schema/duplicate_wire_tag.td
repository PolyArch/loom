include "Dataflow/IR/OperationSchemaBase.td"

def PlainCase : SemanticsCase<0, 103>;
def AlphaSchema : ActorSchema<0, 9001, "::test::AlphaOp", "Compute", PlainCase>;
def BetaSchema : ActorSchema<1, 9001, "::test::BetaOp", "Compute", PlainCase>;

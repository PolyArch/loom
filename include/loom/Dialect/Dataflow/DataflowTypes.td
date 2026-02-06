//===- DataflowTypes.td - Dataflow type definitions ---------*- tablegen -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_DATAFLOW_DATAFLOWTYPES_TD
#define LOOM_DIALECT_DATAFLOW_DATAFLOWTYPES_TD

include "loom/Dialect/Dataflow/DataflowDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

class Dataflow_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Dataflow_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Dataflow_TaggedType : Dataflow_Type<"Tagged", "tagged"> {
  let summary = "Tagged value type pairing a native value with a small integer tag";
  let description = [{
    A tagged value pairs a native value type with a small integer tag.
    The tag is carried alongside the value as part of the data payload.

    Allowed value types: i1, i8, i16, i32, i64, bf16, f16, f32, f64, index.
    Tag type must be a signless integer with width 1-16.

    Example: `!dataflow.tagged<i32, i4>`, `!dataflow.tagged<f32, i5>`
  }];

  let parameters = (ins "::mlir::Type":$valueType, "::mlir::IntegerType":$tagType);
  let assemblyFormat = "`<` $valueType `,` $tagType `>`";
  let genVerifyDecl = 1;
}

#endif // LOOM_DIALECT_DATAFLOW_DATAFLOWTYPES_TD

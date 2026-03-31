#ifndef FCC_DIALECT_FABRIC_FABRICTYPES_TD
#define FCC_DIALECT_FABRIC_FABRICTYPES_TD

include "fcc/Dialect/Fabric/FabricDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"

class Fabric_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Fabric_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// !fabric.bits<N> - width-only routing type
def Fabric_BitsType : Fabric_Type<"Bits", "bits"> {
  let summary = "Width-only data type for routing nodes";
  let description = [{
    Represents a fixed-width bit vector with no semantic type info.
    Used by switches, FIFOs, tag operations, and PE ports.
    Width must be 1-4096 bits. Minimum network granularity is 8 bits.
  }];
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
  let genVerifyDecl = 1;
}

// !fabric.tagged<T, iK> - tagged value type
def Fabric_TaggedType : Fabric_Type<"Tagged", "tagged"> {
  let summary = "Value paired with a small integer tag";
  let description = [{
    Pairs a native/bits value with a small integer tag for temporal
    routing. Tag width is 1-16 bits.
  }];
  let parameters = (ins
    "::mlir::Type":$valueType,
    "::mlir::IntegerType":$tagType
  );
  let assemblyFormat = "`<` $valueType `,` $tagType `>`";
  let genVerifyDecl = 1;
}

#endif

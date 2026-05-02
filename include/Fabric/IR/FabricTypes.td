#ifndef FABRIC_TYPES_TD
#define FABRIC_TYPES_TD

include "Fabric/IR/FabricDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def Fabric_BitsType : Fabric_Type<"Bits", "bits", [
    MemRefElementTypeInterface
]> {
  let summary = "fabric bits type (BW >= 0), with implicit valid/ready handshake";
  let parameters = (ins "unsigned":$width);
  let assemblyFormat = "`<` $width `>`";
}

def Fabric_BitsTagType : Fabric_Type<"BitsTag", "bits_tag"> {
  let summary = "fabric bits+tag type (BW >= 0, TW > 0), with implicit valid/ready handshake";
  let parameters = (ins "unsigned":$width, "unsigned":$tagWidth);
  let assemblyFormat = "`<` $width `,` $tagWidth `>`";
  let genVerifyDecl = 1;
}

def Fabric_AnyType
    : AnyTypeOf<[Fabric_BitsType, Fabric_BitsTagType]>;

#endif // FABRIC_TYPES_TD

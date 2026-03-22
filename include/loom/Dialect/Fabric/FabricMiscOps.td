#ifndef LOOM_DIALECT_FABRIC_FABRICMISCOPS_TD
#define LOOM_DIALECT_FABRIC_FABRICMISCOPS_TD

include "loom/Dialect/Fabric/FabricDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"


// fabric.add_tag - attach constant tag to value
def Fabric_AddTagOp : Fabric_Op<"add_tag", []> {
  let summary = "Attach constant tag to a value";
  let arguments = (ins
    AnyType:$value,
    OptionalAttr<APIntAttr>:$tag
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
  let hasVerifier = 1;
}

// fabric.del_tag - remove tag from tagged value
def Fabric_DelTagOp : Fabric_Op<"del_tag", []> {
  let summary = "Remove tag from tagged value";
  let arguments = (ins AnyType:$tagged);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $tagged attr-dict `:` type($tagged) `->` type($result)
  }];
  let hasVerifier = 1;
}

// fabric.map_tag - transform tag via lookup table
def Fabric_MapTagOp : Fabric_Op<"map_tag", []> {
  let summary = "Transform tag via lookup table";
  let arguments = (ins
    AnyType:$tagged,
    I64Attr:$table_size,
    OptionalAttr<ArrayAttr>:$table
  );
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif

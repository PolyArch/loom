//===- FabricTagOps.td - Fabric tag operations ---------------*- tablegen -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_DIALECT_FABRIC_FABRICTAGOPS_TD
#define LOOM_DIALECT_FABRIC_FABRICTAGOPS_TD

//===----------------------------------------------------------------------===//
// fabric.add_tag
//
// Spec syntax:
//   %tagged = fabric.add_tag %value {tag = 5 : i4} : T -> !dataflow.tagged<T, i4>
//===----------------------------------------------------------------------===//

def Fabric_AddTagOp : Fabric_Op<"add_tag", []> {
  let summary = "Attach a constant tag to a native value";
  let description = [{
    Attaches a constant tag to a native value, producing a tagged value.
    The tag is placed in the high bits and the value in the low bits.
  }];

  let arguments = (ins AnyType:$value, OptionalAttr<APIntAttr>:$tag);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// fabric.del_tag
//
// Spec syntax:
//   %value = fabric.del_tag %tagged : !dataflow.tagged<T, iN> -> T
//===----------------------------------------------------------------------===//

def Fabric_DelTagOp : Fabric_Op<"del_tag", []> {
  let summary = "Remove tag from a tagged value";
  let description = [{
    Removes the tag from a tagged value and forwards the value unchanged.
    The tag is discarded.
  }];

  let arguments = (ins AnyType:$tagged);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $tagged attr-dict `:` type($tagged) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// fabric.map_tag
//
// Spec syntax:
//   %retagged = fabric.map_tag %tagged
//     {table_size = 10, table = [[1 : i1, 5 : i7, 2 : i3], ...]}
//     : !dataflow.tagged<T, i7> -> !dataflow.tagged<T, i3>
//===----------------------------------------------------------------------===//

def Fabric_MapTagOp : Fabric_Op<"map_tag", []> {
  let summary = "Transform tag via lookup table";
  let description = [{
    Transforms the tag of a tagged value using a runtime-configurable lookup
    table while forwarding the value unchanged. Tag width may change between
    input and output.
  }];

  let arguments = (ins AnyType:$tagged,
                       I64Attr:$table_size,
                       OptionalAttr<ArrayAttr>:$table);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $tagged attr-dict `:` type($tagged) `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif // LOOM_DIALECT_FABRIC_FABRICTAGOPS_TD

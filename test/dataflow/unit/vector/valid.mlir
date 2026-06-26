// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @parallelize_pack_i8x4
func.func @parallelize_pack_i8x4(%data: i8, %cont: i1) -> (i8, i8, i8, i8, i4, i32) {
  // CHECK: dataflow.parallelize
  %lane0, %lane1, %lane2, %lane3, %mask =
    dataflow.parallelize %data, %cont {vec_size = 4 : i64}
      : (i8, i1) -> (i8, i8, i8, i8, i4)
  // CHECK: dataflow.pack
  %packed = dataflow.pack %lane0, %lane1, %lane2, %lane3 mask %mask
      {vec_size = 4 : i64} : (i8, i8, i8, i8, i4) -> i32
  return %lane0, %lane1, %lane2, %lane3, %mask, %packed
      : i8, i8, i8, i8, i4, i32
}

// CHECK-LABEL: @unpack_serialize_i8x4
func.func @unpack_serialize_i8x4(%packed: i32, %mask: i4) -> (i8, i8, i8, i8, i8, i1) {
  // CHECK: dataflow.unpack
  %lane0, %lane1, %lane2, %lane3 =
    dataflow.unpack %packed, %mask {vec_size = 4 : i64}
      : (i32, i4) -> (i8, i8, i8, i8)
  // CHECK: dataflow.serialize
  %data, %cont = dataflow.serialize %lane0, %lane1, %lane2, %lane3 mask %mask
      {vec_size = 4 : i64} : (i8, i8, i8, i8, i4) -> (i8, i1)
  return %lane0, %lane1, %lane2, %lane3, %data, %cont
      : i8, i8, i8, i8, i8, i1
}

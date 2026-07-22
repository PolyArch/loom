// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: @canonical_i8x3_boundary
func.func @canonical_i8x3_boundary(
    %data: i8, %scalar_phase: i1) -> (vector<3xi8>, vector<3xi1>, i1,
                                      i24, i3, i8, i1) {
  // CHECK: dataflow.parallelize
  %vector, %mask, %group_phase =
    dataflow.parallelize %data, %scalar_phase
      : (i8, i1) -> (vector<3xi8>, vector<3xi1>, i1)
  // CHECK: dataflow.pack
  %packed = dataflow.pack %vector : vector<3xi8> -> i24
  %packed_mask = dataflow.pack %mask : vector<3xi1> -> i3
  // CHECK: dataflow.serialize
  %scalar, %next_scalar_phase =
    dataflow.serialize %vector, %mask, %group_phase
      : (vector<3xi8>, vector<3xi1>, i1) -> (i8, i1)
  return %vector, %mask, %group_phase, %packed, %packed_mask, %scalar,
      %next_scalar_phase
      : vector<3xi8>, vector<3xi1>, i1, i24, i3, i8, i1
}

// CHECK-LABEL: @float_i96_roundtrip
func.func @float_i96_roundtrip(%packed: i96) -> i96 {
  // CHECK: dataflow.unpack
  %vector = dataflow.unpack %packed : i96 -> vector<3xf32>
  // CHECK: dataflow.pack
  %roundtrip = dataflow.pack %vector : vector<3xf32> -> i96
  return %roundtrip : i96
}

// CHECK-LABEL: @rank_two_i48_roundtrip
func.func @rank_two_i48_roundtrip(%packed: i48) -> i48 {
  // CHECK: dataflow.unpack %{{.*}} : i48 -> vector<2x3xi8>
  %vector = dataflow.unpack %packed : i48 -> vector<2x3xi8>
  // CHECK: dataflow.pack %{{.*}} : vector<2x3xi8> -> i48
  %roundtrip = dataflow.pack %vector : vector<2x3xi8> -> i48
  return %roundtrip : i48
}

// CHECK-LABEL: @rank_three_f32_roundtrip
func.func @rank_three_f32_roundtrip(%packed: i256) -> i256 {
  // CHECK: dataflow.unpack %{{.*}} : i256 -> vector<2x2x2xf32>
  %vector = dataflow.unpack %packed : i256 -> vector<2x2x2xf32>
  // CHECK: dataflow.pack %{{.*}} : vector<2x2x2xf32> -> i256
  %roundtrip = dataflow.pack %vector : vector<2x2x2xf32> -> i256
  return %roundtrip : i256
}

// CHECK-LABEL: @mask_i65_roundtrip
func.func @mask_i65_roundtrip(%packed: i65) -> i65 {
  // CHECK: dataflow.unpack
  %mask = dataflow.unpack %packed : i65 -> vector<65xi1>
  // CHECK: dataflow.pack
  %roundtrip = dataflow.pack %mask : vector<65xi1> -> i65
  return %roundtrip : i65
}

// CHECK-LABEL: @standard_vector_compute
func.func @standard_vector_compute(
    %lhs: vector<3xi32>, %rhs: vector<3xi32>, %value: vector<3xf32>)
    -> (vector<3xi32>, vector<3xi1>, vector<3xf32>) {
  // CHECK: arith.addi
  %sum = arith.addi %lhs, %rhs : vector<3xi32>
  // CHECK: arith.cmpi
  %ordered = arith.cmpi slt, %lhs, %rhs : vector<3xi32>
  // CHECK: math.sqrt
  %root = math.sqrt %value : vector<3xf32>
  return %sum, %ordered, %root
      : vector<3xi32>, vector<3xi1>, vector<3xf32>
}

// Anchors for the PackUnpackRoundTripEliminate typed rewrite. The bit
// representation boundary owns `unpack(pack(v)) = v` and `pack(unpack(i)) = i`
// only when the intermediate result has no other consumer.

// The source program already has the observable behavior the rewrite must
// preserve, so the same expectations are checked before and after the rewrite.
// RUN: loom-dfg-sim %s --graph unpack_of_pack --output %t.src-roundtrip.json
// RUN: FileCheck %s --check-prefix=SCALAR-ITEMS < %t.src-roundtrip.json
// RUN: loom-dfg-sim %s --graph pack_of_unpack_float_bits --output %t.src-bits.json
// RUN: FileCheck %s --check-prefix=FLOAT-BITS < %t.src-bits.json

// RUN: loom-raise-opt --dataflow-rewrite=kind=pack-unpack-round-trip-eliminate %s -o %t.opt.mlir
// RUN: FileCheck %s --check-prefix=OPT < %t.opt.mlir
// RUN: loom-dfg-sim %t.opt.mlir --graph unpack_of_pack --output %t.opt-roundtrip.json
// RUN: FileCheck %s --check-prefix=SCALAR-ITEMS < %t.opt-roundtrip.json
// RUN: loom-dfg-sim %t.opt.mlir --graph pack_of_unpack_float_bits --output %t.opt-bits.json
// RUN: FileCheck %s --check-prefix=FLOAT-BITS < %t.opt-bits.json

// SCALAR-ITEMS: "final_stream_outputs": [
// SCALAR-ITEMS-NEXT: [
// SCALAR-ITEMS-NEXT: "i8:0",
// SCALAR-ITEMS-NEXT: "i8:1",
// SCALAR-ITEMS-NEXT: "i8:2",
// SCALAR-ITEMS-NEXT: "i8:3",
// SCALAR-ITEMS-NEXT: "i8:4"
// SCALAR-ITEMS-NEXT: ],
// SCALAR-ITEMS-NEXT: [
// SCALAR-ITEMS-NEXT: "i1:true",
// SCALAR-ITEMS-NEXT: "i1:true",
// SCALAR-ITEMS-NEXT: "i1:true",
// SCALAR-ITEMS-NEXT: "i1:true",
// SCALAR-ITEMS-NEXT: "i1:true",
// SCALAR-ITEMS-NEXT: "i1:false"
// SCALAR-ITEMS-NEXT: ]
// SCALAR-ITEMS-NEXT: ],
// SCALAR-ITEMS-DAG: "status": "pass"

// The exported payload keeps every signaling-NaN, negative-quiet-NaN, infinity,
// signed-zero and subnormal bit of the six f32 lanes.
// FLOAT-BITS-DAG: "i192:2189397960781083641023532207009296984468382285825"
// FLOAT-BITS-DAG: "status": "pass"

module {
  // OPT-LABEL: dataflow.graph private @unpack_of_pack
  // OPT: %[[VECTOR:[^,]*]], %[[MASK:[^,]*]], %[[GROUP:[^ ]*]] = dataflow.parallelize
  // OPT-NOT: dataflow.pack
  // OPT-NOT: dataflow.unpack
  // OPT: dataflow.serialize %[[VECTOR]], %[[MASK]], %[[GROUP]]
  dataflow.graph private @unpack_of_pack(%start: none) -> (i8, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %five = dataflow.constant %start {const_value = 5 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %item, %scalar_phase = dataflow.stream %zero, %five, %one
        step add while ult : i8
    %vector, %mask, %group_phase =
      dataflow.parallelize %item, %scalar_phase
        : (i8, i1) -> (vector<3xi8>, vector<3xi1>, i1)
    %packed = dataflow.pack %vector : vector<3xi8> -> i24
    %restored = dataflow.unpack %packed : i24 -> vector<3xi8>
    %scalar, %roundtrip_phase =
      dataflow.serialize %restored, %mask, %group_phase
        : (vector<3xi8>, vector<3xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %roundtrip_phase, %start : none
    %close:2 = dataflow.demux %roundtrip_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%scalar, %roundtrip_phase : i8, i1) memories()
        complete(%close#0 : none)
  }

  // A fixed multi-rank vector of floating payload bits round trips exactly.
  // OPT-LABEL: dataflow.graph private @pack_of_unpack_float_bits
  // OPT: %[[BITS:[^ ]*]] = dataflow.constant %{{[^ ]*}} {const_value = 2189397960781083641023532207009296984468382285825 : i192} : i192
  // OPT-NOT: dataflow.unpack
  // OPT-NOT: dataflow.pack
  // OPT: dataflow.sync %{{[^,]*}}, %[[BITS]] :
  dataflow.graph private @pack_of_unpack_float_bits(%start: none) -> i192
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %bits = dataflow.constant %start
        {const_value = 2189397960781083641023532207009296984468382285825 : i192}
        : i192
    %lanes = dataflow.unpack %bits : i192 -> vector<2x3xf32>
    %repacked = dataflow.pack %lanes : vector<2x3xf32> -> i192
    %retired:2 = dataflow.sync %start, %repacked : (none, i192) -> (none, i192)
    dataflow.graph.return values(%retired#1 : i192) streams() memories()
        complete(%retired#0 : none)
  }

  // The packed intermediate is also an exported stream, so deleting the round
  // trip would delete an observed value.
  // OPT-LABEL: dataflow.graph private @packed_side_use
  // OPT: %[[PACKED:[^ ]*]] = dataflow.pack
  // OPT: dataflow.unpack %[[PACKED]]
  dataflow.graph private @packed_side_use(%start: none) -> (i24, i8, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 3, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %five = dataflow.constant %start {const_value = 5 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %item, %scalar_phase = dataflow.stream %zero, %five, %one
        step add while ult : i8
    %vector, %mask, %group_phase =
      dataflow.parallelize %item, %scalar_phase
        : (i8, i1) -> (vector<3xi8>, vector<3xi1>, i1)
    %packed = dataflow.pack %vector : vector<3xi8> -> i24
    %restored = dataflow.unpack %packed : i24 -> vector<3xi8>
    %scalar, %roundtrip_phase =
      dataflow.serialize %restored, %mask, %group_phase
        : (vector<3xi8>, vector<3xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %roundtrip_phase, %start : none
    %close:2 = dataflow.demux %roundtrip_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%packed, %scalar, %roundtrip_phase : i24, i8, i1) memories()
        complete(%close#0 : none)
  }

  // The unpacked lanes are also consumed by an elementwise actor, so the
  // opposite direction is blocked by the same side-use condition.
  // OPT-LABEL: dataflow.graph private @lane_side_use
  // OPT: %[[LANES:[^ ]*]] = dataflow.unpack
  // OPT: dataflow.pack %[[LANES]]
  // OPT: arith.addi %[[LANES]], %[[LANES]]
  dataflow.graph private @lane_side_use(%start: none) -> (i96, i96)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %bits = dataflow.constant %start {const_value = 1 : i96} : i96
    %lanes = dataflow.unpack %bits : i96 -> vector<3xi32>
    %repacked = dataflow.pack %lanes : vector<3xi32> -> i96
    %doubled = arith.addi %lanes, %lanes : vector<3xi32>
    %packed_doubled = dataflow.pack %doubled : vector<3xi32> -> i96
    %retired:3 = dataflow.sync %start, %repacked, %packed_doubled
        : (none, i96, i96) -> (none, i96, i96)
    dataflow.graph.return values(%retired#1, %retired#2 : i96, i96)
        streams() memories() complete(%retired#0 : none)
  }

  // The inner pack would collapse an unproven exceptional state into an
  // ordinary packed value. Activity definedness is the legality owner, so the
  // developer bulk driver must preserve both adapters just like the anchored
  // decision enumerator does.
  // OPT-LABEL: dataflow.graph private @unproven_activity
  // OPT: %[[UNPROVEN:[^ ]*]] = arith.divui
  // OPT: %[[PACKED:[^ ]*]] = dataflow.pack %[[UNPROVEN]]
  // OPT: dataflow.unpack %[[PACKED]]
  dataflow.graph private @unproven_activity(
      %start: none, %lhs: vector<3xi8>, %rhs: vector<3xi8>)
      -> vector<3xi8>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %unproven = arith.divui %lhs, %rhs : vector<3xi8>
    %packed = dataflow.pack %unproven : vector<3xi8> -> i24
    %restored = dataflow.unpack %packed : i24 -> vector<3xi8>
    %retired:2 = dataflow.sync %start, %restored
        : (none, vector<3xi8>) -> (none, vector<3xi8>)
    dataflow.graph.return values(%retired#1 : vector<3xi8>)
        streams() memories() complete(%retired#0 : none)
  }
}

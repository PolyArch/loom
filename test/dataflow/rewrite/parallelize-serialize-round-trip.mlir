// Anchors for the ParallelizeSerializeRoundTripEliminate typed rewrite. Only
// `serialize(parallelize(data, phase))` is an identity, and only when the
// vector, mask and group phase have no other consumer. The reverse direction
// is not an identity because serialize drops inactive lanes and parallelize
// regroups the survivors across the original group boundaries.

// RUN: loom-dfg-sim %s --graph serialize_of_parallelize --output %t.src-tail.json
// RUN: FileCheck %s --check-prefix=PARTIAL-TAIL < %t.src-tail.json
// RUN: loom-dfg-sim %s --graph empty_activation --output %t.src-empty.json
// RUN: FileCheck %s --check-prefix=EMPTY < %t.src-empty.json
// RUN: loom-dfg-sim %s --graph parallelize_of_serialize --output %t.src-compact.json
// RUN: FileCheck %s --check-prefix=COMPACT < %t.src-compact.json

// RUN: loom-raise-opt --dataflow-rewrite=kind=parallelize-serialize-round-trip-eliminate %s -o %t.opt.mlir
// RUN: FileCheck %s --check-prefix=OPT < %t.opt.mlir
// RUN: loom-dfg-sim %t.opt.mlir --graph serialize_of_parallelize --output %t.opt-tail.json
// RUN: FileCheck %s --check-prefix=PARTIAL-TAIL < %t.opt-tail.json
// RUN: loom-dfg-sim %t.opt.mlir --graph empty_activation --output %t.opt-empty.json
// RUN: FileCheck %s --check-prefix=EMPTY < %t.opt-empty.json
// RUN: loom-dfg-sim %t.opt.mlir --graph parallelize_of_serialize --output %t.opt-compact.json
// RUN: FileCheck %s --check-prefix=COMPACT < %t.opt-compact.json

// Five scalar items through width-3 groups leave a partial tail group; the
// ordered scalar items and the closing false phase are unchanged.
// PARTIAL-TAIL: "final_stream_outputs": [
// PARTIAL-TAIL-NEXT: [
// PARTIAL-TAIL-NEXT: "i8:0",
// PARTIAL-TAIL-NEXT: "i8:1",
// PARTIAL-TAIL-NEXT: "i8:2",
// PARTIAL-TAIL-NEXT: "i8:3",
// PARTIAL-TAIL-NEXT: "i8:4"
// PARTIAL-TAIL-NEXT: ],
// PARTIAL-TAIL-NEXT: [
// PARTIAL-TAIL-NEXT: "i1:true",
// PARTIAL-TAIL-NEXT: "i1:true",
// PARTIAL-TAIL-NEXT: "i1:true",
// PARTIAL-TAIL-NEXT: "i1:true",
// PARTIAL-TAIL-NEXT: "i1:true",
// PARTIAL-TAIL-NEXT: "i1:false"
// PARTIAL-TAIL-NEXT: ]
// PARTIAL-TAIL-NEXT: ],
// PARTIAL-TAIL-DAG: "status": "pass"

// An empty activation publishes no data item and exactly one closing phase.
// EMPTY: "final_stream_outputs": [
// EMPTY-NEXT: [],
// EMPTY-NEXT: [
// EMPTY-NEXT: "i1:false"
// EMPTY-NEXT: ]
// EMPTY-NEXT: ],
// EMPTY-DAG: "status": "pass"

// Two width-2 groups with one active lane each compact into a single full
// group, so the regrouped payload is 0x0101 and only one group phase is true.
// Eliminating this direction would instead republish the two original groups.
// COMPACT: "final_stream_outputs": [
// COMPACT-NEXT: [
// COMPACT-NEXT: "i16:257"
// COMPACT-NEXT: ],
// COMPACT-NEXT: [
// COMPACT-NEXT: "i1:true",
// COMPACT-NEXT: "i1:false"
// COMPACT-NEXT: ]
// COMPACT-NEXT: ],
// COMPACT-DAG: "status": "pass"

module {
  // Both scalar results are replaced by the exact original scalar data and
  // scalar phase values.
  // OPT-LABEL: dataflow.graph private @serialize_of_parallelize
  // OPT: %[[ITEM:[^,]*]], %[[PHASE:[^ ]*]] = dataflow.stream
  // OPT-NOT: dataflow.parallelize
  // OPT-NOT: dataflow.serialize
  // OPT: dataflow.graph.return values() streams(%[[ITEM]], %[[PHASE]] : i8, i1)
  dataflow.graph private @serialize_of_parallelize(%start: none) -> (i8, i1)
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
    %scalar, %roundtrip_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<3xi8>, vector<3xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %roundtrip_phase, %start : none
    %close:2 = dataflow.demux %roundtrip_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%scalar, %roundtrip_phase : i8, i1) memories()
        complete(%close#0 : none)
  }

  // OPT-LABEL: dataflow.graph private @empty_activation
  // OPT: %[[ITEM:[^,]*]], %[[PHASE:[^ ]*]] = dataflow.stream
  // OPT-NOT: dataflow.parallelize
  // OPT-NOT: dataflow.serialize
  // OPT: dataflow.graph.return values() streams(%[[ITEM]], %[[PHASE]] : i8, i1)
  dataflow.graph private @empty_activation(%start: none) -> (i8, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %item, %scalar_phase = dataflow.stream %zero, %zero, %one
        step add while ult : i8
    %vector, %mask, %group_phase =
      dataflow.parallelize %item, %scalar_phase
        : (i8, i1) -> (vector<3xi8>, vector<3xi1>, i1)
    %scalar, %roundtrip_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<3xi8>, vector<3xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %roundtrip_phase, %start : none
    %close:2 = dataflow.demux %roundtrip_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%scalar, %roundtrip_phase : i8, i1) memories()
        complete(%close#0 : none)
  }

  // OPT-LABEL: dataflow.graph private @vector_side_use
  // OPT: %[[VECTOR:[^,]*]], %{{[^,]*}}, %{{[^ ]*}} = dataflow.parallelize
  // OPT: dataflow.pack %[[VECTOR]]
  // OPT: dataflow.serialize %[[VECTOR]]
  dataflow.graph private @vector_side_use(%start: none) -> (i24, i8, i1)
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
    %scalar, %roundtrip_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<3xi8>, vector<3xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %roundtrip_phase, %start : none
    %close:2 = dataflow.demux %roundtrip_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%packed, %scalar, %roundtrip_phase : i24, i8, i1) memories()
        complete(%close#0 : none)
  }

  // OPT-LABEL: dataflow.graph private @mask_side_use
  // OPT: %{{[^,]*}}, %[[MASK:[^,]*]], %{{[^ ]*}} = dataflow.parallelize
  // OPT: dataflow.pack %[[MASK]]
  // OPT: dataflow.serialize %{{[^,]*}}, %[[MASK]]
  dataflow.graph private @mask_side_use(%start: none) -> (i3, i8, i1)
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
    %packed_mask = dataflow.pack %mask : vector<3xi1> -> i3
    %scalar, %roundtrip_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<3xi8>, vector<3xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %roundtrip_phase, %start : none
    %close:2 = dataflow.demux %roundtrip_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%packed_mask, %scalar, %roundtrip_phase : i3, i8, i1) memories()
        complete(%close#0 : none)
  }

  // OPT-LABEL: dataflow.graph private @group_phase_side_use
  // OPT: %{{[^,]*}}, %{{[^,]*}}, %[[GROUP:[^ ]*]] = dataflow.parallelize
  // OPT: dataflow.serialize %{{[^,]*}}, %{{[^,]*}}, %[[GROUP]]
  // OPT: dataflow.graph.return values() streams(%[[GROUP]],
  dataflow.graph private @group_phase_side_use(%start: none) -> (i1, i8, i1)
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
    %scalar, %roundtrip_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<3xi8>, vector<3xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %roundtrip_phase, %start : none
    %close:2 = dataflow.demux %roundtrip_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%group_phase, %scalar, %roundtrip_phase : i1, i8, i1) memories()
        complete(%close#0 : none)
  }

  // OPT-LABEL: dataflow.graph private @parallelize_of_serialize
  // OPT: %[[SCALAR:[^,]*]], %[[SPHASE:[^ ]*]] = dataflow.serialize
  // OPT: dataflow.parallelize %[[SCALAR]], %[[SPHASE]]
  dataflow.graph private @parallelize_of_serialize(%start: none) -> (i16, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %two = dataflow.constant %start {const_value = 2 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %ordinal, %group_phase = dataflow.stream %zero, %two, %one
        step add while ult : i8
    %group_units = dataflow.invariant %group_phase, %start : none
    %group_events:2 = dataflow.demux %group_phase, %group_units
        : (i1, none) -> (none, none)
    %packed = dataflow.constant %group_events#1 {const_value = 513 : i16} : i16
    %packed_mask = dataflow.constant %group_events#1 {const_value = 1 : i2} : i2
    %vector = dataflow.unpack %packed : i16 -> vector<2xi8>
    %mask = dataflow.unpack %packed_mask : i2 -> vector<2xi1>
    %scalar, %scalar_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<2xi8>, vector<2xi1>, i1) -> (i8, i1)
    %regrouped, %regrouped_mask, %regrouped_phase =
      dataflow.parallelize %scalar, %scalar_phase
        : (i8, i1) -> (vector<2xi8>, vector<2xi1>, i1)
    %repacked = dataflow.pack %regrouped : vector<2xi8> -> i16
    %units = dataflow.invariant %regrouped_phase, %start : none
    %close:2 = dataflow.demux %regrouped_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%repacked, %regrouped_phase : i16, i1) memories()
        complete(%close#0 : none)
  }
}

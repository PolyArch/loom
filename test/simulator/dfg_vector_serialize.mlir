// RUN: loom-dfg-sim %s --graph partial_roundtrip --output %t.partial.json
// RUN: FileCheck %s --check-prefix=PARTIAL < %t.partial.json
// RUN: loom-dfg-sim %s --graph empty_roundtrip --output %t.empty.json
// RUN: FileCheck %s --check-prefix=EMPTY < %t.empty.json
// RUN: loom-dfg-sim %s --graph zero_mask_group --output %t.zero-mask.json
// RUN: FileCheck %s --check-prefix=ZERO-MASK < %t.zero-mask.json

// PARTIAL: "final_stream_outputs": [
// PARTIAL-NEXT: [
// PARTIAL-NEXT: "i24:131328",
// PARTIAL-NEXT: "i24:1027"
// PARTIAL-NEXT: ],
// PARTIAL-NEXT: [
// PARTIAL-NEXT: "i3:7",
// PARTIAL-NEXT: "i3:3"
// PARTIAL-NEXT: ],
// PARTIAL-NEXT: [
// PARTIAL-NEXT: "i1:true",
// PARTIAL-NEXT: "i1:true",
// PARTIAL-NEXT: "i1:false"
// PARTIAL-NEXT: ],
// PARTIAL-NEXT: [
// PARTIAL-NEXT: "i8:0",
// PARTIAL-NEXT: "i8:1",
// PARTIAL-NEXT: "i8:2",
// PARTIAL-NEXT: "i8:3",
// PARTIAL-NEXT: "i8:4"
// PARTIAL-NEXT: ],
// PARTIAL-NEXT: [
// PARTIAL-NEXT: "i1:true",
// PARTIAL-NEXT: "i1:true",
// PARTIAL-NEXT: "i1:true",
// PARTIAL-NEXT: "i1:true",
// PARTIAL-NEXT: "i1:true",
// PARTIAL-NEXT: "i1:false"
// PARTIAL-NEXT: ]
// PARTIAL-NEXT: ],
// PARTIAL-DAG: "graph": "partial_roundtrip"
// PARTIAL-DAG: "dataflow.parallelize": 6
// PARTIAL-DAG: "dataflow.pack": 4
// PARTIAL-DAG: "dataflow.serialize": 3
// PARTIAL-DAG: "status": "pass"

// EMPTY: "final_stream_outputs": [
// EMPTY-NEXT: [
// EMPTY-NEXT: "i1:false"
// EMPTY-NEXT: ],
// EMPTY-NEXT: [
// EMPTY-NEXT: "i1:false"
// EMPTY-NEXT: ]
// EMPTY-NEXT: ],
// EMPTY-DAG: "graph": "empty_roundtrip"
// EMPTY-DAG: "dataflow.parallelize": 1
// EMPTY-DAG: "dataflow.serialize": 1
// EMPTY-DAG: "status": "pass"

// ZERO-MASK: "final_stream_outputs": [
// ZERO-MASK-NEXT: [],
// ZERO-MASK-NEXT: [
// ZERO-MASK-NEXT: "i1:false"
// ZERO-MASK-NEXT: ]
// ZERO-MASK-NEXT: ],
// ZERO-MASK-DAG: "graph": "zero_mask_group"
// ZERO-MASK-DAG: "dataflow.serialize": 2
// ZERO-MASK-DAG: "status": "pass"

module {
  dataflow.graph private @partial_roundtrip(%start: none)
      -> (i24, i3, i1, i8, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 5, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %five = dataflow.constant %start {const_value = 5 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %item, %scalar_phase = dataflow.stream %zero, %five, %one
        step add while ult : i8
    %vector, %mask, %group_phase =
      dataflow.parallelize %item, %scalar_phase
        : (i8, i1) -> (vector<3xi8>, vector<3xi1>, i1)
    %packed = dataflow.pack %vector : vector<3xi8> -> i24
    %packed_mask = dataflow.pack %mask : vector<3xi1> -> i3
    %scalar, %roundtrip_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<3xi8>, vector<3xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %roundtrip_phase, %start : none
    %close:2 = dataflow.demux %roundtrip_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return
        values()
        streams(%packed, %packed_mask, %group_phase, %scalar,
                %roundtrip_phase : i24, i3, i1, i8, i1)
        memories()
        complete(%close#0 : none)
  }

  dataflow.graph private @empty_roundtrip(%start: none) -> (i1, i1)
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
        streams(%group_phase, %roundtrip_phase : i1, i1) memories()
        complete(%close#0 : none)
  }

  dataflow.graph private @zero_mask_group(%start: none) -> (i8, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %zero = dataflow.constant %start {const_value = 0 : i8} : i8
    %one = dataflow.constant %start {const_value = 1 : i8} : i8
    %ordinal, %group_phase = dataflow.stream %zero, %one, %one
        step add while ult : i8
    %packed = dataflow.constant %start {const_value = 197121 : i32} : i32
    %packed_mask = dataflow.constant %start {const_value = 0 : i4} : i4
    %vector = dataflow.unpack %packed : i32 -> vector<4xi8>
    %mask = dataflow.unpack %packed_mask : i4 -> vector<4xi1>
    %scalar, %scalar_phase =
      dataflow.serialize %vector, %mask, %group_phase
        : (vector<4xi8>, vector<4xi1>, i1) -> (i8, i1)
    %units = dataflow.invariant %scalar_phase, %start : none
    %close:2 = dataflow.demux %scalar_phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values()
        streams(%scalar, %scalar_phase : i8, i1) memories()
        complete(%close#0 : none)
  }
}

// RUN: loom-dfg-sim %s --graph stream_i65_add_cross_bit63 --output %t.add.json
// RUN: FileCheck %s --check-prefix=ADD < %t.add.json
// RUN: loom-dfg-sim %s --graph stream_i65_zero_trip --output %t.zero.json
// RUN: FileCheck %s --check-prefix=ZERO < %t.zero.json
// RUN: loom-dfg-sim %s --graph stream_i65_lshr_wide_amount --output %t.lshr.json
// RUN: FileCheck %s --check-prefix=LSHR < %t.lshr.json
// RUN: loom-dfg-sim %s --graph stream_i65_shift_amount_rejected --output %t.rejected.json
// RUN: FileCheck %s --check-prefix=REJECTED < %t.rejected.json
// RUN: loom-dfg-sim %s --graph stream_i1_single_iteration --output %t.i1.json
// RUN: FileCheck %s --check-prefix=NARROW < %t.i1.json

// ADD: "final_stream_outputs": [
// ADD-NEXT: [
// ADD-NEXT: "i65:18446744073709551614",
// ADD-NEXT: "i65:18446744073709551617"
// ADD-NEXT: ],
// ADD-NEXT: [
// ADD-NEXT: "i1:true",
// ADD-NEXT: "i1:true",
// ADD-NEXT: "i1:false"
// ADD-NEXT: ]
// ADD-NEXT: ],
// ADD-DAG: "status": "pass"

// ZERO: "final_stream_outputs": [
// ZERO-NEXT: [],
// ZERO-NEXT: [
// ZERO-NEXT: "i1:false"
// ZERO-NEXT: ]
// ZERO-NEXT: ],
// ZERO-DAG: "status": "pass"

// LSHR: "final_stream_outputs": [
// LSHR-NEXT: [
// LSHR-NEXT: "i65:18446744073709551616",
// LSHR-NEXT: "i65:1"
// LSHR-NEXT: ],
// LSHR-NEXT: [
// LSHR-NEXT: "i1:true",
// LSHR-NEXT: "i1:true",
// LSHR-NEXT: "i1:false"
// LSHR-NEXT: ]
// LSHR-NEXT: ],
// LSHR-DAG: "status": "pass"

// REJECTED-DAG: "status": "blocked"
// REJECTED-DAG: dataflow.stream shift amount must be in [0, 65), got 18446744073709551616

// NARROW: "final_stream_outputs": [
// NARROW-NEXT: [
// NARROW-NEXT: "i1:true"
// NARROW-NEXT: ],
// NARROW-NEXT: [
// NARROW-NEXT: "i1:true",
// NARROW-NEXT: "i1:false"
// NARROW-NEXT: ]
// NARROW-NEXT: ],
// NARROW-DAG: "status": "pass"

module {
  dataflow.graph private @stream_i65_add_cross_bit63(%start: none) -> (i65, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %init = dataflow.constant %start {const_value = 18446744073709551614 : i65} : i65
    %limit = dataflow.constant %start {const_value = 18446744073709551620 : i65} : i65
    %step = dataflow.constant %start {const_value = 3 : i65} : i65
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while ult : i65
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv, %phase : i65, i1) memories()
        complete(%close#0 : none)
  }

  dataflow.graph private @stream_i65_zero_trip(%start: none) -> (i65, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %init = dataflow.constant %start {const_value = 18446744073709551616 : i65} : i65
    %limit = dataflow.constant %start {const_value = 5 : i65} : i65
    %step = dataflow.constant %start {const_value = 1 : i65} : i65
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while ult : i65
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv, %phase : i65, i1) memories()
        complete(%close#0 : none)
  }

  dataflow.graph private @stream_i65_lshr_wide_amount(%start: none)
      -> (i65, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %init = dataflow.constant %start {const_value = 18446744073709551616 : i65} : i65
    %limit = dataflow.constant %start {const_value = 0 : i65} : i65
    %step = dataflow.constant %start {const_value = 64 : i65} : i65
    %iv, %phase = dataflow.stream %init, %limit, %step
        step lshr while ugt : i65
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv, %phase : i65, i1) memories()
        complete(%close#0 : none)
  }

  dataflow.graph private @stream_i65_shift_amount_rejected(%start: none)
      -> (i65, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %init = dataflow.constant %start {const_value = 1 : i65} : i65
    %limit = dataflow.constant %start {const_value = 2 : i65} : i65
    %step = dataflow.constant %start {const_value = 18446744073709551616 : i65} : i65
    %iv, %phase = dataflow.stream %init, %limit, %step
        step shl while ult : i65
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv, %phase : i65, i1) memories()
        complete(%close#0 : none)
  }

  dataflow.graph private @stream_i1_single_iteration(%start: none) -> (i1, i1)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %init = dataflow.constant %start {const_value = true} : i1
    %limit = dataflow.constant %start {const_value = false} : i1
    %step = dataflow.constant %start {const_value = true} : i1
    %iv, %phase = dataflow.stream %init, %limit, %step
        step add while ugt : i1
    %units = dataflow.invariant %phase, %start : none
    %close:2 = dataflow.demux %phase, %units
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%iv, %phase : i1, i1) memories()
        complete(%close#0 : none)
  }
}

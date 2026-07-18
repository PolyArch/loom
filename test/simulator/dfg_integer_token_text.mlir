// RUN: loom-dfg-sim %s --graph integer_boundaries \
// RUN:   --arg 0=9223372036854775808 \
// RUN:   --arg 1=18446744073709551615 \
// RUN:   --arg 2=9223372036854775807 \
// RUN:   --arg 3=18446744073709551616 \
// RUN:   --arg 4=-1 \
// RUN:   --arg 5=39614081257132168796771975168 \
// RUN:   --arg 6=-1 --output %t.boundaries.json
// RUN: FileCheck %s --check-prefix=BOUNDARY < %t.boundaries.json
// RUN: loom-dfg-sim %s --graph integer_attributes \
// RUN:   --output %t.attributes.json
// RUN: FileCheck %s --check-prefix=ATTRIBUTE < %t.attributes.json
// RUN: not loom-dfg-sim %s --graph echo_i96 --arg 0=-+1 \
// RUN:   --output %t.malformed.json 2>&1 | FileCheck %s --check-prefix=MALFORMED
// RUN: not loom-dfg-sim %s --graph echo_i65 \
// RUN:   --arg 0=36893488147419103232 \
// RUN:   --output %t.positive-overflow.json 2>&1 \
// RUN:   | FileCheck %s --check-prefix=OUT-OF-RANGE
// RUN: not loom-dfg-sim %s --graph echo_i65 \
// RUN:   --arg 0=-18446744073709551617 \
// RUN:   --output %t.negative-overflow.json 2>&1 \
// RUN:   | FileCheck %s --check-prefix=OUT-OF-RANGE

// BOUNDARY: "final_outputs": [
// BOUNDARY-NEXT: "none",
// BOUNDARY-NEXT: "i64:-9223372036854775808",
// BOUNDARY-NEXT: "i64:-1",
// BOUNDARY-NEXT: "i64:9223372036854775807",
// BOUNDARY-NEXT: "i65:18446744073709551616",
// BOUNDARY-NEXT: "i65:36893488147419103231",
// BOUNDARY-NEXT: "i96:39614081257132168796771975168",
// BOUNDARY-NEXT: "i96:79228162514264337593543950335"
// BOUNDARY-NEXT: ],
// BOUNDARY-DAG: "graph": "integer_boundaries"
// BOUNDARY-DAG: "status": "pass"

// ATTRIBUTE: "final_outputs": [
// ATTRIBUTE-NEXT: "none",
// ATTRIBUTE-NEXT: "i64:-1",
// ATTRIBUTE-NEXT: "i65:36893488147419103231",
// ATTRIBUTE-NEXT: "i96:79228162514264337593543950335"
// ATTRIBUTE-NEXT: ],
// ATTRIBUTE-DAG: "graph": "integer_attributes"
// ATTRIBUTE-DAG: "status": "pass"

// MALFORMED: integer argument is not canonical base-10
// OUT-OF-RANGE: integer argument does not fit its declared bit width

module {
  dataflow.graph private @integer_boundaries(
      %start: none,
      %i64_high: i64,
      %i64_max: i64,
      %i64_signed_max: i64,
      %i65_high: i65,
      %i65_negative: i65,
      %i96_high: i96,
      %i96_negative: i96)
      -> (i64, i64, i64, i65, i65, i96, i96)
      attributes {input_segments = array<i32: 7, 0, 0>,
                  result_segments = array<i32: 7, 0, 0>} {
    %published:8 = dataflow.sync
        %start,
        %i64_high,
        %i64_max,
        %i64_signed_max,
        %i65_high,
        %i65_negative,
        %i96_high,
        %i96_negative
        : (none, i64, i64, i64, i65, i65, i96, i96)
        -> (none, i64, i64, i64, i65, i65, i96, i96)
    dataflow.graph.return
        %published#0,
        %published#1,
        %published#2,
        %published#3,
        %published#4,
        %published#5,
        %published#6,
        %published#7
        : none, i64, i64, i64, i65, i65, i96, i96
  }

  dataflow.graph private @echo_i65(%start: none, %value: i65) -> i65
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %published:2 = dataflow.sync %start, %value
        : (none, i65) -> (none, i65)
    dataflow.graph.return %published#0, %published#1 : none, i65
  }

  dataflow.graph private @integer_attributes(%start: none)
      -> (i64, i65, i96)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 3, 0, 0>} {
    %i64_negative =
        dataflow.constant %start {const_value = -1 : i64} : i64
    %i65_negative =
        dataflow.constant %start {const_value = -1 : i65} : i65
    %i96_negative =
        dataflow.constant %start {const_value = -1 : i96} : i96
    %published:4 = dataflow.sync
        %start, %i64_negative, %i65_negative, %i96_negative
        : (none, i64, i65, i96) -> (none, i64, i65, i96)
    dataflow.graph.return
        %published#0, %published#1, %published#2, %published#3
        : none, i64, i65, i96
  }

  dataflow.graph private @echo_i96(%start: none, %value: i96) -> i96
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %published:2 = dataflow.sync %start, %value
        : (none, i96) -> (none, i96)
    dataflow.graph.return %published#0, %published#1 : none, i96
  }
}

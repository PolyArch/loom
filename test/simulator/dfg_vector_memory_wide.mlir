// RUN: loom-dfg-sim %s --graph wide_vector_memory \
// RUN:   --arg 0=1 --arg 1=0xbbcc99aa7788556633441122 \
// RUN:   --memref 2=0,0,0,0,0,0,0,0 --output %t.wide.json
// RUN: FileCheck %s --check-prefix=WIDE < %t.wide.json

// WIDE: "event_count": 2
// WIDE: "arg2": [
// WIDE-NEXT: "i16:0",
// WIDE-NEXT: "i16:4386",
// WIDE-NEXT: "i16:13124",
// WIDE-NEXT: "i16:21862",
// WIDE-NEXT: "i16:30600",
// WIDE-NEXT: "i16:39338",
// WIDE-NEXT: "i16:48076",
// WIDE-NEXT: "i16:0"
// WIDE: "final_outputs": [
// WIDE-NEXT: "none",
// WIDE-NEXT: "vector<6xi16>:0xBBCC99AA7788556633441122"
// WIDE: "dataflow.load": 1
// WIDE-NEXT: "dataflow.store": 1
// WIDE: "status": "pass"

module {
  dataflow.graph private @wide_vector_memory(
      %start: none, %idx: index, %value: vector<6xi16>,
      %mem: memref<?xi16>) -> vector<6xi16>
      attributes {input_segments = array<i32: 2, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %stored = dataflow.store %mem[%idx] %value %start
        : memref<?xi16>, vector<6xi16>
    %loaded, %done = dataflow.load %mem[%idx] %stored
        : memref<?xi16>, vector<6xi16>
    dataflow.graph.return %done, %loaded : none, vector<6xi16>
  }
}

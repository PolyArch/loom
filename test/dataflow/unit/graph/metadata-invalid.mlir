// RUN: loom %s -verify-diagnostics

// The leading start endpoint is protocol state, not an application ABI slot.
// expected-error @+2 {{graph start argument is a protocol endpoint and cannot carry application interface attributes}}
dataflow.graph private @start_metadata(
    %start: none {test.protocol}, %value: i32) -> i32 {
  dataflow.graph.return %start, %value : none, i32
}

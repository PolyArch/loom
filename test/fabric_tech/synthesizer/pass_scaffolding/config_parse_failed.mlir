// RUN: not loom %s -loom-generalize-subgraphs-to-fu='config=%p/broken.yaml' 2>&1 | FileCheck %s

// A YAML config file that fails to parse aborts the pass with an
// `error` diagnostic. Every input func.func is annotated with
// `loom.synth_failed = "config_parse_failed"` for completeness.

// CHECK: error: {{.*}}config_parse_failed
// CHECK: loom.synth_failed = "config_parse_failed"

func.func @pat_addi(%a: i32, %b: i32) -> i32 {
  %r = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> i32 {
    %s = arith.addi %x, %y : i32
    dataflow.yield %s : i32
  }
  return %r : i32
}

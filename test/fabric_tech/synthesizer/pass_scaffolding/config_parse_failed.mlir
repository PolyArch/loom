// RUN: not loom %s -loom-synthesize-configured-functions='config=%p/broken.yaml' 2>&1 | FileCheck %s

// A YAML config file that fails to parse aborts the pass with an
// `error` diagnostic. Every input func.func is annotated with
// `loom.synth_failed = "config_parse_failed"` for completeness.

// CHECK: error: {{.*}}config_parse_failed
// CHECK: loom.synth_failed = "config_parse_failed"

func.func @pat_addi(%a: i32, %b: i32) -> i32 {
  %s = arith.addi %a, %b : i32
  return %s : i32
}

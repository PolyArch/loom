// RUN: loom %s -loom-generalize-subgraphs-to-fu 2>&1 | FileCheck %s

// A func.func that contains zero `dataflow.subgraph` is rejected during
// input validation. The function is annotated with
// `loom.synth_failed = "invalid_input"`, a warning is emitted, and the
// function is not enrolled in any synth group.

// CHECK: warning: {{.*}}func.func @no_subgraph: invalid_input
// CHECK: loom.synth_failed = "invalid_input"

func.func @no_subgraph() {
  return
}

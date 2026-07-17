// RUN: loom %s -loom-synthesize-configured-functions='config=%p/anchor.yaml dump-stats=true' 2>&1 | FileCheck %s

// CHECK: warning:
// CHECK-SAME: group "stream_step_mismatch": synthesis failed: topology_mismatch
// CHECK: note: anchor: dataflow.stream peers require one fixed step_kind
// CHECK: remark: {{.*}}synth-stat group=stream_step_mismatch strategy=anchor reason=topology_mismatch
// CHECK-SAME: covered=0/2 nodes=0/0/0
// CHECK: loom.synth_failed = "topology_mismatch"
// CHECK: loom.synth_failed = "topology_mismatch"

func.func @pat_stream_add(%init: i32, %limit: i32, %step: i32) -> i32
    attributes {loom.synth_group = "stream_step_mismatch"} {
  %iv, %phase = dataflow.stream %init, %limit, %step step add while slt : i32
  return %iv : i32
}

func.func @pat_stream_sdiv(%init: i32, %limit: i32, %step: i32) -> i32
    attributes {loom.synth_group = "stream_step_mismatch"} {
  %iv, %phase = dataflow.stream %init, %limit, %step step sdiv while slt : i32
  return %iv : i32
}

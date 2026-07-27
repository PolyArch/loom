// RUN: not loom %s 2>&1 | FileCheck %s

// A launch whose operand segmentation is malformed reaches the module-level
// extent analysis before its own verification runs. That analysis declines to
// read its segmented operand ranges instead of reporting them, so ODS operand
// segmentation stays the only authority: the run fails normally rather than
// crashing, and the canonical error appears exactly once.

// CHECK-COUNT-1: error: 'dataflow.thread.launch' op 'operandSegmentSizes' attribute cannot have negative elements
// CHECK-NOT: cannot have negative elements

dataflow.thread private @t_malformed_segments domain(#dataflow.thread_domain<dense>)() ctrl (%ctrl: none)
    iv (%i: index) {
  dataflow.thread.yield
}
func.func @launch_malformed_segments(%extent: index) {
  %token = "dataflow.thread.launch"(%extent) <{callee = @t_malformed_segments,
      operandSegmentSizes = array<i32: 0, -1, 1>}>
      : (index) -> !dataflow.thread_token
  return
}

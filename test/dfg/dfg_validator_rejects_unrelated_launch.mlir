// RUN: not %python %S/../app/dfg_validator.py --input %s --symbol target_kernel 2>&1 | FileCheck %s

// CHECK: has no launched dataflow graph for target_kernel

module {
  dataflow.thread private @t_other_kernel(%ctrl: none) ctrl (%thread_ctrl: none) iv (%iv: index) {
    %done = dataflow.graph.launch @g_t_other_kernel_0_0 deps(%thread_ctrl)
        values() stream_inputs() memories() stream_outputs() : (none) -> none
    dataflow.thread.yield
  }

  dataflow.graph private @g_t_other_kernel_0_0(%ctrl: none) -> () {
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph private @g_target_kernel_0(%ctrl: none) -> () {
    // dataflow.graph.launch @g_target_kernel_0 deps(%ctrl) values()
    dataflow.graph.return %ctrl : none
  }
}

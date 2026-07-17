// RUN: not %python %S/../app/dfg_validator.py --input %s --symbol detached_kernel 2>&1 | FileCheck %s

// CHECK: has no dataflow.graph.launch

module {
  dataflow.graph.func private @g_detached_kernel_0(%ctrl: none) -> none {
    dataflow.graph.return %ctrl : none
  }
}

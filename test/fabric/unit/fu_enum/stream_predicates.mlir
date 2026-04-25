// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with one fabric.op[@dataflow.stream] whose hardware supports two
// step_op values and three cont_cond values: 2 x 3 = 6 supported subgraphs.

// CHECK-LABEL: @fu_stream
func.func @fu_stream(%lb: !fabric.bits<32>, %ub: !fabric.bits<32>,
                     %step: !fabric.bits<32>) {
  %i, %r = fabric.fu(%l = %lb : !fabric.bits<32>,
                     %u = %ub : !fabric.bits<32>,
                     %s = %step : !fabric.bits<32>)
                    -> (!fabric.bits<32>, !fabric.bits<1>) {
    %x, %y = fabric.op [@dataflow.stream] (%l, %u, %s)
             {hw_params = [{step_op = ["+=", "*="],
                             cont_cond = ["<", "<=", ">"]}]}
             : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
               -> (!fabric.bits<32>, !fabric.bits<1>)
    fabric.yield %x, %y : !fabric.bits<32>, !fabric.bits<1>
  }

  // 2 x 3 = 6 enumerated subgraphs.
  // CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "<", step_op = "+="} : i32
  // CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "<=", step_op = "+="} : i32
  // CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = ">", step_op = "+="} : i32
  // CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "<", step_op = "*="} : i32
  // CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "<=", step_op = "*="} : i32
  // CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = ">", step_op = "*="} : i32

  return
}

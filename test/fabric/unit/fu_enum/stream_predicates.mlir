// RUN: loom %s -loom-enumerate-fu-subgraphs | FileCheck %s

// FU with one fabric.op[@dataflow.stream] whose hardware supports two
// step_op values and three cont_cond values: 2 x 3 = 6 supported subgraphs.
// To satisfy the spatial_pe uniform-W rule we expose the FU at bits<1>
// throughout (the stream's TypeParam(0) data ports accept any width).

// CHECK-LABEL: fabric.module @fu_stream
fabric.module @fu_stream {
  %lb = builtin.unrealized_conversion_cast to !fabric.bits<1>
  %ub = builtin.unrealized_conversion_cast to !fabric.bits<1>
  %step = builtin.unrealized_conversion_cast to !fabric.bits<1>
  fabric.spatial_pe(%plb = %lb : !fabric.bits<1>,
                    %pub = %ub : !fabric.bits<1>,
                    %pstep = %step : !fabric.bits<1>)
                   -> (!fabric.bits<1>, !fabric.bits<1>) {
    fabric.fu(%l = %plb : !fabric.bits<1>,
              %u = %pub : !fabric.bits<1>,
              %s = %pstep : !fabric.bits<1>)
             -> (!fabric.bits<1>, !fabric.bits<1>) {
      %x, %y = fabric.op [@dataflow.stream] (%l, %u, %s)
               {hw_params = [{step_op = ["+=", "*="],
                               cont_cond = ["<", "<=", ">"]}]}
               : (!fabric.bits<1>, !fabric.bits<1>, !fabric.bits<1>)
                 -> (!fabric.bits<1>, !fabric.bits<1>)
      fabric.yield %x, %y : !fabric.bits<1>, !fabric.bits<1>
    }
  }
  fabric.yield
}

// 2 x 3 = 6 enumerated subgraphs.
// CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "<", step_op = "+="} : i1
// CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "<=", step_op = "+="} : i1
// CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = ">", step_op = "+="} : i1
// CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "<", step_op = "*="} : i1
// CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = "<=", step_op = "*="} : i1
// CHECK-DAG: dataflow.stream %{{.*}}, %{{.*}}, %{{.*}} {cont_cond = ">", step_op = "*="} : i1

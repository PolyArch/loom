// RUN: loom %s | loom | FileCheck %s

// CHECK-LABEL: fabric.module @integer_add_sub_capability
fabric.module @integer_add_sub_capability(
    %a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      // CHECK: fabric.op [@arith.addi, @arith.subi]
      // CHECK-SAME: hw_params = {integer_widths = [32 : i32]}
      // CHECK-SAME: implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>
      %value = fabric.op [@arith.addi, @arith.subi] (%fa, %fb) {
        hw_params = {integer_widths = [32 : i32]},
        implementation_family =
            #fabric.implementation_family<ScalarIntegerAddSub>
      } : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK-LABEL: fabric.module @integer_compare_capability
fabric.module @integer_compare_capability(
    %a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                       %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> () {
      // CHECK: fabric.op [@arith.cmpi]
      // CHECK-SAME: hw_params = {integer_widths = [32 : i32], predicates = ["eq", "slt"]}
      %predicate = fabric.op [@arith.cmpi] (%fa, %fb) {
        hw_params = {
          integer_widths = [32 : i32], predicates = ["eq", "slt"]
        },
        implementation_family =
            #fabric.implementation_family<ScalarIntegerCompareMinMax>
      } : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      fabric.yield
    }
  }
  fabric.yield
}

// CHECK-LABEL: fabric.module @stream_recurrence_capability
fabric.module @stream_recurrence_capability(
    %init : !fabric.bits<32>, %limit : !fabric.bits<32>,
    %step : !fabric.bits<32>) {
  fabric.pe [spatial] (%pinit = %init : !fabric.bits<32>,
                       %plimit = %limit : !fabric.bits<32>,
                       %pstep = %step : !fabric.bits<32>)
      -> !fabric.bits<32> {
    fabric.fu(%finit = %pinit : !fabric.bits<32>,
              %flimit = %plimit : !fabric.bits<32>,
              %fstep = %pstep : !fabric.bits<32>) -> () {
      // CHECK: fabric.op [@dataflow.stream]
      // CHECK-SAME: hw_params = {integer_widths = [32 : i32], predicates = ["slt"], step_kind = "add"}
      %iv, %phase = fabric.op [@dataflow.stream]
          (%finit, %flimit, %fstep) {
        hw_params = {
          integer_widths = [32 : i32], predicates = ["slt"],
          step_kind = "add"
        },
        implementation_family = #fabric.implementation_family<LoopStream>
      } : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
          -> (!fabric.bits<32>, !fabric.bits<1>)
      fabric.yield
    }
  }
  fabric.yield
}

// CHECK-LABEL: fabric.module @gate_token_plane_capability
fabric.module @gate_token_plane_capability(%value : !fabric.bits<32>) {
  fabric.pe [spatial] (%pvalue = %value : !fabric.bits<32>)
      -> !fabric.bits<32> {
    fabric.fu(%fvalue = %pvalue : !fabric.bits<32>) -> () {
      %condition = fabric.op [@arith.cmpi] (%fvalue, %fvalue) {
        hw_params = {integer_widths = [32 : i32], predicates = ["eq"]},
        implementation_family =
            #fabric.implementation_family<ScalarIntegerCompareMinMax>
      } : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
      // CHECK: fabric.op [@dataflow.gate]
      // CHECK-SAME: hw_params = {}
      %accepted, %forwarded = fabric.op [@dataflow.gate]
          (%condition, %fvalue) {
        hw_params = {},
        implementation_family = #fabric.implementation_family<LoopGate>
      } : (!fabric.bits<1>, !fabric.bits<32>)
          -> (!fabric.bits<1>, !fabric.bits<32>)
      fabric.yield
    }
  }
  fabric.yield
}

// RUN: loom %s | FileCheck %s

// Focused shared-reduction anchors under the normative implementation-family
// registry: the loop-control resource set (stream, carry, invariant, gate)
// and the scalar add/subtract datapath the recurrence drives. Every concrete
// fabric.op binds its explicit implementation family and carries exactly the
// typed hw_params record the family descriptor selects.

// CHECK: fabric.module @shared_reduction_adg
// CHECK-DAG: fabric.op [@dataflow.stream]{{.*}}implementation_family = #fabric.implementation_family<LoopStream>
// CHECK-DAG: fabric.op [@dataflow.carry]{{.*}}implementation_family = #fabric.implementation_family<LoopCarry>
// CHECK-DAG: fabric.op [@dataflow.invariant]{{.*}}implementation_family = #fabric.implementation_family<LoopInvariant>
// CHECK-DAG: fabric.op [@dataflow.gate]{{.*}}implementation_family = #fabric.implementation_family<LoopGate>
// CHECK-DAG: fabric.op [@arith.addi, @arith.subi]{{.*}}implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>
// CHECK-DAG: fabric.mem [spatial]
// CHECK-DAG: fabric.switch [spatial]

fabric.module @shared_reduction_adg(%mgr : memref<?x!fabric.bits<32>>,
                                    %init : !fabric.bits<32>,
                                    %limit : !fabric.bits<32>,
                                    %step : !fabric.bits<32>,
                                    %value : !fabric.bits<32>,
                                    %scale : !fabric.bits<32>,
                                    %addr : !fabric.bits<32>,
                                    %ctrl : !fabric.bits<0>) {
  %init_s0, %init_s1 = fabric.switch [spatial] %init
      [{connectivity_table = ["1", "1"]}]
      : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
  %value_s0, %value_s1 = fabric.switch [spatial] %value
      [{connectivity_table = ["1", "1"]}]
      : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
  %idx, %rwc = fabric.pe [spatial] (%pa = %init_s0 : !fabric.bits<32>,
                                    %pb = %limit : !fabric.bits<32>,
                                    %pc = %step : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>,
              %fc = %pc : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %iv, %phase = fabric.op [@dataflow.stream] (%fa, %fb, %fc)
                    {implementation_family =
                       #fabric.implementation_family<LoopStream>,
                     hw_params = {integer_widths = [8 : i32, 16 : i32, 32 : i32, 64 : i32],
                                  predicates = ["slt", "sgt"],
                                  step_kind = "add"}}
                    : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                      -> (!fabric.bits<32>, !fabric.bits<1>)
      fabric.yield %iv : !fabric.bits<32>, %phase : !fabric.bits<1> to !fabric.bits<32>
    }
  }
  %rwc_s0, %rwc_s1, %rwc_s2 = fabric.switch [spatial] %rwc
      [{connectivity_table = ["1", "1", "1"]}]
      : (!fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %carried = fabric.pe [spatial] (%pd = %rwc_s0 : !fabric.bits<32>,
                                  %pe = %init_s1 : !fabric.bits<32>,
                                  %pf = %value_s0 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%cond = %pd : !fabric.bits<32> to !fabric.bits<1>,
              %init_v = %pe : !fabric.bits<32>,
              %next_v = %pf : !fabric.bits<32>) -> !fabric.bits<32> {
      %acc = fabric.op [@dataflow.carry] (%cond, %init_v, %next_v)
             {implementation_family =
                #fabric.implementation_family<LoopCarry>,
              hw_params = {}}
             : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>)
               -> !fabric.bits<32>
      fabric.yield %acc : !fabric.bits<32>
    }
  }
  %sum = fabric.pe [spatial] (%pg = %carried : !fabric.bits<32>,
                              %ph = %value_s1 : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%lhs = %pg : !fabric.bits<32>,
              %rhs = %ph : !fabric.bits<32>) -> !fabric.bits<32> {
      %total = fabric.op [@arith.addi, @arith.subi] (%lhs, %rhs)
               {implementation_family =
                  #fabric.implementation_family<ScalarIntegerAddSub>,
                hw_params = {integer_widths = [8 : i32, 16 : i32, 32 : i32, 64 : i32]}}
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %total : !fabric.bits<32>
    }
  }
  %stable = fabric.pe [spatial] (%pi = %rwc_s1 : !fabric.bits<32>,
                                 %pj = %scale : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%gate_c = %pi : !fabric.bits<32> to !fabric.bits<1>,
              %scale_v = %pj : !fabric.bits<32>) -> !fabric.bits<32> {
      %held = fabric.op [@dataflow.invariant] (%gate_c, %scale_v)
              {implementation_family =
                 #fabric.implementation_family<LoopInvariant>,
               hw_params = {}}
              : (!fabric.bits<1>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %held : !fabric.bits<32>
    }
  }
  %after_cond, %after_value = fabric.pe [spatial] (%pk = %rwc_s2 : !fabric.bits<32>,
                                                   %pl = %sum : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%gcond = %pk : !fabric.bits<32> to !fabric.bits<1>,
              %gvalue = %pl : !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>) {
      %out_c, %out_v = fabric.op [@dataflow.gate] (%gcond, %gvalue)
                       {implementation_family =
                          #fabric.implementation_family<LoopGate>,
                        hw_params = {}}
                       : (!fabric.bits<1>, !fabric.bits<32>)
                         -> (!fabric.bits<1>, !fabric.bits<32>)
      fabric.yield %out_c : !fabric.bits<1> to !fabric.bits<32>, %out_v : !fabric.bits<32>
    }
  }
  %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32, data_width = 32 : i32, dispatch_eligibility = {operation_port_requests = [[0 : i32]], subordinate_requests = []}}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

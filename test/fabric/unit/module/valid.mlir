// RUN: loom %s | loom | FileCheck %s

// Empty module (no inputs, no outputs): parser inserts an implicit
// fabric.yield terminator and the printer omits the implicit terminator
// on the round-trip.
// CHECK-LABEL: fabric.module @m_empty
// CHECK-SAME: ()
// CHECK-NEXT: }
fabric.module @m_empty() {
}

// Module with an explicit fabric.yield terminator. The implicit-terminator
// printer still elides the yield, so the body round-trips as empty.
// CHECK-LABEL: fabric.module @m_explicit_yield
// CHECK-SAME: ()
// CHECK-NEXT: }
fabric.module @m_explicit_yield() {
  fabric.yield
}

// Module body holding the canonical fabric containers (pe, fifo).
// CHECK-LABEL: fabric.module @m_with_inner_ops
// CHECK-SAME: (%{{.*}}: !fabric.bits<32>, %{{.*}}: !fabric.bits<32>)
// CHECK: fabric.pe
// CHECK: fabric.fu
// CHECK: fabric.op
// CHECK: fabric.fifo
fabric.module @m_with_inner_ops(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                         %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  %f = fabric.fifo %r [max_depth = 4, bypassable = false] : !fabric.bits<32>
  fabric.yield
}

// Two distinct modules in one input file: each carries its own sym_name and
// each round-trips independently.
// CHECK-LABEL: fabric.module @m_first
// CHECK-LABEL: fabric.module @m_second
fabric.module @m_first() {
}
fabric.module @m_second() {
}

// Module with declared output types. The yield value types match the
// declared output types exactly (no width relaxation needed).
// CHECK-LABEL: fabric.module @m_with_outputs
// CHECK-SAME: -> (!fabric.bits<32>, !fabric.bits<32>)
fabric.module @m_with_outputs(%a : !fabric.bits<32>, %b : !fabric.bits<32>,
                              %c : !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                         %pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%x = %pa : !fabric.bits<32>, %y = %pb : !fabric.bits<32>)
                  -> !fabric.bits<32> {
      %k = fabric.op [@arith.addi] (%x, %y)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %k : !fabric.bits<32>
    }
  }
  fabric.yield %r, %c : !fabric.bits<32>, !fabric.bits<32>
}

// Explicit broadcast consumes each module input once through a switch and
// exposes two distinct point-to-point output transports.
// CHECK-LABEL: fabric.module @m_explicit_switch_broadcast
// CHECK: %{{.*}}:2 = fabric.switch [spatial]
// CHECK-SAME: route_table = ["10", "10"]
// CHECK: fabric.yield %{{.*}}#0, %{{.*}}#1 : !fabric.bits<32>, !fabric.bits<32>
fabric.module @m_explicit_switch_broadcast(%a : !fabric.bits<32>,
                                            %b : !fabric.bits<32>)
    -> (!fabric.bits<32>, !fabric.bits<32>) {
  %out:2 = fabric.switch [spatial] %a, %b
           [{connectivity_table = ["11", "11"]}]
           {route_table = ["10", "10"], switch_enable = true}
           : (!fabric.bits<32>, !fabric.bits<32>)
          -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield %out#0, %out#1 : !fabric.bits<32>, !fabric.bits<32>
}

// A fabric.mem subordinate capability may satisfy a module memory export.
// CHECK-LABEL: fabric.module @m_memref_sub_export
// CHECK: %[[MEM:[0-9]+]]:3 = fabric.mem [spatial]
// CHECK: fabric.yield %[[MEM]]#0 : memref<?x!fabric.bits<16>>
fabric.module @m_memref_sub_export(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>)
    -> (memref<?x!fabric.bits<16>>) {
  %sub, %data, %done = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = [[0 : i32]]
        }}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (memref<?x!fabric.bits<16>>, !fabric.bits<32>, !fabric.bits<0>)
  fabric.yield %sub : memref<?x!fabric.bits<16>>
}

// An instantiate result preserves the subordinate role of the target
// module's memory result.
// CHECK-LABEL: fabric.module @m_instantiate_memref_sub_export
// CHECK: %[[SUB:[0-9]+]] = fabric.instantiate @m_memref_sub_export
// CHECK: fabric.yield %[[SUB]] : memref<?x!fabric.bits<16>>
fabric.module @m_instantiate_memref_sub_export(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>)
    -> (memref<?x!fabric.bits<16>>) {
  %sub = fabric.instantiate @m_memref_sub_export(
      %mgr : memref<?x!fabric.bits<32>>,
      %addr : !fabric.bits<32>, %ctrl : !fabric.bits<0>)
      -> (memref<?x!fabric.bits<16>>)
      {domain_slot_bindings = array<i64: 0, 0, 0, 1, 0, 0>}
  fabric.yield %sub : memref<?x!fabric.bits<16>>
}

// A subordinate provider result may connect to a manager/requester operand
// and remain available for another provider-side use such as module export.
// CHECK-LABEL: fabric.module @m_memref_sub_to_mgr
// CHECK: %[[PROVIDER:[0-9]+]]:3 = fabric.mem [spatial]
// CHECK: fabric.mem [spatial] mgr(%[[PROVIDER]]#0)
// CHECK: fabric.yield %[[PROVIDER]]#0 : memref<?x!fabric.bits<32>>
fabric.module @m_memref_sub_to_mgr(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr0 : !fabric.bits<32>, %ctrl0 : !fabric.bits<0>,
    %addr1 : !fabric.bits<32>, %ctrl1 : !fabric.bits<0>)
    -> (memref<?x!fabric.bits<32>>) {
  %sub, %data0, %done0 =
      fabric.mem [spatial] mgr(%mgr) load(%addr0, %ctrl0)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = [[0 : i32]]
        }}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
  %data1, %done1 = fabric.mem [spatial] mgr(%sub) load(%addr1, %ctrl1)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield %sub : memref<?x!fabric.bits<32>>
}

// An instantiated provider result may likewise connect to a manager operand.
// CHECK-LABEL: fabric.module @m_instantiate_memref_sub_to_mgr
// CHECK: %[[PROVIDER:[0-9]+]] = fabric.instantiate @m_memref_sub_export
// CHECK: fabric.mem [spatial] mgr(%[[PROVIDER]])
fabric.module @m_instantiate_memref_sub_to_mgr(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr0 : !fabric.bits<32>, %ctrl0 : !fabric.bits<0>,
    %addr1 : !fabric.bits<32>, %ctrl1 : !fabric.bits<0>) {
  %sub = fabric.instantiate @m_memref_sub_export(
      %mgr : memref<?x!fabric.bits<32>>,
      %addr0 : !fabric.bits<32>, %ctrl0 : !fabric.bits<0>)
      -> (memref<?x!fabric.bits<16>>)
      {domain_slot_bindings = array<i64: 0, 0, 0, 1, 0, 0>}
  %data, %done = fabric.mem [spatial] mgr(%sub) load(%addr1, %ctrl1)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 16 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<16>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<16>, !fabric.bits<0>)
  fabric.yield
}

// Manager capabilities are not token transports and may feed multiple
// manager-side consumers.
// CHECK-LABEL: fabric.module @m_memref_manager_multiuse
// CHECK: fabric.mem [spatial] mgr(%[[MGR:[a-zA-Z0-9_]+]])
// CHECK: fabric.mem [spatial] mgr(%[[MGR]])
fabric.module @m_memref_manager_multiuse(
    %mgr : memref<?x!fabric.bits<32>>,
    %addr0 : !fabric.bits<32>, %ctrl0 : !fabric.bits<0>,
    %addr1 : !fabric.bits<32>, %ctrl1 : !fabric.bits<0>) {
  %data0, %done0 = fabric.mem [spatial] mgr(%mgr) load(%addr0, %ctrl0)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  %data1, %done1 = fabric.mem [spatial] mgr(%mgr) load(%addr1, %ctrl1)
      [{load_group_size = 1 : i32, store_group_size = 0 : i32,
        data_width = 32 : i32,
        dispatch_eligibility = {
          operation_port_requests = [[0 : i32]],
          subordinate_requests = []
        }}]
      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
      -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}

// Width relaxation at module-input -> pe operand: the source is
// !fabric.bits<32> and the PE block-arg / inner type is !fabric.bits<16>.
// CHECK-LABEL: fabric.module @m_pe_input_width_relax
// CHECK: fabric.pe [spatial] (%{{.*}} = %{{.*}} : !fabric.bits<32> to !fabric.bits<16>) -> !fabric.bits<16>
fabric.module @m_pe_input_width_relax(%a : !fabric.bits<32>) {
  %r = fabric.pe [spatial] (%pa = %a : !fabric.bits<32> to !fabric.bits<16>)
                        -> !fabric.bits<16> {
    fabric.fu(%fa = %pa : !fabric.bits<16>) -> !fabric.bits<16> {
      %v = fabric.op [@arith.addi] (%fa, %fa)
           {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [1 : i32]}} : (!fabric.bits<16>, !fabric.bits<16>) -> !fabric.bits<16>
      fabric.yield %v : !fabric.bits<16>
    }
  }
  fabric.yield
}

// Width relaxation at fifo operand: input is !fabric.bits<32>, FIFO inner
// width is !fabric.bits<16>. Round-trip preserves the `to` clause.
// CHECK-LABEL: fabric.module @m_fifo_input_width_relax
// CHECK: fabric.fifo %{{.*}} [max_depth = 4, bypassable = false] : !fabric.bits<32> to !fabric.bits<16>
fabric.module @m_fifo_input_width_relax(%a : !fabric.bits<32>) {
  %0 = fabric.fifo %a [max_depth = 4, bypassable = false]
       : !fabric.bits<32> to !fabric.bits<16>
  fabric.yield
}

// Width relaxation at module yield: source !fabric.bits<32> is yielded for a
// declared !fabric.bits<16> module result, low-bit alignment.
// CHECK-LABEL: fabric.module @m_yield_width_relax
// CHECK-SAME: -> !fabric.bits<16>
// CHECK: fabric.yield %{{.*}} : !fabric.bits<32> to !fabric.bits<16>
fabric.module @m_yield_width_relax(%a : !fabric.bits<32>)
    -> (!fabric.bits<16>) {
  fabric.yield %a : !fabric.bits<32> to !fabric.bits<16>
}

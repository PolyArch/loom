// RUN: loom %s -loom-map-subgraph-to-fus | FileCheck %s

// FU offers a 3-port variadic dataflow.sync. The matcher runs VF2
// isomorphism between user patterns and the enumerator's per-bitmask
// templates; a 2-input dataflow.sync user pattern matches the
// bitmask=110 template (after dedup that is the canonical N=2 sync
// template; the matcher doesn't care which bitmask was canonical, only
// that the structural shape matches).

fabric.module @hw_sync3(%a : !fabric.bits<32>, %b : !fabric.bits<32>, %c : !fabric.bits<32>) {
  fabric.pe [spatial] (%pa = %a : !fabric.bits<32>,
                    %pb = %b : !fabric.bits<32>,
                    %pc = %c : !fabric.bits<32>)
                   -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
    fabric.fu(%x = %pa : !fabric.bits<32>,
              %y = %pb : !fabric.bits<32>,
              %z = %pc : !fabric.bits<32>)
             -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) {
      %u, %v, %w = fabric.op [@dataflow.sync] (%x, %y, %z)
                   : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
                     -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
      fabric.yield %u, %v, %w : !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>
    }
  }
  fabric.yield
}

// CHECK-LABEL: @pat_sync2
func.func @pat_sync2(%a: i32, %b: i32) -> (i32, i32) {
  // CHECK: dataflow.subgraph
  // CHECK-SAME: loom.matched_fu = "@hw_sync3#0"
  %r:2 = dataflow.subgraph(%x = %a : i32, %y = %b : i32) -> (i32, i32)
         attributes {loom.is_pattern} {
    %s:2 = dataflow.sync %x, %y : (i32, i32) -> (i32, i32)
    dataflow.yield %s#0, %s#1 : i32, i32
  }
  return %r#0, %r#1 : i32, i32
}

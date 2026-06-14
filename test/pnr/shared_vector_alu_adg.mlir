// RUN: loom %s | FileCheck %s

// Shared vector ALU fabric for elementwise load/compute/store kernels.
// Connectivity is explicit SSA through Fabric switches; coordinates are not
// part of mapping legality.
// CHECK-LABEL: fabric.module @shared_vector_alu_adg
fabric.module @shared_vector_alu_adg(%mgr : memref<?x!fabric.bits<32>>,
                                     %idx0 : !fabric.bits<32>,
                                     %idx1 : !fabric.bits<32>,
                                     %store_idx : !fabric.bits<32>,
                                     %ctrl : !fabric.bits<0>,
                                     %i32a : !fabric.bits<32>,
                                     %i32b : !fabric.bits<32>) {
  // CHECK: fabric.mem [spatial]
  %data0, %done0, %data1, %done1, %store_done =
      fabric.mem [spatial] mgr(%mgr)
        load(%idx0, %ctrl, %idx1, %ctrl)
        store(%store_idx, %store_value, %ctrl)
        [{load_group_size = 2 : i32, store_group_size = 1 : i32}]
        : (memref<?x!fabric.bits<32>>,
           !fabric.bits<32>, !fabric.bits<0>,
           !fabric.bits<32>, !fabric.bits<0>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>,
            !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<0>)

  // CHECK: fabric.switch [spatial]
  %bin0, %bin1, %unary =
      fabric.switch [spatial] %data0, %data1, %i32a
        [{connectivity_table = ["111", "111", "111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)

  // CHECK: fabric.op [@arith.xori]
  %xored = fabric.pe [spatial] (%lhs = %bin0 : !fabric.bits<32>,
                                %rhs = %bin1 : !fabric.bits<32>)
      -> !fabric.bits<32> {
    fabric.fu(%a = %lhs : !fabric.bits<32>,
              %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
      %value = fabric.op [@arith.xori] (%a, %b)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }

  // CHECK: fabric.op [@llvm.intr.bswap]
  %swapped = fabric.pe [spatial] (%value = %unary : !fabric.bits<32>)
      -> !fabric.bits<32> {
    fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
      %result = fabric.op [@llvm.intr.bswap] (%input)
                : (!fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %result : !fabric.bits<32>
    }
  }

  // CHECK: fabric.op [@arith.mulf]
  %product = fabric.pe [spatial] (%lhs = %bin0 : !fabric.bits<32>,
                                  %rhs = %bin1 : !fabric.bits<32>)
      -> !fabric.bits<32> {
    fabric.fu(%a = %lhs : !fabric.bits<32>,
              %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
      %value = fabric.op [@arith.mulf] (%a, %b)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }

  // CHECK: fabric.op [@arith.muli]
  %int_product = fabric.pe [spatial] (%lhs = %bin0 : !fabric.bits<32>,
                                      %rhs = %i32b : !fabric.bits<32>)
      -> !fabric.bits<32> {
    fabric.fu(%a = %lhs : !fabric.bits<32>,
              %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
      %value = fabric.op [@arith.muli] (%a, %b)
               : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %value : !fabric.bits<32>
    }
  }

  // CHECK: fabric.switch [spatial]
  %store_value =
      fabric.switch [spatial] %xored, %swapped, %product, %int_product, %i32b
        [{connectivity_table = ["11111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>, !fabric.bits<32>)
        -> !fabric.bits<32>

  // CHECK: fabric.switch [spatial]
  %sync0, %sync1, %sync2 =
      fabric.switch [spatial] %done0, %done1, %store_done
        [{connectivity_table = ["111", "111", "111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)

  fabric.pe [spatial] (%pa = %sync0 : !fabric.bits<0>,
                       %pb = %sync1 : !fabric.bits<0>,
                       %pc = %sync2 : !fabric.bits<0>)
      -> !fabric.bits<0> {
    fabric.fu(%fa = %pa : !fabric.bits<0>,
              %fb = %pb : !fabric.bits<0>,
              %fc = %pc : !fabric.bits<0>)
        -> !fabric.bits<0> {
      // CHECK: fabric.op [@dataflow.sync]
      %sa, %sb, %sc = fabric.op [@dataflow.sync] (%fa, %fb, %fc)
                      {sw_configs = {bitmask = "111"}}
                      : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
                        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
      fabric.yield %sa : !fabric.bits<0>
    }
  }
  fabric.yield
}

// RUN: loom %s | FileCheck %s

// CHECK: fabric.module @shared_memory_reduction_adg
// CHECK-DAG: fabric.op [@dataflow.constant]
// CHECK-DAG: fabric.op [@arith.cmpi, @llvm.icmp]
// CHECK-DAG: fabric.op [@arith.addf, @arith.subf]
// CHECK-DAG: fabric.op [@arith.mulf]
// CHECK-DAG: fabric.op [@arith.divui, @arith.remui]
// CHECK-DAG: fabric.op [@llvm.intr.fmuladd]
// CHECK-DAG: fabric.op [@arith.cmpf]
// CHECK-DAG: fabric.op [@dataflow.sync]
// CHECK-DAG: fabric.op [@llvm.intr.umin]
// CHECK-DAG: fabric.op [@llvm.intr.smin]
// CHECK-DAG: fabric.op [@llvm.intr.smax]
// CHECK-DAG: fabric.op [@arith.select]
// CHECK-DAG: fabric.op [@dataflow.mux]
// CHECK-DAG: fabric.op [@arith.trunci]
// CHECK-DAG: fabric.op [@arith.index_cast]
// CHECK-DAG: fabric.op [@arith.index_castui]
// CHECK-DAG: fabric.op [@llvm.trunc]
// CHECK-DAG: fabric.op [@llvm.sext]
// CHECK-DAG: fabric.op [@llvm.zext]
// CHECK-DAG: fabric.mem
// CHECK-DAG: fabric.switch

fabric.module @shared_memory_reduction_adg(%mgr : memref<?x!fabric.bits<32>>,
                                    %i32a : !fabric.bits<32>,
                                    %i32b : !fabric.bits<32>,
                                    %i32c : !fabric.bits<32>,
                                    %i32d : !fabric.bits<32>,
                                    %ctrl : !fabric.bits<0>) {
  %sync0_done0, %sync0_done1, %sync0_done2, %sync0_done3, %sync0_done4, %sync0_done5 = fabric.pe [spatial] (%p0 = %sync0_in0 : !fabric.bits<0>,
                    %p1 = %sync0_in1 : !fabric.bits<0>,
                    %p2 = %sync0_in2 : !fabric.bits<0>,
                    %p3 = %sync0_in3 : !fabric.bits<0>,
                    %p4 = %sync0_in4 : !fabric.bits<0>,
                    %p5 = %sync0_in5 : !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) {
    fabric.fu(%f0 = %p0 : !fabric.bits<0>,
              %f1 = %p1 : !fabric.bits<0>,
              %f2 = %p2 : !fabric.bits<0>,
              %f3 = %p3 : !fabric.bits<0>,
              %f4 = %p4 : !fabric.bits<0>,
              %f5 = %p5 : !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) {
      %s0, %s1, %s2, %s3, %s4, %s5 = fabric.op [@dataflow.sync] (%f0, %f1, %f2, %f3, %f4, %f5) {sw_configs = {bitmask = "111111"}} : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
      fabric.yield %s0, %s1, %s2, %s3, %s4, %s5 : !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>
    }
  }
  %sync1_done0, %sync1_done1, %sync1_done2, %sync1_done3, %sync1_done4, %sync1_done5 = fabric.pe [spatial] (%p0 = %sync1_in0 : !fabric.bits<0>,
                    %p1 = %sync1_in1 : !fabric.bits<0>,
                    %p2 = %sync1_in2 : !fabric.bits<0>,
                    %p3 = %sync1_in3 : !fabric.bits<0>,
                    %p4 = %sync1_in4 : !fabric.bits<0>,
                    %p5 = %sync1_in5 : !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) {
    fabric.fu(%f0 = %p0 : !fabric.bits<0>,
              %f1 = %p1 : !fabric.bits<0>,
              %f2 = %p2 : !fabric.bits<0>,
              %f3 = %p3 : !fabric.bits<0>,
              %f4 = %p4 : !fabric.bits<0>,
              %f5 = %p5 : !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) {
      %s0, %s1, %s2, %s3, %s4, %s5 = fabric.op [@dataflow.sync] (%f0, %f1, %f2, %f3, %f4, %f5) {sw_configs = {bitmask = "111111"}} : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
      fabric.yield %s0, %s1, %s2, %s3, %s4, %s5 : !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>
    }
  }
  %sync2_done0, %sync2_done1, %sync2_done2, %sync2_done3, %sync2_done4, %sync2_done5 = fabric.pe [spatial] (%p0 = %sync2_in0 : !fabric.bits<0>,
                    %p1 = %sync2_in1 : !fabric.bits<0>,
                    %p2 = %sync2_in2 : !fabric.bits<0>,
                    %p3 = %sync2_in3 : !fabric.bits<0>,
                    %p4 = %sync2_in4 : !fabric.bits<0>,
                    %p5 = %sync2_in5 : !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) {
    fabric.fu(%f0 = %p0 : !fabric.bits<0>,
              %f1 = %p1 : !fabric.bits<0>,
              %f2 = %p2 : !fabric.bits<0>,
              %f3 = %p3 : !fabric.bits<0>,
              %f4 = %p4 : !fabric.bits<0>,
              %f5 = %p5 : !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) {
      %s0, %s1, %s2, %s3, %s4, %s5 = fabric.op [@dataflow.sync] (%f0, %f1, %f2, %f3, %f4, %f5) {sw_configs = {bitmask = "111111"}} : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
      fabric.yield %s0, %s1, %s2, %s3, %s4, %s5 : !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>
    }
  }
  %sync3_done0, %sync3_done1, %sync3_done2, %sync3_done3, %sync3_done4, %sync3_done5 = fabric.pe [spatial] (%p0 = %sync3_in0 : !fabric.bits<0>,
                    %p1 = %sync3_in1 : !fabric.bits<0>,
                    %p2 = %sync3_in2 : !fabric.bits<0>,
                    %p3 = %sync3_in3 : !fabric.bits<0>,
                    %p4 = %sync3_in4 : !fabric.bits<0>,
                    %p5 = %sync3_in5 : !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) {
    fabric.fu(%f0 = %p0 : !fabric.bits<0>,
              %f1 = %p1 : !fabric.bits<0>,
              %f2 = %p2 : !fabric.bits<0>,
              %f3 = %p3 : !fabric.bits<0>,
              %f4 = %p4 : !fabric.bits<0>,
              %f5 = %p5 : !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) {
      %s0, %s1, %s2, %s3, %s4, %s5 = fabric.op [@dataflow.sync] (%f0, %f1, %f2, %f3, %f4, %f5) {sw_configs = {bitmask = "111111"}} : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>) -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
      fabric.yield %s0, %s1, %s2, %s3, %s4, %s5 : !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>
    }
  }
  %const0 =
      fabric.pe [spatial] (%pa = %const0_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const1 =
      fabric.pe [spatial] (%pa = %const1_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const2 =
      fabric.pe [spatial] (%pa = %const2_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const3 =
      fabric.pe [spatial] (%pa = %const3_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const4 =
      fabric.pe [spatial] (%pa = %const4_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const5 =
      fabric.pe [spatial] (%pa = %const5_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const6 =
      fabric.pe [spatial] (%pa = %const6_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const7 =
      fabric.pe [spatial] (%pa = %const7_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const8 =
      fabric.pe [spatial] (%pa = %const8_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const9 =
      fabric.pe [spatial] (%pa = %const9_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const10 =
      fabric.pe [spatial] (%pa = %const10_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const11 =
      fabric.pe [spatial] (%pa = %const11_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const12 =
      fabric.pe [spatial] (%pa = %const12_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const13 =
      fabric.pe [spatial] (%pa = %const13_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const14 =
      fabric.pe [spatial] (%pa = %const14_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const15 =
      fabric.pe [spatial] (%pa = %const15_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const16 =
      fabric.pe [spatial] (%pa = %const16_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const17 =
      fabric.pe [spatial] (%pa = %const17_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const18 =
      fabric.pe [spatial] (%pa = %const18_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const19 =
      fabric.pe [spatial] (%pa = %const19_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const20 =
      fabric.pe [spatial] (%pa = %const20_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const21 =
      fabric.pe [spatial] (%pa = %const21_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const22 =
      fabric.pe [spatial] (%pa = %const22_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const23 =
      fabric.pe [spatial] (%pa = %const23_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const24 =
      fabric.pe [spatial] (%pa = %const24_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const25 =
      fabric.pe [spatial] (%pa = %const25_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const26 =
      fabric.pe [spatial] (%pa = %const26_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const27 =
      fabric.pe [spatial] (%pa = %const27_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const28 =
      fabric.pe [spatial] (%pa = %const28_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %const29 =
      fabric.pe [spatial] (%pa = %const29_ctrl : !fabric.bits<0> to !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%token = %pa : !fabric.bits<32> to !fabric.bits<0>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000", "0xffffffff"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %wide_const0 =
      fabric.pe [spatial] (%pa = %wide_const0_ctrl : !fabric.bits<0> to !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%token = %pa : !fabric.bits<64> to !fabric.bits<0>) -> !fabric.bits<64> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<64>
          fabric.yield %value : !fabric.bits<64>
        }
      }
  %wide_const1 =
      fabric.pe [spatial] (%pa = %wide_const1_ctrl : !fabric.bits<0> to !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%token = %pa : !fabric.bits<64> to !fabric.bits<0>) -> !fabric.bits<64> {
          %value = fabric.op [@dataflow.constant] (%token)
              {hw_params = [{const_hex_value = ["0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004", "0x00000008"]}]}
              : (!fabric.bits<0>) -> !fabric.bits<64>
          fabric.yield %value : !fabric.bits<64>
        }
      }
  %add0 =
      fabric.pe [spatial] (%lhs = %add0_lhs : !fabric.bits<32>,
                           %rhs = %add0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %add1 =
      fabric.pe [spatial] (%lhs = %add1_lhs : !fabric.bits<32>,
                           %rhs = %add1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %add2 =
      fabric.pe [spatial] (%lhs = %add2_lhs : !fabric.bits<32>,
                           %rhs = %add2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %add3 =
      fabric.pe [spatial] (%lhs = %add3_lhs : !fabric.bits<32>,
                           %rhs = %add3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %add4 =
      fabric.pe [spatial] (%lhs = %add4_lhs : !fabric.bits<32>,
                           %rhs = %add4_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %add5 =
      fabric.pe [spatial] (%lhs = %add5_lhs : !fabric.bits<32>,
                           %rhs = %add5_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %add6 =
      fabric.pe [spatial] (%lhs = %add6_lhs : !fabric.bits<32>,
                           %rhs = %add6_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %add7 =
      fabric.pe [spatial] (%lhs = %add7_lhs : !fabric.bits<32>,
                           %rhs = %add7_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %add8 =
      fabric.pe [spatial] (%lhs = %add8_lhs : !fabric.bits<32>,
                           %rhs = %add8_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %add9 =
      fabric.pe [spatial] (%lhs = %add9_lhs : !fabric.bits<32>,
                           %rhs = %add9_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %add10 =
      fabric.pe [spatial] (%lhs = %add10_lhs : !fabric.bits<32>,
                           %rhs = %add10_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %add11 =
      fabric.pe [spatial] (%lhs = %add11_lhs : !fabric.bits<32>,
                           %rhs = %add11_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mul0 =
      fabric.pe [spatial] (%lhs = %mul0_lhs : !fabric.bits<32>,
                           %rhs = %mul0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.muli] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mul1 =
      fabric.pe [spatial] (%lhs = %mul1_lhs : !fabric.bits<32>,
                           %rhs = %mul1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.muli] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mul2 =
      fabric.pe [spatial] (%lhs = %mul2_lhs : !fabric.bits<32>,
                           %rhs = %mul2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.muli] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mul3 =
      fabric.pe [spatial] (%lhs = %mul3_lhs : !fabric.bits<32>,
                           %rhs = %mul3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.muli] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mul4 =
      fabric.pe [spatial] (%lhs = %mul4_lhs : !fabric.bits<32>,
                           %rhs = %mul4_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.muli] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mul5 =
      fabric.pe [spatial] (%lhs = %mul5_lhs : !fabric.bits<32>,
                           %rhs = %mul5_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.muli] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mul6 =
      fabric.pe [spatial] (%lhs = %mul6_lhs : !fabric.bits<32>,
                           %rhs = %mul6_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.muli] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mul7 =
      fabric.pe [spatial] (%lhs = %mul7_lhs : !fabric.bits<32>,
                           %rhs = %mul7_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.muli] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %udiv0 =
      fabric.pe [spatial] (%lhs = %udiv0_lhs : !fabric.bits<32>,
                           %rhs = %udiv0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.divui, @arith.remui] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %udiv1 =
      fabric.pe [spatial] (%lhs = %udiv1_lhs : !fabric.bits<32>,
                           %rhs = %udiv1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.divui, @arith.remui] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fp_add0 =
      fabric.pe [spatial] (%lhs = %fp_add0_lhs : !fabric.bits<32>,
                           %rhs = %fp_add0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addf, @arith.subf] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fp_add1 =
      fabric.pe [spatial] (%lhs = %fp_add1_lhs : !fabric.bits<32>,
                           %rhs = %fp_add1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addf, @arith.subf] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fp_add2 =
      fabric.pe [spatial] (%lhs = %fp_add2_lhs : !fabric.bits<32>,
                           %rhs = %fp_add2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addf, @arith.subf] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fp_add3 =
      fabric.pe [spatial] (%lhs = %fp_add3_lhs : !fabric.bits<32>,
                           %rhs = %fp_add3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.addf, @arith.subf] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fp_mul0 =
      fabric.pe [spatial] (%lhs = %fp_mul0_lhs : !fabric.bits<32>,
                           %rhs = %fp_mul0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.mulf] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fp_mul1 =
      fabric.pe [spatial] (%lhs = %fp_mul1_lhs : !fabric.bits<32>,
                           %rhs = %fp_mul1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.mulf] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fp_mul2 =
      fabric.pe [spatial] (%lhs = %fp_mul2_lhs : !fabric.bits<32>,
                           %rhs = %fp_mul2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.mulf] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fp_mul3 =
      fabric.pe [spatial] (%lhs = %fp_mul3_lhs : !fabric.bits<32>,
                           %rhs = %fp_mul3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.mulf] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fma0 =
      fabric.pe [spatial] (%lhs = %fma0_lhs : !fabric.bits<32>,
                           %rhs = %fma0_rhs : !fabric.bits<32>,
                           %acc = %fma0_acc : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>,
                  %c = %acc : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.fmuladd] (%a, %b, %c)
                   : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fma1 =
      fabric.pe [spatial] (%lhs = %fma1_lhs : !fabric.bits<32>,
                           %rhs = %fma1_rhs : !fabric.bits<32>,
                           %acc = %fma1_acc : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>,
                  %c = %acc : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.fmuladd] (%a, %b, %c)
                   : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fma2 =
      fabric.pe [spatial] (%lhs = %fma2_lhs : !fabric.bits<32>,
                           %rhs = %fma2_rhs : !fabric.bits<32>,
                           %acc = %fma2_acc : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>,
                  %c = %acc : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.fmuladd] (%a, %b, %c)
                   : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fma3 =
      fabric.pe [spatial] (%lhs = %fma3_lhs : !fabric.bits<32>,
                           %rhs = %fma3_rhs : !fabric.bits<32>,
                           %acc = %fma3_acc : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>,
                  %c = %acc : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.fmuladd] (%a, %b, %c)
                   : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fma4 =
      fabric.pe [spatial] (%lhs = %fma4_lhs : !fabric.bits<32>,
                           %rhs = %fma4_rhs : !fabric.bits<32>,
                           %acc = %fma4_acc : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>,
                  %c = %acc : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.fmuladd] (%a, %b, %c)
                   : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %fma5 =
      fabric.pe [spatial] (%lhs = %fma5_lhs : !fabric.bits<32>,
                           %rhs = %fma5_rhs : !fabric.bits<32>,
                           %acc = %fma5_acc : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>,
                  %c = %acc : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.fmuladd] (%a, %b, %c)
                   : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %and0 =
      fabric.pe [spatial] (%lhs = %and0_lhs : !fabric.bits<32>,
                           %rhs = %and0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.andi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %and1 =
      fabric.pe [spatial] (%lhs = %and1_lhs : !fabric.bits<32>,
                           %rhs = %and1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.andi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %and2 =
      fabric.pe [spatial] (%lhs = %and2_lhs : !fabric.bits<32>,
                           %rhs = %and2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.andi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %and3 =
      fabric.pe [spatial] (%lhs = %and3_lhs : !fabric.bits<32>,
                           %rhs = %and3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.andi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %and4 =
      fabric.pe [spatial] (%lhs = %and4_lhs : !fabric.bits<32>,
                           %rhs = %and4_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.andi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %and5 =
      fabric.pe [spatial] (%lhs = %and5_lhs : !fabric.bits<32>,
                           %rhs = %and5_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.andi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %and6 =
      fabric.pe [spatial] (%lhs = %and6_lhs : !fabric.bits<32>,
                           %rhs = %and6_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.andi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %and7 =
      fabric.pe [spatial] (%lhs = %and7_lhs : !fabric.bits<32>,
                           %rhs = %and7_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.andi] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %or0 =
      fabric.pe [spatial] (%lhs = %or0_lhs : !fabric.bits<32>,
                           %rhs = %or0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.ori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %or1 =
      fabric.pe [spatial] (%lhs = %or1_lhs : !fabric.bits<32>,
                           %rhs = %or1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.ori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %or2 =
      fabric.pe [spatial] (%lhs = %or2_lhs : !fabric.bits<32>,
                           %rhs = %or2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.ori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %or3 =
      fabric.pe [spatial] (%lhs = %or3_lhs : !fabric.bits<32>,
                           %rhs = %or3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.ori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %or4 =
      fabric.pe [spatial] (%lhs = %or4_lhs : !fabric.bits<32>,
                           %rhs = %or4_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.ori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %or5 =
      fabric.pe [spatial] (%lhs = %or5_lhs : !fabric.bits<32>,
                           %rhs = %or5_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.ori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %or6 =
      fabric.pe [spatial] (%lhs = %or6_lhs : !fabric.bits<32>,
                           %rhs = %or6_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.ori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %or7 =
      fabric.pe [spatial] (%lhs = %or7_lhs : !fabric.bits<32>,
                           %rhs = %or7_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.ori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %xor0 =
      fabric.pe [spatial] (%lhs = %xor0_lhs : !fabric.bits<32>,
                           %rhs = %xor0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.xori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %xor1 =
      fabric.pe [spatial] (%lhs = %xor1_lhs : !fabric.bits<32>,
                           %rhs = %xor1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.xori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %xor2 =
      fabric.pe [spatial] (%lhs = %xor2_lhs : !fabric.bits<32>,
                           %rhs = %xor2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.xori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %xor3 =
      fabric.pe [spatial] (%lhs = %xor3_lhs : !fabric.bits<32>,
                           %rhs = %xor3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.xori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %xor4 =
      fabric.pe [spatial] (%lhs = %xor4_lhs : !fabric.bits<32>,
                           %rhs = %xor4_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.xori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %xor5 =
      fabric.pe [spatial] (%lhs = %xor5_lhs : !fabric.bits<32>,
                           %rhs = %xor5_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.xori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %xor6 =
      fabric.pe [spatial] (%lhs = %xor6_lhs : !fabric.bits<32>,
                           %rhs = %xor6_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.xori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %xor7 =
      fabric.pe [spatial] (%lhs = %xor7_lhs : !fabric.bits<32>,
                           %rhs = %xor7_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.xori] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %shift0 =
      fabric.pe [spatial] (%lhs = %shift0_lhs : !fabric.bits<32>,
                           %rhs = %shift0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.shli, @arith.shrsi, @arith.shrui] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %shift1 =
      fabric.pe [spatial] (%lhs = %shift1_lhs : !fabric.bits<32>,
                           %rhs = %shift1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.shli, @arith.shrsi, @arith.shrui] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %shift2 =
      fabric.pe [spatial] (%lhs = %shift2_lhs : !fabric.bits<32>,
                           %rhs = %shift2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.shli, @arith.shrsi, @arith.shrui] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %shift3 =
      fabric.pe [spatial] (%lhs = %shift3_lhs : !fabric.bits<32>,
                           %rhs = %shift3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.shli, @arith.shrsi, @arith.shrui] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %shift4 =
      fabric.pe [spatial] (%lhs = %shift4_lhs : !fabric.bits<32>,
                           %rhs = %shift4_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.shli, @arith.shrsi, @arith.shrui] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %shift5 =
      fabric.pe [spatial] (%lhs = %shift5_lhs : !fabric.bits<32>,
                           %rhs = %shift5_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.shli, @arith.shrsi, @arith.shrui] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %shift6 =
      fabric.pe [spatial] (%lhs = %shift6_lhs : !fabric.bits<32>,
                           %rhs = %shift6_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.shli, @arith.shrsi, @arith.shrui] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %shift7 =
      fabric.pe [spatial] (%lhs = %shift7_lhs : !fabric.bits<32>,
                           %rhs = %shift7_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.shli, @arith.shrsi, @arith.shrui] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %wide_add0 =
      fabric.pe [spatial] (%lhs = %wide_add0_lhs : !fabric.bits<64>,
                           %rhs = %wide_add0_rhs : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%a = %lhs : !fabric.bits<64>,
                  %b = %rhs : !fabric.bits<64>) -> !fabric.bits<64> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
          fabric.yield %value : !fabric.bits<64>
        }
      }
  %wide_add1 =
      fabric.pe [spatial] (%lhs = %wide_add1_lhs : !fabric.bits<64>,
                           %rhs = %wide_add1_rhs : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%a = %lhs : !fabric.bits<64>,
                  %b = %rhs : !fabric.bits<64>) -> !fabric.bits<64> {
          %value = fabric.op [@arith.addi, @arith.subi] (%a, %b)
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
          fabric.yield %value : !fabric.bits<64>
        }
      }
  %wide_mul0 =
      fabric.pe [spatial] (%lhs = %wide_mul0_lhs : !fabric.bits<64>,
                           %rhs = %wide_mul0_rhs : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%a = %lhs : !fabric.bits<64>,
                  %b = %rhs : !fabric.bits<64>) -> !fabric.bits<64> {
          %value = fabric.op [@arith.muli] (%a, %b)
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
          fabric.yield %value : !fabric.bits<64>
        }
      }
  %wide_mul1 =
      fabric.pe [spatial] (%lhs = %wide_mul1_lhs : !fabric.bits<64>,
                           %rhs = %wide_mul1_rhs : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%a = %lhs : !fabric.bits<64>,
                  %b = %rhs : !fabric.bits<64>) -> !fabric.bits<64> {
          %value = fabric.op [@arith.muli] (%a, %b)
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
          fabric.yield %value : !fabric.bits<64>
        }
      }
  %wide_udiv0 =
      fabric.pe [spatial] (%lhs = %wide_udiv0_lhs : !fabric.bits<64>,
                           %rhs = %wide_udiv0_rhs : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%a = %lhs : !fabric.bits<64>,
                  %b = %rhs : !fabric.bits<64>) -> !fabric.bits<64> {
          %value = fabric.op [@arith.divui, @arith.remui] (%a, %b)
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
          fabric.yield %value : !fabric.bits<64>
        }
      }
  %wide_udiv1 =
      fabric.pe [spatial] (%lhs = %wide_udiv1_lhs : !fabric.bits<64>,
                           %rhs = %wide_udiv1_rhs : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%a = %lhs : !fabric.bits<64>,
                  %b = %rhs : !fabric.bits<64>) -> !fabric.bits<64> {
          %value = fabric.op [@arith.divui, @arith.remui] (%a, %b)
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
          fabric.yield %value : !fabric.bits<64>
        }
      }
  %wide_shift0 =
      fabric.pe [spatial] (%lhs = %wide_shift0_lhs : !fabric.bits<64>,
                           %rhs = %wide_shift0_rhs : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%a = %lhs : !fabric.bits<64>,
                  %b = %rhs : !fabric.bits<64>) -> !fabric.bits<64> {
          %value = fabric.op [@arith.shli, @arith.shrsi, @arith.shrui] (%a, %b)
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
          fabric.yield %value : !fabric.bits<64>
        }
      }
  %umin0 =
      fabric.pe [spatial] (%lhs = %umin0_lhs : !fabric.bits<32>,
                           %rhs = %umin0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.umin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %umin1 =
      fabric.pe [spatial] (%lhs = %umin1_lhs : !fabric.bits<32>,
                           %rhs = %umin1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.umin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %umin2 =
      fabric.pe [spatial] (%lhs = %umin2_lhs : !fabric.bits<32>,
                           %rhs = %umin2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.umin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %umin3 =
      fabric.pe [spatial] (%lhs = %umin3_lhs : !fabric.bits<32>,
                           %rhs = %umin3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.umin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smin0 =
      fabric.pe [spatial] (%lhs = %smin0_lhs : !fabric.bits<32>,
                           %rhs = %smin0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smin1 =
      fabric.pe [spatial] (%lhs = %smin1_lhs : !fabric.bits<32>,
                           %rhs = %smin1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smin2 =
      fabric.pe [spatial] (%lhs = %smin2_lhs : !fabric.bits<32>,
                           %rhs = %smin2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smin3 =
      fabric.pe [spatial] (%lhs = %smin3_lhs : !fabric.bits<32>,
                           %rhs = %smin3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smin4 =
      fabric.pe [spatial] (%lhs = %smin4_lhs : !fabric.bits<32>,
                           %rhs = %smin4_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smin5 =
      fabric.pe [spatial] (%lhs = %smin5_lhs : !fabric.bits<32>,
                           %rhs = %smin5_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smin6 =
      fabric.pe [spatial] (%lhs = %smin6_lhs : !fabric.bits<32>,
                           %rhs = %smin6_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smin7 =
      fabric.pe [spatial] (%lhs = %smin7_lhs : !fabric.bits<32>,
                           %rhs = %smin7_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smin8 =
      fabric.pe [spatial] (%lhs = %smin8_lhs : !fabric.bits<32>,
                           %rhs = %smin8_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smin9 =
      fabric.pe [spatial] (%lhs = %smin9_lhs : !fabric.bits<32>,
                           %rhs = %smin9_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smin] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smax0 =
      fabric.pe [spatial] (%lhs = %smax0_lhs : !fabric.bits<32>,
                           %rhs = %smax0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smax] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smax1 =
      fabric.pe [spatial] (%lhs = %smax1_lhs : !fabric.bits<32>,
                           %rhs = %smax1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smax] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smax2 =
      fabric.pe [spatial] (%lhs = %smax2_lhs : !fabric.bits<32>,
                           %rhs = %smax2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smax] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smax3 =
      fabric.pe [spatial] (%lhs = %smax3_lhs : !fabric.bits<32>,
                           %rhs = %smax3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smax] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smax4 =
      fabric.pe [spatial] (%lhs = %smax4_lhs : !fabric.bits<32>,
                           %rhs = %smax4_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smax] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smax5 =
      fabric.pe [spatial] (%lhs = %smax5_lhs : !fabric.bits<32>,
                           %rhs = %smax5_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smax] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smax6 =
      fabric.pe [spatial] (%lhs = %smax6_lhs : !fabric.bits<32>,
                           %rhs = %smax6_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smax] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smax7 =
      fabric.pe [spatial] (%lhs = %smax7_lhs : !fabric.bits<32>,
                           %rhs = %smax7_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smax] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smax8 =
      fabric.pe [spatial] (%lhs = %smax8_lhs : !fabric.bits<32>,
                           %rhs = %smax8_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smax] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %smax9 =
      fabric.pe [spatial] (%lhs = %smax9_lhs : !fabric.bits<32>,
                           %rhs = %smax9_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@llvm.intr.smax] (%a, %b)
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %cmp0 =
      fabric.pe [spatial] (%lhs = %cmp0_lhs : !fabric.bits<32>,
                           %rhs = %cmp0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %cmp1 =
      fabric.pe [spatial] (%lhs = %cmp1_lhs : !fabric.bits<32>,
                           %rhs = %cmp1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %cmp2 =
      fabric.pe [spatial] (%lhs = %cmp2_lhs : !fabric.bits<32>,
                           %rhs = %cmp2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %cmp3 =
      fabric.pe [spatial] (%lhs = %cmp3_lhs : !fabric.bits<32>,
                           %rhs = %cmp3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %cmp4 =
      fabric.pe [spatial] (%lhs = %cmp4_lhs : !fabric.bits<32>,
                           %rhs = %cmp4_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %cmp5 =
      fabric.pe [spatial] (%lhs = %cmp5_lhs : !fabric.bits<32>,
                           %rhs = %cmp5_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %cmp6 =
      fabric.pe [spatial] (%lhs = %cmp6_lhs : !fabric.bits<32>,
                           %rhs = %cmp6_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %cmp7 =
      fabric.pe [spatial] (%lhs = %cmp7_lhs : !fabric.bits<32>,
                           %rhs = %cmp7_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %cmp8 =
      fabric.pe [spatial] (%lhs = %cmp8_lhs : !fabric.bits<32>,
                           %rhs = %cmp8_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %cmp9 =
      fabric.pe [spatial] (%lhs = %cmp9_lhs : !fabric.bits<32>,
                           %rhs = %cmp9_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %cmp10 =
      fabric.pe [spatial] (%lhs = %cmp10_lhs : !fabric.bits<32>,
                           %rhs = %cmp10_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %cmp11 =
      fabric.pe [spatial] (%lhs = %cmp11_lhs : !fabric.bits<32>,
                           %rhs = %cmp11_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %wide_cmp0 =
      fabric.pe [spatial] (%lhs = %wide_cmp0_lhs : !fabric.bits<64>,
                           %rhs = %wide_cmp0_rhs : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%a = %lhs : !fabric.bits<64>,
                  %b = %rhs : !fabric.bits<64>) -> !fabric.bits<64> {
          %pred = fabric.op [@arith.cmpi, @llvm.icmp] (%a, %b)
              {hw_params = [{predicate = ["eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"]}]}
              : (!fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<64>
        }
      }
  %wide_cmp0_pred = fabric.fifo %wide_cmp0 [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %fp_cmp0 =
      fabric.pe [spatial] (%lhs = %fp_cmp0_lhs : !fabric.bits<32>,
                           %rhs = %fp_cmp0_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpf] (%a, %b)
              {hw_params = [{predicate = ["oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %fp_cmp1 =
      fabric.pe [spatial] (%lhs = %fp_cmp1_lhs : !fabric.bits<32>,
                           %rhs = %fp_cmp1_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpf] (%a, %b)
              {hw_params = [{predicate = ["oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %fp_cmp2 =
      fabric.pe [spatial] (%lhs = %fp_cmp2_lhs : !fabric.bits<32>,
                           %rhs = %fp_cmp2_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpf] (%a, %b)
              {hw_params = [{predicate = ["oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %fp_cmp3 =
      fabric.pe [spatial] (%lhs = %fp_cmp3_lhs : !fabric.bits<32>,
                           %rhs = %fp_cmp3_rhs : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%a = %lhs : !fabric.bits<32>,
                  %b = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
          %pred = fabric.op [@arith.cmpf] (%a, %b)
              {hw_params = [{predicate = ["oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult", "ule", "une", "uno"]}]}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<1>
          fabric.yield %pred : !fabric.bits<1> to !fabric.bits<32>
        }
      }
  %select0 =
      fabric.pe [spatial] (%pred = %select0_pred : !fabric.bits<32>,
                           %true_value = %select0_true : !fabric.bits<32>,
                           %false_value = %select0_false : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %true_value : !fabric.bits<32>,
                  %b = %false_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.select] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %select1 =
      fabric.pe [spatial] (%pred = %select1_pred : !fabric.bits<32>,
                           %true_value = %select1_true : !fabric.bits<32>,
                           %false_value = %select1_false : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %true_value : !fabric.bits<32>,
                  %b = %false_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.select] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %select2 =
      fabric.pe [spatial] (%pred = %select2_pred : !fabric.bits<32>,
                           %true_value = %select2_true : !fabric.bits<32>,
                           %false_value = %select2_false : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %true_value : !fabric.bits<32>,
                  %b = %false_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.select] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %select3 =
      fabric.pe [spatial] (%pred = %select3_pred : !fabric.bits<32>,
                           %true_value = %select3_true : !fabric.bits<32>,
                           %false_value = %select3_false : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %true_value : !fabric.bits<32>,
                  %b = %false_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.select] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %select4 =
      fabric.pe [spatial] (%pred = %select4_pred : !fabric.bits<32>,
                           %true_value = %select4_true : !fabric.bits<32>,
                           %false_value = %select4_false : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %true_value : !fabric.bits<32>,
                  %b = %false_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.select] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %select5 =
      fabric.pe [spatial] (%pred = %select5_pred : !fabric.bits<32>,
                           %true_value = %select5_true : !fabric.bits<32>,
                           %false_value = %select5_false : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %true_value : !fabric.bits<32>,
                  %b = %false_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.select] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %select6 =
      fabric.pe [spatial] (%pred = %select6_pred : !fabric.bits<32>,
                           %true_value = %select6_true : !fabric.bits<32>,
                           %false_value = %select6_false : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %true_value : !fabric.bits<32>,
                  %b = %false_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.select] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %select7 =
      fabric.pe [spatial] (%pred = %select7_pred : !fabric.bits<32>,
                           %true_value = %select7_true : !fabric.bits<32>,
                           %false_value = %select7_false : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %true_value : !fabric.bits<32>,
                  %b = %false_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@arith.select] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mux0 =
      fabric.pe [spatial] (%pred = %mux0_pred : !fabric.bits<32>,
                           %false_value = %mux0_false : !fabric.bits<32>,
                           %true_value = %mux0_true : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %false_value : !fabric.bits<32>,
                  %b = %true_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.mux] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mux1 =
      fabric.pe [spatial] (%pred = %mux1_pred : !fabric.bits<32>,
                           %false_value = %mux1_false : !fabric.bits<32>,
                           %true_value = %mux1_true : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %false_value : !fabric.bits<32>,
                  %b = %true_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.mux] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mux2 =
      fabric.pe [spatial] (%pred = %mux2_pred : !fabric.bits<32>,
                           %false_value = %mux2_false : !fabric.bits<32>,
                           %true_value = %mux2_true : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %false_value : !fabric.bits<32>,
                  %b = %true_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.mux] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %mux3 =
      fabric.pe [spatial] (%pred = %mux3_pred : !fabric.bits<32>,
                           %false_value = %mux3_false : !fabric.bits<32>,
                           %true_value = %mux3_true : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%sel = %pred : !fabric.bits<32> to !fabric.bits<1>,
                  %a = %false_value : !fabric.bits<32>,
                  %b = %true_value : !fabric.bits<32>) -> !fabric.bits<32> {
          %value = fabric.op [@dataflow.mux] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %value : !fabric.bits<32>
        }
      }
  %wide_mux0 =
      fabric.pe [spatial] (%pred = %wide_mux0_pred : !fabric.bits<64>,
                           %false_value = %wide_mux0_false : !fabric.bits<64>,
                           %true_value = %wide_mux0_true : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%sel = %pred : !fabric.bits<64> to !fabric.bits<1>,
                  %a = %false_value : !fabric.bits<64>,
                  %b = %true_value : !fabric.bits<64>) -> !fabric.bits<64> {
          %value = fabric.op [@dataflow.mux] (%sel, %a, %b)
              : (!fabric.bits<1>, !fabric.bits<64>, !fabric.bits<64>) -> !fabric.bits<64>
          fabric.yield %value : !fabric.bits<64>
        }
      }
  %cast0 =
      fabric.pe [spatial] (%value = %cast0_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %cast1 =
      fabric.pe [spatial] (%value = %cast1_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %cast2 =
      fabric.pe [spatial] (%value = %cast2_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %cast3 =
      fabric.pe [spatial] (%value = %cast3_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %cast4 =
      fabric.pe [spatial] (%value = %cast4_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %cast5 =
      fabric.pe [spatial] (%value = %cast5_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %cast6 =
      fabric.pe [spatial] (%value = %cast6_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %cast7 =
      fabric.pe [spatial] (%value = %cast7_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %sext0 =
      fabric.pe [spatial] (%value = %sext0_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %sext1 =
      fabric.pe [spatial] (%value = %sext1_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %sext2 =
      fabric.pe [spatial] (%value = %sext2_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %sext3 =
      fabric.pe [spatial] (%value = %sext3_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %sext4 =
      fabric.pe [spatial] (%value = %sext4_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %sext5 =
      fabric.pe [spatial] (%value = %sext5_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %sext6 =
      fabric.pe [spatial] (%value = %sext6_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %sext7 =
      fabric.pe [spatial] (%value = %sext7_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %zext0 =
      fabric.pe [spatial] (%value = %zext0_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %zext1 =
      fabric.pe [spatial] (%value = %zext1_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %zext2 =
      fabric.pe [spatial] (%value = %zext2_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %zext3 =
      fabric.pe [spatial] (%value = %zext3_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %zext4 =
      fabric.pe [spatial] (%value = %zext4_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %zext5 =
      fabric.pe [spatial] (%value = %zext5_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %zext6 =
      fabric.pe [spatial] (%value = %zext6_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %zext7 =
      fabric.pe [spatial] (%value = %zext7_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %trunci0 =
      fabric.pe [spatial] (%value = %trunci0_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@arith.trunci] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %trunci1 =
      fabric.pe [spatial] (%value = %trunci1_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@arith.trunci] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %trunci2 =
      fabric.pe [spatial] (%value = %trunci2_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@arith.trunci] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %trunci3 =
      fabric.pe [spatial] (%value = %trunci3_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@arith.trunci] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %wide_zext0 =
      fabric.pe [spatial] (%value = %wide_zext0_input : !fabric.bits<32> to !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64> to !fabric.bits<32>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<64>
          fabric.yield %result : !fabric.bits<64>
        }
      }
  %wide_zext1 =
      fabric.pe [spatial] (%value = %wide_zext1_input : !fabric.bits<32> to !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64> to !fabric.bits<32>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<64>
          fabric.yield %result : !fabric.bits<64>
        }
      }
  %wide_zext2 =
      fabric.pe [spatial] (%value = %wide_zext2_input : !fabric.bits<32> to !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64> to !fabric.bits<32>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<64>
          fabric.yield %result : !fabric.bits<64>
        }
      }
  %wide_zext3 =
      fabric.pe [spatial] (%value = %wide_zext3_input : !fabric.bits<32> to !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64> to !fabric.bits<32>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.zext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<64>
          fabric.yield %result : !fabric.bits<64>
        }
      }
  %wide_sext0 =
      fabric.pe [spatial] (%value = %wide_sext0_input : !fabric.bits<32> to !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64> to !fabric.bits<32>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<64>
          fabric.yield %result : !fabric.bits<64>
        }
      }
  %wide_sext1 =
      fabric.pe [spatial] (%value = %wide_sext1_input : !fabric.bits<32> to !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64> to !fabric.bits<32>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<64>
          fabric.yield %result : !fabric.bits<64>
        }
      }
  %wide_sext2 =
      fabric.pe [spatial] (%value = %wide_sext2_input : !fabric.bits<32> to !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64> to !fabric.bits<32>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<64>
          fabric.yield %result : !fabric.bits<64>
        }
      }
  %wide_sext3 =
      fabric.pe [spatial] (%value = %wide_sext3_input : !fabric.bits<32> to !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64> to !fabric.bits<32>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.sext] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<64>
          fabric.yield %result : !fabric.bits<64>
        }
      }
  %wide_trunc0_wide =
      fabric.pe [spatial] (%value = %wide_trunc0_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_trunc0 = fabric.fifo %wide_trunc0_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_trunc1_wide =
      fabric.pe [spatial] (%value = %wide_trunc1_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_trunc1 = fabric.fifo %wide_trunc1_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_trunc2_wide =
      fabric.pe [spatial] (%value = %wide_trunc2_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_trunc2 = fabric.fifo %wide_trunc2_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_trunc3_wide =
      fabric.pe [spatial] (%value = %wide_trunc3_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@llvm.trunc] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_trunc3 = fabric.fifo %wide_trunc3_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_index_cast0_wide =
      fabric.pe [spatial] (%value = %wide_index_cast0_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@arith.index_cast] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_index_cast0 = fabric.fifo %wide_index_cast0_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_index_cast1_wide =
      fabric.pe [spatial] (%value = %wide_index_cast1_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@arith.index_cast] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_index_cast1 = fabric.fifo %wide_index_cast1_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_index_cast2_wide =
      fabric.pe [spatial] (%value = %wide_index_cast2_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@arith.index_cast] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_index_cast2 = fabric.fifo %wide_index_cast2_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_index_cast3_wide =
      fabric.pe [spatial] (%value = %wide_index_cast3_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@arith.index_cast] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_index_cast3 = fabric.fifo %wide_index_cast3_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_index_cast4_wide =
      fabric.pe [spatial] (%value = %wide_index_cast4_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@arith.index_cast] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_index_cast4 = fabric.fifo %wide_index_cast4_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_index_cast5_wide =
      fabric.pe [spatial] (%value = %wide_index_cast5_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@arith.index_cast] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_index_cast5 = fabric.fifo %wide_index_cast5_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_index_castui0_wide =
      fabric.pe [spatial] (%value = %wide_index_castui0_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@arith.index_castui] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_index_castui0 = fabric.fifo %wide_index_castui0_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %wide_index_castui1_wide =
      fabric.pe [spatial] (%value = %wide_index_castui1_input : !fabric.bits<64>)
          -> !fabric.bits<64> {
        fabric.fu(%input = %value : !fabric.bits<64>) -> !fabric.bits<64> {
          %result = fabric.op [@arith.index_castui] (%input)
                   : (!fabric.bits<64>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32> to !fabric.bits<64>
        }
      }
  %wide_index_castui1 = fabric.fifo %wide_index_castui1_wide [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<64> to !fabric.bits<32>
  %extui0 =
      fabric.pe [spatial] (%value = %extui0_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@arith.extui] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %extui1 =
      fabric.pe [spatial] (%value = %extui1_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@arith.extui] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %extui2 =
      fabric.pe [spatial] (%value = %extui2_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@arith.extui] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %extui3 =
      fabric.pe [spatial] (%value = %extui3_input : !fabric.bits<32>)
          -> !fabric.bits<32> {
        fabric.fu(%input = %value : !fabric.bits<32>) -> !fabric.bits<32> {
          %result = fabric.op [@arith.extui] (%input)
                   : (!fabric.bits<32>) -> !fabric.bits<32>
          fabric.yield %result : !fabric.bits<32>
        }
      }
  %add0_lhs, %add0_rhs, %add1_lhs, %add1_rhs, %add2_lhs, %add2_rhs, %add3_lhs, %add3_rhs, %add4_lhs, %add4_rhs, %add5_lhs, %add5_rhs, %add6_lhs, %add6_rhs, %add7_lhs, %add7_rhs, %add8_lhs, %add8_rhs, %add9_lhs, %add9_rhs, %add10_lhs, %add10_rhs, %add11_lhs, %add11_rhs, %mul0_lhs, %mul0_rhs, %mul1_lhs, %mul1_rhs, %mul2_lhs, %mul2_rhs, %mul3_lhs, %mul3_rhs, %mul4_lhs, %mul4_rhs, %mul5_lhs, %mul5_rhs, %mul6_lhs, %mul6_rhs, %mul7_lhs, %mul7_rhs, %udiv0_lhs, %udiv0_rhs, %udiv1_lhs, %udiv1_rhs, %fp_add0_lhs, %fp_add0_rhs, %fp_add1_lhs, %fp_add1_rhs, %fp_add2_lhs, %fp_add2_rhs, %fp_add3_lhs, %fp_add3_rhs, %fp_mul0_lhs, %fp_mul0_rhs, %fp_mul1_lhs, %fp_mul1_rhs, %fp_mul2_lhs, %fp_mul2_rhs, %fp_mul3_lhs, %fp_mul3_rhs, %fma0_lhs, %fma0_rhs, %fma0_acc, %fma1_lhs, %fma1_rhs, %fma1_acc, %fma2_lhs, %fma2_rhs, %fma2_acc, %fma3_lhs, %fma3_rhs, %fma3_acc, %fma4_lhs, %fma4_rhs, %fma4_acc, %fma5_lhs, %fma5_rhs, %fma5_acc, %and0_lhs, %and0_rhs, %and1_lhs, %and1_rhs, %and2_lhs, %and2_rhs, %and3_lhs, %and3_rhs, %and4_lhs, %and4_rhs, %and5_lhs, %and5_rhs, %and6_lhs, %and6_rhs, %and7_lhs, %and7_rhs, %or0_lhs, %or0_rhs, %or1_lhs, %or1_rhs, %or2_lhs, %or2_rhs, %or3_lhs, %or3_rhs, %or4_lhs, %or4_rhs, %or5_lhs, %or5_rhs, %or6_lhs, %or6_rhs, %or7_lhs, %or7_rhs, %xor0_lhs, %xor0_rhs, %xor1_lhs, %xor1_rhs, %xor2_lhs, %xor2_rhs, %xor3_lhs, %xor3_rhs, %xor4_lhs, %xor4_rhs, %xor5_lhs, %xor5_rhs, %xor6_lhs, %xor6_rhs, %xor7_lhs, %xor7_rhs, %shift0_lhs, %shift0_rhs, %shift1_lhs, %shift1_rhs, %shift2_lhs, %shift2_rhs, %shift3_lhs, %shift3_rhs, %shift4_lhs, %shift4_rhs, %shift5_lhs, %shift5_rhs, %shift6_lhs, %shift6_rhs, %shift7_lhs, %shift7_rhs, %umin0_lhs, %umin0_rhs, %umin1_lhs, %umin1_rhs, %umin2_lhs, %umin2_rhs, %umin3_lhs, %umin3_rhs, %smin0_lhs, %smin0_rhs, %smin1_lhs, %smin1_rhs, %smin2_lhs, %smin2_rhs, %smin3_lhs, %smin3_rhs, %smin4_lhs, %smin4_rhs, %smin5_lhs, %smin5_rhs, %smin6_lhs, %smin6_rhs, %smin7_lhs, %smin7_rhs, %smin8_lhs, %smin8_rhs, %smin9_lhs, %smin9_rhs, %smax0_lhs, %smax0_rhs, %smax1_lhs, %smax1_rhs, %smax2_lhs, %smax2_rhs, %smax3_lhs, %smax3_rhs, %smax4_lhs, %smax4_rhs, %smax5_lhs, %smax5_rhs, %smax6_lhs, %smax6_rhs, %smax7_lhs, %smax7_rhs, %smax8_lhs, %smax8_rhs, %smax9_lhs, %smax9_rhs, %cmp0_lhs, %cmp0_rhs, %cmp1_lhs, %cmp1_rhs, %cmp2_lhs, %cmp2_rhs, %cmp3_lhs, %cmp3_rhs, %cmp4_lhs, %cmp4_rhs, %cmp5_lhs, %cmp5_rhs, %cmp6_lhs, %cmp6_rhs, %cmp7_lhs, %cmp7_rhs, %cmp8_lhs, %cmp8_rhs, %cmp9_lhs, %cmp9_rhs, %cmp10_lhs, %cmp10_rhs, %cmp11_lhs, %cmp11_rhs, %fp_cmp0_lhs, %fp_cmp0_rhs, %fp_cmp1_lhs, %fp_cmp1_rhs, %fp_cmp2_lhs, %fp_cmp2_rhs, %fp_cmp3_lhs, %fp_cmp3_rhs, %select0_pred, %select0_true, %select0_false, %select1_pred, %select1_true, %select1_false, %select2_pred, %select2_true, %select2_false, %select3_pred, %select3_true, %select3_false, %select4_pred, %select4_true, %select4_false, %select5_pred, %select5_true, %select5_false, %select6_pred, %select6_true, %select6_false, %select7_pred, %select7_true, %select7_false, %mux0_pred, %mux0_false, %mux0_true, %mux1_pred, %mux1_false, %mux1_true, %mux2_pred, %mux2_false, %mux2_true, %mux3_pred, %mux3_false, %mux3_true, %cast0_input, %cast1_input, %cast2_input, %cast3_input, %cast4_input, %cast5_input, %cast6_input, %cast7_input, %sext0_input, %sext1_input, %sext2_input, %sext3_input, %sext4_input, %sext5_input, %sext6_input, %sext7_input, %zext0_input, %zext1_input, %zext2_input, %zext3_input, %zext4_input, %zext5_input, %zext6_input, %zext7_input, %trunci0_input, %trunci1_input, %trunci2_input, %trunci3_input, %wide_zext0_input, %wide_zext1_input, %wide_zext2_input, %wide_zext3_input, %wide_sext0_input, %wide_sext1_input, %wide_sext2_input, %wide_sext3_input, %extui0_input, %extui1_input, %extui2_input, %extui3_input, %load_addr0, %load_addr1, %load_addr2, %load_addr3, %load_addr4, %load_addr5, %load_addr6, %load_addr7, %load_addr8, %load_addr9, %load_addr10, %load_addr11, %load_addr12, %load_addr13, %load_addr14, %load_addr15, %load_addr16, %load_addr17, %store_addr0, %store_value0, %store_addr1, %store_value1, %store_addr2, %store_value2, %store_addr3, %store_value3, %store_addr4, %store_value4, %store_addr5, %store_value5, %store_addr6, %store_value6, %store_addr7, %store_value7, %store_addr8, %store_value8, %wide_route_bridge0_input, %wide_route_bridge1_input =
      fabric.switch [spatial] %i32a, %i32b, %i32c, %i32d, %const0, %const1, %const2, %const3, %const4, %const5, %const6, %const7, %const8, %const9, %const10, %const11, %const12, %const13, %const14, %const15, %const16, %const17, %const18, %const19, %const20, %const21, %const22, %const23, %const24, %const25, %const26, %const27, %const28, %const29, %add0, %add1, %add2, %add3, %add4, %add5, %add6, %add7, %add8, %add9, %add10, %add11, %mul0, %mul1, %mul2, %mul3, %mul4, %mul5, %mul6, %mul7, %udiv0, %udiv1, %fp_add0, %fp_add1, %fp_add2, %fp_add3, %fp_mul0, %fp_mul1, %fp_mul2, %fp_mul3, %fma0, %fma1, %fma2, %fma3, %fma4, %fma5, %and0, %and1, %and2, %and3, %and4, %and5, %and6, %and7, %or0, %or1, %or2, %or3, %or4, %or5, %or6, %or7, %xor0, %xor1, %xor2, %xor3, %xor4, %xor5, %xor6, %xor7, %shift0, %shift1, %shift2, %shift3, %shift4, %shift5, %shift6, %shift7, %umin0, %umin1, %umin2, %umin3, %smin0, %smin1, %smin2, %smin3, %smin4, %smin5, %smin6, %smin7, %smin8, %smin9, %smax0, %smax1, %smax2, %smax3, %smax4, %smax5, %smax6, %smax7, %smax8, %smax9, %cmp0, %cmp1, %cmp2, %cmp3, %cmp4, %cmp5, %cmp6, %cmp7, %cmp8, %cmp9, %cmp10, %cmp11, %wide_cmp0_pred, %fp_cmp0, %fp_cmp1, %fp_cmp2, %fp_cmp3, %select0, %select1, %select2, %select3, %select4, %select5, %select6, %select7, %mux0, %mux1, %mux2, %mux3, %cast0, %cast1, %cast2, %cast3, %cast4, %cast5, %cast6, %cast7, %sext0, %sext1, %sext2, %sext3, %sext4, %sext5, %sext6, %sext7, %zext0, %zext1, %zext2, %zext3, %zext4, %zext5, %zext6, %zext7, %trunci0, %trunci1, %trunci2, %trunci3, %wide_trunc0, %wide_trunc1, %wide_trunc2, %wide_trunc3, %wide_index_cast0, %wide_index_cast1, %wide_index_cast2, %wide_index_cast3, %wide_index_cast4, %wide_index_cast5, %wide_index_castui0, %wide_index_castui1, %extui0, %extui1, %extui2, %extui3, %data0, %data1, %data2, %data3, %data4, %data5, %data6, %data7, %data8, %data9, %data10, %data11, %data12, %data13, %data14, %data15, %data16, %data17
        [{connectivity_table = ["1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"]}]
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  %wide_route_bridge0 = fabric.fifo %wide_route_bridge0_input [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<32> to !fabric.bits<64>
  %wide_route_bridge1 = fabric.fifo %wide_route_bridge1_input [max_depth = 1, bypassable = true] {bypassed = true}
    : !fabric.bits<32> to !fabric.bits<64>
  %wide_add0_lhs, %wide_add0_rhs, %wide_add1_lhs, %wide_add1_rhs, %wide_mul0_lhs, %wide_mul0_rhs, %wide_mul1_lhs, %wide_mul1_rhs, %wide_udiv0_lhs, %wide_udiv0_rhs, %wide_udiv1_lhs, %wide_udiv1_rhs, %wide_shift0_lhs, %wide_shift0_rhs, %wide_cmp0_lhs, %wide_cmp0_rhs, %wide_mux0_pred, %wide_mux0_false, %wide_mux0_true, %wide_trunc0_input, %wide_trunc1_input, %wide_trunc2_input, %wide_trunc3_input, %wide_index_cast0_input, %wide_index_cast1_input, %wide_index_cast2_input, %wide_index_cast3_input, %wide_index_cast4_input, %wide_index_cast5_input, %wide_index_castui0_input, %wide_index_castui1_input =
      fabric.switch [spatial] %wide_const0, %wide_const1, %wide_add0, %wide_add1, %wide_mul0, %wide_mul1, %wide_udiv0, %wide_udiv1, %wide_shift0, %wide_mux0, %wide_zext0, %wide_zext1, %wide_zext2, %wide_zext3, %wide_sext0, %wide_sext1, %wide_sext2, %wide_sext3, %wide_route_bridge0, %wide_route_bridge1
        [{connectivity_table = ["11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111", "11111111111111111111"]}]
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
        -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  %const0_ctrl, %const1_ctrl, %const2_ctrl, %const3_ctrl, %const4_ctrl, %const5_ctrl, %const6_ctrl, %const7_ctrl, %const8_ctrl, %const9_ctrl, %const10_ctrl, %const11_ctrl, %const12_ctrl, %const13_ctrl, %const14_ctrl, %const15_ctrl, %const16_ctrl, %const17_ctrl, %const18_ctrl, %const19_ctrl, %const20_ctrl, %const21_ctrl, %const22_ctrl, %const23_ctrl, %const24_ctrl, %const25_ctrl, %const26_ctrl, %const27_ctrl, %const28_ctrl, %const29_ctrl, %wide_const0_ctrl, %wide_const1_ctrl, %sync0_in0, %sync0_in1, %sync0_in2, %sync0_in3, %sync0_in4, %sync0_in5, %sync1_in0, %sync1_in1, %sync1_in2, %sync1_in3, %sync1_in4, %sync1_in5, %sync2_in0, %sync2_in1, %sync2_in2, %sync2_in3, %sync2_in4, %sync2_in5, %sync3_in0, %sync3_in1, %sync3_in2, %sync3_in3, %sync3_in4, %sync3_in5, %load_ctrl0, %load_ctrl1, %load_ctrl2, %load_ctrl3, %load_ctrl4, %load_ctrl5, %load_ctrl6, %load_ctrl7, %load_ctrl8, %load_ctrl9, %load_ctrl10, %load_ctrl11, %load_ctrl12, %load_ctrl13, %load_ctrl14, %load_ctrl15, %load_ctrl16, %load_ctrl17, %store_ctrl0, %store_ctrl1, %store_ctrl2, %store_ctrl3, %store_ctrl4, %store_ctrl5, %store_ctrl6, %store_ctrl7, %store_ctrl8 =
      fabric.switch [spatial] %ctrl, %sync0_done0, %sync0_done1, %sync0_done2, %sync0_done3, %sync0_done4, %sync0_done5, %sync1_done0, %sync1_done1, %sync1_done2, %sync1_done3, %sync1_done4, %sync1_done5, %sync2_done0, %sync2_done1, %sync2_done2, %sync2_done3, %sync2_done4, %sync2_done5, %sync3_done0, %sync3_done1, %sync3_done2, %sync3_done3, %sync3_done4, %sync3_done5, %done0, %done1, %done2, %done3, %done4, %done5, %done6, %done7, %done8, %done9, %done10, %done11, %done12, %done13, %done14, %done15, %done16, %done17, %store_done0, %store_done1, %store_done2, %store_done3, %store_done4, %store_done5, %store_done6, %store_done7, %store_done8
        [{connectivity_table = ["1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111", "1111111111111111111111111111111111111111111111111111"]}]
        : (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
        -> (!fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  %data0, %done0, %data1, %done1, %data2, %done2, %data3, %done3, %data4, %done4, %data5, %done5, %data6, %done6, %data7, %done7, %data8, %done8, %data9, %done9, %data10, %done10, %data11, %done11, %data12, %done12, %data13, %done13, %data14, %done14, %data15, %done15, %data16, %done16, %data17, %done17, %store_done0, %store_done1, %store_done2, %store_done3, %store_done4, %store_done5, %store_done6, %store_done7, %store_done8 =
      fabric.mem [spatial] mgr(%mgr) load(%load_addr0, %load_ctrl0, %load_addr1, %load_ctrl1, %load_addr2, %load_ctrl2, %load_addr3, %load_ctrl3, %load_addr4, %load_ctrl4, %load_addr5, %load_ctrl5, %load_addr6, %load_ctrl6, %load_addr7, %load_ctrl7, %load_addr8, %load_ctrl8, %load_addr9, %load_ctrl9, %load_addr10, %load_ctrl10, %load_addr11, %load_ctrl11, %load_addr12, %load_ctrl12, %load_addr13, %load_ctrl13, %load_addr14, %load_ctrl14, %load_addr15, %load_ctrl15, %load_addr16, %load_ctrl16, %load_addr17, %load_ctrl17)
                                store(%store_addr0, %store_value0, %store_ctrl0, %store_addr1, %store_value1, %store_ctrl1, %store_addr2, %store_value2, %store_ctrl2, %store_addr3, %store_value3, %store_ctrl3, %store_addr4, %store_value4, %store_ctrl4, %store_addr5, %store_value5, %store_ctrl5, %store_addr6, %store_value6, %store_ctrl6, %store_addr7, %store_value7, %store_ctrl7, %store_addr8, %store_value8, %store_ctrl8)
        [{load_group_size = 18 : i32, store_group_size = 9 : i32}]
        : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>, !fabric.bits<0>)
  fabric.yield
}

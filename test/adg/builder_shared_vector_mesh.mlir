// RUN: loom-adg-builder-test --shared-vector-mesh --output %t.hardware.mlir
// RUN: FileCheck %s --check-prefix=CONTROL < %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE

// CONTROL-LABEL: fabric.module @shared_vector_mesh_adg
// CONTROL: %data0, %done0, %data1, %done1, %store_done = fabric.mem [spatial] mgr(%mgr)
// CONTROL-NEXT: load(%idx0, %load_ctrl0, %idx1, %load_ctrl1)
// CONTROL-NEXT: store(%store_idx, %store_value, %store_ctrl)
// CONTROL: %load_ctrl0, %load_ctrl1, %store_ctrl, %sync0, %sync1, %sync2 = fabric.switch [spatial] %ctrl, %done0, %done1, %store_done
// CONTROL-NEXT: [{connectivity_table = ["1111", "1111", "1111", "1111", "1111", "1111"]}]

// HARDWARE-LABEL: fabric.module @shared_vector_mesh_adg
// HARDWARE-DAG: fabric.mem [spatial]
// HARDWARE-DAG: fabric.switch [spatial]
// HARDWARE-DAG: fabric.op [@arith.xori]
// HARDWARE-DAG: fabric.op [@llvm.intr.bswap]
// HARDWARE-DAG: fabric.op [@dataflow.sync]
// HARDWARE-NOT: fabric.link

// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-vector-math --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE
// RUN: %loom-cc -emit-llvm -O1 -S %S/../app/quat_mult/main_func.cpp -o %t.dir/quat_mult.ll
// RUN: %loom-raise %t.dir/quat_mult.ll -o %t.dir/quat_mult.scf.mlir
// RUN: %loom-lower %t.dir/quat_mult.scf.mlir -o %t.dir/quat_mult.dfg.mlir
// RUN: FileCheck %s --check-prefix=DFG < %t.dir/quat_mult.dfg.mlir
// RUN: loom-pnr-map --dfg-mlir %t.dir/quat_mult.dfg.mlir --graph g_quat_mult_kernel_0 --hardware-mlir %t.hardware.mlir --hardware shared_vector_math_adg --workload quat_mult --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/mapping.json

// DFG-LABEL: dataflow.graph private @g_quat_mult_kernel_0
// DFG-DAG: builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xf32>
// DFG-DAG: builtin.unrealized_conversion_cast %arg3 : !llvm.ptr to memref<?xf32>
// DFG-DAG: builtin.unrealized_conversion_cast %arg2 : !llvm.ptr to memref<?xf32>
// DFG-NOT: dataflow.demux {{.*}} : (i1, !llvm.ptr)
// DFG-NOT: dataflow.invariant {{.*}} : !llvm.ptr
// DFG: dataflow.stream
// DFG: arith.index_cast {{.*}} : i64 to index
// DFG-NOT: dataflow.demux {{.*}} : (i1, !llvm.ptr)
// DFG-NOT: dataflow.invariant {{.*}} : !llvm.ptr
// DFG: dataflow.graph.return

// HARDWARE-LABEL: fabric.module @shared_vector_math_adg
// HARDWARE-DAG: load_group_size = 8 : i32
// HARDWARE-DAG: store_group_size = 4 : i32
// HARDWARE-DAG: fabric.op [@dataflow.constant]
// HARDWARE-DAG: fabric.op [@arith.cmpi, @llvm.icmp]
// HARDWARE-DAG: fabric.op [@arith.ori]
// HARDWARE-DAG: fabric.op [@arith.shli, @arith.shrsi, @arith.shrui]
// HARDWARE-DAG: fabric.op [@arith.mulf]
// HARDWARE-DAG: fabric.op [@llvm.fneg]
// HARDWARE-DAG: fabric.op [@llvm.intr.fmuladd]
// HARDWARE-DAG: fabric.op [@llvm.zext]
// HARDWARE-DAG: fabric.op [@llvm.trunc]
// HARDWARE-DAG: fabric.op [@dataflow.stream]
// HARDWARE-DAG: (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<1>)
// HARDWARE-DAG: fabric.op [@dataflow.carry]
// HARDWARE-DAG: fabric.op [@dataflow.invariant]
// HARDWARE-DAG: fabric.op [@dataflow.gate]
// HARDWARE-DAG: fabric.op [@dataflow.demux]
// HARDWARE-DAG: fabric.op [@dataflow.mux]
// HARDWARE-DAG: fabric.op [@arith.index_cast]
// HARDWARE-DAG: fabric.op [@dataflow.sync]
// HARDWARE-DAG: fabric.mem [spatial]

// MAPPING-DAG: "workload": "quat_mult"
// MAPPING-DAG: "hardware": "shared_vector_math_adg"
// MAPPING-DAG: "unplaced_records": 0
// MAPPING-DAG: "unrouted_edges": 0
// MAPPING-DAG: "status": "pass"

// RUN: loom-adg-builder-test --shared-vector-alu --output %t.hardware.mlir
// RUN: loom %t.hardware.mlir | FileCheck %s --check-prefix=HARDWARE
// RUN: bash %S/../fabric/run_adg_hardware_summary.sh --input %t.hardware.mlir --input-recipe-identity %t.hardware.mlir=adg-builder::shared-vector-alu --output %t.hardware.csv
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.hardware.csv
// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/byte_swap LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/byte_swap/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %t.hardware.mlir --hardware shared_vector_alu_adg --workload byte_swap --output %t.dir/byte.mapping.csv --artifact %t.dir/byte.mapping.json
// RUN: FileCheck %s --check-prefix=BYTE-CSV < %t.dir/byte.mapping.csv
// RUN: FileCheck %s --check-prefix=BYTE-JSON < %t.dir/byte.mapping.json

// HARDWARE-LABEL: fabric.module @shared_vector_alu_adg
// HARDWARE-DAG: fabric.mem [spatial]
// HARDWARE-DAG: fabric.switch [spatial]
// HARDWARE-DAG: fabric.op [@arith.xori]
// HARDWARE-DAG: fabric.op [@llvm.intr.bswap]
// HARDWARE-DAG: fabric.op [@arith.mulf]
// HARDWARE-DAG: fabric.op [@arith.muli]
// HARDWARE-DAG: fabric.op [@arith.addi]
// HARDWARE-DAG: fabric.op [@dataflow.sync]

// SUMMARY: {{.*}}::shared_vector_alu_adg,fabric_module_template,10,0,pass,fabric.module template verified; link_count counts explicit fabric.link records only,mem;pe;switch,spatial,adg-builder::shared-vector-alu,

// BYTE-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// BYTE-CSV-NEXT: byte_swap,shared_vector_alu_adg,byte_swap__g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0__shared_vector_alu_adg,4,4,0,0,pass,mapped software graph to fabric resources

// BYTE-JSON-DAG: "kind": "pnr_mapping"
// BYTE-JSON-DAG: "workload": "byte_swap"
// BYTE-JSON-DAG: "hardware": "shared_vector_alu_adg"
// BYTE-JSON-DAG: "status": "pass"
// BYTE-JSON-DAG: "placed_records": 4
// BYTE-JSON-DAG: "routed_edges": 4
// BYTE-JSON-DAG: "unrouted_edges": 0
// BYTE-JSON-DAG: "segment_kind": "module_path"
// BYTE-JSON-NOT: ".out"
// BYTE-JSON-NOT: ".in"

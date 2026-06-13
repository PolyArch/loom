// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/byte_swap LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/byte_swap/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %S/shared_vector_alu_adg.mlir --hardware shared_vector_alu_adg --workload byte_swap --output %t.dir/byte.mapping.csv --artifact %t.dir/byte.mapping.json
// RUN: FileCheck %s --check-prefix=BYTE-CSV < %t.dir/byte.mapping.csv
// RUN: FileCheck %s --check-prefix=BYTE-JSON < %t.dir/byte.mapping.json

// RUN: env BUILD_DIR=%t.dir/xor_block LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/xor_block/dfg_check.sh
// RUN: loom-pnr-map --dfg-mlir %t.dir/xor_block/main_func.dfg.mlir --graph g_t_xor_block_0_0 --hardware-mlir %S/shared_vector_alu_adg.mlir --hardware shared_vector_alu_adg --workload xor_block --output %t.dir/xor.mapping.csv --artifact %t.dir/xor.mapping.json
// RUN: FileCheck %s --check-prefix=XOR-CSV < %t.dir/xor.mapping.csv
// RUN: FileCheck %s --check-prefix=XOR-JSON < %t.dir/xor.mapping.json

// BYTE-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// BYTE-CSV-NEXT: byte_swap,shared_vector_alu_adg,byte_swap__g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0__shared_vector_alu_adg,4,4,0,0,pass,mapped software graph to fabric resources

// BYTE-JSON-DAG: "kind": "pnr_mapping"
// BYTE-JSON-DAG: "workload": "byte_swap"
// BYTE-JSON-DAG: "hardware": "shared_vector_alu_adg"
// BYTE-JSON-DAG: "status": "pass"
// BYTE-JSON-DAG: "placed_records": 4
// BYTE-JSON-DAG: "routed_edges": 4
// BYTE-JSON-DAG: "unrouted_edges": 0
// BYTE-JSON-DAG: "edge_ref": "dataflow.load#0.result0->llvm.intr.bswap#0.operand0"
// BYTE-JSON-DAG: "edge_ref": "llvm.intr.bswap#0.result0->dataflow.store#0.operand2"
// BYTE-JSON-DAG: "source_endpoint": "shared_vector_alu_adg::mem.load#0.result0"
// BYTE-JSON-DAG: "sink_endpoint": "shared_vector_alu_adg::fabric.switch#0.operand0"
// BYTE-JSON-DAG: "source_endpoint": "shared_vector_alu_adg::fabric.switch#0.operand0"
// BYTE-JSON-DAG: "sink_endpoint": "shared_vector_alu_adg::fabric.switch#0.result2"
// BYTE-JSON-DAG: "source_endpoint": "shared_vector_alu_adg::fabric.switch#1.result0"
// BYTE-JSON-DAG: "sink_endpoint": "shared_vector_alu_adg::mem.store#0.operand1"
// BYTE-JSON-DAG: "segment_kind": "resource_edge"
// BYTE-JSON-DAG: "segment_kind": "module_path"
// BYTE-JSON-NOT: ".out"
// BYTE-JSON-NOT: ".in"

// XOR-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// XOR-CSV-NEXT: xor_block,shared_vector_alu_adg,xor_block__g_t_xor_block_0_0__shared_vector_alu_adg,5,6,0,0,pass,mapped software graph to fabric resources

// XOR-JSON-DAG: "kind": "pnr_mapping"
// XOR-JSON-DAG: "workload": "xor_block"
// XOR-JSON-DAG: "hardware": "shared_vector_alu_adg"
// XOR-JSON-DAG: "status": "pass"
// XOR-JSON-DAG: "placed_records": 5
// XOR-JSON-DAG: "routed_edges": 6
// XOR-JSON-DAG: "unrouted_edges": 0
// XOR-JSON-DAG: "edge_ref": "dataflow.load#0.result0->arith.xori#0.operand1"
// XOR-JSON-DAG: "edge_ref": "dataflow.load#1.result0->arith.xori#0.operand0"
// XOR-JSON-DAG: "edge_ref": "arith.xori#0.result0->dataflow.store#0.operand2"
// XOR-JSON-DAG: "source_endpoint": "shared_vector_alu_adg::mem.load#1.result0"
// XOR-JSON-DAG: "sink_endpoint": "shared_vector_alu_adg::fabric.switch#0.operand1"
// XOR-JSON-DAG: "source_endpoint": "shared_vector_alu_adg::fabric.switch#0.operand1"
// XOR-JSON-DAG: "sink_endpoint": "shared_vector_alu_adg::fabric.switch#0.result0"
// XOR-JSON-DAG: "source_endpoint": "shared_vector_alu_adg::fabric.switch#1.result0"
// XOR-JSON-DAG: "sink_endpoint": "shared_vector_alu_adg::mem.store#0.operand1"
// XOR-JSON-DAG: "segment_kind": "resource_edge"
// XOR-JSON-DAG: "segment_kind": "module_path"
// XOR-JSON-NOT: ".out"
// XOR-JSON-NOT: ".in"

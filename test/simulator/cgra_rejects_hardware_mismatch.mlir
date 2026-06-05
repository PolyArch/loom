// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"toy","status":"pass","operation_semantics_source":"loom.sim.operation_semantics.v1","optimistic_cycles":3}' > %t.dfg.json
// RUN: echo '{"schema_version":1,"kind":"pnr_mapping","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"pass","routed_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.mapping.json
// RUN: echo 'fabric.module @other_adg(%ctrl : !fabric.bits<0>) -> !fabric.bits<0> { fabric.yield %ctrl : !fabric.bits<0> }' > %t.hardware.mlir
// RUN: not loom-cgra-sim --dfg-report %t.dfg.json --mapping-artifact %t.mapping.json --hardware-mlir %t.hardware.mlir --output %t.cgra.json 2>&1 | FileCheck %s

// CHECK: hardware artifact does not contain fabric.module toy_adg

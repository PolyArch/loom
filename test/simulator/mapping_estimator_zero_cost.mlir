// RUN: echo '{"schema_version":"1.0","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"797ff30b56a30808c5465c2a6286abbf0dfb1c587611d3dabc518617cfe437e3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"a4d8534e3918343e0386d9276a5684d20f5e7d35bd47b7e43305ad1f9c76fc3a","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"pass","placed_records":0,"unplaced_records":0,"routed_edges":0,"unrouted_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.mapping.json
// RUN: loom-mapping-estimate --mapping-artifact %t.mapping.json --output %t.estimate.json
// RUN: FileCheck %s < %t.estimate.json

// CHECK-DAG: "kind": "mapping_estimate_report"
// CHECK-DAG: "workload": "toy"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "total_cost_score": 0
// CHECK-NOT: cycles
// CHECK-NOT: final_outputs
// CHECK-NOT: final_memory_state

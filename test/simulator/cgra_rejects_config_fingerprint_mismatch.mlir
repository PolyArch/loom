// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"toy","status":"pass","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","optimistic_cycles":3,"final_outputs":[],"final_memory_state":{}}' > %t.dfg.json
// RUN: echo '{"schema_version":1,"kind":"pnr_mapping","workload":"toy","hardware":"toy_adg","graph":"toy_graph","mapping_id":"toy__toy_adg","status":"pass","routed_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.missing.json
// RUN: not loom-cgra-sim --dfg-report %t.dfg.json --mapping-artifact %t.missing.json --output %t.missing.cgra.json 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: echo '{"schema_version":1,"kind":"pnr_mapping","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"pass","routed_edges":0,"config_records":0,"config_id":"loom.default","config_fingerprint":"0000000000000000000000000000000000000000000000000000000000000000","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"0000000000000000000000000000000000000000000000000000000000000000","placements":[],"routes":[],"config_bitstream":[]}' > %t.mapping.json
// RUN: not loom-cgra-sim --dfg-report %t.dfg.json --mapping-artifact %t.mapping.json --output %t.cgra.json 2>&1 | FileCheck %s

// MISSING: config_missing_required_profile
// CHECK: config_fingerprint_mismatch

// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"toy","status":"pass","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","optimistic_cycles":3}' > %t.dfg.json
// RUN: echo '{"schema_version":1,"kind":"pnr_mapping","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"pass","routed_edges":0,"placements":[],"routes":[]}' > %t.mapping.json
// RUN: not loom-cgra-sim --dfg-report %t.dfg.json --mapping-artifact %t.mapping.json --output %t.cgra.json 2>&1 | FileCheck %s

// CHECK: mapping artifact lacks config_bitstream

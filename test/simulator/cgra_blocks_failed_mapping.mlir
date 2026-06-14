// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"toy","status":"pass","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","optimistic_cycles":3,"final_outputs":[],"final_memory_state":{}}' > %t.dfg.json
// RUN: echo '{"schema_version":1,"kind":"pnr_mapping","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"fail","placed_records":2,"routed_edges":0,"unrouted_edges":1,"unplaced_records":0,"config_records":0,"placements":[{"software":"arith.addi#0","operation":"arith.addi","resource_kind":"fabric.op","hardware":"toy_adg::fabric.op#0","schedule":"spatial"},{"software":"arith.muli#0","operation":"arith.muli","resource_kind":"fabric.op","hardware":"toy_adg::fabric.op#1","schedule":"spatial"}],"routes":[],"config_bitstream":[],"diagnostics":["unrouted software edges lack Fabric ADG connectivity"]}' > %t.mapping.json
// RUN: loom-cgra-sim --dfg-report %t.dfg.json --mapping-artifact %t.mapping.json --output %t.cgra.json
// RUN: FileCheck %s < %t.cgra.json
// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"unsupported_toy","graph":"unsupported_graph","status":"unsupported","metric_definition":"optimistic_pipeline_latency_throughput_sum","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","optimistic_cycles":0,"event_count":0,"wavefront_steps":0,"dynamic_work_items":0,"operation_fire_counts":{},"final_outputs":[],"final_memory_state":{},"diagnostics":["unsupported op: scf.for"]}' > %t.unsupported.dfg.json
// RUN: echo '{"schema_version":1,"kind":"pnr_mapping","workload":"unsupported_toy","hardware":"toy_adg","mapping_id":"unsupported_toy__toy_adg","status":"unsupported","placed_records":0,"routed_edges":0,"unrouted_edges":0,"unplaced_records":0,"config_records":0,"placements":[],"routes":[],"unrouted_edge_details":[],"config_bitstream":[],"diagnostics":["unsupported PnR graph operation: scf.for"]}' > %t.unsupported.mapping.json
// RUN: loom-cgra-sim --dfg-report %t.unsupported.dfg.json --mapping-artifact %t.unsupported.mapping.json --output %t.unsupported.cgra.json
// RUN: FileCheck %s --check-prefix=UNSUPPORTED < %t.unsupported.cgra.json
// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"bad_status_toy","graph":"bad_status_graph","status":"unknown_status","metric_definition":"optimistic_pipeline_latency_throughput_sum","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","optimistic_cycles":0,"event_count":0,"wavefront_steps":0,"dynamic_work_items":0,"operation_fire_counts":{},"final_outputs":[],"final_memory_state":{},"diagnostics":["bad status fixture"]}' > %t.bad_status.dfg.json
// RUN: echo '{"schema_version":1,"kind":"pnr_mapping","workload":"bad_status_toy","hardware":"toy_adg","mapping_id":"bad_status_toy__toy_adg","status":"unsupported","placed_records":0,"routed_edges":0,"unrouted_edges":0,"unplaced_records":0,"config_records":0,"placements":[],"routes":[],"unrouted_edge_details":[],"config_bitstream":[],"diagnostics":["unsupported PnR graph operation: scf.for"]}' > %t.bad_status.mapping.json
// RUN: not loom-cgra-sim --dfg-report %t.bad_status.dfg.json --mapping-artifact %t.bad_status.mapping.json --output %t.bad_status.cgra.json 2>&1 | FileCheck %s --check-prefix=BAD-STATUS

// CHECK-DAG: "kind": "cgra_sim_report"
// CHECK-DAG: "workload": "toy"
// CHECK-DAG: "hardware": "toy_adg"
// CHECK-DAG: "mapping_id": "toy__toy_adg"
// CHECK-DAG: "status": "blocked"
// CHECK-DAG: "difference_classification": "unsupported_scope"
// CHECK-DAG: "hardware_bound_classification": "unsupported_scope"
// CHECK-DAG: "dfg_cycles": 3
// CHECK-DAG: "hardware_aware_cycles": 3
// CHECK-DAG: "routed_edges": 0
// CHECK-DAG: "route_segments": 0
// CHECK-DAG: mapping artifact status fail blocks CGRA-sim

// UNSUPPORTED-DAG: "kind": "cgra_sim_report"
// UNSUPPORTED-DAG: "workload": "unsupported_toy"
// UNSUPPORTED-DAG: "hardware": "toy_adg"
// UNSUPPORTED-DAG: "mapping_id": "unsupported_toy__toy_adg"
// UNSUPPORTED-DAG: "status": "blocked"
// UNSUPPORTED-DAG: "difference_classification": "unsupported_scope"
// UNSUPPORTED-DAG: "dfg_cycles": 0
// UNSUPPORTED-DAG: "hardware_aware_cycles": 0
// UNSUPPORTED-DAG: "routed_edges": 0
// UNSUPPORTED-DAG: DFG-sim report status unsupported blocks CGRA-sim

// BAD-STATUS: DFG report status unknown_status is not supported

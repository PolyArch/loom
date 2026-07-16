// RUN: echo '{"schema_version":"2.0","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"797ff30b56a30808c5465c2a6286abbf0dfb1c587611d3dabc518617cfe437e3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"a4d8534e3918343e0386d9276a5684d20f5e7d35bd47b7e43305ad1f9c76fc3a","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"fail","placed_records":2,"routed_edges":0,"unrouted_edges":1,"unplaced_records":0,"config_records":0,"placements":[{"software":"arith.addi#0","operation":"arith.addi","resource_kind":"fabric.op","hardware":"toy_adg::fabric.op#0","schedule":"spatial"},{"software":"arith.muli#0","operation":"arith.muli","resource_kind":"fabric.op","hardware":"toy_adg::fabric.op#1","schedule":"spatial"}],"routes":[],"config_bitstream":[],"diagnostics":["unrouted software edges lack Fabric ADG connectivity"]}' > %t.mapping.json
// RUN: loom-mapping-estimate --mapping-artifact %t.mapping.json --output %t.estimate.json
// RUN: FileCheck %s < %t.estimate.json
// RUN: not grep -q '"total_cost_score"' %t.estimate.json
// RUN: echo '{"schema_version":"2.0","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"797ff30b56a30808c5465c2a6286abbf0dfb1c587611d3dabc518617cfe437e3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"a4d8534e3918343e0386d9276a5684d20f5e7d35bd47b7e43305ad1f9c76fc3a","workload":"unsupported_toy","hardware":"toy_adg","mapping_id":"unsupported_toy__toy_adg","status":"unsupported","placed_records":0,"routed_edges":0,"unrouted_edges":0,"unplaced_records":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[],"diagnostics":["unsupported PnR graph operation: scf.for"]}' > %t.unsupported.mapping.json
// RUN: loom-mapping-estimate --mapping-artifact %t.unsupported.mapping.json --output %t.unsupported.estimate.json
// RUN: FileCheck %s --check-prefix=BLOCKED < %t.unsupported.estimate.json
// RUN: not grep -q '"total_cost_score"' %t.unsupported.estimate.json
// RUN: echo '{"schema_version":"2.0","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"797ff30b56a30808c5465c2a6286abbf0dfb1c587611d3dabc518617cfe437e3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"a4d8534e3918343e0386d9276a5684d20f5e7d35bd47b7e43305ad1f9c76fc3a","workload":"bad_status_toy","hardware":"toy_adg","mapping_id":"bad_status_toy__toy_adg","status":"unknown_status","placed_records":0,"routed_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.bad-status.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.bad-status.mapping.json --output %t.bad-status.estimate.json 2>&1 | FileCheck %s --check-prefix=BAD-STATUS
// RUN: echo '{"schema_version":1,"kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"797ff30b56a30808c5465c2a6286abbf0dfb1c587611d3dabc518617cfe437e3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"a4d8534e3918343e0386d9276a5684d20f5e7d35bd47b7e43305ad1f9c76fc3a","workload":"numeric_schema_toy","hardware":"toy_adg","mapping_id":"numeric_schema_toy__toy_adg","status":"pass","placed_records":0,"unplaced_records":0,"routed_edges":0,"unrouted_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.numeric-schema.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.numeric-schema.mapping.json --output %t.numeric-schema.estimate.json 2>&1 | FileCheck %s --check-prefix=NUMERIC-SCHEMA
// RUN: echo '{"schema_version":"1","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"797ff30b56a30808c5465c2a6286abbf0dfb1c587611d3dabc518617cfe437e3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"a4d8534e3918343e0386d9276a5684d20f5e7d35bd47b7e43305ad1f9c76fc3a","workload":"bad_schema_toy","hardware":"toy_adg","mapping_id":"bad_schema_toy__toy_adg","status":"pass","placed_records":0,"unplaced_records":0,"routed_edges":0,"unrouted_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.bad-schema.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.bad-schema.mapping.json --output %t.bad-schema.estimate.json 2>&1 | FileCheck %s --check-prefix=BAD-SCHEMA
// RUN: echo '{"schema_version":"2.0","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"797ff30b56a30808c5465c2a6286abbf0dfb1c587611d3dabc518617cfe437e3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"a4d8534e3918343e0386d9276a5684d20f5e7d35bd47b7e43305ad1f9c76fc3a","workload":"bad_config_count_toy","hardware":"toy_adg","mapping_id":"bad_config_count_toy__toy_adg","status":"fail","placed_records":0,"unplaced_records":0,"routed_edges":0,"unrouted_edges":1,"config_records":1,"placements":[],"routes":[],"config_bitstream":[]}' > %t.bad-config-count.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.bad-config-count.mapping.json --output %t.bad-config-count.estimate.json 2>&1 | FileCheck %s --check-prefix=BAD-CONFIG-COUNT

// CHECK-DAG: "kind": "mapping_estimate_report"
// CHECK-DAG: "workload": "toy"
// CHECK-DAG: "status": "blocked"
// CHECK-DAG: mapping artifact status fail prevents a complete mapping estimate

// BLOCKED-DAG: "kind": "mapping_estimate_report"
// BLOCKED-DAG: "workload": "unsupported_toy"
// BLOCKED-DAG: "status": "blocked"

// BAD-STATUS: mapping artifact status unknown_status is not supported
// NUMERIC-SCHEMA: has unsupported schema_version; expected string "2.0"
// BAD-SCHEMA: has unsupported schema_version; expected string "2.0"
// BAD-CONFIG-COUNT: mapping config_records field 1 does not match config_bitstream size 0

// RUN: echo '{"schema_version":"1.0","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"108c413d76c0e8e947c72e22cb489caa2fb22741b36ef14de2279f4173fd7ac3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"c8bccaf54d7c2425294367a4666c93f0ef54937b202958a9943656bd32cc725e","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"fail","placed_records":2,"routed_edges":0,"unrouted_edges":1,"unplaced_records":0,"config_records":0,"placements":[{"software":"arith.addi#0","operation":"arith.addi","resource_kind":"fabric.op","hardware":"toy_adg::fabric.op#0","schedule":"spatial"},{"software":"arith.muli#0","operation":"arith.muli","resource_kind":"fabric.op","hardware":"toy_adg::fabric.op#1","schedule":"spatial"}],"routes":[],"config_bitstream":[],"diagnostics":["unrouted software edges lack Fabric ADG connectivity"]}' > %t.mapping.json
// RUN: loom-mapping-estimate --mapping-artifact %t.mapping.json --output %t.estimate.json
// RUN: FileCheck %s < %t.estimate.json
// RUN: not grep -q '"total_cost_score"' %t.estimate.json
// RUN: echo '{"schema_version":"1.0","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"108c413d76c0e8e947c72e22cb489caa2fb22741b36ef14de2279f4173fd7ac3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"c8bccaf54d7c2425294367a4666c93f0ef54937b202958a9943656bd32cc725e","workload":"unsupported_toy","hardware":"toy_adg","mapping_id":"unsupported_toy__toy_adg","status":"unsupported","placed_records":0,"routed_edges":0,"unrouted_edges":0,"unplaced_records":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[],"diagnostics":["unsupported PnR graph operation: scf.for"]}' > %t.unsupported.mapping.json
// RUN: loom-mapping-estimate --mapping-artifact %t.unsupported.mapping.json --output %t.unsupported.estimate.json
// RUN: FileCheck %s --check-prefix=BLOCKED < %t.unsupported.estimate.json
// RUN: not grep -q '"total_cost_score"' %t.unsupported.estimate.json
// RUN: echo '{"schema_version":"1.0","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"108c413d76c0e8e947c72e22cb489caa2fb22741b36ef14de2279f4173fd7ac3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"c8bccaf54d7c2425294367a4666c93f0ef54937b202958a9943656bd32cc725e","workload":"bad_status_toy","hardware":"toy_adg","mapping_id":"bad_status_toy__toy_adg","status":"unknown_status","placed_records":0,"routed_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.bad-status.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.bad-status.mapping.json --output %t.bad-status.estimate.json 2>&1 | FileCheck %s --check-prefix=BAD-STATUS
// RUN: echo '{"schema_version":1,"kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"108c413d76c0e8e947c72e22cb489caa2fb22741b36ef14de2279f4173fd7ac3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"c8bccaf54d7c2425294367a4666c93f0ef54937b202958a9943656bd32cc725e","workload":"legacy_schema_toy","hardware":"toy_adg","mapping_id":"legacy_schema_toy__toy_adg","status":"pass","placed_records":0,"unplaced_records":0,"routed_edges":0,"unrouted_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.legacy-schema.mapping.json
// RUN: loom-mapping-estimate --mapping-artifact %t.legacy-schema.mapping.json --output %t.legacy-schema.estimate.json
// RUN: FileCheck %s --check-prefix=LEGACY-SCHEMA < %t.legacy-schema.estimate.json
// RUN: echo '{"schema_version":"1","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"108c413d76c0e8e947c72e22cb489caa2fb22741b36ef14de2279f4173fd7ac3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"c8bccaf54d7c2425294367a4666c93f0ef54937b202958a9943656bd32cc725e","workload":"bad_schema_toy","hardware":"toy_adg","mapping_id":"bad_schema_toy__toy_adg","status":"pass","placed_records":0,"unplaced_records":0,"routed_edges":0,"unrouted_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.bad-schema.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.bad-schema.mapping.json --output %t.bad-schema.estimate.json 2>&1 | FileCheck %s --check-prefix=BAD-SCHEMA
// RUN: echo '{"schema_version":"1.0","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"108c413d76c0e8e947c72e22cb489caa2fb22741b36ef14de2279f4173fd7ac3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"c8bccaf54d7c2425294367a4666c93f0ef54937b202958a9943656bd32cc725e","workload":"bad_config_count_toy","hardware":"toy_adg","mapping_id":"bad_config_count_toy__toy_adg","status":"fail","placed_records":0,"unplaced_records":0,"routed_edges":0,"unrouted_edges":1,"config_records":1,"placements":[],"routes":[],"config_bitstream":[]}' > %t.bad-config-count.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.bad-config-count.mapping.json --output %t.bad-config-count.estimate.json 2>&1 | FileCheck %s --check-prefix=BAD-CONFIG-COUNT

// CHECK-DAG: "kind": "mapping_estimate_report"
// CHECK-DAG: "workload": "toy"
// CHECK-DAG: "status": "blocked"
// CHECK-DAG: mapping artifact status fail prevents a complete mapping estimate

// BLOCKED-DAG: "kind": "mapping_estimate_report"
// BLOCKED-DAG: "workload": "unsupported_toy"
// BLOCKED-DAG: "status": "blocked"

// BAD-STATUS: mapping artifact status unknown_status is not supported
// LEGACY-SCHEMA-DAG: "workload": "legacy_schema_toy"
// LEGACY-SCHEMA-DAG: "status": "pass"
// BAD-SCHEMA: has unsupported schema_version; expected 1.0
// BAD-CONFIG-COUNT: mapping config_records field 1 does not match config_bitstream size 0

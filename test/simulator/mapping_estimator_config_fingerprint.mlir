// RUN: echo '{"schema_version":"2.0","kind":"pnr_mapping","workload":"toy","hardware":"toy_adg","graph":"toy_graph","mapping_id":"toy__toy_adg","status":"pass","placed_records":0,"unplaced_records":0,"routed_edges":0,"unrouted_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.missing.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.missing.json --output %t.missing.estimate.json 2>&1 | FileCheck %s --check-prefix=MISSING
// RUN: echo '{"schema_version":"2.0","kind":"pnr_mapping","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"pass","placed_records":0,"unplaced_records":0,"routed_edges":0,"unrouted_edges":0,"config_records":0,"config_id":"loom.default","config_fingerprint":"0000000000000000000000000000000000000000000000000000000000000000","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"0000000000000000000000000000000000000000000000000000000000000000","placements":[],"routes":[],"config_bitstream":[]}' > %t.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.mapping.json --output %t.estimate.json 2>&1 | FileCheck %s

// MISSING: config_missing_required_profile
// CHECK: config_fingerprint_mismatch

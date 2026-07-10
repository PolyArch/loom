// RUN: echo '{"schema_version":"1.0","kind":"pnr_mapping","config_id":"loom.default","config_fingerprint":"108c413d76c0e8e947c72e22cb489caa2fb22741b36ef14de2279f4173fd7ac3","component_config_view":"pnr.mapping.v1","component_config_fingerprint":"c8bccaf54d7c2425294367a4666c93f0ef54937b202958a9943656bd32cc725e","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"pass","placed_records":0,"unplaced_records":1,"routed_edges":0,"unrouted_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.mapping.json --output %t.estimate.json 2>&1 | FileCheck %s

// CHECK: passing mapping artifact has non-zero unplaced_records

// RUN: echo '{"schema_version":"3.0","kind":"pnr_mapping","config_id":"loom.default","resolved_config_identity":"97cdecd0746efcda044cc79d8d66263c874ca27b13bbd037d425510186e1b81d","workload":"toy","hardware":"toy_adg","mapping_id":"toy__toy_adg","status":"pass","placed_records":0,"unplaced_records":1,"routed_edges":0,"unrouted_edges":0,"config_records":0,"placements":[],"routes":[],"config_bitstream":[]}' > %t.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.mapping.json --output %t.estimate.json 2>&1 | FileCheck %s

// CHECK: passing mapping artifact has non-zero unplaced_records

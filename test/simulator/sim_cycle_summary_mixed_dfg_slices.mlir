// RUN: echo '{"kind":"dfg_sim_report","workload":"mixed","graph":"g_mixed_core","status":"blocked","optimistic_cycles":100,"diagnostics":["DFG unsupported for core slice"]}' > %t.dfg0.json
// RUN: echo '{"kind":"dfg_sim_report","workload":"mixed","graph":"g_mixed_checksum","status":"pass","optimistic_cycles":20,"diagnostics":[]}' > %t.dfg1.json
// RUN: echo '{"kind":"cgra_sim_report","workload":"mixed","status":"pass","hardware_aware_cycles":25,"diagnostics":["checksum slice"]}' > %t.cgra0.json
// RUN: loom-sim-cycle-summary --dfg-report %t.dfg0.json --dfg-report %t.dfg1.json --cgra-report %t.cgra0.json --output %t.summary.csv
// RUN: FileCheck %s < %t.summary.csv

// CHECK: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// CHECK-NEXT: mixed,,,blocked,{{.*}}DFG unsupported for core slice

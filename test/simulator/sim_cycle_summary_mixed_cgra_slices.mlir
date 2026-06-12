// RUN: echo '{"kind":"dfg_sim_report","workload":"mixed","graph":"g_mixed_core","status":"pass","optimistic_cycles":100,"diagnostics":[]}' > %t.dfg0.json
// RUN: echo '{"kind":"dfg_sim_report","workload":"mixed","graph":"g_mixed_checksum","status":"pass","optimistic_cycles":20,"diagnostics":[]}' > %t.dfg1.json
// RUN: echo '{"kind":"cgra_sim_report","workload":"mixed","status":"blocked","hardware_aware_cycles":100,"diagnostics":["mapping incomplete for core slice"]}' > %t.cgra0.json
// RUN: echo '{"kind":"cgra_sim_report","workload":"mixed","status":"pass","hardware_aware_cycles":125,"diagnostics":["checksum slice"]}' > %t.cgra1.json
// RUN: loom-sim-cycle-summary --dfg-report %t.dfg0.json --dfg-report %t.dfg1.json --cgra-report %t.cgra0.json --cgra-report %t.cgra1.json --output %t.summary.csv
// RUN: FileCheck %s < %t.summary.csv

// CHECK: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// CHECK-NEXT: mixed,120,,blocked,{{.*}}mapping incomplete for core slice

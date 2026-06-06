// RUN: echo '{"kind":"dfg_sim_report","workload":"vecadd","graph":"g_t_vecadd_0_0","status":"pass","optimistic_cycles":960,"diagnostics":[]}' > %t.dfg0.json
// RUN: echo '{"kind":"dfg_sim_report","workload":"vecadd","graph":"g_t_main_red_0_0","status":"pass","optimistic_cycles":643,"diagnostics":[]}' > %t.dfg1.json
// RUN: echo '{"kind":"cgra_sim_report","workload":"vecadd","status":"pass","hardware_aware_cycles":978,"diagnostics":["core slice"]}' > %t.cgra0.json
// RUN: echo '{"kind":"cgra_sim_report","workload":"vecadd","status":"pass","hardware_aware_cycles":653,"diagnostics":["reduction slice"]}' > %t.cgra1.json
// RUN: loom-sim-cycle-summary --dfg-report %t.dfg0.json --dfg-report %t.dfg1.json --cgra-report %t.cgra0.json --cgra-report %t.cgra1.json --output %t.summary.csv
// RUN: FileCheck %s < %t.summary.csv

// CHECK: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// CHECK-NEXT: vecadd,1603,1631,pass

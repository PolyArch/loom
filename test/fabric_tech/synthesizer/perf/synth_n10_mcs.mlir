// Perf smoke: drive `loom-synth-fu-dump` over a checked-in tier-A
// workload of 10 `dataflow.subgraph` inputs (all members of the
// `alu_int_32` hardware-share group) under the `mcs` strategy. The
// `mcs.yaml` config pins generous `timeout_sec` and `candidate_cap`
// budgets so the n=10 workload completes deterministically. As with
// the other strategy perf tests, the wallclock line is emitted for
// observability only; the test passes as long as synthesis completes
// with `reason=success` and `covered=N/N`. The synthetic input lives
// in `synth_n10.mlir.gen` (suffix `.gen` keeps lit from collecting it
// as its own test) and is regenerated deterministically from
// `gen_synth.py --n 10 --seed 42 --group alu_int_32`.

// RUN: cat %p/synth_n10.mlir.gen | loom-synth-fu-dump - --config %p/mcs.yaml --print-stats --print-ir=false --print-wallclock 2>&1 | FileCheck %s

// CHECK: synth-stat group=alu_int_32 strategy=mcs
// CHECK-SAME: reason=success
// CHECK-SAME: covered=10/10
// CHECK: wallclock_us={{[0-9]+}}

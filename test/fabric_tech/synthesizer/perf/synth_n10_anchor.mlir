// Perf smoke: drive `loom-synth-fu-dump` over a checked-in tier-A
// workload of 10 `dataflow.subgraph` inputs (all members of the
// `alu_int_32` hardware-share group) under the anchor strategy. The
// helper streams the synthetic input on stdin, the dump tool reports
// the canonical `synth-stat` line plus a `wallclock_us=` measurement,
// and FileCheck pins both. We do not gate on a wall-time threshold:
// the wallclock line is emitted purely so the perf surface is
// observable from the lit log; the test passes as long as synthesis
// completes with `reason=success` and `covered=N/N`. The synthetic
// input lives in `synth_n10.mlir.gen` (suffix `.gen` keeps lit from
// collecting it as its own test) and is regenerated deterministically
// from `gen_synth.py --n 10 --seed 42 --group alu_int_32`.

// RUN: cat %p/synth_n10.mlir.gen | loom-synth-fu-dump - --config %p/anchor.yaml --print-stats --print-ir=false --print-wallclock 2>&1 | FileCheck %s

// CHECK: synth-stat group=alu_int_32 strategy=anchor
// CHECK-SAME: reason=success
// CHECK-SAME: covered=10/10
// CHECK: wallclock_us={{[0-9]+}}

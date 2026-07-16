// RUN: not loom-synth-config-test %p/workers_overflow.yaml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=YAML-WORKERS
// RUN: not loom-synth-config-test %p/workers_overflow.toml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TOML-WORKERS
// RUN: not loom-synth-config-test %p/infinite_cost.yaml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=YAML-COST
// RUN: not loom-synth-config-test %p/infinite_cost.toml 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TOML-COST

// YAML-WORKERS: error: synth.parallelism.workers exceeds unsigned range
// TOML-WORKERS: error: synth.parallelism.workers exceeds unsigned range
// YAML-COST: error: synth.cost.{mux,demux,carry}_penalty must all be finite and >= 0
// TOML-COST: error: synth.cost.{mux,demux,carry}_penalty must all be finite and >= 0

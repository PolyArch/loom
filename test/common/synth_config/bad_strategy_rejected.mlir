// RUN: not loom-synth-config-test %p/bad_strategy_rejected.yaml 2>&1 | FileCheck %s

// CHECK: error: synth.strategy must be one of anchor|mcs|incremental|incremental_random, got 'zigzag'

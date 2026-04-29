// RUN: loom-synth-config-test %p/fallback_chain_three.yaml | FileCheck %s

// Flow-style YAML list parses into a 3-element vector in order.

// CHECK: fallback_chain.size=3
// CHECK-NEXT: fallback_chain[0]=anchor
// CHECK-NEXT: fallback_chain[1]=mcs
// CHECK-NEXT: fallback_chain[2]=incremental

// RUN: loom-synth-base-test --list-failure-reasons | FileCheck %s

// The 13 SynthFailureReason snake_case strings (plus the success-sentinel
// `none` placeholder for the `None` enumerator) printed in enum order.
// These strings are the verbatim spec wording stored in the on-IR
// `loom.synth_failed` attribute; any drift between the enum and the
// spec table must be caught by this test.

// CHECK: none
// CHECK-NEXT: cross_share_group
// CHECK-NEXT: topology_mismatch
// CHECK-NEXT: feedback_align_conflict
// CHECK-NEXT: coverage_verify_failed
// CHECK-NEXT: timeout
// CHECK-NEXT: resource_exhausted
// CHECK-NEXT: unsupported_op
// CHECK-NEXT: invalid_input
// CHECK-NEXT: verifier_failed
// CHECK-NEXT: symbol_conflict
// CHECK-NEXT: config_parse_failed
// CHECK-NEXT: no_legal_materialization

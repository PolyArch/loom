// RUN: loom-synth-base-test --synthesize-empty wibble | FileCheck %s

// Unknown strategy names fail through the canonical synthesis entrypoint.

// CHECK: result: success=false reason=invalid_input
// CHECK-NEXT: note: unknown strategy 'wibble'

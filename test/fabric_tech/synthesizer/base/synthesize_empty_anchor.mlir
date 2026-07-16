// RUN: loom-synth-base-test --synthesize-empty anchor | FileCheck %s

// Anchor rejects an empty canonical input group.

// CHECK: result: success=false reason=invalid_input
// CHECK-NEXT: note: anchor: no input functions in synth group

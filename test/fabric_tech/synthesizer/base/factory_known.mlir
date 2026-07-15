// RUN: loom-synth-base-test --make anchor | FileCheck %s

// Anchor is the canonical selectable path. It rejects an empty input group.

// CHECK: result: success=false reason=invalid_input
// CHECK-NEXT: note: anchor: no input functions in synth group

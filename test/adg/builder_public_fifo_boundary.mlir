// RUN: loom-adg-builder-test --public-fifo-boundary --output %t.hardware.mlir
// RUN: loom-adg-builder-test --public-fifo-boundary --output %t.hardware.second.mlir
// RUN: diff %t.hardware.mlir %t.hardware.second.mlir
// RUN: FileCheck %s --check-prefix=BUILDER < %t.hardware.mlir
// RUN: loom %t.hardware.mlir > /dev/null
// RUN: loom-adg-builder-test --topology-matrix-case mixed-temporal-bridge --output %t.recipe.mlir
// RUN: FileCheck %s --check-prefix=RECIPE < %t.recipe.mlir
// RUN: loom %t.recipe.mlir > /dev/null
// RUN: not loom-adg-builder-test --invalid-fifo-spec depth --output %t.depth.mlir 2>&1 | FileCheck %s --check-prefix=DEPTH
// RUN: count 0 < %t.depth.mlir
// RUN: not loom-adg-builder-test --invalid-fifo-spec overflow --output %t.overflow.mlir 2>&1 | FileCheck %s --check-prefix=OVERFLOW
// RUN: count 0 < %t.overflow.mlir
// RUN: not loom-adg-builder-test --invalid-fifo-spec kind --output %t.fifo-kind.mlir 2>&1 | FileCheck %s --check-prefix=FIFO-KIND
// RUN: count 0 < %t.fifo-kind.mlir
// RUN: not loom-adg-builder-test --invalid-fifo-spec bypass --output %t.bypass.mlir 2>&1 | FileCheck %s --check-prefix=BYPASS
// RUN: count 0 < %t.bypass.mlir
// RUN: not loom-adg-builder-test --invalid-boundary-spec shape --output %t.shape.mlir 2>&1 | FileCheck %s --check-prefix=SHAPE
// RUN: count 0 < %t.shape.mlir
// RUN: not loom-adg-builder-test --invalid-boundary-spec direction --output %t.direction.mlir 2>&1 | FileCheck %s --check-prefix=DIRECTION
// RUN: count 0 < %t.direction.mlir
// RUN: not loom-adg-builder-test --invalid-boundary-spec t2t --output %t.t2t.mlir 2>&1 | FileCheck %s --check-prefix=T2T
// RUN: count 0 < %t.t2t.mlir
// RUN: not loom-adg-builder-test --invalid-boundary-spec t2s-spatial --output %t.t2s-spatial.mlir 2>&1 | FileCheck %s --check-prefix=T2S-SPATIAL
// RUN: count 0 < %t.t2s-spatial.mlir
// RUN: not loom-adg-builder-test --invalid-boundary-spec t2s-tag-width --output %t.t2s-tag-width.mlir 2>&1 | FileCheck %s --check-prefix=T2S-TAG-WIDTH
// RUN: count 0 < %t.t2s-tag-width.mlir

// BUILDER-LABEL: fabric.module @public_fifo_boundary_adg(
// BUILDER: %tagged = fabric.boundary [s2t] %data, %tag : (!fabric.bits<32> to !fabric.bits<16>, !fabric.bits<4>) -> !fabric.bits_tag<16, 4>
// BUILDER-NEXT: %queued = fabric.fifo %tagged [max_depth = 4, bypassable = true] {bypassed = false}
// BUILDER-NEXT: : !fabric.bits_tag<16, 4> to !fabric.bits_tag<8, 4>
// BUILDER-NEXT: %untagged, %split_tag = fabric.boundary [t2s] %queued : !fabric.bits_tag<8, 4> -> (!fabric.bits<8>, !fabric.bits<4>)
// BUILDER-NEXT: fabric.yield %untagged, %split_tag : !fabric.bits<8>, !fabric.bits<4>

// RECIPE-LABEL: fabric.module @matrix_mixed_temporal_bridge_adg(
// RECIPE: %tagged = fabric.boundary [s2t] %spatial0, %tag : (!fabric.bits<32>, !fabric.bits<4>) -> !fabric.bits_tag<32, 4>
// RECIPE-NEXT: %queued = fabric.fifo %tagged [max_depth = 4, bypassable = true]
// RECIPE-NEXT: : !fabric.bits_tag<32, 4>
// RECIPE-NEXT: %untagged = fabric.boundary [t2s] %queued : !fabric.bits_tag<32, 4> -> !fabric.bits<32>

// DEPTH: ADG fifo max depth must be greater than zero
// OVERFLOW: ADG fifo max depth exceeds signed i32 range
// FIFO-KIND: operand outer type and inner type must share the same fabric kind
// BYPASS: ADG fifo bypass configuration requires bypassable hardware
// SHAPE: ADG s2t boundary requires exactly two inputs and one result
// DIRECTION: ADG boundary direction is invalid
// T2T: ADG t2t boundary construction is not supported
// T2S-SPATIAL: [t2s] operand must be a !fabric.bits_tag<BW, TW> type
// T2S-TAG-WIDTH: [t2s] result #1 bits-width 8 must equal operand tag-width 4

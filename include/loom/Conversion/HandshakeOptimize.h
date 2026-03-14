//===-- HandshakeOptimize.h - Handshake cleanup helpers ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This header declares cleanup utilities for Handshake IR, including sink
// insertion for unused values and dead code elimination.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_CONVERSION_HANDSHAKEOPTIMIZE_H
#define LOOM_CONVERSION_HANDSHAKEOPTIMIZE_H

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"

namespace loom {

mlir::LogicalResult runHandshakeCleanup(circt::handshake::FuncOp func,
                                        mlir::OpBuilder &builder);

/// Create a module-level pass that runs handshake sink insertion + dead code
/// elimination on every handshake.func. Uses fixed-point iteration so that
/// dead values exposed by canonicalize/CSE are properly cleaned up.
std::unique_ptr<mlir::Pass> createHandshakeCleanupPass();

} // namespace loom

#endif // LOOM_CONVERSION_HANDSHAKEOPTIMIZE_H

//===-- HandshakeOptimize.h - Handshake cleanup helpers ---------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This header declares cleanup utilities for Handshake IR, including sink
// insertion for unused values, dead code elimination, and fork optimization
// to remove redundant dataflow control structures.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_CONVERSION_HANDSHAKEOPTIMIZE_H
#define LOOM_CONVERSION_HANDSHAKEOPTIMIZE_H

#include "mlir/IR/Builders.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"

namespace loom {

mlir::LogicalResult runHandshakeCleanup(circt::handshake::FuncOp func,
                                        mlir::OpBuilder &builder);

} // namespace loom

#endif // LOOM_CONVERSION_HANDSHAKEOPTIMIZE_H

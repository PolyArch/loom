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

#include "circt/Dialect/Handshake/HandshakeOps.h"

namespace loom {

mlir::LogicalResult runHandshakeCleanup(circt::handshake::FuncOp func,
                                        mlir::OpBuilder &builder);

} // namespace loom

#endif // LOOM_CONVERSION_HANDSHAKEOPTIMIZE_H

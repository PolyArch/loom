//===- HandshakeOptimize.h - Handshake cleanup helpers ---------*- C++ -*-===//
//
// Part of the Loom project.
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

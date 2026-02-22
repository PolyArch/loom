//===-- ConnectivityMatrix.h - Routing connectivity ----------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Routing connectivity data structure for ADG-based path queries. Records
// physical edges (output port -> input port) and routing node internals
// (input port -> output port(s)).
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_CONNECTIVITYMATRIX_H
#define LOOM_MAPPER_CONNECTIVITYMATRIX_H

#include "loom/Mapper/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace loom {

struct ConnectivityMatrix {
  /// Physical edges: output port -> input port.
  llvm::DenseMap<IdIndex, IdIndex> outToIn;

  /// Routing node internal: input port -> reachable output port(s).
  llvm::DenseMap<IdIndex, llvm::SmallVector<IdIndex, 4>> inToOut;
};

} // namespace loom

#endif // LOOM_MAPPER_CONNECTIVITYMATRIX_H

//===-- ConfigGenUtil.h - ConfigGen shared utilities -----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_MAPPER_CONFIGGENUTIL_H
#define LOOM_MAPPER_CONFIGGENUTIL_H

#include "loom/Mapper/Graph.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <vector>

namespace loom {
namespace configgen {

/// Get the sym_name attribute from a node, or empty string.
llvm::StringRef getNodeName(const Node *node);

/// Get the op_name attribute from a node, or empty string.
llvm::StringRef getNodeOpName(const Node *node);

/// Get the resource_class attribute from a node, or empty string.
llvm::StringRef getNodeResClass(const Node *node);

/// Get an integer attribute from a node, or a default value.
int64_t getNodeIntAttr(const Node *node, llvm::StringRef name,
                       int64_t dflt = 0);

/// Check if a node has a given attribute.
bool nodeHasAttr(const Node *node, llvm::StringRef name);

/// Pack bits into a word vector, LSB-first across words.
void packBits(std::vector<uint32_t> &words, uint32_t &bitPos,
              uint64_t value, unsigned width);

/// Look up an operation by sym_name attribute, searching from the outermost
/// parent module.
mlir::Operation *lookupSymbolInModule(mlir::Operation *from,
                                      llvm::StringRef name);

} // namespace configgen
} // namespace loom

#endif // LOOM_MAPPER_CONFIGGENUTIL_H

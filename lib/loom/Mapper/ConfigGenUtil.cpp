//===-- ConfigGenUtil.cpp - ConfigGen shared utilities ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Mapper/ConfigGenUtil.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace loom {
namespace configgen {

llvm::StringRef getNodeName(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "sym_name") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

llvm::StringRef getNodeOpName(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "op_name") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

llvm::StringRef getNodeResClass(const Node *node) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == "resource_class") {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

int64_t getNodeIntAttr(const Node *node, llvm::StringRef name,
                       int64_t dflt) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr.getValue()))
        return intAttr.getInt();
    }
  }
  return dflt;
}

bool nodeHasAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name)
      return true;
  }
  return false;
}

void packBits(std::vector<uint32_t> &words, uint32_t &bitPos,
              uint64_t value, unsigned width) {
  for (unsigned b = 0; b < width; ++b) {
    unsigned wordIdx = bitPos / 32;
    unsigned bitIdx = bitPos % 32;
    if (wordIdx >= words.size())
      words.resize(wordIdx + 1, 0);
    if (value & (1ULL << b))
      words[wordIdx] |= (1U << bitIdx);
    ++bitPos;
  }
}

mlir::Operation *lookupSymbolInModule(mlir::Operation *from,
                                      llvm::StringRef name) {
  mlir::Operation *scope = from;
  while (scope->getParentOp())
    scope = scope->getParentOp();

  mlir::Operation *result = nullptr;
  scope->walk([&](mlir::Operation *op) -> mlir::WalkResult {
    if (auto attr = op->getAttrOfType<mlir::StringAttr>("sym_name")) {
      if (attr.getValue() == name) {
        result = op;
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  return result;
}

} // namespace configgen
} // namespace loom

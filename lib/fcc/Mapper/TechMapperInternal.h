#ifndef FCC_MAPPER_TECHMAPPER_INTERNAL_H
#define FCC_MAPPER_TECHMAPPER_INTERNAL_H

#include "fcc/Mapper/TechMapper.h"

#include "fcc/Dialect/Fabric/FabricOps.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <vector>

namespace fcc {
namespace techmapper_detail {

constexpr unsigned kMaxHardwareJoinFanin = 64;

enum class RefKind : uint8_t {
  Input = 0,
  OpResult = 1,
};

struct ValueRef {
  RefKind kind = RefKind::Input;
  unsigned index = 0;
  unsigned resultIndex = 0;

  bool operator==(const ValueRef &other) const {
    return kind == other.kind && index == other.index &&
           resultIndex == other.resultIndex;
  }
};

struct TemplateOp {
  unsigned bodyOpIndex = 0;
  std::string opName;
  bool commutative = false;
  llvm::SmallVector<ValueRef, 4> operands;
};

struct VariantFamily {
  std::string signature;
  std::string hwName;
  llvm::SmallVector<IdIndex, 4> hwNodeIds;
  llvm::SmallVector<mlir::Type, 4> inputTypes;
  llvm::SmallVector<mlir::Type, 4> outputTypes;
  llvm::SmallVector<TemplateOp, 4> ops;
  llvm::SmallVector<std::pair<unsigned, unsigned>, 4> edges;
  llvm::SmallVector<std::optional<ValueRef>, 4> outputs;
  llvm::SmallVector<FUConfigField, 2> configFields;
  bool configurable = false;

  bool isTechFamily() const { return !ops.empty(); }
};

struct Match {
  unsigned familyIndex = 0;
  llvm::SmallVector<IdIndex, 4> swNodesByOp;
  llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 4> operandOrderByOp;
  llvm::SmallVector<TechMapper::PortBinding, 4> inputBindings;
  llvm::SmallVector<TechMapper::PortBinding, 4> outputBindings;
  llvm::SmallVector<IdIndex, 4> internalEdges;
  llvm::SmallVector<FUConfigField, 4> configFields;
};

struct FamilyMatch {
  VariantFamily family;
  Match match;
};

struct DemandMatchStats {
  unsigned structuralStateCount = 0;
  unsigned structuralStateCacheHitCount = 0;
  unsigned structuralStateCacheMissCount = 0;
};

// Locate the ADG node for a function unit identified by PE name and FU name.
IdIndex findFunctionUnitNode(const Graph &adg, llvm::StringRef peName,
                             llvm::StringRef fuName);

// Enumerate all mux/join variant families for a single FunctionUnitOp.
void collectVariantsForFU(fcc::fabric::FunctionUnitOp fuOp,
                          const Node *hwNode,
                          llvm::SmallVectorImpl<VariantFamily> &variants);

// Find all DFG match candidates for a given variant family.
std::vector<Match> findMatchesForFamily(const Graph &dfg,
                                        const VariantFamily &family,
                                        unsigned familyIndex);

// Find all DFG match candidates for a given FunctionUnitOp without globally
// enumerating every structural variant first.
std::vector<FamilyMatch>
findDemandDrivenMatchesForFU(const Graph &dfg, fcc::fabric::FunctionUnitOp fuOp,
                             const Node *hwNode,
                             DemandMatchStats *stats = nullptr);

} // namespace techmapper_detail
} // namespace fcc

#endif // FCC_MAPPER_TECHMAPPER_INTERNAL_H

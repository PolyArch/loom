#include "Fabric/Tech/SubgraphMatcher.h"

#include "Fabric/Tech/SubgraphEnumerator.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace fabric {
namespace {

// Print an attribute to a canonical string form for value-equality compare.
static std::string canonAttr(::mlir::Attribute a) {
  std::string s;
  ::llvm::raw_string_ostream os(s);
  a.print(os);
  return s;
}

// Strip `loom.*` annotations and return a sorted vector of (key, value).
static llvm::SmallVector<std::pair<std::string, std::string>, 4>
stripLoomAttrs(::mlir::ArrayRef<::mlir::NamedAttribute> attrs) {
  llvm::SmallVector<std::pair<std::string, std::string>, 4> out;
  for (::mlir::NamedAttribute na : attrs) {
    auto key = na.getName().getValue();
    if (key.starts_with("loom."))
      continue;
    out.emplace_back(key.str(), canonAttr(na.getValue()));
  }
  llvm::sort(out);
  return out;
}

} // namespace

bool subgraphsStructurallyEqual(::dataflow::SubgraphOp a,
                                 ::dataflow::SubgraphOp b) {
  if (a.getInputs().size() != b.getInputs().size())
    return false;
  if (a.getResultTypes().size() != b.getResultTypes().size())
    return false;
  for (auto [ta, tb] :
       llvm::zip(a.getInputs().getTypes(), b.getInputs().getTypes()))
    if (ta != tb)
      return false;
  for (auto [ta, tb] : llvm::zip(a.getResultTypes(), b.getResultTypes()))
    if (ta != tb)
      return false;

  ::mlir::Block &ba = a.getBody().front();
  ::mlir::Block &bb = b.getBody().front();
  if (ba.getNumArguments() != bb.getNumArguments())
    return false;
  for (auto [aa, bb_] :
       llvm::zip(ba.getArguments(), bb.getArguments()))
    if (aa.getType() != bb_.getType())
      return false;

  // Deterministic value numbering: block args first (in order), then op
  // results in body program order. Two structurally identical subgraphs
  // have identical numberings, so operand references can be compared by
  // their numbers.
  llvm::DenseMap<::mlir::Value, unsigned> numA, numB;
  unsigned ctr = 0;
  for (auto [aa, bb_] :
       llvm::zip(ba.getArguments(), bb.getArguments())) {
    numA[aa] = ctr;
    numB[bb_] = ctr;
    ++ctr;
  }

  ::mlir::Operation *opA = ba.empty() ? nullptr : &ba.front();
  ::mlir::Operation *opB = bb.empty() ? nullptr : &bb.front();
  while (opA && opB) {
    if (opA->getName() != opB->getName())
      return false;
    if (opA->getNumOperands() != opB->getNumOperands())
      return false;
    if (opA->getNumResults() != opB->getNumResults())
      return false;
    for (auto [oa, ob] : llvm::zip(opA->getOperands(), opB->getOperands())) {
      auto na = numA.find(oa);
      auto nb = numB.find(ob);
      if (na == numA.end() || nb == numB.end())
        return false;
      if (na->second != nb->second)
        return false;
    }
    for (auto [ra, rb] : llvm::zip(opA->getResults(), opB->getResults())) {
      if (ra.getType() != rb.getType())
        return false;
      numA[ra] = ctr;
      numB[rb] = ctr;
      ++ctr;
    }
    if (stripLoomAttrs(opA->getAttrs()) != stripLoomAttrs(opB->getAttrs()))
      return false;

    opA = opA->getNextNode();
    opB = opB->getNextNode();
  }
  if (opA || opB)
    return false;
  return true;
}

FuMatchResult mapPatternToFu(::dataflow::SubgraphOp pattern, FuOp fu,
                             ::mlir::ModuleOp tempModule) {
  ::llvm::StringRef unsupported;
  auto cands = enumerateFuSubgraphs(fu, tempModule, "match_tmp", &unsupported);
  FuMatchResult r;
  for (auto &c : cands) {
    if (subgraphsStructurallyEqual(pattern, c.subgraph)) {
      r.matched = true;
      r.fu = fu;
      r.configDescription = c.configDescription;
      r.swConfigsByOp = std::move(c.swConfigsByOp);
      break;
    }
  }
  return r;
}

} // namespace fabric

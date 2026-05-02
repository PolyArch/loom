#include "Fabric/Tech/SubgraphGraphView.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace loom::fabric::tech::detail {

std::string typeKey(::mlir::Type t) {
  std::string s;
  ::llvm::raw_string_ostream os(s);
  t.print(os);
  return s;
}

std::string canonAttr(::mlir::Attribute a) {
  std::string s;
  ::llvm::raw_string_ostream os(s);
  a.print(os);
  return s;
}

::llvm::SmallVector<std::pair<std::string, std::string>, 4>
stripLoomAttrs(::mlir::ArrayRef<::mlir::NamedAttribute> attrs) {
  ::llvm::SmallVector<std::pair<std::string, std::string>, 4> out;
  for (::mlir::NamedAttribute na : attrs) {
    auto key = na.getName().getValue();
    if (key.starts_with("loom."))
      continue;
    out.emplace_back(key.str(), canonAttr(na.getValue()));
  }
  ::llvm::sort(out);
  return out;
}

bool buildGraphView(::dataflow::SubgraphOp sg, GraphView &gv) {
  gv.sg = sg;
  gv.body = &sg.getBody().front();
  gv.numBlockArgs = gv.body->getNumArguments();
  gv.blockArgTypeKeys.reserve(gv.numBlockArgs);
  for (unsigned i = 0; i < gv.numBlockArgs; ++i) {
    auto a = gv.body->getArgument(i);
    Source s{Source::BlockArg, i, 0};
    gv.valueSource[a] = s;
    gv.blockArgTypeKeys.push_back(typeKey(a.getType()));
  }

  // Index body ops first so we can resolve same-block back / forward
  // edges without depending on textual operand order.
  ::llvm::DenseMap<::mlir::Operation *, unsigned> opIdx;
  unsigned i = 0;
  for (::mlir::Operation &op : gv.body->without_terminator()) {
    opIdx[&op] = i;
    for (auto [r, res] : ::llvm::enumerate(op.getResults())) {
      Source s{Source::BodyOp, i, static_cast<unsigned>(r)};
      gv.valueSource[res] = s;
    }
    ++i;
  }
  gv.nodes.resize(i);

  i = 0;
  for (::mlir::Operation &op : gv.body->without_terminator()) {
    NodeInfo &ni = gv.nodes[i];
    ni.op = &op;
    ni.opName = op.getName().getStringRef();
    ni.numOperands = op.getNumOperands();
    ni.numResults = op.getNumResults();
    ni.operands.reserve(ni.numOperands);
    for (::mlir::Value v : op.getOperands()) {
      auto it = gv.valueSource.find(v);
      if (it == gv.valueSource.end())
        return false;
      ni.operands.push_back(it->second);
    }
    ni.resultTypeKeys.reserve(ni.numResults);
    for (::mlir::Type t : op.getResultTypes())
      ni.resultTypeKeys.push_back(typeKey(t));
    ni.sortedOperandTypeKeys.reserve(ni.numOperands);
    for (::mlir::Type t : op.getOperandTypes())
      ni.sortedOperandTypeKeys.push_back(typeKey(t));
    ::llvm::sort(ni.sortedOperandTypeKeys);
    ni.attrKeys = stripLoomAttrs(op.getAttrs());
    ++i;
  }

  // Yield operands (terminator).
  ::mlir::Operation *term = gv.body->getTerminator();
  if (term) {
    for (::mlir::Value v : term->getOperands()) {
      auto it = gv.valueSource.find(v);
      if (it == gv.valueSource.end())
        return false;
      gv.yieldSources.push_back(it->second);
    }
  }
  return true;
}

bool isCommutativeOp(::llvm::StringRef name) {
  // Listed once in canonical order so the runtime has a tight static
  // set. Commentary on the choice (comparison ops) lives at the
  // matcher's call site to avoid drift.
  return name == "arith.addi" || name == "arith.muli" ||
         name == "arith.andi" || name == "arith.ori" || name == "arith.xori" ||
         name == "arith.addf" || name == "arith.mulf" ||
         name == "arith.minsi" || name == "arith.maxsi" ||
         name == "arith.minui" || name == "arith.maxui" ||
         name == "arith.minimumf" || name == "arith.maximumf";
}

} // namespace loom::fabric::tech::detail

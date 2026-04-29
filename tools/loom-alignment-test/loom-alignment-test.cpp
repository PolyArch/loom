// CLI helper for lit tests: drives the loom::fabric::tech Alignment facade
// over each `dataflow.subgraph` in the input MLIR module.
//
// Usage:
//   loom-alignment-test <input.mlir>
//
// For every `func.func` in module order that contains exactly one
// `dataflow.subgraph` op, prints (one block per func, lines separated
// by newlines):
//
//   func @<funcname>:
//     yield-anchors=<N>
//     anchor[i]=<kind>:<details>
//     signature[i]=<op>;<sg>;bw=<n>;arity=<n>;ohash=0x<hex>
//     backedges=<count>
//
// `<kind>` is one of `BodyOp`, `BlockArg`, `BackEdge`.
// `<details>` is `<op-name>#<resultIdx>` for BodyOp / BackEdge, and
// `#<argIdx>` for BlockArg.
// `<sg>` is the share-group integer index from
// `loom::common::findShareGroup` or "-" for singletons.
// `<op>` is the producing op's StringRef (empty string for a BlockArg
// signature; printed as `-`).
// `0x<hex>` is the low 32 bits of the structuralHash, zero-padded to
// 8 nibbles. Truncating to 32 bits keeps lit assertions robust to host
// pointer-width hash drift while still tying the spec's structural
// hash to a single recorded value.
//
// Funcs without exactly one `dataflow.subgraph` are skipped silently
// (lit tests pin the func count themselves).

#include "Common/HwShareGroup.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/Tech/Synthesizer/Alignment.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

static ::llvm::cl::opt<std::string>
    inputPath(::llvm::cl::Positional, ::llvm::cl::desc("<input>"),
              ::llvm::cl::Required);

namespace {

::llvm::StringRef kindName(::loom::fabric::tech::Source::Kind k) {
  switch (k) {
  case ::loom::fabric::tech::Source::BodyOp:
    return "BodyOp";
  case ::loom::fabric::tech::Source::BlockArg:
    return "BlockArg";
  case ::loom::fabric::tech::Source::BackEdge:
    return "BackEdge";
  }
  return "Unknown";
}

void printAnchor(::llvm::raw_ostream &os,
                 const ::loom::fabric::tech::Source &s) {
  os << kindName(s.kind) << ":";
  switch (s.kind) {
  case ::loom::fabric::tech::Source::BlockArg:
    os << "#" << s.argIndex;
    break;
  case ::loom::fabric::tech::Source::BodyOp:
  case ::loom::fabric::tech::Source::BackEdge: {
    ::llvm::StringRef opName =
        s.op ? s.op->getName().getStringRef() : ::llvm::StringRef("<null>");
    os << opName << "#" << s.resultIndex;
    break;
  }
  }
}

void printSignature(::llvm::raw_ostream &os,
                    const ::loom::fabric::tech::NodeSignature &ns) {
  if (ns.op.empty())
    os << "-";
  else
    os << ns.op;
  os << ";";
  if (ns.shareGroup.has_value())
    os << *ns.shareGroup;
  else
    os << "-";
  os << ";bw=" << ns.bitwidth << ";arity=" << ns.arity;
  uint32_t low = static_cast<uint32_t>(ns.structuralHash & 0xFFFFFFFFu);
  os << ";ohash=0x" << ::llvm::format_hex_no_prefix(low, 8);
}

void emitFunc(::llvm::raw_ostream &os, ::mlir::func::FuncOp f) {
  ::llvm::SmallVector<::dataflow::SubgraphOp, 1> sgs;
  f.walk([&](::dataflow::SubgraphOp sg) { sgs.push_back(sg); });
  if (sgs.size() != 1)
    return;
  ::dataflow::SubgraphOp sg = sgs.front();
  os << "func @" << f.getSymName() << ":\n";
  auto anchors = ::loom::fabric::tech::yieldAnchors(sg);
  os << "  yield-anchors=" << anchors.size() << "\n";
  for (size_t i = 0; i < anchors.size(); ++i) {
    os << "  anchor[" << i << "]=";
    printAnchor(os, anchors[i]);
    os << "\n";
    os << "  signature[" << i << "]=";
    auto sig = ::loom::fabric::tech::signatureOf(anchors[i]);
    printSignature(os, sig);
    os << "\n";
  }
  auto bes = ::loom::fabric::tech::backEdges(sg);
  os << "  backedges=" << bes.size() << "\n";
}

} // namespace

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "loom-alignment-test: dump Alignment facade outputs per dataflow.subgraph\n");

  ::mlir::DialectRegistry registry;
  ::mlir::registerAllDialects(registry);
  registry.insert<::fabric::FabricDialect, ::dataflow::DataflowDialect>();
  ::mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto bufOrErr = ::llvm::MemoryBuffer::getFileOrSTDIN(inputPath.getValue());
  if (auto ec = bufOrErr.getError()) {
    ::llvm::errs() << "error: " << ec.message() << "\n";
    return 1;
  }
  ::llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(std::move(*bufOrErr), ::llvm::SMLoc());
  ::mlir::OwningOpRef<::mlir::ModuleOp> mod =
      ::mlir::parseSourceFile<::mlir::ModuleOp>(sm, &ctx);
  if (!mod) {
    ::llvm::errs() << "error: parse failed\n";
    return 1;
  }
  for (::mlir::func::FuncOp f : mod->getOps<::mlir::func::FuncOp>())
    emitFunc(::llvm::outs(), f);
  return 0;
}

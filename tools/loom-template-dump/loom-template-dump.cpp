// CLI helper for lit tests: parse an MLIR file containing one or more
// fabric.fu ops and dump the resulting TemplateLibrary in a stable form
// suitable for FileCheck.
//
// Usage: loom-template-dump <input.mlir>
//
// Output format (one entry per line):
//   tpl#<id> root=<op-name> size=<bodyOps> cfg=<configDescription>

#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Tech/TemplateLibrary.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

static ::llvm::cl::opt<std::string>
    inputPath(::llvm::cl::Positional, ::llvm::cl::desc("<input>"),
              ::llvm::cl::Required);

int main(int argc, char **argv) {
  ::llvm::cl::ParseCommandLineOptions(argc, argv,
                                       "loom-template-dump: parse MLIR file "
                                       "and print TemplateLibrary entries\n");

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

  ::llvm::SmallVector<::fabric::FuOp> fus;
  mod->walk([&](::fabric::FuOp fu) { fus.push_back(fu); });

  auto lib = ::fabric::TemplateLibrary::build(&ctx, fus);
  for (const auto &t : lib->templates()) {
    ::llvm::outs() << "tpl#" << t.id << " root=" << t.rootOpName
                    << " size=" << t.bodyOpCount
                    << " cfg=" << t.configDescription << "\n";
  }
  return 0;
}

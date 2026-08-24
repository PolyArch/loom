#include "ADG/Export.h"

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/Visualization/FabricVisualization.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"

#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::adg {
namespace {

llvm::Error exportError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "adg_export_io: " + message);
}

template <typename Writer>
llvm::Expected<llvm::sys::fs::TempFile>
writeTemporary(llvm::StringRef outputBase, llvm::StringRef suffix,
               Writer &&writer) {
  llvm::SmallString<256> model(outputBase);
  model.append(suffix);
  model.append(".tmp-%%%%%%");
  auto temporary = llvm::sys::fs::TempFile::create(
      model, llvm::sys::fs::owner_read | llvm::sys::fs::owner_write);
  if (!temporary)
    return exportError("unable to create temporary output: " +
                       llvm::toString(temporary.takeError()));

  {
    llvm::raw_fd_ostream output(temporary->FD, false);
    if (llvm::Error error = writer(output)) {
      llvm::consumeError(temporary->discard());
      return std::move(error);
    }
    output.flush();
    if (std::error_code error = output.error()) {
      output.clear_error();
      llvm::consumeError(temporary->discard());
      return exportError("unable to write temporary output: " +
                         error.message());
    }
  }
  return std::move(*temporary);
}

} // namespace

llvm::Error
writeFabricAuthoringMlir(const loom::fabric::FinalizedFabricRoot &root,
                         const loom::ArtifactStore &store,
                         llvm::raw_ostream &output) {
  if (root.view().rootKind() != loom::fabric::FabricRootKind::System)
    return exportError("authoring closure requires a System root");

  std::vector<loom::fabric::FinalizedFabricRoot> dependencies;
  for (const loom::fabric::FabricDirectDependency &dependency :
       root.directDependencies()) {
    if (dependency.role != loom::fabric::FabricDependencyRole::ImportedModule)
      continue;
    auto imported =
        loom::fabric::importEntireFabricRoot(dependency.root, store);
    if (!imported)
      return exportError("cannot import authoring Module dependency: " +
                         llvm::toString(imported.takeError()));
    dependencies.push_back(std::move(*imported));
  }

  auto printRoot = [](const loom::fabric::FinalizedFabricRoot &value) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    if (llvm::Error error = loom::fabric::writeFabricMlir(value, stream))
      return llvm::Expected<std::string>(std::move(error));
    stream.flush();
    return llvm::Expected<std::string>(std::move(text));
  };
  auto rootText = printRoot(root);
  if (!rootText)
    return rootText.takeError();

  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto authoring = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto append = [&](llvm::StringRef text,
                    llvm::StringRef symbol) -> llvm::Error {
    auto parsed = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!parsed)
      return exportError("cannot parse canonical Fabric MLIR projection");
    for (mlir::Operation &operation : parsed->getBody()->getOperations()) {
      mlir::Operation *clone = operation.clone();
      clone->setAttr(mlir::SymbolTable::getSymbolAttrName(),
                     mlir::StringAttr::get(&context, symbol));
      authoring.getBody()->push_back(clone);
    }
    return llvm::Error::success();
  };
  for (auto indexed : llvm::enumerate(dependencies)) {
    auto text = printRoot(indexed.value());
    if (!text)
      return text.takeError();
    const std::string symbol =
        "__loom_imported_module_" + std::to_string(indexed.index());
    if (llvm::Error error = append(*text, symbol))
      return error;
  }
  if (llvm::Error error = append(*rootText, "__loom_imported_system"))
    return error;
  authoring.print(output);
  output << '\n';
  return llvm::Error::success();
}

llvm::Error exportFabricDesign(const loom::fabric::FinalizedFabricRoot &root,
                               const loom::ArtifactStore &store,
                               llvm::StringRef outputBase) {
  if (outputBase.empty())
    return exportError("output base path is empty");

  auto mlir =
      writeTemporary(outputBase, ".mlir", [&](llvm::raw_ostream &output) {
        return loom::fabric::writeFabricMlir(root, output);
      });
  if (!mlir)
    return mlir.takeError();
  llvm::scope_exit discardMlir([&] { llvm::consumeError(mlir->discard()); });

  auto html =
      writeTemporary(outputBase, ".html", [&](llvm::raw_ostream &output) {
        return loom::fabric::writeFabricVisualizationHtml(root, store, output);
      });
  if (!html)
    return html.takeError();
  llvm::scope_exit discardHtml([&] { llvm::consumeError(html->discard()); });

  llvm::SmallString<256> mlirPath(outputBase);
  mlirPath.append(".mlir");
  llvm::SmallString<256> htmlPath(outputBase);
  htmlPath.append(".html");

  if (llvm::Error error = mlir->keep(mlirPath))
    return exportError("unable to publish Fabric MLIR: " +
                       llvm::toString(std::move(error)));
  discardMlir.release();
  if (llvm::Error error = html->keep(htmlPath)) {
    std::error_code cleanup = llvm::sys::fs::remove(mlirPath);
    if (cleanup)
      return exportError("unable to publish Fabric HTML and remove the paired "
                         "MLIR output: " +
                         llvm::toString(std::move(error)) + "; " +
                         cleanup.message());
    return exportError("unable to publish Fabric HTML: " +
                       llvm::toString(std::move(error)));
  }
  discardHtml.release();
  return llvm::Error::success();
}

} // namespace loom::adg

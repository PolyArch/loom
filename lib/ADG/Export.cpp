#include "ADG/Export.h"

#include "Fabric/Visualization/FabricVisualization.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>
#include <utility>

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

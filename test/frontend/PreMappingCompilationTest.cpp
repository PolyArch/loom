#include "Frontend/Compilation/PreMappingCompilation.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <system_error>

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

std::unique_ptr<llvm::Module> parseModule(const char *test,
                                          llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
define i32 @main(i32 %value) {
entry:
  %sum = add i32 %value, %value
  ret i32 %sum
}
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<pre-mapping>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  return module;
}

void exactFabricAndWholeProgramDataflow() {
  const char *test = "exactFabricAndWholeProgramDataflow";
  llvm::SmallString<128> directory;
  std::error_code error =
      llvm::sys::fs::createUniqueDirectory("loom-pre-mapping", directory);
  if (error)
    fail(test, "cannot create artifact store directory: " + error.message());
  loom::ArtifactStore store(directory);
  auto design = take(test, loom::adg::buildBuiltinTarget(
                               store, loom::adg::BuiltinTargetPreset::Small));
  if (design.roots().size() != 1)
    fail(test, "builtin target did not publish one System Fabric root");
  llvm::LLVMContext context;
  auto compiled = take(test, loom::frontend::compileLlvmModuleToPreMapping(
                                 parseModule(test, context),
                                 design.roots().front().reference(), store));
  if (compiled.fabric != design.roots().front().reference())
    fail(test, "pre-Mapping result lost exact Fabric target identity");
  if (!compiled.canonicalDataflow.module().lookupSymbol("main"))
    fail(test, "whole-program Dataflow artifact lost LLVM callable envelope");
  auto view = take(test, compiled.canonicalDataflow.view());
  if (!view.graphs().empty())
    fail(test, "mechanical compilation invented a SpatialCore graph");
  auto published =
      take(test, loom::frontend::publishPreMappingCompilation(compiled, store));
  if (published.fabric != design.roots().front().reference())
    fail(test, "published compilation changed its exact Fabric binding");
  auto importedStructured = take(test, loom::frontend::importStructuredProgram(
                                           published.structuredProgram, store));
  auto importedDataflow = take(test, dataflow::importCanonicalDataflow(
                                         published.canonicalDataflow, store));
  if (importedStructured.identity() != compiled.structuredProgram.identity() ||
      importedDataflow.identity() != compiled.canonicalDataflow.identity())
    fail(test, "published artifacts did not round-trip through their owners");
  std::error_code cleanup = llvm::sys::fs::remove_directories(directory);
  if (cleanup)
    fail(test, "cannot remove artifact store directory: " + cleanup.message());
}

} // namespace

int main() {
  exactFabricAndWholeProgramDataflow();
  llvm::outs() << "pre-Mapping compilation anchor passed\n";
  return EXIT_SUCCESS;
}

#include "Frontend/Compilation/PreMappingCompilation.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"

namespace loom::frontend {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "pre_mapping_compilation_invalid: " + message);
}

} // namespace

llvm::Expected<PreMappingCompilation>
compileLlvmModuleToPreMapping(std::unique_ptr<llvm::Module> module,
                              const ::loom::fabric::FinalizedFabricRoot &fabric,
                              const PreMappingCompilationOptions &options) {
  if (!module)
    return invalid("LLVM module is required");
  if (fabric.reference().schemaIdentity.empty())
    return invalid("finalized Fabric target has no artifact reference");
  auto staticGlobalMemory = projectStaticGlobalMemory(*module);
  if (!staticGlobalMemory)
    return staticGlobalMemory.takeError();
  auto structured = raising::raiseLlvmModuleToStructuredProgram(
      std::move(module), options.raising);
  if (!structured)
    return structured.takeError();
  auto dataflow = lowering::lowerStructuredProgramToCanonicalDataflow(
      *structured, options.lowering);
  if (!dataflow)
    return dataflow.takeError();
  return PreMappingCompilation{fabric.reference(),
                               std::move(*staticGlobalMemory),
                               std::move(*structured), std::move(*dataflow)};
}

llvm::Expected<PreMappingCompilation> compileLlvmModuleToPreMapping(
    std::unique_ptr<llvm::Module> module, const ArtifactRootReference &fabric,
    const ArtifactStore &store, const PreMappingCompilationOptions &options) {
  if (fabric.schemaIdentity.empty())
    return invalid("Fabric artifact reference is required");
  auto imported = ::loom::fabric::importEntireFabricRoot(fabric, store);
  if (!imported)
    return imported.takeError();
  return compileLlvmModuleToPreMapping(std::move(module), *imported, options);
}

llvm::Expected<PublishedPreMappingCompilation>
publishPreMappingCompilation(const PreMappingCompilation &compilation,
                             const ArtifactStore &store) {
  if (compilation.fabric.schemaIdentity.empty())
    return invalid("pre-Mapping compilation has no Fabric binding");
  auto structured =
      publishStructuredProgram(compilation.structuredProgram, store);
  if (!structured)
    return structured.takeError();
  auto dataflow =
      publishCanonicalDataflow(compilation.canonicalDataflow, store);
  if (!dataflow)
    return dataflow.takeError();
  return PublishedPreMappingCompilation{
      compilation.fabric, std::move(*structured), std::move(*dataflow)};
}

} // namespace loom::frontend

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

llvm::Expected<PreMappingCompilation> lowerStructuredCompilationToPreMapping(
    StructuredCompilation compilation,
    const lowering::CanonicalDataflowLoweringOptions &options) {
  auto dataflow = lowering::lowerStructuredProgramToCanonicalDataflow(
      compilation.structuredProgram, options);
  if (!dataflow)
    return dataflow.takeError();
  return PreMappingCompilation{std::move(compilation.fabric),
                               std::move(compilation.staticGlobalMemory),
                               std::move(compilation.structuredProgram),
                               std::move(compilation.sourceProvenance),
                               std::move(compilation.candidateHints),
                               std::move(*dataflow)};
}

llvm::Expected<StructuredCompilation>
raiseLlvmModuleToStructured(std::unique_ptr<llvm::Module> module,
                            const ::loom::fabric::FinalizedFabricRoot &fabric,
                            const raising::StructuredRaisingOptions &options) {
  if (!module)
    return invalid("LLVM module is required");
  if (fabric.reference().schemaIdentity.empty())
    return invalid("finalized Fabric target has no artifact reference");
  auto staticGlobalMemory = projectStaticGlobalMemory(*module);
  if (!staticGlobalMemory)
    return staticGlobalMemory.takeError();
  auto structured = raising::raiseLlvmModuleToStructuredProgramWithProjection(
      std::move(module), options);
  if (!structured)
    return structured.takeError();
  return StructuredCompilation{
      fabric.reference(), std::move(*staticGlobalMemory),
      std::move(structured->artifact), std::move(structured->sourceProvenance),
      std::move(structured->candidateHints)};
}

llvm::Expected<StructuredCompilation>
raiseLlvmModuleToStructured(std::unique_ptr<llvm::Module> module,
                            const ArtifactRootReference &fabric,
                            const ArtifactStore &store,
                            const raising::StructuredRaisingOptions &options) {
  if (fabric.schemaIdentity.empty())
    return invalid("Fabric artifact reference is required");
  auto imported = ::loom::fabric::importEntireFabricRoot(fabric, store);
  if (!imported)
    return imported.takeError();
  return raiseLlvmModuleToStructured(std::move(module), *imported, options);
}

llvm::Expected<PreMappingCompilation>
compileLlvmModuleToPreMapping(std::unique_ptr<llvm::Module> module,
                              const ::loom::fabric::FinalizedFabricRoot &fabric,
                              const PreMappingCompilationOptions &options) {
  auto structured =
      raiseLlvmModuleToStructured(std::move(module), fabric, options.raising);
  if (!structured)
    return structured.takeError();
  return lowerStructuredCompilationToPreMapping(std::move(*structured),
                                                options.lowering);
}

llvm::Expected<PreMappingCompilation> compileLlvmModuleToPreMapping(
    std::unique_ptr<llvm::Module> module, const ArtifactRootReference &fabric,
    const ArtifactStore &store, const PreMappingCompilationOptions &options) {
  auto structured = raiseLlvmModuleToStructured(std::move(module), fabric,
                                                store, options.raising);
  if (!structured)
    return structured.takeError();
  return lowerStructuredCompilationToPreMapping(std::move(*structured),
                                                options.lowering);
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

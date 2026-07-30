#ifndef LOOM_FRONTEND_COMPILATION_PREMAPPINGCOMPILATION_H
#define LOOM_FRONTEND_COMPILATION_PREMAPPINGCOMPILATION_H

#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/StaticGlobalMemory.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/CanonicalDataflowLowering.h"
#include "Frontend/Raising/StructuredRaising.h"

#include "llvm/Support/Error.h"

#include <memory>

namespace llvm {
class Module;
} // namespace llvm

namespace loom::frontend {

/// The exact result at the first mechanical compiler boundary. Fabric remains
/// an invocation binding and static globals remain an ephemeral LLVM-owned
/// projection; neither is copied into the Structured Program Artifact.
struct StructuredCompilation final {
  ArtifactRootReference fabric;
  StaticGlobalMemoryCatalog staticGlobalMemory;
  StructuredProgramCandidate structuredProgram;
  std::vector<StructuredOperationSourceProvenance> sourceProvenance;
};

/// The exact non-Mapping result of one front-end invocation. Fabric is an
/// invocation input used for subsequent capability and Evaluation decisions;
/// it is intentionally not embedded in either software artifact.
struct PreMappingCompilation final {
  ArtifactRootReference fabric;
  StaticGlobalMemoryCatalog staticGlobalMemory;
  StructuredProgramCandidate structuredProgram;
  std::vector<StructuredOperationSourceProvenance> sourceProvenance;
  dataflow::CanonicalDataflowArtifact canonicalDataflow;
};

struct PreMappingCompilationOptions final {
  raising::StructuredRaisingOptions raising;
  lowering::CanonicalDataflowLoweringOptions lowering;
};

/// Persistent projections of a completed pre-Mapping invocation. This is not
/// an Artifact family; each reference remains owned by its existing family.
struct PublishedPreMappingCompilation final {
  ArtifactRootReference fabric;
  ArtifactRootReference structuredProgram;
  ArtifactRootReference canonicalDataflow;
};

llvm::Expected<StructuredCompilation> raiseLlvmModuleToStructured(
    std::unique_ptr<llvm::Module> module,
    const ::loom::fabric::FinalizedFabricRoot &fabric,
    const raising::StructuredRaisingOptions &options = {});

llvm::Expected<StructuredCompilation> raiseLlvmModuleToStructured(
    std::unique_ptr<llvm::Module> module, const ArtifactRootReference &fabric,
    const ArtifactStore &store,
    const raising::StructuredRaisingOptions &options = {});

llvm::Expected<PreMappingCompilation> lowerStructuredCompilationToPreMapping(
    StructuredCompilation compilation,
    const lowering::CanonicalDataflowLoweringOptions &options = {});

/// Runs the mechanical LLVM-to-Structured and Structured-to-Dataflow
/// boundaries against one already-finalized exact Fabric target. Structured
/// optimization, candidate generation, Evaluation, and Mapping remain outside
/// this function; callers must materialize selected decisions before invoking
/// the Dataflow lowerer.
llvm::Expected<PreMappingCompilation>
compileLlvmModuleToPreMapping(std::unique_ptr<llvm::Module> module,
                              const ::loom::fabric::FinalizedFabricRoot &fabric,
                              const PreMappingCompilationOptions &options = {});

/// Resolves the invocation's exact published Fabric through the Fabric owner
/// before running the mechanical frontend boundary.
llvm::Expected<PreMappingCompilation>
compileLlvmModuleToPreMapping(std::unique_ptr<llvm::Module> module,
                              const ArtifactRootReference &fabric,
                              const ArtifactStore &store,
                              const PreMappingCompilationOptions &options = {});

/// Publishes the existing Structured Program and Canonical Dataflow Artifacts
/// through their family owners. Fabric was already published by its owner and
/// is returned only as the exact invocation binding.
llvm::Expected<PublishedPreMappingCompilation>
publishPreMappingCompilation(const PreMappingCompilation &compilation,
                             const ArtifactStore &store);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_COMPILATION_PREMAPPINGCOMPILATION_H

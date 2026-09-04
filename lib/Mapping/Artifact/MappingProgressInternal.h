#ifndef LOOM_LIB_MAPPING_ARTIFACT_MAPPINGPROGRESSINTERNAL_H
#define LOOM_LIB_MAPPING_ARTIFACT_MAPPINGPROGRESSINTERNAL_H

#include "Mapping/Artifact/MappingProgressAnalysis.h"

namespace loom::mapping::progress_detail {

llvm::Error invalid(const llvm::Twine &message);

void appendU64(std::string &bytes, std::uint64_t value);

/// Canonical derived keys shared by the progress projections and kernel.
std::string staticWaitNodeKey(const MappingStaticWaitNode &node);
llvm::Expected<std::string> eventKey(const ArtifactIdentity &dataflowIdentity,
                                     const ::dataflow::EventFamilyKey &event);

std::vector<std::uint32_t>
findDirectedCycle(llvm::ArrayRef<std::vector<std::uint32_t>> edges);
llvm::Expected<std::vector<std::uint32_t>>
initializedFeedbackInputOrdinals(const ::dataflow::CanonicalActorView &actor);
bool isBuffered(
    const std::optional<::loom::fabric::FabricPhysicalTraversalRef> &traversal);

} // namespace loom::mapping::progress_detail

#endif // LOOM_LIB_MAPPING_ARTIFACT_MAPPINGPROGRESSINTERNAL_H

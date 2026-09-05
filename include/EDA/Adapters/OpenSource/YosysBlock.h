#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_YOSYSBLOCK_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_YOSYSBLOCK_H

#include "EDA/Adapters/OpenSource/YosysGateNetlist.h"
#include "Hardware/RTL/BlockGateNetlist.h"

namespace loom::eda::open_source {

inline constexpr dse::CandidateGeneratorKind
    yosysBlockGateNetlistCandidateGeneratorKind(0x59535942);
inline constexpr dse::CandidateGeneratorKind
    yosysHierarchicalBlockGateNetlistCandidateGeneratorKind(0x59535950);

const dse::CandidateGeneratorDescriptor &
yosysBlockGateNetlistCandidateGeneratorDescriptor();
llvm::Error registerYosysBlockGateNetlistCandidateGenerator();
llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindYosysBlockGateNetlistInputs(
    const ArtifactRootReference &blockSource,
    const ArtifactRootReference &implementationPlatform);
llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveYosysBlockGateNetlistBinding(
    const ResolvedYosysGateNetlistConfigView &config);

/// Synthesizes the parent-local logic against one immutable mapped product
/// per distinct direct child in the exact reusable source graph.
const dse::CandidateGeneratorDescriptor &
yosysHierarchicalBlockGateNetlistCandidateGeneratorDescriptor();
llvm::Error registerYosysHierarchicalBlockGateNetlistCandidateGenerator();
llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindYosysHierarchicalBlockGateNetlistInputs(
    const ArtifactRootReference &blockSource,
    const ArtifactRootReference &implementationPlatform,
    llvm::ArrayRef<ArtifactRootReference> compiledChildren);
llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveYosysHierarchicalBlockGateNetlistBinding(
    const ResolvedYosysGateNetlistConfigView &config);

/// Associates the exact portable RTL root with its complete Yosys block
/// product.
inline constexpr dse::CandidateGeneratorKind
    yosysPortableGateImplementationCandidateGeneratorKind(0x59535949);
const dse::CandidateGeneratorDescriptor &
yosysPortableGateImplementationCandidateGeneratorDescriptor();
llvm::Error registerYosysPortableGateImplementationCandidateGenerator();
llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindYosysPortableGateImplementationInputs(
    const ArtifactRootReference &implementation,
    const ArtifactRootReference &blockNetlist);
llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveYosysPortableGateImplementationBinding();

/// Includes the vendor-owned mapped library contract validation in addition
/// to the hardware-owned source/technology/interface closure importer.
llvm::Expected<hardware::rtl::FinalizedBlockGateNetlist>
importYosysBlockGateNetlist(const ArtifactRootReference &reference,
                            const ArtifactStore &artifacts,
                            const BlobStore &blobs);

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_YOSYSBLOCK_H

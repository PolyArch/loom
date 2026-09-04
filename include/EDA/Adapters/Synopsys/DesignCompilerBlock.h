#ifndef LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILERBLOCK_H
#define LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILERBLOCK_H

#include "EDA/Adapters/Synopsys/DesignCompiler.h"
#include "Hardware/RTL/BlockGateNetlist.h"

namespace loom::eda::synopsys {

inline constexpr dse::CandidateGeneratorKind
    designCompilerBlockGateNetlistCandidateGeneratorKind(0x53444342);
inline constexpr dse::CandidateGeneratorKind
    designCompilerHierarchicalBlockGateNetlistCandidateGeneratorKind(
        0x53444350);

const dse::CandidateGeneratorDescriptor &
designCompilerBlockGateNetlistCandidateGeneratorDescriptor();
llvm::Error registerDesignCompilerBlockGateNetlistCandidateGenerator();
llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindDesignCompilerBlockGateNetlistInputs(
    const ArtifactRootReference &blockSource,
    const ArtifactRootReference &implementationPlatform);
llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveDesignCompilerBlockGateNetlistBinding(
    const ResolvedDesignCompilerGateNetlistConfigView &config);

/// Synthesizes the parent-local logic against one immutable mapped product
/// per distinct direct child in the exact reusable source graph.
const dse::CandidateGeneratorDescriptor &
designCompilerHierarchicalBlockGateNetlistCandidateGeneratorDescriptor();
llvm::Error
registerDesignCompilerHierarchicalBlockGateNetlistCandidateGenerator();
llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindDesignCompilerHierarchicalBlockGateNetlistInputs(
    const ArtifactRootReference &blockSource,
    const ArtifactRootReference &implementationPlatform,
    llvm::ArrayRef<ArtifactRootReference> compiledChildren);
llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveDesignCompilerHierarchicalBlockGateNetlistBinding(
    const ResolvedDesignCompilerGateNetlistConfigView &config);

/// Associates an accepted complete mapped root with its exact portable RTL
/// implementation. All source derivation and public interface facts are
/// checked before publishing the existing HardwareImplementation schema.
inline constexpr dse::CandidateGeneratorKind
    designCompilerPortableGateImplementationCandidateGeneratorKind(0x53444349);
const dse::CandidateGeneratorDescriptor &
designCompilerPortableGateImplementationCandidateGeneratorDescriptor();
llvm::Error registerDesignCompilerPortableGateImplementationCandidateGenerator();
llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindDesignCompilerPortableGateImplementationInputs(
    const ArtifactRootReference &implementation,
    const ArtifactRootReference &blockNetlist);
llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveDesignCompilerPortableGateImplementationBinding();

/// Includes the vendor-owned mapped library contract validation in addition
/// to the hardware-owned source/technology/interface closure importer.
llvm::Expected<hardware::rtl::FinalizedBlockGateNetlist>
importDesignCompilerBlockGateNetlist(const ArtifactRootReference &reference,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs);

} // namespace loom::eda::synopsys

#endif // LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILERBLOCK_H

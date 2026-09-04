#ifndef LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILERBLOCK_H
#define LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILERBLOCK_H

#include "EDA/Adapters/Synopsys/DesignCompiler.h"
#include "Hardware/RTL/BlockGateNetlist.h"

namespace loom::eda::synopsys {

inline constexpr dse::CandidateGeneratorKind
    designCompilerBlockGateNetlistCandidateGeneratorKind(0x53444342);

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

/// Includes the vendor-owned mapped library contract validation in addition
/// to the hardware-owned source/technology/interface closure importer.
llvm::Expected<hardware::rtl::FinalizedBlockGateNetlist>
importDesignCompilerBlockGateNetlist(const ArtifactRootReference &reference,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs);

} // namespace loom::eda::synopsys

#endif // LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILERBLOCK_H

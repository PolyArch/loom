#ifndef LOOM_DSE_RTLBLOCKSOURCECANDIDATEGENERATOR_H
#define LOOM_DSE_RTLBLOCKSOURCECANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    rtlBlockSourceCandidateGeneratorKind(0x52425352);
inline constexpr CandidateGeneratorKind
    rtlBlockSourceSubgraphCandidateGeneratorKind(0x52425347);

const CandidateGeneratorDescriptor &
rtlBlockSourceCandidateGeneratorDescriptor();
llvm::Error registerRtlBlockSourceCandidateGenerator();
llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRtlBlockSourceInputs(const ArtifactRootReference &implementation);

/// Definition ordinal in the exact canonical module graph of the input
/// HardwareImplementation. The binding owns its canonical scalar encoding.
llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRtlBlockSourceBinding(std::uint64_t definition);

/// The selected ordinal belongs to the normalized graph of an exact Source.
/// This preserves the parent derivation while avoiding a whole-RTL replay for
/// each member of a bottom-up implementation plan.
const CandidateGeneratorDescriptor &
rtlBlockSourceSubgraphCandidateGeneratorDescriptor();
llvm::Error registerRtlBlockSourceSubgraphCandidateGenerator();
llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRtlBlockSourceSubgraphInputs(const ArtifactRootReference &source);
llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRtlBlockSourceSubgraphBinding(std::uint64_t definition);

/// Replays the exact extraction relation before a parent consumes a reused
/// result. Uses the same provider binding and hardware-owned derivation.
llvm::Error verifyRtlBlockSourceDerivation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactRootReference &source, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::dse

#endif // LOOM_DSE_RTLBLOCKSOURCECANDIDATEGENERATOR_H

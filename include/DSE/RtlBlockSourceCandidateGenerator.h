#ifndef LOOM_DSE_RTLBLOCKSOURCECANDIDATEGENERATOR_H
#define LOOM_DSE_RTLBLOCKSOURCECANDIDATEGENERATOR_H

#include "DSE/CandidateGenerator.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    rtlBlockSourceCandidateGeneratorKind(0x52425352);

const CandidateGeneratorDescriptor &
rtlBlockSourceCandidateGeneratorDescriptor();
llvm::Error registerRtlBlockSourceCandidateGenerator();
llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindRtlBlockSourceInputs(const ArtifactRootReference &implementation);

/// Definition ordinal in the exact canonical module graph of the input
/// HardwareImplementation. The binding owns its canonical scalar encoding.
llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveRtlBlockSourceBinding(std::uint64_t definition);

/// Replays the exact extraction relation before a parent consumes a reused
/// result. Uses the same provider binding and hardware-owned derivation.
llvm::Error verifyRtlBlockSourceDerivation(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactRootReference &source, const ArtifactStore &artifacts,
    const BlobStore &blobs);

} // namespace loom::dse

#endif // LOOM_DSE_RTLBLOCKSOURCECANDIDATEGENERATOR_H

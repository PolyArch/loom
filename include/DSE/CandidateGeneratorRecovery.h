#ifndef LOOM_DSE_CANDIDATEGENERATORRECOVERY_H
#define LOOM_DSE_CANDIDATEGENERATORRECOVERY_H

#include "Common/BlobDigest.h"
#include "DSE/CandidateGenerator.h"
#include "DSE/ExecutionJournal.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::dse {

inline constexpr llvm::StringLiteral
    candidateGeneratorFinalizedWorkRecordSchemaIdentity =
        "loom.dse.candidate_generator_finalized_work";
inline constexpr SchemaVersion
    candidateGeneratorFinalizedWorkRecordSchemaVersion{1, 0};

/// Publishes the immutable owner record for one already validated terminal
/// in-process Generate result. The returned digest is nonsemantic recovery
/// material and never identifies a candidate or formal selection.
llvm::Expected<BlobDigest> publishCandidateGeneratorFinalizedWorkRecord(
    const DseRunKey &runKey, const WorkUnitKey &workUnit,
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const CandidateGeneratorProviderResult &result,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

/// Strictly imports one owner record against the complete expected invocation
/// closure and revalidates all referenced output material.
llvm::Expected<CandidateGeneratorProviderResult>
importCandidateGeneratorFinalizedWorkRecord(
    const BlobDigest &recordDigest, const DseRunKey &runKey,
    const WorkUnitKey &workUnit,
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

} // namespace loom::dse

#endif // LOOM_DSE_CANDIDATEGENERATORRECOVERY_H

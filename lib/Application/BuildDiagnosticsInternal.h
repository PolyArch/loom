#ifndef LOOM_APPLICATION_BUILDDIAGNOSTICSINTERNAL_H
#define LOOM_APPLICATION_BUILDDIAGNOSTICSINTERNAL_H

#include "Application/Build.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <optional>
#include <string>

/// Presentation vocabulary shared by the application build diagnostics
/// (planning, Mapping, and statistics projections) and the pair-decision
/// diagnostics (decision-level projections and the runtime manifest binding).
/// The typed build records remain the semantic owners; every spelling and
/// encoder here exists once.
namespace loom::application::diagnostics_detail {

inline constexpr llvm::StringLiteral applicationPairDecisionSchemaIdentity =
    "loom.application_pair_decision";
inline constexpr llvm::StringLiteral applicationPairDecisionSchemaVersion =
    "1.1";
inline constexpr llvm::StringLiteral applicationPairEvidenceSchemaIdentity =
    "loom.application_pair_evidence";
inline constexpr llvm::StringLiteral applicationPairEvidenceSchemaVersion =
    "1.1";
inline constexpr llvm::StringLiteral applicationPairDispositionSchemaIdentity =
    "loom.application_pair_disposition";
inline constexpr llvm::StringLiteral applicationPairDispositionSchemaVersion =
    "1.1";

llvm::StringRef spelling(dse::JointDesignAttemptDisposition value);
llvm::StringRef spelling(dse::JointDesignQualityDisposition value);
llvm::StringRef spelling(dse::JointDesignQualityIncompleteReason value);
llvm::StringRef spelling(ApplicationMappingRuntimeDisposition value);

llvm::json::Value encodeObjectiveScalar(const ResolvedObjectiveScalar &value);
std::string encodeRoot(const ArtifactRootReference &reference);
void addOptionalUnsigned(llvm::json::Object &object, llvm::StringRef key,
                         std::optional<std::uint64_t> value);
void addOptionalRoot(llvm::json::Object &object, llvm::StringRef key,
                     const std::optional<ArtifactRootReference> &value);
llvm::json::Object
encodeQualityProvenance(const dse::JointDesignQualityProvenance &provenance);
llvm::json::Object
encodePairDecision(const ApplicationPairDecisionRecord &decision);

} // namespace loom::application::diagnostics_detail

#endif // LOOM_APPLICATION_BUILDDIAGNOSTICSINTERNAL_H

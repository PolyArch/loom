#ifndef LOOM_EVALUATION_MODELPROVIDER_H
#define LOOM_EVALUATION_MODELPROVIDER_H

#include "Evaluation/Evidence.h"

#include "Common/ProviderForm.h"
#include "ExternalTool/InvocationBundle.h"

#include "llvm/Support/Error.h"

#include <variant>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
}

namespace loom::evaluation {

/// One provider-produced result before the Evaluation owner finalizes the
/// persistent Evidence root. Normalized observations remain in the ordinary
/// Evidence outcome; descriptor outputs remain in their typed output slots.
struct EvaluationModelResult final {
  std::vector<ModelOutputBinding> outputBindings;
  EvaluationEvidenceOutcome outcome;
};

/// The closed provider implementation forms. The registered form must match
/// the model descriptor's provider form exactly.
struct EvaluationModelInProcessProvider final {
  llvm::Expected<EvaluationModelResult> (*evaluate)(
      const EvaluationRequest &request,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);

  friend bool operator==(const EvaluationModelInProcessProvider &lhs,
                         const EvaluationModelInProcessProvider &rhs) {
    return lhs.evaluate == rhs.evaluate;
  }
};

struct EvaluationModelExternalPrepareImportProvider final {
  llvm::Expected<external_tool::PreparedExternalToolInvocation> (*prepare)(
      const EvaluationRequest &request,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore,
      const external_tool::ExternalToolPreparationContext &context);
  llvm::Expected<EvaluationModelResult> (*import)(
      const EvaluationRequest &request,
      const CaseArtifactResolution &resolution,
      const external_tool::PreparedExternalToolInvocation &prepared,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);

  friend bool operator==(
      const EvaluationModelExternalPrepareImportProvider &lhs,
      const EvaluationModelExternalPrepareImportProvider &rhs) {
    return lhs.prepare == rhs.prepare && lhs.import == rhs.import;
  }
};

using EvaluationModelProviderImplementation =
    std::variant<EvaluationModelInProcessProvider,
                 EvaluationModelExternalPrepareImportProvider>;

/// Runtime availability for one exact static model descriptor. The descriptor
/// owns model semantics and accepted Requests; this record owns only the
/// executable implementation currently available in this process.
struct EvaluationModelProvider final {
  EvaluationModelDescriptorRef descriptor;
  EvaluationModelProviderImplementation implementation;
};

/// Registers one provider with static storage duration. Re-registering the
/// same record is a no-op. A second provider for the same exact descriptor is
/// rejected rather than becoming an ambient selection mechanism.
llvm::Error
registerEvaluationModelProvider(const EvaluationModelProvider &provider);

/// Verifies and executes one exact Request through the in-process provider
/// form. Provider absence is a stable Unsupported outcome for this exact
/// Request; an ExternalPrepareImport provider is never invoked through this
/// facade. Provider-produced output and observation cardinality is validated
/// by the sole Evidence constructor.
llvm::Expected<EvaluationEvidence>
evaluateRequest(const EvaluationRequest &request,
                const CaseArtifactResolution &resolution,
                const ArtifactStore &artifactStore,
                const BlobStore &blobStore);

/// Prepares one deterministic finalized invocation bundle through the exact
/// registered ExternalPrepareImport model provider. The descriptor form is
/// validated before any provider lookup; the caller alone decides whether,
/// where, and when to execute run.sh.
llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareEvaluationModelInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore,
    const external_tool::ExternalToolPreparationContext &context);

/// Strictly imports one prepared invocation through the exact registered
/// ExternalPrepareImport model provider against the full typed closure, then
/// validates and finalizes the canonical Evidence value. The provider import
/// itself returns only the transient model result and never publishes
/// Evidence.
llvm::Expected<EvaluationEvidence>
importEvaluationModelInvocation(
    const EvaluationRequest &request, const CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_MODELPROVIDER_H

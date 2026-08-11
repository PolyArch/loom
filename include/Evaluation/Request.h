#ifndef LOOM_EVALUATION_REQUEST_H
#define LOOM_EVALUATION_REQUEST_H

#include "Evaluation/Case.h"
#include "Evaluation/Finding.h"
#include "Evaluation/Metric.h"
#include "Evaluation/ModelDescriptor.h"

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::evaluation {

class MetricRequestOrdinal {
public:
  explicit constexpr MetricRequestOrdinal(std::uint64_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint64_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(MetricRequestOrdinal lhs,
                                   MetricRequestOrdinal rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(MetricRequestOrdinal lhs,
                                   MetricRequestOrdinal rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint64_t ordinal_;
};

class FindingRequestOrdinal {
public:
  explicit constexpr FindingRequestOrdinal(std::uint64_t ordinal)
      : ordinal_(ordinal) {}
  constexpr std::uint64_t ordinal() const { return ordinal_; }

  friend constexpr bool operator==(FindingRequestOrdinal lhs,
                                   FindingRequestOrdinal rhs) {
    return lhs.ordinal_ == rhs.ordinal_;
  }
  friend constexpr bool operator!=(FindingRequestOrdinal lhs,
                                   FindingRequestOrdinal rhs) {
    return !(lhs == rhs);
  }

private:
  std::uint64_t ordinal_;
};

class MetricRequest {
public:
  static llvm::Expected<MetricRequest>
  get(MetricQuery query, llvm::ArrayRef<EvaluationCondition> conditions,
      const EvaluationCase &evaluationCase,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore);

  const MetricQuery &query() const { return query_; }
  llvm::ArrayRef<EvaluationCondition> conditions() const { return conditions_; }

  friend bool operator==(const MetricRequest &lhs, const MetricRequest &rhs) {
    return lhs.query_ == rhs.query_ && lhs.conditions_ == rhs.conditions_;
  }

private:
  MetricRequest(MetricQuery query, std::vector<EvaluationCondition> conditions)
      : query_(std::move(query)), conditions_(std::move(conditions)) {}

  MetricQuery query_;
  std::vector<EvaluationCondition> conditions_;
};

class FindingRequest {
public:
  static llvm::Expected<FindingRequest>
  get(FindingQuery query, llvm::ArrayRef<EvaluationCondition> conditions,
      const EvaluationCase &evaluationCase,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore);

  const FindingQuery &query() const { return query_; }
  llvm::ArrayRef<EvaluationCondition> conditions() const { return conditions_; }

  friend bool operator==(const FindingRequest &lhs, const FindingRequest &rhs) {
    return lhs.query_ == rhs.query_ && lhs.conditions_ == rhs.conditions_;
  }

private:
  FindingRequest(FindingQuery query,
                 std::vector<EvaluationCondition> conditions)
      : query_(std::move(query)), conditions_(std::move(conditions)) {}

  FindingQuery query_;
  std::vector<EvaluationCondition> conditions_;
};

/// Registry-relative ordering keys shared by EvaluationRequest and every
/// owner that persists request shapes. The keys encode typed query ordinals,
/// scope identities, and canonical conditions; textual encodings are not an
/// ordering authority.
std::vector<std::uint8_t>
canonicalMetricRequestKey(const MetricQuery &query,
                          llvm::ArrayRef<EvaluationCondition> conditions);
std::vector<std::uint8_t>
canonicalFindingRequestKey(const FindingQuery &query,
                           llvm::ArrayRef<EvaluationCondition> conditions);

/// The exact `evaluation.request.1.0` typed root. Its case signature and model
/// descriptor are resolved through model_binding and are not copied fields.
class EvaluationRequest {
public:
  static const ArtifactSchemaDescriptor artifactSchema;

  static llvm::Expected<EvaluationRequest>
  get(const EvaluationCase &evaluationCase,
      llvm::ArrayRef<MetricRequest> metricRequests,
      llvm::ArrayRef<FindingRequest> findingRequests,
      ResolvedModelBinding modelBinding, std::uint64_t replicateIndex,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);

  static llvm::Expected<EvaluationRequest>
  get(EvaluationSubjectBindings subjectBindings,
      std::optional<ArtifactRootReference> workload,
      std::optional<ArtifactRootReference> runtimeInput,
      llvm::ArrayRef<EvaluationCondition> baseConditions,
      llvm::ArrayRef<MetricRequest> metricRequests,
      llvm::ArrayRef<FindingRequest> findingRequests,
      ResolvedModelBinding modelBinding, std::uint64_t replicateIndex,
      const CaseArtifactResolution &resolution,
      const ArtifactStore &artifactStore, const BlobStore &blobStore);

  const EvaluationSubjectBindings &subjectBindings() const {
    return subjectBindings_;
  }
  const std::optional<ArtifactRootReference> &workload() const {
    return workload_;
  }
  const std::optional<ArtifactRootReference> &runtimeInput() const {
    return runtimeInput_;
  }
  llvm::ArrayRef<EvaluationCondition> baseConditions() const {
    return baseConditions_;
  }
  llvm::ArrayRef<MetricRequest> metricRequests() const {
    return metricRequests_;
  }
  llvm::ArrayRef<FindingRequest> findingRequests() const {
    return findingRequests_;
  }
  const ResolvedModelBinding &modelBinding() const { return modelBinding_; }
  std::uint64_t replicateIndex() const { return replicateIndex_; }

  const MetricRequest *resolve(MetricRequestOrdinal ordinal) const;
  const FindingRequest *resolve(FindingRequestOrdinal ordinal) const;

private:
  EvaluationRequest(EvaluationSubjectBindings subjectBindings,
                    std::optional<ArtifactRootReference> workload,
                    std::optional<ArtifactRootReference> runtimeInput,
                    std::vector<EvaluationCondition> baseConditions,
                    std::vector<MetricRequest> metricRequests,
                    std::vector<FindingRequest> findingRequests,
                    ResolvedModelBinding modelBinding,
                    std::uint64_t replicateIndex)
      : subjectBindings_(std::move(subjectBindings)),
        workload_(std::move(workload)), runtimeInput_(std::move(runtimeInput)),
        baseConditions_(std::move(baseConditions)),
        metricRequests_(std::move(metricRequests)),
        findingRequests_(std::move(findingRequests)),
        modelBinding_(std::move(modelBinding)),
        replicateIndex_(replicateIndex) {}

  EvaluationSubjectBindings subjectBindings_;
  std::optional<ArtifactRootReference> workload_;
  std::optional<ArtifactRootReference> runtimeInput_;
  std::vector<EvaluationCondition> baseConditions_;
  std::vector<MetricRequest> metricRequests_;
  std::vector<FindingRequest> findingRequests_;
  ResolvedModelBinding modelBinding_;
  std::uint64_t replicateIndex_;
};

class RequestVerifier {
public:
  RequestVerifier(const CaseArtifactResolution &resolution,
                  const ArtifactStore &artifactStore,
                  const BlobStore &blobStore)
      : resolution_(resolution), artifactStore_(artifactStore),
        blobStore_(blobStore) {}

  llvm::Error verify(const EvaluationRequest &request) const;

private:
  const CaseArtifactResolution &resolution_;
  const ArtifactStore &artifactStore_;
  const BlobStore &blobStore_;
};

const EvaluationModelDescriptor *
resolveEvaluationModelDescriptor(const EvaluationRequest &request);

CanonicalSemanticBytes
canonicalEvaluationRequestBytes(const EvaluationRequest &request);

/// Canonical standalone text for the exact descriptor-owned model binding.
/// This is the same production codec used by EvaluationRequest; DSE and other
/// owners must reuse it rather than copying descriptor/config-view framing.
std::string serializeResolvedModelBinding(const ResolvedModelBinding &binding);
llvm::Expected<ResolvedModelBinding>
parseResolvedModelBinding(llvm::StringRef json);

/// Canonical standalone text for an ordered condition sequence. Individual
/// condition payloads remain owned by the Evaluation condition registry.
std::string
serializeEvaluationConditions(llvm::ArrayRef<EvaluationCondition> conditions);
llvm::Expected<std::vector<EvaluationCondition>>
parseEvaluationConditions(llvm::StringRef json);

std::string serializeEvaluationRequest(const EvaluationRequest &request);
llvm::Expected<EvaluationRequest> parseEvaluationRequest(
    llvm::StringRef json, const CaseArtifactResolution &resolution,
    const ArtifactStore &artifactStore, const BlobStore &blobStore);
ArtifactIdentity evaluationRequestIdentity(const EvaluationRequest &request);
ArtifactRootReference
evaluationRequestReference(const EvaluationRequest &request);
llvm::Expected<ArtifactRootReference>
publishEvaluationRequest(const EvaluationRequest &request,
                         const ArtifactStore &artifactStore);
llvm::Expected<EvaluationRequest>
importEvaluationRequest(const ArtifactRootReference &reference,
                        const CaseArtifactResolution &resolution,
                        const ArtifactStore &artifactStore,
                        const BlobStore &blobStore);

/// Reads the exact Artifact roots directly named by the stored Request. This
/// preparatory projection validates the outer owner envelope and typed nested
/// codecs, but full case and dependency admission remains exclusively in
/// `importEvaluationRequest`.
llvm::Expected<std::vector<ArtifactRootReference>>
importEvaluationRequestArtifactReferences(
    const ArtifactRootReference &reference, const ArtifactStore &artifactStore);

/// A removable derived index over exact case facts. It is never serialized
/// into Request or Evidence, and it is not an Artifact identity.
class EvaluationCaseKey {
public:
  using Storage = std::array<std::uint8_t, 32>;

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const EvaluationCaseKey &lhs,
                         const EvaluationCaseKey &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const EvaluationCaseKey &lhs,
                         const EvaluationCaseKey &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit EvaluationCaseKey(Storage bytes) : bytes_(bytes) {}

  friend EvaluationCaseKey baseCaseKey(const EvaluationCase &evaluationCase);
  friend EvaluationCaseKey metricCaseKey(const EvaluationCase &evaluationCase,
                                         const MetricRequest &request);

  Storage bytes_;
};

EvaluationCaseKey baseCaseKey(const EvaluationCase &evaluationCase);
EvaluationCaseKey metricCaseKey(const EvaluationCase &evaluationCase,
                                const MetricRequest &request);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_REQUEST_H

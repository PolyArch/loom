#ifndef LOOM_EVALUATION_MODELPARAMETERBUNDLE_H
#define LOOM_EVALUATION_MODELPARAMETERBUNDLE_H

#include "Common/Artifact.h"
#include "Common/BlobDigest.h"
#include "Evaluation/OwnerValue.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom {
class ArtifactStore;
class BlobStore;
} // namespace loom

namespace loom::evaluation {

inline constexpr ArtifactSchemaDescriptor modelParameterBundleSchema{
    "loom.model_parameter_bundle", SchemaVersion{1, 0}};

class ModelParameterContractRef final {
public:
  static llvm::Expected<ModelParameterContractRef>
  get(llvm::StringRef ownerRegistryIdentity, SchemaVersion ownerRegistryVersion,
      std::uint32_t ownerLocalContractKind);

  llvm::StringRef ownerRegistryIdentity() const {
    return ownerRegistryIdentity_;
  }
  SchemaVersion ownerRegistryVersion() const { return ownerRegistryVersion_; }
  std::uint32_t ownerLocalContractKind() const {
    return ownerLocalContractKind_;
  }

  friend bool operator==(const ModelParameterContractRef &lhs,
                         const ModelParameterContractRef &rhs) {
    return lhs.ownerRegistryIdentity_ == rhs.ownerRegistryIdentity_ &&
           lhs.ownerRegistryVersion_ == rhs.ownerRegistryVersion_ &&
           lhs.ownerLocalContractKind_ == rhs.ownerLocalContractKind_;
  }
  friend bool operator!=(const ModelParameterContractRef &lhs,
                         const ModelParameterContractRef &rhs) {
    return !(lhs == rhs);
  }

  friend bool operator<(const ModelParameterContractRef &lhs,
                        const ModelParameterContractRef &rhs);

private:
  ModelParameterContractRef(std::string ownerRegistryIdentity,
                            SchemaVersion ownerRegistryVersion,
                            std::uint32_t ownerLocalContractKind)
      : ownerRegistryIdentity_(std::move(ownerRegistryIdentity)),
        ownerRegistryVersion_(ownerRegistryVersion),
        ownerLocalContractKind_(ownerLocalContractKind) {}

  std::string ownerRegistryIdentity_;
  SchemaVersion ownerRegistryVersion_;
  std::uint32_t ownerLocalContractKind_ = 0;
};

/// Canonical owner-local registry framing: u64be owner length, owner bytes,
/// u32be major, u32be minor, and u32be local kind.
std::vector<std::uint8_t> canonicalModelParameterContractReferenceBytes(
    const ModelParameterContractRef &reference);

class ModelParameterBundle final {
public:
  const ModelParameterContractRef &parameterContract() const {
    return parameterContract_;
  }
  const BlobDigest &payloadDigest() const { return payloadDigest_; }

private:
  ModelParameterBundle(ModelParameterContractRef parameterContract,
                       BlobDigest payloadDigest)
      : parameterContract_(std::move(parameterContract)),
        payloadDigest_(std::move(payloadDigest)) {}

  ModelParameterContractRef parameterContract_;
  BlobDigest payloadDigest_;

  friend llvm::Expected<ModelParameterBundle>
  parseModelParameterBundle(llvm::StringRef json);
  friend llvm::Expected<class FinalizedModelParameterBundle>
  finalizeModelParameterBundle(
      const ModelParameterContractRef &parameterContract,
      const OwnerValue &parameters, const ArtifactStore &artifactStore,
      const BlobStore &blobStore);
  friend llvm::Expected<class FinalizedModelParameterBundle>
  importModelParameterBundle(const ArtifactRootReference &reference,
                             const ArtifactStore &artifactStore,
                             const BlobStore &blobStore);
};

class FinalizedModelParameterBundle final {
public:
  const ArtifactRootReference &reference() const { return reference_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  const ModelParameterBundle &bundle() const { return bundle_; }
  const OwnerValue &ownerParameters() const { return parameters_; }

  template <typename T> const std::decay_t<T> *parametersIf() const {
    return parameters_.getIf<T>();
  }

private:
  FinalizedModelParameterBundle(ArtifactRootReference reference,
                                CanonicalSemanticBytes canonicalBytes,
                                ModelParameterBundle bundle,
                                OwnerValue parameters)
      : reference_(std::move(reference)),
        canonicalBytes_(std::move(canonicalBytes)), bundle_(std::move(bundle)),
        parameters_(std::move(parameters)) {}

  ArtifactRootReference reference_;
  CanonicalSemanticBytes canonicalBytes_;
  ModelParameterBundle bundle_;
  OwnerValue parameters_;

  friend llvm::Expected<FinalizedModelParameterBundle>
  importModelParameterBundle(const ArtifactRootReference &reference,
                             const ArtifactStore &artifactStore,
                             const BlobStore &blobStore);
};

CanonicalSemanticBytes
canonicalModelParameterBundleBytes(const ModelParameterBundle &bundle);

std::string serializeModelParameterBundle(const ModelParameterBundle &bundle);

llvm::Expected<ModelParameterBundle>
parseModelParameterBundle(llvm::StringRef json);

llvm::Expected<FinalizedModelParameterBundle>
finalizeModelParameterBundle(const ModelParameterContractRef &parameterContract,
                             const OwnerValue &parameters,
                             const ArtifactStore &artifactStore,
                             const BlobStore &blobStore);

llvm::Expected<FinalizedModelParameterBundle>
importModelParameterBundle(const ArtifactRootReference &reference,
                           const ArtifactStore &artifactStore,
                           const BlobStore &blobStore);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_MODELPARAMETERBUNDLE_H

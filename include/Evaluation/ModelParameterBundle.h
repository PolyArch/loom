#ifndef LOOM_EVALUATION_MODELPARAMETERBUNDLE_H
#define LOOM_EVALUATION_MODELPARAMETERBUNDLE_H

#include "Common/Artifact.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>

namespace loom::evaluation {

inline constexpr ArtifactSchemaDescriptor modelParameterBundleSchema{
    "loom.model_parameter_bundle", SchemaVersion{1, 0}};

class ModelParameterContractRef final {
public:
  static llvm::Expected<ModelParameterContractRef>
  get(llvm::StringRef ownerRegistryIdentity, SchemaVersion ownerRegistryVersion,
      std::uint32_t ownerLocalContractKind) {
    if (ownerRegistryIdentity.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "model parameter owner is empty");
    for (unsigned char character : ownerRegistryIdentity)
      if (character < 0x21 || character > 0x7e)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "model parameter owner is not canonical ASCII");
    return ModelParameterContractRef(ownerRegistryIdentity.str(),
                                     ownerRegistryVersion,
                                     ownerLocalContractKind);
  }

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

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_MODELPARAMETERBUNDLE_H

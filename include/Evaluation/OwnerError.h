#ifndef LOOM_EVALUATION_OWNER_ERROR_H
#define LOOM_EVALUATION_OWNER_ERROR_H

#include "Common/Artifact.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>

namespace loom::evaluation {

class EvaluationOwnerUnavailableError final
    : public llvm::ErrorInfo<EvaluationOwnerUnavailableError> {
public:
  static char ID;

  EvaluationOwnerUnavailableError(std::string ownerIdentity,
                                  SchemaVersion ownerVersion)
      : ownerIdentity_(std::move(ownerIdentity)), ownerVersion_(ownerVersion) {}

  llvm::StringRef ownerIdentity() const { return ownerIdentity_; }
  SchemaVersion ownerVersion() const { return ownerVersion_; }

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  std::string ownerIdentity_;
  SchemaVersion ownerVersion_;
};

llvm::Error evaluationOwnerUnavailable(llvm::StringRef ownerIdentity,
                                       SchemaVersion ownerVersion);

} // namespace loom::evaluation

#endif // LOOM_EVALUATION_OWNER_ERROR_H

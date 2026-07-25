#include "Evaluation/OwnerError.h"

#include "Common/ArtifactText.h"

#include "llvm/Support/raw_ostream.h"

namespace loom::evaluation {

char EvaluationOwnerUnavailableError::ID = 0;

void EvaluationOwnerUnavailableError::log(llvm::raw_ostream &stream) const {
  stream << "evaluation_owner_unavailable: '" << ownerIdentity_ << " "
         << formatSchemaVersion(ownerVersion_) << "'";
}

std::error_code
EvaluationOwnerUnavailableError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Error evaluationOwnerUnavailable(llvm::StringRef ownerIdentity,
                                       SchemaVersion ownerVersion) {
  return llvm::make_error<EvaluationOwnerUnavailableError>(
      ownerIdentity.str(), ownerVersion);
}

} // namespace loom::evaluation

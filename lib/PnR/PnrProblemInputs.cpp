#include "PnR/PnrProblemInputs.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <system_error>
#include <utility>

using namespace loom::mapping;
using namespace loom::pnr;

char PnrProblemInputError::ID;

void PnrProblemInputError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code PnrProblemInputError::convertToErrorCode() const {
  return std::make_error_code(std::errc::invalid_argument);
}

namespace {

llvm::Error identityMismatch(PnrProblemInputErrorCode code,
                             const ArtifactIdentity &expectedIdentity,
                             const ArtifactIdentity &actualIdentity,
                             const llvm::Twine &message) {
  return llvm::make_error<PnrProblemInputError>(code, expectedIdentity,
                                                actualIdentity, message.str());
}

} // namespace

llvm::Error
loom::pnr::validatePnrProblemInputs(const PnrProblemInputs &inputs) {
  if (inputs.techMapping.header().dataflowIdentity != inputs.dataflow.identity)
    return identityMismatch(
        PnrProblemInputErrorCode::TechMappingDataflowIdentityMismatch,
        inputs.dataflow.identity, inputs.techMapping.header().dataflowIdentity,
        "TechMapping dataflow identity does not match the PnR dataflow input");
  if (inputs.techMapping.header().fabricIdentity != inputs.fabric.identity)
    return identityMismatch(
        PnrProblemInputErrorCode::TechMappingFabricIdentityMismatch,
        inputs.fabric.identity, inputs.techMapping.header().fabricIdentity,
        "TechMapping fabric identity does not match the PnR fabric input");
  if (inputs.constraints.dataflowIdentity != inputs.dataflow.identity)
    return identityMismatch(
        PnrProblemInputErrorCode::ConstraintSetDataflowIdentityMismatch,
        inputs.dataflow.identity, inputs.constraints.dataflowIdentity,
        "MappingConstraintSet dataflow identity does not match the PnR "
        "dataflow input");
  if (inputs.constraints.techMappingIdentity != inputs.techMappingIdentity)
    return identityMismatch(
        PnrProblemInputErrorCode::ConstraintSetTechMappingIdentityMismatch,
        inputs.techMappingIdentity, inputs.constraints.techMappingIdentity,
        "MappingConstraintSet TechMapping identity does not match the PnR "
        "TechMapping input");
  if (inputs.constraints.fabricIdentity != inputs.fabric.identity)
    return identityMismatch(
        PnrProblemInputErrorCode::ConstraintSetFabricIdentityMismatch,
        inputs.fabric.identity, inputs.constraints.fabricIdentity,
        "MappingConstraintSet fabric identity does not match the PnR fabric "
        "input");
  return llvm::Error::success();
}

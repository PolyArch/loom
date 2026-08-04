#include "PnR/SpatialCandidateState.h"

#include "SpatialBindingRelationModel.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <system_error>

using namespace loom::pnr;

namespace {

llvm::Error candidateError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial candidate state: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

} // namespace

llvm::Error
SpatialCandidateState::verifyBindingRelation(PnrIndex relation) const {
  const detail::SpatialBindingRelationModel &model =
      problem_->bindingRelations();
  if (relation >= model.relations().relations().size())
    return candidateError("binding relation is out of range");
  if (!model.relationSatisfied(relation, bindingRelationChoices_))
    return candidateError("hard equality or disjoint relation is violated");
  return llvm::Error::success();
}

llvm::Error SpatialCandidateState::verifyBindingRelations() const {
  const detail::SpatialBindingRelationModel &model =
      problem_->bindingRelations();
  if (bindingRelationChoices_.size() != model.decisionCount())
    return candidateError(
        "binding relation choices do not match the frozen decision domain");

  for (auto [realization, binding] : llvm::enumerate(computeBindings_)) {
    const auto expected = model.computeChoiceOrdinal(
        static_cast<PnrIndex>(realization), binding.placement,
        binding.instructionContext);
    if (!expected || bindingRelationChoices_[realization] != *expected)
      return candidateError(
          "compute binding diverges from its relation-domain choice");
  }
  const PnrIndex memoryOffset = model.computeDecisionCount();
  for (auto [realization, binding] : llvm::enumerate(memoryBindings_)) {
    const auto expected = model.memoryChoiceOrdinal(
        static_cast<PnrIndex>(realization), binding.placement);
    if (!expected ||
        bindingRelationChoices_[memoryOffset + realization] != *expected)
      return candidateError(
          "memory binding diverges from its relation-domain choice");
  }
  for (auto [demand, attachment] : llvm::enumerate(portAttachments_)) {
    const auto expected = model.portAttachmentChoiceOrdinal(
        static_cast<PnrIndex>(demand), attachment);
    if (!expected ||
        bindingRelationChoices_[model.portDecisionOffset() + demand] !=
            *expected)
      return candidateError(
          "PortAttachment diverges from its relation-domain choice");
  }
  for (auto [boundary, attachment] :
       llvm::enumerate(graphBoundaryAttachments_)) {
    const auto expected = model.graphBoundaryAttachmentChoiceOrdinal(
        static_cast<PnrIndex>(boundary), attachment);
    if (!expected ||
        bindingRelationChoices_[model.graphBoundaryDecisionOffset() +
                                boundary] != *expected)
      return candidateError(
          "graph-boundary attachment diverges from its relation-domain choice");
  }
  for (PnrIndex relation = 0; relation < model.relations().relations().size();
       ++relation)
    if (llvm::Error error = verifyBindingRelation(relation))
      return error;
  return llvm::Error::success();
}

#include "Evaluation/StandardFindings.h"

#include "Evaluation/Evidence.h"

#include "llvm/Support/Error.h"

namespace loom::evaluation::standard_findings {
namespace {

const ScopeFormDescriptor kWholeCaseScope[] = {
    {ScopeFormRef(0),
     "the entire exact comparison case",
     {},
     WholeExactCaseScope{},
     nullptr}};

llvm::Expected<std::vector<std::uint8_t>>
encodeFunctionalMismatch(const OwnerValue &occurrence) {
  if (!occurrence.getIf<FunctionalMismatchOccurrence>())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "functional mismatch occurrence has the wrong owner type");
  return std::vector<std::uint8_t>{};
}

llvm::Expected<OwnerValue>
decodeFunctionalMismatch(llvm::ArrayRef<std::uint8_t> canonicalPayload) {
  if (!canonicalPayload.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "functional mismatch occurrence payload must be empty");
  return OwnerValue::get(FunctionalMismatchOccurrence{});
}

llvm::Error
validateFunctionalMismatch(const OwnerValue &occurrence,
                           const FindingOccurrenceContext &context) {
  if (!occurrence.getIf<FunctionalMismatchOccurrence>())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "functional mismatch occurrence has the wrong owner type");
  const FindingRequest *request =
      context.request().resolve(context.findingRequestOrdinal());
  if (!request || request->query().kind != FunctionalMismatch)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "functional mismatch occurrence has the wrong request owner");
  return llvm::Error::success();
}

const FindingDescriptor kFunctionalMismatchDescriptor{
    FunctionalMismatch,
    "functional_mismatch",
    "Exact functional observations differ under the model's proven "
    "deterministic relation.",
    kWholeCaseScope,
    {},
    {{"evaluation.functional_mismatch", {1, 0}},
     &encodeFunctionalMismatch,
     &decodeFunctionalMismatch,
     &validateFunctionalMismatch},
    std::nullopt};

} // namespace

llvm::Error registerStandardFindings() {
  return registerFindingDescriptor(kFunctionalMismatchDescriptor);
}

} // namespace loom::evaluation::standard_findings

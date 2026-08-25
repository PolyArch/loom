#include "BuildInternal.h"

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <optional>
#include <utility>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "application pair outcome test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void typedReasonProjection() {
  using PairDisposition = loom::application::ApplicationPairDecisionDisposition;
  using FrontierReason = loom::dse::ResourceTimeFrontierIncompleteReason;
  using RuntimeDisposition =
      loom::application::ApplicationMappingRuntimeDisposition;
  using namespace loom::application::build_detail;

  require(mapResourceTimeFrontierReasonToPairDisposition(
              FrontierReason::BudgetExhausted) ==
              PairDisposition::BudgetExhausted,
          "frontier budget reason lost its pair disposition");
  require(mapResourceTimeFrontierReasonToPairDisposition(
              FrontierReason::CancelledOrTimeout) ==
              PairDisposition::CancelledOrTimeout,
          "frontier cancellation lost its pair disposition");
  require(mapResourceTimeFrontierReasonToPairDisposition(
              FrontierReason::ProofNotEstablished) ==
              PairDisposition::MappingProofNotEstablished,
          "frontier proof gap became a budget outcome");
  require(mapResourceTimeFrontierReasonToPairDisposition(
              FrontierReason::Unsupported) ==
              PairDisposition::UnsupportedSemantic,
          "frontier unsupported reason lost its pair disposition");

  const std::array runtimeCases = {
      std::pair{RuntimeDisposition::Unsupported,
                PairDisposition::UnsupportedSemantic},
      std::pair{RuntimeDisposition::ProofNotEstablished,
                PairDisposition::MappingProofNotEstablished},
      std::pair{RuntimeDisposition::ExecutionFailed,
                PairDisposition::ImplementationFailure},
      std::pair{RuntimeDisposition::CancelledOrTimeout,
                PairDisposition::CancelledOrTimeout}};
  for (const auto &[runtime, expected] : runtimeCases) {
    const auto projected = mapRuntimeDispositionToPairDisposition(runtime);
    require(projected && *projected == expected,
            "runtime reason lost its canonical pair disposition");
  }
}

void spectrumSelectionProjection() {
  using PairDisposition = loom::application::ApplicationPairDecisionDisposition;
  using SpectrumClass = loom::dse::PreMappingSpectrumClass;
  using SpectrumReason = loom::dse::ResourceTimeSpectrumIncompleteReason;
  using namespace loom::application::build_detail;

  require(!classifyResourceTimeSelectionOutcome(std::nullopt, std::nullopt),
          "automatic spectrum selection required absent evidence");
  require(classifyResourceTimeSelectionOutcome(std::nullopt,
                                               SpectrumClass::MaxTemporal) ==
              PairDisposition::MappingProofNotEstablished,
          "explicit endpoint without evidence lost its proof gap");

  const std::array incompleteCases = {
      std::pair{SpectrumReason::Unsupported,
                PairDisposition::UnsupportedSemantic},
      std::pair{SpectrumReason::ProofNotEstablished,
                PairDisposition::MappingProofNotEstablished},
      std::pair{SpectrumReason::CancelledOrTimeout,
                PairDisposition::CancelledOrTimeout}};
  for (const auto &[reason, expected] : incompleteCases) {
    std::optional<loom::dse::ResourceTimeSpectrumFunnelResult> spectrum{
        loom::dse::ResourceTimeSpectrumFunnelResult{
            loom::dse::ResourceTimeSpectrumVerification{
                loom::dse::IncompleteResourceTimeSpectrum{reason, "typed", 0}},
            loom::dse::ResourceTimeSpectrumFunnelAccounting{}}};
    require(classifyResourceTimeSelectionOutcome(
                spectrum, SpectrumClass::MaxTemporal) == expected,
            "spectrum incomplete reason lost its pair disposition");
  }

  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> bytes{};
  loom::ArtifactRootReference root{
      "loom.test.application_pair",
      {1, 0},
      take(loom::ArtifactIdentity::fromBytes(bytes))};
  loom::dse::VerifiedResourceTimeSpectrumScenario scenario;
  scenario.spectrumClass = SpectrumClass::Intermediate;
  std::optional<loom::dse::ResourceTimeSpectrumFunnelResult> verified{
      loom::dse::ResourceTimeSpectrumFunnelResult{
          loom::dse::ResourceTimeSpectrumVerification{
              loom::dse::VerifiedResourceTimeSpectrum{root, root, {scenario}}},
          loom::dse::ResourceTimeSpectrumFunnelAccounting{}}};
  require(!classifyResourceTimeSelectionOutcome(verified,
                                                SpectrumClass::Intermediate),
          "verified requested endpoint was rejected");
  require(classifyResourceTimeSelectionOutcome(verified,
                                               SpectrumClass::MaxSpatial) ==
              PairDisposition::MappingProofNotEstablished,
          "verified non-endpoint schedule satisfied a different request");
}

void incompleteCausePriority() {
  using PairDisposition = loom::application::ApplicationPairDecisionDisposition;
  using loom::application::build_detail::prioritizeIncompletePairDisposition;

  const std::array proofCause = {PairDisposition::MappingProofNotEstablished};
  require(prioritizeIncompletePairDisposition(proofCause, true) ==
              PairDisposition::MappingProofNotEstablished,
          "declared work exhaustion masked a typed proof gap");
  const std::array mixedCauses = {PairDisposition::UnsupportedSemantic,
                                  PairDisposition::CancelledOrTimeout};
  require(prioritizeIncompletePairDisposition(mixedCauses, true) ==
              PairDisposition::CancelledOrTimeout,
          "earlier incomplete evidence masked cancellation");
  require(prioritizeIncompletePairDisposition({}, true) ==
              PairDisposition::BudgetExhausted,
          "unattributed declared work exhaustion lost its fallback");
}

} // namespace

int main() {
  typedReasonProjection();
  spectrumSelectionProjection();
  incompleteCausePriority();
  return 0;
}

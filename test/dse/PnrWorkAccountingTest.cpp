#include "DSE/MappingCandidateGenerator.h"
#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"
#include "PnR/System/SystemPnrGenerator.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "PnR work accounting test failed: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

void requireFailureContains(llvm::Error error, llvm::StringRef fragment) {
  if (!error)
    fail("expected failure containing " + fragment);
  const std::string diagnostic = llvm::toString(std::move(error));
  if (!llvm::StringRef(diagnostic).contains(fragment))
    fail("unexpected diagnostic: " + diagnostic);
}

} // namespace

int main() {
  loom::pnr::SystemPnrGenerationAccounting accounting;
  accounting.plannedSeedAttemptSlots = 3;
  accounting.seedAttemptSlots = 2;
  accounting.plannedInitializerAssignmentAttempts = 11;
  accounting.initializerAssignmentAttempts = 9;
  accounting.plannedEndpointExpansionSlots = 29;
  accounting.endpointExpansionSlots = 23;
  accounting.plannedNegotiationIterationSlots = 7;
  accounting.negotiationIterationSlots = 6;
  accounting.plannedCalibrationProposalSlots = 5;
  accounting.calibrationProposalSlots = 4;
  accounting.plannedAnnealingBaseProposalSlots = 13;
  accounting.annealingBaseProposalSlots = 12;
  accounting.plannedAnnealingMovableProposalSlots = 17;
  accounting.annealingMovableProposalSlots = 16;

  const auto summary =
      loom::dse::rootCompleteSystemPnrCandidateGeneratorWorkSummary(accounting);
  if (summary.size() != loom::dse::pnrCandidateGeneratorWorkUnits.size())
    fail("System work summary changed width");
  const std::uint64_t planned[] = {3, 11, 29, 7, 5, 13, 17, 0, 0};
  const std::uint64_t consumed[] = {2, 9, 23, 6, 4, 12, 16, 0, 0};
  for (std::size_t ordinal = 0; ordinal != summary.size(); ++ordinal)
    if (!(summary[ordinal].unit ==
          loom::dse::CandidateGeneratorWorkUnitRef(ordinal)) ||
        summary[ordinal].planned != planned[ordinal] ||
        summary[ordinal].consumed != consumed[ordinal])
      fail("System work summary replaced owner plans with consumed work");

  requireSuccess(loom::pnr::verifySystemPnrWorkAccounting(
      accounting, /*requireClosedWork=*/false));
  requireFailureContains(loom::pnr::verifySystemPnrWorkAccounting(
                             accounting, /*requireClosedWork=*/true),
                         "work still live");
  accounting.seedAttemptSlots = 4;
  requireFailureContains(loom::pnr::verifySystemPnrWorkAccounting(
                             accounting, /*requireClosedWork=*/false),
                         "consumed work exceeds planned work");
  accounting.seedAttemptSlots = accounting.plannedSeedAttemptSlots;
  accounting.initializerAssignmentAttempts =
      accounting.plannedInitializerAssignmentAttempts;
  accounting.endpointExpansionSlots = accounting.plannedEndpointExpansionSlots;
  accounting.negotiationIterationSlots =
      accounting.plannedNegotiationIterationSlots;
  accounting.calibrationProposalSlots =
      accounting.plannedCalibrationProposalSlots;
  accounting.annealingBaseProposalSlots =
      accounting.plannedAnnealingBaseProposalSlots;
  accounting.annealingMovableProposalSlots =
      accounting.plannedAnnealingMovableProposalSlots;
  requireSuccess(loom::pnr::verifySystemPnrWorkAccounting(
      accounting, /*requireClosedWork=*/true));

  llvm::outs() << "PnR work accounting anchors passed\n";
  return EXIT_SUCCESS;
}

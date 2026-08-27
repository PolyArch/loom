#ifndef LOOM_TEST_DSE_PLANEXECUTIONTESTSUPPORT_H
#define LOOM_TEST_DSE_PLANEXECUTIONTESTSUPPORT_H

#include "Common/Artifact.h"
#include "Config/ResolvedConfig.h"
#include "DSE/InvocationManifest.h"
#include "DSE/ResolvedConfigView.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>

namespace loom {
class ArtifactStore;
}

namespace loom::dse::test_support {

struct PlanExecutionFixture final {
  ResolvedConfig config;
  ResolvedDseConfigView view;
  DseRunClosure closure;
  ArtifactRootReference source;
};

enum class PlanExecutionProviderOutcomeKind : std::uint8_t {
  Candidate,
  ProvenInfeasible,
  CompletedEmpty,
};

llvm::Error registerPlanExecutionTestGenerator();
llvm::Expected<PlanExecutionFixture>
makePlanExecutionFixture(const ArtifactStore &store, std::size_t nodeCount,
                         llvm::StringRef producerIdentity,
                         bool provenInfeasibleSource = false);

void resetPlanExecutionProviderObservations();
void requireConcurrentPlanExecutionProviders(std::uint64_t count);
void requireConcurrentPlanExecutionProvidersAfterSerialPrefix(
    std::uint64_t serialProviderCalls, std::uint64_t count);
void requirePlanExecutionProviderStopObservation();
void setPlanExecutionProviderOutcome(PlanExecutionProviderOutcomeKind outcome);
bool waitForActivePlanExecutionProvider();
std::uint64_t planExecutionProviderCalls();
std::uint64_t maximumConcurrentPlanExecutionProviders();
bool planExecutionProviderObservedStop();
std::uint64_t planExecutionProviderCpuBudgetCores();
std::uint64_t planExecutionProviderMemoryBudgetBytes();

} // namespace loom::dse::test_support

#endif // LOOM_TEST_DSE_PLANEXECUTIONTESTSUPPORT_H

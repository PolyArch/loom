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

llvm::Error registerPlanExecutionTestGenerator();
llvm::Expected<PlanExecutionFixture>
makePlanExecutionFixture(const ArtifactStore &store, std::size_t nodeCount,
                         llvm::StringRef producerIdentity);

void resetPlanExecutionProviderObservations();
void requireConcurrentPlanExecutionProviders(std::uint64_t count);
std::uint64_t planExecutionProviderCalls();
std::uint64_t maximumConcurrentPlanExecutionProviders();

} // namespace loom::dse::test_support

#endif // LOOM_TEST_DSE_PLANEXECUTIONTESTSUPPORT_H

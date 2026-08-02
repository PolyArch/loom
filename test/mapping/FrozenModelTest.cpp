#include "MappingCoreTestSupport.h"

#include "Common/ResolvedConfig.h"
#include "PnR/FrozenModel.h"

#include <type_traits>

namespace loom::mapping::test {
namespace {

static_assert(!std::is_copy_constructible_v<FrozenModel>);
static_assert(!std::is_copy_constructible_v<FrozenRealizationGraph>);
static_assert(!std::is_copy_constructible_v<FrozenRoutingGraph>);

void publishesOneCoupledImmutableModel() {
  TestCase testCase = makeValidCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config = makeSpatialPnrConfigView(__func__);
  PnrProblemInputs inputs = makePnrProblemInputs(testCase, mapping, config);

  FrozenModelHandle first =
      takeExpected(__func__, freezeSpatialPnrModel(inputs));
  FrozenModelHandle second =
      takeExpected(__func__, freezeSpatialPnrModel(inputs));
  if (!first || !second || first->cacheKey() != second->cacheKey() ||
      first->realizations() != second->realizations() ||
      first->routing() != second->routing())
    fail(__func__, "identical inputs did not produce one deterministic model");

  if (first->realizations().physicalEndpoints().size() !=
      first->routing().computeEndpointVertices().size())
    fail(__func__, "aggregate model published inconsistent endpoint domains");
  for (std::size_t index = 0;
       index < first->realizations().physicalEndpoints().size(); ++index) {
    const PnrIndex vertex = first->routing().computeEndpointVertices()[index];
    if (first->routing().routingEndpoints()[vertex].id !=
        first->realizations().physicalEndpoints()[index].id)
      fail(__func__, "aggregate model lost endpoint correspondence");
  }
  if (first->workBudget().empty())
    fail(__func__, "aggregate model omitted its derived work budget");
}

void cacheKeyBindsExactDependenciesAndSelectedView() {
  TestCase testCase = makeValidCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedConfig firstResolved = defaultResolvedConfig();
  ResolvedPnrConfigView firstConfig = takeExpected(
      __func__, projectResolvedSpatialPnrConfigView(firstResolved));
  PnrProblemInputs firstInputs =
      makePnrProblemInputs(testCase, mapping, firstConfig);
  FrozenModelHandle first =
      takeExpected(__func__, freezeSpatialPnrModel(firstInputs));

  PnrProblemInputs changedConstraints = firstInputs;
  changedConstraints.constraints.identity = artifact(243);
  FrozenModelCacheKey changedConstraintKey =
      takeExpected(__func__, deriveFrozenModelCacheKey(changedConstraints));
  if (first->cacheKey() == changedConstraintKey)
    fail(__func__, "changed MappingConstraintSet reused the freeze key");
  llvm::Error changedConstraintError =
      revalidateFrozenModelCacheHit(*first, changedConstraints);
  if (!changedConstraintError)
    fail(__func__, "cache hit accepted a different MappingConstraintSet");
  llvm::consumeError(std::move(changedConstraintError));

  ResolvedConfig changedResolved = firstResolved;
  ++changedResolved.dse.spatialPnr.search.routing.endpointExpansionLimit;
  ResolvedPnrConfigView changedConfig = takeExpected(
      __func__, projectResolvedSpatialPnrConfigView(changedResolved));
  PnrProblemInputs changedConfigInputs{testCase.dataflow, mapping,
                                       testCase.fabric, changedConfig,
                                       firstInputs.constraints};
  FrozenModelCacheKey changedConfigKey =
      takeExpected(__func__, deriveFrozenModelCacheKey(changedConfigInputs));
  if (first->cacheKey() == changedConfigKey)
    fail(__func__, "changed selected PnR view reused the freeze key");
  llvm::Error changedConfigError =
      revalidateFrozenModelCacheHit(*first, changedConfigInputs);
  if (!changedConfigError)
    fail(__func__, "cache hit accepted a different selected PnR view");
  llvm::consumeError(std::move(changedConfigError));

  if (llvm::Error error = revalidateFrozenModelCacheHit(*first, firstInputs))
    fail(__func__, llvm::toString(std::move(error)).c_str());
}

} // namespace

void runFrozenModelTests() {
  publishesOneCoupledImmutableModel();
  cacheKeyBindsExactDependenciesAndSelectedView();
}

} // namespace loom::mapping::test

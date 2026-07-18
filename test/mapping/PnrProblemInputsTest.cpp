#include "MappingCoreTestSupport.h"

#include "PnR/FrozenRoutingGraph.h"
#include "PnR/PnrProblemInputs.h"

#include <type_traits>
#include <utility>

namespace loom::mapping::test {
namespace {

template <typename T, typename = void>
struct HasArtifactIdentityMember : std::false_type {};

template <typename T>
struct HasArtifactIdentityMember<
    T, std::void_t<decltype(std::declval<T>().identity)>> : std::true_type {};

static_assert(!HasArtifactIdentityMember<PnrProblemInputs>::value);
static_assert(!HasArtifactIdentityMember<ResolvedPnrConfigView>::value);
static_assert(!std::is_default_constructible_v<MappingConstraintSetInput>);

PnrProblemInputs
makeProblemInputs(const TestCase &testCase, const ValidatedTechMapping &mapping,
                  const ResolvedPnrConfigView &config,
                  ArtifactIdentity techMappingIdentity = artifact(3),
                  ArtifactIdentity resolvedConfigIdentity = artifact(4),
                  ArtifactIdentity constraintSetIdentity = artifact(5)) {
  return PnrProblemInputs{testCase.dataflow,
                          mapping,
                          techMappingIdentity,
                          testCase.fabric,
                          config,
                          resolvedConfigIdentity,
                          MappingConstraintSetInput{
                              constraintSetIdentity, testCase.dataflow.identity,
                              techMappingIdentity, testCase.fabric.identity}};
}

void expectProblemInputError(const char *test, llvm::Error error,
                             PnrProblemInputErrorCode expectedCode,
                             const ArtifactIdentity &expectedIdentity,
                             const ArtifactIdentity &actualIdentity) {
  if (!error)
    fail(test, "expected PnR problem input failure");
  bool matched = false;
  llvm::handleAllErrors(
      std::move(error), [&](const PnrProblemInputError &inputError) {
        matched = inputError.code() == expectedCode &&
                  inputError.expectedIdentity() == expectedIdentity &&
                  inputError.actualIdentity() == actualIdentity;
      });
  if (!matched)
    fail(test, "received a different PnR problem input failure");
}

template <typename T>
void expectFreezeInputError(const char *test, llvm::Expected<T> result,
                            PnrProblemInputErrorCode expectedCode,
                            const ArtifactIdentity &expectedIdentity,
                            const ArtifactIdentity &actualIdentity) {
  if (result)
    fail(test, "expected freeze input failure");
  expectProblemInputError(test, result.takeError(), expectedCode,
                          expectedIdentity, actualIdentity);
}

void exactFiveInputBindingReachesExistingFreezeBehavior() {
  TestCase testCase = makeValidCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config;
  PnrProblemInputs inputs = makeProblemInputs(testCase, mapping, config);

  FrozenRealizationGraph realizations =
      takeExpected(__func__, freezeRealizationGraph(inputs));
  FrozenRoutingGraph routing =
      takeExpected(__func__, freezeRoutingGraph(inputs));

  if (realizations.computeRealizations().size() !=
          mapping.realizations().size() ||
      routing.computeEndpointVertices().size() !=
          realizations.physicalEndpoints().size())
    fail(__func__, "five-input freeze changed the existing projections");
}

void rejectsEachExactCouplingMismatch() {
  const ArtifactIdentity foreignIdentity = artifact(99);

  {
    TestCase testCase = makeValidCase();
    ValidatedTechMapping mapping = validateCase(__func__, testCase);
    ResolvedPnrConfigView config;
    const ArtifactIdentity mappingDataflowIdentity =
        mapping.header().dataflowIdentity;
    testCase.dataflow.identity = foreignIdentity;
    PnrProblemInputs inputs = makeProblemInputs(testCase, mapping, config);
    expectProblemInputError(
        __func__, validatePnrProblemInputs(inputs),
        PnrProblemInputErrorCode::TechMappingDataflowIdentityMismatch,
        foreignIdentity, mappingDataflowIdentity);
    expectFreezeInputError(
        __func__, freezeRealizationGraph(inputs),
        PnrProblemInputErrorCode::TechMappingDataflowIdentityMismatch,
        foreignIdentity, mappingDataflowIdentity);
  }

  {
    TestCase testCase = makeValidCase();
    ValidatedTechMapping mapping = validateCase(__func__, testCase);
    ResolvedPnrConfigView config;
    const ArtifactIdentity mappingFabricIdentity =
        mapping.header().fabricIdentity;
    testCase.fabric.identity = foreignIdentity;
    PnrProblemInputs inputs = makeProblemInputs(testCase, mapping, config);
    expectProblemInputError(
        __func__, validatePnrProblemInputs(inputs),
        PnrProblemInputErrorCode::TechMappingFabricIdentityMismatch,
        foreignIdentity, mappingFabricIdentity);
    expectFreezeInputError(
        __func__, freezeRoutingGraph(inputs),
        PnrProblemInputErrorCode::TechMappingFabricIdentityMismatch,
        foreignIdentity, mappingFabricIdentity);
  }

  TestCase testCase = makeValidCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config;
  PnrProblemInputs inputs = makeProblemInputs(testCase, mapping, config);

  PnrProblemInputs mismatchedDataflow = inputs;
  mismatchedDataflow.constraints.dataflowIdentity = foreignIdentity;
  expectProblemInputError(
      __func__, validatePnrProblemInputs(mismatchedDataflow),
      PnrProblemInputErrorCode::ConstraintSetDataflowIdentityMismatch,
      testCase.dataflow.identity, foreignIdentity);
  expectFreezeInputError(
      __func__, freezeRealizationGraph(mismatchedDataflow),
      PnrProblemInputErrorCode::ConstraintSetDataflowIdentityMismatch,
      testCase.dataflow.identity, foreignIdentity);

  PnrProblemInputs mismatchedMapping = inputs;
  mismatchedMapping.constraints.techMappingIdentity = foreignIdentity;
  expectProblemInputError(
      __func__, validatePnrProblemInputs(mismatchedMapping),
      PnrProblemInputErrorCode::ConstraintSetTechMappingIdentityMismatch,
      inputs.techMappingIdentity, foreignIdentity);
  expectFreezeInputError(
      __func__, freezeRoutingGraph(mismatchedMapping),
      PnrProblemInputErrorCode::ConstraintSetTechMappingIdentityMismatch,
      inputs.techMappingIdentity, foreignIdentity);

  PnrProblemInputs mismatchedFabric = inputs;
  mismatchedFabric.constraints.fabricIdentity = foreignIdentity;
  expectProblemInputError(
      __func__, validatePnrProblemInputs(mismatchedFabric),
      PnrProblemInputErrorCode::ConstraintSetFabricIdentityMismatch,
      testCase.fabric.identity, foreignIdentity);
  expectFreezeInputError(
      __func__, freezeRealizationGraph(mismatchedFabric),
      PnrProblemInputErrorCode::ConstraintSetFabricIdentityMismatch,
      testCase.fabric.identity, foreignIdentity);
}

void keepsConfigAndConstraintArtifactIdentitiesIndependent() {
  TestCase testCase = makeValidCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView firstConfig;
  ResolvedPnrConfigView secondConfig;
  PnrProblemInputs first = makeProblemInputs(
      testCase, mapping, firstConfig, artifact(3), artifact(4), artifact(5));
  PnrProblemInputs second = makeProblemInputs(
      testCase, mapping, secondConfig, artifact(3), artifact(40), artifact(50));

  if (first.resolvedConfigIdentity == second.resolvedConfigIdentity ||
      first.constraints.identity == second.constraints.identity)
    fail(__func__, "config or constraint identity was not independently bound");
  if (llvm::Error error = validatePnrProblemInputs(first))
    fail(__func__, llvm::toString(std::move(error)).c_str());
  if (llvm::Error error = validatePnrProblemInputs(second))
    fail(__func__, llvm::toString(std::move(error)).c_str());
}

} // namespace

void runPnrProblemInputsTests() {
  exactFiveInputBindingReachesExistingFreezeBehavior();
  rejectsEachExactCouplingMismatch();
  keepsConfigAndConstraintArtifactIdentitiesIndependent();
}

} // namespace loom::mapping::test

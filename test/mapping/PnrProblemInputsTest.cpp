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

template <typename T, typename = void>
struct HasDetachedTechMappingIdentityMember : std::false_type {};

template <typename T>
struct HasDetachedTechMappingIdentityMember<
    T, std::void_t<decltype(std::declval<T>().techMappingIdentity)>>
    : std::true_type {};

template <typename T> struct OwningInputProxy {
  T value;

  operator const T &() const { return value; }
};

template <typename Dataflow, typename TechMapping, typename Fabric,
          typename Config>
inline constexpr bool canConstructPnrProblemInputs =
    std::is_constructible_v<PnrProblemInputs, Dataflow, TechMapping, Fabric,
                            Config, ArtifactIdentity,
                            MappingConstraintSetInput>;

static_assert(!HasArtifactIdentityMember<PnrProblemInputs>::value);
static_assert(!HasArtifactIdentityMember<ResolvedPnrConfigView>::value);
static_assert(!HasDetachedTechMappingIdentityMember<PnrProblemInputs>::value);
static_assert(!std::is_default_constructible_v<MappingConstraintSetInput>);
static_assert(std::is_copy_constructible_v<ValidatedTechMapping>);
static_assert(std::is_move_constructible_v<ValidatedTechMapping>);
static_assert(!std::is_copy_assignable_v<ValidatedTechMapping>);
static_assert(!std::is_move_assignable_v<ValidatedTechMapping>);

static_assert(canConstructPnrProblemInputs<
              const DataflowProgramView &, const ValidatedTechMapping &,
              const FabricHardwareView &, const ResolvedPnrConfigView &>);
static_assert(!canConstructPnrProblemInputs<
              DataflowProgramView &&, const ValidatedTechMapping &,
              const FabricHardwareView &, const ResolvedPnrConfigView &>);
static_assert(!canConstructPnrProblemInputs<
              const DataflowProgramView &, ValidatedTechMapping &&,
              const FabricHardwareView &, const ResolvedPnrConfigView &>);
static_assert(!canConstructPnrProblemInputs<
              const DataflowProgramView &, const ValidatedTechMapping &,
              FabricHardwareView &&, const ResolvedPnrConfigView &>);
static_assert(!canConstructPnrProblemInputs<
              const DataflowProgramView &, const ValidatedTechMapping &,
              const FabricHardwareView &, ResolvedPnrConfigView &&>);
static_assert(!canConstructPnrProblemInputs<
              const DataflowProgramView &&, const ValidatedTechMapping &,
              const FabricHardwareView &, const ResolvedPnrConfigView &>);
static_assert(!canConstructPnrProblemInputs<
              const DataflowProgramView &, const ValidatedTechMapping &&,
              const FabricHardwareView &, const ResolvedPnrConfigView &>);
static_assert(!canConstructPnrProblemInputs<
              const DataflowProgramView &, const ValidatedTechMapping &,
              const FabricHardwareView &&, const ResolvedPnrConfigView &>);
static_assert(!canConstructPnrProblemInputs<
              const DataflowProgramView &, const ValidatedTechMapping &,
              const FabricHardwareView &, const ResolvedPnrConfigView &&>);
static_assert(
    !canConstructPnrProblemInputs<
        OwningInputProxy<DataflowProgramView> &&, const ValidatedTechMapping &,
        const FabricHardwareView &, const ResolvedPnrConfigView &>);
static_assert(
    !canConstructPnrProblemInputs<
        const DataflowProgramView &, OwningInputProxy<ValidatedTechMapping> &&,
        const FabricHardwareView &, const ResolvedPnrConfigView &>);
static_assert(!canConstructPnrProblemInputs<
              const DataflowProgramView &, const ValidatedTechMapping &,
              OwningInputProxy<FabricHardwareView> &&,
              const ResolvedPnrConfigView &>);
static_assert(!canConstructPnrProblemInputs<
              const DataflowProgramView &, const ValidatedTechMapping &,
              const FabricHardwareView &,
              OwningInputProxy<ResolvedPnrConfigView> &&>);

PnrProblemInputs
makeProblemInputs(TestCase &testCase, ValidatedTechMapping &mapping,
                  ResolvedPnrConfigView &config,
                  ArtifactIdentity resolvedConfigIdentity = artifact(4),
                  ArtifactIdentity constraintSetIdentity = artifact(5)) {
  return PnrProblemInputs{testCase.dataflow,
                          mapping,
                          testCase.fabric,
                          config,
                          resolvedConfigIdentity,
                          MappingConstraintSetInput{
                              constraintSetIdentity, testCase.dataflow.identity,
                              mapping.identity(), testCase.fabric.identity}};
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
      inputs.techMapping.identity(), foreignIdentity);
  expectFreezeInputError(
      __func__, freezeRoutingGraph(mismatchedMapping),
      PnrProblemInputErrorCode::ConstraintSetTechMappingIdentityMismatch,
      inputs.techMapping.identity(), foreignIdentity);

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
  PnrProblemInputs first = makeProblemInputs(testCase, mapping, firstConfig,
                                             artifact(4), artifact(5));
  PnrProblemInputs second = makeProblemInputs(testCase, mapping, secondConfig,
                                              artifact(40), artifact(50));

  if (first.resolvedConfigIdentity == second.resolvedConfigIdentity ||
      first.constraints.identity == second.constraints.identity)
    fail(__func__, "config or constraint identity was not independently bound");
  if (llvm::Error error = validatePnrProblemInputs(first))
    fail(__func__, llvm::toString(std::move(error)).c_str());
  if (llvm::Error error = validatePnrProblemInputs(second))
    fail(__func__, llvm::toString(std::move(error)).c_str());
}

void bindsTechMappingIdentityAtValidationBoundary() {
  TestCase testCase = makeValidCase();
  const ArtifactIdentity firstIdentity = artifact(60);
  const ArtifactIdentity secondIdentity = artifact(61);
  ValidatedTechMapping first = takeExpected(
      __func__, validateTechMapping(firstIdentity, testCase.mapping,
                                    testCase.dataflow, testCase.fabric));
  ResolvedPnrConfigView config;
  PnrProblemInputs firstInputs = makeProblemInputs(testCase, first, config);

  PnrProblemInputs relabeledConstraints = firstInputs;
  relabeledConstraints.constraints.techMappingIdentity = secondIdentity;
  expectProblemInputError(
      __func__, validatePnrProblemInputs(relabeledConstraints),
      PnrProblemInputErrorCode::ConstraintSetTechMappingIdentityMismatch,
      firstIdentity, secondIdentity);

  ValidatedTechMapping second = takeExpected(
      __func__, validateTechMapping(secondIdentity, testCase.mapping,
                                    testCase.dataflow, testCase.fabric));
  PnrProblemInputs secondInputs = makeProblemInputs(testCase, second, config);
  if (first.identity() != firstIdentity ||
      second.identity() != secondIdentity ||
      first.identity() == second.identity())
    fail(__func__, "validated mappings did not retain their imported identity");
  if (llvm::Error error = validatePnrProblemInputs(secondInputs))
    fail(__func__, llvm::toString(std::move(error)).c_str());
}

void keepsBorrowedMappingUsableAfterRvalueConstruction() {
  TestCase testCase = makeValidCase();
  ValidatedTechMapping mapping = validateCase(__func__, testCase);
  ResolvedPnrConfigView config;
  PnrProblemInputs inputs = makeProblemInputs(testCase, mapping, config);

  ValidatedTechMapping copiedFromRvalue(std::move(mapping));
  if (copiedFromRvalue.identity() != testCase.techMappingIdentity)
    fail(__func__, "rvalue construction changed the mapping identity");
  if (llvm::Error error = validatePnrProblemInputs(inputs))
    fail(__func__, llvm::toString(std::move(error)).c_str());

  FrozenRealizationGraph realizations =
      takeExpected(__func__, freezeRealizationGraph(inputs));
  FrozenRoutingGraph routing =
      takeExpected(__func__, freezeRoutingGraph(inputs));
  if (realizations.computeRealizations().empty() ||
      routing.computeEndpointVertices().empty())
    fail(__func__, "rvalue construction invalidated a borrowed mapping");
}

} // namespace

void runPnrProblemInputsTests() {
  exactFiveInputBindingReachesExistingFreezeBehavior();
  rejectsEachExactCouplingMismatch();
  keepsConfigAndConstraintArtifactIdentitiesIndependent();
  bindsTechMappingIdentityAtValidationBoundary();
  keepsBorrowedMappingUsableAfterRvalueConstruction();
}

} // namespace loom::mapping::test

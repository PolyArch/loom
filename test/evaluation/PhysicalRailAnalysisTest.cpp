#include "Evaluation/Models/PhysicalRailAnalysis.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/ComponentViewDigest.h"
#include "Config/ResolvedConfig.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::evaluation;
using namespace loom::evaluation::models;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  std::cerr << test.str() << ": " << message << '\n';
  std::exit(1);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectErrorContains(llvm::StringRef test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

ArtifactRootReference
publishHardwareImplementation(const ArtifactStore &store) {
  const std::string body = "physical-rail-analysis-fixture";
  auto identity =
      take(__func__, store.put(hardware::hardwareImplementationSchema,
                               CanonicalSemanticBytes(std::vector<std::uint8_t>(
                                   body.begin(), body.end()))));
  return ArtifactRootReference{
      hardware::hardwareImplementationSchema.identity.str(),
      hardware::hardwareImplementationSchema.version, std::move(identity)};
}

SubjectTargetRef rootTarget(const ArtifactRootReference &hardware) {
  return SubjectTargetRef{hardwareImplementationPhysicalSubjectRole(), hardware,
                          SubjectTarget{hardware}};
}

struct Fixture final {
  platform::FinalizedImplementationPlatform platform;
  ArtifactRootReference hardware;
  CaseArtifactResolution resolution;
};

Fixture makeFixture(const ArtifactStore &store) {
  auto platform = take(
      __func__,
      platform::finalizeImplementationPlatform(
          {platform::AsicTarget{"saed32", "EDK_08_2025"}, {"typical"}}, store));
  ArtifactRootReference hardware = publishHardwareImplementation(store);
  CaseArtifactResolution resolution = take(
      __func__, CaseArtifactResolution::get({{hardware, {platform.reference()}},
                                             {platform.reference(), {}}}));
  return Fixture{std::move(platform), std::move(hardware),
                 std::move(resolution)};
}

std::vector<EvaluationCondition>
conditions(const Fixture &fixture, platform::TechnologyCornerId corner) {
  const SubjectTargetRef target = rootTarget(fixture.hardware);
  return {
      EvaluationCondition{ProcessCornerCondition{
          target,
          platform::TechnologyCornerRef{fixture.platform.reference().artifact,
                                        corner}}},
      EvaluationCondition{SupplyVoltageCondition{
          target, take(__func__, DecimalValue::get(9, -1))}},
      EvaluationCondition{TemperatureCondition{
          target, take(__func__, DecimalValue::get(3, 2))}},
      EvaluationCondition{RequiredClockPeriodCondition{
          target, take(__func__, DecimalValue::get(2, -9))}},
      EvaluationCondition{ActivityBindingCondition{
          target,
          ExplicitAssumptionSource{target,
                                   take(__func__, ExactRatio::get(1, 2)),
                                   take(__func__, ExactRatio::get(1, 10))}}},
  };
}

EvaluationSubjectBindings bindings(const ArtifactRootReference &hardware) {
  return take(__func__,
              EvaluationSubjectBindings::get(
                  {{hardwareImplementationPhysicalSubjectRole(), {hardware}}}));
}

ExternalFileFingerprint fingerprint(std::uint8_t seed) {
  ExternalFileFingerprint::Storage bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<std::uint8_t>(seed + index);
  return take(__func__, ExternalFileFingerprint::fromBytes(bytes));
}

CadenceVoltusStaticRailProviderBinding providerBinding() {
  return {"Voltus 26.10-p001_1",
          {{"libraries/standard_cells.cl", fingerprint(1)},
           {"technology.cl", fingerprint(2)}},
          {"technology.cl", "libraries/standard_cells.cl"}};
}

ResolvedConfig railResolvedConfig() {
  ResolvedConfig config = defaultResolvedConfig();
  config.evaluation.cadenceVoltusStaticRail = providerBinding();
  return config;
}

void exactRequestProjectsOneCompleteConfiguration(const ArtifactStore &store,
                                                  const BlobStore &blobs) {
  const llvm::StringRef test = __func__;
  const Fixture fixture = makeFixture(store);
  if (llvm::Error error = registerCadenceVoltusStaticRailModel())
    fail(test, llvm::toString(std::move(error)));

  const EvaluationCase evaluationCase =
      take(test, EvaluationCase::get(
                     cadenceVoltusStaticRailModelDescriptorRef()
                         .descriptor()
                         ->caseSignature,
                     bindings(fixture.hardware), std::nullopt, std::nullopt,
                     conditions(fixture, platform::TechnologyCornerId(0)),
                     fixture.resolution, store, blobs));
  const MetricRequest metric =
      take(test,
           MetricRequest::get(MetricQuery{MetricKind::MaximumVoltageDrop,
                                          EvaluationScope{ScopeFormRef(0), {}}},
                              {}, evaluationCase, fixture.resolution, store));
  const ResolvedConfig resolvedConfig = railResolvedConfig();
  const std::string canonicalConfig =
      canonicalResolvedConfigJson(resolvedConfig);
  const ResolvedConfig reparsed = take(
      test, parseResolvedConfig(canonicalConfig, "rail-config-round-trip"));
  require(test,
          reparsed.evaluation.cadenceVoltusStaticRail ==
              resolvedConfig.evaluation.cadenceVoltusStaticRail,
          "ResolvedConfig round-trip changed the Voltus provider binding");

  ResolvedModelBinding modelBinding = take(
      test, ResolvedModelBinding::project(
                cadenceVoltusStaticRailModelDescriptorRef(), {}, reparsed));
  require(test,
          !modelBinding.resolvedModelConfig().canonicalViewBytes().empty(),
          "rail model config omitted the exact provider binding");

  const EvaluationRequest request =
      take(test, EvaluationRequest::get(evaluationCase, {metric}, {},
                                        std::move(modelBinding), 0,
                                        fixture.resolution, store, blobs));
  const CompleteRailAnalysisConfiguration projected =
      take(test, projectCompleteRailAnalysisConfiguration(
                     request, fixture.resolution, store, blobs));

  require(test, projected.model == staticExplicitRailAnalysisModelConfig(),
          "projection changed the descriptor-owned rail model config");
  require(test, projected.providerBinding == providerBinding(),
          "projection changed the exact Voltus provider binding");
  require(test,
          projected.providerBinding.powerGridLibraryEntrypoints ==
              std::vector<std::string>(
                  {"technology.cl", "libraries/standard_cells.cl"}),
          "projection changed the ordered PGV entrypoints");
  require(
      test,
      projected.processCorner.corner ==
          platform::TechnologyCornerRef{fixture.platform.reference().artifact,
                                        platform::TechnologyCornerId(0)},
      "projection changed the exact technology corner");
  require(test,
          projected.supplyVoltage.volts == take(test, DecimalValue::get(9, -1)),
          "projection changed the exact supply voltage");
  require(test,
          projected.temperature.kelvin == take(test, DecimalValue::get(3, 2)),
          "projection changed the exact temperature");
  require(test,
          projected.clockPeriod.seconds == take(test, DecimalValue::get(2, -9)),
          "projection changed the exact activity clock period");
  require(test,
          projected.activity.assumption.staticProbability ==
                  take(test, ExactRatio::get(1, 2)) &&
              projected.activity.assumption.transitionsPerClock ==
                  take(test, ExactRatio::get(1, 10)),
          "projection changed the explicit activity assumption");
}

void ownerAndConfigBoundariesRejectInvalidInputs(const ArtifactStore &store,
                                                 const BlobStore &blobs) {
  const llvm::StringRef test = __func__;
  const Fixture fixture = makeFixture(store);
  if (llvm::Error error = registerCadenceVoltusStaticRailModel())
    fail(test, llvm::toString(std::move(error)));

  expectErrorContains(
      test,
      EvaluationCase::get(cadenceVoltusStaticRailModelDescriptorRef()
                              .descriptor()
                              ->caseSignature,
                          bindings(fixture.hardware), std::nullopt,
                          std::nullopt,
                          conditions(fixture, platform::TechnologyCornerId(1)),
                          fixture.resolution, store, blobs),
      "out of range");

  expectErrorContains(
      test,
      ResolvedModelBinding::project(cadenceVoltusStaticRailModelDescriptorRef(),
                                    {}, defaultResolvedConfig()),
      "provider binding is unavailable");

  ResolvedConfig invalidBuild = railResolvedConfig();
  invalidBuild.evaluation.cadenceVoltusStaticRail->stableProviderBuildIdentity =
      " Voltus 26.10-p001_1";
  expectErrorContains(
      test,
      ResolvedModelBinding::project(cadenceVoltusStaticRailModelDescriptorRef(),
                                    {}, invalidBuild),
      "one normalized line");

  ResolvedConfig unsortedMembers = railResolvedConfig();
  std::swap(unsortedMembers.evaluation.cadenceVoltusStaticRail
                ->powerGridLibraryMembers[0],
            unsortedMembers.evaluation.cadenceVoltusStaticRail
                ->powerGridLibraryMembers[1]);
  expectErrorContains(
      test,
      ResolvedModelBinding::project(cadenceVoltusStaticRailModelDescriptorRef(),
                                    {}, unsortedMembers),
      "sorted by relative path");

  ResolvedConfig missingEntrypoints = railResolvedConfig();
  missingEntrypoints.evaluation.cadenceVoltusStaticRail
      ->powerGridLibraryEntrypoints.clear();
  expectErrorContains(
      test,
      ResolvedModelBinding::project(cadenceVoltusStaticRailModelDescriptorRef(),
                                    {}, missingEntrypoints),
      "entrypoint catalog is empty");

  ResolvedConfig duplicateEntrypoints = railResolvedConfig();
  duplicateEntrypoints.evaluation.cadenceVoltusStaticRail
      ->powerGridLibraryEntrypoints = {"technology.cl", "technology.cl"};
  expectErrorContains(
      test,
      ResolvedModelBinding::project(cadenceVoltusStaticRailModelDescriptorRef(),
                                    {}, duplicateEntrypoints),
      "entrypoint catalog contains a duplicate");

  ResolvedConfig foreignEntrypoint = railResolvedConfig();
  foreignEntrypoint.evaluation.cadenceVoltusStaticRail
      ->powerGridLibraryEntrypoints = {"missing.cl"};
  expectErrorContains(
      test,
      ResolvedModelBinding::project(cadenceVoltusStaticRailModelDescriptorRef(),
                                    {}, foreignEntrypoint),
      "entrypoint is absent from the member table");

  const EvaluationModelDescriptor *descriptor =
      cadenceVoltusStaticRailModelDescriptorRef().descriptor();
  require(test, descriptor, "rail model descriptor was not registered");
  ResolvedModelBinding valid = take(
      test,
      ResolvedModelBinding::project(cadenceVoltusStaticRailModelDescriptorRef(),
                                    {}, railResolvedConfig()));
  std::vector<std::uint8_t> noncanonicalConfig(
      valid.resolvedModelConfig().canonicalViewBytes().begin(),
      valid.resolvedModelConfig().canonicalViewBytes().end());
  noncanonicalConfig.push_back(0);
  auto digest =
      take(test, computeComponentViewDigest(
                     descriptor->resolvedConfigView.schemaDescriptorBytes,
                     noncanonicalConfig));
  expectErrorContains(
      test,
      ResolvedModelBinding::adopt(cadenceVoltusStaticRailModelDescriptorRef(),
                                  {}, std::move(noncanonicalConfig), digest),
      "trailing bytes");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one artifact-store root");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  const ArtifactStore store(root.string());
  const BlobStore blobs(root.string());
  exactRequestProjectsOneCompleteConfiguration(store, blobs);
  ownerAndConfigBoundariesRejectInvalidInputs(store, blobs);
  return 0;
}

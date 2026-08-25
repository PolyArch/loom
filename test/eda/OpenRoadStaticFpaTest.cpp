#include "OpenRoadPhysicalTestSupport.h"

#include "Common/ArtifactText.h"
#include "Config/ResolvedConfig.h"
#include "EDA/Adapters/OpenSource/OpenRoadStaticFpa.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/PhysicalRailAnalysis.h"
#include "ExternalTool/InvocationBundle.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::eda::open_source;
using namespace loom::eda::open_source::test;
using namespace loom::evaluation;
using namespace loom::evaluation::models;
using namespace loom::external_tool;
using namespace loom::hardware;

namespace {

constexpr llvm::StringLiteral kSyntheticBuild =
    "OpenROAD synthetic 21512b0ab68c";
constexpr llvm::StringLiteral kPinnedBuild = "21512b0";
constexpr std::array<MetricKind, 4> kAllFpaMetrics{
    MetricKind::LeakagePower, MetricKind::LimitingClockFrequency,
    MetricKind::DynamicPower, MetricKind::TotalArea};

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
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

void requireSuccess(llvm::StringRef test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

template <typename T>
void expectErrorContains(llvm::StringRef test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

SubjectTargetRef rootTarget(const ArtifactRootReference &hardware) {
  return {hardwareImplementationPhysicalSubjectRole(), hardware,
          SubjectTarget{hardware}};
}

struct RequestFixture final {
  EvaluationRequest request;
  CaseArtifactResolution resolution;
};

RequestFixture
makeRequest(const FinalizedHardwareImplementation &hardware,
            const platform::FinalizedImplementationPlatform &platform,
            llvm::StringRef providerBuild, const ArtifactStore &artifacts,
            const BlobStore &blobs, std::int64_t supplyVoltageMillivolts = 1050,
            llvm::ArrayRef<MetricKind> metricKinds = kAllFpaMetrics,
            bool includeActivity = true) {
  const ArtifactRootReference hardwareRef = hardware.reference();
  CaseArtifactResolution resolution =
      take(__func__,
           CaseArtifactResolution::get({{hardwareRef, {platform.reference()}},
                                        {platform.reference(), {}}}));
  const SubjectTargetRef target = rootTarget(hardwareRef);
  const EvaluationSubjectBindings subjects =
      take(__func__,
           EvaluationSubjectBindings::get(
               {{hardwareImplementationPhysicalSubjectRole(), {hardwareRef}}}));
  std::vector<EvaluationCondition> conditions{
      EvaluationCondition{ProcessCornerCondition{
          target,
          {platform.reference().artifact, platform::TechnologyCornerId(0)}}},
      EvaluationCondition{SupplyVoltageCondition{
          target,
          take(__func__, DecimalValue::get(supplyVoltageMillivolts, -3))}},
      EvaluationCondition{TemperatureCondition{
          target, take(__func__, DecimalValue::get(3, 2))}},
      EvaluationCondition{RequiredClockPeriodCondition{
          target, take(__func__, DecimalValue::get(2, -9))}}};
  if (includeActivity)
    conditions.push_back(EvaluationCondition{ActivityBindingCondition{
        target,
        ExplicitAssumptionSource{target, take(__func__, ExactRatio::get(1, 2)),
                                 take(__func__, ExactRatio::get(1, 10))}}});
  const EvaluationCase evaluationCase = take(
      __func__,
      EvaluationCase::get(
          openRoadStaticFpaModelDescriptorRef().descriptor()->caseSignature,
          subjects, std::nullopt, std::nullopt, conditions, resolution,
          artifacts, blobs));
  std::vector<MetricRequest> metrics;
  metrics.reserve(metricKinds.size());
  for (MetricKind kind : metricKinds)
    metrics.push_back(
        take(__func__,
             MetricRequest::get({kind, EvaluationScope{ScopeFormRef(0), {}}},
                                {}, evaluationCase, resolution, artifacts)));

  ResolvedConfig config = defaultResolvedConfig();
  config.evaluation.openRoadStaticFpa =
      OpenRoadStaticFpaProviderBinding{providerBuild.str()};
  const std::string canonical = canonicalResolvedConfigJson(config);
  const ResolvedConfig reparsed =
      take(__func__, parseResolvedConfig(canonical, "openroad-fpa-config"));
  require(__func__,
          reparsed.evaluation.openRoadStaticFpa ==
              config.evaluation.openRoadStaticFpa,
          "ResolvedConfig round-trip changed the OpenROAD FPA binding");
  ResolvedModelBinding model =
      take(__func__, ResolvedModelBinding::project(
                         openRoadStaticFpaModelDescriptorRef(), {}, reparsed));
  EvaluationRequest request =
      take(__func__, EvaluationRequest::get(evaluationCase, std::move(metrics),
                                            {}, std::move(model), 0, resolution,
                                            artifacts, blobs));
  const ArtifactRootReference published =
      take(__func__, publishEvaluationRequest(request, artifacts));
  require(__func__, published == evaluationRequestReference(request),
          "request publication changed identity");
  return {std::move(request), std::move(resolution)};
}

std::filesystem::path
writeAuthoredGpdkRouteTool(const std::filesystem::path &root) {
  const std::filesystem::path tool = root / "authored-gpdk-openroad-route";
  const std::string body = R"sh(#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-version" || "${1:-}" == "--version" ]]; then
  printf '%s\n' '21512b0'
  exit 0
fi
if [[ "$#" -ne 7 || "$1" != "-no_init" || "$2" != "-no_splash" ||
      "$3" != "-no_settings" || "$4" != "-threads" || "$5" != "1" ||
      "$6" != "-exit" || "$7" != "drivers/openroad-routed.tcl" ]]; then
  exit 64
fi
grep -F 'detailed_route -or_seed 1' drivers/openroad-routed.tcl >/dev/null
grep -F 'module top' inputs/netlist/0000.v >/dev/null
grep -F 'DFFX1 launch' inputs/netlist/0000.v >/dev/null
grep -F 'INVX1 combinational' inputs/netlist/0000.v >/dev/null
grep -F 'create_clock -name core_clock -period 2' inputs/constraints/0000.sdc >/dev/null
mkdir -p outputs work
cp inputs/netlist/0000.v outputs/routed.v
cat > outputs/routed.def <<'EOF'
VERSION 5.8 ;
DIVIDERCHAR "/" ;
BUSBITCHARS "[]" ;
DESIGN top ;
UNITS DISTANCE MICRONS 2000 ;
DIEAREA ( 0 0 ) ( 200000 200000 ) ;
COMPONENTS 3 ;
- launch DFFX1 + PLACED ( 40000 40000 ) N ;
- combinational INVX1 + PLACED ( 60000 40000 ) N ;
- capture DFFX1 + PLACED ( 80000 40000 ) N ;
END COMPONENTS
PINS 3 ;
- clk + NET clk + DIRECTION INPUT + USE CLOCK
  + LAYER Metal3 ( -100 -100 ) ( 100 100 ) + FIXED ( 20000 40000 ) N ;
- d + NET d + DIRECTION INPUT + USE SIGNAL
  + LAYER Metal3 ( -100 -100 ) ( 100 100 ) + FIXED ( 20000 60000 ) N ;
- q + NET q + DIRECTION OUTPUT + USE SIGNAL
  + LAYER Metal3 ( -100 -100 ) ( 100 100 ) + FIXED ( 180000 60000 ) N ;
END PINS
NETS 5 ;
- clk ( PIN clk ) ( launch CK ) ( capture CK )
  + ROUTED Metal3 ( 20000 40000 ) ( 80000 40000 ) ;
- d ( PIN d ) ( launch D )
  + ROUTED Metal3 ( 20000 60000 ) ( 40000 60000 ) ;
- launched ( launch Q ) ( combinational A )
  + ROUTED Metal3 ( 48000 40000 ) ( 60000 40000 ) ;
- inverted ( combinational Y ) ( capture D )
  + ROUTED Metal3 ( 60800 40000 ) ( 80000 40000 ) ;
- q ( capture Q ) ( PIN q )
  + ROUTED Metal3 ( 88000 40000 ) ( 180000 40000 ) ( 180000 60000 ) ;
END NETS
END DESIGN
EOF
sed -n '/^set loom_result /,$p' drivers/openroad-routed.tcl > work/publish-result.tcl
tclsh work/publish-result.tcl
)sh";
  requireSuccess(__func__, writeText(tool, body, true));
  return tool;
}

FinalizedHardwareImplementation makeAuthoredRoutedImplementation(
    const std::filesystem::path &root, const OpenRoadGateFixture &fixture,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  requireSuccess(__func__,
                 registerOpenRoadRoutedCandidateGeneratorDescriptor());
  const std::filesystem::path routeTool =
      take(__func__, writeAuthoredOpenRoadRouteTool(root));
  const LocalToolConfig local = makeOpenRoadLocalToolConfig(fixture, routeTool);
  OpenRoadRouteHarness harness =
      take(__func__,
           makeOpenRoadRouteHarness(root / "routed-himpl", fixture, local));
  return take(__func__, runOpenRoadRouteFixture(
                            fixture, harness,
                            makeOpenRoadResolvedExecution(
                                routeTool.string(), kSyntheticBuild, false),
                            artifacts, blobs));
}

EvaluationModelPreparation
prepareAt(const RequestFixture &request, const std::filesystem::path &root,
          llvm::StringRef bundleName, const LocalToolConfig &local,
          const ArtifactStore &artifacts, const BlobStore &blobs) {
  return take(__func__,
              prepareEvaluationModelInvocation(
                  request.request, request.resolution, artifacts, blobs,
                  {local, (root / bundleName.str()).string()}));
}

void requireFailureOutcome(llvm::StringRef test,
                           const EvaluationEvidence &evidence,
                           OutcomeReason reason) {
  const auto *failed =
      std::get_if<ExecutionFailedEvidence>(&evidence.outcome());
  require(test, failed && failed->reason == reason,
          "Evidence carried the wrong execution-failure reason");
}

void resultParserAndDriverAreStrict() {
  const std::string canonical =
      "{\"schema\":\"loom.openroad_static_fpa_result\","
      "\"version\":\"1.0\",\"top\":\"top\","
      "\"limiting_clock_frequency\":{\"value\":\"5.0e+08\","
      "\"unit\":\"hertz\"},"
      "\"total_area\":{\"value\":\"1.2e-10\","
      "\"unit\":\"square_meter\"},"
      "\"dynamic_power\":{\"value\":\"3.4e-03\","
      "\"unit\":\"watt\"},"
      "\"leakage_power\":{\"value\":\"5.6e-04\","
      "\"unit\":\"watt\"}}\n";
  const OpenRoadStaticFpaObservation observation = take(
      __func__, parseOpenRoadStaticFpaResult(canonical, "top", kAllFpaMetrics));
  require(__func__,
          observation.limitingClockFrequencyHertz ==
                  take(__func__, DecimalValue::get(5, 8)) &&
              observation.totalAreaSquareMeters ==
                  take(__func__, DecimalValue::get(12, -11)) &&
              observation.dynamicPowerWatts ==
                  take(__func__, DecimalValue::get(34, -4)) &&
              observation.leakagePowerWatts ==
                  take(__func__, DecimalValue::get(56, -5)),
          "strict parser changed normalized FPA values");

  std::string wrongTop = canonical;
  wrongTop.replace(wrongTop.find("\"top\":\"top\""), 11, "\"top\":\"other\"");
  expectErrorContains(
      __func__, parseOpenRoadStaticFpaResult(wrongTop, "top", kAllFpaMetrics),
      "top");
  std::string wrongUnit = canonical;
  wrongUnit.replace(wrongUnit.find("\"unit\":\"watt\""), 13,
                    "\"unit\":\"volt\"");
  expectErrorContains(
      __func__, parseOpenRoadStaticFpaResult(wrongUnit, "top", kAllFpaMetrics),
      "unit");
  std::string nonfinite = canonical;
  nonfinite.replace(nonfinite.find("3.4e-03"), 7, "NaN");
  expectErrorContains(
      __func__, parseOpenRoadStaticFpaResult(nonfinite, "top", kAllFpaMetrics),
      "finite scientific decimal");
  std::string extra = canonical;
  extra.insert(extra.rfind('}'), ",\"claim\":\"signoff\"");
  expectErrorContains(
      __func__, parseOpenRoadStaticFpaResult(extra, "top", kAllFpaMetrics),
      "shape");

  constexpr std::array<MetricKind, 1> leakageMetric{MetricKind::LeakagePower};
  const std::string leakageOnly =
      "{\"schema\":\"loom.openroad_static_fpa_result\","
      "\"version\":\"1.0\",\"top\":\"top\","
      "\"leakage_power\":{\"value\":\"5.6e-04\","
      "\"unit\":\"watt\"}}\n";
  const OpenRoadStaticFpaObservation leakageObservation =
      take(__func__,
           parseOpenRoadStaticFpaResult(leakageOnly, "top", leakageMetric));
  require(__func__,
          leakageObservation.leakagePowerWatts ==
                  take(__func__, DecimalValue::get(56, -5)) &&
              !leakageObservation.dynamicPowerWatts &&
              !leakageObservation.totalAreaSquareMeters &&
              !leakageObservation.limitingClockFrequencyHertz,
          "strict parser invented an unrequested FPA metric");

  const OpenRoadTechnologyFixture technology =
      syntheticOpenRoadTechnologyFixture();
  const SubjectTargetRef target{
      hardwareImplementationPhysicalSubjectRole(),
      {hardwareImplementationSchema.identity.str(),
       hardwareImplementationSchema.version,
       take(__func__, parseArtifactIdentityHex(std::string(64, 'a')))},
      SubjectTarget{ArtifactRootReference{
          hardwareImplementationSchema.identity.str(),
          hardwareImplementationSchema.version,
          take(__func__, parseArtifactIdentityHex(std::string(64, 'a')))}}};
  const CompleteOpenRoadStaticFpaConfiguration analysis{
      {kSyntheticBuild.str()},
      ProcessCornerCondition{
          target,
          {take(__func__, parseArtifactIdentityHex(std::string(64, 'b'))),
           platform::TechnologyCornerId(0)}},
      SupplyVoltageCondition{target,
                             take(__func__, DecimalValue::get(105, -2))},
      TemperatureCondition{target, take(__func__, DecimalValue::get(3, 2))},
      RequiredClockPeriodCondition{target,
                                   take(__func__, DecimalValue::get(2, -9))},
      std::optional<ActivityBindingCondition>{ActivityBindingCondition{
          target,
          ExplicitAssumptionSource{target,
                                   take(__func__, ExactRatio::get(1, 2)),
                                   take(__func__, ExactRatio::get(1, 10))}}},
      std::optional<ExplicitAssumptionSource>{ExplicitAssumptionSource{
          target, take(__func__, ExactRatio::get(1, 2)),
          take(__func__, ExactRatio::get(1, 10))}},
      std::vector<MetricKind>(kAllFpaMetrics.begin(), kAllFpaMetrics.end())};
  const OpenRoadStaticFpaDriverConfiguration configuration{
      "top",
      {{"inputs/netlist/0.v"},
       {"inputs/constraints/0.sdc"},
       "inputs/database/routed.def",
       "/external/technology.lef",
       "/external/cells.lef",
       "/external/cells.lib"},
      analysis};
  const std::string driver =
      take(__func__, renderOpenRoadStaticFpaDriver(configuration));
  for (llvm::StringRef command :
       {"read_def \"inputs/database/routed.def\"", "set_pvt",
        "set_power_activity -global", "extract_parasitics -version 2.0 -lef_rc",
        "OpenRCX produced no parasitic segments", "sta::find_clk_min_period",
        "rsz::design_area", "sta::design_power",
        "source {drivers/openroad-static-fpa-publish.tcl}"})
    require(__func__, llvm::StringRef(driver).contains(command),
            "FPA driver omitted " + command.str());
  for (llvm::StringRef command :
       {"read_def -incremental", "read_verilog", "link_design"})
    require(__func__, !llvm::StringRef(driver).contains(command),
            "FPA driver reconstructs or drops routed physical state via " +
                command.str());
  require(__func__,
          take(__func__, renderOpenRoadStaticFpaPublisher(kAllFpaMetrics))
                  .find("outputs/openroad-static-fpa-result.json") !=
              std::string::npos,
          "shared publisher omitted the owner result path");
}

void authoredLifecycleSeparatesAllOutcomes(const std::filesystem::path &root) {
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
  const ArtifactStore artifacts((root / "artifacts").string());
  const BlobStore blobs((root / "blobs").string());
  const OpenRoadGateFixture fixture = take(
      __func__, makeOpenRoadGateFixture(root, artifacts, blobs, kSyntheticBuild,
                                        syntheticOpenRoadTechnologyFixture()));
  const FinalizedHardwareImplementation routed =
      makeAuthoredRoutedImplementation(root, fixture, artifacts, blobs);
  requireSuccess(__func__, registerOpenRoadStaticFpaEvaluationProvider());
  const RequestFixture request =
      makeRequest(routed, fixture.platform, kSyntheticBuild, artifacts, blobs);
  const std::filesystem::path tool =
      take(__func__, writeAuthoredOpenRoadStaticFpaTool(root));
  const LocalToolConfig local = makeOpenRoadLocalToolConfig(fixture, tool);

  EvaluationModelPreparation preparation =
      prepareAt(request, root, "fpa-complete", local, artifacts, blobs);
  const auto *prepared =
      std::get_if<PreparedExternalToolInvocation>(&preparation);
  require(__func__, prepared,
          "supported FPA request did not prepare an invocation");
  const std::string manifest =
      take(__func__, readText(root / "fpa-complete" / "tool-invocation.json"));
  require(__func__,
          llvm::StringRef(manifest).contains(
              "outputs/openroad-static-fpa-result.json") &&
              !llvm::StringRef(manifest).contains(
                  "openroad-static-fpa-raw.txt\"") &&
              !llvm::StringRef(manifest).contains("routed-result.json"),
          "FPA bundle declared a scratch or generation report as Evidence");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(*prepared)) == 0,
          "authored FPA invocation failed");
  const EvaluationEvidence evidence =
      take(__func__,
           importEvaluationModelInvocation(request.request, request.resolution,
                                           *prepared, artifacts, blobs));
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  require(__func__, completed && completed->metricResults.size() == 4,
          "FPA importer did not finalize four metric results");
  for (std::size_t index = 0; index < request.request.metricRequests().size();
       ++index) {
    const auto *point = std::get_if<PointObservation>(
        &completed->metricResults[index].observation);
    const auto *decimal =
        point ? std::get_if<DecimalValue>(&point->value) : nullptr;
    require(__func__, decimal,
            "FPA importer did not publish a Decimal Point observation");
    DecimalValue expected = take(__func__, DecimalValue::get(5, 8));
    switch (request.request.metricRequests()[index].query().metric) {
    case MetricKind::LimitingClockFrequency:
      expected = take(__func__, DecimalValue::get(5, 8));
      break;
    case MetricKind::TotalArea:
      expected = take(__func__, DecimalValue::get(12, -11));
      break;
    case MetricKind::DynamicPower:
      expected = take(__func__, DecimalValue::get(34, -4));
      break;
    case MetricKind::LeakagePower:
      expected = take(__func__, DecimalValue::get(56, -5));
      break;
    default:
      fail(__func__, "completed FPA request contains a foreign metric");
    }
    require(__func__, *decimal == expected,
            "FPA importer mapped a value to the wrong MetricKind");
  }

  constexpr std::array<MetricKind, 1> leakageMetric{MetricKind::LeakagePower};
  const RequestFixture leakageRequest =
      makeRequest(routed, fixture.platform, kSyntheticBuild, artifacts, blobs,
                  1050, leakageMetric, false);
  EvaluationModelPreparation leakagePreparation =
      prepareAt(leakageRequest, root, "fpa-leakage", local, artifacts, blobs);
  const auto *leakagePrepared =
      std::get_if<PreparedExternalToolInvocation>(&leakagePreparation);
  require(__func__, leakagePrepared,
          "leakage-only request without activity did not prepare");
  const std::string leakageDriver =
      take(__func__, readText(root / "fpa-leakage" / "drivers" /
                              "openroad-static-fpa.tcl"));
  require(
      __func__,
      !llvm::StringRef(leakageDriver).contains("set_power_activity") &&
          !llvm::StringRef(leakageDriver)
               .contains("dynamic_power_watts=%.12e") &&
          llvm::StringRef(leakageDriver).contains("leakage_power_watts=%.12e"),
      "leakage-only driver added hidden activity or dynamic power");
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(*leakagePrepared)) == 0,
          "authored leakage-only invocation failed");
  const EvaluationEvidence leakageEvidence =
      take(__func__, importEvaluationModelInvocation(
                         leakageRequest.request, leakageRequest.resolution,
                         *leakagePrepared, artifacts, blobs));
  const auto *leakageCompleted =
      std::get_if<CompletedEvidence>(&leakageEvidence.outcome());
  require(__func__,
          leakageCompleted && leakageCompleted->metricResults.size() == 1,
          "leakage-only importer did not preserve the requested metric subset");

  constexpr std::array<MetricKind, 1> dynamicMetric{MetricKind::DynamicPower};
  const RequestFixture missingActivityRequest =
      makeRequest(routed, fixture.platform, kSyntheticBuild, artifacts, blobs,
                  1050, dynamicMetric, false);
  const EvaluationModelPreparation missingActivityPreparation =
      prepareAt(missingActivityRequest, root, "fpa-dynamic-no-activity", local,
                artifacts, blobs);
  const auto *missingActivityEvidence =
      std::get_if<EvaluationEvidence>(&missingActivityPreparation);
  require(__func__,
          missingActivityEvidence &&
              std::holds_alternative<UnsupportedEvidence>(
                  missingActivityEvidence->outcome()),
          "dynamic power without activity did not become typed Unsupported");

  LocalToolConfig unavailable = local;
  unavailable.tools["openroad"].binding.executable =
      (root / "missing-openroad").string();
  EvaluationModelPreparation unavailablePreparation = prepareAt(
      request, root, "fpa-unavailable", unavailable, artifacts, blobs);
  const auto *unavailableEvidence =
      std::get_if<EvaluationEvidence>(&unavailablePreparation);
  require(__func__,
          unavailableEvidence &&
              std::holds_alternative<UnsupportedEvidence>(
                  unavailableEvidence->outcome()) &&
              std::get<UnsupportedEvidence>(unavailableEvidence->outcome())
                      .reason == OutcomeReason::RuntimeCapabilityUnavailable,
          "missing OpenROAD did not become typed Unsupported");

  const RequestFixture gateRequest = makeRequest(
      fixture.gate, fixture.platform, kSyntheticBuild, artifacts, blobs);
  EvaluationModelPreparation gatePreparation = prepareAt(
      gateRequest, root, "fpa-gate-unsupported", local, artifacts, blobs);
  const auto *gateEvidence = std::get_if<EvaluationEvidence>(&gatePreparation);
  require(__func__,
          gateEvidence && std::holds_alternative<UnsupportedEvidence>(
                              gateEvidence->outcome()),
          "non-routed HImpl did not remain explicitly unsupported");

  const std::filesystem::path failedTool =
      take(__func__, writeAuthoredOpenRoadStaticFpaTool(
                         root, AuthoredOpenRoadStaticFpaBehavior::ToolFailure));
  const LocalToolConfig failedLocal =
      makeOpenRoadLocalToolConfig(fixture, failedTool);
  EvaluationModelPreparation failedPreparation =
      prepareAt(request, root, "fpa-failed", failedLocal, artifacts, blobs);
  const auto *failedPrepared =
      std::get_if<PreparedExternalToolInvocation>(&failedPreparation);
  require(__func__, failedPrepared,
          "tool-failure request did not prepare an invocation");
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(*failedPrepared)) == 41,
          "authored FPA failure lost its exit status");
  requireFailureOutcome(__func__,
                        take(__func__, importEvaluationModelInvocation(
                                           request.request, request.resolution,
                                           *failedPrepared, artifacts, blobs)),
                        OutcomeReason::ToolFailure);

  const std::filesystem::path malformedTool = take(
      __func__, writeAuthoredOpenRoadStaticFpaTool(
                    root, AuthoredOpenRoadStaticFpaBehavior::MalformedResult));
  const LocalToolConfig malformedLocal =
      makeOpenRoadLocalToolConfig(fixture, malformedTool);
  EvaluationModelPreparation adapterPreparation =
      prepareAt(request, root, "fpa-adapter", malformedLocal, artifacts, blobs);
  const auto *adapterPrepared =
      std::get_if<PreparedExternalToolInvocation>(&adapterPreparation);
  require(__func__, adapterPrepared,
          "adapter-failure request did not prepare an invocation");
  require(__func__,
          take(__func__,
               executeExternalToolInvocationBundle(*adapterPrepared)) == 0,
          "adapter-failure fixture did not complete its tool attempt");
  requireFailureOutcome(__func__,
                        take(__func__, importEvaluationModelInvocation(
                                           request.request, request.resolution,
                                           *adapterPrepared, artifacts, blobs)),
                        OutcomeReason::AdapterFailure);
}

void evaluateRealOpenRoadFpa(const std::filesystem::path &root,
                             llvm::StringRef executable,
                             llvm::StringRef version,
                             const OpenRoadGateFixture &fixture,
                             const FinalizedHardwareImplementation &routed,
                             const ArtifactStore &artifacts,
                             const BlobStore &blobs) {
  requireSuccess(__func__, registerOpenRoadStaticFpaEvaluationProvider());
  const RequestFixture request =
      makeRequest(routed, fixture.platform, version, artifacts, blobs, 1000);
  const LocalToolConfig realLocal =
      makeOpenRoadLocalToolConfig(fixture, executable.str());
  EvaluationModelPreparation preparation =
      prepareAt(request, root, "fpa", realLocal, artifacts, blobs);
  const auto *prepared =
      std::get_if<PreparedExternalToolInvocation>(&preparation);
  require(__func__, prepared, "real FPA request did not prepare an invocation");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(*prepared)) == 0,
          "real OpenROAD FPA invocation did not complete");
  const EvaluationEvidence evidence =
      take(__func__,
           importEvaluationModelInvocation(request.request, request.resolution,
                                           *prepared, artifacts, blobs));
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  require(__func__, completed && completed->metricResults.size() == 4,
          "real OpenROAD FPA did not finalize four metrics");
}

void realOpenRoadFpaSmoke(const std::filesystem::path &root,
                          llvm::StringRef executable, llvm::StringRef version,
                          const std::filesystem::path &technologyLef,
                          const std::filesystem::path &cellLef,
                          const std::filesystem::path &liberty) {
  version = version.trim();
  require(__func__, std::filesystem::path(executable.str()).is_absolute(),
          "real OpenROAD executable is not absolute");
  require(__func__, version == kPinnedBuild,
          "real OpenROAD version is not the pinned build");
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
  const ArtifactStore artifacts((root / "artifacts").string());
  const BlobStore blobs((root / "blobs").string());
  const OpenRoadTechnologyFixture technology =
      take(__func__, loadGpdk045OpenRoadTechnologyFixture(technologyLef,
                                                          cellLef, liberty));
  const OpenRoadGateFixture fixture =
      take(__func__, makeOpenRoadGateFixture(root, artifacts, blobs, version,
                                             technology));
  requireSuccess(__func__,
                 registerOpenRoadRoutedCandidateGeneratorDescriptor());
  const std::filesystem::path routeTool = writeAuthoredGpdkRouteTool(root);
  const LocalToolConfig routeLocal =
      makeOpenRoadLocalToolConfig(fixture, routeTool);
  OpenRoadRouteHarness routeHarness =
      take(__func__, makeOpenRoadRouteHarness(root / "authored-route", fixture,
                                              routeLocal));
  const FinalizedHardwareImplementation routed = take(
      __func__, runOpenRoadRouteFixture(fixture, routeHarness,
                                        makeOpenRoadResolvedExecution(
                                            routeTool.string(), version, false),
                                        artifacts, blobs));
  evaluateRealOpenRoadFpa(root, executable, version, fixture, routed, artifacts,
                          blobs);
}

} // namespace

int main(int argc, char **argv) {
  if (argc == 8 && llvm::StringRef(argv[1]) == "--real-smoke") {
    realOpenRoadFpaSmoke(std::filesystem::absolute(argv[2]).lexically_normal(),
                         argv[3], argv[4], argv[5], argv[6], argv[7]);
    return EXIT_SUCCESS;
  }
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  resultParserAndDriverAreStrict();
  authoredLifecycleSeparatesAllOutcomes(
      std::filesystem::absolute(argv[1]).lexically_normal());
  return EXIT_SUCCESS;
}

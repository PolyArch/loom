#include "EDA/Adapters/OpenSource/MappedRtlSimulation.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Evaluation/ModelProvider.h"
#include "ExternalTool/Binding.h"
#include "ExternalTool/InvocationBundle.h"
#include "ExternalTool/LocalConfig.h"
#include "ExternalTool/Provider.h"
#include "ExternalTool/ShellProbe.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Simulator/SimulationExecution.h"

#include "MappedRtlSimulationTestSupport.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <variant>

namespace {

const loom::external_tool::PreparedExternalToolInvocation *
externalInvocation(loom::evaluation::EvaluationModelPreparation &preparation) {
  auto *live = std::get_if<loom::evaluation::EvaluationModelPreparedInvocation>(
      &preparation);
  return live ? &live->externalInvocation() : nullptr;
}

constexpr int kFakeToolUsageExitCode = 64;
constexpr std::int64_t kExpectedMappedCycleCount = 8;

struct FakeVerilatorBehavior final {
  std::string version = "Verilator 5.050";
  int compileExitCode = 0;
  int simulationExitCode = 0;
  bool omitResult = false;
};

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "mapped RTL simulation test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T>
void expectErrorContains(llvm::Expected<T> value, llvm::StringRef expected) {
  if (value)
    fail("expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(llvm::StringRef(message).contains(expected), message);
}

void writeFile(const std::filesystem::path &path, llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    fail("cannot write " + path.string());
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output)
    fail("cannot finish writing " + path.string());
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail("cannot read " + path.string());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

void writeExecutable(const std::filesystem::path &path,
                     llvm::StringRef contents) {
  writeFile(path, contents);
  std::filesystem::permissions(path, std::filesystem::perms::owner_read |
                                         std::filesystem::perms::owner_write |
                                         std::filesystem::perms::owner_exec);
}

std::string fakeVerilator(llvm::StringRef result,
                          const FakeVerilatorBehavior &behavior = {}) {
  using namespace loom::eda::open_source;
  std::string script = R"sh(#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -eq 1 && "$1" == "--version" ]]; then
)sh";
  script += "  printf '%s\\n' '" + behavior.version + "'\n";
  script += R"sh(
  exit 0
fi
driver_path=')sh";
  script += mappedRtlVerilatorDriverPath.str();
  script += R"sh('
testbench_path=')sh";
  script += mappedRtlTestbenchPath.str();
  script += R"sh('
simulator_path=')sh";
  script += mappedRtlSimulatorExecutablePath.str();
  script += R"sh('
if [[ "$#" -ne 2 || "$1" != "-f" || "$2" != "$driver_path" ]]; then
  exit )sh";
  script += std::to_string(kFakeToolUsageExitCode);
  script += R"sh(
fi
grep -Fx -- '--binary' "$driver_path" >/dev/null
grep -Fx -- '--build-jobs' "$driver_path" >/dev/null
grep -Fx -- '--top-module' "$driver_path" >/dev/null
)sh";
  script += "grep -Fx -- '" + mappedRtlHarnessTop.str() +
            "' \"$driver_path\" >/dev/null\n";
  script += "grep -F 'module " + mappedRtlHarnessTop.str() +
            "(' \"$testbench_path\" >/dev/null\n";
  if (behavior.compileExitCode != 0) {
    script += "exit " + std::to_string(behavior.compileExitCode) + "\n";
    return script;
  }
  script +=
      R"sh(grep -F 'task automatic loom_cfg_write_' "$testbench_path" >/dev/null
mkdir -p "$(dirname -- "$simulator_path")"
cat > "$simulator_path" <<'LOOM_SIMULATOR'
#!/usr/bin/env bash
set -euo pipefail
)sh";
  if (behavior.simulationExitCode != 0) {
    script += "exit " + std::to_string(behavior.simulationExitCode) + "\n";
  } else if (!behavior.omitResult) {
    script += "mkdir -p outputs\ncat > " + mappedRtlResultPath.str() +
              " <<'LOOM_RESULT'\n";
    script += result.str();
    script += "LOOM_RESULT\n";
  }
  script += R"sh(
LOOM_SIMULATOR
chmod u+x "$simulator_path"
)sh";
  return script;
}

loom::eda::open_source::MappedRtlSimulationResult expectedMappedResult() {
  using namespace loom::eda::open_source;
  using namespace loom::sim;
  MappedRtlSimulationResult result;
  result.terminal = MappedRtlTerminalStatus::Retired;
  result.launchCycle = 3;
  result.retirementCycle = 11;
  result.terminalCycle = 12;
  result.valueResults = {{{llvm::APInt(32, 7)}}};
  result.streamOutputs = {
      {32, {llvm::APInt(32, 0x21)}, StreamTermination::ClosedAfterLast},
      {32, {llvm::APInt(32, 0x3b)}, StreamTermination::ClosedAfterLast}};
  std::vector<SemanticMemoryByte> memory(
      16, SemanticMemoryByte{SemanticState::Defined, 0});
  memory[0].value = 0x11;
  memory[1].value = 0x22;
  memory[2].value = 0x33;
  memory[3].value = 0x44;
  result.memories = {{std::move(memory)}};
  return result;
}

void resultProtocolIsCanonical() {
  using namespace loom::eda::open_source;
  using loom::sim::SemanticMemoryByte;
  using loom::sim::SemanticState;
  using loom::sim::StreamTermination;

  MappedRtlSimulationResult result;
  result.terminal = MappedRtlTerminalStatus::Retired;
  result.launchCycle = 3;
  result.retirementCycle = 11;
  result.terminalCycle = 12;
  result.valueResults = {{{llvm::APInt(8, 0xa5)}}, {std::nullopt}};
  result.streamOutputs = {{4,
                           {llvm::APInt(4, 1), llvm::APInt(4, 14)},
                           StreamTermination::ClosedAfterLast}};
  result.memories = {{{SemanticMemoryByte{SemanticState::Defined, 0x34},
                       SemanticMemoryByte{SemanticState::Undef, 0},
                       SemanticMemoryByte{SemanticState::Poison, 0}}}};
  const std::string encoded = take(renderMappedRtlSimulationResult(result));
  const MappedRtlSimulationResult decoded =
      take(parseMappedRtlSimulationResult(encoded));
  require(decoded.terminal == MappedRtlTerminalStatus::Retired &&
              decoded.launchCycle == 3 && decoded.retirementCycle == 11 &&
              decoded.terminalCycle == 12 && decoded.valueResults.size() == 2 &&
              decoded.valueResults.front().token == llvm::APInt(8, 0xa5) &&
              decoded.streamOutputs.size() == 1 &&
              decoded.streamOutputs.front().tokens.size() == 2 &&
              decoded.memories.size() == 1 &&
              decoded.memories.front().bytes.size() == 3,
          "round trip changed an observation");

  std::string noncanonical = encoded;
  const std::string retired =
      "terminal " +
      mappedRtlTerminalStatusSpelling(MappedRtlTerminalStatus::Retired).str();
  noncanonical.insert(noncanonical.find(retired), "unknown x\n");
  auto rejected = parseMappedRtlSimulationResult(noncanonical);
  require(!rejected, "parser accepted an unknown field");
  llvm::consumeError(rejected.takeError());

  MappedRtlSimulationResult stopped;
  stopped.terminal = MappedRtlTerminalStatus::StoppedByLimit;
  stopped.terminalCycle = 20;
  require(take(parseMappedRtlSimulationResult(
                   take(renderMappedRtlSimulationResult(stopped))))
                  .terminal == MappedRtlTerminalStatus::StoppedByLimit,
          "stopped result did not round trip");
}

void requestFixtureClosesExactOwners() {
  const llvm::StringRef test = __func__;
  loom::deployment::test::TemporaryTree tree("mapped-rtl-request");
  const std::string artifactPath = tree.path("artifacts");
  const std::string blobPath = tree.path("blobs");
  std::filesystem::create_directories(artifactPath);
  std::filesystem::create_directories(blobPath);
  loom::ArtifactStore artifacts(artifactPath);
  loom::BlobStore blobs(blobPath);
  const auto fixture = loom::eda::test::buildMappedRtlRequestFixture(
      test, "Verilator 5.050", artifacts, blobs, tree);
  const auto fabric =
      take(loom::fabric::importEntireFabricRoot(fixture.module, artifacts));
  const auto spatialMapping = take(
      loom::mapping::importSpatialMapping(fixture.spatialMapping, artifacts));
  mlir::MLIRContext relationContext;
  std::size_t activeConfiguredTagBoundaries = 0;
  for (const auto &field :
       spatialMapping.view().configuredHardware().fields()) {
    const auto &owner = field.slot.field.owner.catalog();
    if (owner.kind() !=
        loom::fabric::FabricInventoryOwnerKind::BoundaryOccurrence)
      continue;
    const auto boundary =
        std::get<loom::fabric::FabricBoundaryOccurrenceRef>(owner.payload);
    const auto point = fabric.view().boundaryTagContinuityPoint(boundary);
    require(point.has_value(), "configured boundary lost its continuity kind");
    auto relation = take(
        fabric.view().semanticFieldRelation(field.slot.field, relationContext));
    if (llvm::Error error = relation.validateSemanticValue(field.value.bytes()))
      fail(llvm::toString(std::move(error)));
    if (point->kind ==
        loom::fabric::FabricBoundaryTagContinuityKind::ConfigurableWriter) {
      require(!field.value.bytes().empty() &&
                  (field.value.bytes().front() & 1U) != 0,
              "selected configured-tag boundary remained Disabled");
      ++activeConfiguredTagBoundaries;
    }
  }
  require(fixture.request.workload() == fixture.workload &&
              fixture.request.runtimeInput() == fixture.runtimeInput &&
              !fixture.implementation.implementation().interfaces().empty() &&
              fixture.deployment.reference().schemaIdentity ==
                  loom::deployment::deploymentSchema.identity &&
              activeConfiguredTagBoundaries != 0,
          "real mapped request lost an exact owner");
}

std::int64_t requireCompletedEvidence(
    const loom::eda::test::MappedRtlRequestFixture &fixture,
    const loom::evaluation::EvaluationEvidence &evidence,
    const loom::ArtifactStore &artifacts, const loom::BlobStore &blobs) {
  using namespace loom;
  using namespace loom::evaluation;

  require(evidence.outputBindings().size() == 1 &&
              evidence.outputBindings().front().artifacts.size() == 1,
          "completed mapped RTL run omitted its execution artifact");
  const auto execution = take(sim::importSimulationExecution(
      evidence.outputBindings().front().artifacts.front(), fixture.resolution,
      artifacts, blobs));
  require(std::holds_alternative<sim::RetiredExecution>(execution.terminal()),
          "mapped RTL execution did not retire");
  const auto *published = std::get_if<sim::PublishedValueResult>(
      &execution.spatialFunctionalObservations().valueResults.front());
  require(published && published->value.tokenCount == 1 &&
              published->value.lanes.size() == 1 &&
              published->value.lanes.front().bits == llvm::APInt(32, 7),
          "mapped RTL execution lost its observed value");
  const auto &functional = execution.spatialFunctionalObservations();
  require(functional.streamOutputs.size() == 2,
          "mapped RTL execution omitted a stream observation");
  const auto requireStream = [&](std::size_t ordinal, std::uint64_t expected) {
    const sim::CanonicalStreamSequence &stream =
        functional.streamOutputs[ordinal];
    require(stream.values.tokenCount == 1 && stream.values.lanes.size() == 1 &&
                stream.values.lanes.front().bits == llvm::APInt(32, expected) &&
                stream.termination == sim::StreamTermination::ClosedAfterLast,
            "mapped RTL execution changed an observed stream");
  };
  requireStream(0, 0x21);
  requireStream(1, 0x3b);
  require(functional.memories.size() == 1,
          "mapped RTL execution omitted its memory observation");
  const auto *memory =
      std::get_if<sim::FullMemoryObservation>(&functional.memories.front());
  require(memory && memory->bytes.size() == 16,
          "mapped RTL execution changed the observed memory extent");
  const std::array<std::uint8_t, 4> expectedMemory{0x11, 0x22, 0x33, 0x44};
  for (std::size_t ordinal = 0; ordinal != memory->bytes.size(); ++ordinal) {
    const std::uint8_t expected =
        ordinal < expectedMemory.size() ? expectedMemory[ordinal] : 0;
    require(memory->bytes[ordinal].state == sim::SemanticState::Defined &&
                memory->bytes[ordinal].value == expected,
            "mapped RTL execution changed an observed memory byte");
  }
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  const auto *point = completed && completed->metricResults.size() == 1
                          ? std::get_if<PointObservation>(
                                &completed->metricResults.front().observation)
                          : nullptr;
  const auto *cycles =
      point ? std::get_if<IntegerValue>(&point->value) : nullptr;
  require(cycles && cycles->value() >= 0,
          "mapped RTL Evidence omits a nonnegative cycle count");
  const auto &progress = execution.spatialProgressObservations();
  require(progress.graphRetirementVisible &&
              progress.launchAccepted.referenceCycle.denominator() == 1 &&
              progress.graphRetirementVisible->referenceCycle.denominator() ==
                  1 &&
              progress.graphRetirementVisible->referenceCycle.numerator() >=
                  progress.launchAccepted.referenceCycle.numerator() &&
              static_cast<std::uint64_t>(cycles->value()) ==
                  progress.graphRetirementVisible->referenceCycle.numerator() -
                      progress.launchAccepted.referenceCycle.numerator(),
          "mapped RTL cycle count disagrees with progress coordinates");
  return cycles->value();
}

void requireSelectedHardwareCoverage(
    const loom::eda::test::MappedRtlRequestFixture &fixture,
    const loom::ArtifactStore &artifacts) {
  using namespace loom;
  const auto module =
      take(loom::fabric::importEntireFabricRoot(fixture.module, artifacts));
  const auto mapping = take(
      loom::mapping::importSpatialMapping(fixture.spatialMapping, artifacts));
  std::size_t spatialCompute = 0;
  std::size_t temporalCompute = 0;
  for (const loom::mapping::SpatialComputeBindingView &binding :
       mapping.view().computeBindings()) {
    const auto pe = module.view().parentPeOf(binding.occurrence);
    require(pe.has_value(), "mapped compute binding has no parent PE");
    const auto schedule = module.view().peSchedule(*pe);
    require(schedule.has_value(), "mapped compute binding has no PE schedule");
    spatialCompute += *schedule == ::fabric::Schedule::Spatial;
    temporalCompute += *schedule == ::fabric::Schedule::Temporal;
  }
  require(spatialCompute >= 3 && temporalCompute >= 1,
          "real mapping did not select both PE schedules");
  require(
      mapping.view().memoryEngineBindings().size() == 1 &&
          mapping.view().memoryBindings().size() == 1 &&
          std::holds_alternative<loom::mapping::SpatialMemoryBoundaryProxyView>(
              mapping.view().memoryBindings().front().target),
      "real mapping did not select the external memory service");

  bool sawBoundary = false;
  bool sawBufferedFifo = false;
  bool sawBypassFifo = false;
  std::map<std::string, std::size_t> switchUseCounts;
  const auto inspectTraversal = [&](const auto &optionalTraversal) {
    if (!optionalTraversal)
      return;
    const loom::fabric::FabricPhysicalTraversalRef &traversal =
        *optionalTraversal;
    if (const auto *sw =
            std::get_if<loom::fabric::FabricSwitchTraversalPayload>(
                &traversal.payload)) {
      const auto key = loom::fabric::canonicalFabricBytes(sw->owner);
      ++switchUseCounts[std::string(reinterpret_cast<const char *>(key.data()),
                                    key.size())];
    } else if (const auto *fifo =
                   std::get_if<loom::fabric::FabricFifoTraversalPayload>(
                       &traversal.payload)) {
      sawBufferedFifo |=
          fifo->mode == loom::fabric::FabricFifoTraversalMode::Buffered;
      sawBypassFifo |=
          fifo->mode == loom::fabric::FabricFifoTraversalMode::Bypass;
    } else if (std::holds_alternative<
                   loom::fabric::FabricBoundaryTraversalPayload>(
                   traversal.payload)) {
      sawBoundary = true;
    }
  };
  for (const loom::mapping::SpatialRouteTreeView &route :
       mapping.view().routeTrees()) {
    inspectTraversal(route.localTraversal);
    for (const loom::mapping::SpatialRouteNodeView &node : route.nodes)
      inspectTraversal(node.incomingTraversal);
    for (const loom::mapping::SpatialRouteSinkView &sink : route.sinks)
      inspectTraversal(sink.localTraversal);
  }
  for (const loom::mapping::SpatialRegisterFifoTransferView &transfer :
       mapping.view().registerFifoTransfers()) {
    inspectTraversal(std::optional<loom::fabric::FabricPhysicalTraversalRef>(
        transfer.writeTraversal));
    inspectTraversal(std::optional<loom::fabric::FabricPhysicalTraversalRef>(
        transfer.readTraversal));
  }
  const bool sharedSwitch =
      std::any_of(switchUseCounts.begin(), switchUseCounts.end(),
                  [](const auto &entry) { return entry.second >= 2; });
  require(sawBoundary && sawBufferedFifo && sharedSwitch,
          "real mapping omitted a required hierarchy traversal");
  require(sawBypassFifo,
          "real mapping did not exercise a bypass FIFO traversal");
  require(!mapping.view().registerFifoTransfers().empty(),
          "real mapping did not select a Temporal PE register FIFO");
}

void authoredLifecycleImportsExactEvidence() {
  using namespace loom;
  using namespace loom::eda::open_source;
  using namespace loom::evaluation;
  using namespace loom::external_tool;

  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree("mapped-rtl-lifecycle");
  const std::filesystem::path artifactPath = tree.path("artifacts");
  const std::filesystem::path blobPath = tree.path("blobs");
  ArtifactStore artifacts(artifactPath.string());
  BlobStore blobs(blobPath.string());
  struct PreparedLifecycle final {
    eda::test::MappedRtlRequestFixture fixture;
    PreparedExternalToolInvocation invocation;
  };
  PreparedLifecycle prepared = [&] {
    evaluation::ArtifactImportCacheScope sourceInvocation(artifacts, &blobs);
    auto fixture = eda::test::buildMappedRtlRequestFixture(
        test, "Verilator 5.050", artifacts, blobs, tree);
    const std::string expectedResult =
        take(renderMappedRtlSimulationResult(expectedMappedResult()));
    const std::filesystem::path tool = tree.path("fake-verilator");
    writeExecutable(tool, fakeVerilator(expectedResult));
    LocalToolConfig local;
    local.runtimePolicy = RuntimePolicy::Host;
    local.tools[verilatorProvider().binding.key].binding.executable =
        tool.string();
    const std::filesystem::path bundle = tree.path("bundle");
    EvaluationModelPreparation preparation =
        take(prepareEvaluationModelInvocation(
            fixture.request, fixture.resolution, artifacts, blobs,
            ExternalToolPreparationContext{std::move(local), bundle.string()}));
    auto *invocation = externalInvocation(preparation);
    require(invocation, "supported request did not prepare a Verilator bundle");
    require(!std::filesystem::exists(bundle / mappedRtlResultPath.str()),
            "preparation manufactured a simulation result");
    require(llvm::StringRef(readFile(bundle / mappedRtlTestbenchPath.str()))
                .contains("task automatic loom_cfg_read_"),
            "generated testbench omits configuration readback");
    require(take(executeExternalToolInvocationBundle(*invocation)) == 0,
            "authored Verilator lifecycle failed");
    return PreparedLifecycle{std::move(fixture), std::move(*invocation)};
  }();

  evaluation::ArtifactImportCacheScope independentReplay(artifacts, &blobs);
  const EvaluationEvidence evidence = take(importEvaluationModelInvocation(
      prepared.fixture.request, prepared.fixture.resolution,
      prepared.invocation, artifacts, blobs));
  require(requireCompletedEvidence(prepared.fixture, evidence, artifacts,
                                   blobs) == kExpectedMappedCycleCount,
          "authored lifecycle changed its exact cycle count");
}

void authoredResultTamperIsRejected() {
  using namespace loom;
  using namespace loom::eda::open_source;
  using namespace loom::evaluation;
  using namespace loom::external_tool;

  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree("mapped-rtl-result-tamper");
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const auto fixture = eda::test::buildMappedRtlRequestFixture(
      test, "Verilator 5.050", artifacts, blobs, tree);
  const std::string expectedResult =
      take(renderMappedRtlSimulationResult(expectedMappedResult()));
  const std::filesystem::path tool = tree.path("fake-verilator");
  writeExecutable(tool, fakeVerilator(expectedResult));
  LocalToolConfig local;
  local.runtimePolicy = RuntimePolicy::Host;
  local.tools[verilatorProvider().binding.key].binding.executable =
      tool.string();
  const std::filesystem::path bundle = tree.path("bundle");
  EvaluationModelPreparation preparation =
      take(prepareEvaluationModelInvocation(
          fixture.request, fixture.resolution, artifacts, blobs,
          ExternalToolPreparationContext{std::move(local), bundle.string()}));
  const auto *prepared = externalInvocation(preparation);
  require(prepared, "result-tamper request did not prepare a bundle");
  require(take(executeExternalToolInvocationBundle(*prepared)) == 0,
          "result-tamper lifecycle failed");

  writeFile(bundle / mappedRtlResultPath.str(), expectedResult + " ");
  expectErrorContains(
      importEvaluationModelInvocation(fixture.request, fixture.resolution,
                                      *prepared, artifacts, blobs),
      "completion digest");
}

template <typename Outcome>
void requireTerminalOutcome(
    const loom::eda::test::MappedRtlRequestFixture &fixture,
    const loom::external_tool::PreparedExternalToolInvocation &prepared,
    const loom::ArtifactStore &artifacts, const loom::BlobStore &blobs,
    loom::evaluation::OutcomeReason reason) {
  const loom::evaluation::EvaluationEvidence evidence =
      take(loom::evaluation::importEvaluationModelInvocation(
          fixture.request, fixture.resolution, prepared, artifacts, blobs));
  const auto *outcome = std::get_if<Outcome>(&evidence.outcome());
  require(outcome && outcome->reason == reason,
          "mapped RTL failure has the wrong typed outcome");
  require(evidence.outputBindings().size() == 1 &&
              evidence.outputBindings().front().artifacts.empty(),
          "terminal mapped RTL outcome published an execution Artifact");
}

enum class AuthoredFailureCase : std::uint8_t {
  Compile,
  Simulation,
  MissingResult,
  StoppedByLimit,
  MalformedResult,
  MappingTamper,
  ConfigurationImageTamper,
  VersionMismatch,
};

void authoredFailure(AuthoredFailureCase selected) {
  using namespace loom;
  using namespace loom::eda::open_source;
  using namespace loom::evaluation;
  using namespace loom::external_tool;

  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree("mapped-rtl-failures");
  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const auto fixture = eda::test::buildMappedRtlRequestFixture(
      test, "Verilator 5.050", artifacts, blobs, tree);

  MappedRtlSimulationResult retired = expectedMappedResult();
  const std::string validResult =
      take(renderMappedRtlSimulationResult(retired));

  const auto prepare = [&](llvm::StringRef name, llvm::StringRef result,
                           const FakeVerilatorBehavior &behavior) {
    const std::filesystem::path tool = tree.path((name + "-verilator").str());
    writeExecutable(tool, fakeVerilator(result, behavior));
    LocalToolConfig local;
    local.runtimePolicy = RuntimePolicy::Host;
    local.tools[verilatorProvider().binding.key].binding.executable =
        tool.string();
    const std::filesystem::path bundle = tree.path((name + "-bundle").str());
    EvaluationModelPreparation preparation =
        take(prepareEvaluationModelInvocation(
            fixture.request, fixture.resolution, artifacts, blobs,
            ExternalToolPreparationContext{std::move(local), bundle.string()}));
    const auto *prepared = externalInvocation(preparation);
    require(prepared, "authored failure did not prepare a bundle");
    return *prepared;
  };

  switch (selected) {
  case AuthoredFailureCase::Compile: {
    const auto run =
        prepare("compile-failure", validResult,
                FakeVerilatorBehavior{"Verilator 5.050", 17, 0, false});
    require(take(executeExternalToolInvocationBundle(run)) == 17,
            "compile failure exit status changed");
    requireTerminalOutcome<ExecutionFailedEvidence>(
        fixture, run, artifacts, blobs, OutcomeReason::ToolFailure);
    return;
  }
  case AuthoredFailureCase::Simulation: {
    const auto run =
        prepare("simulation-failure", validResult,
                FakeVerilatorBehavior{"Verilator 5.050", 0, 18, false});
    require(take(executeExternalToolInvocationBundle(run)) == 18,
            "simulation failure exit status changed");
    requireTerminalOutcome<ExecutionFailedEvidence>(
        fixture, run, artifacts, blobs, OutcomeReason::ToolFailure);
    return;
  }
  case AuthoredFailureCase::MissingResult: {
    const auto run =
        prepare("missing-result", validResult,
                FakeVerilatorBehavior{"Verilator 5.050", 0, 0, true});
    require(take(executeExternalToolInvocationBundle(run)) ==
                static_cast<int>(InvocationLauncherExitCode::MissingOutput),
            "missing result exit status changed");
    requireTerminalOutcome<ExecutionFailedEvidence>(
        fixture, run, artifacts, blobs, OutcomeReason::ToolFailure);
    return;
  }
  case AuthoredFailureCase::StoppedByLimit: {
    MappedRtlSimulationResult stopped;
    stopped.terminal = MappedRtlTerminalStatus::StoppedByLimit;
    stopped.terminalCycle = 64;
    const auto run =
        prepare("stopped", take(renderMappedRtlSimulationResult(stopped)), {});
    require(take(executeExternalToolInvocationBundle(run)) == 0,
            "stopped result lifecycle failed");
    requireTerminalOutcome<CancelledOrTimeoutEvidence>(
        fixture, run, artifacts, blobs, OutcomeReason::ExecutionLimitReached);
    return;
  }
  case AuthoredFailureCase::MalformedResult: {
    const auto run = prepare("malformed", "not a mapped result\n", {});
    require(take(executeExternalToolInvocationBundle(run)) == 0,
            "malformed result lifecycle failed before import");
    expectErrorContains(importEvaluationModelInvocation(fixture.request,
                                                        fixture.resolution, run,
                                                        artifacts, blobs),
                        "mapped_rtl_result_invalid");
    return;
  }
  case AuthoredFailureCase::MappingTamper: {
    const auto run = prepare("mapping-tamper", validResult, {});
    const std::filesystem::path path = std::filesystem::path(run.bundleRoot) /
                                       "inputs/semantic/spatial-mapping.mlir";
    writeFile(path, readFile(path) + " ");
    require(
        take(executeExternalToolInvocationBundle(run)) ==
            static_cast<int>(InvocationLauncherExitCode::BundleContentMismatch),
        "mapping tamper was not rejected before tool execution");
    expectErrorContains(importEvaluationModelInvocation(fixture.request,
                                                        fixture.resolution, run,
                                                        artifacts, blobs),
                        "bundle content changed");
    return;
  }
  case AuthoredFailureCase::ConfigurationImageTamper: {
    const auto run = prepare("image-tamper", validResult, {});
    const std::filesystem::path directory =
        std::filesystem::path(run.bundleRoot) /
        "inputs/semantic/configuration-images";
    std::filesystem::directory_iterator image(directory), end;
    require(image != end && image->is_regular_file(),
            "mapped bundle contains no configuration image");
    writeFile(image->path(), readFile(image->path()) + " ");
    require(
        take(executeExternalToolInvocationBundle(run)) ==
            static_cast<int>(InvocationLauncherExitCode::BundleContentMismatch),
        "configuration-image tamper was not rejected before execution");
    return;
  }
  case AuthoredFailureCase::VersionMismatch: {
    const std::filesystem::path tool = tree.path("mismatched-verilator");
    writeExecutable(tool, fakeVerilator(validResult, FakeVerilatorBehavior{
                                                         "Verilator 5.051"}));
    LocalToolConfig local;
    local.runtimePolicy = RuntimePolicy::Host;
    local.tools[verilatorProvider().binding.key].binding.executable =
        tool.string();
    expectErrorContains(
        prepareEvaluationModelInvocation(
            fixture.request, fixture.resolution, artifacts, blobs,
            ExternalToolPreparationContext{std::move(local),
                                           tree.path("mismatched-bundle")}),
        "resolved Verilator build differs");
    return;
  }
  }
  llvm_unreachable("closed authored failure case");
}

void realVerilatorLifecycle() {
  using namespace loom;
  using namespace loom::evaluation;
  using namespace loom::external_tool;

  const llvm::StringRef test = __func__;
  deployment::test::TemporaryTree tree("mapped-rtl-verilator");
  const std::filesystem::path probeRoot = tree.path("probe");
  std::filesystem::create_directories(probeRoot);
  LocalToolConfig local;
  local.runtimePolicy = RuntimePolicy::Host;
  const ExternalToolProviderDescriptor &provider = verilatorProvider();
  ShellToolBindingProbe probe(probeRoot.string(), provider.versionProbe);
  const ResolvedToolBinding binding =
      take(resolveToolBinding(provider.binding, local,
                              captureToolEnvironment(provider.binding), probe));
  local.tools[provider.binding.key].binding.executable = binding.executable;
  constexpr std::uint64_t mappedRtlCycleLimit = 64;
  local.tools[provider.binding.key].providerOptions["max_cycles"] =
      mappedRtlCycleLimit;

  ArtifactStore artifacts(tree.path("artifacts"));
  BlobStore blobs(tree.path("blobs"));
  const auto fixture = eda::test::buildMappedRtlRequestFixture(
      test, binding.version, artifacts, blobs, tree,
      eda::test::MappedRtlFixtureTopology::HeterogeneousPortable);
  requireSelectedHardwareCoverage(fixture, artifacts);
  const std::filesystem::path bundle = tree.path("bundle");
  EvaluationModelPreparation preparation =
      take(prepareEvaluationModelInvocation(
          fixture.request, fixture.resolution, artifacts, blobs,
          ExternalToolPreparationContext{std::move(local), bundle.string()}));
  const auto *prepared = externalInvocation(preparation);
  require(prepared, "real Verilator request did not prepare a bundle");
  const int status = take(executeExternalToolInvocationBundle(*prepared));
  if (status != 0)
    fail("real Verilator lifecycle failed: " +
         readFile(bundle / "outputs/stderr.log"));
  const EvaluationEvidence evidence = take(importEvaluationModelInvocation(
      fixture.request, fixture.resolution, *prepared, artifacts, blobs));
  if (evidence.outputBindings().size() != 1 ||
      evidence.outputBindings().front().artifacts.size() != 1)
    fail("real mapped RTL execution did not complete: " +
         readFile(bundle / eda::open_source::mappedRtlResultPath.str()) +
         "\nsimulator stdout:\n" + readFile(bundle / "outputs/stdout.log"));
  require(requireCompletedEvidence(fixture, evidence, artifacts, blobs) > 0,
          "real Verilator execution retired without elapsed cycles");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected one test selector");
  if (llvm::Error error =
          loom::eda::open_source::registerMappedRtlSimulationProvider())
    fail(llvm::toString(std::move(error)));
  const llvm::StringRef selector(argv[1]);
  if (selector == "--real-verilator") {
    realVerilatorLifecycle();
    return EXIT_SUCCESS;
  }
  if (selector == "--result-protocol")
    resultProtocolIsCanonical();
  else if (selector == "--request-fixture")
    requestFixtureClosesExactOwners();
  else if (selector == "--authored-lifecycle")
    authoredLifecycleImportsExactEvidence();
  else if (selector == "--result-tamper")
    authoredResultTamperIsRejected();
  else if (selector == "--compile-failure")
    authoredFailure(AuthoredFailureCase::Compile);
  else if (selector == "--simulation-failure")
    authoredFailure(AuthoredFailureCase::Simulation);
  else if (selector == "--missing-result")
    authoredFailure(AuthoredFailureCase::MissingResult);
  else if (selector == "--stopped-by-limit")
    authoredFailure(AuthoredFailureCase::StoppedByLimit);
  else if (selector == "--malformed-result")
    authoredFailure(AuthoredFailureCase::MalformedResult);
  else if (selector == "--mapping-tamper")
    authoredFailure(AuthoredFailureCase::MappingTamper);
  else if (selector == "--configuration-image-tamper")
    authoredFailure(AuthoredFailureCase::ConfigurationImageTamper);
  else if (selector == "--version-mismatch")
    authoredFailure(AuthoredFailureCase::VersionMismatch);
  else
    fail("unknown test selector");
  return EXIT_SUCCESS;
}

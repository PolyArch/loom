#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "EDA/Adapters/Cadence/Voltus.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "Evaluation/ModelProvider.h"
#include "Evaluation/Models/PhysicalRailAnalysis.h"
#include "ExternalTool/InvocationBundle.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "Hardware/Implementation/PhysicalRepresentationIndex.h"
#include "ImplementationPlatform/ImplementationPlatform.h"

#include "ConfigurationABI2TestSupport.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using namespace loom;
using namespace loom::eda;
using namespace loom::eda::cadence;
using namespace loom::evaluation;
using namespace loom::evaluation::models;
using namespace loom::external_tool;
using namespace loom::hardware;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  std::cerr << test.str() << ": " << message << '\n';
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

template <typename T>
void expectErrorContains(llvm::StringRef test, llvm::Expected<T> value,
                         llvm::StringRef expected) {
  if (value)
    fail(test, "expected an error");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

std::vector<std::uint8_t> bytes(llvm::StringRef value) {
  return {value.bytes_begin(), value.bytes_end()};
}

void writeFile(const std::filesystem::path &path, llvm::StringRef contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    fail(__func__, "cannot write " + path.string());
  output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
  if (!output)
    fail(__func__, "cannot finish writing " + path.string());
}

std::string readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail(__func__, "cannot read " + path.string());
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

ExternalFileFingerprint fingerprint(llvm::StringRef contents) {
  return take(__func__, ExternalFileFingerprint::fromBytes(
                            llvm::SHA256::hash(bytes(contents))));
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *instance = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *instance;
}

loom::fabric::FinalizedFabricRoot makeModule(const ArtifactStore &store) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @rail_fixture(
          %a: !fabric.bits<1>) -> !fabric.bits<1> {
        fabric.yield %a : !fabric.bits<1>
      }
    }
  )mlir",
                                                        &context());
  require(__func__, static_cast<bool>(source), "cannot parse Fabric fixture");
  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(__func__, static_cast<bool>(root), "Fabric fixture has no root");
  return take(__func__, loom::fabric::finalizeFabricRoot(root, store));
}

std::string physicalDef(bool multiplePowerNets, bool partialNetwork = false) {
  std::string result =
      "VERSION 5.8 ;\n"
      "DESIGN top ;\n"
      "PINS 2 ;\n"
      "- VPWR + NET power_main + DIRECTION INOUT + USE POWER "
      "+ LAYER M4 ( 0 0 ) ( 10 10 ) + FIXED ( 20 20 ) N ;\n"
      "- VGND + NET ground_main + DIRECTION INOUT + USE GROUND "
      "+ LAYER M4 ( 0 0 ) ( 10 10 ) + FIXED ( 40 20 ) N ;\n"
      "END PINS\n"
      "SPECIALNETS " +
      std::to_string(multiplePowerNets ? 3 : 2) +
      " ;\n"
      "- power_main + USE POWER " +
      std::string(partialNetwork ? ";\n"
                                 : "+ ROUTED M4 ( 20 20 ) ( 100 20 ) ;\n") +
      "- ground_main + USE GROUND + ROUTED M4 ( 40 20 ) ( 100 40 ) ;\n";
  if (multiplePowerNets)
    result += "- auxiliary + USE POWER + ROUTED M4 ( 0 0 ) ( 10 0 ) ;\n";
  result += "END SPECIALNETS\nEND DESIGN\n";
  return result;
}

ImplementationPayload putPayload(const BlobStore &blobs, PayloadRole role,
                                 llvm::StringRef name,
                                 llvm::StringRef contents) {
  return {role, name.str(), take(__func__, blobs.put(bytes(contents)))};
}

FinalizedHardwareImplementation makePhysicalImplementation(
    const platform::FinalizedImplementationPlatform &platform,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    bool multiplePowerNets, bool partialNetwork = false) {
  const loom::fabric::FinalizedFabricRoot module = makeModule(artifacts);
  const loom::fabric::FinalizedFabricRoot system = take(
      __func__, hardware::test::makeSingleSpatialCoreSystem(module, artifacts));
  const FinalizedConfigurationABI abi = take(
      __func__, finalizeConfigurationABI(
                    ConfigurationABIDraft{system.reference(), {}}, artifacts));

  std::vector<ImplementationPayload> payloads{
      putPayload(blobs, PayloadRole::Netlist, "netlist/top.v",
                 "module top(input clk); helper u_helper(); endmodule\n"),
      putPayload(blobs, PayloadRole::Netlist, "netlist/helper.v",
                 "module helper; endmodule\n"),
      putPayload(blobs, PayloadRole::PhysicalDatabase, "database/top.def",
                 physicalDef(multiplePowerNets, partialNetwork)),
      putPayload(blobs, PayloadRole::GenerationConstraint,
                 "constraints/top.sdc", "create_clock -period 2 clk\n")};
  const RepresentationFormatDescriptorRef format =
      take(__func__, RepresentationFormatDescriptorRef::get(
                         RepresentationFormatKind::IndexedDefPhysical));
  const RepresentationLocator top{RepresentationObjectKind::PhysicalObject,
                                  "top"};
  PhysicalRepresentationIndexPayload index = take(
      __func__,
      createPhysicalRepresentationIndexPayload(
          format, RepresentationRootVariant::AsicPhysical,
          RepresentationPhysicalStage::Routed, top, "index/physical.json",
          payloads,
          {{{RepresentationObjectKind::PhysicalObject, "top"}, std::nullopt}},
          {}));
  const std::string indexBytes =
      take(__func__, serializePhysicalRepresentationIndexPayloadJson(index));
  payloads.push_back(putPayload(blobs, PayloadRole::RepresentationIndex,
                                index.indexLogicalName, indexBytes));
  ImplementationRepresentationRoot representation =
      take(__func__, createImplementationRepresentationRoot(
                         RepresentationRootVariant::AsicPhysical,
                         RepresentationPhysicalStage::Routed, format, top,
                         std::move(payloads)));

  ExternalImplementationContractCatalog contracts =
      take(__func__, makeKnownAsicStandardCellContractCatalog());
  require(__func__,
          contracts.find(cadenceGenusStandardCellContractRef) &&
              contracts.find(synopsysDesignCompilerStandardCellContractRef) &&
              contracts.find(openSourceYosysStandardCellContractRef),
          "shared standard-cell contract catalog is incomplete");
  return take(__func__,
              finalizeHardwareImplementation(
                  HardwareImplementationDraft{system.reference(),
                                              abi.reference(),
                                              {},
                                              std::move(representation),
                                              platform.reference(),
                                              {},
                                              {},
                                              {},
                                              {}},
                  contracts, artifacts, blobs));
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
            const std::vector<ExternalFileTreeMember> &pgvMembers,
            const ArtifactStore &artifacts) {
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
  const std::vector<EvaluationCondition> conditions{
      EvaluationCondition{ProcessCornerCondition{
          target,
          {platform.reference().artifact, platform::TechnologyCornerId(0)}}},
      EvaluationCondition{SupplyVoltageCondition{
          target, take(__func__, DecimalValue::get(9, -1))}},
      EvaluationCondition{TemperatureCondition{
          target, take(__func__, DecimalValue::get(3, 2))}},
      EvaluationCondition{RequiredClockPeriodCondition{
          target, take(__func__, DecimalValue::get(2, -9))}},
      EvaluationCondition{ActivityBindingCondition{
          target, ExplicitAssumptionSource{
                      target, take(__func__, ExactRatio::get(1, 2)),
                      take(__func__, ExactRatio::get(1, 10))}}}};
  const EvaluationCase evaluationCase = take(
      __func__, EvaluationCase::get(cadenceVoltusStaticRailModelDescriptorRef()
                                        .descriptor()
                                        ->caseSignature,
                                    subjects, std::nullopt, std::nullopt,
                                    conditions, resolution, artifacts));
  const MetricRequest metric = take(
      __func__, MetricRequest::get({MetricKind::MaximumVoltageDrop,
                                    EvaluationScope{ScopeFormRef(0), {}}},
                                   {}, evaluationCase, resolution, artifacts));
  ResolvedConfig config = defaultResolvedConfig();
  config.evaluation.cadenceVoltusStaticRail =
      CadenceVoltusStaticRailProviderBinding{
          "@(#)CDS: Voltus v26.10-p001_1",
          pgvMembers,
          {"technology.cl", "cells/stdcells.cl"}};
  ResolvedModelBinding model = take(
      __func__, ResolvedModelBinding::project(
                    cadenceVoltusStaticRailModelDescriptorRef(), {}, config));
  EvaluationRequest request =
      take(__func__,
           EvaluationRequest::get(evaluationCase, {metric}, {},
                                  std::move(model), 0, resolution, artifacts));
  const ArtifactRootReference published =
      take(__func__, publishEvaluationRequest(request, artifacts));
  require(__func__, published == evaluationRequestReference(request),
          "request publication changed identity");
  return {std::move(request), std::move(resolution)};
}

LocalToolConfig localConfig(const std::filesystem::path &tool,
                            const std::filesystem::path &pgvRoot) {
  LocalToolConfig local;
  local.runtimePolicy = RuntimePolicy::Host;
  local.tools["voltus"].binding.executable =
      std::filesystem::canonical(tool).string();
  local.externalFileTrees[cadenceVoltusPowerGridLibraryInputSlot.str()] =
      std::filesystem::canonical(pgvRoot).string();
  return local;
}

void completeAndUnsupportedLifecycles(const std::filesystem::path &root) {
  const ArtifactStore artifacts((root / "artifacts").string());
  const BlobStore blobs((root / "blobs").string());
  std::filesystem::create_directories(root / "artifacts");
  std::filesystem::create_directories(root / "blobs");
  const platform::FinalizedImplementationPlatform platform =
      take(__func__,
           platform::finalizeImplementationPlatform(
               {platform::AsicTarget{"saed32", "EDK_08_2025"}, {"typical"}},
               artifacts));

  const std::filesystem::path pgvRoot = root / "pgv";
  const std::string technology = "technology-pgv\n";
  const std::string cells = "standard-cell-pgv\n";
  writeFile(pgvRoot / "technology.cl", technology);
  writeFile(pgvRoot / "cells/stdcells.cl", cells);
  const std::vector<ExternalFileTreeMember> pgvMembers{
      {"cells/stdcells.cl", fingerprint(cells)},
      {"technology.cl", fingerprint(technology)}};

  const std::filesystem::path tool = root / "fake-voltus";
  writeFile(tool, R"sh(#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "-version" ]]; then
  printf '@(#)CDS: Voltus v26.10-p001_1\n'
  exit 0
fi
if [[ "$#" -ne 4 || "$1" != "-no_gui" || "$2" != "-batch" ||
      "$3" != "-files" || "$4" != "drivers/voltus-rail.tcl" ]]; then
  exit 64
fi
grep -F 'set_rail_analysis_mode -method static -accuracy hd' drivers/voltus-rail.tcl >/dev/null
grep -F 'report_power_rail_results -plot ivdd' drivers/voltus-rail.tcl >/dev/null
grep -F 'netlist/helper.v' drivers/voltus-rail.tcl >/dev/null
grep -F 'netlist/top.v' drivers/voltus-rail.tcl >/dev/null
mkdir -p work
printf '# domain voltage-drop report\n1.25e-2 instance0 power_main ground_main\n4.42943e-2 instance1 power_main ground_main\n3.7e-2 instance2 power_main ground_main\n' > work/voltus-ivdd.rpt
tclsh drivers/voltus-rail-publish.tcl
)sh");
  std::filesystem::permissions(tool, std::filesystem::perms::owner_read |
                                         std::filesystem::perms::owner_write |
                                         std::filesystem::perms::owner_exec);

  if (llvm::Error error = registerVoltusRailEvaluationProvider())
    fail(__func__, llvm::toString(std::move(error)));

  const FinalizedHardwareImplementation physical =
      makePhysicalImplementation(platform, artifacts, blobs, false);
  const RequestFixture fixture =
      makeRequest(physical, platform, pgvMembers, artifacts);
  const std::filesystem::path bundle = root / "complete";
  EvaluationModelPreparation preparation =
      take(__func__, prepareEvaluationModelInvocation(
                         fixture.request, fixture.resolution, artifacts, blobs,
                         {localConfig(tool, pgvRoot), bundle.string()}));
  const auto *prepared =
      std::get_if<PreparedExternalToolInvocation>(&preparation);
  require(__func__, prepared,
          "supported rail request did not prepare a bundle");
  const std::string driver = readFile(bundle / "drivers/voltus-rail.tcl");
  require(__func__,
          llvm::StringRef(driver).contains("-accuracy hd") &&
              llvm::StringRef(driver).find("technology.cl") <
                  llvm::StringRef(driver).find("cells/stdcells.cl"),
          "prepared driver lost fixed accuracy or ordered PGV entrypoints");
  require(__func__,
          take(__func__, executeExternalToolInvocationBundle(*prepared)) == 0,
          "authored Voltus lifecycle failed");
  const EvaluationEvidence evidence =
      take(__func__,
           importEvaluationModelInvocation(fixture.request, fixture.resolution,
                                           *prepared, artifacts, blobs));
  const auto *completed = std::get_if<CompletedEvidence>(&evidence.outcome());
  require(__func__, completed && completed->metricResults.size() == 1,
          "Voltus import did not publish one completed metric");
  const auto *point = std::get_if<PointObservation>(
      &completed->metricResults.front().observation);
  const auto *drop = point ? std::get_if<DecimalValue>(&point->value) : nullptr;
  require(__func__,
          drop && *drop == take(__func__, DecimalValue::get(442943, -7)),
          "Voltus observation was not normalized into voltage Evidence");
  writeFile(bundle / "outputs/voltus-rail-result.json",
            "{\"schema\":\"loom.cadence.voltus_rail_result\","
            "\"version\":\"1.0\","
            "\"maximum_voltage_drop_volts\":\"9e-1\"}\n");
  expectErrorContains(
      __func__,
      importEvaluationModelInvocation(fixture.request, fixture.resolution,
                                      *prepared, artifacts, blobs),
      "completion digest");

  const FinalizedHardwareImplementation multiDomain =
      makePhysicalImplementation(platform, artifacts, blobs, true);
  const RequestFixture unsupported =
      makeRequest(multiDomain, platform, pgvMembers, artifacts);
  EvaluationModelPreparation rejected =
      take(__func__,
           prepareEvaluationModelInvocation(
               unsupported.request, unsupported.resolution, artifacts, blobs,
               {localConfig(tool, pgvRoot), (root / "unsupported").string()}));
  const auto *unsupportedEvidence = std::get_if<EvaluationEvidence>(&rejected);
  require(__func__,
          unsupportedEvidence &&
              std::holds_alternative<UnsupportedEvidence>(
                  unsupportedEvidence->outcome()) &&
              std::get<UnsupportedEvidence>(unsupportedEvidence->outcome())
                      .reason == OutcomeReason::RuntimeCapabilityUnavailable,
          "multi-domain rail request was not finalized as typed Unsupported");

  const FinalizedHardwareImplementation partialNetwork =
      makePhysicalImplementation(platform, artifacts, blobs, false, true);
  const RequestFixture partial =
      makeRequest(partialNetwork, platform, pgvMembers, artifacts);
  EvaluationModelPreparation partialRejected = take(
      __func__,
      prepareEvaluationModelInvocation(
          partial.request, partial.resolution, artifacts, blobs,
          {localConfig(tool, pgvRoot), (root / "partial-network").string()}));
  const auto *partialEvidence =
      std::get_if<EvaluationEvidence>(&partialRejected);
  require(
      __func__,
      partialEvidence &&
          std::holds_alternative<UnsupportedEvidence>(
              partialEvidence->outcome()) &&
          std::get<UnsupportedEvidence>(partialEvidence->outcome()).reason ==
              OutcomeReason::RuntimeCapabilityUnavailable,
      "partial rail network was not finalized as typed Unsupported");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test root");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  completeAndUnsupportedLifecycles(root);
  return EXIT_SUCCESS;
}

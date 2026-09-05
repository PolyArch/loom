#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobDigest.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CampaignRunner.h"
#include "DSE/DataflowEvaluationAcquisition.h"
#include "DSE/DataflowRewriteCandidateGenerator.h"
#include "DSE/EvidenceObligation.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "DSE/FpaCampaign.h"
#include "DSE/FuReverseSynthesis.h"
#include "DSE/FuReverseSynthesisWorkflow.h"
#include "DSE/GroundTruthPlan.h"
#include "DSE/InvocationManifest.h"
#include "DSE/JointDesignExploration.h"
#include "DSE/MappingCandidateGenerator.h"
#include "DSE/ModelParameterCalibrationAcquisition.h"
#include "DSE/ModelParameterTrainingCandidateGenerator.h"
#include "DSE/PortableSpatialCoreRtlCandidateGenerator.h"
#include "DSE/ProductionOwners.h"
#include "DSE/RootCompleteSpatialPnrCandidateGenerator.h"
#include "DSE/RootCompleteSystemPnrCandidateGenerator.h"
#include "DSE/RootCompleteTechMappingCandidateGenerator.h"
#include "DSE/SpatialMappingEvaluationAcquisition.h"
#include "DSE/SpatialMappingFeedbackCandidateGenerator.h"
#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "DSE/SpatialTopologyCandidateGenerator.h"
#include "DSE/StructuredEvaluationAcquisition.h"
#include "DSE/StructuredExecutionShapeCandidateGenerator.h"
#include "DSE/StructuredMemoryCommunicationCandidateGenerator.h"
#include "DSE/StructuredOwnershipCandidateGenerator.h"
#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "DSE/StructuredSpecialMathAccuracyCandidateGenerator.h"
#include "DSE/SystemCompositionCandidateGenerator.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "EDA/Adapters/OpenSource/OpenRoadRouted.h"
#include "EDA/Adapters/OpenSource/OpenRoadStaticFpa.h"
#include "EDA/Adapters/OpenSource/YosysBlock.h"
#include "EDA/Adapters/OpenSource/YosysGateNetlist.h"
#include "EDA/Adapters/Synopsys/DesignCompilerBlock.h"
#include "Evaluation/Evidence.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/Request.h"
#include "ExternalTool/ExternalFile.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Evaluation/ProductionRegistry.h"
#include "ExternalTool/LocalConfig.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;

llvm::cl::opt<std::string>
    configPath("config", llvm::cl::Required,
               llvm::cl::desc("canonical resolved configuration JSON"),
               llvm::cl::value_desc("path"));
llvm::cl::opt<std::string> artifactStorePath(
    "artifact-store", llvm::cl::Required,
    llvm::cl::desc("existing content-addressed ArtifactStore root"),
    llvm::cl::value_desc("directory"));
llvm::cl::opt<std::string>
    blobStorePath("blob-store", llvm::cl::Required,
                  llvm::cl::desc("existing content-addressed BlobStore root"),
                  llvm::cl::value_desc("directory"));
llvm::cl::opt<std::string>
    runRoot("run-root", llvm::cl::Required,
            llvm::cl::desc("durable execution journal directory"),
            llvm::cl::value_desc("directory"));
llvm::cl::opt<std::string>
    producerBuild("producer-build", llvm::cl::Required,
                  llvm::cl::desc("semantic producer build identity"),
                  llvm::cl::value_desc("identity"));
llvm::cl::list<std::string> semanticInputFiles(
    "semantic-input",
    llvm::cl::desc("JSON file containing one exact semantic input root"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::opt<std::string> fuReverseSynthesisDataflow(
    "fu-reverse-synthesis-dataflow",
    llvm::cl::desc("canonical Dataflow MLIR graph set for the bounded "
                   "reverse-FU workflow"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<std::string> fuReverseSynthesisEvidence(
    "fu-reverse-synthesis-evidence",
    llvm::cl::desc("derived reverse-FU artifact and replay evidence JSON"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::list<std::string> evidenceInputFiles(
    "preexisting-evidence",
    llvm::cl::desc("JSON file containing one exact preexisting Evidence root"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::list<std::string> jointApplicationScopeFiles(
    "joint-application-scope",
    llvm::cl::desc("canonical Spatial workload-root set admitted as one "
                   "application scope"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::list<std::string> jointSystemRootFiles(
    "joint-system-root",
    llvm::cl::desc("exact Fabric System root admitted to joint Mapping"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::list<std::string> jointPhysicalTimingProfileFiles(
    "joint-physical-timing-profile",
    llvm::cl::desc("exact physical timing profile root for a joint target "
                   "Module"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::opt<bool> jointNormalizedPhysicalTiming(
    "joint-normalized-physical-timing",
    llvm::cl::desc("explicitly publish and bind target-neutral normalized "
                   "timing for every joint target Module"),
    llvm::cl::init(false));
llvm::cl::opt<std::uint64_t> jointPairLimit(
    "joint-pair-limit",
    llvm::cl::desc(
        "maximum joint software/System pairs; zero admits the full product"),
    llvm::cl::init(0));
llvm::cl::opt<std::uint64_t> jointSpatialMappingLimit(
    "joint-spatial-mapping-limit",
    llvm::cl::desc("maximum SpatialMapping roots joined for each pair"),
    llvm::cl::init(0));
llvm::cl::opt<std::uint64_t> jointTechMappingLimit(
    "joint-tech-mapping-limit",
    llvm::cl::desc("maximum TechMapping candidates admitted to Spatial PnR "
                   "for each target Module"),
    llvm::cl::init(0));
llvm::cl::opt<std::string> resolvedConfigOutputPath(
    "resolved-config-output",
    llvm::cl::desc("optional canonical executed ResolvedConfig JSON output"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::opt<std::uint64_t>
    workerCount("workers", llvm::cl::desc("concurrent plan workers"),
                llvm::cl::init(1));
llvm::cl::opt<std::uint64_t> siteCpu("site-cpu",
                                     llvm::cl::desc("site CPU-core capacity"),
                                     llvm::cl::init(1));
llvm::cl::opt<std::uint64_t>
    siteMemory("site-memory-bytes",
               llvm::cl::desc("site memory capacity in bytes"),
               llvm::cl::init(0));
llvm::cl::opt<std::uint64_t>
    siteScratch("site-scratch-bytes",
                llvm::cl::desc("site scratch capacity in bytes"),
                llvm::cl::init(0));
llvm::cl::opt<std::uint64_t>
    workCpu("work-cpu", llvm::cl::desc("default in-process CPU claim"),
            llvm::cl::init(1));
llvm::cl::opt<std::uint64_t>
    workMemory("work-memory-bytes",
               llvm::cl::desc("default in-process memory claim in bytes"),
               llvm::cl::init(0));
llvm::cl::opt<std::uint64_t>
    workScratch("work-scratch-bytes",
                llvm::cl::desc("default in-process scratch claim in bytes"),
                llvm::cl::init(0));
llvm::cl::opt<std::uint64_t> maximumDispatches(
    "maximum-dispatches",
    llvm::cl::desc(
        "stop after this many new work-unit dispatches; zero means unlimited"),
    llvm::cl::init(0));

llvm::cl::opt<std::string> localToolConfigPath(
    "local-tool-config",
    llvm::cl::desc("local external-tool execution configuration"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<bool> prepareOnly(
    "prepare-only",
    llvm::cl::desc("prepare external invocations without executing them"),
    llvm::cl::init(false));
llvm::cl::opt<std::uint64_t>
    externalCpu("external-cpu",
                llvm::cl::desc("CPU claim for external execution"),
                llvm::cl::init(1));
llvm::cl::opt<std::uint64_t>
    externalMemory("external-memory-bytes",
                   llvm::cl::desc("memory claim for external execution"),
                   llvm::cl::init(0));
llvm::cl::opt<std::uint64_t>
    externalScratch("external-scratch-bytes",
                    llvm::cl::desc("scratch claim for external execution"),
                    llvm::cl::init(0));
llvm::cl::opt<bool> claimLicense(
    "claim-license",
    llvm::cl::desc("claim one exact license unit per external binding"),
    llvm::cl::init(false));
llvm::cl::list<std::string> externalBindingCapacities(
    "external-binding-capacity",
    llvm::cl::desc("exact external binding digest and capacity as HEX=UNITS"),
    llvm::cl::value_desc("binding"), llvm::cl::ZeroOrMore);
llvm::cl::opt<std::uint64_t> externalBindingDefaultCapacity(
    "external-binding-default-capacity",
    llvm::cl::desc("units granted to every external tool binding without an "
                   "explicit capacity; zero keeps them inadmissible"),
    llvm::cl::init(0));
llvm::cl::list<std::string> licenseBindingCapacities(
    "license-binding-capacity",
    llvm::cl::desc("exact license binding digest and capacity as HEX=UNITS"),
    llvm::cl::value_desc("binding"), llvm::cl::ZeroOrMore);

enum class GroundTruthCampaignKind : std::uint8_t {
  None,
  Generic,
  Fpa,
};

llvm::cl::opt<GroundTruthCampaignKind> groundTruthCampaign(
    "ground-truth-campaign",
    llvm::cl::desc("apply generic or FPA campaign time gates"),
    llvm::cl::values(clEnumValN(GroundTruthCampaignKind::Generic, "generic",
                                "generic ground-truth collection policy"),
                     clEnumValN(GroundTruthCampaignKind::Fpa, "fpa",
                                "four-hour FPA active-time policy")),
    llvm::cl::init(GroundTruthCampaignKind::None));
llvm::cl::opt<std::uint64_t>
    pilotDispatchCount("pilot-dispatches",
                       llvm::cl::desc("deterministic pilot prefix size"),
                       llvm::cl::init(1));
llvm::cl::opt<std::uint64_t> minimumPilotObservations(
    "minimum-pilot-observations",
    llvm::cl::desc("minimum terminal pilot work units for admission"),
    llvm::cl::init(1));
llvm::cl::opt<std::string> progressPath(
    "progress-jsonl",
    llvm::cl::desc("append removable operational projections to path or '-'"),
    llvm::cl::value_desc("path"), llvm::cl::init("-"));
llvm::cl::opt<std::uint64_t> progressIntervalMilliseconds(
    "progress-interval-ms",
    llvm::cl::desc("live projection interval in milliseconds"),
    llvm::cl::init(1000));

llvm::cl::list<std::string> fpaPhysicalSystemFiles(
    "fpa-physical-system",
    llvm::cl::desc("Fabric System root lowered to RTL, synthesized, and routed "
                   "for the offline FPA ground-truth campaign"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::list<std::string> fpaPhysicalRtlFiles(
    "fpa-physical-rtl",
    llvm::cl::desc("exact portable SpatialCore RTL implementation root "
                   "synthesized and routed for the campaign"),
    llvm::cl::value_desc("path"), llvm::cl::ZeroOrMore);
llvm::cl::opt<std::string> fpaAsicTechnology(
    "fpa-asic-technology",
    llvm::cl::desc("ASIC technology identity of the implementation platform"),
    llvm::cl::value_desc("identity"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaAsicRelease(
    "fpa-asic-release",
    llvm::cl::desc("ASIC library release identity of the platform"),
    llvm::cl::value_desc("identity"), llvm::cl::init(""));
llvm::cl::list<std::string> fpaTechnologyCorners(
    "fpa-technology-corner",
    llvm::cl::desc("technology corner key admitted by the platform"),
    llvm::cl::value_desc("key"), llvm::cl::ZeroOrMore);
llvm::cl::opt<std::string> fpaSelectedCorner(
    "fpa-selected-corner",
    llvm::cl::desc("platform corner key bound to synthesis and routing"),
    llvm::cl::value_desc("key"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaYosysBuild(
    "fpa-yosys-build",
    llvm::cl::desc("exact Yosys version probe output bound as the stable "
                   "synthesis provider build"),
    llvm::cl::value_desc("identity"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaOpenRoadBuild(
    "fpa-openroad-build",
    llvm::cl::desc("exact OpenROAD version probe output bound as the stable "
                   "routing and static FPA provider build"),
    llvm::cl::value_desc("identity"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaPlacementPath(
    "fpa-placement",
    llvm::cl::desc("placement parameter JSON (die/core area, site, pin layers, "
                   "density) for the routed implementation"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaTechnologyLefKey(
    "fpa-technology-lef",
    llvm::cl::desc("local external-file key of the technology LEF"),
    llvm::cl::value_desc("key"), llvm::cl::init("technology"));
llvm::cl::opt<std::string>
    fpaCellLefKey("fpa-cell-lef",
                  llvm::cl::desc("local external-file key of the cell LEF"),
                  llvm::cl::value_desc("key"), llvm::cl::init("cells"));
llvm::cl::opt<std::string> fpaLibertyKey(
    "fpa-liberty",
    llvm::cl::desc("local external-file key of the standard-cell Liberty"),
    llvm::cl::value_desc("key"), llvm::cl::init("timing"));
llvm::cl::opt<std::string> fpaPhysicalOutputPath(
    "fpa-physical-output",
    llvm::cl::desc("physical implementation report JSON output"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::opt<std::string> fpaCollectionTrainingPath(
    "fpa-collection-training",
    llvm::cl::desc("root-set JSON of routed HardwareImplementations in the "
                   "Training partition"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaCollectionValidationPath(
    "fpa-collection-validation",
    llvm::cl::desc("root-set JSON of routed HardwareImplementations in the "
                   "Validation partition"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaCollectionHeldOutPath(
    "fpa-collection-held-out",
    llvm::cl::desc("root-set JSON of routed HardwareImplementations in the "
                   "HeldOut partition"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaOperatingConditionsPath(
    "fpa-operating-conditions",
    llvm::cl::desc("evaluation conditions JSON anchored on the first Training "
                   "implementation"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaCollectionOutputPath(
    "fpa-collection-output",
    llvm::cl::desc("ground-truth collection and leaf characterization report "
                   "JSON output"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::opt<std::string> fpaModelTrainingPath(
    "fpa-model-training",
    llvm::cl::desc("root-set JSON of Training ground-truth Evidence"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaModelValidationPath(
    "fpa-model-validation",
    llvm::cl::desc("root-set JSON of Validation ground-truth Evidence"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaModelHeldOutPath(
    "fpa-model-held-out",
    llvm::cl::desc("root-set JSON of HeldOut ground-truth Evidence"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<std::uint64_t>
    fpaModelSeed("fpa-model-seed",
                 llvm::cl::desc("deterministic GBDT training seed"),
                 llvm::cl::init(1));
llvm::cl::opt<unsigned> fpaModelTrees("fpa-model-trees",
                                      llvm::cl::desc("GBDT tree count"),
                                      llvm::cl::init(4));
llvm::cl::opt<unsigned> fpaModelDepth("fpa-model-depth",
                                      llvm::cl::desc("GBDT maximum depth"),
                                      llvm::cl::init(2));
llvm::cl::opt<unsigned> fpaModelMinimumRows(
    "fpa-model-minimum-rows",
    llvm::cl::desc("minimum Training rows per GBDT leaf"), llvm::cl::init(1));
llvm::cl::opt<unsigned> fpaModelLearningRateNumerator(
    "fpa-model-learning-rate-numerator",
    llvm::cl::desc("GBDT learning-rate numerator"), llvm::cl::init(1));
llvm::cl::opt<unsigned> fpaModelLearningRateDenominator(
    "fpa-model-learning-rate-denominator",
    llvm::cl::desc("GBDT learning-rate denominator"), llvm::cl::init(2));
llvm::cl::opt<std::string> fpaModelMaximumValidationError(
    "fpa-model-maximum-validation-error",
    llvm::cl::desc("Validation error gate as decimal COEFFICIENTeEXPONENT"),
    llvm::cl::value_desc("decimal"), llvm::cl::init("1e0"));
llvm::cl::opt<std::string> fpaModelMaximumHeldOutError(
    "fpa-model-maximum-held-out-error",
    llvm::cl::desc("HeldOut release gate as decimal COEFFICIENTeEXPONENT"),
    llvm::cl::value_desc("decimal"), llvm::cl::init("1e0"));
llvm::cl::opt<std::string> fpaModelOutputPath(
    "fpa-model-output",
    llvm::cl::desc("model training, calibration, and release report JSON"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));
llvm::cl::opt<std::string> fpaReleasedWeightOutputPath(
    "fpa-released-weight-output",
    llvm::cl::desc("root-reference JSON of the released "
                   "EdaPredictionModelWeight bundle"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

volatile std::sig_atomic_t stopSignal = 0;

void requestStopFromSignal(int) { stopSignal = 1; }

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "loom_dse_invalid: " + message);
}

llvm::Expected<ArtifactRootReference> loadRootReference(llvm::StringRef path) {
  return loadArtifactRootReferenceJsonFile(path);
}

llvm::Expected<std::vector<ArtifactRootReference>>
loadRootReferences(llvm::ArrayRef<std::string> paths) {
  std::vector<ArtifactRootReference> roots;
  roots.reserve(paths.size());
  for (const std::string &path : paths) {
    auto root = loadRootReference(path);
    if (!root)
      return llvm::joinErrors(
          invalid(llvm::Twine("cannot load root binding '") + path + "'"),
          root.takeError());
    roots.push_back(std::move(*root));
  }
  llvm::sort(roots, artifactRootReferenceLess);
  if (std::adjacent_find(roots.begin(), roots.end()) != roots.end())
    return invalid("root bindings contain a duplicate reference");
  return roots;
}

llvm::Expected<ArtifactRootReference>
publishFuReverseSynthesisDataflow(llvm::StringRef path,
                                  const ArtifactStore &store) {
  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(), "cannot read %s",
                                   path.str().c_str());
  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registry.insert<::dataflow::DataflowDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
  if (!module)
    return invalid("cannot parse reverse-FU canonical Dataflow MLIR");
  auto dataflow = ::dataflow::finalizeCanonicalDataflow(*module);
  if (!dataflow)
    return dataflow.takeError();
  return ::dataflow::publishCanonicalDataflow(*dataflow, store);
}

llvm::json::Object rootReferenceJson(const ArtifactRootReference &reference) {
  return llvm::json::Object{
      {"schema", reference.schemaIdentity},
      {"schema_version", formatSchemaVersion(reference.schemaVersion)},
      {"artifact", formatArtifactIdentityHex(reference.artifact)}};
}

llvm::json::Array
rootReferenceArray(llvm::ArrayRef<ArtifactRootReference> references) {
  llvm::json::Array result;
  for (const ArtifactRootReference &reference : references)
    result.push_back(rootReferenceJson(reference));
  return result;
}

llvm::Expected<ArtifactRootReference>
publishResolvedConfigReference(const ResolvedConfig &config,
                               const ArtifactStore &artifacts) {
  auto identity = artifacts.put(ResolvedConfig::artifactSchema,
                                canonicalResolvedConfigBytes(config));
  if (!identity)
    return identity.takeError();
  if (*identity != resolvedConfigIdentity(config))
    return invalid("ResolvedConfig publication changed its identity");
  return ArtifactRootReference{ResolvedConfig::artifactSchema.identity.str(),
                               ResolvedConfig::artifactSchema.version,
                               std::move(*identity)};
}

llvm::Error writeJsonObject(llvm::StringRef path, llvm::json::Object report) {
  llvm::SmallString<256> parent(path);
  llvm::sys::path::remove_filename(parent);
  if (!parent.empty())
    if (std::error_code error = llvm::sys::fs::create_directories(parent))
      return llvm::errorCodeToError(error);
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::errorCodeToError(error);
  output << llvm::formatv("{0:2}", llvm::json::Value(std::move(report)))
         << '\n';
  return llvm::Error::success();
}

llvm::Error writeFuReverseSynthesisRejectionEvidence(
    llvm::StringRef path, const ArtifactRootReference &dataflow,
    FuReverseSynthesisFailure failure, llvm::StringRef diagnostic,
    const ResolvedConfig &config, const ArtifactStore &artifacts) {
  auto imported = ::dataflow::importCanonicalDataflow(dataflow, artifacts);
  if (!imported)
    return imported.takeError();
  auto view = imported->view();
  if (!view)
    return view.takeError();
  auto resolvedConfig = publishResolvedConfigReference(config, artifacts);
  if (!resolvedConfig)
    return resolvedConfig.takeError();

  llvm::json::Object report;
  report["projection_kind"] = "fu_reverse_synthesis_workflow";
  report["projection_format"] = 1;
  report["dataflow"] = rootReferenceJson(dataflow);
  report["resolved_config"] = rootReferenceJson(*resolvedConfig);
  report["graph_count"] = view->graphs().size();
  llvm::json::Object typedFailure;
  typedFailure["stage"] = "preflight";
  typedFailure["kind"] = fuReverseSynthesisFailureSpelling(failure);
  typedFailure["diagnostic"] = diagnostic;
  report["typed_failure"] = std::move(typedFailure);
  return writeJsonObject(path, std::move(report));
}

struct FuReverseSynthesisRequiredOutput final {
  llvm::StringLiteral name;
  PlanOutputRef output;
};

std::array<FuReverseSynthesisRequiredOutput, 10>
requiredFuReverseSynthesisOutputs(
    const FuReverseSynthesisCandidateWorkflow &workflow) {
  return {{{"module", workflow.module()},
           {"tech_mappings", workflow.techMappings()},
           {"joint_tech_mapping", workflow.jointTechMapping()},
           {"system", workflow.system()},
           {"physical_timing_profiles", workflow.physicalTimingProfiles()},
           {"configuration_abi", workflow.configurationAbi()},
           {"spatial_mappings", workflow.spatialMappings()},
           {"joint_spatial_mappings", workflow.jointSpatialMappings()},
           {"system_mappings", workflow.systemMappings()},
           {"portable_rtl_implementations",
            workflow.portableRtlImplementations()}}};
}

llvm::json::Object projectFuReverseSynthesisRequiredOutputs(
    const FuReverseSynthesisCandidateWorkflow &workflow,
    const CompletedDsePlanExecution &execution,
    llvm::json::Array &failingRequiredOutputs) {
  llvm::json::Object roots;
  for (const FuReverseSynthesisRequiredOutput &required :
       requiredFuReverseSynthesisOutputs(workflow)) {
    llvm::ArrayRef<ArtifactRootReference> available;
    if (execution.hasOutput(required.output))
      available = execution.resolve(required.output);
    roots[required.name] = rootReferenceArray(available);
    if (available.empty())
      failingRequiredOutputs.push_back(required.name.str());
  }
  return roots;
}

llvm::json::Object
projectInvocationOutcome(const InvocationControllerOutcome &outcome) {
  llvm::json::Object projected;
  if (const auto *selection =
          std::get_if<InvocationCompletedSelection>(&outcome)) {
    projected["kind"] = "completed_selection";
    projected["selected"] = rootReferenceArray(selection->selected);
    projected["satisfied_evidence"] =
        rootReferenceArray(selection->satisfiedEvidence);
    return projected;
  }
  if (const auto *noCandidate =
          std::get_if<InvocationCompletedNoFeasibleCandidate>(&outcome)) {
    projected["kind"] = "completed_no_feasible_candidate";
    projected["satisfied_evidence"] =
        rootReferenceArray(noCandidate->satisfiedEvidence);
    return projected;
  }
  const auto &incomplete = std::get<InvocationIncomplete>(outcome);
  projected["kind"] = "incomplete";
  projected["node"] = incomplete.planNodeOrdinal;
  projected["reason"] = toString(incomplete.reason).str();
  llvm::json::Array obligations;
  for (EvidenceObligationTemplateRef obligation :
       incomplete.unsatisfiedObligations)
    obligations.push_back(obligation.ordinal());
  projected["unsatisfied_obligations"] = std::move(obligations);
  projected["retained_artifacts"] =
      rootReferenceArray(incomplete.retainedArtifacts);
  projected["retained_evidence"] =
      rootReferenceArray(incomplete.retainedEvidence);
  return projected;
}

llvm::json::Array projectGenerateOutcomes(const InvocationManifest &manifest) {
  llvm::json::Array outcomes;
  for (const InvocationGenerateRecord &record : manifest.generateRecords()) {
    llvm::json::Object projected;
    projected["node"] = record.invocation.planNodeOrdinal;
    const CandidateGeneratorDescriptor *descriptor =
        record.invocation.generatorBinding.descriptorRef().descriptor();
    if (descriptor) {
      projected["generator"] = descriptor->spelling.str();
      projected["generator_kind"] = descriptor->kind.ordinal();
    }
    if (!record.completed) {
      projected["kind"] = "incomplete";
      if (record.invocation.incompleteReason)
        projected["reason"] = candidateGeneratorIncompleteReasonSpelling(
                                  *record.invocation.incompleteReason)
                                  .str();
    } else if (record.invocation.infeasibilityProof) {
      projected["kind"] = "proven_infeasible";
      projected["proof_kind"] =
          record.invocation.infeasibilityProof->kind.ordinal();
      projected["proof_witness"] =
          llvm::toHex(record.invocation.infeasibilityProof->witness, true);
    } else {
      projected["kind"] = "completed";
    }
    outcomes.push_back(std::move(projected));
  }
  return outcomes;
}

llvm::Error writeFuReverseSynthesisEvidence(
    llvm::StringRef path, const FuReverseSynthesisCandidateWorkflow &workflow,
    const DsePlanExecutionOutcome &outcome,
    const InvocationManifestReference &invocation,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  llvm::json::Object report;
  report["projection_kind"] = "fu_reverse_synthesis_workflow";
  report["projection_format"] = 1;
  report["dataflow"] = rootReferenceJson(workflow.dataflow());
  report["resolved_config"] = rootReferenceJson(invocation.resolvedConfig());

  auto manifest = importInvocationManifest(invocation, artifacts, blobs);
  if (!manifest)
    return manifest.takeError();
  auto dataflow =
      ::dataflow::importCanonicalDataflow(workflow.dataflow(), artifacts);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  report["graph_count"] = dataflowView->graphs().size();

  llvm::json::Object invocationProjection;
  invocationProjection["run_key"] =
      llvm::toHex(invocation.occurrence().runKey.bytes(), true);
  invocationProjection["occurrence"] =
      invocation.occurrence().occurrenceOrdinal;
  invocationProjection["manifest_blob"] =
      formatBlobDigestHex(invocation.blob());
  invocationProjection["outcome"] =
      projectInvocationOutcome(manifest->outcome());
  invocationProjection["generate_outcomes"] =
      projectGenerateOutcomes(*manifest);
  report["invocation"] = std::move(invocationProjection);

  const CompletedDsePlanExecution *available = nullptr;
  bool incompleteExecution = false;
  if (const auto *completed =
          std::get_if<CompletedDsePlanExecution>(&outcome)) {
    available = completed;
  } else {
    available =
        &std::get<IncompleteDsePlanExecution>(outcome).availableExecution();
    incompleteExecution = true;
  }
  std::uint64_t dispatchCount = 0;
  for (std::size_t ordinal = 0;
       ordinal != available->generateInvocations().size(); ++ordinal)
    dispatchCount += available->generateInvocationWasDispatched(ordinal);
  report["generate_invocation_count"] = available->generateInvocations().size();
  report["dispatch_count"] = dispatchCount;
  report["covered_graph_count"] =
      available->resolve(workflow.techMappings()).size();

  llvm::json::Array failingRequiredOutputs;
  llvm::json::Object workflowProjection;
  workflowProjection["required_outputs"] =
      projectFuReverseSynthesisRequiredOutputs(workflow, *available,
                                               failingRequiredOutputs);
  workflowProjection["failing_required_outputs"] =
      std::move(failingRequiredOutputs);
  if (incompleteExecution) {
    workflowProjection["disposition"] = "incomplete";
  } else {
    auto disposition = classifyFuReverseSynthesisWorkflow(workflow, *available);
    if (!disposition)
      return disposition.takeError();
    if (*disposition ==
        FuReverseSynthesisWorkflowDisposition::RequiredOutputMissing) {
      workflowProjection["disposition"] = "required_output_missing";
    } else {
      auto projected = projectFuReverseSynthesisWorkflowArtifacts(
          workflow, *available, artifacts, blobs);
      if (!projected)
        return projected.takeError();
      workflowProjection["disposition"] = "complete_candidate";
    }
  }
  report["workflow"] = std::move(workflowProjection);

  return writeJsonObject(path, std::move(report));
}

llvm::Expected<std::vector<ArtifactRootReference>>
publishNormalizedPhysicalTimingProfiles(
    llvm::ArrayRef<ArtifactRootReference> systems, const ArtifactStore &store) {
  std::map<ArtifactIdentity::Storage, ArtifactRootReference> profiles;
  for (const ArtifactRootReference &systemReference : systems) {
    auto artifact =
        loom::fabric::importEntireFabricRoot(systemReference, store);
    if (!artifact)
      return artifact.takeError();
    auto system = loom::fabric::requireSystemRoot(artifact->view());
    if (!system)
      return system.takeError();
    for (const loom::fabric::AccCoreOccurrenceRef core :
         system->artifact().accCoreOccurrences()) {
      const auto target = system->spatialCoreTarget(core);
      if (!target || target->dependencyOrdinal >=
                         system->artifact().importedModules().size())
        return invalid("joint System AccCore target does not resolve");
      const auto &module =
          system->artifact().importedModules()[target->dependencyOrdinal];
      if (profiles.count(module.identity().bytes()))
        continue;
      auto profile =
          loom::fabric::projectNormalizedFabricPhysicalTimingProfile(module);
      if (!profile)
        return profile.takeError();
      auto published =
          loom::fabric::publishFabricPhysicalTimingProfile(*profile, store);
      if (!published)
        return published.takeError();
      profiles.emplace(module.identity().bytes(), std::move(*published));
    }
  }
  std::vector<ArtifactRootReference> result;
  result.reserve(profiles.size());
  for (auto &[identity, reference] : profiles) {
    (void)identity;
    result.push_back(std::move(reference));
  }
  llvm::sort(result, artifactRootReferenceLess);
  return result;
}

void canonicalizeRootUnion(std::vector<ArtifactRootReference> &roots) {
  llvm::sort(roots, artifactRootReferenceLess);
  roots.erase(std::unique(roots.begin(), roots.end()), roots.end());
}

llvm::Error writeResolvedConfig(llvm::StringRef path,
                                const ResolvedConfig &config) {
  std::error_code error;
  llvm::raw_fd_ostream output(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return llvm::errorCodeToError(error);
  output << canonicalResolvedConfigJson(config) << '\n';
  return llvm::Error::success();
}

llvm::Expected<std::uint64_t> parsePositiveInteger(llvm::StringRef text) {
  std::uint64_t value = 0;
  if (text.getAsInteger(10, value) || value == 0)
    return invalid("resource capacity must be a positive integer");
  return value;
}

llvm::Expected<std::vector<CountedSiteResource>>
parseCapacities(llvm::ArrayRef<std::string> spellings, bool license) {
  std::vector<CountedSiteResource> capacities;
  capacities.reserve(spellings.size());
  for (const std::string &spelling : spellings) {
    const std::size_t separator = spelling.rfind('=');
    if (separator == std::string::npos)
      return invalid("binding capacity must use HEX=UNITS");
    auto digest =
        parseBlobDigestHex(llvm::StringRef(spelling).take_front(separator));
    if (!digest)
      return digest.takeError();
    auto units = parsePositiveInteger(
        llvm::StringRef(spelling).drop_front(separator + 1));
    if (!units)
      return units.takeError();
    capacities.push_back({license
                              ? SiteResourceKey::licenseBinding(*digest)
                              : SiteResourceKey::externalToolBinding(*digest),
                          *units});
  }
  llvm::sort(capacities,
             [](const CountedSiteResource &lhs,
                const CountedSiteResource &rhs) { return lhs.key < rhs.key; });
  for (std::size_t index = 1; index < capacities.size(); ++index)
    if (capacities[index - 1].key == capacities[index].key)
      return invalid("binding capacities contain a duplicate digest");
  return capacities;
}

llvm::StringRef admissionReason(CampaignAdmissionFailureReason reason) {
  switch (reason) {
  case CampaignAdmissionFailureReason::InsufficientPilotObservations:
    return "insufficient_pilot_observations";
  case CampaignAdmissionFailureReason::PreparedAttemptIncomplete:
    return "prepared_attempt_incomplete";
  case CampaignAdmissionFailureReason::SampleActiveWallTimeLimit:
    return "sample_active_wall_time_limit";
  case CampaignAdmissionFailureReason::CampaignActiveWallTimeLimit:
    return "campaign_active_wall_time_limit";
  case CampaignAdmissionFailureReason::EstimatedCompletionLimit:
    return "estimated_completion_limit";
  case CampaignAdmissionFailureReason::ThroughputUnavailable:
    return "throughput_unavailable";
  }
  llvm_unreachable("closed campaign admission reason");
}

int reportPlanOutcome(const DsePlanExecutionOutcome &outcome) {
  if (std::holds_alternative<CompletedDsePlanExecution>(outcome)) {
    llvm::errs() << "campaign_result=completed search_complete=true\n";
    return EXIT_SUCCESS;
  }
  const auto &incomplete = std::get<IncompleteDsePlanExecution>(outcome);
  if (!incomplete.executionStopped()) {
    llvm::errs() << "campaign_result=completed search_complete=false node="
                 << incomplete.nodeOrdinal()
                 << " reason=" << toString(incomplete.reason()) << '\n';
    return EXIT_SUCCESS;
  }
  llvm::errs() << "campaign_result=incomplete node=" << incomplete.nodeOrdinal()
               << " search_complete=false reason="
               << toString(incomplete.reason()) << '\n';
  return 2;
}

std::size_t outputCount(const CompletedDsePlanExecution &completed,
                        llvm::ArrayRef<PlanOutputRef> outputs) {
  std::size_t count = 0;
  for (PlanOutputRef output : outputs)
    if (completed.hasOutput(output))
      count += completed.resolve(output).size();
  return count;
}

void reportJointOutputs(const DsePlanExecutionOutcome &outcome,
                        llvm::ArrayRef<JointDesignPlanPair> pairs) {
  const CompletedDsePlanExecution *completed =
      std::get_if<CompletedDsePlanExecution>(&outcome);
  if (!completed)
    completed =
        &std::get<IncompleteDsePlanExecution>(outcome).availableExecution();
  for (std::size_t index = 0; index != pairs.size(); ++index) {
    const JointDesignPlanPair &pair = pairs[index];
    llvm::errs() << "joint_pair=" << index << " tech_mappings="
                 << outputCount(*completed, pair.techMappings)
                 << " spatial_mappings="
                 << outputCount(*completed, pair.spatialMappings)
                 << " system_mappings="
                 << (completed->hasOutput(pair.systemMappings)
                         ? completed->resolve(pair.systemMappings).size()
                         : 0)
                 << '\n';
  }
}

llvm::Expected<evaluation::DecimalValue>
parseDecimalSpelling(llvm::StringRef spelling) {
  auto [coefficientText, exponentText] = spelling.split('e');
  std::int64_t coefficient = 0;
  std::int64_t exponent = 0;
  if (coefficientText.getAsInteger(10, coefficient) ||
      (!exponentText.empty() && exponentText.getAsInteger(10, exponent)))
    return invalid("decimal must be spelled COEFFICIENTeEXPONENT");
  return evaluation::DecimalValue::get(coefficient, exponent);
}

llvm::json::Object decimalJson(const evaluation::DecimalValue &value) {
  return llvm::json::Object{{"coefficient", value.coefficient()},
                            {"base10_exponent", value.base10Exponent()}};
}

llvm::Expected<std::vector<ArtifactRootReference>>
loadRootSet(llvm::StringRef path) {
  auto roots = loadArtifactRootReferenceSetJsonFile(path);
  if (!roots)
    return llvm::joinErrors(
        invalid(llvm::Twine("cannot load root set '") + path + "'"),
        roots.takeError());
  return roots;
}

llvm::Expected<ExternalFileFingerprint>
fingerprintLocalExternalFile(const external_tool::LocalToolConfig &local,
                             llvm::StringRef key) {
  const auto found = local.externalFiles.find(key.str());
  if (found == local.externalFiles.end())
    return invalid(llvm::Twine("local tool configuration has no external file "
                               "'") +
                   key + "'");
  return external_tool::fingerprintExternalFile(found->second);
}

llvm::json::Array
availableRoots(const CompletedDsePlanExecution &execution, PlanOutputRef output) {
  return execution.hasOutput(output) ? rootReferenceArray(execution.resolve(output))
                                     : llvm::json::Array{};
}

llvm::StringRef leafReasonSpelling(FpaCharacterizationUnavailableReason reason) {
  switch (reason) {
  case FpaCharacterizationUnavailableReason::RoutedAsicImplementationUnavailable:
    return "routed_asic_implementation_unavailable";
  case FpaCharacterizationUnavailableReason::IndependentlyRoutedLeafUnavailable:
    return "independently_routed_leaf_unavailable";
  }
  llvm_unreachable("closed FPA characterization unavailability reason");
}

llvm::Expected<llvm::json::Object>
leafCharacterizationJson(const FpaCharacterizationUnavailable &leaf) {
  llvm::json::Object result;
  const loom::fabric::FabricModuleDomainMemberRef &member = leaf.target.leaf;
  loom::fabric::FabricByteWriter writer;
  loom::fabric::encodeFabricRef(writer, member);
  result["canonical_ref_hex"] = llvm::toHex(writer.take(), true);
  switch (member.kind()) {
  case loom::fabric::FabricModuleDomainMemberKind::Boundary: {
    const auto &boundary =
        std::get<loom::fabric::FabricModuleBoundaryEndpointRef>(member.payload);
    result["leaf_kind"] = "transport_boundary";
    result["direction"] =
        boundary.direction == loom::fabric::FabricPortDirection::Input
            ? "input"
            : "output";
    result["ordinal"] = boundary.ordinal;
    break;
  }
  case loom::fabric::FabricModuleDomainMemberKind::Internal: {
    const auto &owner =
        std::get<loom::fabric::FabricModulePhysicalOwnerRef>(member.payload);
    switch (owner.kind()) {
    case loom::fabric::FabricModulePhysicalOwnerKind::PeOccurrence:
      result["leaf_kind"] = "pe_occurrence";
      result["entity"] =
          std::get<loom::fabric::FabricPeOccurrenceRef>(owner.payload()).id();
      break;
    case loom::fabric::FabricModulePhysicalOwnerKind::FuOccurrence:
      result["leaf_kind"] = "fu_occurrence";
      result["entity"] =
          std::get<loom::fabric::FabricFuOccurrenceRef>(owner.payload()).id();
      break;
    case loom::fabric::FabricModulePhysicalOwnerKind::MemoryOccurrence:
      result["leaf_kind"] = "memory_occurrence";
      result["entity"] =
          std::get<loom::fabric::FabricMemoryOccurrenceRef>(owner.payload())
              .id();
      break;
    case loom::fabric::FabricModulePhysicalOwnerKind::SwitchOccurrence:
      result["leaf_kind"] = "switch_occurrence";
      result["entity"] =
          std::get<loom::fabric::FabricSwitchOccurrenceRef>(owner.payload())
              .id();
      break;
    default:
      return invalid("leaf breadth admitted a non-leaf physical owner");
    }
    break;
  }
  }
  result["reason"] = leafReasonSpelling(leaf.reason);
  return result;
}

struct FpaCollectionPartitionInputs final {
  llvm::StringLiteral name;
  std::vector<ArtifactRootReference> hardwareImplementations;
  PlanOutputRef evidence;
};

llvm::Expected<llvm::json::Object> fpaPartitionReport(
    const FpaCollectionPartitionInputs &partition,
    const CompletedDsePlanExecution &execution, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  llvm::json::Object result;
  result["hardware_implementations"] =
      rootReferenceArray(partition.hardwareImplementations);
  llvm::json::Array evidence;
  if (execution.hasOutput(partition.evidence))
    for (const ArtifactRootReference &root :
         execution.resolve(partition.evidence)) {
      auto sample = evaluation::models::importFpaTrainingEvidenceSample(
          root, artifacts, blobs);
      if (!sample)
        return sample.takeError();
      evidence.push_back(llvm::json::Object{
          {"evidence", rootReferenceJson(root)},
          {"limiting_clock_frequency_hz",
           decimalJson(sample->observation.limitingClockFrequency)},
          {"total_area_square_meters", decimalJson(sample->observation.totalArea)},
          {"dynamic_power_watts", decimalJson(sample->observation.dynamicPower)},
          {"leakage_power_watts", decimalJson(sample->observation.leakagePower)},
          {"sample_group_key_hex", llvm::toHex(sample->sampleGroupKey, true)},
          {"ground_truth_target_key_hex",
           llvm::toHex(sample->groundTruthTargetKey, true)}});
    }
  result["evidence"] = std::move(evidence);
  llvm::json::Array leafReports;
  for (const ArtifactRootReference &hardware :
       partition.hardwareImplementations) {
    auto breadth =
        assessFpaLeafCharacterizationBreadth(hardware, artifacts, blobs);
    if (!breadth)
      return breadth.takeError();
    llvm::json::Array leaves;
    std::map<std::string, std::uint64_t> counts;
    for (const FpaCharacterizationUnavailable &leaf : *breadth) {
      auto projected = leafCharacterizationJson(leaf);
      if (!projected)
        return projected.takeError();
      ++counts[(*projected->getString("leaf_kind")).str() + ":" +
               leafReasonSpelling(leaf.reason).str()];
      leaves.push_back(std::move(*projected));
    }
    llvm::json::Object summary;
    for (const auto &[key, count] : counts)
      summary[key] = count;
    leafReports.push_back(llvm::json::Object{
        {"hardware_implementation", rootReferenceJson(hardware)},
        {"leaf_count", static_cast<std::uint64_t>(breadth->size())},
        {"summary", std::move(summary)},
        {"leaves", std::move(leaves)}});
  }
  result["leaf_characterization"] = std::move(leafReports);
  return result;
}

llvm::Expected<llvm::json::Array> calibrationMetricsJson(
    const ArtifactRootReference &calibrationEvidence,
    const ArtifactRootReference &bundle,
    llvm::ArrayRef<ArtifactRootReference> sourceEvidence,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto resolution =
      evaluation::models::resolveFpaCalibrationCaseArtifactResolution(
          bundle, sourceEvidence, artifacts, blobs);
  if (!resolution)
    return resolution.takeError();
  auto requestReference = evaluation::importEvaluationEvidenceRequestReference(
      calibrationEvidence, artifacts);
  if (!requestReference)
    return requestReference.takeError();
  auto request = evaluation::importEvaluationRequest(*requestReference,
                                                     *resolution, artifacts, blobs);
  if (!request)
    return request.takeError();
  auto evidence = evaluation::importEvaluationEvidence(
      calibrationEvidence, *resolution, artifacts, blobs);
  if (!evidence)
    return evidence.takeError();
  llvm::json::Array metrics;
  const auto *completed =
      std::get_if<evaluation::CompletedEvidence>(&evidence->outcome());
  if (!completed)
    return metrics;
  for (std::size_t ordinal = 0;
       ordinal != completed->metricResults.size() &&
       ordinal != request->metricRequests().size();
       ++ordinal) {
    const evaluation::MetricRequest &metric = request->metricRequests()[ordinal];
    llvm::json::Object entry{{"metric", toString(metric.query().metric)}};
    for (const evaluation::EvaluationCondition &condition : metric.conditions())
      if (const auto *quantile =
              std::get_if<evaluation::QuantileCondition>(&condition.payload))
        entry["quantile"] = llvm::json::Object{
            {"numerator", quantile->probability.numerator()},
            {"denominator", quantile->probability.denominator()}};
    const auto *point = std::get_if<evaluation::PointObservation>(
        &completed->metricResults[ordinal].observation);
    const auto *decimal =
        point ? std::get_if<evaluation::DecimalValue>(&point->value) : nullptr;
    entry["value"] = decimal ? llvm::json::Value(decimalJson(*decimal))
                             : llvm::json::Value(nullptr);
    metrics.push_back(std::move(entry));
  }
  return metrics;
}

llvm::Expected<int> run() {
  loom::fabric::FabricArtifactImportSession fabricImportSession;
  llvm::scope_exit emitFabricImportStatistics([&] {
    loom::fabric::emitFabricArtifactImportSessionStatistics(
        loom::fabric::FabricArtifactImportVerificationDomain::SourceInvocation,
        loom::InvocationDiagnosticStage::SpatialPnr,
        fabricImportSession.statistics());
  });
  if (progressIntervalMilliseconds == 0)
    return invalid("progress interval must be positive");
  if (prepareOnly && localToolConfigPath.empty())
    return invalid("prepare-only requires a local tool configuration");
  const bool authorJointPlan =
      !jointApplicationScopeFiles.empty() || !jointSystemRootFiles.empty();
  const bool authorFuReverseSynthesis = !fuReverseSynthesisDataflow.empty() ||
                                        !fuReverseSynthesisEvidence.empty();
  if (fuReverseSynthesisDataflow.empty() != fuReverseSynthesisEvidence.empty())
    return invalid("reverse-FU authoring requires both its Dataflow input and "
                   "evidence output");
  if (authorFuReverseSynthesis && authorJointPlan)
    return invalid("reverse-FU and joint plan authoring are mutually "
                   "exclusive");
  if (authorFuReverseSynthesis && !semanticInputFiles.empty())
    return invalid("reverse-FU authoring owns its exact semantic input");
  if (authorFuReverseSynthesis &&
      groundTruthCampaign.getValue() != GroundTruthCampaignKind::None)
    return invalid("reverse-FU authoring does not admit a ground-truth "
                   "campaign wrapper");
  const bool authorFpaPhysical =
      !fpaPhysicalSystemFiles.empty() || !fpaPhysicalRtlFiles.empty();
  const bool authorFpaCollection = !fpaCollectionTrainingPath.empty() ||
                                   !fpaCollectionValidationPath.empty() ||
                                   !fpaCollectionHeldOutPath.empty();
  const bool authorFpaModel = !fpaModelTrainingPath.empty() ||
                              !fpaModelValidationPath.empty() ||
                              !fpaModelHeldOutPath.empty();
  if (authorFpaPhysical + authorFpaCollection + authorFpaModel > 1)
    return invalid("FPA physical, collection, and model authoring are "
                   "mutually exclusive");
  if ((authorFpaPhysical || authorFpaCollection || authorFpaModel) &&
      (authorJointPlan || authorFuReverseSynthesis))
    return invalid("FPA campaign authoring owns its exact plan");
  if (authorFpaPhysical && localToolConfigPath.empty())
    return invalid("FPA physical implementation requires the local tool "
                   "configuration that names its technology files");
  if (authorFpaCollection &&
      (fpaCollectionTrainingPath.empty() || fpaCollectionValidationPath.empty() ||
       fpaCollectionHeldOutPath.empty() || fpaOperatingConditionsPath.empty()))
    return invalid("FPA collection requires Training, Validation, and HeldOut "
                   "root sets and operating conditions");
  if (authorFpaCollection &&
      groundTruthCampaign.getValue() != GroundTruthCampaignKind::Fpa)
    return invalid("FPA collection runs only under the FPA campaign policy");
  if (authorFpaModel &&
      (fpaModelTrainingPath.empty() || fpaModelValidationPath.empty() ||
       fpaModelHeldOutPath.empty()))
    return invalid("FPA model release requires Training, Validation, and "
                   "HeldOut Evidence root sets");
  if (jointApplicationScopeFiles.empty() != jointSystemRootFiles.empty())
    return invalid("joint plan authoring requires both application scopes "
                   "and System root frontiers");
  if (authorJointPlan && jointSpatialMappingLimit == 0)
    return invalid("joint plan authoring requires a positive SpatialMapping "
                   "join limit");
  if (!authorJointPlan && (jointPairLimit.getNumOccurrences() != 0 ||
                           jointSpatialMappingLimit.getNumOccurrences() != 0))
    return invalid("joint policy requires joint software and System roots");
  if (!llvm::sys::fs::is_directory(artifactStorePath) ||
      !llvm::sys::fs::is_directory(blobStorePath))
    return invalid("ArtifactStore and BlobStore roots must already exist");
  if (std::error_code error = llvm::sys::fs::create_directories(runRoot))
    return llvm::errorCodeToError(error);

  if (llvm::Error error = registerProductionDseOwners())
    return error;
  // The offline EDA providers are production owners of loom-dse: the plan
  // executor dispatches them only through the configured external site.
  if (llvm::Error error =
          eda::synopsys::registerDesignCompilerGateNetlistCandidateGenerator())
    return error;
  if (llvm::Error error =
          eda::synopsys::registerDesignCompilerBlockGateNetlistCandidateGenerator())
    return error;
  if (llvm::Error error = eda::synopsys::
          registerDesignCompilerHierarchicalBlockGateNetlistCandidateGenerator())
    return error;
  if (llvm::Error error = eda::synopsys::
          registerDesignCompilerPortableGateImplementationCandidateGenerator())
    return error;
  if (llvm::Error error =
          eda::open_source::registerYosysBlockGateNetlistCandidateGenerator())
    return error;
  if (llvm::Error error = eda::open_source::
          registerYosysHierarchicalBlockGateNetlistCandidateGenerator())
    return error;
  if (llvm::Error error =
          eda::open_source::registerYosysGateNetlistCandidateGenerator())
    return error;
  if (llvm::Error error =
          eda::open_source::registerOpenRoadRoutedCandidateGenerator())
    return error;
  if (llvm::Error error =
          eda::open_source::registerOpenRoadStaticFpaEvaluationProvider())
    return error;
  auto config = loadResolvedConfig(configPath);
  if (!config)
    return config.takeError();
  if (!fpaOpenRoadBuild.empty()) {
    const evaluation::models::OpenRoadStaticFpaProviderBinding binding{
        fpaOpenRoadBuild};
    if (config->evaluation.openRoadStaticFpa &&
        !(*config->evaluation.openRoadStaticFpa == binding))
      return invalid("resolved configuration binds another OpenROAD static "
                     "FPA provider build");
    config->evaluation.openRoadStaticFpa = binding;
  }
  auto semanticInputs = loadRootReferences(semanticInputFiles);
  if (!semanticInputs)
    return semanticInputs.takeError();
  auto preexistingEvidence = loadRootReferences(evidenceInputFiles);
  if (!preexistingEvidence)
    return preexistingEvidence.takeError();

  ArtifactStore artifacts(artifactStorePath);
  BlobStore blobs(blobStorePath);
  std::optional<FuReverseSynthesisCandidateWorkflow> fuWorkflow;
  if (authorFuReverseSynthesis) {
    auto dataflow = publishFuReverseSynthesisDataflow(
        fuReverseSynthesisDataflow, artifacts);
    if (!dataflow)
      return dataflow.takeError();
    auto workflow =
        buildFuReverseSynthesisCandidateWorkflow(*dataflow, *config, artifacts);
    if (!workflow) {
      std::optional<FuReverseSynthesisFailure> failure;
      std::string diagnostic;
      llvm::Error remaining = llvm::handleErrors(
          workflow.takeError(), [&](const FuReverseSynthesisError &error) {
            failure = error.failure();
            diagnostic = error.diagnostic().str();
          });
      if (remaining)
        return std::move(remaining);
      if (!failure)
        return invalid("reverse-FU admission lost its typed failure");
      llvm::Error reportError = writeFuReverseSynthesisRejectionEvidence(
          fuReverseSynthesisEvidence, *dataflow, *failure, diagnostic, *config,
          artifacts);
      return llvm::joinErrors(
          llvm::make_error<FuReverseSynthesisError>(*failure, diagnostic),
          std::move(reportError));
    }
    semanticInputs->push_back(*dataflow);
    *config = workflow->resolvedConfig();
    fuWorkflow.emplace(std::move(*workflow));
  }
  std::vector<JointDesignPlanPair> jointPairOutputs;
  if (authorJointPlan) {
    std::vector<std::vector<ArtifactRootReference>> applicationScopes;
    applicationScopes.reserve(jointApplicationScopeFiles.size());
    for (const std::string &path : jointApplicationScopeFiles) {
      auto scope = loadArtifactRootReferenceSetJsonFile(path);
      if (!scope)
        return llvm::joinErrors(
            invalid(llvm::Twine("cannot load application scope '") + path +
                    "'"),
            scope.takeError());
      applicationScopes.push_back(std::move(*scope));
    }
    auto systems = loadRootReferences(jointSystemRootFiles);
    if (!systems)
      return systems.takeError();
    if (jointNormalizedPhysicalTiming &&
        !jointPhysicalTimingProfileFiles.empty())
      return invalid("joint normalized timing and exact timing profile roots "
                     "are mutually exclusive");
    if (!jointNormalizedPhysicalTiming &&
        jointPhysicalTimingProfileFiles.empty())
      return invalid("joint plan requires exact physical timing profile roots "
                     "or --joint-normalized-physical-timing");
    llvm::Expected<std::vector<ArtifactRootReference>> timingProfiles =
        jointNormalizedPhysicalTiming
            ? publishNormalizedPhysicalTimingProfiles(*systems, artifacts)
            : loadRootReferences(jointPhysicalTimingProfileFiles);
    if (!timingProfiles)
      return timingProfiles.takeError();
    if (applicationScopes.size() >
        std::numeric_limits<std::uint64_t>::max() / systems->size())
      return invalid("joint pair count overflows u64");
    const std::uint64_t completePairCount =
        static_cast<std::uint64_t>(applicationScopes.size()) *
        static_cast<std::uint64_t>(systems->size());
    const std::uint64_t pairLimit =
        jointPairLimit == 0 ? completePairCount : jointPairLimit;
    auto policy = JointDesignPolicy::get(
        applicationScopes.size(), systems->size(), pairLimit,
        jointTechMappingLimit, jointSpatialMappingLimit);
    if (!policy)
      return policy.takeError();
    auto plan = buildJointDesignExplorationPlan(
        {std::move(applicationScopes), std::move(*systems)}, *timingProfiles,
        *policy, *config, artifacts);
    if (!plan)
      return plan.takeError();
    std::vector<ArtifactRootReference> jointInputs =
        projectJointDesignSemanticInputs(*plan);
    semanticInputs->insert(semanticInputs->end(), jointInputs.begin(),
                           jointInputs.end());
    canonicalizeRootUnion(*semanticInputs);
    llvm::errs() << "joint_frontier_eligible="
                 << plan->frontier.eligiblePairCount
                 << " retained=" << plan->frontier.pairs.size()
                 << " truncated=" << (plan->frontier.truncated ? 1 : 0)
                 << " analytic_evaluated="
                 << plan->frontier.analyticEvaluatedPairCount
                 << " analytic_deferred="
                 << plan->frontier.analyticDeferredPairCount << '\n';
    jointPairOutputs = plan->pairOutputs;
    *config = std::move(plan->resolvedConfig);
  }
  std::optional<FpaPhysicalImplementationPlan> fpaPhysical;
  if (authorFpaPhysical) {
    auto local = external_tool::loadLocalToolConfig(localToolConfigPath);
    if (!local)
      return local.takeError();
    auto systems = loadRootReferences(fpaPhysicalSystemFiles);
    if (!systems)
      return systems.takeError();
    auto rtlImplementations = loadRootReferences(fpaPhysicalRtlFiles);
    if (!rtlImplementations)
      return rtlImplementations.takeError();
    if (fpaAsicTechnology.empty() || fpaAsicRelease.empty() ||
        fpaTechnologyCorners.empty())
      return invalid("FPA physical implementation requires the ASIC target "
                     "and its corners");
    if (!rtlImplementations->empty() &&
        (fpaYosysBuild.empty() || fpaOpenRoadBuild.empty() ||
         fpaPlacementPath.empty()))
      return invalid("FPA synthesis and routing require the provider builds "
                     "and placement parameters");
    eda::open_source::OpenRoadPlacementParameters placement{};
    if (!fpaPlacementPath.empty()) {
      auto placementText = llvm::MemoryBuffer::getFile(fpaPlacementPath);
      if (!placementText)
        return llvm::createStringError(placementText.getError(),
                                       "cannot read %s",
                                       fpaPlacementPath.c_str());
      auto parsed = eda::open_source::parseOpenRoadPlacementParametersJson(
          (*placementText)->getBuffer());
      if (!parsed)
        return parsed.takeError();
      placement = std::move(*parsed);
    }
    auto technologyLef = fingerprintLocalExternalFile(*local, fpaTechnologyLefKey);
    if (!technologyLef)
      return technologyLef.takeError();
    auto cellLef = fingerprintLocalExternalFile(*local, fpaCellLefKey);
    if (!cellLef)
      return cellLef.takeError();
    auto liberty = fingerprintLocalExternalFile(*local, fpaLibertyKey);
    if (!liberty)
      return liberty.takeError();
    auto plan = buildFpaPhysicalImplementationPlan(
        FpaPhysicalImplementationRequest{
            std::move(*systems),
            std::move(*rtlImplementations),
            loom::platform::AsicTarget{fpaAsicTechnology, fpaAsicRelease},
            std::vector<std::string>(fpaTechnologyCorners.begin(),
                                     fpaTechnologyCorners.end()),
            fpaSelectedCorner.empty() ? fpaTechnologyCorners.front()
                                      : fpaSelectedCorner.getValue(),
            fpaYosysBuild,
            fpaOpenRoadBuild,
            placement,
            *technologyLef,
            *cellLef,
            *liberty},
        *config, artifacts);
    if (!plan)
      return plan.takeError();
    semanticInputs->insert(semanticInputs->end(), plan->semanticInputs.begin(),
                           plan->semanticInputs.end());
    canonicalizeRootUnion(*semanticInputs);
    *config = plan->resolvedConfig;
    fpaPhysical.emplace(std::move(*plan));
  }
  std::optional<FpaGroundTruthCollectionPlan> fpaCollection;
  std::vector<FpaCollectionPartitionInputs> fpaCollectionPartitions;
  if (authorFpaCollection) {
    auto training = loadRootSet(fpaCollectionTrainingPath);
    if (!training)
      return training.takeError();
    auto validation = loadRootSet(fpaCollectionValidationPath);
    if (!validation)
      return validation.takeError();
    auto heldOut = loadRootSet(fpaCollectionHeldOutPath);
    if (!heldOut)
      return heldOut.takeError();
    auto conditionsText = llvm::MemoryBuffer::getFile(fpaOperatingConditionsPath);
    if (!conditionsText)
      return llvm::createStringError(conditionsText.getError(),
                                     "cannot read %s",
                                     fpaOperatingConditionsPath.c_str());
    auto conditions =
        evaluation::parseEvaluationConditions((*conditionsText)->getBuffer());
    if (!conditions)
      return conditions.takeError();
    auto plan = buildFpaGroundTruthCollectionPlan(
        FpaGroundTruthPlanInputs{{*training}, {*validation}, {*heldOut},
                                 std::move(*conditions)},
        *config, artifacts, blobs);
    if (!plan)
      return plan.takeError();
    for (const auto &partition : {*training, *validation, *heldOut})
      semanticInputs->insert(semanticInputs->end(), partition.begin(),
                             partition.end());
    canonicalizeRootUnion(*semanticInputs);
    fpaCollectionPartitions = {
        {"training", std::move(*training), plan->trainingEvidence},
        {"validation", std::move(*validation), plan->validationEvidence},
        {"held_out", std::move(*heldOut), plan->heldOutEvidence}};
    *config = plan->resolvedConfig;
    fpaCollection.emplace(std::move(*plan));
  }
  std::optional<ResolvedGroundTruthPlan> fpaModel;
  std::array<std::vector<ArtifactRootReference>, 3> fpaModelEvidence;
  if (authorFpaModel) {
    auto training = loadRootSet(fpaModelTrainingPath);
    if (!training)
      return training.takeError();
    auto validation = loadRootSet(fpaModelValidationPath);
    if (!validation)
      return validation.takeError();
    auto heldOut = loadRootSet(fpaModelHeldOutPath);
    if (!heldOut)
      return heldOut.takeError();
    auto validationGate = parseDecimalSpelling(fpaModelMaximumValidationError);
    if (!validationGate)
      return validationGate.takeError();
    auto heldOutGate = parseDecimalSpelling(fpaModelMaximumHeldOutError);
    if (!heldOutGate)
      return heldOutGate.takeError();
    fpaModelEvidence = {*training, *validation, *heldOut};
    GroundTruthPlanInputs inputs;
    inputs.fpa = GroundTruthModelTrack{
        GroundTruthEvidencePartitions{std::move(*training),
                                      std::move(*validation),
                                      std::move(*heldOut), std::nullopt},
        DeterministicGbdtTrainingConfig{fpaModelSeed, fpaModelTrees,
                                        fpaModelDepth, fpaModelMinimumRows,
                                        fpaModelLearningRateNumerator,
                                        fpaModelLearningRateDenominator},
        *validationGate, *heldOutGate};
    auto plan = buildGroundTruthPlan(*config, std::move(inputs));
    if (!plan)
      return plan.takeError();
    semanticInputs->insert(semanticInputs->end(),
                           plan->semanticInputs().begin(),
                           plan->semanticInputs().end());
    canonicalizeRootUnion(*semanticInputs);
    preexistingEvidence->insert(preexistingEvidence->end(),
                                plan->preexistingEvidence().begin(),
                                plan->preexistingEvidence().end());
    canonicalizeRootUnion(*preexistingEvidence);
    *config = plan->resolvedConfig();
    fpaModel.emplace(std::move(*plan));
  }
  auto view = projectResolvedDseConfigView(*config);
  if (!view)
    return view.takeError();
  if (!resolvedConfigOutputPath.empty())
    if (llvm::Error error =
            writeResolvedConfig(resolvedConfigOutputPath, *config))
      return std::move(error);
  auto publishedConfig = publishResolvedConfigReference(*config, artifacts);
  if (!publishedConfig)
    return publishedConfig.takeError();
  auto producer = DseProducerSemanticBuildIdentity::get(producerBuild);
  if (!producer)
    return producer.takeError();
  auto closure = DseRunClosure::get(std::move(*producer), *semanticInputs,
                                    *config, *preexistingEvidence, artifacts);
  if (!closure)
    return closure.takeError();
  auto journal = openExecutionJournal(runRoot, *closure, *view);
  if (!journal)
    return journal.takeError();

  auto toolCapacities = parseCapacities(externalBindingCapacities, false);
  if (!toolCapacities)
    return toolCapacities.takeError();
  auto licenseCapacities = parseCapacities(licenseBindingCapacities, true);
  if (!licenseCapacities)
    return licenseCapacities.takeError();
  auto capacity =
      SiteCapacity::get(siteCpu, siteMemory, siteScratch, *toolCapacities,
                        *licenseCapacities, externalBindingDefaultCapacity);
  if (!capacity)
    return capacity.takeError();
  auto scheduler = SiteScheduler::create(std::move(*capacity));
  if (!scheduler)
    return scheduler.takeError();
  auto inProcessClaim =
      SiteResourceClaim::get(workCpu, workMemory, workScratch);
  if (!inProcessClaim)
    return inProcessClaim.takeError();

  std::optional<ExternalExecutionSite> externalSite;
  if (!localToolConfigPath.empty()) {
    auto localConfig = external_tool::loadLocalToolConfig(localToolConfigPath);
    if (!localConfig)
      return localConfig.takeError();
    externalSite = ExternalExecutionSite{
        std::move(*localConfig),
        prepareOnly ? ExternalAttemptDisposition::PrepareOnly
                    : ExternalAttemptDisposition::ExecutePrepared,
        externalCpu,
        externalMemory,
        externalScratch,
        claimLicense};
  }
  auto executionPolicy = PlanExecutionPolicy::get(
      workerCount, std::move(*inProcessClaim), std::move(externalSite), {},
      maximumDispatches == 0 ? std::optional<std::uint64_t>{}
                             : std::optional<std::uint64_t>{maximumDispatches});
  if (!executionPolicy)
    return executionPolicy.takeError();

  std::unique_ptr<llvm::raw_fd_ostream> progressFile;
  llvm::raw_ostream *progress = &llvm::outs();
  if (progressPath != "-") {
    std::error_code error;
    progressFile = std::make_unique<llvm::raw_fd_ostream>(
        progressPath, error, llvm::sys::fs::OF_Append);
    if (error)
      return llvm::errorCodeToError(error);
    progress = progressFile.get();
  }

  stopSignal = 0;
  std::signal(SIGINT, requestStopFromSignal);
  std::signal(SIGTERM, requestStopFromSignal);
  std::atomic<bool> monitoring{true};
  std::condition_variable monitorChanged;
  std::mutex monitorMutex;
  std::string monitorError;
  std::thread monitor([&] {
    std::unique_lock<std::mutex> lock(monitorMutex);
    while (monitoring.load(std::memory_order_relaxed)) {
      lock.unlock();
      if (stopSignal != 0 && !journal->gracefulStopRequested()) {
        if (llvm::Error error = stopDseExecution(
                *journal, GracefulStopPolicy::FinishAtomicOwnerBoundary)) {
          monitorError = llvm::toString(std::move(error));
          break;
        }
      }
      auto projection =
          projectDseOperationalState(*journal, *scheduler, workerCount);
      if (!projection) {
        monitorError = llvm::toString(projection.takeError());
        break;
      }
      if (llvm::Error error =
              writeDseOperationalProjectionJsonLine(*projection, *progress)) {
        monitorError = llvm::toString(std::move(error));
        break;
      }
      progress->flush();
      lock.lock();
      monitorChanged.wait_for(
          lock, std::chrono::milliseconds(progressIntervalMilliseconds),
          [&] { return !monitoring.load(std::memory_order_relaxed); });
    }
  });

  using ExecutionResult =
      std::variant<DsePlanExecutionOutcome, CampaignExecutionResult>;
  llvm::Expected<ExecutionResult> executionResult =
      [&]() -> llvm::Expected<ExecutionResult> {
    const GroundTruthCampaignKind campaignKind = groundTruthCampaign.getValue();
    if (campaignKind == GroundTruthCampaignKind::None) {
      auto outcome =
          resumeDsePlan(*view, *closure, *journal, *scheduler, *executionPolicy,
                        artifacts, blobs, InvocationManifestRetention::Retain);
      if (!outcome)
        return outcome.takeError();
      return ExecutionResult{std::in_place_index<0>, std::move(*outcome)};
    }
    if (campaignKind == GroundTruthCampaignKind::Generic) {
      auto campaignPolicy = CampaignExecutionPolicy::get(
          pilotDispatchCount, minimumPilotObservations);
      if (!campaignPolicy)
        return campaignPolicy.takeError();
      auto outcome = runGroundTruthCampaign(*view, *closure, *campaignPolicy,
                                            *executionPolicy, *scheduler,
                                            *journal, artifacts, blobs);
      if (!outcome)
        return outcome.takeError();
      return ExecutionResult{std::in_place_index<1>, std::move(*outcome)};
    }
    auto campaignPolicy = makeFpaGroundTruthCampaignPolicy(
        pilotDispatchCount, minimumPilotObservations);
    if (!campaignPolicy)
      return campaignPolicy.takeError();
    auto outcome = runFpaGroundTruthCampaign(*view, *closure, *campaignPolicy,
                                             *executionPolicy, *scheduler,
                                             *journal, artifacts, blobs);
    if (!outcome)
      return outcome.takeError();
    return ExecutionResult{std::in_place_index<1>, std::move(*outcome)};
  }();
  monitoring.store(false, std::memory_order_relaxed);
  monitorChanged.notify_all();
  monitor.join();
  if (!executionResult) {
    llvm::Error executionError = executionResult.takeError();
    auto activeOccurrence = journal->currentInvocationOccurrence();
    if (activeOccurrence)
      executionError = llvm::joinErrors(std::move(executionError),
                                        journal->releaseInvocationOccurrence());
    else
      llvm::consumeError(activeOccurrence.takeError());
    if (!monitorError.empty())
      return llvm::joinErrors(std::move(executionError), invalid(monitorError));
    return std::move(executionError);
  }

  DsePlanExecutionOutcome *invocationOutcome = nullptr;
  std::optional<CampaignAdmissionFailureReason> campaignAdmissionFailure;
  if (auto *direct = std::get_if<DsePlanExecutionOutcome>(&*executionResult)) {
    invocationOutcome = direct;
  } else {
    CampaignExecutionResult &campaign =
        std::get<CampaignExecutionResult>(*executionResult);
    if (auto *executed = std::get_if<CampaignExecution>(&campaign))
      invocationOutcome = &executed->outcome;
    else {
      CampaignAdmissionRefusal &refusal =
          std::get<CampaignAdmissionRefusal>(campaign);
      invocationOutcome = &refusal.outcome;
      campaignAdmissionFailure = refusal.reason;
    }
  }
  auto invocation =
      finalizeDsePlanInvocation(*closure, *config, *invocationOutcome, *journal,
                                artifacts, blobs, campaignAdmissionFailure);
  if (!invocation) {
    llvm::Error manifestError = invocation.takeError();
    if (!monitorError.empty())
      return llvm::joinErrors(std::move(manifestError), invalid(monitorError));
    return std::move(manifestError);
  }
  llvm::errs() << "invocation_manifest resolved_config_schema="
               << invocation->resolvedConfig().schemaIdentity
               << " resolved_config_version="
               << invocation->resolvedConfig().schemaVersion.major << '.'
               << invocation->resolvedConfig().schemaVersion.minor
               << " resolved_config_identity="
               << llvm::toHex(invocation->resolvedConfig().artifact.bytes(),
                              true)
               << " run_key="
               << llvm::toHex(invocation->occurrence().runKey.bytes(), true)
               << " occurrence=" << invocation->occurrence().occurrenceOrdinal
               << " blob=" << llvm::toHex(invocation->blob().bytes(), true)
               << '\n';
  if (!monitorError.empty())
    return invalid(monitorError);
  if (fuWorkflow)
    if (llvm::Error error = writeFuReverseSynthesisEvidence(
            fuReverseSynthesisEvidence, *fuWorkflow, *invocationOutcome,
            *invocation, artifacts, blobs))
      return std::move(error);

  auto finalProjection =
      projectDseOperationalState(*journal, *scheduler, workerCount);
  if (!finalProjection)
    return finalProjection.takeError();
  if (llvm::Error error =
          writeDseOperationalProjectionJsonLine(*finalProjection, *progress))
    return error;
  progress->flush();

  int exitCode = EXIT_SUCCESS;
  if (groundTruthCampaign.getValue() != GroundTruthCampaignKind::None) {
    CampaignExecutionResult &campaignOutcome =
        std::get<CampaignExecutionResult>(*executionResult);
    if (const auto *refused =
            std::get_if<CampaignAdmissionRefusal>(&campaignOutcome)) {
      llvm::errs() << "campaign_result=admission_refused reason="
                   << admissionReason(refused->reason) << '\n';
      exitCode = 3;
    } else {
      exitCode = reportPlanOutcome(
          std::get<CampaignExecution>(campaignOutcome).outcome);
    }
  } else {
    DsePlanExecutionOutcome &outcome =
        std::get<DsePlanExecutionOutcome>(*executionResult);
    if (!jointPairOutputs.empty())
      reportJointOutputs(outcome, jointPairOutputs);
    exitCode = reportPlanOutcome(outcome);
  }

  const DsePlanExecutionOutcome &finalOutcome =
      groundTruthCampaign.getValue() != GroundTruthCampaignKind::None
          ? [&]() -> const DsePlanExecutionOutcome & {
              const CampaignExecutionResult &campaignOutcome =
                  std::get<CampaignExecutionResult>(*executionResult);
              if (const auto *refused =
                      std::get_if<CampaignAdmissionRefusal>(&campaignOutcome))
                return refused->outcome;
              return std::get<CampaignExecution>(campaignOutcome).outcome;
            }()
          : std::get<DsePlanExecutionOutcome>(*executionResult);
  const CompletedDsePlanExecution *availableExecution =
      std::get_if<CompletedDsePlanExecution>(&finalOutcome);
  const bool executionComplete = availableExecution != nullptr;
  if (!availableExecution)
    availableExecution =
        &std::get<IncompleteDsePlanExecution>(finalOutcome).availableExecution();
  const auto planDisposition = [&]() -> llvm::json::Object {
    llvm::json::Object disposition{{"complete", executionComplete}};
    if (const auto *incomplete =
            std::get_if<IncompleteDsePlanExecution>(&finalOutcome)) {
      disposition["node"] = incomplete->nodeOrdinal();
      disposition["reason"] = toString(incomplete->reason()).str();
      disposition["execution_stopped"] = incomplete->executionStopped();
    }
    return disposition;
  };
  if (fpaPhysical && !fpaPhysicalOutputPath.empty()) {
    llvm::json::Object report{
        {"schema", "loom.fpa_physical_implementation_report"},
        {"version", "1.0"},
        {"resolved_config", rootReferenceJson(*publishedConfig)},
        {"implementation_platform",
         rootReferenceJson(fpaPhysical->implementationPlatform)},
        {"technology_corner_id", fpaPhysical->technologyCorner.entity.value()},
        {"yosys_provider_build", fpaYosysBuild.getValue()},
        {"openroad_provider_build", fpaOpenRoadBuild.getValue()},
        {"plan", planDisposition()}};
    llvm::json::Array systems;
    for (const FpaRtlStageOutputs &outputs : fpaPhysical->rtlStages)
      systems.push_back(llvm::json::Object{
          {"system", rootReferenceJson(outputs.system)},
          {"configuration_abi", rootReferenceJson(outputs.configurationAbi)},
          {"portable_rtl", availableRoots(*availableExecution, outputs.rtl)}});
    report["systems"] = std::move(systems);
    llvm::json::Array implementations;
    for (const FpaPhysicalStageOutputs &outputs : fpaPhysical->physicalStages)
      implementations.push_back(llvm::json::Object{
          {"portable_rtl", rootReferenceJson(outputs.rtlImplementation)},
          {"gate_netlist",
           availableRoots(*availableExecution, outputs.gateNetlist)},
          {"routed", availableRoots(*availableExecution, outputs.routed)}});
    report["implementations"] = std::move(implementations);
    if (llvm::Error error =
            writeJsonObject(fpaPhysicalOutputPath, std::move(report)))
      return std::move(error);
  }
  if (fpaCollection && !fpaCollectionOutputPath.empty()) {
    llvm::json::Object report{
        {"schema", "loom.fpa_ground_truth_collection_report"},
        {"version", "1.0"},
        {"resolved_config", rootReferenceJson(*publishedConfig)},
        {"static_fpa_provider_build",
         config->evaluation.openRoadStaticFpa
             ? llvm::json::Value(config->evaluation.openRoadStaticFpa
                                     ->stableProviderBuildIdentity)
             : llvm::json::Value(nullptr)},
        {"campaign_active_wall_time_limit_ns",
         CampaignExecutionPolicy::maximumCampaignActiveWallTimeNanoseconds},
        {"sample_active_wall_time_limit_ns",
         CampaignExecutionPolicy::maximumSampleActiveWallTimeNanoseconds},
        {"plan", planDisposition()}};
    llvm::json::Object partitions;
    for (const FpaCollectionPartitionInputs &partition : fpaCollectionPartitions) {
      auto projected = fpaPartitionReport(partition, *availableExecution,
                                          artifacts, blobs);
      if (!projected)
        return projected.takeError();
      partitions[partition.name] = std::move(*projected);
    }
    report["partitions"] = std::move(partitions);
    if (llvm::Error error =
            writeJsonObject(fpaCollectionOutputPath, std::move(report)))
      return std::move(error);
  }
  if (fpaModel && !fpaModelOutputPath.empty()) {
    const GroundTruthTrackOutputs &outputs = *fpaModel->fpaOutputs();
    llvm::json::Object report{
        {"schema", "loom.fpa_model_release_report"},
        {"version", "1.0"},
        {"resolved_config", rootReferenceJson(*publishedConfig)},
        {"training_evidence", rootReferenceArray(fpaModelEvidence[0])},
        {"validation_evidence", rootReferenceArray(fpaModelEvidence[1])},
        {"held_out_evidence", rootReferenceArray(fpaModelEvidence[2])},
        {"trained_bundle",
         availableRoots(*availableExecution, outputs.trainedBundle)},
        {"released_bundle",
         availableRoots(*availableExecution, outputs.releasedBundle)},
        {"plan", planDisposition()}};
    llvm::ArrayRef<ArtifactRootReference> released =
        availableExecution->hasOutput(outputs.releasedBundle)
            ? availableExecution->resolve(outputs.releasedBundle)
            : llvm::ArrayRef<ArtifactRootReference>{};
    llvm::ArrayRef<ArtifactRootReference> trained =
        availableExecution->hasOutput(outputs.trainedBundle)
            ? availableExecution->resolve(outputs.trainedBundle)
            : llvm::ArrayRef<ArtifactRootReference>{};
    const std::array<std::pair<llvm::StringLiteral, PlanOutputRef>, 2>
        calibrations = {{{"validation_calibration", outputs.validationEvidence},
                         {"held_out_calibration", outputs.heldOutEvidence}}};
    for (std::size_t ordinal = 0; ordinal != calibrations.size(); ++ordinal) {
      llvm::json::Array entries;
      const ArtifactRootReference *bundle =
          !trained.empty() ? &trained.front() : nullptr;
      if (bundle && availableExecution->hasOutput(calibrations[ordinal].second))
        for (const ArtifactRootReference &evidence :
             availableExecution->resolve(calibrations[ordinal].second)) {
          auto metrics = calibrationMetricsJson(
              evidence, *bundle, fpaModelEvidence[ordinal + 1], artifacts, blobs);
          if (!metrics)
            return metrics.takeError();
          entries.push_back(llvm::json::Object{
              {"evidence", rootReferenceJson(evidence)},
              {"metrics", std::move(*metrics)}});
        }
      report[calibrations[ordinal].first] = std::move(entries);
    }
    if (llvm::Error error =
            writeJsonObject(fpaModelOutputPath, std::move(report)))
      return std::move(error);
    if (!fpaReleasedWeightOutputPath.empty()) {
      if (released.size() != 1)
        return invalid("FPA model release did not publish exactly one bundle");
      if (llvm::Error error = writeArtifactRootReferenceJsonFile(
              fpaReleasedWeightOutputPath, released.front()))
        return std::move(error);
    }
  }
  return exitCode;
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "execute one resolved Loom DSE plan\n");
  auto result = run();
  if (!result) {
    bool typed = false;
    llvm::Error remaining = llvm::handleErrors(
        result.takeError(), [&](const FuReverseSynthesisError &error) {
          typed = true;
          llvm::errs() << "error: fu_reverse_synthesis_failure="
                       << fuReverseSynthesisFailureSpelling(error.failure())
                       << " diagnostic=" << error.diagnostic() << '\n';
        });
    if (remaining)
      llvm::errs() << "error: " << llvm::toString(std::move(remaining)) << '\n';
    else if (!typed)
      llvm_unreachable("successful error handling requires a typed error");
    return EXIT_FAILURE;
  }
  return *result;
}

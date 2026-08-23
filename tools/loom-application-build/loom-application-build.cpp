#include "ADG/Builtin.h"
#include "Application/Build.h"
#include "Application/BuildDiagnostics.h"
#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Common/ExecutionControl.h"
#include "Config/ResolvedConfig.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Deployment/Package.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Payload/AcceleratorFinalLink.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

using MonotonicClock = std::chrono::steady_clock;

std::uint64_t elapsedNanoseconds(MonotonicClock::time_point begin) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             MonotonicClock::now() - begin)
      .count();
}

class ApplicationBuildOperationTimer final {
public:
  explicit ApplicationBuildOperationTimer(
      loom::application::ApplicationBuildOperation operation)
      : operation_(operation), begin_(MonotonicClock::now()) {}

  ~ApplicationBuildOperationTimer() {
    loom::application::emitApplicationBuildOperationStatistics(
        {operation_, elapsedNanoseconds(begin_), 1});
  }

  ApplicationBuildOperationTimer(const ApplicationBuildOperationTimer &) =
      delete;
  ApplicationBuildOperationTimer &
  operator=(const ApplicationBuildOperationTimer &) = delete;

private:
  loom::application::ApplicationBuildOperation operation_;
  MonotonicClock::time_point begin_;
};

llvm::cl::opt<std::string> driverArgumentsOutput(
    "driver-arguments-output",
    llvm::cl::desc("Write the System-derived compiler arguments"),
    llvm::cl::value_desc("path"));
llvm::cl::opt<std::string> finalLinkOutput(
    "final-link-output",
    llvm::cl::desc("Consume one completed compiler final-link output"),
    llvm::cl::value_desc("path"));
llvm::cl::opt<std::string>
    deploymentOutput("deployment-output",
                     llvm::cl::desc("Publish the Deployment package"),
                     llvm::cl::value_desc("path"), llvm::cl::Required);
llvm::cl::opt<std::string>
    accelerationProfile("acceleration-profile",
                        llvm::cl::desc("Resolved configuration selector"),
                        llvm::cl::value_desc("selector"));
llvm::cl::opt<std::string> hardwarePath("hardware",
                                        llvm::cl::desc("External Fabric MLIR"),
                                        llvm::cl::value_desc("path"));
llvm::cl::opt<std::string>
    visualizationPath("visualization",
                      llvm::cl::desc("Mapping visualization destination"),
                      llvm::cl::value_desc("path"));
llvm::cl::list<std::string> operatorProtocolSymbols(
    "operator-protocol-symbol",
    llvm::cl::desc("Select a defined callable as an operator protocol root"),
    llvm::cl::value_desc("symbol"), llvm::cl::ZeroOrMore);
llvm::cl::opt<std::uint64_t> mappingTechCandidateLimit(
    "mapping-tech-candidate-limit",
    llvm::cl::desc("maximum TechMapping candidates admitted to Spatial PnR "
                   "for each target Module"),
    llvm::cl::init(8));
inline constexpr std::uint64_t kDefaultMappingWallTimeLimitMilliseconds =
    120000;
llvm::cl::opt<std::uint64_t> mappingWallTimeLimitMilliseconds(
    "mapping-wall-time-limit-ms",
    llvm::cl::desc("cooperative pre-Mapping and Mapping wall-time limit"),
    llvm::cl::init(kDefaultMappingWallTimeLimitMilliseconds));
llvm::cl::opt<std::string> mappingStoppingPolicy(
    "mapping-stopping-policy",
    llvm::cl::desc(
        "Mapping stopping policy: first_verified or bounded_quality"),
    llvm::cl::init("first_verified"));
llvm::cl::opt<std::string> mappingSpectrumEndpoint(
    "mapping-spectrum-endpoint",
    llvm::cl::desc("Spectrum ranking focus: automatic, max_temporal, "
                   "max_spatial, or intermediate"),
    llvm::cl::init("automatic"));

llvm::Error productError(llvm::StringRef kind, const llvm::Twine &message);

llvm::Expected<loom::dse::JointDesignStoppingPolicy>
parseMappingStoppingPolicy() {
  if (mappingStoppingPolicy == "first_verified")
    return loom::dse::JointDesignStoppingPolicy::FirstVerified;
  if (mappingStoppingPolicy == "bounded_quality")
    return loom::dse::JointDesignStoppingPolicy::BoundedQuality;
  return productError("loom_mapping_stopping_policy_invalid",
                      "expected first_verified or bounded_quality");
}

llvm::Expected<loom::dse::PreMappingSpectrumEndpoint>
parseMappingSpectrumEndpoint() {
  if (mappingSpectrumEndpoint == "automatic")
    return loom::dse::PreMappingSpectrumEndpoint::Automatic;
  if (mappingSpectrumEndpoint == "max_temporal")
    return loom::dse::PreMappingSpectrumEndpoint::MaxTemporal;
  if (mappingSpectrumEndpoint == "max_spatial")
    return loom::dse::PreMappingSpectrumEndpoint::MaxSpatial;
  if (mappingSpectrumEndpoint == "intermediate")
    return loom::dse::PreMappingSpectrumEndpoint::Intermediate;
  return productError("loom_mapping_spectrum_endpoint_invalid",
                      "expected automatic, max_temporal, max_spatial, or "
                      "intermediate");
}

llvm::Error productError(llvm::StringRef kind, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 kind + ": " + message);
}

struct ProductMappingExecutionDeadline final {
  std::uint64_t requestedMilliseconds = 0;
  std::uint64_t notAfterUnixNanoseconds = 0;
  MonotonicClock::time_point begin = MonotonicClock::now();
  MonotonicClock::time_point notAfter = MonotonicClock::time_point::max();

  bool stopRequested() const { return MonotonicClock::now() >= notAfter; }

  std::optional<MonotonicClock::duration> remainingTime() const {
    const auto now = MonotonicClock::now();
    return now >= notAfter ? MonotonicClock::duration::zero() : notAfter - now;
  }
};

bool productMappingStopRequested(const void *opaque) {
  return static_cast<const ProductMappingExecutionDeadline *>(opaque)
      ->stopRequested();
}

std::optional<MonotonicClock::duration>
productMappingRemainingTime(const void *opaque) {
  return static_cast<const ProductMappingExecutionDeadline *>(opaque)
      ->remainingTime();
}

llvm::Expected<std::optional<ProductMappingExecutionDeadline>>
makeProductMappingExecutionDeadline() {
  const std::uint64_t requested = mappingWallTimeLimitMilliseconds;
  if (requested == 0)
    return productError("loom_mapping_execution_policy_invalid",
                        "Mapping wall-time limit must be positive");
  constexpr std::uint64_t nanosecondsPerMillisecond = 1000000;
  if (requested >
      std::numeric_limits<std::uint64_t>::max() / nanosecondsPerMillisecond)
    return productError("loom_mapping_execution_policy_invalid",
                        "Mapping wall-time limit overflows nanoseconds");
  const auto elapsed = std::chrono::system_clock::now().time_since_epoch();
  const auto signedNow =
      std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  if (signedNow <= 0)
    return productError("loom_mapping_execution_policy_invalid",
                        "system clock cannot represent a Mapping deadline");
  const std::uint64_t now = static_cast<std::uint64_t>(signedNow);
  const std::uint64_t duration = requested * nanosecondsPerMillisecond;
  if (now > std::numeric_limits<std::uint64_t>::max() - duration)
    return productError("loom_mapping_execution_policy_invalid",
                        "Mapping deadline overflows the system clock");
  const auto begin = MonotonicClock::now();
  return std::optional<ProductMappingExecutionDeadline>{
      ProductMappingExecutionDeadline{
          requested, now + duration, begin,
          begin + std::chrono::milliseconds(requested)}};
}

class ProductMappingExecutionPolicyReporter final {
public:
  explicit ProductMappingExecutionPolicyReporter(
      const std::optional<ProductMappingExecutionDeadline> &deadline)
      : deadline_(deadline) {}

  ~ProductMappingExecutionPolicyReporter() {
    if (!deadline_)
      return;
    loom::application::emitApplicationMappingExecutionPolicyStatistics(
        {deadline_->requestedMilliseconds, deadline_->notAfterUnixNanoseconds,
         elapsedNanoseconds(deadline_->begin), deadline_->stopRequested()});
  }

private:
  const std::optional<ProductMappingExecutionDeadline> &deadline_;
};

class ProductWorkspace final {
public:
  static llvm::Expected<std::unique_ptr<ProductWorkspace>>
  create(llvm::StringRef destinationPath) {
    llvm::SmallString<256> destination(destinationPath);
    if (std::error_code error = llvm::sys::fs::make_absolute(destination))
      return productError("loom_product_workspace_invalid",
                          "cannot resolve Deployment output: " +
                              error.message());
    llvm::sys::path::remove_dots(destination, true);
    const llvm::StringRef filename = llvm::sys::path::filename(destination);
    if (filename.empty() || filename == "." || filename == "..")
      return productError("loom_product_workspace_invalid",
                          "Deployment output has no directory name");
    llvm::SmallString<256> parent = llvm::sys::path::parent_path(destination);
    if (parent.empty())
      parent = ".";
    if (!llvm::sys::fs::is_directory(parent))
      return productError("loom_product_workspace_invalid",
                          "Deployment output parent is not a directory");
    llvm::SmallString<256> pattern(parent);
    llvm::sys::path::append(pattern, ("." + filename + ".loom-work").str());
    llvm::SmallString<256> root;
    if (std::error_code error =
            llvm::sys::fs::createUniqueDirectory(pattern, root)) {
      return productError("loom_product_workspace_invalid",
                          "cannot create bounded workspace at '" + pattern +
                              "': " + error.message());
    }
    auto workspace = std::unique_ptr<ProductWorkspace>(
        new ProductWorkspace(root.str(), destination.str()));
    for (llvm::StringRef directory :
         {llvm::StringRef(workspace->artifactPath_),
          llvm::StringRef(workspace->blobPath_),
          llvm::StringRef(workspace->journalPath_),
          llvm::StringRef(workspace->linkerPath_)}) {
      if (std::error_code error =
              llvm::sys::fs::create_directories(directory)) {
        std::error_code ignored;
        std::filesystem::remove_all(workspace->root_, ignored);
        return productError("loom_product_workspace_invalid",
                            "cannot initialize workspace: " + error.message());
      }
    }
    return workspace;
  }

  ~ProductWorkspace() {
    std::error_code ignored;
    std::filesystem::remove_all(root_, ignored);
  }

  const loom::ArtifactStore &artifacts() const { return artifacts_; }
  const loom::BlobStore &blobs() const { return blobs_; }
  llvm::StringRef deploymentPath() const { return deploymentPath_; }
  llvm::StringRef journalPath() const { return journalPath_; }
  llvm::StringRef linkerPath() const { return linkerPath_; }

private:
  static std::string child(llvm::StringRef root, llvm::StringRef name) {
    llvm::SmallString<256> path(root);
    llvm::sys::path::append(path, name);
    return path.str().str();
  }

  ProductWorkspace(llvm::StringRef root, llvm::StringRef deploymentPath)
      : root_(root.str()), deploymentPath_(deploymentPath.str()),
        artifactPath_(child(root, "artifacts")),
        blobPath_(child(root, "blobs")), journalPath_(child(root, "journal")),
        linkerPath_(child(root, "linker")), artifacts_(artifactPath_),
        blobs_(blobPath_) {}

  std::string root_;
  std::string deploymentPath_;
  std::string artifactPath_;
  std::string blobPath_;
  std::string journalPath_;
  std::string linkerPath_;
  loom::ArtifactStore artifacts_;
  loom::BlobStore blobs_;
};

struct PreparedProductTarget final {
  std::unique_ptr<ProductWorkspace> workspace;
  loom::ResolvedConfig config;
  loom::fabric::FinalizedFabricRoot system;
  std::vector<loom::ArtifactRootReference> physicalTimingProfiles;
  loom::CompilerTargetPolicy compilerPolicy;
  loom::CompilerTargetCommandLineProjection commandLine;
};

bool sameCommandLineTarget(
    const loom::CompilerTargetCommandLineProjection &lhs,
    const loom::CompilerTargetCommandLineProjection &rhs) {
  return lhs.targetTriple == rhs.targetTriple &&
         lhs.architecture == rhs.architecture && lhs.abi == rhs.abi &&
         lhs.codeModel == rhs.codeModel && lhs.backendCpu == rhs.backendCpu &&
         lhs.ltoFeatures == rhs.ltoFeatures &&
         lhs.positionIndependent == rhs.positionIndependent;
}

llvm::Error validateRequestedProductCapabilities() {
  if (!hardwarePath.empty())
    return productError("loom_hardware_import_unsupported",
                        "external Fabric MLIR import is not yet available");
  if (!visualizationPath.empty())
    return productError("loom_visualization_export_unsupported",
                        "product visualization export is not yet available");
  return llvm::Error::success();
}

llvm::Expected<loom::CompilerTargetCommandLineProjection>
prepareProductDriverTarget() {
  ApplicationBuildOperationTimer timer(
      loom::application::ApplicationBuildOperation::ProductTargetPreparation);
  if (llvm::Error error = validateRequestedProductCapabilities())
    return std::move(error);
  auto config = loom::resolveConfigProfile(accelerationProfile);
  if (!config)
    return config.takeError();
  if (!loom::adg::findBuiltinTargetDescriptor(
          config->hardwareTarget.templateIdentity,
          config->hardwareTarget.schemaVersion.major,
          config->hardwareTarget.schemaVersion.minor) ||
      !loom::adg::isValidBuiltinTargetScale(config->hardwareTarget.parameters))
    return productError("loom_product_target_invalid",
                        "resolved builtin target descriptor is invalid");
  auto architecture = loom::adg::getBuiltinInstructionCoreArchitecture();
  if (!architecture)
    return architecture.takeError();
  return loom::projectCompilerTargetCommandLine(
      *architecture, loom::portableRiscV64CompilerTargetPolicy());
}

llvm::Expected<PreparedProductTarget> prepareProductTarget() {
  ApplicationBuildOperationTimer timer(
      loom::application::ApplicationBuildOperation::ProductTargetPreparation);
  if (llvm::Error error = validateRequestedProductCapabilities())
    return std::move(error);
  std::string deploymentPath = deploymentOutput.getValue();
  auto workspace = ProductWorkspace::create(deploymentPath);
  if (!workspace)
    return workspace.takeError();
  auto config = loom::resolveConfigProfile(accelerationProfile);
  if (!config)
    return config.takeError();
  config->dse.spatialPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  config->dse.systemPnr.search.completionGoal =
      loom::ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  auto design = loom::adg::buildBuiltinTarget(
      (*workspace)->artifacts(), config->hardwareTarget.templateIdentity,
      config->hardwareTarget.schemaVersion.major,
      config->hardwareTarget.schemaVersion.minor,
      config->hardwareTarget.parameters);
  if (!design)
    return design.takeError();
  if (design->roots().size() != 1)
    return productError("loom_product_target_invalid",
                        "resolved target did not produce one System root");
  auto system = loom::fabric::importEntireFabricRoot(
      design->roots().front().reference(), (*workspace)->artifacts());
  if (!system)
    return system.takeError();
  auto systemView = loom::fabric::requireSystemRoot(system->view());
  if (!systemView)
    return systemView.takeError();

  std::vector<loom::ArtifactRootReference> timingReferences;
  auto timing =
      loom::fabric::projectNormalizedSystemPhysicalTimingProfiles(*systemView);
  if (!timing)
    return timing.takeError();
  timingReferences.reserve(timing->size());
  for (const auto &profile : *timing) {
    auto published = loom::fabric::publishFabricPhysicalTimingProfile(
        profile, (*workspace)->artifacts());
    if (!published)
      return published.takeError();
    timingReferences.push_back(std::move(*published));
  }

  loom::CompilerTargetPolicy compilerPolicy =
      loom::portableRiscV64CompilerTargetPolicy();
  auto compilerTargets = loom::resolveSystemCompilerTargetBindings(
      *system, compilerPolicy, (*workspace)->artifacts());
  if (!compilerTargets)
    return compilerTargets.takeError();
  auto commandLine =
      loom::projectCompilerTargetCommandLine(compilerTargets->host().binding());
  if (!commandLine)
    return commandLine.takeError();
  for (const auto &group : compilerTargets->instructionGroups()) {
    auto groupCommandLine =
        loom::projectCompilerTargetCommandLine(group.binding().binding());
    if (!groupCommandLine)
      return groupCommandLine.takeError();
    if (!sameCommandLineTarget(*commandLine, *groupCommandLine))
      return productError(
          "loom_product_target_unsupported",
          "one final-linked module cannot serve the heterogeneous compiler "
          "target cohort");
  }
  return PreparedProductTarget{
      std::move(*workspace),     std::move(*config),
      std::move(*system),        std::move(timingReferences),
      std::move(compilerPolicy), std::move(*commandLine)};
}

std::vector<std::string> projectDriverArguments(
    const loom::CompilerTargetCommandLineProjection &target) {
  std::vector<std::string> result;
  result.push_back("--target=" + target.targetTriple);
  result.push_back("-march=" + target.architecture);
  result.push_back("-mabi=" + target.abi);
  result.push_back("-mcmodel=" + target.codeModel);
  result.push_back("-mcpu=" + target.backendCpu);
  result.push_back("-B" LOOM_LLVM_TOOLS_DIR);
  result.push_back("-fuse-ld=lld");
  result.push_back("-nostdlib");
  result.push_back("-Wl,--entry=main");
  result.push_back("-flto=full");
  result.push_back("-ffat-lto-objects");
  result.push_back("-Wl,--fat-lto-objects");
  result.push_back("-Wl,--save-temps=resolution");
  result.push_back("-Wl,--save-temps=precodegen");
  result.push_back("-Wl,--unresolved-symbols=ignore-all");
  result.push_back("-Wl,--lto-O1");
  if (!target.ltoFeatures.empty()) {
    result.push_back("-Xlinker");
    result.push_back("--plugin-opt=-mattr=" + target.ltoFeatures);
  }
  if (target.positionIndependent)
    result.push_back("-fPIC");
  return result;
}

llvm::Error
writeDriverArguments(const loom::CompilerTargetCommandLineProjection &target) {
  std::error_code error;
  llvm::raw_fd_ostream output(driverArgumentsOutput, error,
                              llvm::sys::fs::OF_None);
  if (error)
    return productError("loom_product_driver_projection_invalid",
                        "cannot open driver argument output: " +
                            error.message());
  for (const std::string &argument : projectDriverArguments(target)) {
    output << argument;
    output.write('\0');
  }
  output.close();
  if (output.has_error())
    return productError("loom_product_driver_projection_invalid",
                        "cannot write driver argument output");
  return llvm::Error::success();
}

struct ProductFinalLinkArtifacts final {
  std::unique_ptr<llvm::Module> linkedModule;
};

llvm::Expected<ProductFinalLinkArtifacts>
importProductFinalLink(llvm::LLVMContext &context) {
  ApplicationBuildOperationTimer timer(
      loom::application::ApplicationBuildOperation::FinalLinkImport);
  const std::string resolutionPath = finalLinkOutput + ".resolution.txt";
  const std::string bitcodePath = finalLinkOutput + ".0.5.precodegen.bc";
  llvm::FileRemover removeResolution(resolutionPath);
  llvm::FileRemover removeBitcode(bitcodePath);

  auto resolution = llvm::MemoryBuffer::getFile(resolutionPath, false, false);
  if (!resolution)
    return productError("loom_final_link_artifact_missing",
                        "cannot read LLD resolution output: " +
                            resolution.getError().message());
  auto bitcode = llvm::MemoryBuffer::getFile(bitcodePath, false, false);
  if (!bitcode)
    return productError("loom_final_link_artifact_missing",
                        "cannot read LLD pre-code-generation bitcode: " +
                            bitcode.getError().message());
  auto module = loom::importLldAcceleratorFinalLink(
      (*resolution)->getMemBufferRef(), (*bitcode)->getMemBufferRef(), context);
  if (!module)
    return module.takeError();

  return ProductFinalLinkArtifacts{std::move(*module)};
}

llvm::Expected<loom::application::PreparedApplicationBuild>
prepareMappedApplication(const llvm::Module &module,
                         PreparedProductTarget &target,
                         loom::ExecutionControlView executionControl,
                         loom::dse::JointDesignStoppingPolicy stoppingPolicy,
                         loom::dse::PreMappingSpectrumEndpoint endpoint) {
  constexpr std::uint64_t kSoftwareFrontierLimit = 8;
  constexpr std::uint64_t kHardwareFrontierLimit = 8;
  constexpr std::uint64_t kSpatialMappingFrontierLimit = 32;
  const llvm::Function *entry = module.getFunction("main");
  if (!entry || entry->isDeclaration())
    return productError("loom_application_entry_unsupported",
                        "the final-linked module has no defined main entry");
  if (entry->isVarArg() || !entry->arg_empty())
    return productError(
        "loom_application_entry_unsupported",
        "the initial product entry supports only a nullary main function");

  auto jointPolicy = loom::dse::JointDesignPolicy::get(
      kSoftwareFrontierLimit, kHardwareFrontierLimit, kSoftwareFrontierLimit,
      mappingTechCandidateLimit, kSpatialMappingFrontierLimit);
  if (!jointPolicy)
    return jointPolicy.takeError();
  loom::frontend::PreMappingCompilationOptions compilationOptions;
  loom::dse::PreMappingExplorationOptions preMappingOptions{
      {compilationOptions.lowering,
       {loom::evaluation::MetricRequestOrdinal(0),
        loom::ResolvedObjectiveDirection::Minimize, kSoftwareFrontierLimit},
       1}};
  preMappingOptions.ownership.selectionMode =
      loom::dse::StructuredOwnershipSelectionMode::SemanticConformance;
  preMappingOptions.frontier.stoppingPolicy = stoppingPolicy;
  preMappingOptions.executionControl = executionControl;
  loom::application::ApplicationSourceInvocation sourceInvocation;
  sourceInvocation.entrySymbol = "main";
  sourceInvocation.observeReturnValue = !entry->getReturnType()->isVoidTy();

  loom::dse::ResourceTimeFrontierPolicy resourceTimePolicy;
  resourceTimePolicy.spectrumEndpoint = endpoint;
  auto outcome = loom::application::prepareApplicationBuild(
      module,
      {std::move(sourceInvocation),
       std::vector<std::string>(operatorProtocolSymbols.begin(),
                                operatorProtocolSymbols.end()),
       target.system.reference(),
       target.physicalTimingProfiles,
       target.config,
       std::move(*jointPolicy),
       std::move(compilationOptions),
       std::move(preMappingOptions),
       std::move(resourceTimePolicy)},
      target.workspace->artifacts(), target.workspace->blobs());
  if (!outcome)
    return outcome.takeError();
  if (auto *prepared =
          std::get_if<loom::application::PreparedApplicationBuild>(&*outcome))
    return std::move(*prepared);
  if (auto *incomplete =
          std::get_if<loom::dse::IncompletePreMappingExploration>(&*outcome))
    return productError("loom_pre_mapping_incomplete",
                        "candidate exploration ended at node " +
                            llvm::Twine(incomplete->planNodeOrdinal.value_or(
                                std::numeric_limits<std::uint64_t>::max())) +
                            " with reason " +
                            loom::dse::toString(incomplete->reason));
  if (auto *incomplete = std::get_if<
          loom::application::IncompleteApplicationResourceTimePlanning>(
          &*outcome))
    return productError(
        "loom_resource_time_planning_incomplete",
        "resource-time planning ended with reason " +
            loom::dse::resourceTimeFrontierIncompleteReasonSpelling(
                incomplete->reason));
  if (std::holds_alternative<loom::dse::CompletedPreMappingNoFeasibleCandidate>(
          *outcome))
    return productError("loom_pre_mapping_no_feasible_candidate",
                        "no verified software candidate was selected");
  const auto &unsupported =
      std::get<loom::application::UnsupportedApplicationBuild>(*outcome);
  switch (unsupported.kind) {
  case loom::application::ApplicationBuildUnsupportedKind::RootCoordinates:
    return productError("loom_application_unsupported",
                        "root coordinates are not statically enumerable for "
                        "launch " +
                            llvm::Twine(unsupported.root.entity.value()));
  case loom::application::ApplicationBuildUnsupportedKind::
      DirectInvocationBoundary:
    return productError("loom_application_unsupported",
                        "root has no replaceable direct invocation boundary "
                        "for launch " +
                            llvm::Twine(unsupported.root.entity.value()));
  case loom::application::ApplicationBuildUnsupportedKind::
      DynamicInvocationBoundary:
    return productError(
        "loom_application_unsupported",
        "dynamic invocation value capture is not exact for launch " +
            llvm::Twine(unsupported.root.entity.value()));
  }
  llvm_unreachable("closed ApplicationBuildUnsupportedKind");
}

llvm::Expected<loom::application::ApplicationMappingExecution>
executeProductMapping(
    const loom::application::PreparedApplicationBuild &prepared,
    PreparedProductTarget &target,
    std::optional<std::uint64_t> dispatchNotAfterUnixNanoseconds,
    loom::dse::JointDesignStoppingPolicy stoppingPolicy) {
  auto producer = loom::dse::DseProducerSemanticBuildIdentity::get(
      loom::application::applicationBuildProducerIdentity);
  if (!producer)
    return producer.takeError();
  auto capacity = loom::dse::SiteCapacity::get(1, 0, 0);
  if (!capacity)
    return capacity.takeError();
  auto claim = loom::dse::SiteResourceClaim::get(1, 0, 0);
  if (!claim)
    return claim.takeError();
  auto policy = loom::dse::PlanExecutionPolicy::get(
      1, std::move(*claim), std::nullopt, {}, std::nullopt,
      dispatchNotAfterUnixNanoseconds);
  if (!policy)
    return policy.takeError();
  std::optional<loom::dse::JointBoundedQualityPolicy> boundedQuality;
  if (stoppingPolicy == loom::dse::JointDesignStoppingPolicy::BoundedQuality) {
    auto quality = loom::application::makeApplicationBoundedQualityPolicy(
        prepared, *policy, target.workspace->artifacts(),
        target.workspace->blobs());
    if (!quality)
      return quality.takeError();
    boundedQuality.emplace(std::move(*quality));
  }
  auto execution = loom::application::executeApplicationMapping(
      prepared,
      {std::move(*producer),
       target.workspace->journalPath().str(),
       {},
       std::move(boundedQuality),
       std::move(*capacity),
       std::move(*policy)},
      target.workspace->artifacts(), target.workspace->blobs());
  if (!execution)
    return execution.takeError();
  std::size_t mappingCount = 0;
  for (const loom::dse::JointMappedPair &pair :
       execution->execution.mappedPairs)
    mappingCount += pair.systemMappings.size();
  if (const auto *incomplete =
          std::get_if<loom::dse::IncompleteDsePlanExecution>(
              &execution->execution.planExecution)) {
    const auto *generationReason =
        std::get_if<loom::dse::CandidateGeneratorIncompleteReason>(
            &incomplete->reason());
    const bool hasUsableBoundedResult =
        mappingCount != 0 && !incomplete->executionStopped() &&
        generationReason &&
        (*generationReason == loom::dse::CandidateGeneratorIncompleteReason::
                                  SemanticLimitReached ||
         *generationReason == loom::dse::CandidateGeneratorIncompleteReason::
                                  ProofNotEstablished);
    if (!hasUsableBoundedResult)
      return productError("loom_mapping_incomplete",
                          "joint Mapping ended at node " +
                              llvm::Twine(incomplete->nodeOrdinal()) +
                              " with reason " +
                              loom::dse::toString(incomplete->reason()));
  }
  if (mappingCount == 0)
    return productError("loom_mapping_no_feasible_candidate",
                        "joint Mapping selected no SystemMapping");
  if (stoppingPolicy == loom::dse::JointDesignStoppingPolicy::BoundedQuality &&
      execution->execution.summary.qualityDisposition !=
          loom::dse::JointDesignQualityDisposition::Complete) {
    return productError(
        "loom_mapping_quality_incomplete",
        "BoundedQuality did not establish a complete application QoR result");
  }
  if (!execution->execution.summary.selectedMapping)
    return productError("loom_mapping_selection_incomplete",
                        "Mapping returned candidates without a selected root");
  if (prepared.resourceTimePolicy.spectrumEndpoint !=
      loom::dse::PreMappingSpectrumEndpoint::Automatic) {
    const auto requestedClass = [&]() {
      switch (prepared.resourceTimePolicy.spectrumEndpoint) {
      case loom::dse::PreMappingSpectrumEndpoint::MaxTemporal:
        return loom::dse::PreMappingSpectrumClass::MaxTemporal;
      case loom::dse::PreMappingSpectrumEndpoint::MaxSpatial:
        return loom::dse::PreMappingSpectrumClass::MaxSpatial;
      case loom::dse::PreMappingSpectrumEndpoint::Intermediate:
        return loom::dse::PreMappingSpectrumClass::Intermediate;
      case loom::dse::PreMappingSpectrumEndpoint::Automatic:
        llvm_unreachable("automatic spectrum endpoint was not requested");
      }
      llvm_unreachable("unknown spectrum endpoint");
    }();
    const bool verified = llvm::any_of(
        execution->candidateOutcomes, [&](const auto &outcome) {
          if (execution->execution.summary.selectedPlanOrdinal &&
              outcome.planOrdinal !=
                  *execution->execution.summary.selectedPlanOrdinal)
            return false;
          if (!execution->execution.summary.selectedMapping ||
              !llvm::is_contained(outcome.systemMappings,
                                  *execution->execution.summary.selectedMapping))
            return false;
          if (!outcome.resourceTimeSpectrum)
            return false;
          const auto *spectrum = std::get_if<loom::dse::VerifiedResourceTimeSpectrum>(
              &outcome.resourceTimeSpectrum->verification);
          return spectrum && llvm::any_of(
                                 spectrum->scenarios, [&](const auto &scenario) {
                                   return scenario.spectrumClass == requestedClass;
                                 });
        });
    if (!verified)
      return productError(
          "loom_mapping_spectrum_endpoint_unsupported",
          "requested endpoint has no verified SystemMapping schedule");
  }
  return std::move(*execution);
}

llvm::Error publishProductDeployment(const ProductFinalLinkArtifacts &finalLink,
                                     PreparedProductTarget &target) {
  auto deadline = makeProductMappingExecutionDeadline();
  if (!deadline)
    return deadline.takeError();
  auto stoppingPolicy = parseMappingStoppingPolicy();
  if (!stoppingPolicy)
    return stoppingPolicy.takeError();
  auto spectrumEndpoint = parseMappingSpectrumEndpoint();
  if (!spectrumEndpoint)
    return spectrumEndpoint.takeError();
  ProductMappingExecutionPolicyReporter reporter(*deadline);
  const loom::ExecutionControlView executionControl =
      *deadline
          ? loom::ExecutionControlView{&**deadline, productMappingStopRequested,
                                       productMappingRemainingTime}
          : loom::ExecutionControlView{};
  auto prepared = prepareMappedApplication(*finalLink.linkedModule, target,
                                           executionControl, *stoppingPolicy,
                                           *spectrumEndpoint);
  if (!prepared)
    return prepared.takeError();
  auto mapping = executeProductMapping(
      *prepared, target,
      *deadline
          ? std::optional<std::uint64_t>{(*deadline)->notAfterUnixNanoseconds}
          : std::nullopt,
      *stoppingPolicy);
  if (!mapping)
    return mapping.takeError();
  loom::mapping::SystemMappingImportSession systemMappingImportSession(
      target.workspace->artifacts(), 1);
  loom::deployment::ConfigurationImageProjectionSession projectionSession(
      target.workspace->artifacts(), 1);
  auto deployment = loom::application::buildApplicationDeployment(
      *prepared, *mapping, *finalLink.linkedModule,
      {target.compilerPolicy, {target.workspace->linkerPath().str()}},
      target.workspace->artifacts(), target.workspace->blobs());
  if (!deployment) {
    loom::deployment::emitConfigurationImageProjectionSessionStatistics(
        loom::deployment::ConfigurationImageProjectionVerificationDomain::
            SourceInvocation,
        projectionSession.statistics());
    loom::mapping::emitSystemMappingImportSessionStatistics(
        loom::mapping::SystemMappingImportVerificationDomain::SourceInvocation,
        systemMappingImportSession.statistics());
    return deployment.takeError();
  }
  const auto packageBegin = MonotonicClock::now();
  llvm::Error packageError = loom::deployment::publishDeploymentPackage(
      deployment->deployment, target.workspace->deploymentPath(),
      target.workspace->artifacts(), target.workspace->blobs());
  loom::application::emitApplicationBuildOperationStatistics(
      {loom::application::ApplicationBuildOperation::PackagePublication,
       elapsedNanoseconds(packageBegin), 1});
  loom::deployment::emitConfigurationImageProjectionSessionStatistics(
      loom::deployment::ConfigurationImageProjectionVerificationDomain::
          SourceInvocation,
      projectionSession.statistics());
  loom::mapping::emitSystemMappingImportSessionStatistics(
      loom::mapping::SystemMappingImportVerificationDomain::SourceInvocation,
      systemMappingImportSession.statistics());
  return packageError;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Loom application build helper\n");
  loom::fabric::FabricArtifactImportSession importSession;
  loom::hardware::ConfigurationABIImportSession configurationAbiImportSession;
  const bool projectsArguments = !driverArgumentsOutput.empty();
  const bool buildsDeployment = !finalLinkOutput.empty();
  if (projectsArguments == buildsDeployment) {
    llvm::errs() << "loom-application-build: error: select exactly one "
                    "product action\n";
    return 1;
  }

  if (projectsArguments) {
    auto commandLine = prepareProductDriverTarget();
    if (!commandLine) {
      llvm::errs() << "loom-application-build: error: "
                   << llvm::toString(commandLine.takeError()) << '\n';
      return 1;
    }
    if (llvm::Error error = writeDriverArguments(*commandLine)) {
      llvm::errs() << "loom-application-build: error: "
                   << llvm::toString(std::move(error)) << '\n';
      return 1;
    }
    return 0;
  }

  auto target = prepareProductTarget();
  if (!target) {
    llvm::errs() << "loom-application-build: error: "
                 << llvm::toString(target.takeError()) << '\n';
    return 1;
  }

  llvm::LLVMContext context;
  auto finalLink = importProductFinalLink(context);
  if (!finalLink) {
    llvm::errs() << "loom-application-build: error: "
                 << llvm::toString(finalLink.takeError()) << '\n';
    return 1;
  }
  if (llvm::Error error = publishProductDeployment(*finalLink, *target)) {
    llvm::errs() << "loom-application-build: error: "
                 << llvm::toString(std::move(error)) << '\n';
    return 1;
  }
  return 0;
}

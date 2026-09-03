#include "Application/ProductBuild.h"

#include "BuildInternal.h"

#include "ADG/Builtin.h"
#include "Application/Build.h"
#include "Application/BuildDiagnostics.h"
#include "Application/Manifest.h"
#include "Application/Package.h"
#include "Application/ProductVisualization.h"
#include "Application/SourceAdmission.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Common/ExecutionControl.h"
#include "Config/ResolvedConfig.h"
#include "DSE/CandidateGenerator.h"
#include "Deployment/HardwareConfigurationImage.h"
#include "Evaluation/ModelParameterBundle.h"
#include "Evaluation/Models/FpaParameterContract.h"
#include "Evaluation/ProductionRegistry.h"
#include "Evaluation/Request.h"
#include "ExternalTool/LocalConfig.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Payload/AcceleratorFinalLink.h"
#include "Hardware/Configuration/ConfigurationABI.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::application {
namespace {

constexpr llvm::StringLiteral portfolioFreestandingMathLibraryOption{"-lm"};

using MonotonicClock = std::chrono::steady_clock;

llvm::Error productError(llvm::StringRef kind, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 kind + ": " + message);
}

std::uint64_t elapsedNanoseconds(MonotonicClock::time_point begin) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             MonotonicClock::now() - begin)
      .count();
}

class ApplicationBuildOperationTimer final {
public:
  explicit ApplicationBuildOperationTimer(ApplicationBuildOperation operation)
      : operation_(operation), begin_(MonotonicClock::now()) {}

  ~ApplicationBuildOperationTimer() {
    emitApplicationBuildOperationStatistics(
        {operation_, elapsedNanoseconds(begin_), 1});
  }

  ApplicationBuildOperationTimer(const ApplicationBuildOperationTimer &) =
      delete;
  ApplicationBuildOperationTimer &
  operator=(const ApplicationBuildOperationTimer &) = delete;

private:
  ApplicationBuildOperation operation_;
  MonotonicClock::time_point begin_;
};

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

llvm::Expected<ProductMappingExecutionDeadline>
makeProductMappingExecutionDeadline(std::uint64_t requested) {
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
  return ProductMappingExecutionDeadline{
      requested, now + duration, begin,
      begin + std::chrono::milliseconds(requested)};
}

class ProductMappingExecutionPolicyReporter final {
public:
  explicit ProductMappingExecutionPolicyReporter(
      const ProductMappingExecutionDeadline &deadline)
      : deadline_(deadline) {}

  ~ProductMappingExecutionPolicyReporter() {
    emitApplicationMappingExecutionPolicyStatistics(
        {deadline_.requestedMilliseconds, deadline_.notAfterUnixNanoseconds,
         elapsedNanoseconds(deadline_.begin), deadline_.stopRequested()});
  }

private:
  const ProductMappingExecutionDeadline &deadline_;
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
            llvm::sys::fs::createUniqueDirectory(pattern, root))
      return productError("loom_product_workspace_invalid",
                          "cannot create bounded workspace at '" + pattern +
                              "': " + error.message());
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

  const ArtifactStore &artifacts() const { return artifacts_; }
  const BlobStore &blobs() const { return blobs_; }
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
  ArtifactStore artifacts_;
  BlobStore blobs_;
};

struct PreparedProductTarget final {
  struct PortfolioInput final {
    SelectedApplicationInput selection;
    AdmittedApplicationSource source;
    std::string repositoryRoot;
  };

  std::unique_ptr<ProductWorkspace> workspace;
  ResolvedConfig config;
  fabric::FinalizedFabricRoot system;
  std::vector<ArtifactRootReference> physicalTimingProfiles;
  CompilerTargetPolicy compilerPolicy;
  CompilerTargetCommandLineProjection commandLine;
  std::optional<PortfolioInput> portfolioInput;
  std::optional<ArtifactRootReference> fpaWeight;
  std::vector<evaluation::EvaluationCondition> fpaConditions;
};

struct PreparedProductFpaInputs final {
  std::optional<ArtifactRootReference> weight;
  std::vector<evaluation::EvaluationCondition> conditions;
};

llvm::Expected<PreparedProductFpaInputs>
prepareProductFpaInputs(const ProductBuildOptions &options,
                        const ProductWorkspace &workspace) {
  if (options.fpaWeightRootPath.empty())
    return PreparedProductFpaInputs{};
  if (llvm::Error error = evaluation::registerProductionEvaluationRegistry())
    return std::move(error);

  auto reference = loadArtifactRootReferenceJsonFile(options.fpaWeightRootPath);
  if (!reference)
    return productError("loom_fpa_weight_invalid",
                        llvm::toString(reference.takeError()));
  const ArtifactStore sourceArtifacts(options.fpaArtifactStorePath);
  const BlobStore sourceBlobs(options.fpaBlobStorePath);
  auto bundle =
      evaluation::importModelParameterBundleRoot(*reference, sourceArtifacts);
  if (!bundle)
    return productError("loom_fpa_weight_invalid",
                        llvm::toString(bundle.takeError()));
  if (bundle->parameterContract() !=
      evaluation::models::fpaModelParameterContractRef())
    return productError("loom_fpa_weight_invalid",
                        "model bundle does not use the FPA contract");

  std::vector<evaluation::EvaluationCondition> conditions;
  if (!options.fpaConditionsPath.empty()) {
    auto buffer =
        llvm::MemoryBuffer::getFile(options.fpaConditionsPath, false, false);
    if (!buffer)
      return productError("loom_fpa_conditions_invalid",
                          "cannot read FPA conditions: " +
                              buffer.getError().message());
    auto parsed = evaluation::parseEvaluationConditions((*buffer)->getBuffer());
    if (!parsed)
      return productError("loom_fpa_conditions_invalid",
                          llvm::toString(parsed.takeError()));
    conditions = std::move(*parsed);
  }

  auto canonical = sourceArtifacts.get(evaluation::modelParameterBundleSchema,
                                       reference->artifact);
  if (!canonical)
    return productError("loom_fpa_weight_invalid",
                        llvm::toString(canonical.takeError()));
  auto copiedRoot = workspace.artifacts().put(
      evaluation::modelParameterBundleSchema, *canonical);
  if (!copiedRoot)
    return copiedRoot.takeError();
  if (*copiedRoot != reference->artifact)
    return productError("loom_fpa_weight_invalid",
                        "bundle root identity changed during import");
  auto copiedBytes = workspace.blobs().importVerified(
      bundle->payloadDigest(), sourceBlobs,
      evaluation::models::maximumFpaModelParameterPayloadBytes);
  if (!copiedBytes)
    return productError("loom_fpa_weight_invalid",
                        llvm::toString(copiedBytes.takeError()));
  auto weight = evaluation::models::importEdaPredictionModelWeight(
      *reference, workspace.artifacts(), workspace.blobs());
  if (!weight)
    return productError("loom_fpa_weight_invalid",
                        llvm::toString(weight.takeError()));
  return PreparedProductFpaInputs{weight->reference(), std::move(conditions)};
}

llvm::Error validatePortfolioBuildOptions(const BuildSelection &build) {
  const auto isProductOwned = [](llvm::StringRef option) {
    return option == "-target" || option == "--target" || option == "-march" ||
           option == "-mabi" || option == "-mcmodel" || option == "-mcpu" ||
           option == "-B" || option == "-o" || option == "-x" ||
           option == "-Xlinker" || option == "-nostdlib" ||
           option.starts_with("-target=") || option.starts_with("--target=") ||
           option.starts_with("-march=") || option.starts_with("-mabi=") ||
           option.starts_with("-mcmodel=") || option.starts_with("-mcpu=") ||
           option.starts_with("-B") || option.starts_with("-o") ||
           option.starts_with("-working-directory") ||
           option.starts_with("-fuse-ld=") || option.starts_with("-flto") ||
           option.starts_with("-fno-lto") ||
           option.starts_with("-ffat-lto-objects") ||
           option.starts_with("-fno-fat-lto-objects") ||
           option.starts_with("-Wl,--entry") ||
           option.starts_with("-Wl,--fat-lto-objects") ||
           option.starts_with("-Wl,--save-temps") ||
           option.starts_with("-Wl,--unresolved-symbols") ||
           option.starts_with("-Wl,--lto-O") ||
           option.starts_with("-Wl,--plugin-opt=-mattr=");
  };
  for (const std::string &option : build.compilerOptions)
    if (isProductOwned(option))
      return productError(
          "loom_portfolio_build_invalid",
          "manifest compiler option '" + option +
              "' conflicts with product-owned compiler target policy");
  for (const std::string &option : build.linkOptions)
    if (isProductOwned(option))
      return productError(
          "loom_portfolio_build_invalid",
          "manifest link option '" + option +
              "' conflicts with product-owned compiler target policy");
    else if (llvm::StringRef(option).starts_with("-l") &&
             option != portfolioFreestandingMathLibraryOption)
      return productError(
          "loom_portfolio_build_invalid",
          "manifest link option '" + option +
              "' names a library unavailable to the product freestanding "
              "runtime");
  return llvm::Error::success();
}

llvm::Expected<std::optional<PreparedProductTarget::PortfolioInput>>
resolvePortfolioInput(const ProductBuildOptions &options) {
  if (options.portfolioManifestPath.empty())
    return std::optional<PreparedProductTarget::PortfolioInput>{};
  auto manifest = loadApplicationManifest(options.portfolioManifestPath);
  if (!manifest)
    return productError("loom_portfolio_manifest_invalid",
                        llvm::toString(manifest.takeError()));
  auto selection =
      selectApplicationInput(*manifest, options.portfolioApplicationIdentity,
                             options.portfolioInputName);
  if (!selection)
    return productError("loom_portfolio_selection_invalid",
                        llvm::toString(selection.takeError()));
  if (llvm::Error error = validatePortfolioBuildOptions(selection->build))
    return std::move(error);
  std::optional<llvm::StringRef> cacheRoot;
  if (!options.portfolioCacheRoot.empty())
    cacheRoot = options.portfolioCacheRoot;
  auto outcome = admitApplicationSource(
      *manifest, options.portfolioApplicationIdentity,
      options.portfolioInputName, options.portfolioRepositoryRoot, cacheRoot);
  if (!outcome)
    return productError("loom_portfolio_source_invalid",
                        llvm::toString(outcome.takeError()));
  if (const auto *unavailable =
          std::get_if<UnavailableApplicationSource>(&*outcome))
    return productError("loom_portfolio_source_unavailable",
                        llvm::Twine("application '") +
                            unavailable->applicationIdentity + "' requires " +
                            toString(unavailable->reason) + " at '" +
                            unavailable->path + "'");
  auto source = std::get<AdmittedApplicationSource>(std::move(*outcome));
  return std::optional<PreparedProductTarget::PortfolioInput>{
      PreparedProductTarget::PortfolioInput{std::move(*selection),
                                            std::move(source),
                                            options.portfolioRepositoryRoot}};
}

bool sameCommandLineTarget(const CompilerTargetCommandLineProjection &lhs,
                           const CompilerTargetCommandLineProjection &rhs) {
  return lhs.targetTriple == rhs.targetTriple &&
         lhs.architecture == rhs.architecture && lhs.abi == rhs.abi &&
         lhs.codeModel == rhs.codeModel && lhs.backendCpu == rhs.backendCpu &&
         lhs.ltoFeatures == rhs.ltoFeatures &&
         lhs.positionIndependent == rhs.positionIndependent;
}

llvm::Expected<fabric::FinalizedFabricRoot>
finalizeExternalSystem(llvm::StringRef path, const ArtifactStore &store) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return productError("loom_hardware_import_missing",
                        "cannot read external Fabric MLIR: " +
                            buffer.getError().message());

  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());
  auto source = mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
  if (!source)
    return productError("loom_hardware_import_malformed",
                        "cannot parse external Fabric MLIR");

  std::vector<ArtifactRootReference> importedModules;
  for (::fabric::ModuleOp module : source->getOps<::fabric::ModuleOp>()) {
    auto finalized = fabric::finalizeFabricRoot(module, store);
    if (!finalized)
      return productError("loom_hardware_module_invalid",
                          llvm::toString(finalized.takeError()));
    importedModules.push_back(finalized->reference());
  }

  llvm::SmallVector<::fabric::SystemOp, 2> systems;
  for (::fabric::SystemOp system : source->getOps<::fabric::SystemOp>())
    systems.push_back(system);
  if (systems.size() != 1)
    return productError("loom_hardware_root_invalid",
                        "external Fabric MLIR must contain exactly one System "
                        "root");
  for (mlir::Operation &operation : source->getBody()->getOperations())
    if (!mlir::isa<::fabric::ModuleOp, ::fabric::SystemOp>(operation))
      return productError("loom_hardware_root_invalid",
                          "external Fabric MLIR contains a non-Fabric root");

  auto finalized =
      fabric::finalizeFabricRoot(systems.front(), importedModules, store);
  if (!finalized)
    return productError("loom_hardware_system_invalid",
                        llvm::toString(finalized.takeError()));
  auto imported = fabric::importEntireFabricRoot(finalized->reference(), store);
  if (!imported)
    return productError("loom_hardware_reimport_failed",
                        llvm::toString(imported.takeError()));
  auto systemView = fabric::requireSystemRoot(imported->view());
  if (!systemView)
    return productError("loom_hardware_root_invalid",
                        llvm::toString(systemView.takeError()));
  return std::move(*imported);
}

llvm::Expected<PreparedProductTarget>
prepareProductTarget(const ProductBuildOptions &options) {
  ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::ProductTargetPreparation);
  auto portfolioInput = resolvePortfolioInput(options);
  if (!portfolioInput)
    return portfolioInput.takeError();
  auto workspace = ProductWorkspace::create(options.deploymentOutput);
  if (!workspace)
    return workspace.takeError();
  auto fpa = prepareProductFpaInputs(options, **workspace);
  if (!fpa)
    return fpa.takeError();
  auto profile = resolveConfigProfileWithProvenance(options.accelerationProfile);
  if (!profile)
    return profile.takeError();
  auto config = std::move(profile->config);
  // A product build needs one verified Mapping, not the best Mapping in the
  // configured restart budget. Exhausting every restart multiplies Spatial and
  // System PnR by the restart count for a result the product path discards, so
  // a builtin preset stops at its first verified candidate. An explicit
  // ResolvedConfig remains the single policy owner: when the profile names a
  // configuration file, its completion goals are published and executed
  // exactly as written.
  if (isBuiltinConfigProfile(options.accelerationProfile) ||
      !profile->spatialPnrAuthored) {
    config.dse.spatialPnr.search.completionGoal =
        ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  }
  if (isBuiltinConfigProfile(options.accelerationProfile) ||
      !profile->systemPnrAuthored) {
    config.dse.systemPnr.search.completionGoal =
        ResolvedPnrCompletionGoal::FirstVerifiedCandidate;
  }
  auto publishedConfig = (*workspace)
                             ->artifacts()
                             .put(ResolvedConfig::artifactSchema,
                                  canonicalResolvedConfigBytes(config));
  if (!publishedConfig)
    return publishedConfig.takeError();
  if (*publishedConfig != resolvedConfigIdentity(config))
    return productError("loom_product_target_invalid",
                        "resolved configuration publication changed its "
                        "identity");

  llvm::Expected<fabric::FinalizedFabricRoot> system =
      options.externalHardwarePath.empty()
      ? [&]() -> llvm::Expected<fabric::FinalizedFabricRoot> {
    if (!adg::findBuiltinTargetDescriptor(
            config.hardwareTarget.templateIdentity,
            config.hardwareTarget.schemaVersion.major,
            config.hardwareTarget.schemaVersion.minor) ||
        !adg::isValidBuiltinTargetScale(config.hardwareTarget.parameters))
      return productError("loom_product_target_invalid",
                          "resolved builtin target descriptor is invalid");
    auto design = adg::buildBuiltinTarget(
        (*workspace)->artifacts(), config.hardwareTarget.templateIdentity,
        config.hardwareTarget.schemaVersion.major,
        config.hardwareTarget.schemaVersion.minor,
        config.hardwareTarget.parameters);
    if (!design)
      return design.takeError();
    if (design->roots().size() != 1)
      return productError("loom_product_target_invalid",
                          "resolved target did not produce one System root");
    return fabric::importEntireFabricRoot(design->roots().front().reference(),
                                          (*workspace)->artifacts());
  }()
      : finalizeExternalSystem(options.externalHardwarePath,
                               (*workspace)->artifacts());
  if (!system)
    return system.takeError();
  auto systemView = fabric::requireSystemRoot(system->view());
  if (!systemView)
    return systemView.takeError();

  std::vector<ArtifactRootReference> timingReferences;
  auto timing =
      fabric::projectNormalizedSystemPhysicalTimingProfiles(*systemView);
  if (!timing)
    return timing.takeError();
  timingReferences.reserve(timing->size());
  for (const auto &profile : *timing) {
    auto published = fabric::publishFabricPhysicalTimingProfile(
        profile, (*workspace)->artifacts());
    if (!published)
      return published.takeError();
    timingReferences.push_back(std::move(*published));
  }

  CompilerTargetPolicy compilerPolicy = portableRiscV64CompilerTargetPolicy();
  auto compilerTargets = resolveSystemCompilerTargetBindings(
      *system, compilerPolicy, (*workspace)->artifacts());
  if (!compilerTargets)
    return compilerTargets.takeError();
  auto commandLine =
      projectCompilerTargetCommandLine(compilerTargets->host().binding());
  if (!commandLine)
    return commandLine.takeError();
  for (const auto &group : compilerTargets->instructionGroups()) {
    auto groupCommandLine =
        projectCompilerTargetCommandLine(group.binding().binding());
    if (!groupCommandLine)
      return groupCommandLine.takeError();
    if (!sameCommandLineTarget(*commandLine, *groupCommandLine))
      return productError(
          "loom_product_target_unsupported",
          "one final-linked module cannot serve the heterogeneous compiler "
          "target cohort");
  }
  return PreparedProductTarget{
      std::move(*workspace),      std::move(config),
      std::move(*system),         std::move(timingReferences),
      std::move(compilerPolicy),  std::move(*commandLine),
      std::move(*portfolioInput), std::move(fpa->weight),
      std::move(fpa->conditions)};
}

std::vector<std::string> projectDriverArguments(
    const CompilerTargetCommandLineProjection &target,
    const std::optional<PreparedProductTarget::PortfolioInput> &portfolioInput,
    llvm::StringRef linkerWorkspace) {
  std::vector<std::string> result;
  result.push_back("--target=" + target.targetTriple);
  result.push_back("-march=" + target.architecture);
  result.push_back("-mabi=" + target.abi);
  result.push_back("-mcmodel=" + target.codeModel);
  result.push_back("-mcpu=" + target.backendCpu);
  result.push_back("-B" LOOM_LLVM_TOOLS_DIR);
  result.push_back("-fuse-ld=lld");
  result.push_back("-nostdlib");
  const llvm::StringRef entrySymbol =
      portfolioInput && portfolioInput->selection.build.productExecution
          ? portfolioInput->selection.build.productExecution->entrySymbol
          : llvm::StringRef("main");
  result.push_back("-Wl,--entry=" + entrySymbol.str());
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
  if (portfolioInput) {
    result.push_back("-o");
    result.push_back(
        (std::filesystem::path(linkerWorkspace.str()) / "portfolio.elf")
            .string());
    result.push_back("-working-directory=" + portfolioInput->repositoryRoot);
    result.insert(result.end(),
                  portfolioInput->selection.build.compilerOptions.begin(),
                  portfolioInput->selection.build.compilerOptions.end());
    result.push_back("-x");
    result.push_back(toString(portfolioInput->selection.build.language).str());
    for (const std::string &source : portfolioInput->selection.build.sources)
      result.push_back(
          (std::filesystem::path(portfolioInput->source.sourceRoot) / source)
              .string());
    for (const std::string &option :
         portfolioInput->selection.build.linkOptions)
      if (option != portfolioFreestandingMathLibraryOption)
        result.push_back(option);
  }
  return result;
}

struct ProductFinalLinkArtifacts final {
  std::unique_ptr<llvm::Module> linkedModule;
};

llvm::Expected<ProductFinalLinkArtifacts>
importProductFinalLink(llvm::StringRef output, llvm::LLVMContext &context) {
  ApplicationBuildOperationTimer timer(
      ApplicationBuildOperation::FinalLinkImport);
  const std::string resolutionPath = (output + ".resolution.txt").str();
  const std::string bitcodePath = (output + ".0.5.precodegen.bc").str();

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
  llvm::SmallString<256> inputDirectory(output);
  if (std::error_code error = llvm::sys::fs::make_absolute(inputDirectory))
    return productError("loom_final_link_artifact_invalid",
                        "cannot resolve final-link output directory: " +
                            error.message());
  llvm::sys::path::remove_filename(inputDirectory);
  auto module = importLldAcceleratorFinalLink((*resolution)->getMemBufferRef(),
                                              (*bitcode)->getMemBufferRef(),
                                              context, inputDirectory);
  if (!module)
    return module.takeError();
  return ProductFinalLinkArtifacts{std::move(*module)};
}

llvm::Expected<std::vector<std::uint8_t>>
readPinnedBytes(llvm::StringRef path, const BlobDigest &expected,
                llvm::StringRef role) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return productError("loom_portfolio_input_invalid",
                        "cannot read " + role + ": " +
                            buffer.getError().message());
  const llvm::StringRef contents = (*buffer)->getBuffer();
  const llvm::ArrayRef<std::uint8_t> bytes(
      reinterpret_cast<const std::uint8_t *>(contents.data()),
      contents.size());
  const BlobDigest actual = computeBlobDigest(bytes);
  if (actual != expected)
    return productError("loom_portfolio_input_invalid",
                        role + " changed after source admission");
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}

llvm::Expected<std::vector<std::uint8_t>> decodeHexSampleOracle(
    llvm::StringRef path, const OracleSelection &oracle,
    const WorkloadExecutionProfile &profile,
    std::uint64_t bytesPerSample) {
  auto bytes = readPinnedBytes(path, oracle.digest, "product oracle");
  if (!bytes)
    return bytes.takeError();
  if (oracle.kind != OracleKind::Exact ||
      oracle.encoding != OracleEncoding::HexSampleLines)
    return productError("loom_portfolio_oracle_invalid",
                        "product oracle has the wrong typed encoding");
  if (bytesPerSample > std::numeric_limits<std::uint64_t>::max() / 2 ||
      profile.measuredSamples >
          std::numeric_limits<std::uint64_t>::max() / bytesPerSample)
    return productError("loom_portfolio_oracle_invalid",
                        "product oracle byte extent overflows uint64");
  const std::uint64_t expectedBytes =
      profile.measuredSamples * bytesPerSample;
  if (expectedBytes > std::numeric_limits<std::size_t>::max())
    return productError("loom_portfolio_oracle_invalid",
                        "product oracle exceeds host addressability");

  const auto hexValue = [](unsigned char character) -> std::optional<unsigned> {
    if (character >= '0' && character <= '9')
      return character - '0';
    if (character >= 'a' && character <= 'f')
      return character - 'a' + 10;
    return std::nullopt;
  };
  llvm::StringRef remaining(
      reinterpret_cast<const char *>(bytes->data()), bytes->size());
  std::vector<std::uint8_t> decoded;
  decoded.reserve(static_cast<std::size_t>(expectedBytes));
  std::uint64_t sample = 0;
  while (!remaining.empty()) {
    const auto split = remaining.split('\n');
    const llvm::StringRef line = split.first;
    remaining = split.second;
    if (sample >= profile.measuredSamples)
      return productError("loom_portfolio_oracle_invalid",
                          "product oracle has too many sample rows");
    const std::string prefix =
        "measured_sample=" + std::to_string(sample) + " output=";
    if (!line.starts_with(prefix))
      return productError("loom_portfolio_oracle_invalid",
                          "product oracle sample ordinals are not dense");
    const llvm::StringRef encoded = line.drop_front(prefix.size());
    if (encoded.size() != bytesPerSample * 2)
      return productError("loom_portfolio_oracle_invalid",
                          "product oracle sample has the wrong byte extent");
    for (std::size_t index = 0; index != encoded.size(); index += 2) {
      const std::optional<unsigned> high = hexValue(encoded[index]);
      const std::optional<unsigned> low = hexValue(encoded[index + 1]);
      if (!high || !low)
        return productError("loom_portfolio_oracle_invalid",
                            "product oracle contains non-lowercase-hex data");
      decoded.push_back(static_cast<std::uint8_t>((*high << 4) | *low));
    }
    ++sample;
  }
  if (sample != profile.measuredSamples || decoded.size() != expectedBytes)
    return productError("loom_portfolio_oracle_invalid",
                        "product oracle does not cover every measured sample");
  return decoded;
}

sim::CanonicalValueSequence fixedU64(std::uint64_t value) {
  return {1, {sim::SemanticLane::defined(llvm::APInt(64, value))}};
}

sim::RuntimeMemoryObject definedMemoryObject(
    llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<sim::SemanticMemoryByte> memory;
  memory.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    memory.push_back({sim::SemanticState::Defined, byte});
  return sim::RuntimeMemoryObject(std::move(memory));
}

struct PreparedPortfolioProductInvocation final {
  ApplicationSourceInvocation sourceInvocation;
  ProductOracleContract oracle;
};

llvm::Expected<PreparedPortfolioProductInvocation>
preparePortfolioProductInvocation(
    const PreparedProductTarget::PortfolioInput &portfolio,
    const BlobStore &blobs) {
  const BuildSelection &build = portfolio.selection.build;
  if (!build.productExecution)
    return productError("loom_portfolio_build_invalid",
                        "portfolio input has no product execution contract");
  const auto admittedInput = llvm::find_if(
      portfolio.source.inputs, [&](const AdmittedApplicationInput &input) {
        return input.inputName == portfolio.selection.input.name;
      });
  if (admittedInput == portfolio.source.inputs.end() ||
      admittedInput->cachedInputs.size() !=
          portfolio.selection.cachedInputs.size())
    return productError("loom_portfolio_input_invalid",
                        "admission lost the selected product input");

  ApplicationSourceInvocation invocation;
  invocation.entrySymbol = build.productExecution->entrySymbol;
  invocation.observeReturnValue = true;
  for (const auto indexed :
       llvm::enumerate(portfolio.selection.cachedInputs)) {
    const CachedInput &selected = indexed.value();
    const AdmittedCachedInput &admitted =
        admittedInput->cachedInputs[indexed.index()];
    if (selected.logicalName != admitted.logicalName)
      return productError("loom_portfolio_input_invalid",
                          "admitted cache order differs from the manifest");
    auto bytes = readPinnedBytes(admitted.path, selected.digest,
                                 "cached product input");
    if (!bytes)
      return bytes.takeError();
    invocation.argumentPlan.push_back(sim::StructuredRuntimeMemoryInput{});
    invocation.argumentPlan.push_back(fixedU64(bytes->size()));
    invocation.memoryObjects.push_back(definedMemoryObject(*bytes));
    invocation.pointerBindings.push_back(
        {static_cast<std::uint64_t>(indexed.index() * 2),
         static_cast<std::uint64_t>(indexed.index()), 0});
  }

  const WorkloadExecutionProfile &profile = portfolio.selection.input.profile;
  const std::uint64_t bytesPerSample =
      build.productExecution->measuredOutputBytesPerSample;
  if (profile.measuredSamples >
      std::numeric_limits<std::uint64_t>::max() / bytesPerSample)
    return productError("loom_portfolio_profile_invalid",
                        "product output extent overflows uint64");
  const std::uint64_t outputBytes =
      profile.measuredSamples * bytesPerSample;
  if (outputBytes == 0 ||
      outputBytes > std::numeric_limits<std::size_t>::max())
    return productError("loom_portfolio_profile_invalid",
                        "product output extent is not addressable");
  invocation.argumentPlan.push_back(fixedU64(profile.warmupSamples));
  invocation.argumentPlan.push_back(fixedU64(profile.measuredSamples));
  const std::uint64_t outputArgument = invocation.argumentPlan.size();
  invocation.argumentPlan.push_back(sim::StructuredRuntimeMemoryInput{});
  invocation.argumentPlan.push_back(fixedU64(outputBytes));
  invocation.memoryObjects.push_back(definedMemoryObject(
      std::vector<std::uint8_t>(static_cast<std::size_t>(outputBytes), 0)));
  invocation.pointerBindings.push_back(
      {outputArgument,
       static_cast<std::uint64_t>(invocation.memoryObjects.size() - 1), 0});
  invocation.memoryObservables.push_back(
      {outputArgument, sim::MemoryObservationForm::FullState,
       deployment::HostExternalInterfaceDirection::Output});

  auto expectedBytes = decodeHexSampleOracle(
      admittedInput->oraclePath, portfolio.selection.input.oracle, profile,
      bytesPerSample);
  if (!expectedBytes)
    return expectedBytes.takeError();
  auto expectedOutput = blobs.put(*expectedBytes);
  if (!expectedOutput)
    return expectedOutput.takeError();
  return PreparedPortfolioProductInvocation{
      std::move(invocation),
      ProductOracleContract{ProductEntryABI::CachedInputsProfileOutputV1,
                            build.productExecution->entrySymbol,
                            profile.warmupSamples,
                            profile.measuredSamples,
                            bytesPerSample,
                            *expectedOutput,
                            static_cast<std::uint64_t>(
                                portfolio.selection.cachedInputs.size())}};
}

llvm::Expected<PreparedApplicationBuild> prepareMappedApplication(
    const llvm::Module &module, PreparedProductTarget &target,
    const ProductBuildOptions &options, ExecutionControlView executionControl) {
  constexpr std::uint64_t softwareFrontierLimit = 8;
  constexpr std::uint64_t hardwareFrontierLimit = 8;
  constexpr std::uint64_t spatialMappingFrontierLimit = 32;
  const bool productPortfolio =
      target.portfolioInput &&
      target.portfolioInput->selection.build.productExecution.has_value();
  const llvm::StringRef entrySymbol =
      productPortfolio
          ? llvm::StringRef(target.portfolioInput->selection.build
                                .productExecution->entrySymbol)
          : llvm::StringRef("main");
  const llvm::Function *entry = module.getFunction(entrySymbol);
  if (!entry || entry->isDeclaration())
    return productError("loom_application_entry_unsupported",
                        "the final-linked module has no defined product entry");
  if (entry->isVarArg())
    return productError("loom_application_entry_unsupported",
                        "the product entry cannot be variadic");

  ApplicationSourceInvocation sourceInvocation;
  std::optional<ProductOracleContract> productOracle;
  if (productPortfolio) {
    auto prepared = preparePortfolioProductInvocation(*target.portfolioInput,
                                                      target.workspace->blobs());
    if (!prepared)
      return prepared.takeError();
    const std::size_t cachedInputCount =
        target.portfolioInput->selection.cachedInputs.size();
    const std::size_t expectedArguments = cachedInputCount * 2 + 4;
    if (!entry->getReturnType()->isIntegerTy(32) ||
        entry->arg_size() != expectedArguments)
      return productError(
          "loom_application_entry_unsupported",
          "product entry differs from its manifest-derived ABI");
    for (const auto indexed : llvm::enumerate(entry->args())) {
      const std::size_t ordinal = indexed.index();
      const bool pointer =
          (ordinal < cachedInputCount * 2 && ordinal % 2 == 0) ||
          ordinal == cachedInputCount * 2 + 2;
      if (pointer != indexed.value().getType()->isPointerTy() ||
          (!pointer && !indexed.value().getType()->isIntegerTy(64)))
        return productError(
            "loom_application_entry_unsupported",
            "product entry argument differs from its manifest-derived ABI");
    }
    sourceInvocation = std::move(prepared->sourceInvocation);
    productOracle = std::move(prepared->oracle);
  } else {
    if (!entry->arg_empty() || entry->getReturnType()->isVoidTy())
      return productError(
          "loom_application_entry_unsupported",
          "the default product entry requires a nullary value-returning main");
    sourceInvocation.entrySymbol = entrySymbol.str();
    sourceInvocation.observeReturnValue = true;
  }

  auto jointPolicy = dse::JointDesignPolicy::get(
      softwareFrontierLimit, hardwareFrontierLimit, softwareFrontierLimit,
      options.mappingTechCandidateLimit, spatialMappingFrontierLimit);
  if (!jointPolicy)
    return jointPolicy.takeError();
  frontend::PreMappingCompilationOptions compilationOptions;
  dse::PreMappingExplorationOptions preMappingOptions{
      {compilationOptions.lowering,
       {evaluation::MetricRequestOrdinal(0),
        ResolvedObjectiveDirection::Minimize, softwareFrontierLimit},
       1}};
  preMappingOptions.ownership.selectionMode =
      dse::StructuredOwnershipSelectionMode::SemanticConformance;
  preMappingOptions.frontier.stoppingPolicy = options.mappingStoppingPolicy;
  preMappingOptions.executionControl = executionControl;
  dse::ResourceTimeFrontierPolicy resourceTimePolicy;
  resourceTimePolicy.spectrumEndpoint = options.mappingSpectrumEndpoint;
  auto outcome = prepareApplicationBuild(
      module,
      {std::move(sourceInvocation), options.operatorProtocolSymbols,
       target.system.reference(), target.physicalTimingProfiles, target.config,
       std::move(*jointPolicy), std::move(compilationOptions),
       std::move(preMappingOptions), std::move(resourceTimePolicy),
       target.portfolioInput ? std::optional<SelectedApplicationInput>(
                                   target.portfolioInput->selection)
                             : std::nullopt,
       target.fpaWeight, target.fpaConditions, std::move(productOracle)},
      target.workspace->artifacts(), target.workspace->blobs());
  if (!outcome)
    return outcome.takeError();
  if (auto *prepared = std::get_if<PreparedApplicationBuild>(&*outcome))
    return std::move(*prepared);
  if (auto *incomplete =
          std::get_if<dse::IncompletePreMappingExploration>(&*outcome))
    return productError("loom_pre_mapping_incomplete",
                        "candidate exploration ended at node " +
                            llvm::Twine(incomplete->planNodeOrdinal.value_or(
                                std::numeric_limits<std::uint64_t>::max())) +
                            " with reason " +
                            dse::toString(incomplete->reason));
  if (auto *incomplete =
          std::get_if<IncompleteApplicationResourceTimePlanning>(&*outcome))
    return productError("loom_resource_time_planning_incomplete",
                        "resource-time planning ended with reason " +
                            dse::resourceTimeFrontierIncompleteReasonSpelling(
                                incomplete->reason));
  if (const auto *noFeasible =
          std::get_if<dse::CompletedPreMappingNoFeasibleCandidate>(
              &*outcome)) {
    // The pair decision already states the host path; the product build
    // names the refused hardware capability instead of a generic absence.
    if (build_detail::classifyPreMappingNoFeasibleOutcome(*noFeasible) ==
        ApplicationPairDecisionDisposition::ExactHardwareIncompatible)
      return productError(
          "loom_pre_mapping_exact_hardware_incompatible",
          "the exact Fabric keeps the verified host path: " +
              build_detail::exactFabricOwnershipRejection(*noFeasible)
                  ->message);
    return productError("loom_pre_mapping_no_feasible_candidate",
                        "no verified software candidate was selected");
  }
  const auto &unsupported = std::get<UnsupportedApplicationBuild>(*outcome);
  switch (unsupported.kind) {
  case ApplicationBuildUnsupportedKind::RootCoordinates:
    return productError("loom_application_unsupported",
                        "root coordinates are not statically enumerable for "
                        "launch " +
                            llvm::Twine(unsupported.root.entity.value()));
  case ApplicationBuildUnsupportedKind::DirectInvocationBoundary:
    return productError("loom_application_unsupported",
                        "root has no replaceable direct invocation boundary "
                        "for launch " +
                            llvm::Twine(unsupported.root.entity.value()));
  case ApplicationBuildUnsupportedKind::DynamicInvocationBoundary:
    return productError(
        "loom_application_unsupported",
        "dynamic invocation value capture is not exact for launch " +
            llvm::Twine(unsupported.root.entity.value()));
  }
  llvm_unreachable("closed ApplicationBuildUnsupportedKind");
}

llvm::Expected<ApplicationMappingExecution>
executeProductMapping(const PreparedApplicationBuild &prepared,
                      PreparedProductTarget &target,
                      const ProductBuildOptions &options,
                      const external_tool::LocalToolConfig &localToolConfig,
                      std::uint64_t dispatchNotAfterUnixNanoseconds,
                      ExecutionControlView executionControl) {
  auto producer = dse::DseProducerSemanticBuildIdentity::get(
      applicationBuildProducerIdentity);
  if (!producer)
    return producer.takeError();
  const std::uint32_t candidateWorkerCount = dse::defaultCandidateWorkerCount();
  auto capacity = dse::SiteCapacity::get(candidateWorkerCount, 0, 0);
  if (!capacity)
    return capacity.takeError();
  auto claim = dse::SiteResourceClaim::get(candidateWorkerCount, 0, 0);
  if (!claim)
    return claim.takeError();
  std::optional<dse::ExternalExecutionSite> externalSite;
  if (!options.localToolConfigPath.empty())
    externalSite = dse::ExternalExecutionSite{
        localToolConfig,
        dse::ExternalAttemptDisposition::ExecutePrepared,
        1,
        0,
        0,
        false};
  auto policy = dse::PlanExecutionPolicy::get(
      1, std::move(*claim), std::move(externalSite), {}, std::nullopt,
      dispatchNotAfterUnixNanoseconds);
  if (!policy)
    return policy.takeError();
  std::optional<dse::JointBoundedQualityPolicy> boundedQuality;
  if (options.mappingStoppingPolicy ==
      dse::JointDesignStoppingPolicy::BoundedQuality) {
    auto quality = makeApplicationBoundedQualityPolicy(
        prepared, *policy, target.workspace->artifacts(),
        target.workspace->blobs());
    if (!quality)
      return quality.takeError();
    boundedQuality.emplace(std::move(*quality));
  }
  auto execution = executeApplicationMapping(
      prepared,
      {std::move(*producer),
       target.workspace->journalPath().str(),
       {},
       std::move(boundedQuality),
       std::move(*capacity),
       std::move(*policy),
       options.externalHardwarePath.empty()
           ? dse::JointHardwareExplorationScope::BoundedHardwareReopen
           : dse::JointHardwareExplorationScope::FixedSystemFrontier,
       executionControl,
       options.mappingRepairCandidateLimit},
      target.workspace->artifacts(), target.workspace->blobs());
  if (!execution)
    return execution.takeError();
  std::size_t mappingCount = 0;
  for (const dse::JointMappedPair &pair : execution->execution.mappedPairs)
    mappingCount += pair.systemMappings.size();
  if (const auto *incomplete = std::get_if<dse::IncompleteDsePlanExecution>(
          &execution->execution.planExecution)) {
    const auto *generationReason =
        std::get_if<dse::CandidateGeneratorIncompleteReason>(
            &incomplete->reason());
    const bool hasUsableBoundedResult =
        mappingCount != 0 && !incomplete->executionStopped() &&
        generationReason &&
        (*generationReason ==
             dse::CandidateGeneratorIncompleteReason::SemanticLimitReached ||
         *generationReason ==
             dse::CandidateGeneratorIncompleteReason::ProofNotEstablished);
    if (!hasUsableBoundedResult)
      return productError("loom_mapping_incomplete",
                          "joint Mapping ended at node " +
                              llvm::Twine(incomplete->nodeOrdinal()) +
                              " with reason " +
                              dse::toString(incomplete->reason()));
  }
  if (mappingCount == 0 &&
      !execution->execution.summary.qualityIncompleteCandidate)
    return productError("loom_mapping_no_feasible_candidate",
                        "joint Mapping selected no SystemMapping");
  if (options.mappingStoppingPolicy ==
      dse::JointDesignStoppingPolicy::BoundedQuality) {
    switch (execution->execution.summary.qualityDisposition) {
    case dse::JointDesignQualityDisposition::Complete:
      break;
    case dse::JointDesignQualityDisposition::Unsupported:
      return productError("loom_mapping_quality_unsupported",
                          "application QoR acquisition was unsupported");
    case dse::JointDesignQualityDisposition::ProofNotEstablished:
      return productError(
          "loom_mapping_quality_proof_not_established",
          "application QoR acquisition did not establish proof");
    case dse::JointDesignQualityDisposition::ExecutionFailed:
      return productError("loom_mapping_quality_execution_failed",
                          "application QoR acquisition execution failed");
    case dse::JointDesignQualityDisposition::CancelledOrTimeout:
      return productError("loom_mapping_quality_cancelled_or_timeout",
                          "application QoR acquisition was cancelled or "
                          "timed out");
    case dse::JointDesignQualityDisposition::NotRequested:
      return productError("loom_mapping_quality_not_requested",
                          "bounded application QoR acquisition did not run");
    }
  }
  if (mappingCount == 0)
    return productError("loom_mapping_no_feasible_candidate",
                        "joint Mapping selected no SystemMapping");
  if (!execution->execution.summary.selectedMapping)
    return productError("loom_mapping_selection_incomplete",
                        "Mapping returned candidates without a selected root");
  if (prepared.resourceTimePolicy.spectrumEndpoint !=
      dse::PreMappingSpectrumEndpoint::Automatic) {
    const auto requestedClass = [&]() {
      switch (prepared.resourceTimePolicy.spectrumEndpoint) {
      case dse::PreMappingSpectrumEndpoint::MaxTemporal:
        return dse::PreMappingSpectrumClass::MaxTemporal;
      case dse::PreMappingSpectrumEndpoint::MaxSpatial:
        return dse::PreMappingSpectrumClass::MaxSpatial;
      case dse::PreMappingSpectrumEndpoint::Intermediate:
        return dse::PreMappingSpectrumClass::Intermediate;
      case dse::PreMappingSpectrumEndpoint::Automatic:
        llvm_unreachable("automatic spectrum endpoint was not requested");
      }
      llvm_unreachable("unknown spectrum endpoint");
    }();
    const bool verified =
        llvm::any_of(execution->candidateOutcomes, [&](const auto &outcome) {
          if (execution->execution.summary.selectedPlanOrdinal &&
              outcome.planOrdinal !=
                  *execution->execution.summary.selectedPlanOrdinal)
            return false;
          if (!execution->execution.summary.selectedMapping ||
              !llvm::is_contained(
                  outcome.systemMappings,
                  *execution->execution.summary.selectedMapping))
            return false;
          if (!outcome.resourceTimeSpectrum)
            return false;
          const auto *spectrum = std::get_if<dse::VerifiedResourceTimeSpectrum>(
              &outcome.resourceTimeSpectrum->verification);
          return spectrum &&
                 llvm::any_of(spectrum->scenarios, [&](const auto &scenario) {
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

llvm::Error publishProductDeployment(
    const ProductFinalLinkArtifacts &finalLink, PreparedProductTarget &target,
    const ProductBuildOptions &options,
    const external_tool::LocalToolConfig &localToolConfig) {
  auto deadline = makeProductMappingExecutionDeadline(
      options.mappingWallTimeLimitMilliseconds);
  if (!deadline)
    return deadline.takeError();
  ProductMappingExecutionPolicyReporter reporter(*deadline);
  const ExecutionControlView executionControl{
      &*deadline, productMappingStopRequested, productMappingRemainingTime};
  auto prepared = prepareMappedApplication(*finalLink.linkedModule, target,
                                           options, executionControl);
  if (!prepared)
    return prepared.takeError();
  auto mapping = executeProductMapping(
      *prepared, target, options, localToolConfig,
      deadline->notAfterUnixNanoseconds, executionControl);
  if (!mapping)
    return mapping.takeError();
  auto deployment = [&]() -> llvm::Expected<ApplicationDeploymentArtifacts> {
    mapping::SystemMappingImportSession systemMappingImportSession(
        target.workspace->artifacts(), 64);
    deployment::ConfigurationImageProjectionSession projectionSession(
        target.workspace->artifacts(), 64);
    auto built = buildApplicationDeployment(
        *prepared, *mapping, *finalLink.linkedModule,
        {target.compilerPolicy,
         {target.workspace->linkerPath().str()},
         executionControl},
        target.workspace->artifacts(), target.workspace->blobs());
    deployment::emitConfigurationImageProjectionSessionStatistics(
        deployment::ConfigurationImageProjectionVerificationDomain::
            SourceInvocation,
        projectionSession.statistics());
    mapping::emitSystemMappingImportSessionStatistics(
        mapping::SystemMappingImportVerificationDomain::SourceInvocation,
        systemMappingImportSession.statistics());
    return built;
  }();
  if (!deployment)
    return deployment.takeError();
  if (!options.visualizationPath.empty())
    if (llvm::Error error = exportProductVisualization(
            options.visualizationPath, target.system, *prepared, *mapping,
            *deployment, target.workspace->artifacts(),
            target.workspace->blobs()))
      return error;
  const auto packageBegin = MonotonicClock::now();
  llvm::Error packageError = publishApplicationPackage(
      *deployment, target.workspace->deploymentPath(),
      target.workspace->artifacts(), target.workspace->blobs());
  emitApplicationBuildOperationStatistics(
      {ApplicationBuildOperation::PackagePublication,
       elapsedNanoseconds(packageBegin), 1});
  return packageError;
}

llvm::Error validateProductOptions(const ProductBuildOptions &options) {
  if (options.deploymentOutput.empty())
    return productError("loom_product_option_invalid",
                        "Deployment output is required");
  if (!options.externalHardwarePath.empty() &&
      !options.accelerationProfile.empty())
    return productError("loom_product_option_invalid",
                        "external hardware and an acceleration profile are "
                        "mutually exclusive");
  if (options.mappingTechCandidateLimit == 0)
    return productError("loom_product_option_invalid",
                        "TechMapping candidate limit must be positive");
  if (options.mappingWallTimeLimitMilliseconds == 0)
    return productError("loom_product_option_invalid",
                        "Mapping wall-time limit must be positive");
  if (options.mappingRepairCandidateLimit &&
      *options.mappingRepairCandidateLimit == 0)
    return productError("loom_product_option_invalid",
                        "Mapping repair candidate limit must be positive");
  const unsigned fpaStorageCount =
      static_cast<unsigned>(!options.fpaWeightRootPath.empty()) +
      static_cast<unsigned>(!options.fpaArtifactStorePath.empty()) +
      static_cast<unsigned>(!options.fpaBlobStorePath.empty());
  if (fpaStorageCount != 0 && fpaStorageCount != 3)
    return productError(
        "loom_product_option_invalid",
        "FPA weight root, ArtifactStore, and BlobStore must be selected "
        "together");
  if (!options.fpaConditionsPath.empty() && fpaStorageCount != 3)
    return productError("loom_product_option_invalid",
                        "FPA conditions require a frozen model weight");
  if (fpaStorageCount == 3 &&
      options.mappingStoppingPolicy !=
          dse::JointDesignStoppingPolicy::BoundedQuality)
    return productError(
        "loom_product_option_invalid",
        "a frozen FPA weight requires bounded_quality Mapping selection");
  const unsigned portfolioSelectorCount =
      static_cast<unsigned>(!options.portfolioManifestPath.empty()) +
      static_cast<unsigned>(!options.portfolioRepositoryRoot.empty()) +
      static_cast<unsigned>(!options.portfolioApplicationIdentity.empty()) +
      static_cast<unsigned>(!options.portfolioInputName.empty());
  if ((portfolioSelectorCount != 0 && portfolioSelectorCount != 4) ||
      (!options.portfolioCacheRoot.empty() && portfolioSelectorCount != 4))
    return productError(
        "loom_product_option_invalid",
        "portfolio manifest, repository, application, and input must be "
        "selected together");
  for (auto indexed : llvm::enumerate(options.operatorProtocolSymbols)) {
    if (indexed.value().empty())
      return productError("loom_product_option_invalid",
                          "operator protocol symbol must be nonempty");
    if (llvm::is_contained(
            llvm::ArrayRef<std::string>(options.operatorProtocolSymbols)
                .take_front(indexed.index()),
            indexed.value()))
      return productError("loom_product_option_invalid",
                          "operator protocol symbol is duplicated");
  }
  return llvm::Error::success();
}

} // namespace

class ProductBuildInvocation::Impl final {
public:
  Impl(ProductBuildOptions options,
       external_tool::LocalToolConfig localToolConfig,
       std::unique_ptr<fabric::FabricArtifactImportSession> fabricImportSession,
       std::unique_ptr<hardware::ConfigurationABIImportSession>
           configurationAbiImportSession,
       PreparedProductTarget target)
      : options_(std::move(options)),
        localToolConfig_(std::move(localToolConfig)),
        fabricImportSession_(std::move(fabricImportSession)),
        configurationAbiImportSession_(
            std::move(configurationAbiImportSession)),
        target_(std::move(target)) {}

  std::vector<std::string> compilerArguments() const {
    return projectDriverArguments(target_.commandLine, target_.portfolioInput,
                                  target_.workspace->linkerPath());
  }

  llvm::Error buildFromFinalLink(llvm::StringRef finalLinkOutput) {
    llvm::LLVMContext context;
    auto finalLink = importProductFinalLink(finalLinkOutput, context);
    if (!finalLink)
      return finalLink.takeError();
    return publishProductDeployment(*finalLink, target_, options_,
                                    localToolConfig_);
  }

private:
  ProductBuildOptions options_;
  external_tool::LocalToolConfig localToolConfig_;
  std::unique_ptr<fabric::FabricArtifactImportSession> fabricImportSession_;
  std::unique_ptr<hardware::ConfigurationABIImportSession>
      configurationAbiImportSession_;
  PreparedProductTarget target_;
};

llvm::Expected<dse::JointDesignStoppingPolicy>
parseProductMappingStoppingPolicy(llvm::StringRef spelling) {
  if (spelling == "first_verified")
    return dse::JointDesignStoppingPolicy::FirstVerified;
  if (spelling == "bounded_quality")
    return dse::JointDesignStoppingPolicy::BoundedQuality;
  return productError("loom_mapping_stopping_policy_invalid",
                      "expected first_verified or bounded_quality");
}

llvm::Expected<dse::PreMappingSpectrumEndpoint>
parseProductMappingSpectrumEndpoint(llvm::StringRef spelling) {
  if (spelling == "automatic")
    return dse::PreMappingSpectrumEndpoint::Automatic;
  if (spelling == "max_temporal")
    return dse::PreMappingSpectrumEndpoint::MaxTemporal;
  if (spelling == "max_spatial")
    return dse::PreMappingSpectrumEndpoint::MaxSpatial;
  if (spelling == "intermediate")
    return dse::PreMappingSpectrumEndpoint::Intermediate;
  return productError("loom_mapping_spectrum_endpoint_invalid",
                      "expected automatic, max_temporal, max_spatial, or "
                      "intermediate");
}

llvm::Expected<std::unique_ptr<ProductBuildInvocation>>
ProductBuildInvocation::create(ProductBuildOptions options) {
  if (llvm::Error error = validateProductOptions(options))
    return std::move(error);
  llvm::Expected<external_tool::LocalToolConfig> localToolConfig =
      options.localToolConfigPath.empty()
          ? llvm::Expected<external_tool::LocalToolConfig>(
                external_tool::defaultLocalToolConfig())
          : external_tool::loadLocalToolConfig(options.localToolConfigPath);
  if (!localToolConfig)
    return productError("loom_local_tool_config_invalid",
                        llvm::toString(localToolConfig.takeError()));
  auto fabricImportSession =
      std::make_unique<fabric::FabricArtifactImportSession>();
  auto configurationAbiImportSession =
      std::make_unique<hardware::ConfigurationABIImportSession>();
  auto target = prepareProductTarget(options);
  if (!target)
    return target.takeError();
  if (target->portfolioInput) {
    if (!options.operatorProtocolSymbols.empty())
      return productError(
          "loom_portfolio_build_invalid",
          "a portfolio selection derives operator protocol symbols from the "
          "manifest");
    options.operatorProtocolSymbols =
        target->portfolioInput->selection.build.operatorProtocolSymbols;
  }
  return std::unique_ptr<ProductBuildInvocation>(new ProductBuildInvocation(
      std::make_unique<Impl>(std::move(options), std::move(*localToolConfig),
                             std::move(fabricImportSession),
                             std::move(configurationAbiImportSession),
                             std::move(*target))));
}

ProductBuildInvocation::ProductBuildInvocation(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

ProductBuildInvocation::ProductBuildInvocation(
    ProductBuildInvocation &&) noexcept = default;

ProductBuildInvocation &
ProductBuildInvocation::operator=(ProductBuildInvocation &&) noexcept = default;

ProductBuildInvocation::~ProductBuildInvocation() = default;

std::vector<std::string> ProductBuildInvocation::compilerArguments() const {
  return impl_->compilerArguments();
}

llvm::Error
ProductBuildInvocation::buildFromFinalLink(llvm::StringRef finalLinkOutput) {
  return impl_->buildFromFinalLink(finalLinkOutput);
}

} // namespace loom::application

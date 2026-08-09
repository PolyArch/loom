#include "EDA/Adapters/Cadence/Voltus.h"

#include "EDA/Adapters/AsicStandardCellContracts.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Evaluation/ModelProvider.h"
#include "ExternalTool/ExternalFile.h"
#include "ExternalTool/Provider.h"
#include "ExternalTool/RuntimeBinding.h"
#include "ExternalTool/ShellProbe.h"
#include "Hardware/Implementation/PhysicalRepresentationIndex.h"
#include "ImplementationPlatform/ImplementationPlatform.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <filesystem>
#include <string>
#include <vector>

namespace loom::eda::cadence {
namespace {

constexpr CadenceImplementationState acceptedStates[]{
    {hardware::RepresentationRootVariant::AsicPhysical,
     hardware::RepresentationPhysicalStage::Routed},
};
constexpr llvm::StringLiteral providerInputs[]{
    evaluation::models::cadenceVoltusPowerGridLibraryInputSlot};
constexpr llvm::StringLiteral declaredOutputs[]{
    "outputs/voltus-rail-result.json"};

const CadenceInvocationDescriptor descriptor{
    &external_tool::voltusProvider(),
    evaluation::models::cadenceVoltusRailImplementationSemanticIdentity,
    CadenceOperation::RailEvaluation,
    acceptedStates,
    true,
    true,
    true,
    providerInputs,
    declaredOutputs,
};

llvm::Error parserError(const llvm::Twine &detail) {
  return makeCadenceAdapterError(CadenceAdapterFailureKind::ParserFailure,
                                 descriptor.implementationSemanticIdentity,
                                 detail);
}

llvm::Error invalid(const llvm::Twine &detail) {
  return makeCadenceAdapterError(CadenceAdapterFailureKind::DescriptorMismatch,
                                 descriptor.implementationSemanticIdentity,
                                 detail);
}

std::string decimalSpelling(evaluation::DecimalValue value) {
  return std::to_string(value.coefficient()) + "e" +
         std::to_string(value.base10Exponent());
}

std::string ratioExpression(evaluation::ExactRatio value) {
  return "[expr {double(" + std::to_string(value.numerator()) + ") / " +
         std::to_string(value.denominator()) + "}]";
}

llvm::Expected<std::string> tclList(llvm::ArrayRef<std::string> values) {
  if (values.empty())
    return invalid("Tcl list input is empty");
  std::string result = "[list";
  for (const std::string &value : values) {
    auto word = renderTclWord(descriptor.implementationSemanticIdentity, value);
    if (!word)
      return word.takeError();
    result += " " + *word;
  }
  result += "]";
  return result;
}

llvm::Expected<std::vector<std::string>> resolvePgvEntrypoints(
    const external_tool::ResolvedExternalFileTree &tree,
    const evaluation::models::CadenceVoltusStaticRailProviderBinding &binding) {
  if (tree.members != binding.powerGridLibraryMembers)
    return invalid("resolved PGV tree differs from the exact model binding");
  const std::filesystem::path root(tree.absolutePath);
  std::vector<std::string> paths;
  paths.reserve(binding.powerGridLibraryEntrypoints.size());
  for (const std::string &entrypoint : binding.powerGridLibraryEntrypoints) {
    const std::filesystem::path path = (root / entrypoint).lexically_normal();
    const std::string relative = path.lexically_relative(root).generic_string();
    if (path.parent_path().empty() || relative.empty() || relative == ".." ||
        llvm::StringRef(relative).starts_with("../"))
      return invalid("PGV entrypoint escapes the resolved tree");
    paths.push_back(path.string());
  }
  return paths;
}

struct VoltusRailInvocationFacts final {
  hardware::FinalizedHardwareImplementation implementation;
  platform::FinalizedImplementationPlatform platform;
  evaluation::models::CompleteRailAnalysisConfiguration analysis;
  hardware::DefSingleSupplyNetwork supplyNetwork;
  external_tool::ExternalToolSemanticContract semanticContract;
  std::vector<external_tool::MaterializedBundleFile> semanticInputs;
  std::vector<std::string> netlists;
  std::vector<std::string> constraints;
  std::string physicalDatabase;
  std::string top;
};

using VoltusRailFactsOrUnsupported =
    std::variant<VoltusRailInvocationFacts, evaluation::UnsupportedEvidence>;

llvm::Expected<std::string> blobText(const BlobStore &blobs,
                                     const BlobDigest &digest) {
  auto contents = blobs.get(digest);
  if (!contents)
    return contents.takeError();
  return std::string(reinterpret_cast<const char *>(contents->data()),
                     contents->size());
}

llvm::Expected<VoltusRailFactsOrUnsupported>
invocationFacts(const evaluation::EvaluationRequest &request,
                const evaluation::CaseArtifactResolution &resolution,
                const ArtifactStore &artifacts, const BlobStore &blobs) {
  using namespace evaluation;
  using namespace evaluation::models;
  using namespace hardware;

  auto analysis =
      projectCompleteRailAnalysisConfiguration(request, resolution, artifacts);
  if (!analysis)
    return analysis.takeError();
  const auto subjects = request.subjectBindings().subjects(
      hardwareImplementationPhysicalSubjectRole());
  if (subjects.size() != 1)
    return invalid("rail request does not bind one HardwareImplementation");

  auto contracts = makeKnownAsicStandardCellContractCatalog();
  if (!contracts)
    return contracts.takeError();
  auto implementation = importHardwareImplementation(
      subjects.front(), *contracts, artifacts, blobs);
  if (!implementation)
    return implementation.takeError();
  const HardwareImplementation &hardware = implementation->implementation();
  const ImplementationRepresentationRoot &representation =
      hardware.representationRoot();
  if (representation.variant != RepresentationRootVariant::AsicPhysical ||
      representation.stage != RepresentationPhysicalStage::Routed ||
      representation.formatRef.kind() !=
          RepresentationFormatKind::IndexedDefPhysical ||
      representation.top.kind != RepresentationObjectKind::PhysicalObject)
    return VoltusRailFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  if (!hardware.implementationPlatform())
    return VoltusRailFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};

  auto target = platform::importImplementationPlatform(
      *hardware.implementationPlatform(), artifacts);
  if (!target)
    return target.takeError();
  if (!std::holds_alternative<platform::AsicTarget>(
          target->platform().target()))
    return VoltusRailFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  if (analysis->processCorner.corner.artifact != target->reference().artifact ||
      !target->platform().findTechnologyCorner(
          analysis->processCorner.corner.entity))
    return invalid(
        "rail Request corner is outside the implementation's exact platform");

  std::vector<external_tool::MaterializedBundleFile> semanticInputs;
  std::vector<std::string> netlists;
  std::vector<std::string> constraints;
  std::string physicalDatabase;
  std::string physicalDatabaseContents;
  for (const ImplementationPayload &payload : representation.payloads) {
    if (payload.role != PayloadRole::Netlist &&
        payload.role != PayloadRole::GenerationConstraint &&
        payload.role != PayloadRole::PhysicalDatabase)
      continue;
    auto contents = blobText(blobs, payload.blobDigest);
    if (!contents)
      return contents.takeError();
    const std::string path =
        "inputs/implementation/" + payload.canonicalLogicalName;
    if (llvm::Error error = validateBundleInputPath(
            descriptor.implementationSemanticIdentity, path))
      return std::move(error);
    semanticInputs.push_back(
        {path, *contents, implementation->reference(), false});
    if (payload.role == PayloadRole::Netlist)
      netlists.push_back(path);
    else if (payload.role == PayloadRole::GenerationConstraint)
      constraints.push_back(path);
    else {
      physicalDatabase = path;
      physicalDatabaseContents = std::move(*contents);
    }
  }
  if (netlists.empty() || constraints.empty() || physicalDatabase.empty())
    return invalid("indexed DEF closure has an incomplete logical payload set");

  auto def = parseDefPhysicalDesign(physicalDatabaseContents,
                                    representation.top.canonicalName,
                                    *representation.stage);
  if (!def)
    return def.takeError();
  auto supplyNetwork = deriveDefSingleSupplyNetwork(*def);
  if (!supplyNetwork)
    return VoltusRailFactsOrUnsupported{
        UnsupportedEvidence{OutcomeReason::RuntimeCapabilityUnavailable}};
  auto semanticContract =
      evaluation::deriveExternalToolSemanticContract(request);
  if (!semanticContract)
    return semanticContract.takeError();
  const std::string top = representation.top.canonicalName;

  return VoltusRailFactsOrUnsupported{VoltusRailInvocationFacts{
      std::move(*implementation), std::move(*target), std::move(*analysis),
      std::move(*supplyNetwork), std::move(*semanticContract),
      std::move(semanticInputs), std::move(netlists), std::move(constraints),
      std::move(physicalDatabase), top}};
}

VoltusRailInvocationConfiguration
invocationConfiguration(const VoltusRailInvocationFacts &facts) {
  return {facts.top,           facts.netlists,
          facts.constraints,   facts.physicalDatabase,
          facts.supplyNetwork, facts.analysis};
}

CadenceBundleInputs bundleInputs(const VoltusRailInvocationFacts &facts,
                                 CadenceFrozenInvocation frozen) {
  return {
      facts.semanticContract,
      &facts.implementation.implementation().representationRoot(),
      facts.implementation.implementation().implementationPlatform(),
      &facts.platform,
      platform::encodeTechnologyCornerRef(facts.analysis.processCorner.corner),
      std::move(frozen),
      facts.semanticInputs};
}

llvm::Expected<evaluation::EvaluationModelProviderPreparation>
prepareEvaluationProvider(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const external_tool::ExternalToolPreparationContext &context) {
  using namespace external_tool;
  auto factsOrUnsupported =
      invocationFacts(request, resolution, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  if (const auto *unsupported =
          std::get_if<evaluation::UnsupportedEvidence>(&*factsOrUnsupported))
    return evaluation::EvaluationModelProviderPreparation{*unsupported};
  const auto &facts = std::get<VoltusRailInvocationFacts>(*factsOrUnsupported);

  auto externalTrees = resolveExternalFileTrees(
      {{evaluation::models::cadenceVoltusPowerGridLibraryInputSlot.str(),
        facts.analysis.providerBinding.powerGridLibraryMembers}},
      context.localConfig);
  if (!externalTrees)
    return externalTrees.takeError();

  const ExternalToolProviderDescriptor &toolProvider = voltusProvider();
  const std::filesystem::path destination(context.bundleDestination);
  const std::filesystem::path probeRoot = destination.parent_path();
  ShellToolBindingProbe toolProbe(probeRoot.string(),
                                  toolProvider.versionProbe);
  auto tool = resolveToolBinding(toolProvider.binding, context.localConfig,
                                 captureToolEnvironment(toolProvider.binding),
                                 toolProbe);
  if (!tool)
    return tool.takeError();
  if (tool->version !=
      facts.analysis.providerBinding.stableProviderBuildIdentity)
    return invalid("resolved Voltus build differs from the model binding");

  std::vector<std::string> inheritEnvironment;
  const auto configured =
      context.localConfig.tools.find(toolProvider.binding.key);
  if (configured != context.localConfig.tools.end())
    inheritEnvironment = configured->second.inheritEnvironment;

  const ExternalToolProviderDescriptor &containerProvider =
      polyArchContainerProvider();
  ShellToolBindingProbe containerProbe(probeRoot.string(),
                                       containerProvider.versionProbe);
  auto runtime = resolveInvocationRuntime(
      *tool, context.localConfig, containerProvider.binding,
      captureToolEnvironment(containerProvider.binding), containerProbe,
      toolProvider.runtimeCompatibility,
      [&](const ResolvedToolBinding &resolvedTool,
          const ResolvedToolBinding &container,
          llvm::StringRef os) -> llvm::Expected<std::optional<std::string>> {
        return probeContainerToolComposition(probeRoot.string(), resolvedTool,
                                             toolProvider.versionProbe,
                                             container, os, inheritEnvironment);
      });
  if (!runtime)
    return runtime.takeError();

  CadenceFrozenInvocation frozen{std::move(*tool),
                                 toolProvider.versionProbe,
                                 std::move(*runtime),
                                 containerProvider.versionProbe,
                                 std::move(inheritEnvironment),
                                 {},
                                 std::move(*externalTrees)};
  CadenceBundleInputs inputs = bundleInputs(facts, std::move(frozen));
  auto specification =
      makeVoltusRailBundleSpec(inputs, invocationConfiguration(facts));
  if (!specification)
    return specification.takeError();
  auto prepared = finalizeExternalToolInvocationBundle(
      context.bundleDestination, std::move(*specification));
  if (!prepared)
    return prepared.takeError();
  return evaluation::EvaluationModelProviderPreparation{std::move(*prepared)};
}

llvm::Expected<evaluation::EvaluationModelResult> importEvaluationProvider(
    const evaluation::EvaluationRequest &request,
    const evaluation::CaseArtifactResolution &resolution,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  using namespace evaluation;
  using namespace external_tool;
  auto factsOrUnsupported =
      invocationFacts(request, resolution, artifacts, blobs);
  if (!factsOrUnsupported)
    return factsOrUnsupported.takeError();
  const auto *facts =
      std::get_if<VoltusRailInvocationFacts>(&*factsOrUnsupported);
  if (!facts)
    return invalid("prepared Voltus invocation is no longer supported");

  const ExternalToolProviderDescriptor &toolProvider = voltusProvider();
  CadenceFrozenInvocation frozen{
      ResolvedToolBinding{
          toolProvider.binding.key,
          ToolBindingSource::Explicit,
          "/unavailable/voltus",
          facts->analysis.providerBinding.stableProviderBuildIdentity,
          {},
          {},
          std::nullopt,
          std::nullopt},
      toolProvider.versionProbe,
      InvocationRuntimeBinding{},
      polyArchContainerProvider().versionProbe,
      {},
      {},
      {ResolvedExternalFileTree{
          models::cadenceVoltusPowerGridLibraryInputSlot.str(),
          {},
          {},
          facts->analysis.providerBinding.powerGridLibraryMembers}}};
  CadenceBundleInputs inputs = bundleInputs(*facts, std::move(frozen));
  auto observation = importVoltusRailObservation(prepared, inputs);
  if (!observation)
    return observation.takeError();
  return EvaluationModelResult{
      {},
      CompletedEvidence{
          {MetricResult{UncertaintyKind::ExactWithinModel,
                        PointObservation{
                            MetricValue{observation->maximumVoltageDropVolts}},
                        {}}},
          {}}};
}

} // namespace

const CadenceInvocationDescriptor &voltusRailDescriptor() { return descriptor; }

llvm::Expected<VoltusRailObservation>
parseVoltusRailObservation(llvm::StringRef contents) {
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return parserError("rail result is malformed JSON: " +
                       llvm::toString(parsed.takeError()));
  const llvm::json::Object *object = parsed->getAsObject();
  if (!object || object->size() != 3)
    return parserError("rail result shape is invalid");
  const auto schema = object->getString("schema");
  const auto version = object->getString("version");
  const auto maximum = object->getString("maximum_voltage_drop_volts");
  if (!schema || *schema != "loom.cadence.voltus_rail_result" || !version ||
      *version != "1.0" || !maximum)
    return parserError("rail result fields are invalid");
  auto parsedMaximum =
      parseCadenceDecimal(descriptor.implementationSemanticIdentity,
                          "maximum_voltage_drop_volts", *maximum, true);
  if (!parsedMaximum)
    return parsedMaximum.takeError();
  return VoltusRailObservation{*parsedMaximum};
}

static llvm::StringRef voltusRailPublisher() {
  return R"tcl(set loom_report [open {work/voltus-ivdd.rpt} r]
set loom_maximum {}
while {[gets $loom_report loom_line] >= 0} {
  set loom_fields [regexp -inline -all {\S+} $loom_line]
  if {[llength $loom_fields] < 2} { continue }
  set loom_value [lindex $loom_fields 0]
  if {![regexp {^[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$} $loom_value]} { continue }
  if {$loom_maximum eq {} || [expr {double($loom_value) > double($loom_maximum)}]} {
    set loom_maximum $loom_value
  }
}
close $loom_report
if {$loom_maximum eq {}} { error {Voltus ivdd report has no observation} }
set loom_output [open {outputs/voltus-rail-result.json} w]
puts $loom_output [format {{"schema":"loom.cadence.voltus_rail_result","version":"1.0","maximum_voltage_drop_volts":"%s"}} $loom_maximum]
close $loom_output
)tcl";
}

llvm::Expected<std::string> renderVoltusRailDriver(
    const VoltusRailInvocationConfiguration &configuration,
    llvm::ArrayRef<std::string> powerGridLibraryEntrypoints) {
  using namespace evaluation;
  using namespace evaluation::models;
  if (!isPortableHdlIdentifier(configuration.top))
    return invalid("representation top is not a portable HDL identifier");
  if (configuration.analysis.model != staticExplicitRailAnalysisModelConfig())
    return invalid("rail analysis model is not the exact static contract");

  std::vector<std::string> requiredPaths = configuration.netlists;
  requiredPaths.insert(requiredPaths.end(),
                       configuration.generationConstraints.begin(),
                       configuration.generationConstraints.end());
  requiredPaths.push_back(configuration.physicalDatabase);
  for (const std::string &path : requiredPaths)
    if (llvm::Error error = validateBundleInputPath(
            descriptor.implementationSemanticIdentity, path))
      return std::move(error);

  auto netlists = tclList(configuration.netlists);
  auto pgvs = tclList(powerGridLibraryEntrypoints);
  auto top = renderTclWord(descriptor.implementationSemanticIdentity,
                           configuration.top);
  auto physical = renderTclWord(descriptor.implementationSemanticIdentity,
                                configuration.physicalDatabase);
  auto power = renderTclWord(descriptor.implementationSemanticIdentity,
                             configuration.supplyNetwork.powerNet);
  auto ground = renderTclWord(descriptor.implementationSemanticIdentity,
                              configuration.supplyNetwork.groundNet);
  if (!netlists)
    return netlists.takeError();
  if (!pgvs)
    return pgvs.takeError();
  if (!top)
    return top.takeError();
  if (!physical)
    return physical.takeError();
  if (!power)
    return power.takeError();
  if (!ground)
    return ground.takeError();

  const std::string voltage =
      decimalSpelling(configuration.analysis.supplyVoltage.volts);
  const std::string temperatureKelvin =
      decimalSpelling(configuration.analysis.temperature.kelvin);
  const std::string period =
      decimalSpelling(configuration.analysis.clockPeriod.seconds) + "s";
  const std::string transitionDensity = ratioExpression(
      configuration.analysis.activity.assumption.transitionsPerClock);
  const std::string staticProbability = ratioExpression(
      configuration.analysis.activity.assumption.staticProbability);

  std::string commands = "read_lib -pgv " + *pgvs + "\n";
  commands += "read_verilog " + *netlists + "\n";
  commands += "set_top_module " + *top + " -ignore_undefined_cell\n";
  for (const std::string &constraint : configuration.generationConstraints) {
    auto word =
        renderTclWord(descriptor.implementationSemanticIdentity, constraint);
    if (!word)
      return word.takeError();
    commands += "read_sdc " + *word + "\n";
  }
  commands += "specify_def " + *physical + "\n";
  commands += "set_power_analysis_mode -reset\n";
  commands += "set_power_analysis_mode -method static "
              "-write_static_currents true -create_binary_db true "
              "-power_grid_library " +
              *pgvs + " -default_supply " + voltage + "\n";
  commands += "set_default_switching_activity -global_activity " +
              transitionDensity + " -duty " + staticProbability + " -period {" +
              period + "}\n";
  commands += "file mkdir {work/voltus-power}\n";
  commands += "set_power_output_dir {work/voltus-power}\n";
  commands += "report_power -outfile {work/voltus-power.rpt}\n";
  commands += "set_power_data -reset\n";
  commands += "set_power_data -power_directory {work/voltus-power}\n";
  commands += "set_pg_nets -net " + *power + " -voltage " + voltage + "\n";
  commands += "set_pg_nets -net " + *ground + " -voltage 0\n";
  commands += "set_rail_analysis_domain -name {loom_complete_domain} "
              "-pwrnets [list " +
              *power + "] -gndnets [list " + *ground + "]\n";
  commands += "set_rail_analysis_mode -method static -accuracy hd "
              "-power_grid_library " +
              *pgvs + " -temperature [expr {" + temperatureKelvin +
              " - 273.15}]\n";
  commands += "create_power_pads -net " + *power +
              " -auto_fetch -honor_pin_connection\n";
  commands += "create_power_pads -net " + *ground +
              " -auto_fetch -honor_pin_connection\n";
  commands += "analyze_rail -type domain -output {work/voltus-rail} "
              "{loom_complete_domain}\n";
  commands += "report_power_rail_results -plot ivdd -nets [list " + *power +
              " " + *ground +
              "] -limit 2147483647 -ignore_limit_bound "
              "-filename {work/voltus-ivdd.rpt}\n";

  return renderCadenceTclBatch(commands,
                               "source {drivers/voltus-rail-publish.tcl}\n");
}

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeVoltusRailBundleSpec(
    const CadenceBundleInputs &inputs,
    const VoltusRailInvocationConfiguration &configuration) {
  if (llvm::Error error = validateCadenceInvocationInputs(descriptor, inputs))
    return std::move(error);

  const auto powerGridLibrary =
      llvm::find_if(inputs.frozen.externalFileTrees, [](const auto &tree) {
        return tree.providerInputSlot ==
               evaluation::models::cadenceVoltusPowerGridLibraryInputSlot;
      });
  if (powerGridLibrary == inputs.frozen.externalFileTrees.end())
    return makeCadenceAdapterError(
        CadenceAdapterFailureKind::MissingProviderInput,
        descriptor.implementationSemanticIdentity,
        "power_grid_library must be one exact external file tree");
  if (configuration.analysis.providerBinding.stableProviderBuildIdentity !=
      inputs.frozen.tool.version)
    return invalid("resolved Voltus build differs from the model binding");
  if (!inputs.technologyCorner ||
      platform::encodeTechnologyCornerRef(
          configuration.analysis.processCorner.corner) !=
          *inputs.technologyCorner)
    return invalid("rail model and invocation technology corners differ");
  auto pgvs = resolvePgvEntrypoints(*powerGridLibrary,
                                    configuration.analysis.providerBinding);
  if (!pgvs)
    return pgvs.takeError();

  std::vector<std::string> required = configuration.netlists;
  required.insert(required.end(), configuration.generationConstraints.begin(),
                  configuration.generationConstraints.end());
  required.push_back(configuration.physicalDatabase);
  if (llvm::Error error =
          validateCadenceSemanticInputs(descriptor, inputs, required))
    return std::move(error);
  auto driver = renderVoltusRailDriver(configuration, *pgvs);
  if (!driver)
    return driver.takeError();
  return makeCadenceInvocationBundleSpec(
      descriptor, inputs,
      {{inputs.frozen.tool.executable, "-no_gui", "-batch", "-files",
        "drivers/voltus-rail.tcl"}},
      {{"drivers/voltus-rail.tcl", std::move(*driver), std::nullopt, false},
       {"drivers/voltus-rail-publish.tcl", voltusRailPublisher().str(),
        std::nullopt, false}});
}

llvm::Expected<VoltusRailObservation> importVoltusRailObservation(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const CadenceBundleInputs &inputs) {
  auto imported = importCadenceInvocation(descriptor, prepared, inputs);
  if (!imported)
    return imported.takeError();
  auto result = readCadenceDeclaredOutput(descriptor, *imported,
                                          "outputs/voltus-rail-result.json");
  if (!result)
    return result.takeError();
  return parseVoltusRailObservation(*result);
}

llvm::Error registerVoltusRailEvaluationProvider() {
  if (llvm::Error error =
          evaluation::models::registerCadenceVoltusStaticRailModel())
    return error;
  static const evaluation::EvaluationModelProvider provider{
      evaluation::models::cadenceVoltusStaticRailModelDescriptorRef(),
      evaluation::EvaluationModelExternalPrepareImportProvider{
          &prepareEvaluationProvider, &importEvaluationProvider}};
  return evaluation::registerEvaluationModelProvider(provider);
}

} // namespace loom::eda::cadence

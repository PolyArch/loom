#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Config/ResolvedConfig.h"
#include "DSE/FabricTemplateCandidateGenerator.h"
#include "DSE/HardwareDecision.h"
#include "DSE/SpatialMicroarchitectureCandidateGenerator.h"
#include "DSE/SpatialTopologyCandidateGenerator.h"
#include "DSE/SystemCompositionCandidateGenerator.h"
#include "DSE/TechMappingHardwareFeedback.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/SpatialPhysicalDemandProjection.h"
#include "Mapping/Tech/TechMappingHardwareDemand.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "hardware candidate generator test failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

void requireError(llvm::Error error, llvm::StringRef expected) {
  if (!error)
    fail("expected an error containing '" + expected + "'");
  std::string message = llvm::toString(std::move(error));
  if (message.find(expected.str()) == std::string::npos)
    fail("unexpected error: " + message);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    std::error_code error = llvm::sys::fs::createUniqueDirectory(
        "loom-hardware-candidate-generator", path_);
    if (error)
      fail("cannot create test directory: " + error.message());
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }

private:
  llvm::SmallString<128> path_;
};

const loom::dse::CompletedCandidateGeneratorResult &
completed(const loom::dse::CandidateGeneratorProviderResult &result) {
  const auto *value = std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
      &result.outcome);
  if (!value)
    fail("provider did not complete");
  return *value;
}

struct Fixture final {
  TemporaryDirectory directory;
  loom::ArtifactStore store;
  loom::BlobStore blobs;

  Fixture()
      : store(directory.path()),
        blobs((llvm::Twine(directory.path()) + "/blobs").str()) {}
};

loom::fabric::FinalizedFabricRoot
generateBuiltinSystem(Fixture &fixture, loom::adg::BuiltinTargetPreset preset,
                      const loom::adg::BuiltinTargetScale &scale) {
  const auto &descriptor = loom::adg::getBuiltinTargetDescriptor(preset);
  auto config = take(loom::dse::resolveFabricTemplateConfig(
      descriptor.templateIdentity, descriptor.schemaMajor,
      descriptor.schemaMinor, scale));
  auto inputs = take(loom::dse::bindFabricTemplateCandidateGeneratorInputs());
  auto binding =
      take(loom::dse::resolveFabricTemplateCandidateGeneratorBinding(config));
  auto result = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, fixture.store, fixture.blobs));
  const auto &output = completed(result).outputBindings.front().artifacts;
  require(output.size() == 1, "template generator did not return one System");
  auto system =
      take(loom::fabric::importEntireFabricRoot(output.front(), fixture.store));
  require(system.view().rootKind() == loom::fabric::FabricRootKind::System,
          "template generator output is not a System");
  return system;
}

loom::fabric::FinalizedFabricRoot
generateBuiltinSystem(Fixture &fixture,
                      loom::adg::BuiltinTargetPreset preset =
                          loom::adg::BuiltinTargetPreset::Small) {
  return generateBuiltinSystem(
      fixture, preset, loom::adg::getBuiltinTargetDescriptor(preset).scale);
}

loom::fabric::FinalizedFabricRoot
importBuiltinModule(const loom::fabric::FinalizedFabricRoot &system,
                    Fixture &fixture) {
  require(system.directDependencies().size() == 1,
          "builtin System did not retain one Module dependency");
  const auto &dependency = system.directDependencies().front();
  require(dependency.role == loom::fabric::FabricDependencyRole::ImportedModule,
          "builtin dependency is not an ImportedModule");
  auto module = take(
      loom::fabric::importEntireFabricRoot(dependency.root, fixture.store));
  require(module.view().rootKind() == loom::fabric::FabricRootKind::Module,
          "builtin dependency is not a Module");
  return module;
}

void parameterizedTemplateScale(
    Fixture &fixture, const loom::fabric::FinalizedFabricRoot &defaultSystem) {
  loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
  resolved.hardwareTarget.templateIdentity =
      loom::adg::builtinSmallTarget.templateIdentity.str();
  resolved.hardwareTarget.schemaVersion = {
      loom::adg::builtinSmallTarget.schemaMajor,
      loom::adg::builtinSmallTarget.schemaMinor};
  auto &scale = resolved.hardwareTarget.parameters;
  scale = loom::adg::builtinSmallTarget.scale;
  scale.accCoreCount = 2;
  scale.meshDimension = 5;
  scale.spatialPeCount = 13;
  scale.temporalPeCount = 5;
  scale.temporalResidentContexts = 3;
  auto projected =
      take(loom::dse::projectResolvedFabricTemplateConfigView(resolved));
  auto inputs = take(loom::dse::bindFabricTemplateCandidateGeneratorInputs());
  auto binding = take(
      loom::dse::resolveFabricTemplateCandidateGeneratorBinding(projected));
  auto generated = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, fixture.store, fixture.blobs));
  const auto &outputs = completed(generated).outputBindings.front().artifacts;
  require(outputs.size() == 1,
          "resolved hardware target did not generate one System");
  auto system = take(
      loom::fabric::importEntireFabricRoot(outputs.front(), fixture.store));
  require(system.reference() != defaultSystem.reference(),
          "resolved target scale did not change the System identity");
  require(system.view().accCoreOccurrences().size() == scale.accCoreCount,
          "resolved AccCore count did not reach the System artifact");
  auto module = importBuiltinModule(system, fixture);
  const std::uint64_t expectedMeshLinkFifos =
      16 * scale.meshDimension * (scale.meshDimension - 1);
  const std::uint64_t expectedAdapterFifos =
      3 * (scale.spatialMemoryCount + scale.temporalMemoryCount) +
      2 * scale.temporalPeCount * scale.crossScheduleBoundaryLanesPerTemporalPe;
  require(module.view().fifoOccurrences().size() ==
              expectedMeshLinkFifos + expectedAdapterFifos,
          "resolved mesh dimension did not reach the Module topology");
  std::uint64_t instructionContexts = 0;
  for (const auto &pe : module.view().peOccurrences())
    instructionContexts += module.view().inventorySize(
        loom::fabric::FabricInventoryOwnerRef::of(pe),
        loom::fabric::FabricInventoryKind::InstructionContext);
  const std::uint64_t expectedContexts =
      scale.spatialPeCount + static_cast<std::uint64_t>(scale.temporalPeCount) *
                                 scale.temporalResidentContexts;
  require(instructionContexts == expectedContexts,
          "resolved PE scale did not reach the Module context inventory");
}

void localMemoryPortVariantRoundTrip() {
  for (const loom::adg::LocalMemoryPortVariant variant :
       {loom::adg::LocalMemoryPortVariant::ElementOnly,
        loom::adg::LocalMemoryPortVariant::VectorOnly,
        loom::adg::LocalMemoryPortVariant::SeparateElementVector,
        loom::adg::LocalMemoryPortVariant::SharedElementVector}) {
    loom::ResolvedConfig resolved = loom::defaultResolvedConfig();
    resolved.hardwareTarget.parameters.localMemoryPortVariant = variant;
    auto projected =
        take(loom::dse::projectResolvedFabricTemplateConfigView(resolved));
    auto adopted = take(loom::dse::adoptResolvedFabricTemplateConfigView(
        loom::dse::resolvedFabricTemplateConfigSchemaBytes(),
        projected.canonicalViewBytes(), projected.digest()));
    require(adopted.scale().localMemoryPortVariant == variant,
            "template config changed the local-memory port variant");
  }

  loom::ResolvedConfig invalid = loom::defaultResolvedConfig();
  invalid.hardwareTarget.parameters.localMemoryPortVariant =
      static_cast<loom::adg::LocalMemoryPortVariant>(255);
  requireError(
      loom::dse::projectResolvedFabricTemplateConfigView(invalid).takeError(),
      "template base scale is invalid");
}

void strictConfigAdmission() {
  requireError(
      loom::dse::resolveSpatialTopologyRewriteConfig({}, 1).takeError(),
      "nonempty");
  std::vector<loom::dse::SpatialTopologyDecisionDomain> duplicate = {
      loom::dse::RemoveOccurrenceDomain{
          {take(loom::fabric::FabricModulePhysicalOwnerRef::create(
              loom::fabric::FabricFifoOccurrenceRef(0)))}},
      loom::dse::RemoveOccurrenceDomain{
          {take(loom::fabric::FabricModulePhysicalOwnerRef::create(
              loom::fabric::FabricFifoOccurrenceRef(1)))}},
  };
  requireError(
      loom::dse::resolveSpatialTopologyRewriteConfig(duplicate, 1).takeError(),
      "domain key");
  std::vector<loom::dse::SpatialMicroarchitectureDecisionDomain> domains = {
      loom::dse::ResizeFifoDomain{loom::fabric::FabricFifoOccurrenceRef(0),
                                  {1}}};
  requireError(
      loom::dse::resolveSpatialMicroarchitectureRewriteConfig(domains, 0)
          .takeError(),
      "positive");
  domains.front() = loom::dse::ResizeFifoDomain{
      loom::fabric::FabricFifoOccurrenceRef(0), {0}};
  requireError(
      loom::dse::resolveSpatialMicroarchitectureRewriteConfig(domains, 1)
          .takeError(),
      "positive");
  domains.front() = loom::dse::ChangeFuInventoryDomain{
      loom::fabric::FabricPeOccurrenceRef(0), {{}}};
  requireError(
      loom::dse::resolveSpatialMicroarchitectureRewriteConfig(domains, 1)
          .takeError(),
      "nonempty");
  domains.front() = loom::dse::ResizeFifoDomain{
      loom::fabric::FabricFifoOccurrenceRef(0), {1, 1}};
  requireError(
      loom::dse::resolveSpatialMicroarchitectureRewriteConfig(domains, 1)
          .takeError(),
      "duplicate");
  domains.front() = loom::dse::ResizeTemporalOperandBufferDomain{
      loom::fabric::FabricPeOccurrenceRef(0), {0}};
  requireError(
      loom::dse::resolveSpatialMicroarchitectureRewriteConfig(domains, 1)
          .takeError(),
      "positive");

  std::vector<loom::dse::SpatialTopologyDecisionDomain> emptyConnections = {
      loom::dse::AdjustParallelConnectionCountDomain{{{}}}};
  requireError(
      loom::dse::resolveSpatialTopologyRewriteConfig(emptyConnections, 1)
          .takeError(),
      "nonempty");
}

void computeContextFeedbackRoundTrip(
    Fixture &fixture, const loom::fabric::FinalizedFabricRoot &module) {
  require(!module.view().fuOccurrences().empty(),
          "builtin Module has no FU occurrence for Hall feedback");
  const auto definition =
      module.view().fuTemplateOf(module.view().fuOccurrences().front());
  require(definition.has_value(), "FU occurrence has no template");
  const auto capabilities = module.view().fuCapabilityTemplates(*definition);
  require(!capabilities.empty(), "FU template has no capability");
  const loom::fabric::FabricFuCapabilityTemplateRef capability{*definition, 0};
  auto placements =
      take(loom::mapping::deriveSpatialComputeContextPlacementDomain(
          capability, module.view()));
  std::vector<loom::fabric::InstructionContextRef> contexts;
  for (const auto &placement : placements)
    contexts.insert(contexts.end(), placement.contexts.begin(),
                    placement.contexts.end());
  require(!contexts.empty(), "FU capability has no compatible context");

  const std::uint64_t demandCount = contexts.size() + 1;
  const std::vector<loom::mapping::TechMappingComputeContextHallDemandGroup>
      groups = {{capability, demandCount, contexts}};
  auto feedback = take(loom::mapping::TechMappingComputeContextHallDeficit::get(
      demandCount, contexts.size(), groups));
  std::vector<std::uint8_t> bytes =
      loom::mapping::encodeTechMappingComputeContextHallFeedback(feedback);
  auto adopted = take(loom::mapping::adoptTechMappingComputeContextHallFeedback(
      bytes, module.view()));
  require(adopted.deficit() == 1 && adopted.hallDemandCount() == demandCount &&
              adopted.hallContextValueCount() == contexts.size(),
          "Hall feedback did not rebuild its Fabric context relation");

  auto domains = take(loom::dse::projectTechMappingComputeContextGrowthDomains(
      adopted, module.view()));
  require(!domains.empty(),
          "Hall feedback produced no Temporal PE growth action");
  for (const auto &domain : domains) {
    const auto *resize =
        std::get_if<loom::dse::ResizeInstructionStoreDomain>(&domain);
    require(resize && !resize->capacities.empty(),
            "Hall feedback produced a non-context growth action");
    const std::uint64_t current =
        module.view().peResidentContextCount(resize->target);
    require(resize->capacities.front() == current + 1 &&
                resize->capacities.back() == current + adopted.deficit(),
            "Hall feedback did not order minimal-to-complete growth");
  }
  auto joint = take(loom::dse::projectTechMappingComputeContextJointGrowthPlan(
      adopted, module.view()));
  require(joint.addedContextCount == adopted.deficit() &&
              !joint.decisions.empty(),
          "joint Hall growth did not close the exact minimum deficit");
  std::uint64_t jointGrowth = 0;
  for (const loom::dse::ResizeInstructionStore &decision : joint.decisions) {
    const std::uint64_t current =
        module.view().peResidentContextCount(decision.target);
    require(decision.instructionCapacity > current,
            "joint Hall growth contains a non-growth decision");
    jointGrowth += decision.instructionCapacity - current;
  }
  require(jointGrowth == adopted.deficit(),
          "joint Hall growth overprovisioned the observed relation");
  auto jointConfig =
      take(loom::dse::resolveSpatialMicroarchitectureRewriteConfig(
          {loom::dse::ResizeInstructionStoresDomain{joint.decisions}}, 1));
  auto jointInputs =
      take(loom::dse::bindSpatialMicroarchitectureCandidateGeneratorInputs(
          {module.reference()}));
  auto jointBinding =
      take(loom::dse::resolveSpatialMicroarchitectureCandidateGeneratorBinding(
          jointConfig));
  auto jointResult = take(loom::dse::invokeCandidateGenerator(
      jointInputs, jointBinding, fixture.store, fixture.blobs));
  const auto &jointCompleted = completed(jointResult);
  require(jointCompleted.outputBindings.front().artifacts.size() == 1 &&
              jointCompleted.lineageEdges.size() == 1,
          "joint Hall growth did not publish one typed Module child");
  auto jointLineage = take(loom::dse::adoptSpatialMicroarchitectureDecision(
      jointCompleted.lineageEdges.front().ownerPayload));
  const auto *jointDecision =
      std::get_if<loom::dse::ResizeInstructionStores>(&jointLineage.decision);
  require(jointDecision &&
              jointDecision->stores.size() == joint.decisions.size(),
          "joint Hall growth lineage lost its atomic resize set");
  auto jointModule = take(loom::fabric::importEntireFabricRoot(
      jointCompleted.outputBindings.front().artifacts.front(), fixture.store));
  std::uint64_t parentContexts = 0;
  std::uint64_t childContexts = 0;
  for (const auto pe : module.view().peOccurrences())
    parentContexts += module.view().peResidentContextCount(pe);
  for (const auto pe : jointModule.view().peOccurrences())
    childContexts += jointModule.view().peResidentContextCount(pe);
  require(childContexts == parentContexts + adopted.deficit(),
          "joint Hall growth child changed the requested total capacity");
  auto growthConfig =
      take(loom::dse::resolveSpatialMicroarchitectureRewriteConfig(
          domains, domains.size()));
  auto growthInputs =
      take(loom::dse::bindSpatialMicroarchitectureCandidateGeneratorInputs(
          {module.reference()}));
  auto growthBinding =
      take(loom::dse::resolveSpatialMicroarchitectureCandidateGeneratorBinding(
          growthConfig));
  auto growth = take(loom::dse::invokeCandidateGenerator(
      growthInputs, growthBinding, fixture.store, fixture.blobs));
  require(!completed(growth).outputBindings.front().artifacts.empty(),
          "Hall feedback growth actions produced no Module child");
  require(completed(growth).lineageEdges.size() ==
              completed(growth).outputBindings.front().artifacts.size(),
          "Hall feedback growth child lost typed decision lineage");

  bytes.push_back(0);
  requireError(loom::mapping::adoptTechMappingComputeContextHallFeedback(
                   bytes, module.view())
                   .takeError(),
               "trailing bytes");
}

void topologyRewrite(Fixture &fixture,
                     const loom::fabric::FinalizedFabricRoot &module) {
  auto moduleTemplate = module.view().moduleRootTemplate();
  require(moduleTemplate.has_value(), "builtin Module has no root template");
  const std::uint64_t inputCount = module.view().moduleBoundaryEndpointCount(
      *moduleTemplate, loom::fabric::FabricPortDirection::Input);
  std::vector<loom::fabric::FabricTransportEndpointRef> outputSources;
  for (const auto &attachment :
       module.view().moduleBoundaryTransportAttachments())
    if (attachment.boundary.direction ==
        loom::fabric::FabricPortDirection::Output)
      outputSources.push_back(attachment.endpoint);
  require(outputSources.size() > 1,
          "builtin Module has no shrinkable transport output boundary");
  outputSources.pop_back();
  std::vector<loom::dse::SpatialTopologyDecisionDomain> domains = {
      loom::dse::ChangeBoundaryInventoryDomain{
          {{inputCount, std::move(outputSources)}}}};
  auto config =
      take(loom::dse::resolveSpatialTopologyRewriteConfig(domains, 1));
  auto inputs = take(loom::dse::bindSpatialTopologyCandidateGeneratorInputs(
      {module.reference()}));
  auto binding =
      take(loom::dse::resolveSpatialTopologyCandidateGeneratorBinding(config));
  const std::vector<std::uint8_t> parentBytes(
      module.canonicalBytes().bytes().begin(),
      module.canonicalBytes().bytes().end());
  auto result = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, fixture.store, fixture.blobs));
  require(completed(result).outputBindings.front().artifacts.size() == 1,
          "topology decision did not produce a child");
  require(completed(result).lineageEdges.size() == 1,
          "topology child lost its decision lineage");
  require(module.canonicalBytes().bytes() ==
              llvm::ArrayRef<std::uint8_t>(parentBytes),
          "topology generator mutated the finalized parent");
}

void occurrenceInventoryRewrite(
    Fixture &fixture, const loom::fabric::FinalizedFabricRoot &module) {
  require(!module.view().fuOccurrences().empty(),
          "builtin Module has no FU occurrence prototype");
  std::vector<loom::fabric::FabricModulePhysicalOwnerRef> prototypes = {
      take(loom::fabric::FabricModulePhysicalOwnerRef::create(
          module.view().fuOccurrences().front()))};
  std::vector<loom::dse::SpatialTopologyDecisionDomain> addDomains = {
      loom::dse::AddOccurrenceDomain{prototypes}};
  auto addConfig = take(loom::dse::resolveSpatialTopologyRewriteConfig(
      addDomains, prototypes.size()));
  auto addInputs = take(loom::dse::bindSpatialTopologyCandidateGeneratorInputs(
      {module.reference()}));
  auto addBinding = take(
      loom::dse::resolveSpatialTopologyCandidateGeneratorBinding(addConfig));
  auto addResult = take(loom::dse::invokeCandidateGenerator(
      addInputs, addBinding, fixture.store, fixture.blobs));
  const auto &addedOutputs =
      completed(addResult).outputBindings.front().artifacts;
  require(!addedOutputs.empty(),
          "AddOccurrence produced no finalized Module child");
  auto added = take(loom::fabric::importEntireFabricRoot(addedOutputs.front(),
                                                         fixture.store));

  const auto prototypeTemplate =
      module.view().fuTemplateOf(module.view().fuOccurrences().front());
  require(prototypeTemplate.has_value(), "FU prototype has no template");
  std::vector<loom::fabric::FabricModulePhysicalOwnerRef> targets;
  for (auto target : added.view().fuOccurrences())
    if (added.view().fuTemplateOf(target) == prototypeTemplate)
      targets.push_back(
          take(loom::fabric::FabricModulePhysicalOwnerRef::create(target)));
  require(!targets.empty(),
          "added Module has no occurrence of the prototype FU template");
  std::vector<loom::dse::SpatialTopologyDecisionDomain> removeDomains = {
      loom::dse::RemoveOccurrenceDomain{targets}};
  auto removeConfig = take(loom::dse::resolveSpatialTopologyRewriteConfig(
      removeDomains, targets.size()));
  auto removeInputs =
      take(loom::dse::bindSpatialTopologyCandidateGeneratorInputs(
          {added.reference()}));
  auto removeBinding = take(
      loom::dse::resolveSpatialTopologyCandidateGeneratorBinding(removeConfig));
  auto removeResult = take(loom::dse::invokeCandidateGenerator(
      removeInputs, removeBinding, fixture.store, fixture.blobs));
  const auto &removedOutputs =
      completed(removeResult).outputBindings.front().artifacts;
  require(llvm::is_contained(removedOutputs, module.reference()),
          "RemoveOccurrence could not recover the pre-add Module identity");
}

loom::fabric::FinalizedFabricRoot
microarchitectureRewrite(Fixture &fixture,
                         const loom::fabric::FinalizedFabricRoot &module) {
  require(!module.view().fifoOccurrences().empty(),
          "builtin Module has no FIFO occurrence");
  std::vector<loom::dse::SpatialMicroarchitectureDecisionDomain> domains = {
      loom::dse::ResizeFifoDomain{module.view().fifoOccurrences().front(),
                                  {1, 2, 4, 8}}};
  auto config =
      take(loom::dse::resolveSpatialMicroarchitectureRewriteConfig(domains, 4));
  auto inputs =
      take(loom::dse::bindSpatialMicroarchitectureCandidateGeneratorInputs(
          {module.reference()}));
  auto binding =
      take(loom::dse::resolveSpatialMicroarchitectureCandidateGeneratorBinding(
          config));
  auto result = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, fixture.store, fixture.blobs));
  require(!completed(result).outputBindings.front().artifacts.empty(),
          "microarchitecture decisions produced no valid child");
  require(completed(result).lineageEdges.size() ==
              completed(result).outputBindings.front().artifacts.size(),
          "microarchitecture children lost decision lineage");
  return take(loom::fabric::importEntireFabricRoot(
      completed(result).outputBindings.front().artifacts.front(),
      fixture.store));
}

void temporalOperandBufferRewrite(
    Fixture &fixture, const loom::fabric::FinalizedFabricRoot &module) {
  using BufferSignature =
      std::pair<::fabric::OperandBufferMode, std::uint32_t>;
  const auto bufferSignatures = [](const loom::fabric::FabricArtifactView &view) {
    std::map<BufferSignature, std::uint64_t> result;
    for (const auto pe : view.peOccurrences()) {
      const auto mode = view.peOperandBufferMode(pe);
      if (mode)
        ++result[{*mode, view.peOperandBufferSize(pe)}];
    }
    return result;
  };
  std::optional<loom::fabric::FabricPeOccurrenceRef> target;
  for (const auto pe : module.view().peOccurrences())
    if (module.view().peSchedule(pe) == ::fabric::Schedule::Temporal) {
      target = pe;
      break;
    }
  require(target.has_value(), "builtin Module has no Temporal PE");
  const auto parentMode = module.view().peOperandBufferMode(*target);
  const std::uint32_t parentEntries =
      module.view().peOperandBufferSize(*target);
  require(parentMode.has_value() && parentEntries != 0,
          "Temporal PE has no operand-buffer declaration");
  const ::fabric::OperandBufferMode childMode =
      *parentMode == ::fabric::OperandBufferMode::PerInputPort
          ? ::fabric::OperandBufferMode::AllFuShare
          : ::fabric::OperandBufferMode::PerInputPort;
  require(parentEntries != std::numeric_limits<std::uint32_t>::max(),
          "Temporal operand-buffer fixture cannot grow");
  const std::uint32_t childEntries = parentEntries + 1;
  const auto parentSignatures = bufferSignatures(module.view());
  const auto replaceSignature = [](auto &signatures, BufferSignature oldValue,
                                   BufferSignature newValue) {
    auto old = signatures.find(oldValue);
    if (old == signatures.end() || old->second == 0)
      fail("parent operand-buffer signature is absent");
    if (--old->second == 0)
      signatures.erase(old);
    ++signatures[newValue];
  };

  std::vector<loom::dse::SpatialMicroarchitectureDecisionDomain> domains = {
      loom::dse::ChangeTemporalOperandBufferModeDomain{*target, {childMode}},
      loom::dse::ResizeTemporalOperandBufferDomain{*target, {childEntries}}};
  auto config =
      take(loom::dse::resolveSpatialMicroarchitectureRewriteConfig(domains, 2));
  auto inputs =
      take(loom::dse::bindSpatialMicroarchitectureCandidateGeneratorInputs(
          {module.reference()}));
  auto binding =
      take(loom::dse::resolveSpatialMicroarchitectureCandidateGeneratorBinding(
          config));
  auto generated = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, fixture.store, fixture.blobs));
  const auto &result = completed(generated);
  require(result.outputBindings.front().artifacts.size() == 2 &&
              result.lineageEdges.size() == 2,
          "operand-buffer decisions did not publish two exact children");

  bool sawMode = false;
  bool sawResize = false;
  for (const loom::dse::CandidateGeneratorLineageEdge &lineage :
       result.lineageEdges) {
    require(lineage.parents ==
                std::vector<loom::ArtifactRootReference>{module.reference()},
            "operand-buffer child lost its exact parent");
    auto decision = take(loom::dse::adoptSpatialMicroarchitectureDecision(
        lineage.ownerPayload));
    auto child = take(loom::fabric::importEntireFabricRoot(lineage.output,
                                                           fixture.store));
    auto impact = loom::dse::projectHardwareImpact(decision, lineage.output);
    require(impact.family ==
                    loom::dse::HardwareMutationFamily::TemporalOperandBuffer &&
                impact.locality ==
                    loom::dse::HardwareMutationLocality::LocalCone &&
                impact.spatial.placementRoots.size() == 1,
            "operand-buffer child lost its typed local impact");
    if (const auto *mode = std::get_if<
            loom::dse::ChangeTemporalOperandBufferMode>(&decision.decision)) {
      auto expected = parentSignatures;
      replaceSignature(expected, {*parentMode, parentEntries},
                       {childMode, parentEntries});
      require(mode->target == *target && mode->mode == childMode &&
                  bufferSignatures(child.view()) == expected &&
                  impact.spatial.kind ==
                      loom::dse::HardwareMappingImpactKind::Reopen,
              "operand-buffer mode child changed another hardware fact");
      sawMode = true;
      continue;
    }
    const auto *resize =
        std::get_if<loom::dse::ResizeTemporalOperandBuffer>(&decision.decision);
    auto expected = parentSignatures;
    replaceSignature(expected, {*parentMode, parentEntries},
                     {*parentMode, childEntries});
    require(resize && resize->target == *target &&
                resize->entriesPerAllocationUnit == childEntries &&
                bufferSignatures(child.view()) == expected &&
                impact.spatial.kind ==
                    loom::dse::HardwareMappingImpactKind::Rebase,
            "operand-buffer resize child changed another hardware fact");
    sawResize = true;
  }
  require(sawMode && sawResize,
          "operand-buffer decision lineage is incomplete");
  require(bufferSignatures(module.view()) == parentSignatures,
          "operand-buffer child mutated its immutable parent");
}

bool hasSpecialMath(const loom::fabric::FabricArtifactView &view,
                    loom::fabric::FabricFuOccurrenceRef fu) {
  const auto definition = view.fuTemplateOf(fu);
  if (!definition)
    fail("FU occurrence has no template");
  return llvm::any_of(
      view.resolvedFabricOpCapabilities(*definition),
      [](const auto &operation) {
        return ::fabric::implementationFamily(operation.implementationFamily)
                   .typedAdmissionProvider ==
               ::fabric::TypedAdmissionProviderId::ScalarSpecialMathAdmission;
      });
}

std::uint64_t
specialMathContextCount(const loom::fabric::FabricArtifactView &view) {
  std::uint64_t result = 0;
  for (const auto pe : view.peOccurrences()) {
    bool capable = false;
    for (const auto fu : view.fuOccurrences())
      capable |= view.parentPeOf(fu) == pe && hasSpecialMath(view, fu);
    if (capable)
      result += view.peResidentContextCount(pe);
  }
  return result;
}

void redistributeFuCapability(Fixture &fixture,
                              const loom::fabric::FinalizedFabricRoot &module) {
  std::optional<loom::fabric::FabricPeOccurrenceRef> target;
  std::optional<loom::fabric::FabricFuOccurrenceRef> prototype;
  for (const auto pe : module.view().peOccurrences()) {
    if (module.view().peSchedule(pe) != ::fabric::Schedule::Temporal)
      continue;
    bool capable = false;
    for (const auto fu : module.view().fuOccurrences()) {
      if (module.view().parentPeOf(fu) != pe ||
          !hasSpecialMath(module.view(), fu))
        continue;
      capable = true;
      prototype = fu;
    }
    if (!capable && !target)
      target = pe;
  }
  require(target && prototype,
          "builtin Module lacks distinct Temporal capability sites");

  std::vector<loom::fabric::FabricFuOccurrenceRef> inventory;
  for (const auto fu : module.view().fuOccurrences())
    if (module.view().parentPeOf(fu) == target)
      inventory.push_back(fu);
  inventory.push_back(*prototype);
  std::vector<loom::dse::SpatialMicroarchitectureDecisionDomain> domains = {
      loom::dse::ChangeFuInventoryDomain{*target, {inventory}}};
  auto config =
      take(loom::dse::resolveSpatialMicroarchitectureRewriteConfig(domains, 1));
  auto inputs =
      take(loom::dse::bindSpatialMicroarchitectureCandidateGeneratorInputs(
          {module.reference()}));
  auto binding =
      take(loom::dse::resolveSpatialMicroarchitectureCandidateGeneratorBinding(
          config));
  auto result = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, fixture.store, fixture.blobs));
  const auto &outputs = completed(result).outputBindings.front().artifacts;
  require(outputs.size() == 1,
          "cross-PE FU inventory decision did not produce one child");
  auto child = take(
      loom::fabric::importEntireFabricRoot(outputs.front(), fixture.store));
  require(child.view().fuOccurrences().size() ==
              module.view().fuOccurrences().size() + 1,
          "cross-PE FU inventory decision did not retain and add capabilities");
  require(specialMathContextCount(child.view()) >
              specialMathContextCount(module.view()),
          "cross-PE FU inventory decision did not redistribute capability");
}

void hardwareImpactMatrix(
    const loom::fabric::FinalizedFabricRoot &system,
    const loom::fabric::FinalizedFabricRoot &module) {
  const auto requireModuleImpact =
      [&](loom::dse::SpatialMicroarchitectureDecision decision,
          loom::dse::HardwareMutationFamily family,
          loom::dse::HardwareMutationLocality locality,
          loom::dse::HardwareMappingImpactKind techKind,
          loom::dse::HardwareMappingImpactKind spatialKind) {
        const loom::dse::SpatialMicroarchitectureCandidateDecision candidate{
            module.reference(), std::move(decision), {}};
        const auto impact = loom::dse::projectHardwareImpact(candidate);
        require(impact.family == family && impact.locality == locality &&
                    impact.tech.kind == techKind &&
                    impact.spatial.kind == spatialKind &&
                    !impact.spatial.placementRoots.empty(),
                "hardware mutation matrix lost a typed Module impact");
      };

  if (!module.view().peOccurrences().empty()) {
    const auto pe = module.view().peOccurrences().front();
    requireModuleImpact(
        loom::dse::ResizeInstructionStore{pe, 2},
        loom::dse::HardwareMutationFamily::InstructionCapacity,
        loom::dse::HardwareMutationLocality::LocalCone,
        loom::dse::HardwareMappingImpactKind::Rebase,
        loom::dse::HardwareMappingImpactKind::Rebase);
    requireModuleImpact(
        loom::dse::ResizeTemporalOperandBuffer{
            pe, module.view().peOperandBufferSize(pe) + 1},
        loom::dse::HardwareMutationFamily::TemporalOperandBuffer,
        loom::dse::HardwareMutationLocality::LocalCone,
        loom::dse::HardwareMappingImpactKind::Rebase,
        loom::dse::HardwareMappingImpactKind::Rebase);
    requireModuleImpact(
        loom::dse::ChangeTemporalOperandBufferMode{
            pe, ::fabric::OperandBufferMode::PerInstruction},
        loom::dse::HardwareMutationFamily::TemporalOperandBuffer,
        loom::dse::HardwareMutationLocality::LocalCone,
        loom::dse::HardwareMappingImpactKind::Rebase,
        loom::dse::HardwareMappingImpactKind::Reopen);
  }
  if (!module.view().memoryOccurrences().empty()) {
    const auto memory = module.view().memoryOccurrences().front();
    requireModuleImpact(
        loom::dse::ResizeMemory{memory, 4096},
        loom::dse::HardwareMutationFamily::SpatialMemory,
        loom::dse::HardwareMutationLocality::LocalCone,
        loom::dse::HardwareMappingImpactKind::Rebase,
        loom::dse::HardwareMappingImpactKind::Rebase);
    requireModuleImpact(
        loom::dse::ChangeMemoryOperationTable{memory, memory},
        loom::dse::HardwareMutationFamily::SpatialMemory,
        loom::dse::HardwareMutationLocality::GlobalReopen,
        loom::dse::HardwareMappingImpactKind::Reopen,
        loom::dse::HardwareMappingImpactKind::Reopen);
  }
  if (!module.view().fifoOccurrences().empty()) {
    const auto fifo = module.view().fifoOccurrences().front();
    requireModuleImpact(
        loom::dse::ResizeFifo{fifo, 2},
        loom::dse::HardwareMutationFamily::SpatialFifo,
        loom::dse::HardwareMutationLocality::LocalCone,
        loom::dse::HardwareMappingImpactKind::Rebase,
        loom::dse::HardwareMappingImpactKind::Rebase);
    requireModuleImpact(
        loom::dse::ChangeFifoBypassCapability{fifo, true},
        loom::dse::HardwareMutationFamily::SpatialFifo,
        loom::dse::HardwareMutationLocality::LocalCone,
        loom::dse::HardwareMappingImpactKind::Rebase,
        loom::dse::HardwareMappingImpactKind::Reopen);
  }
  if (!module.view().fuOccurrences().empty())
    requireModuleImpact(
        loom::dse::ChangeFuCapability{module.view().fuOccurrences().front(),
                                      module.view().fuOccurrences().front()},
        loom::dse::HardwareMutationFamily::FuCapability,
        loom::dse::HardwareMutationLocality::GlobalReopen,
        loom::dse::HardwareMappingImpactKind::Reopen,
        loom::dse::HardwareMappingImpactKind::Reopen);
  if (!module.view().switchOccurrences().empty())
    requireModuleImpact(
        loom::dse::ChangeSwitchModeOrScheduleCapacity{
            module.view().switchOccurrences().front(),
            module.view().switchOccurrences().front()},
        loom::dse::HardwareMutationFamily::SpatialSwitch,
        loom::dse::HardwareMutationLocality::GlobalReopen,
        loom::dse::HardwareMappingImpactKind::Reopen,
        loom::dse::HardwareMappingImpactKind::Reopen);

  auto systemView = take(loom::fabric::requireSystemRoot(system.view()));
  require(!system.view().accCoreOccurrences().empty() &&
              !systemView.transportResources().empty(),
          "hardware mutation matrix lacks System resources");
  const auto core = system.view().accCoreOccurrences().front();
  const auto transport = systemView.transportResources().front();
  auto addImpact = loom::dse::projectHardwareImpact(
      loom::dse::SystemCompositionCandidateDecision{
          system.reference(),
          loom::dse::AddAccCore{core, module.reference()},
          {}, {}});
  require(addImpact.family == loom::dse::HardwareMutationFamily::SystemAccCore &&
              addImpact.system.kind ==
                  loom::dse::HardwareMappingImpactKind::Rebase,
          "AddAccCore lost its System rebase impact");
  auto contextImpact = loom::dse::projectHardwareImpact(
      loom::dse::SystemCompositionCandidateDecision{
          system.reference(),
          loom::dse::SelectInstructionCoreRealization{
              loom::fabric::InstructionCoreContextRef{core},
              loom::fabric::InstructionCoreContextRef{core}},
          {}, {}});
  require(contextImpact.family ==
                  loom::dse::HardwareMutationFamily::SystemInstructionContext &&
              contextImpact.locality ==
                  loom::dse::HardwareMutationLocality::LocalCone &&
              contextImpact.system.kind ==
                  loom::dse::HardwareMappingImpactKind::Reopen,
          "InstructionCore context lost its System reopen impact");
  auto transportImpact = loom::dse::projectHardwareImpact(
      loom::dse::SystemCompositionCandidateDecision{
          system.reference(), loom::dse::ChangeTransportResource{transport,
                                                                  transport},
          {}, {}});
  require(transportImpact.family ==
                  loom::dse::HardwareMutationFamily::SystemTransport &&
              transportImpact.system.kind ==
                  loom::dse::HardwareMappingImpactKind::Reopen,
          "System transport lost its typed reopen impact");
}

void systemCompositionRewrite(Fixture &fixture,
                              const loom::fabric::FinalizedFabricRoot &system,
                              const loom::fabric::FinalizedFabricRoot &module) {
  require(!system.view().accCoreOccurrences().empty(),
          "builtin System has no AccCore prototype");
  std::vector<loom::dse::SystemCompositionDecisionDomain> domains = {
      loom::dse::AddAccCoreDomain{system.view().accCoreOccurrences().front(),
                                  {module.reference()}}};
  auto config =
      take(loom::dse::resolveSystemCompositionRewriteConfig(domains, 1));
  auto inputs = take(loom::dse::bindSystemCompositionCandidateGeneratorInputs(
      {system.reference()}, {module.reference()}));
  auto binding = take(
      loom::dse::resolveSystemCompositionCandidateGeneratorBinding(config));
  auto result = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, fixture.store, fixture.blobs));
  const auto &outputs = completed(result).outputBindings.front().artifacts;
  require(outputs.size() == 1,
          "AddAccCore did not produce one finalized System child");
  auto child = take(
      loom::fabric::importEntireFabricRoot(outputs.front(), fixture.store));
  require(child.view().rootKind() == loom::fabric::FabricRootKind::System,
          "System composition child has the wrong root kind");
  require(
      child.view().accCoreOccurrences().size() ==
          system.view().accCoreOccurrences().size() + 1,
      "System composition child did not retain distinct AccCore occurrences");

  std::vector<loom::dse::SystemCompositionDecisionDomain> removeDomains = {
      loom::dse::RemoveAccCoreDomain{
          {child.view().accCoreOccurrences().back()}}};
  auto removeConfig =
      take(loom::dse::resolveSystemCompositionRewriteConfig(removeDomains, 1));
  auto removeInputs =
      take(loom::dse::bindSystemCompositionCandidateGeneratorInputs(
          {child.reference()}, {module.reference()}));
  auto removeBinding =
      take(loom::dse::resolveSystemCompositionCandidateGeneratorBinding(
          removeConfig));
  auto removeResult = take(loom::dse::invokeCandidateGenerator(
      removeInputs, removeBinding, fixture.store, fixture.blobs));
  const auto &removeOutputs =
      completed(removeResult).outputBindings.front().artifacts;
  require(removeOutputs.size() == 1,
          "RemoveAccCore did not produce one finalized System child");
  auto removed = take(loom::fabric::importEntireFabricRoot(
      removeOutputs.front(), fixture.store));
  require(removed.view().accCoreOccurrences().size() ==
              system.view().accCoreOccurrences().size(),
          "RemoveAccCore did not remove one occurrence-qualified core");

  std::vector<loom::fabric::AccCoreOccurrenceRef> connectedTargets(
      system.view().accCoreOccurrences().begin(),
      system.view().accCoreOccurrences().end());
  std::vector<loom::dse::SystemCompositionDecisionDomain>
      connectedRemoveDomains = {
          loom::dse::RemoveAccCoreDomain{connectedTargets}};
  auto connectedRemoveConfig =
      take(loom::dse::resolveSystemCompositionRewriteConfig(
          connectedRemoveDomains, connectedTargets.size()));
  auto connectedRemoveInputs =
      take(loom::dse::bindSystemCompositionCandidateGeneratorInputs(
          {system.reference()}, {module.reference()}));
  auto connectedRemoveBinding =
      take(loom::dse::resolveSystemCompositionCandidateGeneratorBinding(
          connectedRemoveConfig));
  auto connectedRemoveResult = take(loom::dse::invokeCandidateGenerator(
      connectedRemoveInputs, connectedRemoveBinding, fixture.store,
      fixture.blobs));
  const auto &connectedRemoveOutputs =
      completed(connectedRemoveResult).outputBindings.front().artifacts;
  require(!connectedRemoveOutputs.empty(),
          "connected AccCore removal produced no finalized System child");
  for (const auto &output : connectedRemoveOutputs) {
    auto connectedChild =
        take(loom::fabric::importEntireFabricRoot(output, fixture.store));
    require(connectedChild.view().accCoreOccurrences().size() + 1 ==
                system.view().accCoreOccurrences().size(),
            "connected AccCore removal retained the target occurrence");
  }
}

void spatialAttachmentRewrite(
    Fixture &fixture, const loom::fabric::FinalizedFabricRoot &system,
    const loom::fabric::FinalizedFabricRoot &replacementModule) {
  const auto target = system.view().accCoreOccurrences().front();
  std::vector<loom::dse::SystemCompositionDecisionDomain> domains = {
      loom::dse::ReplaceSpatialAttachmentDomain{
          target, {replacementModule.reference()}}};
  auto config =
      take(loom::dse::resolveSystemCompositionRewriteConfig(domains, 1));
  auto inputs = take(loom::dse::bindSystemCompositionCandidateGeneratorInputs(
      {system.reference()}, {replacementModule.reference()}));
  auto binding = take(
      loom::dse::resolveSystemCompositionCandidateGeneratorBinding(config));
  auto result = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, fixture.store, fixture.blobs));
  const auto &outputs = completed(result).outputBindings.front().artifacts;
  require(outputs.size() == 1,
          "ReplaceSpatialAttachment did not produce one finalized child");
  auto child = take(
      loom::fabric::importEntireFabricRoot(outputs.front(), fixture.store));
  auto systemView = take(loom::fabric::requireSystemRoot(child.view()));
  std::uint64_t replacementCount = 0;
  for (auto core : child.view().accCoreOccurrences()) {
    auto selected = systemView.spatialCoreTarget(core);
    require(selected.has_value(),
            "replaced System has an AccCore without a SpatialCore target");
    require(selected->dependencyOrdinal < child.directDependencies().size(),
            "replaced AccCore has an invalid dependency ordinal");
    replacementCount +=
        child.directDependencies()[selected->dependencyOrdinal].root ==
        replacementModule.reference();
  }
  require(replacementCount == 1,
          "ReplaceSpatialAttachment did not select one replacement Module");
}

bool sameTransportShape(const loom::fabric::FabricArtifactView &view,
                        loom::fabric::SystemTransportResourceRef left,
                        loom::fabric::SystemTransportResourceRef right) {
  const auto leftOwner =
      loom::fabric::FabricTransportEndpointOwnerRef::of(left);
  const auto rightOwner =
      loom::fabric::FabricTransportEndpointOwnerRef::of(right);
  const std::uint64_t count = view.transportEndpointCount(leftOwner);
  if (count != view.transportEndpointCount(rightOwner))
    return false;
  for (std::uint64_t ordinal = 0; ordinal < count; ++ordinal) {
    const loom::fabric::FabricTransportEndpointRef leftEndpoint{leftOwner,
                                                                ordinal};
    const loom::fabric::FabricTransportEndpointRef rightEndpoint{rightOwner,
                                                                 ordinal};
    if (view.transportEndpointDirection(leftEndpoint) !=
            view.transportEndpointDirection(rightEndpoint) ||
        view.transportEndpointType(leftEndpoint) !=
            view.transportEndpointType(rightEndpoint))
      return false;
  }
  return true;
}

void transportResourceRewrite(Fixture &fixture,
                              const loom::fabric::FinalizedFabricRoot &system,
                              const loom::fabric::FinalizedFabricRoot &module) {
  auto systemView = take(loom::fabric::requireSystemRoot(system.view()));
  require(systemView.transportResources().size() > 1,
          "builtin System has no transport-resource prototypes");
  const auto target = systemView.transportResources().front();
  std::vector<loom::fabric::SystemTransportResourceRef> prototypes;
  for (auto prototype : systemView.transportResources().drop_front())
    if (sameTransportShape(system.view(), target, prototype))
      prototypes.push_back(prototype);
  require(!prototypes.empty(),
          "builtin System has no shape-compatible transport prototype");

  std::vector<loom::dse::SystemCompositionDecisionDomain> domains = {
      loom::dse::ChangeTransportResourceDomain{target, prototypes}};
  auto config = take(loom::dse::resolveSystemCompositionRewriteConfig(
      domains, prototypes.size()));
  auto inputs = take(loom::dse::bindSystemCompositionCandidateGeneratorInputs(
      {system.reference()}, {module.reference()}));
  auto binding = take(
      loom::dse::resolveSystemCompositionCandidateGeneratorBinding(config));
  auto result = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, fixture.store, fixture.blobs));
  require(completed(result).outputBindings.front().artifacts.empty(),
          "equivalent transport prototypes changed canonical identity");
  require(completed(result).lineageEdges.empty(),
          "transport no-op incorrectly produced decision lineage");
  for (const auto &output :
       completed(result).outputBindings.front().artifacts) {
    auto child =
        take(loom::fabric::importEntireFabricRoot(output, fixture.store));
    require(child.view().rootKind() == loom::fabric::FabricRootKind::System,
            "transport-resource rewrite produced a non-System child");
  }
}

void systemRelationalNoOps(Fixture &fixture,
                           const loom::fabric::FinalizedFabricRoot &system,
                           const loom::fabric::FinalizedFabricRoot &module) {
  auto systemView = take(loom::fabric::requireSystemRoot(system.view()));
  require(!system.view().accCoreOccurrences().empty(),
          "builtin System has no InstructionCore context");
  require(!system.view().pointConnections().empty(),
          "builtin System has no transport connection");

  const loom::fabric::FabricSpatialAttachmentRecordView *memoryAttachment =
      nullptr;
  for (const auto &attachment : systemView.spatialAttachments())
    if (attachment.spatialEndpoint.memory() && attachment.serviceEndpoint) {
      memoryAttachment = &attachment;
      break;
    }
  require(memoryAttachment,
          "builtin System has no spatial memory-service attachment");

  const auto core = system.view().accCoreOccurrences().front();
  const auto transport = system.view().pointConnections().front();
  std::vector<loom::dse::SystemCompositionDecisionDomain> domains = {
      loom::dse::SelectInstructionCoreRealizationDomain{
          loom::fabric::InstructionCoreContextRef{core},
          {loom::fabric::InstructionCoreContextRef{core}}},
      loom::dse::ChangeTransportConnectionDomain{transport.destination,
                                                 {transport.source}},
      loom::dse::ChangeServiceOrMemoryAttachmentDomain{
          loom::dse::ChangeSpatialMemoryAttachmentDomain{
              *memoryAttachment->spatialEndpoint.memory(),
              {*memoryAttachment->serviceEndpoint}}},
  };
  if (!system.view().memoryServiceConnections().empty()) {
    const auto memory = system.view().memoryServiceConnections().front();
    domains.push_back(loom::dse::ChangeServiceOrMemoryAttachmentDomain{
        loom::dse::ChangeMemoryServiceConnectionDomain{memory.destination,
                                                       {memory.source}}});
  }
  auto config = take(loom::dse::resolveSystemCompositionRewriteConfig(
      domains, domains.size()));
  auto inputs = take(loom::dse::bindSystemCompositionCandidateGeneratorInputs(
      {system.reference()}, {module.reference()}));
  auto binding = take(
      loom::dse::resolveSystemCompositionCandidateGeneratorBinding(config));
  auto result = take(loom::dse::invokeCandidateGenerator(
      inputs, binding, fixture.store, fixture.blobs));
  require(completed(result).outputBindings.front().artifacts.empty(),
          "relational no-op changed canonical System identity");
  require(completed(result).lineageEdges.empty(),
          "relational no-op incorrectly produced decision lineage");
}

} // namespace

int main() {
  strictConfigAdmission();
  localMemoryPortVariantRoundTrip();
  Fixture fixture;
  auto system = generateBuiltinSystem(fixture);
  parameterizedTemplateScale(fixture, system);
  auto module = importBuiltinModule(system, fixture);
  computeContextFeedbackRoundTrip(fixture, module);
  topologyRewrite(fixture, module);
  occurrenceInventoryRewrite(fixture, module);
  auto replacementModule = microarchitectureRewrite(fixture, module);
  temporalOperandBufferRewrite(fixture, module);
  redistributeFuCapability(fixture, module);
  hardwareImpactMatrix(system, module);
  systemCompositionRewrite(fixture, system, module);
  spatialAttachmentRewrite(fixture, system, replacementModule);
  transportResourceRewrite(fixture, system, module);
  systemRelationalNoOps(fixture, system, module);
  return EXIT_SUCCESS;
}

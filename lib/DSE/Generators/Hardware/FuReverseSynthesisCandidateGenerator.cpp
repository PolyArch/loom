#include "DSE/FuReverseSynthesis.h"

#include "Common/BlobStore.h"
#include "Fabric/Identity/FabricPhysicalTiming.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

enum InputSlot : std::uint32_t { DataflowInput, InputSlotCount };
enum OutputSlot : std::uint32_t {
  ModuleOutput = static_cast<std::uint32_t>(FuReverseSynthesisOutput::Module),
  MappingOutput =
      static_cast<std::uint32_t>(FuReverseSynthesisOutput::TechMapping),
  JointMappingOutput =
      static_cast<std::uint32_t>(FuReverseSynthesisOutput::JointTechMapping),
  SystemOutput = static_cast<std::uint32_t>(FuReverseSynthesisOutput::System),
  PhysicalTimingProfileOutput = static_cast<std::uint32_t>(
      FuReverseSynthesisOutput::PhysicalTimingProfile),
  ConfigurationAbiOutput =
      static_cast<std::uint32_t>(FuReverseSynthesisOutput::ConfigurationAbi),
  OutputSlotCount,
};
enum class LineageOutputKind : std::uint8_t {
  Module,
  TechMapping,
  System,
  PhysicalTimingProfile,
  ConfigurationAbi,
  JointTechMapping,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {
        {{CandidateGeneratorInputSlotRef(DataflowInput), "dataflow",
          PlanValueRole::CandidateSet, &::dataflow::canonicalDataflowSchema,
          PlanValueCardinality::ExactlyOne}}};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, OutputSlotCount>
    outputSlots = {{
        {CandidateGeneratorOutputSlotRef(ModuleOutput), "fabric_module",
         PlanValueRole::CandidateSet, &::loom::fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorOutputSlotRef(MappingOutput), "tech_mapping",
         PlanValueRole::CandidateSet, &::loom::mapping::mappingArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {CandidateGeneratorOutputSlotRef(JointMappingOutput),
         "joint_tech_mapping", PlanValueRole::CandidateSet,
         &::loom::mapping::mappingArtifactSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorOutputSlotRef(SystemOutput), "fabric_system",
         PlanValueRole::CandidateSet, &::loom::fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorOutputSlotRef(PhysicalTimingProfileOutput),
         "physical_timing_profile", PlanValueRole::CandidateSet,
         &::loom::fabric::fabricPhysicalTimingProfileArtifactSchema,
         PlanValueCardinality::ExactlyOne},
        {CandidateGeneratorOutputSlotRef(ConfigurationAbiOutput),
         "configuration_abi", PlanValueRole::CandidateSet,
         &::loom::hardware::configurationAbiSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "mapping_invocation"},
}};

constexpr llvm::StringLiteral lineageDescriptor =
    "loom.hardware.fu_reverse_synthesis.lineage.3";
constexpr llvm::StringLiteral outcomeDescriptor =
    "loom.hardware.fu_reverse_synthesis.outcome.3";

llvm::ArrayRef<std::uint8_t> lineageSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(lineageDescriptor.data()),
          lineageDescriptor.size()};
}

llvm::ArrayRef<std::uint8_t> outcomeSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(outcomeDescriptor.data()),
          outcomeDescriptor.size()};
}

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fu_reverse_synthesis_generator_invalid: " +
                                     message);
}

std::vector<::dataflow::GraphRef>
completeGraphSet(const ::dataflow::CanonicalDataflowProgramView &dataflow) {
  std::vector<::dataflow::GraphRef> graphs;
  graphs.reserve(dataflow.graphs().size());
  for (const ::dataflow::CanonicalGraphView &graph : dataflow.graphs())
    graphs.push_back(graph.ref);
  return graphs;
}

llvm::Error validateLineagePayload(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &output,
    llvm::ArrayRef<ArtifactRootReference> parents, const ArtifactStore &store) {
  if (bytes.size() != 1 ||
      bytes.front() >
          static_cast<std::uint8_t>(LineageOutputKind::JointTechMapping))
    return invalid("lineage payload is not canonical");
  const auto kind = static_cast<LineageOutputKind>(bytes.front());
  const ArtifactRootReference *dataflow = nullptr;
  const ArtifactRootReference *fabric = nullptr;
  for (const ArtifactRootReference &parent : parents) {
    if (parent.schemaIdentity == ::dataflow::canonicalDataflowSchema.identity &&
        parent.schemaVersion == ::dataflow::canonicalDataflowSchema.version)
      dataflow = &parent;
    if (parent.schemaIdentity ==
            ::loom::fabric::fabricArtifactSchema.identity &&
        parent.schemaVersion == ::loom::fabric::fabricArtifactSchema.version)
      fabric = &parent;
  }

  if (kind == LineageOutputKind::Module) {
    if (parents.size() != 1 || !dataflow ||
        output.schemaIdentity !=
            ::loom::fabric::fabricArtifactSchema.identity ||
        output.schemaVersion != ::loom::fabric::fabricArtifactSchema.version)
      return invalid("Module lineage has the wrong closure");
    auto importedDataflow =
        ::dataflow::importCanonicalDataflow(*dataflow, store);
    if (!importedDataflow)
      return importedDataflow.takeError();
    auto synthesizedFabric =
        ::loom::fabric::importEntireFabricRoot(output, store);
    if (!synthesizedFabric)
      return synthesizedFabric.takeError();
    if (synthesizedFabric->view().rootKind() !=
            ::loom::fabric::FabricRootKind::Module ||
        synthesizedFabric->view().fuTemplates().size() != 1)
      return invalid("Fabric lineage output is not a bounded FU Module");
    auto dataflowView = importedDataflow->view();
    if (!dataflowView)
      return dataflowView.takeError();
    const std::vector<::dataflow::GraphRef> graphs =
        completeGraphSet(*dataflowView);
    if (llvm::Error error = verifyScalarIntegerAddSubFuFabricLineage(
            *dataflowView, graphs, *synthesizedFabric, store))
      return error;
    return llvm::Error::success();
  }

  if (kind == LineageOutputKind::TechMapping) {
    if (parents.size() != 2 || !dataflow || !fabric ||
        output.schemaIdentity !=
            ::loom::mapping::mappingArtifactSchema.identity ||
        output.schemaVersion != ::loom::mapping::mappingArtifactSchema.version)
      return invalid("TechMapping lineage has the wrong closure");
    auto importedDataflow =
        ::dataflow::importCanonicalDataflow(*dataflow, store);
    if (!importedDataflow)
      return importedDataflow.takeError();
    auto mapping = ::loom::mapping::importTechMapping(output, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() != dataflow->artifact ||
        mapping->view().fabricIdentity() != fabric->artifact ||
        mapping->view().covers().size() != 1)
      return invalid("TechMapping lineage does not bind its exact parents");
    auto dataflowView = importedDataflow->view();
    if (!dataflowView)
      return dataflowView.takeError();
    auto synthesizedFabric =
        ::loom::fabric::importEntireFabricRoot(*fabric, store);
    if (!synthesizedFabric)
      return synthesizedFabric.takeError();
    const std::vector<::dataflow::GraphRef> graphs =
        completeGraphSet(*dataflowView);
    return verifyScalarIntegerAddSubFuMappingLineage(
        *dataflowView, graphs, *synthesizedFabric, *mapping, store);
  }

  if (kind == LineageOutputKind::System) {
    if (parents.size() != 1 || !fabric ||
        output.schemaIdentity !=
            ::loom::fabric::fabricArtifactSchema.identity ||
        output.schemaVersion != ::loom::fabric::fabricArtifactSchema.version)
      return invalid("System lineage has the wrong closure");
    auto importedParent =
        ::loom::fabric::importEntireFabricRoot(*fabric, store);
    if (!importedParent)
      return importedParent.takeError();
    auto system = ::loom::fabric::importEntireFabricRoot(output, store);
    if (!system)
      return system.takeError();
    return verifyScalarIntegerAddSubFuSystemIdentity(*importedParent, *system,
                                                     store);
  }

  if (kind == LineageOutputKind::JointTechMapping) {
    if (parents.size() != 2 || !dataflow || !fabric ||
        output.schemaIdentity !=
            ::loom::mapping::mappingArtifactSchema.identity ||
        output.schemaVersion != ::loom::mapping::mappingArtifactSchema.version)
      return invalid("joint TechMapping lineage has the wrong closure");
    auto importedDataflow =
        ::dataflow::importCanonicalDataflow(*dataflow, store);
    if (!importedDataflow)
      return importedDataflow.takeError();
    auto dataflowView = importedDataflow->view();
    if (!dataflowView)
      return dataflowView.takeError();
    auto synthesizedFabric =
        ::loom::fabric::importEntireFabricRoot(*fabric, store);
    if (!synthesizedFabric)
      return synthesizedFabric.takeError();
    auto mapping = ::loom::mapping::importTechMapping(output, store);
    if (!mapping)
      return mapping.takeError();
    const std::vector<::dataflow::GraphRef> graphs =
        completeGraphSet(*dataflowView);
    return verifyScalarIntegerAddSubFuJointMappingLineage(
        *dataflowView, graphs, *synthesizedFabric, *mapping, store);
  }

  if (kind == LineageOutputKind::PhysicalTimingProfile) {
    if (parents.size() != 1 || !fabric)
      return invalid("timing lineage has the wrong closure");
    auto importedParent =
        ::loom::fabric::importEntireFabricRoot(*fabric, store);
    if (!importedParent)
      return importedParent.takeError();
    if (output.schemaIdentity !=
            ::loom::fabric::fabricPhysicalTimingProfileArtifactSchema
                .identity ||
        output.schemaVersion !=
            ::loom::fabric::fabricPhysicalTimingProfileArtifactSchema.version)
      return invalid("timing lineage has the wrong output schema");
    return verifyScalarIntegerAddSubFuPhysicalTimingLineage(*importedParent,
                                                            output, store);
  }

  if (parents.size() != 1 || !fabric)
    return invalid("ConfigurationABI lineage has the wrong closure");
  if (output.schemaIdentity !=
          ::loom::hardware::configurationAbiSchema.identity ||
      output.schemaVersion != ::loom::hardware::configurationAbiSchema.version)
    return invalid("ConfigurationABI lineage has the wrong output schema");
  auto system = ::loom::fabric::importEntireFabricRoot(*fabric, store);
  if (!system)
    return system.takeError();
  return verifyScalarIntegerAddSubFuConfigurationAbiLineage(*system, output,
                                                            store);
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    lineageSchemaBytes(), validateLineagePayload};

llvm::Error
validateOutcome(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                llvm::ArrayRef<CandidateGeneratorOutputBinding> outputs,
                llvm::ArrayRef<CandidateGeneratorLineageEdge> lineageEdges,
                bool completed, const ArtifactStore &store) {
  if (inputs.size() != 1 || inputs.front().artifacts.size() != 1 ||
      outputs.size() != OutputSlotCount ||
      outputs[ModuleOutput].slot !=
          CandidateGeneratorOutputSlotRef(ModuleOutput) ||
      outputs[MappingOutput].slot !=
          CandidateGeneratorOutputSlotRef(MappingOutput) ||
      outputs[JointMappingOutput].slot !=
          CandidateGeneratorOutputSlotRef(JointMappingOutput) ||
      outputs[SystemOutput].slot !=
          CandidateGeneratorOutputSlotRef(SystemOutput) ||
      outputs[PhysicalTimingProfileOutput].slot !=
          CandidateGeneratorOutputSlotRef(PhysicalTimingProfileOutput) ||
      outputs[ConfigurationAbiOutput].slot !=
          CandidateGeneratorOutputSlotRef(ConfigurationAbiOutput))
    return invalid("outcome does not have the canonical slot closure");

  auto dataflow = ::dataflow::importCanonicalDataflow(
      inputs.front().artifacts.front(), store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();
  const std::vector<::dataflow::GraphRef> graphs =
      completeGraphSet(*dataflowView);

  const auto &moduleOutputs = outputs[ModuleOutput].artifacts;
  const auto &mappingOutputs = outputs[MappingOutput].artifacts;
  const auto &jointMappingOutputs = outputs[JointMappingOutput].artifacts;
  const auto &systemOutputs = outputs[SystemOutput].artifacts;
  const auto &timingOutputs = outputs[PhysicalTimingProfileOutput].artifacts;
  const auto &abiOutputs = outputs[ConfigurationAbiOutput].artifacts;
  if (moduleOutputs.size() > 1 || systemOutputs.size() > 1 ||
      timingOutputs.size() > 1 || abiOutputs.size() > 1 ||
      jointMappingOutputs.size() > 1)
    return invalid("outcome contains a non-unique fixed synthesis artifact");
  const bool hasFixedClosure = moduleOutputs.size() == 1;
  if ((systemOutputs.size() == 1) != hasFixedClosure ||
      (timingOutputs.size() == 1) != hasFixedClosure ||
      (abiOutputs.size() == 1) != hasFixedClosure ||
      (!jointMappingOutputs.empty() && !hasFixedClosure) ||
      (!mappingOutputs.empty() && !hasFixedClosure))
    return invalid("outcome has an incomplete fixed synthesis closure");
  if (completed && (!hasFixedClosure || jointMappingOutputs.size() != 1 ||
                    mappingOutputs.size() != dataflowView->graphs().size()))
    return invalid("completed outcome does not cover the complete graph "
                   "domain");

  if (hasFixedClosure) {
    auto module =
        ::loom::fabric::importEntireFabricRoot(moduleOutputs.front(), store);
    if (!module)
      return module.takeError();
    auto system =
        ::loom::fabric::importEntireFabricRoot(systemOutputs.front(), store);
    if (!system)
      return system.takeError();
    if (llvm::Error error =
            verifyScalarIntegerAddSubFuSystemIdentity(*module, *system, store))
      return error;
    if (llvm::Error error = verifyScalarIntegerAddSubFuPhysicalTimingLineage(
            *module, timingOutputs.front(), store))
      return error;
    if (llvm::Error error = verifyScalarIntegerAddSubFuConfigurationAbiLineage(
            *system, abiOutputs.front(), store))
      return error;
    if (!jointMappingOutputs.empty()) {
      auto jointMapping = ::loom::mapping::importTechMapping(
          jointMappingOutputs.front(), store);
      if (!jointMapping)
        return jointMapping.takeError();
      if (llvm::Error error = verifyScalarIntegerAddSubFuJointMappingLineage(
              *dataflowView, graphs, *module, *jointMapping, store))
        return error;
    }
  }

  std::vector<::dataflow::GraphRef> coveredGraphs;
  coveredGraphs.reserve(mappingOutputs.size());
  for (const ArtifactRootReference &reference : mappingOutputs) {
    auto mapping = ::loom::mapping::importTechMapping(reference, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() != dataflowView->identity() ||
        mapping->view().fabricIdentity() != moduleOutputs.front().artifact ||
        mapping->view().covers().size() != 1)
      return invalid("outcome mapping does not bind its exact synthesis "
                     "domain");
    const ::dataflow::GraphRef graph = mapping->view().covers().front();
    if (llvm::is_contained(coveredGraphs, graph))
      return invalid("outcome maps one graph more than once");
    if (!llvm::any_of(dataflowView->graphs(), [&](const auto &candidate) {
          return candidate.ref == graph;
        }))
      return invalid("outcome mapping covers a graph outside its input");
    coveredGraphs.push_back(graph);
  }
  if (completed)
    for (const ::dataflow::CanonicalGraphView &graph : dataflowView->graphs())
      if (!llvm::is_contained(coveredGraphs, graph.ref))
        return invalid("completed outcome omits a graph mapping");

  std::size_t outputCount = 0;
  for (const CandidateGeneratorOutputBinding &binding : outputs)
    outputCount += binding.artifacts.size();
  if (lineageEdges.size() != outputCount)
    return invalid("outcome lineage is not exact for its returned artifacts");
  for (const CandidateGeneratorLineageEdge &edge : lineageEdges) {
    if (edge.outputSlot.ordinal() >= outputs.size() ||
        !llvm::is_contained(outputs[edge.outputSlot.ordinal()].artifacts,
                            edge.output))
      return invalid("outcome contains internal synthesis lineage");
  }
  for (const CandidateGeneratorOutputBinding &binding : outputs)
    for (const ArtifactRootReference &artifact : binding.artifacts)
      if (llvm::count_if(lineageEdges, [&](const auto &edge) {
            return edge.outputSlot == binding.slot && edge.output == artifact;
          }) != 1)
        return invalid("outcome does not have one lineage edge per artifact");
  return llvm::Error::success();
}

const CandidateGeneratorOwnerOutcomeContract outcomeContract{
    outcomeSchemaBytes(), validateOutcome};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = ::loom::mapping::adoptResolvedTechMappingConfigView(
      ::loom::mapping::resolvedTechMappingConfigSchemaDescriptorBytes(), bytes,
      digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store, const BlobStore &,
               const CandidateGeneratorInvocationView &invocation) {
  if (inputBindings.size() != 1 ||
      inputBindings.front().slot !=
          CandidateGeneratorInputSlotRef(DataflowInput) ||
      inputBindings.front().artifacts.size() != 1)
    return invalid("provider requires one Dataflow root");
  const ArtifactRootReference &dataflowReference =
      inputBindings.front().artifacts.front();
  auto dataflow = ::dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflow)
    return dataflow.takeError();
  auto view = dataflow->view();
  if (!view)
    return view.takeError();
  auto mappingConfig = ::loom::mapping::adoptResolvedTechMappingConfigView(
      ::loom::mapping::resolvedTechMappingConfigSchemaDescriptorBytes(),
      binding.canonicalConfigBytes(), binding.configDigest());
  if (!mappingConfig)
    return mappingConfig.takeError();

  std::vector<::dataflow::GraphRef> graphs;
  graphs.reserve(view->graphs().size());
  for (const ::dataflow::CanonicalGraphView &graph : view->graphs())
    graphs.push_back(graph.ref);
  const auto incompleteResult = [&](CandidateGeneratorIncompleteReason reason) {
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            reason,
            {{CandidateGeneratorOutputSlotRef(ModuleOutput), {}},
             {CandidateGeneratorOutputSlotRef(MappingOutput), {}},
             {CandidateGeneratorOutputSlotRef(JointMappingOutput), {}},
             {CandidateGeneratorOutputSlotRef(SystemOutput), {}},
             {CandidateGeneratorOutputSlotRef(PhysicalTimingProfileOutput), {}},
             {CandidateGeneratorOutputSlotRef(ConfigurationAbiOutput), {}}},
            {}},
        {{CandidateGeneratorWorkUnitRef(0), graphs.size() + 1, 0}}};
  };
  if (graphs.empty())
    return incompleteResult(CandidateGeneratorIncompleteReason::Unsupported);
  const std::optional<std::uint64_t> mappingLimit =
      invocation.maximumOutputArtifacts(
          CandidateGeneratorOutputSlotRef(MappingOutput));
  const std::array fixedOutputs = {
      CandidateGeneratorOutputSlotRef(ModuleOutput),
      CandidateGeneratorOutputSlotRef(JointMappingOutput),
      CandidateGeneratorOutputSlotRef(SystemOutput),
      CandidateGeneratorOutputSlotRef(PhysicalTimingProfileOutput),
      CandidateGeneratorOutputSlotRef(ConfigurationAbiOutput)};
  const bool fixedOutputLimited = llvm::any_of(fixedOutputs, [&](auto slot) {
    const std::optional<std::uint64_t> limit =
        invocation.maximumOutputArtifacts(slot);
    return limit && *limit < 1;
  });
  if (fixedOutputLimited || (mappingLimit && *mappingLimit < graphs.size()))
    return incompleteResult(
        CandidateGeneratorIncompleteReason::SemanticLimitReached);
  if (invocation.stopRequested())
    return incompleteResult(
        CandidateGeneratorIncompleteReason::CancelledOrTimeout);

  auto synthesized = attemptScalarIntegerAddSubFuSynthesis(
      *view, graphs, *mappingConfig, store, invocation.executionControl());
  if (!synthesized) {
    std::optional<FuReverseSynthesisFailure> kind;
    std::string diagnostic;
    llvm::Error remaining = llvm::handleErrors(
        synthesized.takeError(), [&](const FuReverseSynthesisError &error) {
          kind = error.failure();
          diagnostic = error.diagnostic().str();
        });
    if (remaining)
      return std::move(remaining);
    if (!kind)
      return invalid("bounded synthesis lost its typed failure");
    switch (*kind) {
    case FuReverseSynthesisFailure::EmptyGraphSet:
    case FuReverseSynthesisFailure::UnsupportedGraphInterface:
    case FuReverseSynthesisFailure::UnsupportedActorInventory:
    case FuReverseSynthesisFailure::UnsupportedActorSchema:
    case FuReverseSynthesisFailure::UnsupportedActorProjection:
    case FuReverseSynthesisFailure::UnsupportedGraphTopology:
      return incompleteResult(CandidateGeneratorIncompleteReason::Unsupported);
    case FuReverseSynthesisFailure::CancelledOrTimeout:
      return incompleteResult(
          CandidateGeneratorIncompleteReason::CancelledOrTimeout);
    default:
      return invalid("bounded synthesis failed: " + diagnostic);
    }
  }

  if (synthesized->termination() ==
      FuReverseSynthesisFailure::MappingInfeasible)
    return invalid("synthesized FU failed its required graph mapping");

  auto systemArtifacts = materializeScalarIntegerAddSubFuSystemArtifacts(
      synthesized->fabric(), store);
  if (!systemArtifacts)
    return systemArtifacts.takeError();
  if (llvm::Error error = verifyScalarIntegerAddSubFuSystemLineage(
          synthesized->fabric(), *systemArtifacts, store))
    return std::move(error);

  std::vector<ArtifactRootReference> mappingOutputs;
  mappingOutputs.reserve(synthesized->perGraphMappings().size());
  std::vector<CandidateGeneratorLineageEdge> lineage;
  lineage.reserve(5 + synthesized->perGraphMappings().size());
  lineage.push_back(CandidateGeneratorLineageEdge{
      CandidateGeneratorLineageEdgeKind::CandidateDecision,
      CandidateGeneratorOutputSlotRef(ModuleOutput),
      synthesized->fabric().reference(),
      {dataflowReference},
      {static_cast<std::uint8_t>(LineageOutputKind::Module)}});
  for (const ::loom::mapping::FinalizedTechMapping &mapping :
       synthesized->perGraphMappings()) {
    mappingOutputs.push_back(mapping.reference());
    lineage.push_back(CandidateGeneratorLineageEdge{
        CandidateGeneratorLineageEdgeKind::CandidateDecision,
        CandidateGeneratorOutputSlotRef(MappingOutput),
        mapping.reference(),
        {dataflowReference, synthesized->fabric().reference()},
        {static_cast<std::uint8_t>(LineageOutputKind::TechMapping)}});
  }
  if (synthesized->jointMapping())
    lineage.push_back(CandidateGeneratorLineageEdge{
        CandidateGeneratorLineageEdgeKind::CandidateDecision,
        CandidateGeneratorOutputSlotRef(JointMappingOutput),
        synthesized->jointMapping()->reference(),
        {dataflowReference, synthesized->fabric().reference()},
        {static_cast<std::uint8_t>(LineageOutputKind::JointTechMapping)}});
  lineage.push_back(CandidateGeneratorLineageEdge{
      CandidateGeneratorLineageEdgeKind::CandidateDecision,
      CandidateGeneratorOutputSlotRef(SystemOutput),
      systemArtifacts->system().reference(),
      {synthesized->fabric().reference()},
      {static_cast<std::uint8_t>(LineageOutputKind::System)}});
  lineage.push_back(CandidateGeneratorLineageEdge{
      CandidateGeneratorLineageEdgeKind::CandidateDecision,
      CandidateGeneratorOutputSlotRef(PhysicalTimingProfileOutput),
      systemArtifacts->physicalTimingProfile(),
      {synthesized->fabric().reference()},
      {static_cast<std::uint8_t>(LineageOutputKind::PhysicalTimingProfile)}});
  lineage.push_back(CandidateGeneratorLineageEdge{
      CandidateGeneratorLineageEdgeKind::CandidateDecision,
      CandidateGeneratorOutputSlotRef(ConfigurationAbiOutput),
      systemArtifacts->configurationAbi().reference(),
      {systemArtifacts->system().reference()},
      {static_cast<std::uint8_t>(LineageOutputKind::ConfigurationAbi)}});
  std::vector<CandidateGeneratorOutputBinding> retained = {
      {CandidateGeneratorOutputSlotRef(ModuleOutput),
       {synthesized->fabric().reference()}},
      {CandidateGeneratorOutputSlotRef(MappingOutput),
       std::move(mappingOutputs)},
      {CandidateGeneratorOutputSlotRef(JointMappingOutput),
       synthesized->jointMapping()
           ? std::vector<ArtifactRootReference>{synthesized->jointMapping()
                                                    ->reference()}
           : std::vector<ArtifactRootReference>{}},
      {CandidateGeneratorOutputSlotRef(SystemOutput),
       {systemArtifacts->system().reference()}},
      {CandidateGeneratorOutputSlotRef(PhysicalTimingProfileOutput),
       {systemArtifacts->physicalTimingProfile()}},
      {CandidateGeneratorOutputSlotRef(ConfigurationAbiOutput),
       {systemArtifacts->configurationAbi().reference()}}};
  const std::vector<CandidateGeneratorWorkUnitSummary> summary = {
      {CandidateGeneratorWorkUnitRef(0),
       synthesized->plannedMappingInvocations(),
       synthesized->consumedMappingInvocations()}};
  if (synthesized->termination()) {
    CandidateGeneratorIncompleteReason reason;
    switch (*synthesized->termination()) {
    case FuReverseSynthesisFailure::MappingIncomplete:
      reason = CandidateGeneratorIncompleteReason::ProofNotEstablished;
      break;
    case FuReverseSynthesisFailure::CancelledOrTimeout:
      reason = CandidateGeneratorIncompleteReason::CancelledOrTimeout;
      break;
    default:
      return invalid("bounded synthesis returned a non-retainable partial "
                     "termination");
    }
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{reason, std::move(retained),
                                           std::move(lineage)},
        summary};
  }
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{std::move(retained),
                                        std::move(lineage)},
      summary};
}

const CandidateGeneratorDescriptor descriptor{
    fuReverseSynthesisCandidateGeneratorKind,
    "fu_reverse_synthesis",
    "loom.hardware.fu_reverse_synthesis.generator.v3",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{
        ::loom::mapping::resolvedTechMappingConfigSchemaDescriptorBytes(),
        validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &lineageContract,
    ProviderForm::InProcess,
    nullptr,
    &outcomeContract,
};

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeProvider}};

} // namespace

const CandidateGeneratorDescriptor &
fuReverseSynthesisCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerFuReverseSynthesisCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindFuReverseSynthesisCandidateGeneratorInputs(
    const ArtifactRootReference &dataflow) {
  if (llvm::Error error = registerFuReverseSynthesisCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> inputs = {
      {CandidateGeneratorInputSlotRef(DataflowInput), {dataflow}}};
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), inputs))
    return std::move(error);
  return inputs;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveFuReverseSynthesisCandidateGeneratorBinding(
    const ::loom::mapping::ResolvedTechMappingConfigView &mappingConfig) {
  if (llvm::Error error = registerFuReverseSynthesisCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), mappingConfig.canonicalViewBytes(),
      mappingConfig.digest());
}

} // namespace loom::dse

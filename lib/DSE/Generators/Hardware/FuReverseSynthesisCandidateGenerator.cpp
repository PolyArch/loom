#include "DSE/FuReverseSynthesis.h"

#include "Common/BlobStore.h"

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
enum OutputSlot : std::uint32_t { FabricOutput, MappingOutput };
enum class LineageOutputKind : std::uint8_t { Fabric, TechMapping };

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {
        {{CandidateGeneratorInputSlotRef(DataflowInput), "dataflow",
          PlanValueRole::CandidateSet, &::dataflow::canonicalDataflowSchema,
          PlanValueCardinality::ExactlyOne}}};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 2> outputSlots = {{
    {CandidateGeneratorOutputSlotRef(FabricOutput), "fabric",
     PlanValueRole::CandidateSet, &::loom::fabric::fabricArtifactSchema,
     PlanValueCardinality::FiniteSet},
    {CandidateGeneratorOutputSlotRef(MappingOutput), "tech_mapping",
     PlanValueRole::CandidateSet, &::loom::mapping::mappingArtifactSchema,
     PlanValueCardinality::FiniteSet},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "graph_binding"},
}};

constexpr llvm::StringLiteral lineageDescriptor =
    "loom.hardware.fu_reverse_synthesis.lineage.1";
constexpr llvm::StringLiteral outcomeDescriptor =
    "loom.hardware.fu_reverse_synthesis.outcome.1";

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

llvm::Error validateLineagePayload(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &output,
    llvm::ArrayRef<ArtifactRootReference> parents, const ArtifactStore &store) {
  if (bytes.size() != 1 ||
      bytes.front() > static_cast<std::uint8_t>(LineageOutputKind::TechMapping))
    return invalid("lineage payload is not canonical");
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
  if (!dataflow)
    return invalid("lineage omits its canonical Dataflow parent");
  auto importedDataflow = ::dataflow::importCanonicalDataflow(*dataflow, store);
  if (!importedDataflow)
    return importedDataflow.takeError();

  if (bytes.front() == static_cast<std::uint8_t>(LineageOutputKind::Fabric)) {
    if (parents.size() != 1 ||
        output.schemaIdentity !=
            ::loom::fabric::fabricArtifactSchema.identity ||
        output.schemaVersion != ::loom::fabric::fabricArtifactSchema.version)
      return invalid("Fabric lineage has the wrong closure");
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
    if (llvm::Error error = verifyScalarIntegerAddSubFuFabricLineage(
            *dataflowView, *synthesizedFabric, store))
      return error;
    return llvm::Error::success();
  }

  if (parents.size() != 2 || !fabric ||
      output.schemaIdentity !=
          ::loom::mapping::mappingArtifactSchema.identity ||
      output.schemaVersion != ::loom::mapping::mappingArtifactSchema.version)
    return invalid("TechMapping lineage has the wrong closure");
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
  if (llvm::Error error = verifyScalarIntegerAddSubFuMappingLineage(
          *dataflowView, *synthesizedFabric, *mapping, store))
    return error;
  return llvm::Error::success();
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    lineageSchemaBytes(), validateLineagePayload};

llvm::Error
validateOutcome(llvm::ArrayRef<CandidateGeneratorInputBinding> inputs,
                llvm::ArrayRef<CandidateGeneratorOutputBinding> outputs,
                llvm::ArrayRef<CandidateGeneratorLineageEdge> lineageEdges,
                bool completed, const ArtifactStore &store) {
  if (inputs.size() != 1 || inputs.front().artifacts.size() != 1 ||
      outputs.size() != 2 ||
      outputs[FabricOutput].slot !=
          CandidateGeneratorOutputSlotRef(FabricOutput) ||
      outputs[MappingOutput].slot !=
          CandidateGeneratorOutputSlotRef(MappingOutput))
    return invalid("outcome does not have the canonical slot closure");

  auto dataflow = ::dataflow::importCanonicalDataflow(
      inputs.front().artifacts.front(), store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();

  const auto &fabricOutputs = outputs[FabricOutput].artifacts;
  const auto &mappingOutputs = outputs[MappingOutput].artifacts;
  if (fabricOutputs.size() > 1 ||
      (!mappingOutputs.empty() && fabricOutputs.empty()))
    return invalid("outcome has no unique Fabric owner for its mappings");
  if (completed && (fabricOutputs.size() != 1 ||
                    mappingOutputs.size() != dataflowView->graphs().size()))
    return invalid("completed outcome does not cover the complete graph "
                   "domain");

  std::vector<::dataflow::GraphRef> coveredGraphs;
  coveredGraphs.reserve(mappingOutputs.size());
  for (const ArtifactRootReference &reference : mappingOutputs) {
    auto mapping = ::loom::mapping::importTechMapping(reference, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() != dataflowView->identity() ||
        mapping->view().fabricIdentity() != fabricOutputs.front().artifact ||
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

  const std::size_t outputCount = fabricOutputs.size() + mappingOutputs.size();
  if (lineageEdges.size() != outputCount)
    return invalid("outcome lineage is not exact for its returned artifacts");
  for (const CandidateGeneratorLineageEdge &edge : lineageEdges)
    if (!llvm::is_contained(outputs[edge.outputSlot.ordinal()].artifacts,
                            edge.output))
      return invalid("outcome contains internal synthesis lineage");
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
            {{CandidateGeneratorOutputSlotRef(FabricOutput), {}},
             {CandidateGeneratorOutputSlotRef(MappingOutput), {}}},
            {}},
        {{CandidateGeneratorWorkUnitRef(0), graphs.size(), 0}}};
  };
  if (graphs.empty())
    return incompleteResult(CandidateGeneratorIncompleteReason::Unsupported);
  const std::optional<std::uint64_t> fabricLimit =
      invocation.maximumOutputArtifacts(
          CandidateGeneratorOutputSlotRef(FabricOutput));
  const std::optional<std::uint64_t> mappingLimit =
      invocation.maximumOutputArtifacts(
          CandidateGeneratorOutputSlotRef(MappingOutput));
  if ((fabricLimit && *fabricLimit < 1) ||
      (mappingLimit && *mappingLimit < graphs.size()))
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

  std::vector<ArtifactRootReference> mappingOutputs;
  mappingOutputs.reserve(synthesized->mappings().size());
  std::vector<CandidateGeneratorLineageEdge> lineage;
  lineage.reserve(1 + synthesized->mappings().size());
  lineage.push_back(CandidateGeneratorLineageEdge{
      CandidateGeneratorLineageEdgeKind::CandidateDecision,
      CandidateGeneratorOutputSlotRef(FabricOutput),
      synthesized->fabric().reference(),
      {dataflowReference},
      {static_cast<std::uint8_t>(LineageOutputKind::Fabric)}});
  for (const ::loom::mapping::FinalizedTechMapping &mapping :
       synthesized->mappings()) {
    mappingOutputs.push_back(mapping.reference());
    lineage.push_back(CandidateGeneratorLineageEdge{
        CandidateGeneratorLineageEdgeKind::CandidateDecision,
        CandidateGeneratorOutputSlotRef(MappingOutput),
        mapping.reference(),
        {dataflowReference, synthesized->fabric().reference()},
        {static_cast<std::uint8_t>(LineageOutputKind::TechMapping)}});
  }
  std::vector<CandidateGeneratorOutputBinding> retained = {
      {CandidateGeneratorOutputSlotRef(FabricOutput),
       {synthesized->fabric().reference()}},
      {CandidateGeneratorOutputSlotRef(MappingOutput),
       std::move(mappingOutputs)}};
  const std::vector<CandidateGeneratorWorkUnitSummary> summary = {
      {CandidateGeneratorWorkUnitRef(0), synthesized->plannedGraphBindings(),
       synthesized->consumedGraphBindings()}};
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
    "loom.hardware.fu_reverse_synthesis.generator.v1",
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

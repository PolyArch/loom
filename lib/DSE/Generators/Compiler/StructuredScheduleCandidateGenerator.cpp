#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/MappingDebugLog.h"
#include "Config/ResolvedConfig.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/StructuredSchedule.h"

#include "Frontend/IR/StructuredProgramArtifact.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.structured_schedule_generator.config.3.0";

enum InputSlot : std::uint32_t {
  StructuredProgramsInput,
  FabricInput,
  InputSlotCount,
};

constexpr std::array<CandidateGeneratorInputSlotDescriptor, InputSlotCount>
    inputSlots = {{
        {CandidateGeneratorInputSlotRef(StructuredProgramsInput),
         "structured_program", PlanValueRole::CandidateSet,
         &frontend::structuredProgramArtifactSchema,
         PlanValueCardinality::FiniteSet},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {{
    {CandidateGeneratorOutputSlotRef(0), "structured_program",
     PlanValueRole::CandidateSet, &frontend::structuredProgramArtifactSchema,
     PlanValueCardinality::FiniteSet},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 5> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "loop_scope"},
    {CandidateGeneratorWorkUnitRef(1), "candidate_materialization"},
    {CandidateGeneratorWorkUnitRef(2), "schedule_coordinate"},
    {CandidateGeneratorWorkUnitRef(3), "bounded_candidate"},
    {CandidateGeneratorWorkUnitRef(4), "polyhedral_dependence_query"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "structured_schedule_generator_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

std::vector<std::uint8_t>
encodeConfig(std::uint64_t limit,
             std::optional<std::uint64_t> maximumMaterializationAttempts,
             StructuredScheduleGenerationIntent intent) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(17);
  const auto appendU64 = [&](std::uint64_t value) {
    for (unsigned shift = 56; shift != 0; shift -= 8)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
    bytes.push_back(static_cast<std::uint8_t>(value));
  };
  appendU64(limit);
  appendU64(maximumMaterializationAttempts.value_or(0));
  bytes.push_back(static_cast<std::uint8_t>(intent));
  return bytes;
}

struct DecodedConfig final {
  std::uint64_t limit;
  std::optional<std::uint64_t> maximumMaterializationAttempts;
  StructuredScheduleGenerationIntent intent;
};

llvm::Expected<DecodedConfig> decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() < 17)
    return invalid("truncated schedule generator config");
  if (bytes.size() != 17)
    return invalid("config has trailing bytes");
  const auto readU64 = [&](std::size_t offset) {
    std::uint64_t value = 0;
    for (std::uint8_t byte : bytes.slice(offset, 8))
      value = (value << 8) | byte;
    return value;
  };
  const std::uint64_t limit = readU64(0);
  if (limit == 0)
    return invalid("scope expansion limit must be positive");
  const std::uint64_t materializationLimit = readU64(8);
  const auto intent =
      static_cast<StructuredScheduleGenerationIntent>(bytes.back());
  if (intent > StructuredScheduleGenerationIntent::ForbidLogicalThreadDomain)
    return invalid("config has an unknown generation intent");
  return DecodedConfig{limit,
                       materializationLimit == 0
                           ? std::nullopt
                           : std::optional<std::uint64_t>(materializationLimit),
                       intent};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedStructuredScheduleGeneratorConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Error validateDecisionPayload(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &output,
    llvm::ArrayRef<ArtifactRootReference> parents, const ArtifactStore &store) {
  auto adopted = frontend::adoptStructuredScheduleDecision(bytes);
  if (!adopted)
    return adopted.takeError();
  const ArtifactRootReference *structuredParent = nullptr;
  const ArtifactRootReference *fabricParent = nullptr;
  for (const ArtifactRootReference &parent : parents) {
    if (parent.schemaIdentity ==
            frontend::structuredProgramArtifactSchema.identity &&
        parent.schemaVersion ==
            frontend::structuredProgramArtifactSchema.version) {
      if (structuredParent)
        return invalid("schedule lineage has multiple Structured parents");
      structuredParent = &parent;
      continue;
    }
    if (parent.schemaIdentity == fabric::fabricArtifactSchema.identity &&
        parent.schemaVersion == fabric::fabricArtifactSchema.version) {
      if (fabricParent)
        return invalid("schedule lineage has multiple Fabric parents");
      fabricParent = &parent;
    }
  }
  if (parents.size() != 2 || !structuredParent || !fabricParent ||
      adopted->loop.parent != structuredParent->artifact)
    return invalid(
        "schedule decision does not bind one exact Structured/Fabric pair");
  auto parent = frontend::importStructuredProgram(*structuredParent, store);
  if (!parent)
    return parent.takeError();
  auto lineageFabric = fabric::importEntireFabricRoot(*fabricParent, store);
  if (!lineageFabric)
    return lineageFabric.takeError();
  auto view = parent->view();
  if (!view)
    return view.takeError();
  auto loop = view->resolve(adopted->loop);
  if (!loop)
    return loop.takeError();
  const bool exactLoop =
      adopted->kind == frontend::StructuredScheduleDecisionKind::Vectorize
          ? llvm::isa_and_nonnull<mlir::scf::ForOp, mlir::affine::AffineForOp>(
                loop->operation)
          : llvm::isa_and_nonnull<mlir::scf::ForOp>(loop->operation);
  if (!exactLoop)
    return invalid("schedule decision does not reference its exact loop kind");
  auto child = frontend::importStructuredProgram(output, store);
  if (!child)
    return child.takeError();
  return frontend::verifyStructuredScheduleDerivation(*parent, *lineageFabric,
                                                      *adopted, *child);
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    frontend::structuredScheduleDecisionSchemaBytes(), validateDecisionPayload};

const CandidateGeneratorDescriptor descriptor{
    structuredScheduleCandidateGeneratorKind,
    "compiler.structured_schedule",
    "loom.compiler.structured_schedule.generator.v10",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    &lineageContract,
    ProviderForm::InProcess,
};

const ArtifactRootReference &
singleInput(llvm::ArrayRef<CandidateGeneratorInputBinding> bindings,
            InputSlot slot) {
  return bindings[slot].artifacts.front();
}

llvm::Expected<CandidateGeneratorProviderResult> invokeScheduleProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store, const BlobStore &blobs,
    const CandidateGeneratorInvocationView &invocationView) {
  auto config = adoptResolvedStructuredScheduleGeneratorConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();
  StructuredOwnershipInvocation *invocation =
      detail::StructuredOwnershipInvocationAccess::current();
  std::optional<fabric::FinalizedFabricRoot> importedFabric;
  const fabric::FinalizedFabricRoot *exactFabric = nullptr;
  if (invocation) {
    exactFabric =
        &detail::StructuredOwnershipInvocationAccess::fabric(*invocation);
    if (singleInput(inputBindings, FabricInput) != exactFabric->reference())
      return invalid("Fabric input differs from the bound invocation");
  } else {
    auto imported = fabric::importEntireFabricRoot(
        singleInput(inputBindings, FabricInput), store);
    if (!imported)
      return imported.takeError();
    importedFabric.emplace(std::move(*imported));
    exactFabric = &*importedFabric;
  }
  std::vector<ArtifactRootReference> outputs;
  if (config->generationIntent() !=
      StructuredScheduleGenerationIntent::RequireLogicalThreadDomain)
    outputs = inputBindings[StructuredProgramsInput].artifacts;
  const std::optional<std::uint64_t> maximumOutputs =
      invocationView.maximumOutputArtifacts(CandidateGeneratorOutputSlotRef(0));
  bool truncated = false;
  bool cancelled = false;
  if (maximumOutputs && outputs.size() > *maximumOutputs) {
    outputs.erase(outputs.begin() + static_cast<std::size_t>(*maximumOutputs),
                  outputs.end());
    truncated = true;
  }
  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      seenOutputs(outputs.begin(), outputs.end(), &artifactRootReferenceLess);
  std::vector<CandidateGeneratorLineageEdge> lineageEdges;
  std::uint64_t inspectedLoopScopes = 0;
  std::uint64_t inspectedDecisionCoordinates = 0;
  std::uint64_t inspectedPolyhedralDependenceQueries = 0;
  std::uint64_t generatedProposalCount = 0;
  std::uint64_t selectedProposalCount = 0;
  std::uint64_t materializationAttempts = 0;
  std::uint64_t scopRefusalCount = 0;
  std::uint64_t logicalDomainDecisionCount = 0;
  std::uint64_t ownedLogicalDomainDecisionCount = 0;
  std::uint64_t materializedLogicalDomainCount = 0;
  std::uint64_t nonFinalizableLogicalDomainCount = 0;
  std::uint64_t exactFabricRejectedLogicalDomainCount = 0;
  bool stopGeneration = truncated;
  bool proofIncomplete = false;
  const auto recordScopRefusal = [&](const frontend::StructuredEntityRef &loop,
                                     frontend::StructuredScopRefusalKind kind) {
    ++scopRefusalCount;
    proofIncomplete |= kind == frontend::StructuredScopRefusalKind::
                                   ProviderScheduleBudgetExhausted ||
                       kind == frontend::StructuredScopRefusalKind::
                                   PolyhedralMaterializationUnavailable;
    mapping_debug::emit(
        mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "structured_scop_refusal";
          fields["loop_ordinal"] = loop.ordinal;
          fields["refusal_kind"] = static_cast<std::uint64_t>(kind);
        });
  };
  for (const ArtifactRootReference &reference :
       inputBindings[StructuredProgramsInput].artifacts) {
    if (stopGeneration)
      break;
    if (invocationView.stopRequested()) {
      cancelled = true;
      break;
    }
    auto parent = frontend::importStructuredProgram(reference, store);
    if (!parent)
      return parent.takeError();
    // A logical domain may have been introduced by an earlier provider and
    // therefore need not carry a newly owned Spatial region. Preserve that
    // exact candidate before applying the region filter used for fresh
    // schedule decisions.
    if (config->generationIntent() ==
            StructuredScheduleGenerationIntent::RequireLogicalThreadDomain &&
        invocation) {
      auto hasLogicalDomain =
          invocation->selectedCandidateHasLogicalThreadDomain(reference);
      if (!hasLogicalDomain)
        return hasLogicalDomain.takeError();
      if (*hasLogicalDomain) {
        if (maximumOutputs && outputs.size() == *maximumOutputs) {
          truncated = true;
          stopGeneration = true;
        } else {
          if (seenOutputs.insert(reference).second)
            outputs.push_back(reference);
          ++ownedLogicalDomainDecisionCount;
          ++materializedLogicalDomainCount;
        }
        continue;
      }
    }
    std::optional<frontend::StructuredEntityRef> trackedSpatialRegion;
    std::vector<frontend::StructuredOperationSourceProvenance> sourceProvenance;
    if (invocation) {
      auto tracked =
          detail::StructuredOwnershipInvocationAccess::ownedSpatialRegion(
              *invocation, reference);
      if (!tracked)
        return tracked.takeError();
      trackedSpatialRegion = *tracked;
      auto provenance =
          detail::StructuredOwnershipInvocationAccess::sourceProvenance(
              *invocation, reference);
      if (!provenance)
        return provenance.takeError();
      sourceProvenance.assign(provenance->begin(), provenance->end());
    }
    if (config->generationIntent() ==
            StructuredScheduleGenerationIntent::RequireLogicalThreadDomain &&
        !trackedSpatialRegion)
      continue;
    const std::optional<frontend::StructuredEntityRef> schedulingScope =
        config->generationIntent() ==
                StructuredScheduleGenerationIntent::RequireLogicalThreadDomain
            ? trackedSpatialRegion
            : std::nullopt;
    auto decisions = frontend::enumerateStructuredScheduleDecisions(
        *parent, *exactFabric, config->scopeExpansionLimit(), schedulingScope);
    if (!decisions)
      return decisions.takeError();
    if (decisions->inspectedLoopScopes >
        std::numeric_limits<std::uint64_t>::max() - inspectedLoopScopes)
      return invalid("loop-scope accounting overflows u64");
    inspectedLoopScopes += decisions->inspectedLoopScopes;
    for (const frontend::StructuredScopRefusal &refusal : decisions->refusals)
      recordScopRefusal(refusal.loop, refusal.kind);
    if (decisions->inspectedDecisionCoordinates >
        std::numeric_limits<std::uint64_t>::max() -
            inspectedDecisionCoordinates)
      return invalid("schedule-coordinate accounting overflows u64");
    inspectedDecisionCoordinates += decisions->inspectedDecisionCoordinates;
    if (decisions->inspectedPolyhedralDependenceQueries >
        std::numeric_limits<std::uint64_t>::max() -
            inspectedPolyhedralDependenceQueries)
      return invalid("polyhedral dependence-query accounting overflows u64");
    inspectedPolyhedralDependenceQueries +=
        decisions->inspectedPolyhedralDependenceQueries;
    for (const frontend::StructuredScheduleProposal &proposal :
         decisions->proposals) {
      if (invocationView.stopRequested()) {
        cancelled = true;
        stopGeneration = true;
        break;
      }
      const frontend::StructuredScheduleDecision &decision =
          proposal.decision();
      const bool producesLogicalThreadDomain =
          decision.kind ==
              frontend::StructuredScheduleDecisionKind::Parallelize ||
          decision.kind ==
              frontend::StructuredScheduleDecisionKind::ParallelizeNest;
      logicalDomainDecisionCount += producesLogicalThreadDomain ? 1 : 0;
      if ((config->generationIntent() ==
               StructuredScheduleGenerationIntent::RequireLogicalThreadDomain &&
           !producesLogicalThreadDomain) ||
          (config->generationIntent() ==
               StructuredScheduleGenerationIntent::ForbidLogicalThreadDomain &&
           producesLogicalThreadDomain))
        continue;
      if (config->generationIntent() ==
          StructuredScheduleGenerationIntent::RequireLogicalThreadDomain)
        ++ownedLogicalDomainDecisionCount;
      if (generatedProposalCount == std::numeric_limits<std::uint64_t>::max())
        return invalid("generated proposal accounting overflows u64");
      ++generatedProposalCount;
      if ((maximumOutputs && outputs.size() == *maximumOutputs) ||
          (config->maximumMaterializationAttempts() &&
           materializationAttempts ==
               *config->maximumMaterializationAttempts())) {
        truncated = true;
        stopGeneration = true;
        break;
      }
      ++selectedProposalCount;
      ++materializationAttempts;
      auto child = frontend::materializeStructuredScheduleProposal(
          *parent, proposal, *exactFabric, trackedSpatialRegion,
          sourceProvenance);
      if (!child) {
        bool rejected = false;
        llvm::Error unhandled = llvm::handleErrors(
            child.takeError(),
            [&](const frontend::StructuredScheduleProposalRefusal &error) {
              rejected = true;
              recordScopRefusal(error.loop(), error.kind());
            },
            [&](const frontend::SpatialOwnershipCandidateRejection &error) {
              rejected = true;
              if (producesLogicalThreadDomain)
                switch (error.kind()) {
                case frontend::SpatialOwnershipCandidateRejectionKind::
                    NonFinalizable:
                  ++nonFinalizableLogicalDomainCount;
                  break;
                case frontend::SpatialOwnershipCandidateRejectionKind::
                    ExactFabricInadmissible:
                  ++exactFabricRejectedLogicalDomainCount;
                  break;
                }
            });
        if (unhandled)
          return std::move(unhandled);
        if (!rejected)
          return invalid("schedule candidate failed without a classified "
                         "outcome");
        continue;
      }
      if (child->structuredProgram.identity() == parent->identity()) {
        if (child->structuredProgram.canonicalBytes().bytes() !=
            parent->canonicalBytes().bytes())
          return invalid("schedule materialization produced an identity "
                         "collision with changed canonical bytes");
        continue;
      }
      const ArtifactRootReference childReference{
          frontend::structuredProgramArtifactSchema.identity.str(),
          frontend::structuredProgramArtifactSchema.version,
          child->structuredProgram.identity()};
      if (seenOutputs.find(childReference) != seenOutputs.end())
        continue;
      auto published =
          frontend::publishStructuredProgram(child->structuredProgram, store);
      if (!published)
        return published.takeError();
      if (*published != childReference)
        return invalid("published schedule child changed its exact reference");
      if (invocation)
        if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
                recordScheduleCandidate(*invocation, reference, *published,
                                        decision, std::move(*child), store))
          return std::move(error);
      auto ownerPayload = frontend::encodeStructuredScheduleDecision(decision);
      if (!ownerPayload)
        return ownerPayload.takeError();
      lineageEdges.push_back(CandidateGeneratorLineageEdge{
          CandidateGeneratorLineageEdgeKind::CandidateDecision,
          CandidateGeneratorOutputSlotRef(0),
          *published,
          {reference, exactFabric->reference()},
          std::move(*ownerPayload)});
      seenOutputs.insert(*published);
      outputs.push_back(std::move(*published));
      materializedLogicalDomainCount += producesLogicalThreadDomain ? 1 : 0;
    }
  }
  if (config->generationIntent() ==
      StructuredScheduleGenerationIntent::RequireLogicalThreadDomain)
    mapping_debug::emit(
        mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "structured_schedule_generation";
          fields["logical_domain_decision_count"] = logicalDomainDecisionCount;
          fields["owned_logical_domain_decision_count"] =
              ownedLogicalDomainDecisionCount;
          fields["materialized_logical_domain_count"] =
              materializedLogicalDomainCount;
          fields["non_finalizable_logical_domain_count"] =
              nonFinalizableLogicalDomainCount;
          fields["exact_fabric_rejected_logical_domain_count"] =
              exactFabricRejectedLogicalDomainCount;
        });
  if (scopRefusalCount != 0)
    mapping_debug::emit(
        mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "structured_scop_refusal_summary";
          fields["refusal_count"] = scopRefusalCount;
        });
  std::vector<CandidateGeneratorOutputBinding> outputBindings = {
      {CandidateGeneratorOutputSlotRef(0), std::move(outputs)}};
  CandidateGeneratorProviderOutcome outcome =
      cancelled
          ? CandidateGeneratorProviderOutcome{IncompleteCandidateGeneratorResult{
                CandidateGeneratorIncompleteReason::CancelledOrTimeout,
                std::move(outputBindings), std::move(lineageEdges)}}
      : truncated
          ? CandidateGeneratorProviderOutcome{IncompleteCandidateGeneratorResult{
                CandidateGeneratorIncompleteReason::SemanticLimitReached,
                std::move(outputBindings), std::move(lineageEdges)}}
      : proofIncomplete
          ? CandidateGeneratorProviderOutcome{IncompleteCandidateGeneratorResult{
                CandidateGeneratorIncompleteReason::ProofNotEstablished,
                std::move(outputBindings), std::move(lineageEdges)}}
          : CandidateGeneratorProviderOutcome{CompletedCandidateGeneratorResult{
                std::move(outputBindings), std::move(lineageEdges)}};
  return CandidateGeneratorProviderResult{
      std::move(outcome),
      {{CandidateGeneratorWorkUnitRef(0), inspectedLoopScopes,
        inspectedLoopScopes},
       {CandidateGeneratorWorkUnitRef(1), selectedProposalCount,
        materializationAttempts},
       {CandidateGeneratorWorkUnitRef(2), inspectedDecisionCoordinates,
        inspectedDecisionCoordinates},
       {CandidateGeneratorWorkUnitRef(3), generatedProposalCount,
        selectedProposalCount},
       {CandidateGeneratorWorkUnitRef(4), inspectedPolyhedralDependenceQueries,
        inspectedPolyhedralDependenceQueries}}};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeScheduleProvider}};

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedStructuredScheduleGeneratorConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedStructuredScheduleGeneratorConfigView>
projectResolvedStructuredScheduleGeneratorConfigView(
    const ResolvedConfig &config, StructuredScheduleGenerationIntent intent,
    std::optional<std::uint64_t> maximumMaterializationAttempts) {
  const std::uint64_t limit = config.dse.schedule.scopeExpansionLimit;
  if (limit == 0)
    return invalid("scope expansion limit must be positive");
  if (intent > StructuredScheduleGenerationIntent::ForbidLogicalThreadDomain)
    return invalid("generation intent is unknown");
  if (maximumMaterializationAttempts && *maximumMaterializationAttempts == 0)
    return invalid("materialization attempt limit must be positive");
  std::vector<std::uint8_t> bytes =
      encodeConfig(limit, maximumMaterializationAttempts, intent);
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedStructuredScheduleGeneratorConfigView(
      limit, maximumMaterializationAttempts, intent, std::move(bytes),
      std::move(*digest));
}

llvm::Expected<ResolvedStructuredScheduleGeneratorConfigView>
adoptResolvedStructuredScheduleGeneratorConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  auto config = decodeConfig(canonicalViewBytes);
  if (!config)
    return config.takeError();
  std::vector<std::uint8_t> reencoded = encodeConfig(
      config->limit, config->maximumMaterializationAttempts, config->intent);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalViewBytes)
    return invalid("decoded config does not re-encode to the source bytes");
  return ResolvedStructuredScheduleGeneratorConfigView(
      config->limit, config->maximumMaterializationAttempts, config->intent,
      std::move(reencoded), digest);
}

const CandidateGeneratorDescriptor &
structuredScheduleCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerStructuredScheduleCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindStructuredScheduleCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabric) {
  if (llvm::Error error = registerStructuredScheduleCandidateGenerator())
    return std::move(error);
  std::vector<CandidateGeneratorInputBinding> bindings = {
      {CandidateGeneratorInputSlotRef(StructuredProgramsInput),
       structuredPrograms.vec()},
      {CandidateGeneratorInputSlotRef(FabricInput), {fabric}},
  };
  if (llvm::Error error = validateCandidateGeneratorInputBindings(
          descriptor.reference(), bindings))
    return std::move(error);
  return bindings;
}

llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveStructuredScheduleCandidateGeneratorBinding(
    const ResolvedStructuredScheduleGeneratorConfigView &config) {
  if (llvm::Error error = registerStructuredScheduleCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse

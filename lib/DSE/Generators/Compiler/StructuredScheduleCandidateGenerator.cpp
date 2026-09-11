#include "DSE/StructuredScheduleCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Evaluation/Models/SystemRuntimeAnalytic.h"
#include "Common/MappingDebugLog.h"
#include "Config/ResolvedConfig.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Frontend/Compilation/StructuredSchedule.h"

#include "Frontend/IR/StructuredProgramArtifact.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/Support/Error.h"

#include <algorithm>
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
  const bool acceptsAffineLoop =
      adopted->kind == frontend::StructuredScheduleDecisionKind::Vectorize ||
      adopted->kind ==
          frontend::StructuredScheduleDecisionKind::PolyhedralSchedule;
  const bool exactLoop =
      acceptsAffineLoop
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
    "loom.compiler.structured_schedule.generator.v22",
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

bool producesLogicalThreadDomain(
    const frontend::StructuredScheduleDecision &decision) {
  return decision.kind ==
             frontend::StructuredScheduleDecisionKind::Parallelize ||
         decision.kind ==
             frontend::StructuredScheduleDecisionKind::ParallelizeNest;
}

bool isTiledPolyhedralPrefix(
    const frontend::StructuredScheduleDecision &decision) {
  return decision.kind ==
             frontend::StructuredScheduleDecisionKind::PolyhedralSchedule &&
         decision.factor != 0;
}

/// Strip-mining stands in as the tiled prefix of a loop the exact SCoP
/// refused, such as a loop already carrying vector transfers; like a
/// polyhedral tile, its factor is the iterations one tile performs.
bool isStripMinedPrefix(const frontend::StructuredScheduleDecision &decision) {
  return decision.kind == frontend::StructuredScheduleDecisionKind::Tile &&
         decision.factor > 1;
}

llvm::Expected<frontend::MaterializedStructuredScheduleCandidate>
cloneMaterializedScheduleCandidate(
    const frontend::MaterializedStructuredScheduleCandidate &candidate) {
  auto program = frontend::importStructuredProgram(
      candidate.structuredProgram.identity(),
      candidate.structuredProgram.canonicalBytes());
  if (!program)
    return program.takeError();
  return frontend::MaterializedStructuredScheduleCandidate{
      std::move(*program), candidate.trackedSpatialRegion,
      candidate.transformedScheduleRoots, candidate.sourceProvenance};
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
    proofIncomplete |=
        frontend::classifyStructuredScopRefusal(kind) ==
        frontend::StructuredScopRefusalDisposition::IncompleteProof;
    mapping_debug::emit(
        mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "structured_scop_refusal";
          fields["parent"] = formatArtifactIdentityHex(loop.parent);
          fields["loop_ordinal"] = loop.ordinal;
          fields["refusal_kind"] =
              frontend::structuredScopRefusalKindSpelling(kind);
          fields["refusal_kind_ordinal"] = static_cast<std::uint64_t>(kind);
        });
  };
  const auto accountDecisionDomain =
      [&](const frontend::StructuredScheduleDecisionDomain &domain)
      -> llvm::Error {
    if (domain.inspectedLoopScopes >
        std::numeric_limits<std::uint64_t>::max() - inspectedLoopScopes)
      return invalid("loop-scope accounting overflows u64");
    inspectedLoopScopes += domain.inspectedLoopScopes;
    for (const frontend::StructuredScopRefusal &refusal : domain.refusals)
      recordScopRefusal(refusal.loop, refusal.kind);
    if (domain.inspectedDecisionCoordinates >
        std::numeric_limits<std::uint64_t>::max() -
            inspectedDecisionCoordinates)
      return invalid("schedule-coordinate accounting overflows u64");
    inspectedDecisionCoordinates += domain.inspectedDecisionCoordinates;
    if (domain.inspectedPolyhedralDependenceQueries >
        std::numeric_limits<std::uint64_t>::max() -
            inspectedPolyhedralDependenceQueries)
      return invalid("polyhedral dependence-query accounting overflows u64");
    inspectedPolyhedralDependenceQueries +=
        domain.inspectedPolyhedralDependenceQueries;
    return llvm::Error::success();
  };
  const auto accountGeneratedProposal = [&]() -> llvm::Error {
    if (generatedProposalCount == std::numeric_limits<std::uint64_t>::max())
      return invalid("generated proposal accounting overflows u64");
    ++generatedProposalCount;
    return llvm::Error::success();
  };
  const auto consumeMaterializationAttempt = [&]() {
    if ((maximumOutputs && outputs.size() == *maximumOutputs) ||
        (config->maximumMaterializationAttempts() &&
         materializationAttempts ==
             *config->maximumMaterializationAttempts())) {
      truncated = true;
      stopGeneration = true;
      return false;
    }
    ++selectedProposalCount;
    ++materializationAttempts;
    return true;
  };
  const auto materializeProposal =
      [&](const ArtifactRootReference &rejectionOwner,
          const frontend::StructuredProgramCandidate &parent,
          const frontend::StructuredScheduleProposal &proposal,
          std::optional<frontend::StructuredEntityRef> spatialRegion,
          llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
              provenance)
      -> llvm::Expected<
          std::optional<frontend::MaterializedStructuredScheduleCandidate>> {
    const frontend::StructuredScheduleDecision &decision = proposal.decision();
    const bool logical = producesLogicalThreadDomain(decision);
    auto child = frontend::materializeStructuredScheduleProposal(
        parent, proposal, *exactFabric, spatialRegion, provenance);
    if (!child) {
      bool rejected = false;
      llvm::Error unhandled = llvm::handleErrors(
          child.takeError(),
          [&](const frontend::StructuredScheduleProposalRefusal &error) {
            rejected = true;
            recordScopRefusal(error.loop(), error.kind());
          },
          [&](const frontend::SpatialOwnershipCandidateRejection &error)
              -> llvm::Error {
            rejected = true;
            if (logical)
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
            mapping_debug::emit(
                mapping_debug::Level::Detail,
                mapping_debug::Stage::DataflowLowering,
                mapping_debug::Event::DerivedContext,
                [&](llvm::json::Object &fields) {
                  fields["context_kind"] =
                      "structured_schedule_materialization_rejection";
                  fields["parent"] =
                      formatArtifactIdentityHex(parent.identity());
                  fields["loop_ordinal"] = decision.loop.ordinal;
                  fields["decision_kind"] =
                      frontend::structuredScheduleDecisionKindSpelling(
                          decision.kind);
                  fields["factor"] = decision.vector ? decision.vector->shape.front() : decision.factor;
                  fields["rejection_kind_ordinal"] =
                      static_cast<std::uint64_t>(error.kind());
                  fields["diagnostic"] = error.message();
                });
            if (invocation)
              return detail::StructuredOwnershipInvocationAccess::
                  recordFinalizationRejection(
                      *invocation, rejectionOwner,
                      {error.kind(), error.message(), error.memoryContract()});
            return llvm::Error::success();
          });
      if (unhandled)
        return std::move(unhandled);
      if (!rejected)
        return invalid(
            "schedule candidate failed without a classified outcome");
      return std::optional<frontend::MaterializedStructuredScheduleCandidate>{};
    }
    if (child->structuredProgram.identity() == parent.identity()) {
      if (child->structuredProgram.canonicalBytes().bytes() !=
          parent.canonicalBytes().bytes())
        return invalid("schedule materialization produced an identity "
                       "collision with changed canonical bytes");
      return std::optional<frontend::MaterializedStructuredScheduleCandidate>{};
    }
    return std::optional<frontend::MaterializedStructuredScheduleCandidate>(
        std::move(*child));
  };
  const auto publishDecision =
      [&](const ArtifactRootReference &parentReference,
          const frontend::StructuredScheduleDecision &decision,
          frontend::MaterializedStructuredScheduleCandidate child)
      -> llvm::Expected<ArtifactRootReference> {
    const ArtifactRootReference childReference{
        frontend::structuredProgramArtifactSchema.identity.str(),
        frontend::structuredProgramArtifactSchema.version,
        child.structuredProgram.identity()};
    auto published =
        frontend::publishStructuredProgram(child.structuredProgram, store);
    if (!published)
      return published.takeError();
    if (*published != childReference)
      return invalid("published schedule child changed its exact reference");
    if (invocation)
      if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
              recordScheduleCandidate(*invocation, parentReference, *published,
                                      decision, std::move(child), store))
        return std::move(error);
    auto ownerPayload = frontend::encodeStructuredScheduleDecision(decision);
    if (!ownerPayload)
      return ownerPayload.takeError();
    mapping_debug::emit(
        mapping_debug::Level::Summary, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::Candidate, [&](llvm::json::Object &fields) {
          fields["operation"] = "structured_schedule_candidate";
          fields["decision_kind"] =
              frontend::structuredScheduleDecisionKindSpelling(decision.kind);
          fields["factor"] = decision.vector ? decision.vector->shape.front() : decision.factor;
          fields["loop_ordinal"] = decision.loop.ordinal;
          fields["parent"] =
              formatArtifactIdentityHex(parentReference.artifact);
          fields["child"] = formatArtifactIdentityHex(published->artifact);
        });
    lineageEdges.push_back(CandidateGeneratorLineageEdge{
        CandidateGeneratorLineageEdgeKind::CandidateDecision,
        CandidateGeneratorOutputSlotRef(0),
        *published,
        {parentReference, exactFabric->reference()},
        std::move(*ownerPayload)});
    return *published;
  };
  struct ParentSchedule final {
    ArtifactRootReference reference;
    frontend::StructuredProgramCandidate program;
    std::optional<frontend::StructuredEntityRef> spatialRegion;
    std::vector<frontend::StructuredOperationSourceProvenance> provenance;
    frontend::StructuredScheduleDecisionDomain domain;
    std::array<std::vector<std::size_t>, 2> proposalOrdinals;
    /// The widest admitted vector coordinate, explored as the head of a
    /// vector, strip-mine, parallelize chain before the plain prefixes.
    std::optional<std::size_t> vectorChainOrdinal;
  };
  enum class ScheduleSearchPhase : std::size_t { Direct, TiledPrefix };
  std::vector<ParentSchedule> parents;
  auto systemRoot = fabric::requireSystemRoot(exactFabric->view());
  if (!systemRoot)
    return systemRoot.takeError();
  const std::uint64_t accCoreCount =
      std::max<std::size_t>(1, systemRoot->artifact().accCoreOccurrences().size());
  // The outstanding requests the shared memory service grants one AccCore;
  // the same platform projection the analytic runtime model reads.
  std::uint64_t memoryOutstandingRequests = 1;
  if (auto platform =
          evaluation::models::projectSystemPlatformModel(*exactFabric)) {
    memoryOutstandingRequests =
        std::max<std::uint64_t>(1, platform->accCoreOutstandingRequests);
  } else {
    // A Fabric without a shared memory service grants no overlap; the
    // smallest admitted unroll then stands in.
    llvm::consumeError(platform.takeError());
  }
  // The tiled prefixes of one decision domain: proven polyhedral tiles, and
  // strip-mining where the exact SCoP admitted no tile for the loop, such as
  // a loop already carrying vector transfers. A tiled prefix exists to become
  // one logical thread per AccCore, so the tile counts nearest the AccCore
  // count come first; among equal distances the coarser tile amortizes more
  // activation overhead.
  const auto tiledPrefixOrdinals =
      [&](const frontend::StructuredProgramCandidate &program,
          const frontend::StructuredScheduleDecisionDomain &domain) {
        std::vector<std::size_t> prefixes;
        for (std::size_t ordinal = 0; ordinal != domain.proposals.size();
             ++ordinal)
          if (isTiledPolyhedralPrefix(domain.proposals[ordinal].decision()))
            prefixes.push_back(ordinal);
        for (std::size_t ordinal = 0; ordinal != domain.proposals.size();
             ++ordinal) {
          const auto &decision = domain.proposals[ordinal].decision();
          if (!isStripMinedPrefix(decision))
            continue;
          const bool polyhedralPrefixExists = llvm::any_of(
              domain.proposals,
              [&](const frontend::StructuredScheduleProposal &proposal) {
                return proposal.decision().loop == decision.loop &&
                       isTiledPolyhedralPrefix(proposal.decision());
              });
          if (!polyhedralPrefixExists)
            prefixes.push_back(ordinal);
        }
        const auto tileDistance = [&](std::size_t ordinal) -> std::uint64_t {
          const auto &decision = domain.proposals[ordinal].decision();
          if (decision.factor == 0)
            return std::numeric_limits<std::uint64_t>::max();
          auto entity = program.view();
          if (!entity) {
            llvm::consumeError(entity.takeError());
            return std::numeric_limits<std::uint64_t>::max();
          }
          auto loop = entity->resolve(decision.loop);
          if (!loop) {
            llvm::consumeError(loop.takeError());
            return std::numeric_limits<std::uint64_t>::max();
          }
          const std::optional<std::uint64_t> trip =
              frontend::structuredLoopStaticTripCount(loop->operation);
          if (!trip)
            return std::numeric_limits<std::uint64_t>::max();
          const std::uint64_t tiles = *trip / decision.factor;
          return tiles > accCoreCount ? tiles - accCoreCount
                                      : accCoreCount - tiles;
        };
        std::stable_sort(prefixes.begin(), prefixes.end(),
                         [&](std::size_t left, std::size_t right) {
                           const std::uint64_t leftDistance =
                               tileDistance(left);
                           const std::uint64_t rightDistance =
                               tileDistance(right);
                           if (leftDistance != rightDistance)
                             return leftDistance < rightDistance;
                           return domain.proposals[left].decision().factor >
                                  domain.proposals[right].decision().factor;
                         });
        return prefixes;
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
    // Schedule decisions transform the owned Spatial carrier. Loops outside
    // the tracked region never reach the fabric, so enumerating them only
    // spends the materialization budget on proposals that cannot close.
    const std::optional<frontend::StructuredEntityRef> &schedulingScope =
        trackedSpatialRegion;
    auto decisions = frontend::enumerateStructuredScheduleDecisions(
        *parent, *exactFabric, config->scopeExpansionLimit(), schedulingScope);
    if (!decisions)
      return decisions.takeError();
    if (llvm::Error error = accountDecisionDomain(*decisions))
      return std::move(error);
    ParentSchedule schedule{reference,
                            std::move(*parent),
                            trackedSpatialRegion,
                            std::move(sourceProvenance),
                            std::move(*decisions),
                            {}};
    for (std::size_t ordinal = 0; ordinal != schedule.domain.proposals.size();
         ++ordinal) {
      const auto &decision = schedule.domain.proposals[ordinal].decision();
      const bool logical = producesLogicalThreadDomain(decision);
      logicalDomainDecisionCount += logical ? 1 : 0;
      const auto intent = config->generationIntent();
      if ((intent !=
               StructuredScheduleGenerationIntent::RequireLogicalThreadDomain ||
           logical) &&
          (intent !=
               StructuredScheduleGenerationIntent::ForbidLogicalThreadDomain ||
           !logical)) {
        schedule
            .proposalOrdinals[static_cast<std::size_t>(
                ScheduleSearchPhase::Direct)]
            .push_back(ordinal);
        if (intent ==
            StructuredScheduleGenerationIntent::RequireLogicalThreadDomain)
          ++ownedLogicalDomainDecisionCount;
      }
    }
    if (config->generationIntent() ==
        StructuredScheduleGenerationIntent::RequireLogicalThreadDomain) {
      schedule.proposalOrdinals[static_cast<std::size_t>(
          ScheduleSearchPhase::TiledPrefix)] =
          tiledPrefixOrdinals(schedule.program, schedule.domain);
      for (std::size_t ordinal = 0;
           ordinal != schedule.domain.proposals.size(); ++ordinal) {
        if (schedule.domain.proposals[ordinal].decision().kind !=
            frontend::StructuredScheduleDecisionKind::Vectorize)
          continue;
        schedule.vectorChainOrdinal = ordinal;
        break;
      }
    }
    mapping_debug::emit(
        mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "structured_schedule_parent";
          fields["parent"] = formatArtifactIdentityHex(reference.artifact);
          fields["tracked_spatial_region"] =
              static_cast<bool>(schedule.spatialRegion);
          llvm::json::Object kinds;
          for (const auto &proposal : schedule.domain.proposals) {
            const llvm::StringRef kind =
                frontend::structuredScheduleDecisionKindSpelling(
                    proposal.decision().kind);
            kinds[kind] = kinds.getInteger(kind).value_or(0) + 1;
          }
          fields["proposal_kinds"] = std::move(kinds);
          fields["refusal_count"] = schedule.domain.refusals.size();
          fields["direct_proposals"] =
              schedule
                  .proposalOrdinals[static_cast<std::size_t>(
                      ScheduleSearchPhase::Direct)]
                  .size();
          fields["tiled_prefixes"] =
              schedule
                  .proposalOrdinals[static_cast<std::size_t>(
                      ScheduleSearchPhase::TiledPrefix)]
                  .size();
          fields["vector_chain"] =
              static_cast<bool>(schedule.vectorChainOrdinal);
        });
    parents.push_back(std::move(schedule));
  }

  // Materializes one tiled prefix of `program` and every proven Parallelize
  // terminal on the transformed roots it introduces. `publishAncestors` runs
  // once before the prefix is published, so an intermediate stage reaches the
  // store only when a terminal below it materializes. Returns whether a
  // terminal was published.
  const auto exploreTiledPrefix =
      [&](const ArtifactRootReference &rejectionOwner,
          const ArtifactRootReference &reference,
          const frontend::StructuredProgramCandidate &program,
          const std::optional<frontend::StructuredEntityRef> &spatialRegion,
          llvm::ArrayRef<frontend::StructuredOperationSourceProvenance>
              provenance,
          const frontend::StructuredScheduleProposal &proposal,
          llvm::function_ref<llvm::Error()> publishAncestors)
      -> llvm::Expected<bool> {
    const frontend::StructuredScheduleDecision &decision = proposal.decision();
    if (llvm::Error error = accountGeneratedProposal())
      return std::move(error);
    if (!consumeMaterializationAttempt())
      return false;
    auto materialized = materializeProposal(rejectionOwner, program, proposal,
                                            spatialRegion, provenance);
    if (!materialized)
      return materialized.takeError();
    if (!*materialized)
      return false;
    frontend::MaterializedStructuredScheduleCandidate &prefix = **materialized;
    if (prefix.transformedScheduleRoots.empty())
      return invalid("tiled prefix lost its transformed schedule roots");
    const ArtifactRootReference prefixReference{
        frontend::structuredProgramArtifactSchema.identity.str(),
        frontend::structuredProgramArtifactSchema.version,
        prefix.structuredProgram.identity()};
    bool prefixPublished = false;
    bool terminalPublished = false;
    for (const frontend::StructuredEntityRef &transformedRoot :
         prefix.transformedScheduleRoots) {
      if (stopGeneration)
        break;
      if (invocationView.stopRequested()) {
        cancelled = true;
        stopGeneration = true;
        break;
      }
      auto terminalDomain = frontend::enumerateStructuredScheduleDecisions(
          prefix.structuredProgram, *exactFabric,
          config->scopeExpansionLimit(), transformedRoot);
      if (!terminalDomain)
        return terminalDomain.takeError();
      if (llvm::Error error = accountDecisionDomain(*terminalDomain))
        return std::move(error);
      const bool terminalAvailable = llvm::any_of(
          terminalDomain->proposals,
          [&](const frontend::StructuredScheduleProposal &terminal) {
            return terminal.decision().kind ==
                       frontend::StructuredScheduleDecisionKind::Parallelize &&
                   terminal.decision().loop == transformedRoot;
          });
      if (!terminalAvailable)
        mapping_debug::emit(
            mapping_debug::Level::Detail,
            mapping_debug::Stage::DataflowLowering,
            mapping_debug::Event::DerivedContext,
            [&](llvm::json::Object &fields) {
              fields["context_kind"] = "structured_schedule_prefix_terminal";
              fields["prefix"] =
                  formatArtifactIdentityHex(prefixReference.artifact);
              fields["decision_kind"] =
                  frontend::structuredScheduleDecisionKindSpelling(
                      decision.kind);
              fields["factor"] = decision.factor;
              fields["terminal_proposals"] = terminalDomain->proposals.size();
              if (auto view = prefix.structuredProgram.view()) {
                if (auto root = view->resolve(transformedRoot)) {
                  std::string text;
                  llvm::raw_string_ostream stream(text);
                  root->operation->print(stream);
                  fields["transformed_root"] = std::move(text);
                } else {
                  llvm::consumeError(root.takeError());
                }
              } else {
                llvm::consumeError(view.takeError());
              }
            });
      for (const frontend::StructuredScheduleProposal &terminalProposal :
           terminalDomain->proposals) {
        if (invocationView.stopRequested()) {
          cancelled = true;
          stopGeneration = true;
          break;
        }
        const frontend::StructuredScheduleDecision &terminalDecision =
            terminalProposal.decision();
        const bool logical = producesLogicalThreadDomain(terminalDecision);
        logicalDomainDecisionCount += logical ? 1 : 0;
        if (terminalDecision.kind !=
                frontend::StructuredScheduleDecisionKind::Parallelize ||
            terminalDecision.loop != transformedRoot)
          continue;
        ++ownedLogicalDomainDecisionCount;
        if (llvm::Error error = accountGeneratedProposal())
          return std::move(error);
        if (!consumeMaterializationAttempt())
          break;
        auto materializedTerminal = materializeProposal(
            rejectionOwner, prefix.structuredProgram, terminalProposal,
            prefix.trackedSpatialRegion, prefix.sourceProvenance);
        if (!materializedTerminal)
          return materializedTerminal.takeError();
        if (!*materializedTerminal)
          continue;
        const ArtifactRootReference terminalReference{
            frontend::structuredProgramArtifactSchema.identity.str(),
            frontend::structuredProgramArtifactSchema.version,
            (*materializedTerminal)->structuredProgram.identity()};
        if (seenOutputs.find(terminalReference) != seenOutputs.end())
          continue;
        if (!prefixPublished) {
          if (llvm::Error error = publishAncestors())
            return std::move(error);
          auto prefixClone = cloneMaterializedScheduleCandidate(prefix);
          if (!prefixClone)
            return prefixClone.takeError();
          auto publishedPrefix =
              publishDecision(reference, decision, std::move(*prefixClone));
          if (!publishedPrefix)
            return publishedPrefix.takeError();
          if (*publishedPrefix != prefixReference)
            return invalid("published tiled prefix changed identity");
          prefixPublished = true;
        }
        mapping_debug::emit(
            mapping_debug::Level::Detail,
            mapping_debug::Stage::DataflowLowering,
            mapping_debug::Event::DerivedContext,
            [&](llvm::json::Object &fields) {
              fields["context_kind"] = "structured_schedule_terminal";
              fields["prefix"] =
                  formatArtifactIdentityHex(prefixReference.artifact);
              fields["terminal"] =
                  formatArtifactIdentityHex(terminalReference.artifact);
              const auto &terminal = **materializedTerminal;
              if (auto view = terminal.structuredProgram.view()) {
                if (terminal.trackedSpatialRegion) {
                  if (auto region =
                          view->resolve(*terminal.trackedSpatialRegion)) {
                    std::string text;
                    llvm::raw_string_ostream stream(text);
                    region->operation->print(stream);
                    fields["spatial_region"] = std::move(text);
                  } else {
                    llvm::consumeError(region.takeError());
                  }
                }
              } else {
                llvm::consumeError(view.takeError());
              }
            });
        auto publishedTerminal =
            publishDecision(prefixReference, terminalDecision,
                            std::move(**materializedTerminal));
        if (!publishedTerminal)
          return publishedTerminal.takeError();
        seenOutputs.insert(*publishedTerminal);
        outputs.push_back(std::move(*publishedTerminal));
        ++materializedLogicalDomainCount;
        terminalPublished = true;
      }
    }
    return terminalPublished;
  };

  // The exact SCoP cannot re-vectorize a symbolic tile, so the vector shape
  // comes first and the tile is strip-mined below it: the widest admitted
  // shape moves a tile's elements in the fewest memory transactions. An
  // optional unroll stage replicates the vector body so more memory actors
  // keep requests in flight. Every stage is published only when a terminal
  // below it materializes.
  struct ChainStage final {
    ArtifactRootReference reference;
    frontend::StructuredScheduleDecision decision;
    const frontend::MaterializedStructuredScheduleCandidate *candidate;
    bool published = false;
  };
  const auto exploreVectorChain =
      [&](const ParentSchedule &parent,
          const frontend::StructuredScheduleProposal &vectorProposal,
          bool unrollStage) -> llvm::Expected<bool> {
    if (llvm::Error error = accountGeneratedProposal())
      return std::move(error);
    if (!consumeMaterializationAttempt())
      return false;
    auto materialized =
        materializeProposal(parent.reference, parent.program, vectorProposal,
                            parent.spatialRegion, parent.provenance);
    if (!materialized)
      return materialized.takeError();
    if (!*materialized)
      return false;
    const auto referenceOf =
        [](const frontend::MaterializedStructuredScheduleCandidate &stage) {
          return ArtifactRootReference{
              frontend::structuredProgramArtifactSchema.identity.str(),
              frontend::structuredProgramArtifactSchema.version,
              stage.structuredProgram.identity()};
        };
    const auto enumerateStage =
        [&](const frontend::MaterializedStructuredScheduleCandidate &stage) {
          return frontend::enumerateStructuredScheduleDecisions(
              stage.structuredProgram, *exactFabric,
              config->scopeExpansionLimit(), stage.trackedSpatialRegion);
        };
    std::vector<ChainStage> stages;
    stages.push_back({referenceOf(**materialized), vectorProposal.decision(),
                      &**materialized});
    auto stageDomain = enumerateStage(**materialized);
    if (!stageDomain)
      return stageDomain.takeError();
    if (llvm::Error error = accountDecisionDomain(*stageDomain))
      return std::move(error);
    std::optional<frontend::MaterializedStructuredScheduleCandidate> unrolled;
    if (unrollStage) {
      // A memory actor holds one request in flight, so copies beyond the
      // service's outstanding slots add actors without overlap. Take the
      // smallest admitted unroll that fills those slots, or the widest
      // admitted one when none reaches them.
      const frontend::StructuredScheduleProposal *widest = nullptr;
      for (const auto &proposal : stageDomain->proposals) {
        if (proposal.decision().kind !=
            frontend::StructuredScheduleDecisionKind::Unroll)
          continue;
        auto view = (**materialized).structuredProgram.view();
        if (!view)
          return view.takeError();
        auto loop = view->resolve(proposal.decision().loop);
        if (!loop)
          return loop.takeError();
        std::uint64_t memoryActors = 0;
        loop->operation->walk([&](mlir::Operation *operation) {
          auto schema = dataflow::operationSchemaOf(operation);
          memoryActors += schema && dataflow::actorKind(*schema) ==
                                        dataflow::CanonicalDataflowActorKind::Memory
                              ? 1
                              : 0;
        });
        const std::uint64_t neededCopies =
            memoryActors == 0
                ? 1
                : (memoryOutstandingRequests + memoryActors - 1) / memoryActors;
        const std::uint64_t factor = proposal.decision().factor;
        const bool fills = factor >= neededCopies;
        if (!widest) {
          widest = &proposal;
          continue;
        }
        const std::uint64_t current = widest->decision().factor;
        const bool currentFills = current >= neededCopies;
        if ((fills && (!currentFills || factor < current)) ||
            (!fills && !currentFills && factor > current))
          widest = &proposal;
      }
      if (widest) {
        if (llvm::Error error = accountGeneratedProposal())
          return std::move(error);
        if (!consumeMaterializationAttempt())
          return false;
        const auto &vectorStage = **materialized;
        auto materializedUnroll = materializeProposal(
            parent.reference, vectorStage.structuredProgram, *widest,
            vectorStage.trackedSpatialRegion, vectorStage.sourceProvenance);
        if (!materializedUnroll)
          return materializedUnroll.takeError();
        if (*materializedUnroll) {
          unrolled.emplace(std::move(**materializedUnroll));
          stages.push_back(
              {referenceOf(*unrolled), widest->decision(), &*unrolled});
          stageDomain = enumerateStage(*unrolled);
          if (!stageDomain)
            return stageDomain.takeError();
          if (llvm::Error error = accountDecisionDomain(*stageDomain))
            return std::move(error);
        }
      }
    }
    const frontend::MaterializedStructuredScheduleCandidate &leaf =
        *stages.back().candidate;
    const ArtifactRootReference leafReference = stages.back().reference;
    const std::vector<std::size_t> stagePrefixes =
        tiledPrefixOrdinals(leaf.structuredProgram, *stageDomain);
    mapping_debug::emit(
        mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
        mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
          fields["context_kind"] = "structured_schedule_vector_stage";
          fields["parent"] = formatArtifactIdentityHex(parent.reference.artifact);
          fields["stage"] = formatArtifactIdentityHex(leafReference.artifact);
          fields["stage_count"] = stages.size();
          llvm::json::Object kinds;
          for (const auto &proposal : stageDomain->proposals) {
            const llvm::StringRef kind =
                frontend::structuredScheduleDecisionKindSpelling(
                    proposal.decision().kind);
            kinds[kind] = kinds.getInteger(kind).value_or(0) + 1;
          }
          fields["proposal_kinds"] = std::move(kinds);
          fields["tiled_prefixes"] = stagePrefixes.size();
          if (auto view = leaf.structuredProgram.view()) {
            if (leaf.trackedSpatialRegion) {
              if (auto region = view->resolve(*leaf.trackedSpatialRegion)) {
                std::string text;
                llvm::raw_string_ostream stream(text);
                region->operation->print(stream);
                fields["spatial_region"] = std::move(text);
              } else {
                llvm::consumeError(region.takeError());
              }
            }
          } else {
            llvm::consumeError(view.takeError());
          }
        });
    const auto publishStages = [&]() -> llvm::Error {
      ArtifactRootReference lineageParent = parent.reference;
      for (ChainStage &stage : stages) {
        if (!stage.published) {
          auto clone = cloneMaterializedScheduleCandidate(*stage.candidate);
          if (!clone)
            return clone.takeError();
          auto published =
              publishDecision(lineageParent, stage.decision, std::move(*clone));
          if (!published)
            return published.takeError();
          if (*published != stage.reference)
            return invalid("published chain stage changed identity");
          stage.published = true;
        }
        lineageParent = stage.reference;
      }
      return llvm::Error::success();
    };
    for (std::size_t ordinal : stagePrefixes) {
      if (stopGeneration)
        break;
      auto published = exploreTiledPrefix(
          parent.reference, leafReference, leaf.structuredProgram,
          leaf.trackedSpatialRegion, leaf.sourceProvenance,
          stageDomain->proposals[ordinal], publishStages);
      if (!published)
        return published.takeError();
      if (*published)
        return true;
    }
    return false;
  };

  // A logical-domain search first explores the vector chain, then coarse
  // proven tiles to amortize activation overhead. Parents share each round; a
  // prefix and its independent terminal proof remain one step, with every
  // materialization charged.
  for (ScheduleSearchPhase phase :
       {ScheduleSearchPhase::TiledPrefix, ScheduleSearchPhase::Direct}) {
    const auto chainCount = [&](const ParentSchedule &parent) -> std::size_t {
      return phase == ScheduleSearchPhase::TiledPrefix &&
                     parent.vectorChainOrdinal
                 ? 1
                 : 0;
    };
    std::size_t rounds = 0;
    for (const ParentSchedule &parent : parents)
      rounds = std::max(
          rounds,
          chainCount(parent) +
              parent.proposalOrdinals[static_cast<std::size_t>(phase)].size());
    for (std::size_t round = 0; round != rounds && !stopGeneration; ++round) {
      for (auto [parentIndex, parent] : llvm::enumerate(parents)) {
        if (stopGeneration)
          break;
        if (invocationView.stopRequested()) {
          cancelled = true;
          stopGeneration = true;
          break;
        }
        const auto &ordinals =
            parent.proposalOrdinals[static_cast<std::size_t>(phase)];
        const std::size_t chains = chainCount(parent);
        if (round >= chains + ordinals.size())
          continue;
        if (round < chains) {
          // Alternate the unroll stage across parents so both chain shapes
          // are explored within one grant.
          auto published = exploreVectorChain(
              parent, parent.domain.proposals[*parent.vectorChainOrdinal],
              parentIndex % 2 == 1);
          if (!published)
            return published.takeError();
          continue;
        }
        const auto &reference = parent.reference;
        const auto &proposal = parent.domain.proposals[ordinals[round - chains]];
        const auto &decision = proposal.decision();
        if (phase == ScheduleSearchPhase::TiledPrefix) {
          auto published = exploreTiledPrefix(
              reference, reference, parent.program, parent.spatialRegion,
              parent.provenance, proposal,
              [] { return llvm::Error::success(); });
          if (!published)
            return published.takeError();
          continue;
        }
        if (llvm::Error error = accountGeneratedProposal())
          return std::move(error);
        if (!consumeMaterializationAttempt())
          break;
        auto materialized =
            materializeProposal(reference, parent.program, proposal,
                                parent.spatialRegion, parent.provenance);
        if (!materialized)
          return materialized.takeError();
        if (!*materialized)
          continue;
        const ArtifactRootReference childReference{
            frontend::structuredProgramArtifactSchema.identity.str(),
            frontend::structuredProgramArtifactSchema.version,
            (*materialized)->structuredProgram.identity()};
        if (seenOutputs.find(childReference) != seenOutputs.end())
          continue;
        auto published =
            publishDecision(reference, decision, std::move(**materialized));
        if (!published)
          return published.takeError();
        seenOutputs.insert(*published);
        outputs.push_back(std::move(*published));
        materializedLogicalDomainCount +=
            producesLogicalThreadDomain(decision) ? 1 : 0;
      }
    }
  }
  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::DerivedContext, [&](llvm::json::Object &fields) {
        fields["context_kind"] = "structured_schedule_generation";
        fields["materialization_attempts"] = materializationAttempts;
        fields["truncated"] = truncated;
        fields["input_count"] =
            inputBindings[StructuredProgramsInput].artifacts.size();
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

#include "DSE/StructuredSpecialMathAccuracyCandidateGenerator.h"
#include "DSE/StructuredOwnershipInvocationInternal.h"
#include "Dataflow/IR/OperationSchema.h"

#include "Common/ArtifactStore.h"
#include "Common/MappingDebugLog.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/OwnershipCandidateGenerator.h"
#include "Frontend/Compilation/StructuredSpecialMathAccuracy.h"
#include "Frontend/IR/LoomOps.h"
#include "Frontend/IR/StructuredProgramArtifact.h"
#include "Frontend/Lowering/GraphParallelLowering.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.structured_special_math_accuracy_generator.config.2.0";

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

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 2> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "accuracy_decision"},
    {CandidateGeneratorWorkUnitRef(1), "mechanical_accuracy_closure"},
}};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "structured_special_math_accuracy_generator_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

std::vector<std::uint8_t>
encodeConfig(std::optional<std::uint64_t> maximumMaterializationAttempts) {
  const std::uint64_t value = maximumMaterializationAttempts.value_or(0);
  std::vector<std::uint8_t> bytes;
  bytes.reserve(8);
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
  return bytes;
}

llvm::Expected<std::optional<std::uint64_t>>
decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != 8)
    return invalid("config must contain one u64 materialization bound");
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes)
    value = (value << 8) | byte;
  return value == 0 ? std::optional<std::uint64_t>{}
                    : std::optional<std::uint64_t>{value};
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Error validateDecisionPayload(
    llvm::ArrayRef<std::uint8_t> bytes, const ArtifactRootReference &,
    llvm::ArrayRef<ArtifactRootReference> parents, const ArtifactStore &store) {
  auto adopted = frontend::adoptStructuredSpecialMathAccuracyDecision(bytes);
  if (!adopted)
    return adopted.takeError();
  if (parents.size() != 1 ||
      parents.front().schemaIdentity !=
          frontend::structuredProgramArtifactSchema.identity ||
      parents.front().schemaVersion !=
          frontend::structuredProgramArtifactSchema.version)
    return invalid(
        "accuracy decision does not have one exact Structured parent");
  auto parent = frontend::importStructuredProgram(parents.front(), store);
  if (!parent)
    return parent.takeError();
  auto domain =
      frontend::enumerateStructuredSpecialMathAccuracyDecisions(*parent);
  if (!domain)
    return domain.takeError();
  if (!llvm::is_contained(*domain, *adopted))
    return invalid("accuracy decision is outside the exact parent domain");
  return llvm::Error::success();
}

const CandidateGeneratorOwnerLineagePayloadContract lineageContract{
    frontend::structuredSpecialMathAccuracyDecisionSchemaBytes(),
    validateDecisionPayload};

const CandidateGeneratorDescriptor descriptor{
    structuredSpecialMathAccuracyCandidateGeneratorKind,
    "compiler.structured_special_math_accuracy",
    "loom.compiler.structured_special_math_accuracy.generator.v2",
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

llvm::Expected<frontend::MaterializedStructuredOwnershipCandidate>
cloneCandidate(
    const frontend::MaterializedStructuredOwnershipCandidate &candidate) {
  auto clone = frontend::importStructuredProgram(
      candidate.structuredProgram.identity(),
      candidate.structuredProgram.canonicalBytes());
  if (!clone)
    return clone.takeError();
  return frontend::MaterializedStructuredOwnershipCandidate{
      std::move(*clone), candidate.ownedSpatialRegion,
      candidate.blockActivityLineage, candidate.sourceProvenance};
}

llvm::Expected<frontend::MaterializedStructuredOwnershipCandidate>
cloneRoot(StructuredOwnershipInvocation *invocation,
          const ArtifactRootReference &reference, const ArtifactStore &store) {
  if (invocation)
    return detail::StructuredOwnershipInvocationAccess::
        clonePreClosureCandidate(*invocation, reference);
  auto imported = frontend::importStructuredProgram(reference, store);
  if (!imported)
    return imported.takeError();
  return frontend::MaterializedStructuredOwnershipCandidate{
      std::move(*imported), std::nullopt, {}, {}};
}

llvm::Expected<std::optional<frontend::MaterializedOwnershipCandidate>>
finalizeCandidate(
    frontend::MaterializedStructuredOwnershipCandidate candidate,
    const ArtifactRootReference &generatorInput,
    StructuredOwnershipInvocation *invocation,
    const fabric::FinalizedFabricRoot &fabric,
    const lowering::CanonicalDataflowLoweringOptions &loweringOptions) {
  // A refused leaf leaves no candidate behind; the bound invocation retains
  // the typed refusal against the Ownership coordinate of the generator input
  // the leaf descends from, so the application decision can state it. The
  // leaf itself is not anchored: its mechanical and decision edges reach the
  // invocation only on admission, so its own lineage cannot be resolved here.
  const auto retainRejection =
      [&](StructuredOwnershipCandidateRejectionRecord rejection)
      -> llvm::Error {
    if (!invocation)
      return llvm::Error::success();
    return detail::StructuredOwnershipInvocationAccess::
        recordFinalizationRejection(*invocation, generatorInput,
                                    std::move(rejection));
  };
  if (candidate.ownedSpatialRegion) {
    auto view = candidate.structuredProgram.view();
    if (!view)
      return view.takeError();
    auto owner = view->resolve(*candidate.ownedSpatialRegion);
    if (!owner)
      return owner.takeError();
    if (!llvm::isa_and_nonnull<loom::SpatialRegionOp>(owner->operation))
      return invalid("candidate Spatial owner does not resolve to its carrier");
    if (std::optional<std::string> rejection =
            lowering::explainSpatialCarrierParallelRejection(
                owner->operation)) {
      if (llvm::Error error = retainRejection(
              {frontend::SpatialOwnershipCandidateRejectionKind::NonFinalizable,
               std::move(*rejection), std::nullopt}))
        return std::move(error);
      return std::optional<frontend::MaterializedOwnershipCandidate>{};
    }
  }

  auto finalized = frontend::finalizeSpatialOwnershipCandidate(
      std::move(candidate), fabric, loweringOptions);
  if (finalized)
    return std::optional<frontend::MaterializedOwnershipCandidate>(
        std::move(*finalized));

  std::optional<frontend::SpatialOwnershipCandidateRejectionKind> rejectionKind;
  std::string rejectionMessage;
  std::optional<dataflow::MemoryContractClass> rejectionContract;
  llvm::Error unhandled = llvm::handleErrors(
      finalized.takeError(),
      [&](const frontend::SpatialOwnershipCandidateRejection &rejection) {
        rejectionKind = rejection.kind();
        rejectionMessage = rejection.message();
        rejectionContract = rejection.memoryContract();
      });
  if (unhandled)
    return std::move(unhandled);
  if (!rejectionKind)
    return invalid("candidate failed without a classified rejection");
  if (llvm::Error error = retainRejection(
          {*rejectionKind, rejectionMessage, rejectionContract}))
    return std::move(error);
  mapping_debug::emit(
      mapping_debug::Level::Detail, mapping_debug::Stage::DataflowLowering,
      mapping_debug::Event::MappingFailure, [&](llvm::json::Object &fields) {
        fields["failure_scope"] = "structured_candidate_finalization";
        fields["closure_status"] = "proven_infeasible";
        fields["rejection_kind"] =
            *rejectionKind == frontend::SpatialOwnershipCandidateRejectionKind::
                                  NonFinalizable
                ? "non_finalizable"
                : "exact_fabric_inadmissible";
        fields["diagnostic"] = rejectionMessage;
        if (rejectionContract)
          fields["memory_contract"] =
              dataflow::memoryContractClassSpelling(*rejectionContract);
      });
  return std::optional<frontend::MaterializedOwnershipCandidate>{};
}

struct PendingAccuracyLineage final {
  ArtifactRootReference parent;
  ArtifactRootReference child;
  std::optional<frontend::StructuredSpecialMathAccuracyDecision> decision;
};

struct AdmittedAccuracyLeaf final {
  ArtifactRootReference reference;
  std::vector<PendingAccuracyLineage> lineage;
  frontend::MaterializedOwnershipCandidate candidate;
};

struct MechanicalAccuracyClosure final {
  bool changed = false;
  bool complete = true;
};

llvm::Expected<MechanicalAccuracyClosure> closeMechanicalAccuracy(
    frontend::MaterializedStructuredOwnershipCandidate &candidate,
    std::uint64_t maximumMaterializations,
    std::uint64_t consumedChoiceAttempts,
    std::uint64_t &plannedMechanicalClosures,
    std::uint64_t &consumedMechanicalClosures) {
  MechanicalAccuracyClosure result;
  while (true) {
    auto domain = frontend::enumerateStructuredSpecialMathAccuracyDecisions(
        candidate.structuredProgram);
    if (!domain)
      return domain.takeError();
    if (domain->size() != 1)
      return result;
    if (plannedMechanicalClosures ==
        std::numeric_limits<std::uint64_t>::max())
      return invalid("planned mechanical-closure accounting overflows u64");
    ++plannedMechanicalClosures;
    if (consumedChoiceAttempts > maximumMaterializations ||
        consumedMechanicalClosures >
            maximumMaterializations - consumedChoiceAttempts) {
      return invalid("accuracy materialization accounting exceeds its bound");
    }
    if (consumedChoiceAttempts + consumedMechanicalClosures ==
        maximumMaterializations) {
      result.complete = false;
      return result;
    }
    ++consumedMechanicalClosures;
    auto child = frontend::materializeStructuredSpecialMathAccuracyDecision(
        std::move(candidate), domain->front());
    if (!child)
      return child.takeError();
    candidate = std::move(*child);
    result.changed = true;
  }
}

llvm::Expected<CandidateGeneratorProviderResult>
invokeProvider(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &store, const BlobStore &blobs,
               const CandidateGeneratorInvocationView &invocationView) {
  auto config = adoptResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
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
  const lowering::CanonicalDataflowLoweringOptions loweringOptions =
      invocation ? detail::StructuredOwnershipInvocationAccess::loweringOptions(
                       *invocation)
                 : lowering::CanonicalDataflowLoweringOptions{};

  std::vector<AdmittedAccuracyLeaf> admittedLeaves;
  std::vector<PendingAccuracyLineage> lineagePath;
  std::set<ArtifactRootReference, decltype(&artifactRootReferenceLess)>
      expanded(&artifactRootReferenceLess);
  const std::optional<std::uint64_t> maximumOutputs =
      invocationView.maximumOutputArtifacts(CandidateGeneratorOutputSlotRef(0));
  const std::uint64_t maximumMaterializations =
      config->maximumMaterializationAttempts().value_or(
          std::numeric_limits<std::uint64_t>::max());
  std::uint64_t plannedChoiceAttempts = 0;
  std::uint64_t consumedChoiceAttempts = 0;
  std::uint64_t plannedMechanicalClosures = 0;
  std::uint64_t consumedMechanicalClosures = 0;
  bool outputLimitReached = false;

  const ArtifactRootReference *generatorInput = nullptr;
  std::function<llvm::Error(ArtifactRootReference,
                            frontend::MaterializedStructuredOwnershipCandidate)>
      expand = [&](ArtifactRootReference reference,
                   frontend::MaterializedStructuredOwnershipCandidate candidate)
      -> llvm::Error {
    if (outputLimitReached)
      return llvm::Error::success();
    if (!expanded.insert(reference).second)
      return llvm::Error::success();
    auto domain = frontend::enumerateStructuredSpecialMathAccuracyDecisions(
        candidate.structuredProgram);
    if (!domain)
      return domain.takeError();
    if (domain->empty()) {
      if (maximumOutputs && admittedLeaves.size() >= *maximumOutputs) {
        outputLimitReached = true;
        return llvm::Error::success();
      }
      auto finalized =
          finalizeCandidate(std::move(candidate), *generatorInput, invocation,
                            *exactFabric, loweringOptions);
      if (!finalized)
        return finalized.takeError();
      if (!*finalized)
        return llvm::Error::success();
      admittedLeaves.push_back(
          {std::move(reference), lineagePath, std::move(**finalized)});
      return llvm::Error::success();
    }
    if (domain->size() == 1)
      return invalid("mechanical accuracy was not normalized before search");
    if (domain->size() >
        std::numeric_limits<std::uint64_t>::max() - plannedChoiceAttempts)
      return invalid("planned accuracy-decision accounting overflows u64");
    plannedChoiceAttempts += domain->size();

    for (const frontend::StructuredSpecialMathAccuracyDecision &decision :
         *domain) {
      if (consumedChoiceAttempts > maximumMaterializations ||
          consumedMechanicalClosures >
              maximumMaterializations - consumedChoiceAttempts)
        return invalid("accuracy materialization accounting exceeds its bound");
      if (consumedChoiceAttempts + consumedMechanicalClosures ==
          maximumMaterializations) {
        outputLimitReached = true;
        break;
      }
      ++consumedChoiceAttempts;
      auto branch = cloneCandidate(candidate);
      if (!branch)
        return branch.takeError();
      auto child = frontend::materializeStructuredSpecialMathAccuracyDecision(
          std::move(*branch), decision);
      if (!child)
        return child.takeError();
      auto normalized = closeMechanicalAccuracy(
          *child, maximumMaterializations, consumedChoiceAttempts,
          plannedMechanicalClosures, consumedMechanicalClosures);
      if (!normalized)
        return normalized.takeError();
      if (!normalized->complete) {
        outputLimitReached = true;
        break;
      }
      auto published =
          frontend::publishStructuredProgram(child->structuredProgram, store);
      if (!published)
        return published.takeError();
      lineagePath.push_back({reference, *published, decision});
      llvm::Error expansion = expand(*published, std::move(*child));
      lineagePath.pop_back();
      if (expansion)
        return expansion;
    }
    return llvm::Error::success();
  };

  for (const ArtifactRootReference &reference :
       inputBindings[StructuredProgramsInput].artifacts) {
    if (outputLimitReached)
      break;
    generatorInput = &reference;
    auto candidate = cloneRoot(invocation, reference, store);
    if (!candidate)
      return candidate.takeError();
    auto normalized = closeMechanicalAccuracy(
        *candidate, maximumMaterializations, consumedChoiceAttempts,
        plannedMechanicalClosures, consumedMechanicalClosures);
    if (!normalized)
      return normalized.takeError();
    if (!normalized->complete) {
      outputLimitReached = true;
      break;
    }
    ArtifactRootReference normalizedReference = reference;
    if (normalized->changed) {
      auto published = frontend::publishStructuredProgram(
          candidate->structuredProgram, store);
      if (!published)
        return published.takeError();
      normalizedReference = *published;
      lineagePath.push_back({reference, normalizedReference, std::nullopt});
    }
    llvm::Error expansion = expand(normalizedReference, std::move(*candidate));
    if (normalized->changed)
      lineagePath.pop_back();
    if (expansion)
      return std::move(expansion);
  }

  std::vector<ArtifactRootReference> outputs;
  std::vector<CandidateGeneratorLineageEdge> lineageEdges;
  for (AdmittedAccuracyLeaf &leaf : admittedLeaves) {
    for (const PendingAccuracyLineage &step : leaf.lineage) {
      if (!step.decision) {
        if (invocation)
          if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
                  recordSpecialMathAccuracyMechanicalCandidate(
                      *invocation, step.parent, step.child, store))
            return std::move(error);
        lineageEdges.push_back(CandidateGeneratorLineageEdge{
            CandidateGeneratorLineageEdgeKind::MechanicalDerivation,
            CandidateGeneratorOutputSlotRef(0),
            step.child,
            {},
            {}});
        continue;
      }
      if (invocation)
        if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
                recordSpecialMathAccuracyDerivation(*invocation, step.parent,
                                                    step.child, *step.decision,
                                                    store))
          return std::move(error);
      auto ownerPayload =
          frontend::encodeStructuredSpecialMathAccuracyDecision(*step.decision);
      if (!ownerPayload)
        return ownerPayload.takeError();
      lineageEdges.push_back(CandidateGeneratorLineageEdge{
          CandidateGeneratorLineageEdgeKind::CandidateDecision,
          CandidateGeneratorOutputSlotRef(0),
          step.child,
          {step.parent},
          std::move(*ownerPayload)});
    }
    if (invocation)
      if (llvm::Error error = detail::StructuredOwnershipInvocationAccess::
              recordSpecialMathAccuracyFinalCandidate(
                  *invocation, leaf.reference, std::move(leaf.candidate),
                  store))
        return std::move(error);
    outputs.push_back(std::move(leaf.reference));
  }

  CandidateGeneratorOutputBinding output{CandidateGeneratorOutputSlotRef(0),
                                         std::move(outputs)};
  std::vector<CandidateGeneratorWorkUnitSummary> workSummary = {
      {CandidateGeneratorWorkUnitRef(0), plannedChoiceAttempts,
       consumedChoiceAttempts},
      {CandidateGeneratorWorkUnitRef(1), plannedMechanicalClosures,
       consumedMechanicalClosures}};
  if (outputLimitReached)
    return CandidateGeneratorProviderResult{
        IncompleteCandidateGeneratorResult{
            CandidateGeneratorIncompleteReason::SemanticLimitReached,
            {std::move(output)}, std::move(lineageEdges)},
        std::move(workSummary)};
  return CandidateGeneratorProviderResult{
      CompletedCandidateGeneratorResult{{std::move(output)},
                                        std::move(lineageEdges)},
      std::move(workSummary)};
}

const CandidateGeneratorProvider provider{
    descriptor.reference(),
    CandidateGeneratorInProcessProvider{invokeProvider}};

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedStructuredSpecialMathAccuracyGeneratorConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedStructuredSpecialMathAccuracyGeneratorConfigView>
projectResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
    std::optional<std::uint64_t> maximumMaterializationAttempts) {
  if (maximumMaterializationAttempts &&
      *maximumMaterializationAttempts == 0)
    return invalid("materialization attempt limit must be positive");
  std::vector<std::uint8_t> bytes =
      encodeConfig(maximumMaterializationAttempts);
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
      maximumMaterializationAttempts, std::move(bytes), std::move(*digest));
}

llvm::Expected<ResolvedStructuredSpecialMathAccuracyGeneratorConfigView>
adoptResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  auto decoded = decodeConfig(canonicalViewBytes);
  if (!decoded)
    return decoded.takeError();
  return ResolvedStructuredSpecialMathAccuracyGeneratorConfigView(
      *decoded, encodeConfig(*decoded), digest);
}

const CandidateGeneratorDescriptor &
structuredSpecialMathAccuracyCandidateGeneratorDescriptor() {
  return descriptor;
}

llvm::Error registerStructuredSpecialMathAccuracyCandidateGenerator() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(descriptor))
    return error;
  return registerCandidateGeneratorProvider(provider);
}

llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindStructuredSpecialMathAccuracyCandidateGeneratorInputs(
    llvm::ArrayRef<ArtifactRootReference> structuredPrograms,
    const ArtifactRootReference &fabric) {
  if (llvm::Error error =
          registerStructuredSpecialMathAccuracyCandidateGenerator())
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
resolveStructuredSpecialMathAccuracyCandidateGeneratorBinding(
    const ResolvedStructuredSpecialMathAccuracyGeneratorConfigView &config) {
  if (llvm::Error error =
          registerStructuredSpecialMathAccuracyCandidateGenerator())
    return std::move(error);
  return ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), config.canonicalViewBytes(), config.digest());
}

} // namespace loom::dse

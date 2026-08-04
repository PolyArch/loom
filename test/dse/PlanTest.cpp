#include "DSE/Plan.h"

#include "Common/ArtifactStore.h"
#include "Common/ComponentViewDigest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace loom;
using namespace loom::dse;

[[noreturn]] void fail(const std::string &message) {
  std::cerr << "dse plan test failure: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireErrorContains(llvm::Error error, llvm::StringRef needle) {
  const std::string message = llvm::toString(std::move(error));
  if (message.find(needle.str()) == std::string::npos)
    fail("expected error containing '" + needle.str() + "', got: " + message);
}

constexpr ArtifactSchemaDescriptor sourceSchema{"loom.test.plan_source",
                                                SchemaVersion{1, 0}};
constexpr ArtifactSchemaDescriptor candidateSchema{"loom.test.plan_candidate",
                                                   SchemaVersion{1, 0}};
constexpr std::array<std::uint8_t, 4> configSchema = {0x50, 0x4c, 0x41, 0x4e};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  if (bytes != llvm::ArrayRef<std::uint8_t>({0x01}))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "plan test config is not canonical");
  return validateComponentViewDigest(configSchema, bytes, digest);
}

constexpr std::array<CandidateGeneratorInputSlotDescriptor, 1> sourceInputs = {
    {{CandidateGeneratorInputSlotRef(0), "source", PlanValueRole::CandidateSet,
      &sourceSchema, PlanValueCardinality::ExactlyOne}}};
constexpr std::array<CandidateGeneratorInputSlotDescriptor, 1> candidateInputs =
    {{{CandidateGeneratorInputSlotRef(0), "parent", PlanValueRole::CandidateSet,
       &candidateSchema, PlanValueCardinality::FiniteSet}}};
constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputs = {{{
    CandidateGeneratorOutputSlotRef(0),
    "candidate",
    PlanValueRole::CandidateSet,
    &candidateSchema,
    PlanValueCardinality::FiniteSet,
}}};
constexpr std::array<PromotionAcquisitionInputSlotDescriptor, 1>
    promotionInputs = {{{PromotionAcquisitionInputSlotRef(0), "candidate",
                         PlanValueRole::CandidateSet, &candidateSchema,
                         PlanValueCardinality::FiniteSet}}};

const CandidateGeneratorDescriptor sourceGenerator{
    CandidateGeneratorKind(0x7fff1000),
    "test.plan.source",
    "loom.test.plan.source.v1",
    sourceInputs,
    outputs,
    ResolvedDseConfigViewContract{configSchema, validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    {},
    {},
};

const CandidateGeneratorDescriptor transformGenerator{
    CandidateGeneratorKind(0x7fff1001),
    "test.plan.transform",
    "loom.test.plan.transform.v1",
    candidateInputs,
    outputs,
    ResolvedDseConfigViewContract{configSchema, validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    {},
    {},
};

const CandidateGeneratorDescriptor unavailableGenerator{
    CandidateGeneratorKind(0x7fff1002),
    "test.plan.unavailable",
    "loom.test.plan.unavailable.v1",
    candidateInputs,
    outputs,
    ResolvedDseConfigViewContract{configSchema, validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    {},
    {},
};

const PromotionAcquisitionDescriptor objectiveAcquisition{
    PromotionAcquisitionKind(0x7fff2000),
    "test.plan.objective",
    "loom.test.plan.objective.v1",
    promotionInputs,
    PromotionAcquisitionInputSlotRef(0),
    evaluation::CaseSubjectRoleRef(0),
    ResolvedDseConfigViewContract{configSchema, validateConfig},
};

const PromotionAcquisitionDescriptor unavailableAcquisition{
    PromotionAcquisitionKind(0x7fff2001),
    "test.plan.unavailable_objective",
    "loom.test.plan.unavailable_objective.v1",
    promotionInputs,
    PromotionAcquisitionInputSlotRef(0),
    evaluation::CaseSubjectRoleRef(0),
    ResolvedDseConfigViewContract{configSchema, validateConfig},
};

ArtifactRootReference makeReference(const ArtifactSchemaDescriptor &schema,
                                    std::uint8_t fill) {
  std::array<std::uint8_t, ArtifactIdentity::byteSize> bytes{};
  bytes.fill(fill);
  return ArtifactRootReference{schema.identity.str(), schema.version,
                               take(ArtifactIdentity::fromBytes(bytes))};
}

GeneratePlanNodeDefinition makeNode(CandidateGeneratorDescriptorRef descriptor,
                                    std::vector<PlanInputBinding> inputs,
                                    const ComponentViewDigest &digest) {
  return GeneratePlanNodeDefinition{
      descriptor, std::move(inputs), {0x01}, digest};
}

PromotePlanNodeDefinition
makePromoteNode(PromotionAcquisitionDescriptorRef descriptor,
                PlanInputBinding input, const ComponentViewDigest &digest,
                CandidateSelectionPolicy selection) {
  return PromotePlanNodeDefinition{
      descriptor,
      {std::move(input)},
      {0x01},
      digest,
      QualityGatePolicyRef(0),
      std::move(selection),
      PromotePurpose::CandidateSelection,
  };
}

llvm::Expected<CandidateGeneratorInvocationOutcome>
generateSource(llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
               const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &) {
  if (inputBindings.size() != 1 ||
      inputBindings.front().artifacts.size() != 1 ||
      binding.descriptorRef() != sourceGenerator.reference())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "source provider received invalid inputs");
  const ArtifactRootReference first = makeReference(candidateSchema, 0x31);
  const ArtifactRootReference second = makeReference(candidateSchema, 0x22);
  return CompletedCandidateGeneratorInvocation{{
      {CandidateGeneratorOutputSlotRef(0), {first, second, first}},
  }};
}

llvm::Expected<PromotionAcquisitionOutcome>
acquireObjectives(const ResolvedPromotionAcquisitionBinding &binding,
                  const ArtifactStore &) {
  const PromotionAcquisitionInputBinding *candidates =
      binding.findInputBinding(PromotionAcquisitionInputSlotRef(0));
  if (!candidates || candidates->artifacts.size() != 2)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "objective provider received invalid input");

  return CompletedPromotionAcquisition{};
}

void registerOwners() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(sourceGenerator))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerCandidateGeneratorDescriptor(transformGenerator))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerCandidateGeneratorDescriptor(unavailableGenerator))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerPromotionAcquisitionDescriptor(objectiveAcquisition))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerPromotionAcquisitionDescriptor(unavailableAcquisition))
    fail(llvm::toString(std::move(error)));

  const CandidateGeneratorProvider sourceProvider{sourceGenerator.reference(),
                                                  generateSource};
  if (llvm::Error error = registerCandidateGeneratorProvider(sourceProvider))
    fail(llvm::toString(std::move(error)));
  const PromotionAcquisitionProvider objectiveProvider{
      objectiveAcquisition.reference(), acquireObjectives};
  if (llvm::Error error =
          registerPromotionAcquisitionProvider(objectiveProvider))
    fail(llvm::toString(std::move(error)));
}

void exerciseOrderedTypedUseDef() {
  registerOwners();
  const ComponentViewDigest digest = take(computeComponentViewDigest(
      configSchema, std::array<std::uint8_t, 1>{0x01}));
  const ArtifactRootReference source = makeReference(sourceSchema, 0x11);
  const ResolvedObjectiveCatalogs objectiveCatalogs{};
  const std::vector<QualityGatePolicy> qualityGates = {
      take(QualityGatePolicy::get({}))};

  std::vector<DsePlanNodeDefinition> nodes;
  nodes.push_back(makeNode(sourceGenerator.reference(),
                           {ExactPlanArtifacts{{source, source}}}, digest));
  nodes.push_back(makePromoteNode(objectiveAcquisition.reference(),
                                  PlanOutputRef{0, 0}, digest,
                                  AllPassingSelection{}));
  ResolvedDsePlan plan =
      take(ResolvedDsePlan::get(nodes, objectiveCatalogs, qualityGates));
  const auto &generate = std::get<ResolvedGeneratePlanNode>(plan.nodes()[0]);
  const auto &promote = std::get<ResolvedPromotePlanNode>(plan.nodes()[1]);
  if (plan.nodes().size() != 2 || generate.inputBindings().size() != 1 ||
      std::get<ExactPlanArtifacts>(generate.inputBindings()[0])
              .artifacts.size() != 1 ||
      std::get<PlanOutputRef>(promote.inputBindings()[0]) !=
          PlanOutputRef{0, 0})
    fail("resolved mixed plan did not preserve canonical typed use-def");
  const PlanValueDescriptor *produced = plan.resolve(PlanOutputRef{1, 0});
  if (!produced || produced->role != PlanValueRole::CandidateSet ||
      produced->schema != candidateSchema ||
      produced->cardinality != PlanValueCardinality::FiniteSet)
    fail("resolved Promote output did not derive the candidate contract");
  const PlanValueDescriptor *evidence = plan.resolve(PlanOutputRef{1, 1});
  if (!evidence || evidence->role != PlanValueRole::EvidenceSet ||
      evidence->schema != evaluation::EvaluationEvidence::artifactSchema)
    fail("resolved Promote output did not derive the Evidence contract");

  llvm::SmallString<128> storePath;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory("loom-dse-plan", storePath))
    fail("cannot create plan test ArtifactStore: " + error.message());
  ArtifactStore store(storePath);
  DsePlanExecutionOutcome execution = take(executeDsePlan(plan, store));
  const auto *completed = std::get_if<CompletedDsePlanExecution>(&execution);
  if (!completed)
    fail("available mixed plan did not complete");
  llvm::ArrayRef<ArtifactRootReference> generated =
      completed->resolve(PlanOutputRef{0, 0});
  llvm::ArrayRef<ArtifactRootReference> selected =
      completed->resolve(PlanOutputRef{1, 0});
  if (generated.size() != 2 ||
      generated.front() != makeReference(candidateSchema, 0x22) ||
      selected.size() != 2 ||
      selected.front() != makeReference(candidateSchema, 0x22) ||
      selected.back() != makeReference(candidateSchema, 0x31) ||
      !completed->resolve(PlanOutputRef{1, 1}).empty())
    fail("mixed execution did not canonicalize and select its candidates");

  const std::vector<DsePlanNodeDefinition> unavailableNodes = {
      makeNode(sourceGenerator.reference(), {ExactPlanArtifacts{{source}}},
               digest),
      makeNode(unavailableGenerator.reference(), {PlanOutputRef{0, 0}}, digest),
  };
  ResolvedDsePlan unavailablePlan = take(
      ResolvedDsePlan::get(unavailableNodes, objectiveCatalogs, qualityGates));
  DsePlanExecutionOutcome unavailable =
      take(executeDsePlan(unavailablePlan, store));
  const auto *incomplete =
      std::get_if<IncompleteDsePlanExecution>(&unavailable);
  const auto *reason =
      incomplete
          ? std::get_if<CandidateGeneratorIncompleteReason>(&incomplete->reason)
          : nullptr;
  if (!incomplete || incomplete->nodeOrdinal != 1 || !reason ||
      *reason != CandidateGeneratorIncompleteReason::ProviderUnavailable ||
      incomplete->completedPrefix.resolve(PlanOutputRef{0, 0}).size() != 2)
    fail("missing Generate provider did not produce typed Incomplete");

  const std::vector<DsePlanNodeDefinition> unavailablePromotionNodes = {
      makeNode(sourceGenerator.reference(), {ExactPlanArtifacts{{source}}},
               digest),
      makePromoteNode(unavailableAcquisition.reference(), PlanOutputRef{0, 0},
                      digest, AllPassingSelection{}),
  };
  ResolvedDsePlan unavailablePromotion = take(ResolvedDsePlan::get(
      unavailablePromotionNodes, objectiveCatalogs, qualityGates));
  DsePlanExecutionOutcome unavailablePromotionOutcome =
      take(executeDsePlan(unavailablePromotion, store));
  const auto *promotionIncomplete =
      std::get_if<IncompleteDsePlanExecution>(&unavailablePromotionOutcome);
  const auto *promotionReason =
      promotionIncomplete ? std::get_if<PromotionAcquisitionIncompleteReason>(
                                &promotionIncomplete->reason)
                          : nullptr;
  if (!promotionIncomplete || promotionIncomplete->nodeOrdinal != 1 ||
      !promotionReason ||
      *promotionReason !=
          PromotionAcquisitionIncompleteReason::ProviderUnavailable)
    fail("missing Promote provider did not produce typed Incomplete");
  llvm::sys::fs::remove_directories(storePath);

  std::vector<DsePlanNodeDefinition> forward;
  forward.push_back(
      makeNode(transformGenerator.reference(), {PlanOutputRef{1, 0}}, digest));
  auto rejectedForward =
      ResolvedDsePlan::get(forward, objectiveCatalogs, qualityGates);
  if (rejectedForward)
    fail("plan accepted a forward use-def edge");
  requireErrorContains(rejectedForward.takeError(), "earlier node");

  std::vector<DsePlanNodeDefinition> foreign;
  foreign.push_back(makeNode(
      sourceGenerator.reference(),
      {ExactPlanArtifacts{{makeReference(candidateSchema, 0x22)}}}, digest));
  auto rejectedForeign =
      ResolvedDsePlan::get(foreign, objectiveCatalogs, qualityGates);
  if (rejectedForeign)
    fail("plan accepted a foreign static artifact schema");
  requireErrorContains(rejectedForeign.takeError(), "artifact schema");
}

} // namespace

int main() {
  exerciseOrderedTypedUseDef();
  return 0;
}

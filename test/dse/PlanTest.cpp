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

const CandidateGeneratorDescriptor sourceGenerator{
    CandidateGeneratorKind(0x7fff1000),
    "test.plan.source",
    "loom.test.plan.source.v1",
    sourceInputs,
    outputs,
    CandidateGeneratorConfigViewContract{configSchema, validateConfig},
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
    CandidateGeneratorConfigViewContract{configSchema, validateConfig},
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
    CandidateGeneratorConfigViewContract{configSchema, validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    {},
    {},
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

llvm::Expected<CandidateGeneratorInvocationOutcome>
generateSource(const ResolvedCandidateGeneratorBinding &binding,
               const ArtifactStore &) {
  if (binding.inputBindings().size() != 1 ||
      binding.inputBindings().front().artifacts.size() != 1)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "source provider received invalid inputs");
  const ArtifactRootReference first = makeReference(candidateSchema, 0x31);
  const ArtifactRootReference second = makeReference(candidateSchema, 0x22);
  return CompletedCandidateGeneratorInvocation{{
      {CandidateGeneratorOutputSlotRef(0), {first, second, first}},
  }};
}

llvm::Expected<CandidateGeneratorInvocationOutcome>
transformCandidates(const ResolvedCandidateGeneratorBinding &binding,
                    const ArtifactStore &) {
  if (binding.inputBindings().size() != 1 ||
      binding.inputBindings().front().artifacts.size() != 2)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "transform provider did not receive a set");
  return CompletedCandidateGeneratorInvocation{{
      {CandidateGeneratorOutputSlotRef(0),
       {binding.inputBindings().front().artifacts.back()}},
  }};
}

void exerciseOrderedTypedUseDef() {
  if (llvm::Error error = registerCandidateGeneratorDescriptor(sourceGenerator))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerCandidateGeneratorDescriptor(transformGenerator))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error =
          registerCandidateGeneratorDescriptor(unavailableGenerator))
    fail(llvm::toString(std::move(error)));
  const CandidateGeneratorProvider sourceProvider{sourceGenerator.reference(),
                                                  generateSource};
  const CandidateGeneratorProvider transformProvider{
      transformGenerator.reference(), transformCandidates};
  if (llvm::Error error = registerCandidateGeneratorProvider(sourceProvider))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = registerCandidateGeneratorProvider(transformProvider))
    fail(llvm::toString(std::move(error)));
  const ComponentViewDigest digest = take(computeComponentViewDigest(
      configSchema, std::array<std::uint8_t, 1>{0x01}));
  const ArtifactRootReference source = makeReference(sourceSchema, 0x11);

  std::vector<GeneratePlanNodeDefinition> nodes;
  nodes.push_back(makeNode(sourceGenerator.reference(),
                           {ExactPlanArtifacts{{source, source}}}, digest));
  nodes.push_back(
      makeNode(transformGenerator.reference(), {PlanOutputRef{0, 0}}, digest));
  ResolvedGeneratePlan plan = take(ResolvedGeneratePlan::get(std::move(nodes)));
  if (plan.nodes().size() != 2 || plan.nodes()[0].inputBindings().size() != 1 ||
      std::get<ExactPlanArtifacts>(plan.nodes()[0].inputBindings()[0])
              .artifacts.size() != 1 ||
      std::get<PlanOutputRef>(plan.nodes()[1].inputBindings()[0]) !=
          PlanOutputRef{0, 0})
    fail("resolved Generate plan did not preserve canonical typed use-def");
  const PlanValueDescriptor *produced = plan.resolve(PlanOutputRef{1, 0});
  if (!produced || produced->role != PlanValueRole::CandidateSet ||
      produced->schema != candidateSchema ||
      produced->cardinality != PlanValueCardinality::FiniteSet)
    fail("resolved plan output did not derive the generator slot contract");

  llvm::SmallString<128> storePath;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory("loom-dse-plan", storePath))
    fail("cannot create plan test ArtifactStore: " + error.message());
  ArtifactStore store(storePath);
  GeneratePlanExecutionOutcome execution =
      take(executeGeneratePlan(plan, store));
  const auto *completed =
      std::get_if<CompletedGeneratePlanExecution>(&execution);
  if (!completed)
    fail("available Generate plan did not complete");
  llvm::ArrayRef<ArtifactRootReference> finalCandidates =
      completed->resolve(PlanOutputRef{1, 0});
  if (finalCandidates.size() != 1 ||
      finalCandidates.front() != makeReference(candidateSchema, 0x31))
    fail("Generate execution did not canonicalize and forward outputs");

  ResolvedGeneratePlan unavailablePlan = take(ResolvedGeneratePlan::get({
      makeNode(sourceGenerator.reference(), {ExactPlanArtifacts{{source}}},
               digest),
      makeNode(unavailableGenerator.reference(), {PlanOutputRef{0, 0}}, digest),
  }));
  GeneratePlanExecutionOutcome unavailable =
      take(executeGeneratePlan(unavailablePlan, store));
  const auto *incomplete =
      std::get_if<IncompleteGeneratePlanExecution>(&unavailable);
  if (!incomplete || incomplete->nodeOrdinal != 1 ||
      incomplete->reason !=
          CandidateGeneratorIncompleteReason::ProviderUnavailable ||
      incomplete->completedPrefix.resolve(PlanOutputRef{0, 0}).size() != 2)
    fail("missing provider did not produce typed Incomplete");
  llvm::sys::fs::remove_directories(storePath);

  std::vector<GeneratePlanNodeDefinition> forward;
  forward.push_back(
      makeNode(transformGenerator.reference(), {PlanOutputRef{1, 0}}, digest));
  auto rejectedForward = ResolvedGeneratePlan::get(std::move(forward));
  if (rejectedForward)
    fail("plan accepted a forward use-def edge");
  requireErrorContains(rejectedForward.takeError(), "earlier node");

  std::vector<GeneratePlanNodeDefinition> foreign;
  foreign.push_back(makeNode(
      sourceGenerator.reference(),
      {ExactPlanArtifacts{{makeReference(candidateSchema, 0x22)}}}, digest));
  auto rejectedForeign = ResolvedGeneratePlan::get(std::move(foreign));
  if (rejectedForeign)
    fail("plan accepted a foreign static artifact schema");
  requireErrorContains(rejectedForeign.takeError(), "artifact schema");
}

} // namespace

int main() {
  exerciseOrderedTypedUseDef();
  return 0;
}

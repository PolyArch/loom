#include "DSE/StructuredScheduleCandidateGenerator.h"

#include "Common/ArtifactStore.h"
#include "Config/ResolvedConfig.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Frontend/Compilation/StructuredSchedule.h"
#include "Frontend/IR/StructuredProgramArtifact.h"

#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.structured_schedule_generator.config.1.0";

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
         PlanValueCardinality::NonEmptySet},
        {CandidateGeneratorInputSlotRef(FabricInput), "fabric",
         PlanValueRole::CandidateSet, &fabric::fabricArtifactSchema,
         PlanValueCardinality::ExactlyOne},
    }};

constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {{
    {CandidateGeneratorOutputSlotRef(0), "structured_program",
     PlanValueRole::CandidateSet, &frontend::structuredProgramArtifactSchema,
     PlanValueCardinality::NonEmptySet},
}};

constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 2> workUnits = {{
    {CandidateGeneratorWorkUnitRef(0), "loop_scope"},
    {CandidateGeneratorWorkUnitRef(1), "schedule_decision"},
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

std::vector<std::uint8_t> encodeConfig(std::uint64_t limit) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(8);
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(limit >> shift));
  bytes.push_back(static_cast<std::uint8_t>(limit));
  return bytes;
}

llvm::Expected<std::uint64_t> decodeConfig(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() < 8)
    return invalid("truncated scope expansion limit");
  std::uint64_t limit = 0;
  for (std::uint8_t byte : bytes.take_front(8))
    limit = (limit << 8) | byte;
  if (bytes.size() != 8)
    return invalid("config has trailing bytes");
  if (limit == 0)
    return invalid("scope expansion limit must be positive");
  return limit;
}

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedStructuredScheduleGeneratorConfigView(
      descriptorBytes(), bytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

const CandidateGeneratorDescriptor descriptor{
    structuredScheduleCandidateGeneratorKind,
    "compiler.structured_schedule",
    "loom.compiler.structured_schedule.generator.v1",
    inputSlots,
    outputSlots,
    ResolvedDseConfigViewContract{descriptorBytes(), validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    {},
};

const ArtifactRootReference &
singleInput(llvm::ArrayRef<CandidateGeneratorInputBinding> bindings,
            InputSlot slot) {
  return bindings[slot].artifacts.front();
}

llvm::Expected<CandidateGeneratorInvocationOutcome> invokeScheduleProvider(
    llvm::ArrayRef<CandidateGeneratorInputBinding> inputBindings,
    const ResolvedCandidateGeneratorBinding &binding,
    const ArtifactStore &store) {
  auto config = adoptResolvedStructuredScheduleGeneratorConfigView(
      descriptorBytes(), binding.canonicalConfigBytes(),
      binding.configDigest());
  if (!config)
    return config.takeError();
  auto fabric = fabric::importEntireFabricRoot(
      singleInput(inputBindings, FabricInput), store);
  if (!fabric)
    return fabric.takeError();

  std::vector<ArtifactRootReference> outputs =
      inputBindings[StructuredProgramsInput].artifacts;
  for (const ArtifactRootReference &reference :
       inputBindings[StructuredProgramsInput].artifacts) {
    auto parent = frontend::importStructuredProgram(reference, store);
    if (!parent)
      return parent.takeError();
    auto decisions = frontend::enumerateStructuredScheduleDecisions(
        *parent, *fabric, config->scopeExpansionLimit());
    if (!decisions)
      return decisions.takeError();
    outputs.reserve(outputs.size() + decisions->size());
    for (const frontend::StructuredScheduleDecision &decision : *decisions) {
      auto child =
          frontend::materializeStructuredScheduleDecision(*parent, decision);
      if (!child)
        return child.takeError();
      auto published = frontend::publishStructuredProgram(*child, store);
      if (!published)
        return published.takeError();
      outputs.push_back(std::move(*published));
    }
  }
  return CandidateGeneratorInvocationOutcome{
      CompletedCandidateGeneratorInvocation{{
          {CandidateGeneratorOutputSlotRef(0), std::move(outputs)},
      }}};
}

const CandidateGeneratorProvider provider{descriptor.reference(),
                                          invokeScheduleProvider};

} // namespace

llvm::ArrayRef<std::uint8_t>
resolvedStructuredScheduleGeneratorConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedStructuredScheduleGeneratorConfigView>
projectResolvedStructuredScheduleGeneratorConfigView(
    const ResolvedConfig &config) {
  const std::uint64_t limit = config.dse.schedule.scopeExpansionLimit;
  if (limit == 0)
    return invalid("scope expansion limit must be positive");
  std::vector<std::uint8_t> bytes = encodeConfig(limit);
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedStructuredScheduleGeneratorConfigView(limit, std::move(bytes),
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
  auto limit = decodeConfig(canonicalViewBytes);
  if (!limit)
    return limit.takeError();
  std::vector<std::uint8_t> reencoded = encodeConfig(*limit);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalViewBytes)
    return invalid("decoded config does not re-encode to the source bytes");
  return ResolvedStructuredScheduleGeneratorConfigView(
      *limit, std::move(reencoded), digest);
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

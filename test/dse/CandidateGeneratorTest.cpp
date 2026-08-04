#include "DSE/CandidateGenerator.h"

#include "Common/ComponentViewDigest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <vector>

namespace {

using loom::ArtifactIdentity;
using loom::ArtifactRootReference;
using loom::ArtifactSchemaDescriptor;
using loom::ComponentViewDigest;
using loom::SchemaVersion;
using namespace loom::dse;

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "candidate generator test failure: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

void requireErrorContains(llvm::Error error, llvm::StringRef expected) {
  if (!error)
    fail("expected an error");
  std::string message = llvm::toString(std::move(error));
  if (message.find(expected.str()) == std::string::npos)
    fail(("error did not contain expected text: " + expected).str());
}

constexpr ArtifactSchemaDescriptor inputSchema{"loom.test.generator_input",
                                               SchemaVersion{1, 0}};
constexpr ArtifactSchemaDescriptor outputSchema{"loom.test.generator_output",
                                                SchemaVersion{1, 0}};
constexpr std::array<std::uint8_t, 4> configSchema = {0x54, 0x45, 0x53, 0x54};

llvm::Error validateConfig(llvm::ArrayRef<std::uint8_t> bytes,
                           const ComponentViewDigest &digest) {
  if (bytes != llvm::ArrayRef<std::uint8_t>({0x01, 0x02}))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "test config bytes are not canonical");
  return loom::validateComponentViewDigest(configSchema, bytes, digest);
}

constexpr std::array<CandidateGeneratorInputSlotDescriptor, 1> inputSlots = {{{
    CandidateGeneratorInputSlotRef(0),
    "subject",
    PlanValueRole::CandidateSet,
    &inputSchema,
    PlanValueCardinality::ExactlyOne,
}}};
constexpr std::array<CandidateGeneratorOutputSlotDescriptor, 1> outputSlots = {
    {{CandidateGeneratorOutputSlotRef(0), "candidate",
      PlanValueRole::CandidateSet, &outputSchema,
      PlanValueCardinality::FiniteSet}}};
constexpr std::array<CandidateGeneratorWorkUnitDescriptor, 1> workUnits = {{{
    CandidateGeneratorWorkUnitRef(0),
    "candidate_attempt",
}}};

const CandidateGeneratorDescriptor descriptor{
    CandidateGeneratorKind(0x7fff0001),
    "test.generator",
    "loom.test.generator.v1",
    inputSlots,
    outputSlots,
    CandidateGeneratorConfigViewContract{configSchema, validateConfig},
    CandidateGeneratorDeterminism::Deterministic,
    workUnits,
    {},
};

ArtifactRootReference makeReference(const ArtifactSchemaDescriptor &schema,
                                    std::uint8_t fill) {
  std::array<std::uint8_t, ArtifactIdentity::byteSize> bytes{};
  bytes.fill(fill);
  return ArtifactRootReference{schema.identity.str(), schema.version,
                               take(ArtifactIdentity::fromBytes(bytes))};
}

void exerciseRegistryAndBinding() {
  requireSuccess(registerCandidateGeneratorDescriptor(descriptor));
  requireSuccess(registerCandidateGeneratorDescriptor(descriptor));

  const CandidateGeneratorDescriptor *resolved =
      descriptor.reference().descriptor();
  if (resolved != &descriptor ||
      findCandidateGeneratorDescriptor(descriptor.kind) != &descriptor)
    fail("exact descriptor reference did not resolve the registered owner");

  const std::array<std::uint8_t, 2> canonicalConfig = {0x01, 0x02};
  const ComponentViewDigest digest =
      take(loom::computeComponentViewDigest(configSchema, canonicalConfig));
  const ArtifactRootReference subject = makeReference(inputSchema, 0x11);
  std::vector<CandidateGeneratorInputBinding> inputs = {
      {CandidateGeneratorInputSlotRef(0), {subject, subject}}};
  auto binding = take(ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), std::move(inputs), canonicalConfig, digest));
  if (binding.descriptorRef() != descriptor.reference() ||
      binding.inputBindings().size() != 1 ||
      binding.inputBindings().front().artifacts.size() != 1 ||
      !llvm::equal(binding.canonicalConfigBytes(), canonicalConfig) ||
      binding.configDigest() != digest)
    fail("resolved binding did not preserve exact descriptor-owned inputs");

  std::vector<CandidateGeneratorInputBinding> wrongSchema = {
      {CandidateGeneratorInputSlotRef(0), {makeReference(outputSchema, 0x22)}}};
  auto rejected = ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), std::move(wrongSchema), canonicalConfig, digest);
  if (rejected)
    fail("binding accepted a foreign input schema");
  requireErrorContains(rejected.takeError(), "does not accept artifact schema");

  const ComponentViewDigest wrongDigest = take(loom::computeComponentViewDigest(
      configSchema, std::array<std::uint8_t, 1>{0x03}));
  std::vector<CandidateGeneratorInputBinding> validInputs = {
      {CandidateGeneratorInputSlotRef(0), {makeReference(inputSchema, 0x33)}}};
  auto stale = ResolvedCandidateGeneratorBinding::get(
      descriptor.reference(), std::move(validInputs), canonicalConfig,
      wrongDigest);
  if (stale)
    fail("binding accepted a stale config digest");
  requireErrorContains(stale.takeError(), "digest");

  CandidateGeneratorDescriptor conflicting = descriptor;
  conflicting.spelling = "test.generator.conflict";
  requireErrorContains(registerCandidateGeneratorDescriptor(conflicting),
                       "conflicting registration");
}

} // namespace

int main() {
  exerciseRegistryAndBinding();
  return 0;
}

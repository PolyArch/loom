#include "CompilerTargetBindingInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Fabric/Artifact/FabricSystemContracts.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace loom {
namespace {

constexpr char architectureFingerprintDomain[] =
    "loom.compiler.architecture.fingerprint.v1\0";

llvm::Error bindingError(llvm::StringRef kind, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 kind + ": " + message);
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::StringRef asText(llvm::ArrayRef<std::uint8_t> bytes) {
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

bool providerEqual(const LlvmProviderIdentity &lhs,
                   const LlvmProviderIdentity &rhs) {
  return lhs.repositoryIdentity == rhs.repositoryIdentity &&
         lhs.fullCommitIdentity == rhs.fullCommitIdentity;
}

bool reconstructedEqual(const CompilerTargetBinding &binding,
                        const detail::ReconstructedCompilerTarget &target) {
  return providerEqual(binding.compilerProvider(), target.provider) &&
         binding.targetTriple() == target.targetTriple &&
         binding.dataLayout() == target.dataLayout &&
         binding.objectFormat() == target.objectFormat &&
         llvm::equal(binding.backendFeatures(), target.backendFeatures) &&
         llvm::equal(binding.targetScopeBindings(), target.targetScopeBindings);
}

llvm::Expected<std::vector<std::uint8_t>>
architectureBytes(const CompilerProcessorArchitectureRef &processor,
                  const ArtifactStore &store) {
  auto architecture = detail::resolveProcessorArchitecture(processor, store);
  if (!architecture)
    return architecture.takeError();
  return fabric::encodeInstructionCoreArchitecturalContract(*architecture);
}

} // namespace

llvm::Expected<ArchitectureFingerprint>
ArchitectureFingerprint::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return bindingError("architecture_fingerprint_invalid",
                        "fingerprint requires exactly 32 bytes");
  Storage storage;
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return ArchitectureFingerprint(storage);
}

ArchitectureFingerprint computeArchitectureFingerprint(
    const fabric::InstructionCoreArchitecturalContract &contract) {
  const std::vector<std::uint8_t> contractBytes = llvm::cantFail(
      fabric::encodeInstructionCoreArchitecturalContract(contract));
  std::vector<std::uint8_t> preimage;
  constexpr std::size_t domainSize = sizeof(architectureFingerprintDomain) - 1;
  preimage.reserve(domainSize + 8 + contractBytes.size());
  preimage.insert(preimage.end(), architectureFingerprintDomain,
                  architectureFingerprintDomain + domainSize);
  appendU64Be(preimage, contractBytes.size());
  preimage.insert(preimage.end(), contractBytes.begin(), contractBytes.end());
  return llvm::cantFail(
      ArchitectureFingerprint::fromBytes(llvm::SHA256::hash(preimage)));
}

std::string
formatArchitectureFingerprintHex(const ArchitectureFingerprint &fingerprint) {
  static constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(ArchitectureFingerprint::byteSize * 2);
  for (std::uint8_t byte : fingerprint.bytes()) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0x0f]);
  }
  return result;
}

llvm::Expected<ArchitectureFingerprint>
parseArchitectureFingerprintHex(llvm::StringRef spelling) {
  if (spelling.size() != ArchitectureFingerprint::byteSize * 2)
    return bindingError("architecture_fingerprint_invalid",
                        "fingerprint must use exactly 64 lowercase "
                        "hexadecimal characters");
  auto nibble = [](char value) -> int {
    if (value >= '0' && value <= '9')
      return value - '0';
    if (value >= 'a' && value <= 'f')
      return value - 'a' + 10;
    return -1;
  };
  std::array<std::uint8_t, ArchitectureFingerprint::byteSize> bytes;
  for (std::size_t index = 0; index < spelling.size(); index += 2) {
    const int high = nibble(spelling[index]);
    const int low = nibble(spelling[index + 1]);
    if (high < 0 || low < 0)
      return bindingError("architecture_fingerprint_invalid",
                          "fingerprint must use lowercase hexadecimal");
    bytes[index / 2] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return ArchitectureFingerprint::fromBytes(bytes);
}

CompilerProcessorArchitectureRef
CompilerProcessorArchitectureRef::host(Host reference) {
  return CompilerProcessorArchitectureRef(std::move(reference));
}

CompilerProcessorArchitectureRef
CompilerProcessorArchitectureRef::instruction(Instruction reference) {
  return CompilerProcessorArchitectureRef(std::move(reference));
}

const ArtifactIdentity &
CompilerProcessorArchitectureRef::fabricArtifact() const {
  return std::visit(
      [](const auto &reference) -> const ArtifactIdentity & {
        return reference.artifact;
      },
      value_);
}

llvm::Expected<CompilerTargetBinding>
decodeCompilerTargetBinding(llvm::StringRef canonicalJson,
                            const ArtifactStore &store) {
  auto fields = detail::parseCompilerTargetBindingFields(canonicalJson);
  if (!fields)
    return fields.takeError();
  auto architecture = detail::resolveProcessorArchitecture(
      fields->processorArchitecture, store);
  if (!architecture)
    return architecture.takeError();
  const ArchitectureFingerprint expectedFingerprint =
      computeArchitectureFingerprint(*architecture);
  if (fields->architectureFingerprint != expectedFingerprint)
    return bindingError("architecture_fingerprint_mismatch",
                        "stored fingerprint is not derived from the exact "
                        "Fabric architecture contract");
  auto reconstructed = detail::reconstructCompilerTarget(
      *architecture, fields->backendAbi, fields->codeModel,
      fields->relocationModel, fields->backendCpu);
  if (!reconstructed)
    return reconstructed.takeError();

  CompilerTargetBinding binding(
      std::move(fields->processorArchitecture), fields->architectureFingerprint,
      std::move(fields->provider), std::move(fields->targetTriple),
      std::move(fields->dataLayout), fields->backendAbi, fields->objectFormat,
      fields->codeModel, fields->relocationModel, std::move(fields->backendCpu),
      std::move(fields->backendFeatures),
      std::move(fields->targetScopeBindings),
      std::move(fields->supportComponents));
  if (!reconstructedEqual(binding, *reconstructed))
    return bindingError("compiler_target_reconstruction_mismatch",
                        "stored target fields are not the exact projection of "
                        "the pinned LLVM TargetMachine");
  if (detail::serializeCompilerTargetBinding(binding) != canonicalJson)
    return bindingError("compiler_target_binding_not_canonical",
                        "stored JSON is not the production canonical encoding");
  return binding;
}

llvm::Expected<FinalizedCompilerTargetBinding>
resolveCompilerTargetBinding(const CompilerProcessorArchitectureRef &processor,
                             const CompilerTargetPolicy &policy,
                             const ArtifactStore &store) {
  auto architecture = detail::resolveProcessorArchitecture(processor, store);
  if (!architecture)
    return architecture.takeError();
  auto target = detail::reconstructCompilerTarget(
      *architecture, policy.backendAbi, policy.codeModel,
      policy.relocationModel, policy.backendCpu);
  if (!target)
    return target.takeError();
  auto support =
      detail::canonicalizeSupportComponents(policy.supportComponents);
  if (!support)
    return support.takeError();

  CompilerTargetBinding binding(
      processor, computeArchitectureFingerprint(*architecture),
      std::move(target->provider), std::move(target->targetTriple),
      std::move(target->dataLayout), policy.backendAbi, target->objectFormat,
      policy.codeModel, policy.relocationModel, policy.backendCpu,
      std::move(target->backendFeatures),
      std::move(target->targetScopeBindings), std::move(*support));
  const std::string json = detail::serializeCompilerTargetBinding(binding);
  auto strict = decodeCompilerTargetBinding(json, store);
  if (!strict)
    return strict.takeError();
  CanonicalSemanticBytes bytes(
      std::vector<std::uint8_t>(json.begin(), json.end()));
  auto identity = store.put(compilerTargetBindingSchema, bytes);
  if (!identity)
    return identity.takeError();
  return importCompilerTargetBinding(
      {compilerTargetBindingSchema.identity.str(),
       compilerTargetBindingSchema.version, *identity},
      store);
}

llvm::Expected<FinalizedCompilerTargetBinding>
importCompilerTargetBinding(const ArtifactRootReference &reference,
                            const ArtifactStore &store) {
  if (reference.schemaIdentity != compilerTargetBindingSchema.identity ||
      reference.schemaVersion != compilerTargetBindingSchema.version)
    return bindingError("compiler_target_binding_schema_unsupported",
                        "reference is not loom.compiler_target_binding 1.0");
  auto bytes = store.get(reference);
  if (!bytes)
    return bytes.takeError();
  auto binding = decodeCompilerTargetBinding(asText(bytes->bytes()), store);
  if (!binding)
    return binding.takeError();
  return FinalizedCompilerTargetBinding(reference, std::move(*bytes),
                                        std::move(*binding));
}

llvm::Error requireCompilerTargetCompatibility(
    const CompilerTargetBinding &binding,
    const CompilerProcessorArchitectureRef &processor,
    const ArtifactStore &store) {
  if (binding.processorArchitecture().isHost() != processor.isHost())
    return bindingError("processor_kind_mismatch",
                        "HostCore and InstructionCore target bindings are "
                        "independently exact");
  auto selectedBytes =
      architectureBytes(binding.processorArchitecture(), store);
  if (!selectedBytes)
    return selectedBytes.takeError();
  auto candidateBytes = architectureBytes(processor, store);
  if (!candidateBytes)
    return candidateBytes.takeError();
  if (*selectedBytes != *candidateBytes)
    return bindingError("processor_architecture_incompatible",
                        "the exact Fabric architecture contracts differ");
  return llvm::Error::success();
}

} // namespace loom

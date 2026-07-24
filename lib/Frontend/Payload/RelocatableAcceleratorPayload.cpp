#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ComponentViewDigest.h"
#include "Frontend/Payload/BuildConfig.h"
#include "Frontend/Payload/LlvmModuleNormalization.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom {
namespace {

llvm::Error rejected(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::ArrayRef<std::uint8_t> asBytes(llvm::StringRef text) {
  return {reinterpret_cast<const std::uint8_t *>(text.data()), text.size()};
}

llvm::StringRef asText(llvm::ArrayRef<std::uint8_t> bytes) {
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

/// Appends a variable byte sequence as its u64 big-endian length followed by
/// its exact bytes.
void appendFramed(std::vector<std::uint8_t> &bytes,
                  llvm::ArrayRef<std::uint8_t> field) {
  appendU64Be(bytes, static_cast<std::uint64_t>(field.size()));
  bytes.insert(bytes.end(), field.begin(), field.end());
}

void appendFramed(std::vector<std::uint8_t> &bytes, llvm::StringRef field) {
  appendFramed(bytes, asBytes(field));
}

/// Appends a fixed digest as its raw bytes, with no length prefix.
void appendFixed(std::vector<std::uint8_t> &bytes,
                 llvm::ArrayRef<std::uint8_t> digest) {
  bytes.insert(bytes.end(), digest.begin(), digest.end());
}

/// Raw size of the stored normalized-bitcode SHA-256 digest.
constexpr std::size_t bitcodeDigestSize = 32;

/// Sequential reader over canonical root bytes. Every read is bounded by the
/// remaining input, so a truncated or overlong framed length is a typed
/// rejection rather than an out-of-range access. The first rejection is kept
/// and later reads yield nothing, which lets the root be read as the field list
/// it is and checked once.
class RootReader {
public:
  explicit RootReader(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::ArrayRef<std::uint8_t> framed() {
    if (!rejection_.empty())
      return {};
    if (remaining() < 8) {
      reject("relocatable_payload_encoding_truncated: a framed length does not "
             "fit in the canonical bytes");
      return {};
    }
    std::uint64_t length = 0;
    for (unsigned index = 0; index < 8; ++index)
      length = (length << 8) | bytes_[offset_ + index];
    offset_ += 8;
    if (length > remaining()) {
      reject("relocatable_payload_encoding_overflow: a framed length exceeds "
             "the remaining canonical bytes");
      return {};
    }
    return take(static_cast<std::size_t>(length));
  }

  llvm::ArrayRef<std::uint8_t> fixed(std::size_t size) {
    if (!rejection_.empty())
      return {};
    if (remaining() < size) {
      reject("relocatable_payload_encoding_truncated: a fixed digest does not "
             "fit in the canonical bytes");
      return {};
    }
    return take(size);
  }

  /// The first encoding rejection, or the trailing-data rejection when the
  /// complete root was read but bytes remain.
  llvm::Error takeError() {
    if (rejection_.empty() && offset_ != bytes_.size())
      reject("relocatable_payload_encoding_trailing: the canonical bytes carry "
             "data after the canonical root");
    if (rejection_.empty())
      return llvm::Error::success();
    return rejected(rejection_);
  }

private:
  void reject(llvm::StringRef message) { rejection_ = message.str(); }

  std::uint64_t remaining() const {
    return static_cast<std::uint64_t>(bytes_.size() - offset_);
  }

  llvm::ArrayRef<std::uint8_t> take(std::size_t size) {
    const llvm::ArrayRef<std::uint8_t> field = bytes_.slice(offset_, size);
    offset_ += size;
    return field;
  }

  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
  std::string rejection_;
};

AbiCompatibilityKeyInputs keyInputs(const LlvmProviderIdentity &provider,
                                    llvm::StringRef targetTriple,
                                    llvm::StringRef dataLayout,
                                    const ResolvedFrontendConfigView &view) {
  AbiCompatibilityKeyInputs inputs;
  inputs.repositoryIdentity = provider.repositoryIdentity;
  inputs.fullCommitIdentity = provider.fullCommitIdentity;
  inputs.canonicalTargetTriple = targetTriple;
  inputs.canonicalDataLayout = dataLayout;
  inputs.viewSchemaDescriptorBytes = view.schemaDescriptorBytes();
  inputs.viewCanonicalBytes = view.canonicalViewBytes();
  return inputs;
}

llvm::Error disagreement(llvm::StringRef field) {
  return rejected(
      "relocatable_payload_incompatible: the payloads disagree on " + field);
}

} // namespace

const LlvmProviderIdentity &buildSelectedLlvmProvider() {
  static const LlvmProviderIdentity provider{
      LOOM_LLVM_PROVIDER_REPOSITORY_IDENTITY,
      LOOM_LLVM_PROVIDER_COMMIT_IDENTITY};
  return provider;
}

RelocatableAcceleratorPayload::RelocatableAcceleratorPayload(
    LlvmProviderIdentity provider, std::string targetTriple,
    std::string dataLayout, AbiCompatibilityKey abiKey,
    ResolvedFrontendConfigView view, std::vector<std::uint8_t> bitcode,
    std::array<std::uint8_t, 32> bitcodeDigest)
    : provider_(std::move(provider)), targetTriple_(std::move(targetTriple)),
      dataLayout_(std::move(dataLayout)), abiKey_(abiKey), view_(view),
      bitcode_(std::move(bitcode)), bitcodeDigest_(bitcodeDigest) {}

llvm::Expected<RelocatableAcceleratorPayload>
RelocatableAcceleratorPayload::create(
    llvm::ArrayRef<std::uint8_t> sourceBitcode,
    const ResolvedFrontendConfigView &frontendConfigView) {
  llvm::Expected<NormalizedLlvmModule> normalized =
      normalizeLlvmModule(sourceBitcode);
  if (!normalized)
    return normalized.takeError();

  const LlvmProviderIdentity &provider = buildSelectedLlvmProvider();
  const AbiCompatibilityKey abiKey = computeAbiCompatibilityKey(
      keyInputs(provider, normalized->canonicalTargetTriple,
                normalized->canonicalDataLayout, frontendConfigView));
  return RelocatableAcceleratorPayload(
      provider, normalized->canonicalTargetTriple,
      normalized->canonicalDataLayout, abiKey, frontendConfigView,
      std::move(normalized->bitcode), normalized->bitcodeDigest);
}

CanonicalSemanticBytes
RelocatableAcceleratorPayload::canonicalSemanticBytes() const {
  std::vector<std::uint8_t> bytes;
  appendFramed(bytes, provider_.repositoryIdentity);
  appendFramed(bytes, provider_.fullCommitIdentity);
  appendFramed(bytes, targetTriple_);
  appendFramed(bytes, dataLayout_);
  appendFixed(bytes, abiKey_.bytes());
  appendFramed(bytes, view_.schemaDescriptorBytes());
  appendFramed(bytes, view_.canonicalViewBytes());
  appendFixed(bytes, view_.digest().bytes());
  appendFixed(bytes, bitcodeDigest_);
  appendFramed(bytes, bitcode_);
  return CanonicalSemanticBytes(std::move(bytes));
}

ArtifactIdentity RelocatableAcceleratorPayload::identity() const {
  return finalizeArtifactIdentity(artifactSchema, canonicalSemanticBytes());
}

llvm::Expected<RelocatableAcceleratorPayload>
decodeRelocatableAcceleratorPayload(
    const ArtifactSchemaDescriptor &schema,
    llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  const ArtifactSchemaDescriptor &supported =
      RelocatableAcceleratorPayload::artifactSchema;
  if (schema.identity != supported.identity ||
      schema.version != supported.version)
    return rejected("relocatable_payload_schema_unsupported: this reader only "
                    "decodes " +
                    supported.identity + " 1.0");

  // The canonical root in its fixed field order.
  RootReader reader(canonicalBytes);
  const llvm::ArrayRef<std::uint8_t> repository = reader.framed();
  const llvm::ArrayRef<std::uint8_t> commit = reader.framed();
  const llvm::ArrayRef<std::uint8_t> targetTriple = reader.framed();
  const llvm::ArrayRef<std::uint8_t> dataLayout = reader.framed();
  const llvm::ArrayRef<std::uint8_t> abiKeyBytes =
      reader.fixed(AbiCompatibilityKey::byteSize);
  const llvm::ArrayRef<std::uint8_t> viewDescriptor = reader.framed();
  const llvm::ArrayRef<std::uint8_t> viewBytes = reader.framed();
  const llvm::ArrayRef<std::uint8_t> viewDigestBytes =
      reader.fixed(ComponentViewDigest::byteSize);
  const llvm::ArrayRef<std::uint8_t> bitcodeDigestBytes =
      reader.fixed(bitcodeDigestSize);
  const llvm::ArrayRef<std::uint8_t> moduleBytes = reader.framed();
  if (llvm::Error error = reader.takeError())
    return std::move(error);

  const LlvmProviderIdentity &provider = buildSelectedLlvmProvider();
  if (asText(repository) != llvm::StringRef(provider.repositoryIdentity) ||
      asText(commit) != llvm::StringRef(provider.fullCommitIdentity))
    return rejected("llvm_provider_mismatch: the payload records LLVM provider "
                    "'" +
                    asText(repository) + "' at commit '" + asText(commit) +
                    "', which is not the provider this build selected");

  llvm::Expected<ComponentViewDigest> viewDigest =
      ComponentViewDigest::fromBytes(viewDigestBytes);
  if (!viewDigest)
    return viewDigest.takeError();
  llvm::Expected<ResolvedFrontendConfigView> view =
      adoptResolvedFrontendConfigView(viewDescriptor, viewBytes, *viewDigest);
  if (!view)
    return view.takeError();

  // The stored module is the authority, so it is reparsed, verified, and
  // rewritten through the same production writer before anything derived from
  // it is trusted.
  llvm::Expected<NormalizedLlvmModule> normalized =
      normalizeLlvmModule(moduleBytes);
  if (!normalized)
    return normalized.takeError();
  if (llvm::ArrayRef<std::uint8_t>(normalized->bitcode) != moduleBytes)
    return rejected("normalized_bitcode_not_canonical: the stored module bytes "
                    "are not what the production writer contract emits for "
                    "this module");
  if (llvm::ArrayRef<std::uint8_t>(normalized->bitcodeDigest) !=
      bitcodeDigestBytes)
    return rejected("normalized_bitcode_digest_mismatch: the stored digest is "
                    "not the digest of the stored module bytes");
  if (asText(targetTriple) !=
      llvm::StringRef(normalized->canonicalTargetTriple))
    return rejected("target_triple_mismatch: the payload records target triple "
                    "'" +
                    asText(targetTriple) + "' but its module projects '" +
                    normalized->canonicalTargetTriple + "'");
  if (asText(dataLayout) != llvm::StringRef(normalized->canonicalDataLayout))
    return rejected("data_layout_mismatch: the payload records data layout '" +
                    asText(dataLayout) + "' but its module projects '" +
                    normalized->canonicalDataLayout + "'");

  llvm::Expected<AbiCompatibilityKey> abiKey =
      AbiCompatibilityKey::fromBytes(abiKeyBytes);
  if (!abiKey)
    return abiKey.takeError();
  if (llvm::Error error = validateAbiCompatibilityKey(
          keyInputs(provider, normalized->canonicalTargetTriple,
                    normalized->canonicalDataLayout, *view),
          *abiKey))
    return std::move(error);

  return RelocatableAcceleratorPayload(
      provider, normalized->canonicalTargetTriple,
      normalized->canonicalDataLayout, *abiKey, *view,
      std::move(normalized->bitcode), normalized->bitcodeDigest);
}

llvm::Error requireRelocatablePayloadCompatibility(
    const RelocatableAcceleratorPayload &lhs,
    const RelocatableAcceleratorPayload &rhs) {
  if (lhs.llvmProvider().repositoryIdentity !=
      rhs.llvmProvider().repositoryIdentity)
    return disagreement("the LLVM provider repository identity");
  if (lhs.llvmProvider().fullCommitIdentity !=
      rhs.llvmProvider().fullCommitIdentity)
    return disagreement("the LLVM provider commit identity");
  if (lhs.targetTriple() != rhs.targetTriple())
    return disagreement("the canonical target triple");
  if (lhs.dataLayout() != rhs.dataLayout())
    return disagreement("the canonical data layout");
  if (lhs.abiCompatibilityKey() != rhs.abiCompatibilityKey())
    return disagreement("the ABI compatibility key");

  // The complete frontend view is compared, not only its digest, so a later
  // view schema version cannot pass this preflight on a digest alone.
  const ResolvedFrontendConfigView &lhsView = lhs.frontendConfigView();
  const ResolvedFrontendConfigView &rhsView = rhs.frontendConfigView();
  if (lhsView.schemaDescriptorBytes() != rhsView.schemaDescriptorBytes())
    return disagreement("the frontend config view schema descriptor");
  if (lhsView.canonicalViewBytes() != rhsView.canonicalViewBytes())
    return disagreement("the frontend config view canonical bytes");
  if (lhsView.digest() != rhsView.digest())
    return disagreement("the frontend config view digest");
  return llvm::Error::success();
}

} // namespace loom

#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ComponentViewDigest.h"
#include "Frontend/Payload/BuildConfig.h"
#include "Frontend/Payload/LlvmModuleNormalization.h"
#include "RelocatablePayloadRootCodec.h"

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
  // The view digest is derived on demand, so it is held while it is framed.
  const ComponentViewDigest viewDigest = view_.digest();
  detail::RelocatablePayloadRoot root;
  root.repositoryIdentity = asBytes(provider_.repositoryIdentity);
  root.fullCommitIdentity = asBytes(provider_.fullCommitIdentity);
  root.targetTriple = asBytes(targetTriple_);
  root.dataLayout = asBytes(dataLayout_);
  root.abiCompatibilityKey = abiKey_.bytes();
  root.viewSchemaDescriptor = view_.schemaDescriptorBytes();
  root.viewCanonicalBytes = view_.canonicalViewBytes();
  root.viewDigest = viewDigest.bytes();
  root.normalizedBitcodeDigest = bitcodeDigest_;
  root.normalizedBitcode = bitcode_;
  return CanonicalSemanticBytes(detail::encodeRelocatablePayloadRoot(root));
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

  llvm::Expected<detail::RelocatablePayloadRoot> root =
      detail::decodeRelocatablePayloadRoot(canonicalBytes);
  if (!root)
    return root.takeError();
  const llvm::ArrayRef<std::uint8_t> repository = root->repositoryIdentity;
  const llvm::ArrayRef<std::uint8_t> commit = root->fullCommitIdentity;
  const llvm::ArrayRef<std::uint8_t> targetTriple = root->targetTriple;
  const llvm::ArrayRef<std::uint8_t> dataLayout = root->dataLayout;
  const llvm::ArrayRef<std::uint8_t> abiKeyBytes = root->abiCompatibilityKey;
  const llvm::ArrayRef<std::uint8_t> viewDescriptor =
      root->viewSchemaDescriptor;
  const llvm::ArrayRef<std::uint8_t> viewBytes = root->viewCanonicalBytes;
  const llvm::ArrayRef<std::uint8_t> viewDigestBytes = root->viewDigest;
  const llvm::ArrayRef<std::uint8_t> bitcodeDigestBytes =
      root->normalizedBitcodeDigest;
  const llvm::ArrayRef<std::uint8_t> moduleBytes = root->normalizedBitcode;

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

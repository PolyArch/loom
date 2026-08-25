#include "Evaluation/ModelParameterBundle.h"

#include "CanonicalSupport.h"
#include "Evaluation/ArtifactImportCache.h"
#include "Evaluation/ModelParameter.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace loom::evaluation {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "model_parameter_bundle_invalid: " + message);
}

class Decoder final {
public:
  explicit Decoder(llvm::ArrayRef<std::uint8_t> bytes) : remaining_(bytes) {}

  llvm::Expected<std::uint32_t> u32(llvm::StringRef description) {
    if (remaining_.size() < 4)
      return invalid(description + " is truncated");
    const std::uint32_t value =
        (static_cast<std::uint32_t>(remaining_[0]) << 24) |
        (static_cast<std::uint32_t>(remaining_[1]) << 16) |
        (static_cast<std::uint32_t>(remaining_[2]) << 8) |
        static_cast<std::uint32_t>(remaining_[3]);
    remaining_ = remaining_.drop_front(4);
    return value;
  }

  llvm::Expected<std::uint64_t> u64(llvm::StringRef description) {
    if (remaining_.size() < 8)
      return invalid(description + " is truncated");
    std::uint64_t value = 0;
    for (std::uint8_t byte : remaining_.take_front(8))
      value = (value << 8) | byte;
    remaining_ = remaining_.drop_front(8);
    return value;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>>
  bytes(std::uint64_t size, llvm::StringRef description) {
    if (size > remaining_.size())
      return invalid(description + " is truncated");
    llvm::ArrayRef<std::uint8_t> result = remaining_.take_front(size);
    remaining_ = remaining_.drop_front(size);
    return result;
  }

  llvm::ArrayRef<std::uint8_t> remaining() const { return remaining_; }

private:
  llvm::ArrayRef<std::uint8_t> remaining_;
};

struct DecodedBundleFields final {
  ModelParameterContractRef reference;
  BlobDigest digest;
};

llvm::Expected<DecodedBundleFields>
decodeCanonicalBundle(llvm::ArrayRef<std::uint8_t> bytes) {
  Decoder decoder(bytes);
  auto ownerSize = decoder.u64("owner identity length");
  if (!ownerSize)
    return ownerSize.takeError();
  auto ownerBytes = decoder.bytes(*ownerSize, "owner identity");
  if (!ownerBytes)
    return ownerBytes.takeError();
  const std::string owner(ownerBytes->begin(), ownerBytes->end());
  auto major = decoder.u32("owner registry major version");
  if (!major)
    return major.takeError();
  auto minor = decoder.u32("owner registry minor version");
  if (!minor)
    return minor.takeError();
  auto kind = decoder.u32("owner-local contract kind");
  if (!kind)
    return kind.takeError();
  if (decoder.remaining().size() != BlobDigest::byteSize)
    return invalid("canonical root does not end in one payload digest");
  auto digest = BlobDigest::fromBytes(decoder.remaining());
  if (!digest)
    return digest.takeError();
  auto reference = ModelParameterContractRef::get(
      owner, SchemaVersion{*major, *minor}, *kind);
  if (!reference)
    return reference.takeError();
  if (!findModelParameterContract(*reference))
    return invalid("parameter contract is unregistered");
  return DecodedBundleFields{std::move(*reference), std::move(*digest)};
}

llvm::Expected<OwnerValue>
adoptCanonicalParameters(const ModelParameterContractDescriptor &contract,
                         llvm::ArrayRef<std::uint8_t> payload) {
  if (contract.maximumPayloadBytes &&
      payload.size() > *contract.maximumPayloadBytes)
    return invalid("parameter payload exceeds its contract byte bound");
  auto parameters = contract.adopt(payload);
  if (!parameters)
    return parameters.takeError();
  if (!*parameters)
    return invalid("parameter contract adopted an empty owner value");
  auto encoded = contract.encode(*parameters);
  if (!encoded)
    return encoded.takeError();
  if (!std::equal(encoded->begin(), encoded->end(), payload.begin(),
                  payload.end()))
    return invalid("parameter payload is not canonical under its contract");
  auto targetKey = contract.parameterGroundTruthTargetKey(*parameters);
  if (!targetKey)
    return targetKey.takeError();
  if (targetKey->empty())
    return invalid("parameter payload has an empty ground-truth target key");
  return parameters;
}

} // namespace

CanonicalSemanticBytes
canonicalModelParameterBundleBytes(const ModelParameterBundle &bundle) {
  std::vector<std::uint8_t> bytes =
      canonicalModelParameterContractReferenceBytes(bundle.parameterContract());
  bytes.insert(bytes.end(), bundle.payloadDigest().bytes().begin(),
               bundle.payloadDigest().bytes().end());
  return CanonicalSemanticBytes(std::move(bytes));
}

std::string serializeModelParameterBundle(const ModelParameterBundle &bundle) {
  llvm::SmallString<256> storage;
  llvm::raw_svector_ostream stream(storage);
  llvm::json::OStream json(stream);
  json.object([&] {
    json.attribute("owner_registry_identity",
                   bundle.parameterContract().ownerRegistryIdentity());
    json.attribute("owner_registry_version_major",
                   bundle.parameterContract().ownerRegistryVersion().major);
    json.attribute("owner_registry_version_minor",
                   bundle.parameterContract().ownerRegistryVersion().minor);
    json.attribute("owner_local_contract_kind",
                   bundle.parameterContract().ownerLocalContractKind());
    json.attribute("payload_blob_digest",
                   formatBlobDigestHex(bundle.payloadDigest()));
  });
  return stream.str().str();
}

llvm::Expected<ModelParameterBundle>
parseModelParameterBundle(llvm::StringRef jsonText) {
  auto parsed = llvm::json::parse(jsonText);
  if (!parsed)
    return invalid("root is not valid JSON");
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("root must be an object");
  if (llvm::Error error = detail::rejectUnknownFields(
          *root, "ModelParameterBundle",
          {"owner_registry_identity", "owner_registry_version_major",
           "owner_registry_version_minor", "owner_local_contract_kind",
           "payload_blob_digest"}))
    return std::move(error);
  auto owner = detail::requireString(*root, "owner_registry_identity",
                                     "ModelParameterBundle");
  if (!owner)
    return owner.takeError();
  auto major = detail::requireUnsigned(*root, "owner_registry_version_major",
                                       "ModelParameterBundle");
  if (!major)
    return major.takeError();
  auto minor = detail::requireUnsigned(*root, "owner_registry_version_minor",
                                       "ModelParameterBundle");
  if (!minor)
    return minor.takeError();
  auto kind = detail::requireUnsigned(*root, "owner_local_contract_kind",
                                      "ModelParameterBundle");
  if (!kind)
    return kind.takeError();
  auto digestText = detail::requireString(*root, "payload_blob_digest",
                                          "ModelParameterBundle");
  if (!digestText)
    return digestText.takeError();
  if (*major > std::numeric_limits<std::uint32_t>::max() ||
      *minor > std::numeric_limits<std::uint32_t>::max() ||
      *kind > std::numeric_limits<std::uint32_t>::max())
    return invalid("contract reference integer exceeds uint32");
  auto reference = ModelParameterContractRef::get(
      *owner,
      SchemaVersion{static_cast<std::uint32_t>(*major),
                    static_cast<std::uint32_t>(*minor)},
      static_cast<std::uint32_t>(*kind));
  if (!reference)
    return reference.takeError();
  if (!findModelParameterContract(*reference))
    return invalid("parameter contract is unregistered");
  auto digest = parseBlobDigestHex(*digestText);
  if (!digest)
    return digest.takeError();
  ModelParameterBundle bundle(std::move(*reference), std::move(*digest));
  if (serializeModelParameterBundle(bundle) != jsonText)
    return invalid("JSON is not canonical");
  return bundle;
}

llvm::Expected<FinalizedModelParameterBundle>
finalizeModelParameterBundle(const ModelParameterContractRef &parameterContract,
                             const OwnerValue &parameters,
                             const ArtifactStore &artifactStore,
                             const BlobStore &blobStore) {
  const ModelParameterContractDescriptor *contract =
      findModelParameterContract(parameterContract);
  if (!contract)
    return invalid("parameter contract is unregistered");
  if (!parameters)
    return invalid("parameter owner value is empty");
  auto payload = contract->encode(parameters);
  if (!payload)
    return payload.takeError();
  auto adopted = adoptCanonicalParameters(*contract, *payload);
  if (!adopted)
    return adopted.takeError();
  auto digest = blobStore.put(*payload);
  if (!digest)
    return digest.takeError();
  ModelParameterBundle bundle(parameterContract, std::move(*digest));
  CanonicalSemanticBytes canonical = canonicalModelParameterBundleBytes(bundle);
  auto identity = artifactStore.put(modelParameterBundleSchema, canonical);
  if (!identity)
    return identity.takeError();
  return importModelParameterBundle({modelParameterBundleSchema.identity.str(),
                                     modelParameterBundleSchema.version,
                                     *identity},
                                    artifactStore, blobStore);
}

llvm::Expected<FinalizedModelParameterBundle>
importModelParameterBundle(const ArtifactRootReference &reference,
                           const ArtifactStore &artifactStore,
                           const BlobStore &blobStore) {
  const std::array<ArtifactRootReference, 1> references{reference};
  auto imported = importCachedArtifact<FinalizedModelParameterBundle>(
      artifactStore, &blobStore, references,
      [&]() -> llvm::Expected<FinalizedModelParameterBundle> {
        if (reference.schemaIdentity != modelParameterBundleSchema.identity ||
            reference.schemaVersion != modelParameterBundleSchema.version)
          return invalid("reference has the wrong Artifact schema");
        auto canonical =
            artifactStore.get(modelParameterBundleSchema, reference.artifact);
        if (!canonical)
          return canonical.takeError();
        auto decoded = decodeCanonicalBundle(canonical->bytes());
        if (!decoded)
          return decoded.takeError();
        ModelParameterBundle bundle(std::move(decoded->reference),
                                    std::move(decoded->digest));
        CanonicalSemanticBytes reencoded =
            canonicalModelParameterBundleBytes(bundle);
        if (!std::equal(reencoded.bytes().begin(), reencoded.bytes().end(),
                        canonical->bytes().begin(), canonical->bytes().end()))
          return invalid("stored bundle root is not canonical");
        const ModelParameterContractDescriptor *contract =
            findModelParameterContract(bundle.parameterContract());
        auto payload = contract->maximumPayloadBytes
                           ? blobStore.get(bundle.payloadDigest(),
                                           *contract->maximumPayloadBytes)
                           : blobStore.get(bundle.payloadDigest());
        if (!payload)
          return payload.takeError();
        auto parameters = adoptCanonicalParameters(*contract, *payload);
        if (!parameters)
          return parameters.takeError();
        return FinalizedModelParameterBundle(reference, std::move(*canonical),
                                             std::move(bundle),
                                             std::move(*parameters));
      },
      [&](const FinalizedModelParameterBundle &cached)
          -> llvm::Expected<std::uint64_t> {
        const ModelParameterContractDescriptor *contract =
            findModelParameterContract(cached.bundle().parameterContract());
        if (!contract)
          return invalid("parameter contract is unregistered");
        return contract->maximumPayloadBytes
                   ? blobStore.verify(cached.bundle().payloadDigest(),
                                      *contract->maximumPayloadBytes)
                   : blobStore.verify(cached.bundle().payloadDigest());
      });
  if (!imported)
    return imported.takeError();
  return **imported;
}

} // namespace loom::evaluation

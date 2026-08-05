#include "DSE/EvidenceObligationSetConfig.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace loom::dse {
namespace {

constexpr llvm::StringLiteral configDescriptor =
    "loom.evidence_obligation_set.config.1.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "evidence_obligation_set_config_invalid: " +
                                     message);
}

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(configDescriptor.data()),
          configDescriptor.size()};
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

llvm::Expected<std::uint32_t> readU32(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (bytes.size() - offset < 4)
    return invalid("truncated u32 field");
  std::uint32_t value = 0;
  for (unsigned ordinal = 0; ordinal != 4; ++ordinal)
    value = (value << 8) | bytes[offset++];
  return value;
}

llvm::Expected<std::uint64_t> readU64(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (bytes.size() - offset < 8)
    return invalid("truncated u64 field");
  std::uint64_t value = 0;
  for (unsigned ordinal = 0; ordinal != 8; ++ordinal)
    value = (value << 8) | bytes[offset++];
  return value;
}

llvm::Expected<std::vector<EvidenceObligationTemplateRef>>
canonicalReferences(llvm::ArrayRef<EvidenceObligationTemplateRef> references) {
  std::vector<EvidenceObligationTemplateRef> canonical(references.begin(),
                                                       references.end());
  llvm::sort(canonical, [](EvidenceObligationTemplateRef lhs,
                           EvidenceObligationTemplateRef rhs) {
    return lhs.ordinal() < rhs.ordinal();
  });
  if (std::adjacent_find(canonical.begin(), canonical.end()) != canonical.end())
    return invalid("Evidence obligation set contains a duplicate reference");
  return canonical;
}

std::vector<std::uint8_t>
encode(llvm::ArrayRef<EvidenceObligationTemplateRef> references) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(8 + references.size() * 4);
  appendU64(bytes, references.size());
  for (EvidenceObligationTemplateRef reference : references)
    appendU32(bytes, reference.ordinal());
  return bytes;
}

llvm::Expected<std::vector<EvidenceObligationTemplateRef>>
decode(llvm::ArrayRef<std::uint8_t> bytes) {
  std::size_t offset = 0;
  auto count = readU64(bytes, offset);
  if (!count)
    return count.takeError();
  if (*count > (bytes.size() - offset) / 4 ||
      *count > std::numeric_limits<std::size_t>::max())
    return invalid("Evidence obligation count exceeds remaining bytes");
  std::vector<EvidenceObligationTemplateRef> references;
  references.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t index = 0; index != *count; ++index) {
    auto ordinal = readU32(bytes, offset);
    if (!ordinal)
      return ordinal.takeError();
    references.emplace_back(*ordinal);
  }
  if (offset != bytes.size())
    return invalid("config has trailing bytes");
  auto canonical = canonicalReferences(references);
  if (!canonical)
    return canonical.takeError();
  if (*canonical != references)
    return invalid("Evidence obligation references are not canonical");
  return references;
}

} // namespace

llvm::ArrayRef<std::uint8_t> resolvedEvidenceObligationSetConfigSchemaBytes() {
  return descriptorBytes();
}

llvm::Expected<ResolvedEvidenceObligationSetConfigView>
projectResolvedEvidenceObligationSetConfigView(
    llvm::ArrayRef<EvidenceObligationTemplateRef> evidenceObligations) {
  auto canonical = canonicalReferences(evidenceObligations);
  if (!canonical)
    return canonical.takeError();
  std::vector<std::uint8_t> bytes = encode(*canonical);
  auto digest = computeComponentViewDigest(descriptorBytes(), bytes);
  if (!digest)
    return digest.takeError();
  return ResolvedEvidenceObligationSetConfigView(
      std::move(*canonical), std::move(bytes), std::move(*digest));
}

llvm::Expected<ResolvedEvidenceObligationSetConfigView>
adoptResolvedEvidenceObligationSetConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return invalid("config descriptor does not match the exact owner");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  auto references = decode(canonicalViewBytes);
  if (!references)
    return references.takeError();
  std::vector<std::uint8_t> reencoded = encode(*references);
  if (llvm::ArrayRef<std::uint8_t>(reencoded) != canonicalViewBytes)
    return invalid("decoded config does not re-encode to the source bytes");
  return ResolvedEvidenceObligationSetConfigView(std::move(*references),
                                                 std::move(reencoded), digest);
}

llvm::Error validateResolvedEvidenceObligationSetConfigView(
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  auto adopted = adoptResolvedEvidenceObligationSetConfigView(
      descriptorBytes(), canonicalViewBytes, digest);
  if (!adopted)
    return adopted.takeError();
  return llvm::Error::success();
}

llvm::Expected<std::vector<EvidenceObligationTemplateRef>>
resolveEvidenceObligationSetConfig(
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes) {
  return decode(canonicalViewBytes);
}

} // namespace loom::dse

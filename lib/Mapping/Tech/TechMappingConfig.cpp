#include "Mapping/Tech/TechMappingConfig.h"

#include "Config/ResolvedConfig.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <cstdint>

namespace loom::mapping {
namespace {

constexpr llvm::StringLiteral descriptor = "loom.tech_mapping.config.2.0";

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(descriptor.data()),
          descriptor.size()};
}

void writeU64(std::uint64_t value, llvm::MutableArrayRef<std::uint8_t> bytes) {
  for (unsigned ordinal = 0; ordinal < 8; ++ordinal)
    bytes[ordinal] = static_cast<std::uint8_t>(value >> (8 * (7 - ordinal)));
}

std::uint64_t readU64(llvm::ArrayRef<std::uint8_t> bytes) {
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes)
    value = (value << 8) | byte;
  return value;
}

llvm::Error validateLimits(llvm::ArrayRef<std::uint64_t> limits) {
  if (llvm::is_contained(limits, 0))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "tech_mapping_config_invalid: semantic limits must be positive");
  return llvm::Error::success();
}

} // namespace

llvm::ArrayRef<std::uint8_t> resolvedTechMappingConfigSchemaDescriptorBytes() {
  return descriptorBytes();
}

ResolvedTechMappingConfigView::ResolvedTechMappingConfigView(
    std::array<std::uint64_t, 4> limits)
    : limits_(limits) {
  for (auto [ordinal, value] : llvm::enumerate(limits_))
    writeU64(value, llvm::MutableArrayRef(bytes_).slice(ordinal * 8, 8));
}

llvm::ArrayRef<std::uint8_t>
ResolvedTechMappingConfigView::schemaDescriptorBytes() const {
  return resolvedTechMappingConfigSchemaDescriptorBytes();
}

ComponentViewDigest ResolvedTechMappingConfigView::digest() const {
  return llvm::cantFail(computeComponentViewDigest(schemaDescriptorBytes(),
                                                   canonicalViewBytes()));
}

llvm::Expected<ResolvedTechMappingConfigView>
projectResolvedTechMappingConfigView(const ResolvedConfig &config) {
  const std::array<std::uint64_t, 4> limits = {
      config.dse.techMapping.matchRowAttemptLimit,
      config.dse.techMapping.partialCoverExpansionLimit,
      config.dse.techMapping.candidateEvaluationLimit,
      config.dse.techMapping.candidatePublicationLimit};
  if (llvm::Error error = validateLimits(limits))
    return std::move(error);
  return ResolvedTechMappingConfigView(limits);
}

llvm::Expected<ResolvedTechMappingConfigView>
deriveTechMappingConfigWithPublicationLimit(
    const ResolvedTechMappingConfigView &config,
    std::uint64_t candidatePublicationLimit) {
  const std::array<std::uint64_t, 4> limits = {
      config.matchRowAttemptLimit(), config.partialCoverExpansionLimit(),
      config.candidateEvaluationLimit(),
      std::min(config.candidatePublicationLimit(),
               candidatePublicationLimit)};
  if (llvm::Error error = validateLimits(limits))
    return std::move(error);
  return ResolvedTechMappingConfigView(limits);
}

llvm::Expected<ResolvedTechMappingConfigView>
adoptResolvedTechMappingConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "tech_mapping_config_descriptor_mismatch: "
                                   "expected exact 2.0 descriptor");
  if (canonicalViewBytes.size() != 32)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "tech_mapping_config_bytes_invalid: expected four u64be fields");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  const std::array<std::uint64_t, 4> limits = {
      readU64(canonicalViewBytes.slice(0, 8)),
      readU64(canonicalViewBytes.slice(8, 8)),
      readU64(canonicalViewBytes.slice(16, 8)),
      readU64(canonicalViewBytes.slice(24, 8))};
  if (llvm::Error error = validateLimits(limits))
    return std::move(error);
  return ResolvedTechMappingConfigView(limits);
}

} // namespace loom::mapping

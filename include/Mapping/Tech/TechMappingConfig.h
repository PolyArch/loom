#ifndef LOOM_MAPPING_TECH_TECHMAPPINGCONFIG_H
#define LOOM_MAPPING_TECH_TECHMAPPINGCONFIG_H

#include "Common/ComponentViewDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>

namespace loom {
struct ResolvedConfig;
}

namespace loom::mapping {

llvm::ArrayRef<std::uint8_t> resolvedTechMappingConfigSchemaDescriptorBytes();

class ResolvedTechMappingConfigView final {
public:
  std::uint64_t matchRowAttemptLimit() const { return limits_[0]; }
  std::uint64_t partialCoverExpansionLimit() const { return limits_[1]; }
  std::uint64_t candidateEvaluationLimit() const { return limits_[2]; }
  std::uint64_t candidatePublicationLimit() const { return limits_[3]; }

  llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes() const;
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const { return bytes_; }
  ComponentViewDigest digest() const;

private:
  explicit ResolvedTechMappingConfigView(std::array<std::uint64_t, 4> limits);

  std::array<std::uint64_t, 4> limits_;
  std::array<std::uint8_t, 32> bytes_;

  friend llvm::Expected<ResolvedTechMappingConfigView>
  projectResolvedTechMappingConfigView(const ResolvedConfig &config);
  friend llvm::Expected<ResolvedTechMappingConfigView>
  adoptResolvedTechMappingConfigView(
      llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
      llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
      const ComponentViewDigest &digest);
};

llvm::Expected<ResolvedTechMappingConfigView>
projectResolvedTechMappingConfigView(const ResolvedConfig &config);

llvm::Expected<ResolvedTechMappingConfigView>
adoptResolvedTechMappingConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

} // namespace loom::mapping

#endif // LOOM_MAPPING_TECH_TECHMAPPINGCONFIG_H

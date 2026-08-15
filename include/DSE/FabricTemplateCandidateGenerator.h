#ifndef LOOM_DSE_FABRICTEMPLATECANDIDATEGENERATOR_H
#define LOOM_DSE_FABRICTEMPLATECANDIDATEGENERATOR_H

#include "ADG/BuiltinDescriptor.h"
#include "DSE/CandidateGenerator.h"

namespace loom {
struct ResolvedConfig;
}

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    fabricTemplateCandidateGeneratorKind(12);

class ResolvedFabricTemplateConfigView final {
public:
  const loom::adg::BuiltinTargetScale &scale() const { return scale_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedFabricTemplateConfigView(loom::adg::BuiltinTargetScale scale,
                                   std::vector<std::uint8_t> canonicalBytes,
                                   ComponentViewDigest digest)
      : scale_(scale), canonicalBytes_(std::move(canonicalBytes)),
        digest_(digest) {}

  loom::adg::BuiltinTargetScale scale_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedFabricTemplateConfigView>
  resolveFabricTemplateConfig(llvm::StringRef, std::uint32_t, std::uint32_t,
                              const loom::adg::BuiltinTargetScale &);
  friend llvm::Expected<ResolvedFabricTemplateConfigView>
  adoptResolvedFabricTemplateConfigView(llvm::ArrayRef<std::uint8_t>,
                                        llvm::ArrayRef<std::uint8_t>,
                                        const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t> resolvedFabricTemplateConfigSchemaBytes();
llvm::Expected<ResolvedFabricTemplateConfigView> resolveFabricTemplateConfig(
    llvm::StringRef templateIdentity, std::uint32_t schemaMajor,
    std::uint32_t schemaMinor, const loom::adg::BuiltinTargetScale &scale);
llvm::Expected<ResolvedFabricTemplateConfigView>
projectResolvedFabricTemplateConfigView(const ResolvedConfig &config);
llvm::Expected<ResolvedFabricTemplateConfigView>
adoptResolvedFabricTemplateConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const CandidateGeneratorDescriptor &
fabricTemplateCandidateGeneratorDescriptor();
llvm::Error registerFabricTemplateCandidateGenerator();
llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindFabricTemplateCandidateGeneratorInputs();
llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveFabricTemplateCandidateGeneratorBinding(
    const ResolvedFabricTemplateConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_FABRICTEMPLATECANDIDATEGENERATOR_H

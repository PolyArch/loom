#ifndef LOOM_DSE_FABRICTEMPLATECANDIDATEGENERATOR_H
#define LOOM_DSE_FABRICTEMPLATECANDIDATEGENERATOR_H

#include "ADG/BuiltinDescriptor.h"
#include "DSE/CandidateGenerator.h"

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    fabricTemplateCandidateGeneratorKind(12);

class ResolvedFabricTemplateConfigView final {
public:
  loom::adg::BuiltinTargetPreset preset() const { return preset_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedFabricTemplateConfigView(loom::adg::BuiltinTargetPreset preset,
                                   std::vector<std::uint8_t> canonicalBytes,
                                   ComponentViewDigest digest)
      : preset_(preset), canonicalBytes_(std::move(canonicalBytes)),
        digest_(digest) {}

  loom::adg::BuiltinTargetPreset preset_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedFabricTemplateConfigView>
      resolveFabricTemplateConfig(loom::adg::BuiltinTargetPreset);
  friend llvm::Expected<ResolvedFabricTemplateConfigView>
  adoptResolvedFabricTemplateConfigView(llvm::ArrayRef<std::uint8_t>,
                                        llvm::ArrayRef<std::uint8_t>,
                                        const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t> resolvedFabricTemplateConfigSchemaBytes();
llvm::Expected<ResolvedFabricTemplateConfigView>
resolveFabricTemplateConfig(loom::adg::BuiltinTargetPreset preset);
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

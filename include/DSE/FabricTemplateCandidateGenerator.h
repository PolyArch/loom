#ifndef LOOM_DSE_FABRICTEMPLATECANDIDATEGENERATOR_H
#define LOOM_DSE_FABRICTEMPLATECANDIDATEGENERATOR_H

#include "ADG/BuiltinDescriptor.h"
#include "Common/Artifact.h"
#include "DSE/CandidateGenerator.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom {
struct ResolvedConfig;
}

namespace loom::dse {

inline constexpr CandidateGeneratorKind
    fabricTemplateCandidateGeneratorKind(12);

/// One mined composite FU template a built Module offers, named by the
/// canonical shape key mining assigns it, together with the number of Spatial
/// PE sites that receive an occurrence.
struct MinedCompositeFuSelectionEntry final {
  std::vector<std::uint8_t> shapeKey;
  std::uint32_t occurrences = 0;
};

/// The typed decision domain that lets a builtin template offer mined
/// composite FUs. It names the exact canonical Dataflow the shapes were mined
/// from and the selected shapes; it carries no FU structure at all. The miner
/// and the canonical capability derivation stay the only owners of that
/// structure, and the generator re-derives it from this selection, so the
/// config can never disagree with them.
struct MinedCompositeFuSelection final {
  ArtifactIdentity dataflow;
  std::vector<MinedCompositeFuSelectionEntry> templates;
};

class ResolvedFabricTemplateConfigView final {
public:
  const loom::adg::BuiltinTargetScale &scale() const { return scale_; }
  const std::optional<MinedCompositeFuSelection> &
  minedCompositeFus() const {
    return minedCompositeFus_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedFabricTemplateConfigView(
      loom::adg::BuiltinTargetScale scale,
      std::optional<MinedCompositeFuSelection> minedCompositeFus,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : scale_(scale), minedCompositeFus_(std::move(minedCompositeFus)),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  loom::adg::BuiltinTargetScale scale_;
  std::optional<MinedCompositeFuSelection> minedCompositeFus_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedFabricTemplateConfigView>
  resolveFabricTemplateConfig(llvm::StringRef, std::uint32_t, std::uint32_t,
                              const loom::adg::BuiltinTargetScale &,
                              const std::optional<MinedCompositeFuSelection> &);
  friend llvm::Expected<ResolvedFabricTemplateConfigView>
  adoptResolvedFabricTemplateConfigView(llvm::ArrayRef<std::uint8_t>,
                                        llvm::ArrayRef<std::uint8_t>,
                                        const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t> resolvedFabricTemplateConfigSchemaBytes();
llvm::Expected<ResolvedFabricTemplateConfigView> resolveFabricTemplateConfig(
    llvm::StringRef templateIdentity, std::uint32_t schemaMajor,
    std::uint32_t schemaMinor, const loom::adg::BuiltinTargetScale &scale,
    const std::optional<MinedCompositeFuSelection> &minedCompositeFus = {});
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
/// Binds the canonical Dataflow a mined selection names. A config without a
/// mined selection binds nothing, which is why the slot admits zero or one
/// artifact rather than exactly one.
llvm::Expected<std::vector<CandidateGeneratorInputBinding>>
bindFabricTemplateCandidateGeneratorInputs(
    const std::optional<ArtifactRootReference> &dataflow = std::nullopt);
llvm::Expected<ResolvedCandidateGeneratorBinding>
resolveFabricTemplateCandidateGeneratorBinding(
    const ResolvedFabricTemplateConfigView &config);

} // namespace loom::dse

#endif // LOOM_DSE_FABRICTEMPLATECANDIDATEGENERATOR_H

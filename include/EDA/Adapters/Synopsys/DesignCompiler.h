#ifndef LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILER_H
#define LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILER_H

#include "EDA/Adapters/Synopsys/Common.h"

#include "Common/ComponentViewDigest.h"
#include "Common/ExternalFileFingerprint.h"
#include "DSE/CandidateGenerator.h"
#include "ExternalTool/Provider.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <utility>
#include <vector>

namespace loom::eda::synopsys {

inline constexpr llvm::StringLiteral designCompilerGateNetlistOutputPath =
    "outputs/design-compiler-gate-netlist.v";

inline constexpr dse::CandidateGeneratorKind
    designCompilerGateNetlistCandidateGeneratorKind(0x53444347);

class ResolvedDesignCompilerGateNetlistConfigView final {
public:
  llvm::StringRef stableProviderBuildIdentity() const {
    return stableProviderBuildIdentity_;
  }
  const platform::TechnologyCornerRef &technologyCorner() const {
    return technologyCorner_;
  }
  const ExternalFileFingerprint &standardCellLiberty() const {
    return standardCellLiberty_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedDesignCompilerGateNetlistConfigView(
      std::string stableProviderBuildIdentity,
      platform::TechnologyCornerRef technologyCorner,
      ExternalFileFingerprint standardCellLiberty,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : stableProviderBuildIdentity_(std::move(stableProviderBuildIdentity)),
        technologyCorner_(std::move(technologyCorner)),
        standardCellLiberty_(std::move(standardCellLiberty)),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::string stableProviderBuildIdentity_;
  platform::TechnologyCornerRef technologyCorner_;
  ExternalFileFingerprint standardCellLiberty_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedDesignCompilerGateNetlistConfigView>
      createResolvedDesignCompilerGateNetlistConfigView(
          llvm::StringRef, platform::TechnologyCornerRef,
          ExternalFileFingerprint);
  friend llvm::Expected<ResolvedDesignCompilerGateNetlistConfigView>
  adoptResolvedDesignCompilerGateNetlistConfigView(llvm::ArrayRef<std::uint8_t>,
                                                   llvm::ArrayRef<std::uint8_t>,
                                                   const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedDesignCompilerGateNetlistConfigSchemaDescriptorBytes();

llvm::Expected<ResolvedDesignCompilerGateNetlistConfigView>
createResolvedDesignCompilerGateNetlistConfigView(
    llvm::StringRef stableProviderBuildIdentity,
    platform::TechnologyCornerRef technologyCorner,
    ExternalFileFingerprint standardCellLiberty);

llvm::Expected<ResolvedDesignCompilerGateNetlistConfigView>
adoptResolvedDesignCompilerGateNetlistConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const dse::CandidateGeneratorDescriptor &
designCompilerGateNetlistCandidateGeneratorDescriptor();

llvm::Error registerDesignCompilerGateNetlistCandidateGenerator();

llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindDesignCompilerGateNetlistInputs(
    const ArtifactRootReference &rtlImplementation,
    const ArtifactRootReference &implementationPlatform);

llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveDesignCompilerGateNetlistBinding(
    const ResolvedDesignCompilerGateNetlistConfigView &config);

llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareDesignCompilerGateNetlistInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const external_tool::ExternalToolPreparationContext &context);

llvm::Expected<dse::CandidateGeneratorProviderResult>
importDesignCompilerGateNetlistInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<hardware::ExternalImplementationContractCatalog>
makeSynopsysStandardCellContractCatalog();

llvm::Expected<std::string> renderSynopsysStandardCellBlackBoxContract(
    const ExternalFileFingerprint &standardCellLiberty,
    llvm::ArrayRef<hardware::RepresentationLocator> unresolvedDefinitions);

llvm::Expected<hardware::FinalizedHardwareImplementation>
importDesignCompilerGateNetlistImplementation(
    const ArtifactRootReference &reference, const ArtifactStore &artifacts,
    const BlobStore &blobs);

struct DesignCompilerGateNetlist final {
  std::string verilog;
};

const SynopsysInvocationDescriptor &designCompilerDescriptor();

/// The source view selects whether synthesis may change definition boundaries.
/// This is fixed by the registered implementation or block generator.
enum class DesignCompilerHierarchy : std::uint8_t {
  Optimize,
  PreserveDefinitions,
};

/// Shared nonsemantic resolution boundary for Design Compiler generators.
llvm::Expected<SynopsysFrozenInvocation> resolveDesignCompilerInvocation(
    const ResolvedDesignCompilerGateNetlistConfigView &config,
    const external_tool::ExternalToolPreparationContext &context);

llvm::Expected<std::string> renderDesignCompilerDriver(
    llvm::StringRef top, llvm::ArrayRef<std::string> rtlSources,
    llvm::ArrayRef<std::string> generationConstraints,
    llvm::StringRef targetLibrary,
    DesignCompilerHierarchy hierarchy = DesignCompilerHierarchy::Optimize);

llvm::Expected<DesignCompilerGateNetlist>
parseDesignCompilerGateNetlist(llvm::StringRef contents, llvm::StringRef top);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeDesignCompilerBundleSpec(const SynopsysBundleInputs &inputs,
                             llvm::StringRef top,
                             llvm::ArrayRef<std::string> rtlSources,
                             llvm::ArrayRef<std::string> generationConstraints);

llvm::Expected<DesignCompilerGateNetlist> importDesignCompilerGateNetlist(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs, llvm::StringRef top);

} // namespace loom::eda::synopsys

#endif // LOOM_EDA_ADAPTERS_SYNOPSYS_DESIGNCOMPILER_H

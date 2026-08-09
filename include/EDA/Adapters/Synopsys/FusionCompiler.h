#ifndef LOOM_EDA_ADAPTERS_SYNOPSYS_FUSIONCOMPILER_H
#define LOOM_EDA_ADAPTERS_SYNOPSYS_FUSIONCOMPILER_H

#include "EDA/Adapters/Synopsys/Common.h"

#include "Common/ComponentViewDigest.h"
#include "DSE/CandidateGenerator.h"
#include "ExternalTool/ExternalFile.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::synopsys {

inline constexpr dse::CandidateGeneratorKind
    fusionCompilerRoutedCandidateGeneratorKind(0x53464352);

class ResolvedFusionCompilerRoutedConfigView final {
public:
  llvm::StringRef stableProviderBuildIdentity() const {
    return stableProviderBuildIdentity_;
  }
  const platform::TechnologyCornerRef &technologyCorner() const {
    return technologyCorner_;
  }
  llvm::ArrayRef<external_tool::ExternalFileTreeMember>
  referenceLibraryMembers() const {
    return referenceLibraryMembers_;
  }
  const ExternalFileFingerprint &earlyParasiticTech() const {
    return earlyParasiticTech_;
  }
  const ExternalFileFingerprint &lateParasiticTech() const {
    return lateParasiticTech_;
  }
  const ExternalFileFingerprint &parasiticLayerMap() const {
    return parasiticLayerMap_;
  }
  llvm::StringRef floorplanDef() const { return floorplanDef_; }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalBytes_;
  }
  const ComponentViewDigest &digest() const { return digest_; }

private:
  ResolvedFusionCompilerRoutedConfigView(
      std::string stableProviderBuildIdentity,
      platform::TechnologyCornerRef technologyCorner,
      std::vector<external_tool::ExternalFileTreeMember>
          referenceLibraryMembers,
      ExternalFileFingerprint earlyParasiticTech,
      ExternalFileFingerprint lateParasiticTech,
      ExternalFileFingerprint parasiticLayerMap, std::string floorplanDef,
      std::vector<std::uint8_t> canonicalBytes, ComponentViewDigest digest)
      : stableProviderBuildIdentity_(std::move(stableProviderBuildIdentity)),
        technologyCorner_(std::move(technologyCorner)),
        referenceLibraryMembers_(std::move(referenceLibraryMembers)),
        earlyParasiticTech_(std::move(earlyParasiticTech)),
        lateParasiticTech_(std::move(lateParasiticTech)),
        parasiticLayerMap_(std::move(parasiticLayerMap)),
        floorplanDef_(std::move(floorplanDef)),
        canonicalBytes_(std::move(canonicalBytes)), digest_(digest) {}

  std::string stableProviderBuildIdentity_;
  platform::TechnologyCornerRef technologyCorner_;
  std::vector<external_tool::ExternalFileTreeMember> referenceLibraryMembers_;
  ExternalFileFingerprint earlyParasiticTech_;
  ExternalFileFingerprint lateParasiticTech_;
  ExternalFileFingerprint parasiticLayerMap_;
  std::string floorplanDef_;
  std::vector<std::uint8_t> canonicalBytes_;
  ComponentViewDigest digest_;

  friend llvm::Expected<ResolvedFusionCompilerRoutedConfigView>
      createResolvedFusionCompilerRoutedConfigView(
          llvm::StringRef, platform::TechnologyCornerRef,
          std::vector<external_tool::ExternalFileTreeMember>,
          ExternalFileFingerprint, ExternalFileFingerprint,
          ExternalFileFingerprint, llvm::StringRef);
  friend llvm::Expected<ResolvedFusionCompilerRoutedConfigView>
  adoptResolvedFusionCompilerRoutedConfigView(llvm::ArrayRef<std::uint8_t>,
                                              llvm::ArrayRef<std::uint8_t>,
                                              const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedFusionCompilerRoutedConfigSchemaDescriptorBytes();

llvm::Expected<ResolvedFusionCompilerRoutedConfigView>
createResolvedFusionCompilerRoutedConfigView(
    llvm::StringRef stableProviderBuildIdentity,
    platform::TechnologyCornerRef technologyCorner,
    std::vector<external_tool::ExternalFileTreeMember> referenceLibraryMembers,
    ExternalFileFingerprint earlyParasiticTech,
    ExternalFileFingerprint lateParasiticTech,
    ExternalFileFingerprint parasiticLayerMap, llvm::StringRef floorplanDef);

llvm::Expected<ResolvedFusionCompilerRoutedConfigView>
adoptResolvedFusionCompilerRoutedConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const dse::CandidateGeneratorDescriptor &
fusionCompilerRoutedCandidateGeneratorDescriptor();

llvm::Error registerFusionCompilerRoutedCandidateGenerator();

llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindFusionCompilerRoutedInputs(
    const ArtifactRootReference &gateNetlistImplementation,
    const ArtifactRootReference &implementationPlatform);

llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveFusionCompilerRoutedBinding(
    const ResolvedFusionCompilerRoutedConfigView &config);

llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareFusionCompilerRoutedInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const external_tool::ExternalToolPreparationContext &context);

llvm::Expected<dse::CandidateGeneratorProviderResult>
importFusionCompilerRoutedInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<hardware::FinalizedHardwareImplementation>
importFusionCompilerRoutedImplementation(const ArtifactRootReference &reference,
                                         const ArtifactStore &artifacts,
                                         const BlobStore &blobs);

struct FusionCompilerPhysicalSnapshot final {
  hardware::RepresentationPhysicalStage stage;
  std::string netlistVerilog;
  std::string designExchangeFormat;
  std::string generationConstraints;
};

const SynopsysInvocationDescriptor &fusionCompilerDescriptor();

llvm::Expected<std::string> renderFusionCompilerDriver(
    llvm::StringRef top, llvm::StringRef gateNetlist,
    llvm::StringRef generationConstraint, llvm::StringRef floorplan,
    llvm::StringRef referenceLibrary, llvm::StringRef earlyParasiticTech,
    llvm::StringRef lateParasiticTech, llvm::StringRef parasiticLayerMap);

llvm::Expected<FusionCompilerPhysicalSnapshot>
parseFusionCompilerPhysicalSnapshot(
    llvm::StringRef netlist, llvm::StringRef designExchangeFormat,
    llvm::StringRef generationConstraints, llvm::StringRef top,
    hardware::RepresentationPhysicalStage stage);

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeFusionCompilerBundleSpec(const SynopsysBundleInputs &inputs,
                             llvm::StringRef top, llvm::StringRef gateNetlist,
                             llvm::StringRef generationConstraint,
                             llvm::StringRef floorplan);

llvm::Expected<FusionCompilerPhysicalSnapshot>
importFusionCompilerPhysicalSnapshot(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs, llvm::StringRef top);

llvm::Expected<hardware::FinalizedHardwareImplementation>
publishFusionCompilerPhysicalImplementation(
    const hardware::FinalizedHardwareImplementation &source,
    const FusionCompilerPhysicalSnapshot &snapshot,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<hardware::FinalizedHardwareImplementation>
publishFusionCompilerPhysicalImplementation(
    const hardware::FinalizedHardwareImplementation &source,
    const FusionCompilerPhysicalSnapshot &snapshot,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::eda::synopsys

#endif // LOOM_EDA_ADAPTERS_SYNOPSYS_FUSIONCOMPILER_H

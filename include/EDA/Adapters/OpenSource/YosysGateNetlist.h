#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_YOSYSGATENETLIST_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_YOSYSGATENETLIST_H

#include "Common/ComponentViewDigest.h"
#include "Common/ExternalFileFingerprint.h"
#include "DSE/CandidateGenerator.h"
#include "ExternalTool/Provider.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::open_source {

inline constexpr dse::CandidateGeneratorKind
    yosysGateNetlistCandidateGeneratorKind(0x59535953);

class ResolvedYosysGateNetlistConfigView final {
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
  ResolvedYosysGateNetlistConfigView(
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

  friend llvm::Expected<ResolvedYosysGateNetlistConfigView>
      createResolvedYosysGateNetlistConfigView(llvm::StringRef,
                                               platform::TechnologyCornerRef,
                                               ExternalFileFingerprint);
  friend llvm::Expected<ResolvedYosysGateNetlistConfigView>
  adoptResolvedYosysGateNetlistConfigView(llvm::ArrayRef<std::uint8_t>,
                                          llvm::ArrayRef<std::uint8_t>,
                                          const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedYosysGateNetlistConfigSchemaDescriptorBytes();

llvm::Expected<ResolvedYosysGateNetlistConfigView>
createResolvedYosysGateNetlistConfigView(
    llvm::StringRef stableProviderBuildIdentity,
    platform::TechnologyCornerRef technologyCorner,
    ExternalFileFingerprint standardCellLiberty);

llvm::Expected<ResolvedYosysGateNetlistConfigView>
adoptResolvedYosysGateNetlistConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const dse::CandidateGeneratorDescriptor &
yosysGateNetlistCandidateGeneratorDescriptor();

llvm::Error registerYosysGateNetlistCandidateGenerator();

llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindYosysGateNetlistInputs(const ArtifactRootReference &rtlImplementation,
                           const ArtifactRootReference &implementationPlatform);

llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveYosysGateNetlistBinding(
    const ResolvedYosysGateNetlistConfigView &config);

llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareYosysGateNetlistInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const external_tool::ExternalToolPreparationContext &context);

llvm::Expected<dse::CandidateGeneratorProviderResult>
importYosysGateNetlistInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

llvm::Expected<hardware::ExternalImplementationContractCatalog>
makeYosysStandardCellContractCatalog();

/// Imports a Yosys-owned GateNetlist through its exact external dependency
/// contract. Generic HardwareImplementation import cannot infer that contract.
llvm::Expected<hardware::FinalizedHardwareImplementation>
importYosysGateNetlistImplementation(const ArtifactRootReference &reference,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs);

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_YOSYSGATENETLIST_H

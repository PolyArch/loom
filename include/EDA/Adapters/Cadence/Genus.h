#ifndef LOOM_EDA_ADAPTERS_CADENCE_GENUS_H
#define LOOM_EDA_ADAPTERS_CADENCE_GENUS_H

#include "Common/ComponentViewDigest.h"
#include "Common/ExternalFileFingerprint.h"
#include "DSE/CandidateGenerator.h"
#include "Hardware/Implementation/HardwareImplementation.h"
#include "ImplementationPlatform/TechnologyCorner.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::cadence {

inline constexpr dse::CandidateGeneratorKind
    genusGateNetlistCandidateGeneratorKind(0x4347454e);

class ResolvedGenusGateNetlistConfigView final {
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
  ResolvedGenusGateNetlistConfigView(
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

  friend llvm::Expected<ResolvedGenusGateNetlistConfigView>
      createResolvedGenusGateNetlistConfigView(llvm::StringRef,
                                               platform::TechnologyCornerRef,
                                               ExternalFileFingerprint);
  friend llvm::Expected<ResolvedGenusGateNetlistConfigView>
  adoptResolvedGenusGateNetlistConfigView(llvm::ArrayRef<std::uint8_t>,
                                          llvm::ArrayRef<std::uint8_t>,
                                          const ComponentViewDigest &);
};

llvm::ArrayRef<std::uint8_t>
resolvedGenusGateNetlistConfigSchemaDescriptorBytes();

llvm::Expected<ResolvedGenusGateNetlistConfigView>
createResolvedGenusGateNetlistConfigView(
    llvm::StringRef stableProviderBuildIdentity,
    platform::TechnologyCornerRef technologyCorner,
    ExternalFileFingerprint standardCellLiberty);

llvm::Expected<ResolvedGenusGateNetlistConfigView>
adoptResolvedGenusGateNetlistConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

const dse::CandidateGeneratorDescriptor &
genusGateNetlistCandidateGeneratorDescriptor();

llvm::Error registerGenusGateNetlistCandidateGenerator();

llvm::Expected<std::vector<dse::CandidateGeneratorInputBinding>>
bindGenusGateNetlistInputs(const ArtifactRootReference &rtlImplementation,
                           const ArtifactRootReference &implementationPlatform);

llvm::Expected<dse::ResolvedCandidateGeneratorBinding>
resolveGenusGateNetlistBinding(
    const ResolvedGenusGateNetlistConfigView &config);

struct GenusGateNetlist final {
  std::string verilog;
};

llvm::Expected<std::string>
renderGenusGateNetlistDriver(llvm::StringRef top,
                             llvm::ArrayRef<std::string> rtlSources,
                             llvm::ArrayRef<std::string> generationConstraints,
                             llvm::StringRef standardCellLiberty);

llvm::Expected<GenusGateNetlist> parseGenusGateNetlist(llvm::StringRef contents,
                                                       llvm::StringRef top);

llvm::Expected<hardware::ExternalImplementationContractCatalog>
makeCadenceStandardCellContractCatalog();

/// Imports a Genus-owned GateNetlist through its exact external dependency
/// contract. Generic HardwareImplementation import cannot infer that contract.
llvm::Expected<hardware::FinalizedHardwareImplementation>
importGenusGateNetlistImplementation(const ArtifactRootReference &reference,
                                     const ArtifactStore &artifacts,
                                     const BlobStore &blobs);

} // namespace loom::eda::cadence

#endif // LOOM_EDA_ADAPTERS_CADENCE_GENUS_H

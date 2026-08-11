#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_OPENROAD_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_OPENROAD_H

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
#include <vector>

namespace loom::eda::open_source {

struct OpenRoadRectangleNanometers final {
  std::uint64_t lowerXNanometers;
  std::uint64_t lowerYNanometers;
  std::uint64_t upperXNanometers;
  std::uint64_t upperYNanometers;

  friend bool operator==(const OpenRoadRectangleNanometers &lhs,
                         const OpenRoadRectangleNanometers &rhs) {
    return lhs.lowerXNanometers == rhs.lowerXNanometers &&
           lhs.lowerYNanometers == rhs.lowerYNanometers &&
           lhs.upperXNanometers == rhs.upperXNanometers &&
           lhs.upperYNanometers == rhs.upperYNanometers;
  }
};

/// Exact result-affecting values owned by the placed-state generator binding.
/// Coordinates use integer nanometers; density uses parts per million.
struct OpenRoadPlacementParameters final {
  OpenRoadRectangleNanometers dieArea;
  OpenRoadRectangleNanometers coreArea;
  std::string siteName;
  std::string horizontalPinLayer;
  std::string verticalPinLayer;
  std::uint32_t placementDensityPpm;

  friend bool operator==(const OpenRoadPlacementParameters &lhs,
                         const OpenRoadPlacementParameters &rhs) {
    return lhs.dieArea == rhs.dieArea && lhs.coreArea == rhs.coreArea &&
           lhs.siteName == rhs.siteName &&
           lhs.horizontalPinLayer == rhs.horizontalPinLayer &&
           lhs.verticalPinLayer == rhs.verticalPinLayer &&
           lhs.placementDensityPpm == rhs.placementDensityPpm;
  }
};

enum class OpenRoadExternalFileKind : std::uint8_t {
  TechnologyLef = 0,
  CellLef = 1,
  Liberty = 2,
};

struct OpenRoadExternalFile final {
  OpenRoadExternalFileKind kind;
  std::string logicalName;
  ExternalFileFingerprint fingerprint;

  friend bool operator==(const OpenRoadExternalFile &lhs,
                         const OpenRoadExternalFile &rhs) {
    return lhs.kind == rhs.kind && lhs.logicalName == rhs.logicalName &&
           lhs.fingerprint == rhs.fingerprint;
  }
};

/// The complete semantic binding for one placed-state attempt. Machine-local
/// paths and a resolved executable are deliberately absent.
struct OpenRoadPlacedConfig final {
  std::string providerBuild;
  platform::TechnologyCornerRef corner;
  OpenRoadPlacementParameters placement;
  std::vector<OpenRoadExternalFile> externalFiles;

  friend bool operator==(const OpenRoadPlacedConfig &lhs,
                         const OpenRoadPlacedConfig &rhs) {
    return lhs.providerBuild == rhs.providerBuild && lhs.corner == rhs.corner &&
           lhs.placement == rhs.placement &&
           lhs.externalFiles == rhs.externalFiles;
  }
};

llvm::Error validateOpenRoadPlacementParameters(
    const OpenRoadPlacementParameters &parameters);

std::string openRoadExternalFileInputSlot(const OpenRoadExternalFile &file);

llvm::ArrayRef<std::uint8_t> openRoadPlacedConfigSchemaDescriptorBytes();

llvm::Expected<std::vector<std::uint8_t>>
encodeOpenRoadPlacedConfig(const OpenRoadPlacedConfig &config);

llvm::Expected<OpenRoadPlacedConfig>
decodeOpenRoadPlacedConfig(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Error validateCanonicalOpenRoadPlacedConfig(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ComponentViewDigest &suppliedDigest);

inline constexpr dse::CandidateGeneratorKind
    openRoadPlacedCandidateGeneratorKind(11);

const dse::CandidateGeneratorDescriptor &
openRoadPlacedCandidateGeneratorDescriptor();

/// Registers only the semantic descriptor for callers that inject an already
/// resolved execution closure through the direct helper below.
llvm::Error registerOpenRoadPlacedCandidateGeneratorDescriptor();

/// Registers the canonical provider facade. Preparation resolves and freezes
/// the configured OpenROAD tool and runtime before delegating to the same
/// strict invocation lifecycle used by the direct helper.
llvm::Error registerOpenRoadPlacedCandidateGenerator();

/// A coordinator-frozen nonsemantic execution closure. Construction and tool
/// discovery happen before adapter preparation; preparation only validates
/// this value, resolves declared external files, and finalizes a fresh bundle.
struct OpenRoadResolvedExecution final {
  external_tool::ExternalToolProviderDescriptor provider;
  external_tool::ResolvedToolBinding tool;
  external_tool::InvocationRuntimeBinding runtime;
  external_tool::ToolVersionProbe containerVersionProbe;
};

llvm::Expected<OpenRoadResolvedExecution> resolveOpenRoadExecution(
    llvm::StringRef providerBuild,
    const external_tool::ExternalToolPreparationContext &context);

llvm::Error
validateOpenRoadResolvedExecution(const OpenRoadResolvedExecution &execution,
                                  llvm::StringRef providerBuild);

llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareOpenRoadPlacedInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const OpenRoadResolvedExecution &execution,
    const external_tool::ExternalToolPreparationContext &context);

/// Strictly imports one exact attempt. A successful placed result publishes an
/// indexed AsicPhysical HardwareImplementation. Ordinary provider failures
/// remain dense non-publishing outcomes; integrity and incomplete attempts are
/// errors.
llvm::Expected<dse::CandidateGeneratorProviderResult>
importOpenRoadPlacedInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

/// The exact driver projection. Artifact payloads use bundle-relative paths;
/// external files use the frozen absolute paths recorded by the invocation
/// manifest.
struct OpenRoadPlacedDriverFiles final {
  std::vector<std::string> netlists;
  std::vector<std::string> constraints;
  std::string technologyLef;
  std::vector<std::string> cellLefs;
  std::vector<std::string> libertyFiles;
};

/// Renders the deterministic OpenROAD placed-state driver. It performs no
/// routing, extraction, timing, power, physical-verification, or signoff
/// operation and writes only outputs/placed.odb and the canonical result
/// marker outputs/placed-result.json.
llvm::Expected<std::string>
renderOpenRoadPlacedDriver(llvm::StringRef topModule,
                           const OpenRoadPlacementParameters &parameters,
                           const OpenRoadPlacedDriverFiles &files);

/// Ephemeral proof that the declared placed-state driver reached its final
/// publication point. It is neither HardwareImplementation nor Evidence.
struct OpenRoadPlacedAttemptResult final {
  std::string topModule;

  friend bool operator==(const OpenRoadPlacedAttemptResult &lhs,
                         const OpenRoadPlacedAttemptResult &rhs) {
    return lhs.topModule == rhs.topModule;
  }
};

/// Strictly parses the authored result protocol and rejects noncanonical JSON,
/// later-stage claims, and unknown fields.
llvm::Expected<OpenRoadPlacedAttemptResult>
parseOpenRoadPlacedAttemptResult(llvm::StringRef contents);

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_OPENROAD_H

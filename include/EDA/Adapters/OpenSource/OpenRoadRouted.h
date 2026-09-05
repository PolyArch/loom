#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_OPENROADROUTED_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_OPENROADROUTED_H

#include "EDA/Adapters/OpenSource/OpenRoad.h"

#include <optional>

namespace loom::eda::open_source {

/// Explicit routing policy over names owned by the exact technology LEF.
/// Pins at or below viaAccessCutoffLayer require via access; a missing cutoff
/// retains the pinned native provider's policy.
struct OpenRoadRoutingParameters final {
  std::string minimumRoutingLayer;
  std::string maximumRoutingLayer;
  std::optional<std::string> viaAccessCutoffLayer;

  friend bool operator==(const OpenRoadRoutingParameters &lhs,
                         const OpenRoadRoutingParameters &rhs) {
    return lhs.minimumRoutingLayer == rhs.minimumRoutingLayer &&
           lhs.maximumRoutingLayer == rhs.maximumRoutingLayer &&
           lhs.viaAccessCutoffLayer == rhs.viaAccessCutoffLayer;
  }
};

struct OpenRoadRoutedConfig final {
  OpenRoadPlacedConfig physical;
  OpenRoadRoutingParameters routing;

  friend bool operator==(const OpenRoadRoutedConfig &lhs,
                         const OpenRoadRoutedConfig &rhs) {
    return lhs.physical == rhs.physical && lhs.routing == rhs.routing;
  }
};

llvm::Error
validateOpenRoadRoutingParameters(const OpenRoadRoutingParameters &parameters);
llvm::Expected<OpenRoadRoutingParameters>
parseOpenRoadRoutingParametersJson(llvm::StringRef json);
llvm::ArrayRef<std::uint8_t> openRoadRoutedConfigSchemaDescriptorBytes();
llvm::Expected<std::vector<std::uint8_t>>
encodeOpenRoadRoutedConfig(const OpenRoadRoutedConfig &config);
llvm::Expected<OpenRoadRoutedConfig>
decodeOpenRoadRoutedConfig(llvm::ArrayRef<std::uint8_t> bytes);
llvm::Error validateCanonicalOpenRoadRoutedConfig(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ComponentViewDigest &suppliedDigest);

inline constexpr dse::CandidateGeneratorKind
    openRoadRoutedCandidateGeneratorKind(0x4f525254);

const dse::CandidateGeneratorDescriptor &
openRoadRoutedCandidateGeneratorDescriptor();

llvm::Error registerOpenRoadRoutedCandidateGeneratorDescriptor();
llvm::Error registerOpenRoadRoutedCandidateGenerator();

enum class OpenRoadRoutedInputKind : std::uint8_t {
  GateNetlist,
  PlacedDatabase,
};

struct OpenRoadRoutedDriverFiles final {
  OpenRoadRoutedInputKind inputKind;
  std::vector<std::string> netlists;
  std::optional<std::string> placedDatabase;
  std::vector<std::string> constraints;
  std::string technologyLef;
  std::vector<std::string> cellLefs;
  std::vector<std::string> libertyFiles;
};

llvm::Expected<std::string>
renderOpenRoadRoutedDriver(llvm::StringRef topModule,
                           const OpenRoadPlacementParameters &parameters,
                           const OpenRoadRoutingParameters &routing,
                           const OpenRoadRoutedDriverFiles &files);

struct OpenRoadRoutedAttemptResult final {
  std::string topModule;

  friend bool operator==(const OpenRoadRoutedAttemptResult &lhs,
                         const OpenRoadRoutedAttemptResult &rhs) {
    return lhs.topModule == rhs.topModule;
  }
};

llvm::Expected<OpenRoadRoutedAttemptResult>
parseOpenRoadRoutedAttemptResult(llvm::StringRef contents);

llvm::Expected<external_tool::PreparedExternalToolInvocation>
prepareOpenRoadRoutedInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs,
    const OpenRoadResolvedExecution &execution,
    const external_tool::ExternalToolPreparationContext &context);

llvm::Expected<dse::CandidateGeneratorProviderResult>
importOpenRoadRoutedInvocation(
    llvm::ArrayRef<dse::CandidateGeneratorInputBinding> inputs,
    const dse::ResolvedCandidateGeneratorBinding &binding,
    const external_tool::PreparedExternalToolInvocation &prepared,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_OPENROADROUTED_H

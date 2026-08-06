#ifndef LOOM_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAIN_H
#define LOOM_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAIN_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "Mapping/Artifact/SystemPresburger.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {

namespace detail {
struct SystemPnrSearchDomainViewBuilder;
}

class SystemPnrSearchDomainDigest final {
public:
  using Storage = std::array<std::uint8_t, 32>;
  static constexpr std::size_t byteSize = 32;

  static llvm::Expected<SystemPnrSearchDomainDigest>
  fromBytes(llvm::ArrayRef<std::uint8_t> bytes);

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const SystemPnrSearchDomainDigest &lhs,
                         const SystemPnrSearchDomainDigest &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const SystemPnrSearchDomainDigest &lhs,
                         const SystemPnrSearchDomainDigest &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit SystemPnrSearchDomainDigest(Storage bytes) : bytes_(bytes) {}

  friend llvm::Expected<SystemPnrSearchDomainDigest>
  computeSystemPnrSearchDomainDigest(
      llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
      llvm::ArrayRef<std::uint8_t> canonicalViewBytes);

  Storage bytes_;
};

llvm::ArrayRef<std::uint8_t> systemPnrSearchDomainSchemaDescriptorBytes();

llvm::Expected<SystemPnrSearchDomainDigest> computeSystemPnrSearchDomainDigest(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes);

llvm::Error validateSystemPnrSearchDomainDigest(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const SystemPnrSearchDomainDigest &digest);

using SystemSearchBindingKey = std::variant<::dataflow::RootThreadLaunchRef,
                                            ::dataflow::RootedGraphLaunchRef>;

struct SystemPresburgerBindingPartition final {
  SystemSearchBindingKey key;
  std::vector<::loom::mapping::SystemPresburgerCell> cells;
};

struct SystemBindingPartitionPlan final {
  std::vector<SystemPresburgerBindingPartition> bindings;
};

struct SystemSearchAtomDomains final {
  std::optional<std::vector<::loom::fabric::AccCoreOccurrenceRef>>
      compatibleAccCores;
  std::optional<std::vector<ArtifactRootReference>> compatibleSpatialMappings;
  std::optional<std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>
      compatibleServiceRegions;
  std::optional<std::vector<::loom::fabric::FabricTransportEndpointRef>>
      compatibleTransportEndpoints;
};

struct SystemSearchAtom final {
  ::loom::mapping::SystemPresburgerCell cell;
  SystemSearchAtomDomains domains;
};

struct SystemSearchBindingDomain final {
  SystemSearchBindingKey key;
  std::vector<SystemSearchAtom> atoms;
};

struct SystemTransferSourceTerminalKey final {
  ::loom::mapping::CanonicalServiceLegKey leg;
};

struct SystemTransferSinkTerminalKey final {
  ::loom::mapping::CanonicalServiceLegKey leg;
  ::dataflow::StructuralOrdinal sinkOrdinal = 0;
};

using SystemTransferTerminalKey = std::variant<SystemTransferSourceTerminalKey,
                                               SystemTransferSinkTerminalKey>;

struct SystemSearchTransferTerminalDomain final {
  SystemTransferTerminalKey key;
  std::vector<::loom::fabric::FabricTransportEndpointRef>
      compatibleTransportEndpoints;
};

struct SystemSearchServiceDomain final {
  ::loom::mapping::SystemServiceObligationKey key;
  std::optional<std::vector<::loom::fabric::FabricMemoryServiceRegionRef>>
      compatibleServiceRegions;
  std::vector<SystemSearchTransferTerminalDomain> transferTerminals;
};

enum class UnsupportedSystemPnrSearchDomainReason : std::uint32_t {
  DynamicWorkStableKeyProjectionUnavailable = 0,
  RootedGraphMayDomainProjectionUnavailable = 1,
  ServiceTransformProjectionUnavailable = 2,
};

class UnsupportedSystemPnrSearchDomain final
    : public llvm::ErrorInfo<UnsupportedSystemPnrSearchDomain> {
public:
  static char ID;

  UnsupportedSystemPnrSearchDomain(
      UnsupportedSystemPnrSearchDomainReason reason, std::string message)
      : reason_(reason), message_(std::move(message)) {}

  UnsupportedSystemPnrSearchDomainReason reason() const { return reason_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  UnsupportedSystemPnrSearchDomainReason reason_;
  std::string message_;
};

class SystemPnrSearchDomainView final {
public:
  const ArtifactRootReference &dataflowReference() const {
    return dataflowReference_;
  }
  const ArtifactRootReference &fabricReference() const {
    return fabricReference_;
  }
  const ArtifactRootReference &constraintReference() const {
    return constraintReference_;
  }
  llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches() const {
    return rootThreadLaunches_;
  }
  llvm::ArrayRef<SystemSearchBindingDomain> bindings() const {
    return bindings_;
  }
  llvm::ArrayRef<SystemSearchServiceDomain> serviceObligations() const {
    return serviceObligations_;
  }
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const {
    return canonicalViewBytes_;
  }
  const SystemPnrSearchDomainDigest &digest() const { return digest_; }

private:
  SystemPnrSearchDomainView(
      ArtifactRootReference dataflowReference,
      ArtifactRootReference fabricReference,
      ArtifactRootReference constraintReference,
      std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches,
      std::vector<SystemSearchBindingDomain> bindings,
      std::vector<SystemSearchServiceDomain> serviceObligations,
      std::vector<std::uint8_t> canonicalViewBytes,
      SystemPnrSearchDomainDigest digest)
      : dataflowReference_(std::move(dataflowReference)),
        fabricReference_(std::move(fabricReference)),
        constraintReference_(std::move(constraintReference)),
        rootThreadLaunches_(std::move(rootThreadLaunches)),
        bindings_(std::move(bindings)),
        serviceObligations_(std::move(serviceObligations)),
        canonicalViewBytes_(std::move(canonicalViewBytes)),
        digest_(std::move(digest)) {}

  ArtifactRootReference dataflowReference_;
  ArtifactRootReference fabricReference_;
  ArtifactRootReference constraintReference_;
  std::vector<::dataflow::RootThreadLaunchRef> rootThreadLaunches_;
  std::vector<SystemSearchBindingDomain> bindings_;
  std::vector<SystemSearchServiceDomain> serviceObligations_;
  std::vector<std::uint8_t> canonicalViewBytes_;
  SystemPnrSearchDomainDigest digest_;

  friend llvm::Expected<SystemPnrSearchDomainView> projectSystemPnrSearchDomain(
      const ::dataflow::CanonicalDataflowProgramView &,
      const ::loom::fabric::FabricSystemRootView &,
      const ::loom::mapping::FinalizedSystemMappingConstraintSet &,
      const SystemBindingPartitionPlan &, llvm::ArrayRef<ArtifactRootReference>,
      const ArtifactStore &);
  friend llvm::Expected<SystemPnrSearchDomainView> adoptSystemPnrSearchDomain(
      llvm::ArrayRef<std::uint8_t>, llvm::ArrayRef<std::uint8_t>,
      const SystemPnrSearchDomainDigest &, const ArtifactStore &);
  friend struct detail::SystemPnrSearchDomainViewBuilder;
};

llvm::Expected<SystemBindingPartitionPlan>
projectWholeDomainPresburgerPartitionPlan(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    llvm::ArrayRef<::dataflow::RootThreadLaunchRef> rootThreadLaunches);

llvm::Expected<SystemPnrSearchDomainView> projectSystemPnrSearchDomain(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const ::loom::fabric::FabricSystemRootView &fabric,
    const ::loom::mapping::FinalizedSystemMappingConstraintSet &constraints,
    const SystemBindingPartitionPlan &partitionPlan,
    llvm::ArrayRef<ArtifactRootReference> spatialMappings,
    const ArtifactStore &store);

llvm::Expected<SystemPnrSearchDomainView>
adoptSystemPnrSearchDomain(llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
                           llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
                           const SystemPnrSearchDomainDigest &digest,
                           const ArtifactStore &store);

} // namespace loom::pnr

#endif // LOOM_PNR_SYSTEM_SYSTEMPNRSEARCHDOMAIN_H

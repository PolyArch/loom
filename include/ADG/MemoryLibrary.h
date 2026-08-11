#ifndef LOOM_ADG_MEMORYLIBRARY_H
#define LOOM_ADG_MEMORYLIBRARY_H

#include "ADG/Builder.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom::adg {

/// Exact temporal parameters of a catalog local-memory recipe. Absence from
/// LocalMemoryParameters selects a Spatial operation engine.
struct TemporalMemoryParameters final {
  std::uint32_t tagWidth = 0;
  std::uint64_t residentContextCount = 0;
};

/// Transient typed bounds used to construct one exact memory access domain.
/// They are projected into ordinary operation endpoints and service
/// capabilities and do not become a separate persistent authority.
struct MemoryAccessDomainParameters final {
  std::uint32_t dataPayloadBits = 0;
  std::optional<std::uint32_t> indexedAddressPayloadBits;
  std::uint32_t maskPayloadBits = 0;
  std::optional<::fabric::UnsignedDomain> rootRelativeIndexWidths;
};

/// Independent physical widths of one local memory interface.
struct MemoryInterfaceParameters final {
  MemoryAccessDomainParameters accessDomain;
  std::uint32_t scalarAddressPayloadBits = 0;
  std::uint32_t serviceBeatWidthBits = 0;
};

/// Parameters shared by the catalog's local load/store recipes. The selected
/// recipe owns its exact typed scalar and vector access domains. The optional
/// manager endpoint admits the same operation requests through an external
/// service while retaining the local storage target.
struct LocalMemoryParameters final {
  std::uint64_t capacityBytes = 0;
  MemoryInterfaceParameters interface;
  std::optional<TemporalMemoryParameters> temporal;
  bool managerEndpoint = false;
};

/// Parameters for an Operation Engine whose only storage target is an
/// external service reached through its manager endpoint.
struct ManagerMemoryParameters final {
  MemoryInterfaceParameters interface;
  std::optional<TemporalMemoryParameters> temporal;
};

/// Exact address range, access domain, and independent beat width of one
/// catalog System memory-service recipe.
struct SystemMemoryParameters final {
  std::uint64_t addressBaseBytes = 0;
  std::uint64_t capacityBytes = 0;
  MemoryAccessDomainParameters accessDomain;
  std::uint32_t serviceBeatWidthBits = 0;
};

/// Matching owner and endpoint contracts for one typed System memory service.
/// Both records are ordinary Fabric contracts and introduce no helper-owned
/// persistent identity.
struct SystemMemorySpec final {
  ::fabric::MemoryServiceContractRecord contract;
  loom::fabric::CanonicalServiceCapabilitySet capabilities;
};

/// Builds one ordinary MemorySpec through the canonical Fabric memory owners.
/// The returned spec is consumed by SpatialCoreBuilder::addMemory; this helper
/// owns no persistent schema or identity of its own.
llvm::Expected<MemorySpec>
makeHybrid32LocalMemory(LocalMemoryParameters parameters);

/// Builds the general-purpose catalog memory domain. Its exact operation
/// endpoint and service widths come only from `parameters`; the recipe adds
/// the registered 64-bit element domain and optional indexed-address form.
llvm::Expected<MemorySpec>
makeGeneral64LocalMemory(LocalMemoryParameters parameters);

/// Builds the same Hybrid32 Operation Engine without a Local Memory Service.
/// Every admitted operation dispatches through the single manager endpoint.
llvm::Expected<MemorySpec>
makeHybrid32ManagerMemory(ManagerMemoryParameters parameters);

/// Builds the same General64 Operation Engine without a Local Memory Service.
/// Every admitted operation dispatches through the single manager endpoint.
llvm::Expected<MemorySpec>
makeGeneral64ManagerMemory(ManagerMemoryParameters parameters);

llvm::Expected<SystemMemorySpec>
makeHybrid32SystemMemory(SystemMemoryParameters parameters,
                         loom::fabric::ServiceRateContractRecord serviceRate);

llvm::Expected<SystemMemorySpec>
makeGeneral64SystemMemory(SystemMemoryParameters parameters,
                          loom::fabric::ServiceRateContractRecord serviceRate);

} // namespace loom::adg

#endif // LOOM_ADG_MEMORYLIBRARY_H

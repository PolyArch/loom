#ifndef LOOM_ADG_MEMORYLIBRARY_H
#define LOOM_ADG_MEMORYLIBRARY_H

#include "ADG/Builder.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace loom::adg {

/// Exact temporal parameters of the catalog local-memory recipe. Absence from
/// Hybrid32LocalMemoryParameters selects a Spatial operation engine.
struct TemporalMemoryParameters final {
  std::uint32_t tagWidth = 0;
  std::uint64_t residentContextCount = 0;
};

/// Parameters of the catalog's local 128-bit load/store memory. The optional
/// manager endpoint admits the same operation requests through an external
/// service while retaining the local storage target.
struct Hybrid32LocalMemoryParameters final {
  std::uint64_t capacityBytes = 0;
  std::optional<TemporalMemoryParameters> temporal;
  bool managerEndpoint = false;
};

/// Address range of the catalog's System memory-service recipe. The helper
/// admits the same scalar 32-bit and contiguous four-lane 32-bit accesses as
/// the local-memory recipe, but publishes them through the System service
/// plane.
struct Hybrid32SystemMemoryParameters final {
  std::uint64_t addressBaseBytes = 0;
  std::uint64_t capacityBytes = 0;
};

/// Matching owner and endpoint contracts for one typed System memory service.
/// Both records are ordinary Fabric contracts and introduce no helper-owned
/// persistent identity.
struct Hybrid32SystemMemorySpec final {
  ::fabric::MemoryServiceContractRecord contract;
  loom::fabric::CanonicalServiceCapabilitySet capabilities;
};

/// Builds one ordinary MemorySpec through the canonical Fabric memory owners.
/// The returned spec is consumed by SpatialCoreBuilder::addMemory; this helper
/// owns no persistent schema or identity of its own.
llvm::Expected<MemorySpec>
makeHybrid32LocalMemory(Hybrid32LocalMemoryParameters parameters);

llvm::Expected<Hybrid32SystemMemorySpec>
makeHybrid32SystemMemory(Hybrid32SystemMemoryParameters parameters,
                         loom::fabric::ServiceRateContractRecord serviceRate);

} // namespace loom::adg

#endif // LOOM_ADG_MEMORYLIBRARY_H

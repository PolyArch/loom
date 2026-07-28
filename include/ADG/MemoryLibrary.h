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

/// Parameters shared by the catalog's local 128-bit load/store recipes. The
/// selected recipe owns its exact typed scalar and vector access domains. The
/// optional manager endpoint admits the same operation requests through an
/// external service while retaining the local storage target.
struct LocalMemoryParameters final {
  std::uint64_t capacityBytes = 0;
  std::optional<TemporalMemoryParameters> temporal;
  bool managerEndpoint = false;
};

/// Address range shared by the catalog's System memory-service recipes.
struct SystemMemoryParameters final {
  std::uint64_t addressBaseBytes = 0;
  std::uint64_t capacityBytes = 0;
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

/// Builds the general-purpose builtin memory domain. It retains the exact
/// contiguous four-lane 32-bit vector geometry of Hybrid32 and additionally
/// admits common scalar 64-bit accesses through the same 128-bit datapath.
llvm::Expected<MemorySpec>
makeGeneral64LocalMemory(LocalMemoryParameters parameters);

llvm::Expected<SystemMemorySpec>
makeHybrid32SystemMemory(SystemMemoryParameters parameters,
                         loom::fabric::ServiceRateContractRecord serviceRate);

llvm::Expected<SystemMemorySpec>
makeGeneral64SystemMemory(SystemMemoryParameters parameters,
                         loom::fabric::ServiceRateContractRecord serviceRate);

} // namespace loom::adg

#endif // LOOM_ADG_MEMORYLIBRARY_H

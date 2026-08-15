#include "Fabric/Identity/FabricMemoryInternalConnection.h"

#include "Fabric/Identity/FabricRefBytes.h"

#include <map>
#include <utility>
#include <vector>

namespace loom::fabric {

FabricMemoryInternalConnectionClosure
deriveFabricMemoryInternalConnectionClosure(
    llvm::ArrayRef<FabricMemoryInternalConnectionUse> uses) {
  struct Counts final {
    std::uint64_t producers = 0;
    std::uint64_t consumers = 0;
  };
  using Key = std::pair<std::vector<std::uint8_t>, FabricOrdinal>;
  std::map<Key, Counts> counts;
  for (const FabricMemoryInternalConnectionUse &use : uses) {
    Counts &selected =
        counts[{canonicalFabricBytes(use.occurrence), use.connection}];
    if (use.kind == FabricMemoryInternalConnectionUseKind::Producer)
      ++selected.producers;
    else
      ++selected.consumers;
  }
  for (const auto &[key, selected] : counts) {
    (void)key;
    if (selected.producers == 0)
      return FabricMemoryInternalConnectionClosure::Open;
    if (selected.producers > 1)
      return FabricMemoryInternalConnectionClosure::MultipleProducers;
    if (selected.consumers == 0)
      return FabricMemoryInternalConnectionClosure::Open;
  }
  return FabricMemoryInternalConnectionClosure::Closed;
}

} // namespace loom::fabric

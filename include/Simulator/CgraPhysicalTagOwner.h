#ifndef LOOM_SIMULATOR_CGRAPHYSICALTAGOWNER_H
#define LOOM_SIMULATOR_CGRAPHYSICALTAGOWNER_H

#include "Dataflow/IR/DataflowCanonicalEntity.h"

#include <cstdint>
#include <variant>

namespace loom::sim {

/// Exact Mapping owner of one Physical Tag carried by the frozen execution
/// plan. Route segments and PE-local register-FIFO transfers are distinct
/// Mapping decisions; the plan-local dense tag ordinal is never persisted as
/// their identity.
struct CgraRoutePhysicalTagOwner final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  std::uint64_t segmentOrdinal = 0;
};

struct CgraRegisterFifoPhysicalTagOwner final {
  ::dataflow::CanonicalGraphProducerEndpointRef producer;
  ::dataflow::CanonicalGraphConsumerEndpointRef consumer;
};

using CgraPhysicalTagMappingOwner =
    std::variant<CgraRoutePhysicalTagOwner,
                 CgraRegisterFifoPhysicalTagOwner>;

} // namespace loom::sim

#endif // LOOM_SIMULATOR_CGRAPHYSICALTAGOWNER_H

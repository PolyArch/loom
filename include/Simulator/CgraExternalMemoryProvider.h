#ifndef LOOM_SIMULATOR_CGRAEXTERNALMEMORYPROVIDER_H
#define LOOM_SIMULATOR_CGRAEXTERNALMEMORYPROVIDER_H

#include "Fabric/Identity/FabricRefs.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <vector>

namespace loom::sim {

enum class CgraExternalMemoryOperation : std::uint32_t {
  Read = 0,
  Write = 1,
};

/// One active element of a logical memory actor firing. Offsets are relative
/// to the canonical runtime memory object. A write carries exactly byteCount
/// bytes; a read carries none.
struct CgraExternalMemoryElement final {
  std::uint64_t byteOffset = 0;
  std::uint64_t byteCount = 0;
  std::vector<std::uint8_t> writeData;
};

/// Transient projection of one accepted memory actor firing whose selected
/// Mapping target crosses a manager endpoint. Canonical Dataflow owns the
/// operation shape, Mapping owns the endpoint, and the provider owns external
/// timing and dynamic state.
struct CgraExternalMemoryRequest final {
  ::loom::fabric::ManagerEndpointRef endpoint;
  std::uint64_t objectOrdinal = 0;
  CgraExternalMemoryOperation operation = CgraExternalMemoryOperation::Read;
  std::vector<CgraExternalMemoryElement> elements;
  SpatialEventCoordinate readyCoordinate;
};

/// Read elements retain request order. A write response has no read data.
struct CgraExternalMemoryResponse final {
  std::vector<std::vector<std::uint8_t>> readData;
};

/// Execution-scoped provider for manager-dispatched CGRA memory requests.
/// The call returns only after the selected external service has completed the
/// one logical request. Provider-internal beats remain invisible to actor
/// firing and retirement identity.
class CgraExternalMemoryProvider {
public:
  virtual ~CgraExternalMemoryProvider() = default;

  virtual llvm::Expected<CgraExternalMemoryResponse>
  transact(const CgraExternalMemoryRequest &request) = 0;
};

} // namespace loom::sim

#endif // LOOM_SIMULATOR_CGRAEXTERNALMEMORYPROVIDER_H

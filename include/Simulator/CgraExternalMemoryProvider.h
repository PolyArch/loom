#ifndef LOOM_SIMULATOR_CGRAEXTERNALMEMORYPROVIDER_H
#define LOOM_SIMULATOR_CGRAEXTERNALMEMORYPROVIDER_H

#include "Fabric/Identity/FabricRefs.h"
#include "Simulator/SimulationExecution.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

namespace loom::sim {

namespace detail {
class CgraMemoryRuntime;
}

/// An execution-owned logical request identity. The retained domain prevents
/// a completion from another execution, including a destroyed execution whose
/// storage address has since been reused, from naming this request.
class CgraExternalMemoryRequestId final {
public:
  bool operator==(const CgraExternalMemoryRequestId &other) const {
    return domain_ == other.domain_ &&
           semanticActorOrdinal_ == other.semanticActorOrdinal_ &&
           actorOccurrenceOrdinal_ == other.actorOccurrenceOrdinal_;
  }

private:
  struct Domain final {};

  CgraExternalMemoryRequestId(std::shared_ptr<const Domain> domain,
                              std::uint64_t semanticActorOrdinal,
                              std::uint64_t actorOccurrenceOrdinal)
      : domain_(std::move(domain)), semanticActorOrdinal_(semanticActorOrdinal),
        actorOccurrenceOrdinal_(actorOccurrenceOrdinal) {}

  std::shared_ptr<const Domain> domain_;
  std::uint64_t semanticActorOrdinal_;
  std::uint64_t actorOccurrenceOrdinal_;

  friend class detail::CgraMemoryRuntime;
};

enum class CgraExternalMemoryOperation : std::uint32_t {
  Read = 0,
  Write = 1,
};

/// One contiguous external transfer of a logical memory actor firing. Offsets
/// are relative to the canonical runtime memory object. A write carries
/// exactly byteCount bytes; a read carries none. A contiguous access
/// contributes one element for each run of adjacent active lanes, so a fully
/// active vector firing is one element; element and indexed geometry
/// contribute one element for each active lane.
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
  CgraExternalMemoryRequestId id;
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

/// The provider retained the request and will complete it through the issuing
/// execution session. One logical response still covers every request element.
struct CgraExternalMemoryPending final {};

using CgraExternalMemorySubmission =
    std::variant<CgraExternalMemoryResponse, CgraExternalMemoryPending>;

/// Execution-scoped provider for manager-dispatched CGRA memory requests.
/// A completed local service returns its response. An external service may
/// retain the request and return Pending; the execution keeps advancing every
/// actor whose linearization the consistency domain does not order behind that
/// request. Provider-internal beats remain invisible to actor firing and
/// retirement identity.
class CgraExternalMemoryProvider {
public:
  virtual ~CgraExternalMemoryProvider() = default;

  virtual llvm::Expected<CgraExternalMemorySubmission>
  submit(const CgraExternalMemoryRequest &request) = 0;

  /// Concurrent requests the selected external service guarantees. The
  /// execution submits at most this many requests before consuming the
  /// response of the earliest one. It is the exact service contract's
  /// outstanding-operation guarantee, never a simulator-chosen depth.
  virtual std::uint64_t outstandingCapacity() const = 0;
};

} // namespace loom::sim

#endif // LOOM_SIMULATOR_CGRAEXTERNALMEMORYPROVIDER_H

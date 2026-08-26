#ifndef LOOM_PNR_SPATIALGLOBALROUTINGCLOSURE_H
#define LOOM_PNR_SPATIALGLOBALROUTINGCLOSURE_H

#include "PnR/SpatialActionExecutor.h"

#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>

namespace loom::pnr {

enum class SpatialGlobalRoutingClosureFailureKind : std::uint8_t {
  UnroutedObligation,
  RouteCapacityOveruse,
  TagResidentCapacityOveruse,
  TagUnassigned,
  TagConflict,
};

class SpatialGlobalRoutingClosureFailure final
    : public llvm::ErrorInfo<SpatialGlobalRoutingClosureFailure> {
public:
  static char ID;

  SpatialGlobalRoutingClosureFailure(
      SpatialGlobalRoutingClosureFailureKind kind, std::string message)
      : kind_(kind), message_(std::move(message)) {}

  SpatialGlobalRoutingClosureFailureKind kind() const { return kind_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  SpatialGlobalRoutingClosureFailureKind kind_;
  std::string message_;
};

/// Executes one final Global TransportRoutingAction through the ordinary
/// Spatial Action transaction. This owner closes route, resident capacity,
/// and route-derived tag state; complete final verification additionally
/// requires every other Mapping violation owner.
class SpatialGlobalRoutingClosureScratch final {
public:
  llvm::Error run(SpatialCandidateState &candidate,
                  SpatialPnrWorkLedgerView workLedger = {});

  std::uint64_t endpointExpansionCount() const {
    return actionExecutor_.endpointExpansionCount();
  }
  std::uint64_t negotiationIterationCount() const {
    return actionExecutor_.negotiationIterationCount();
  }
  HandshakeProjectionStatistics handshakeProjectionStatistics() const {
    return actionExecutor_.handshakeProjectionStatistics();
  }

  std::size_t retainedStorageBytes() const;

private:
  SpatialActionExecutorScratch actionExecutor_;
};

} // namespace loom::pnr

#endif // LOOM_PNR_SPATIALGLOBALROUTINGCLOSURE_H

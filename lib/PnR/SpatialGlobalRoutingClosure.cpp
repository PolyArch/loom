#include "PnR/SpatialGlobalRoutingClosure.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <system_error>
#include <utility>

using namespace loom::pnr;

char SpatialGlobalRoutingClosureFailure::ID;

void SpatialGlobalRoutingClosureFailure::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code SpatialGlobalRoutingClosureFailure::convertToErrorCode() const {
  return std::make_error_code(std::errc::resource_unavailable_try_again);
}

namespace {

llvm::Error closureError(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial global routing closure: %s", message.str().c_str());
}

llvm::Error verifyRoutingClosure(const SpatialCandidateState &candidate) {
  if (candidate.unroutedObligationCount() != 0)
    return llvm::make_error<SpatialGlobalRoutingClosureFailure>(
        SpatialGlobalRoutingClosureFailureKind::UnroutedObligation,
        "Global Action left an unrouted sink obligation");
  if (candidate.routeCapacityOveruse() != 0)
    return llvm::make_error<SpatialGlobalRoutingClosureFailure>(
        SpatialGlobalRoutingClosureFailureKind::RouteCapacityOveruse,
        "Global Action left route-resource overuse");
  if (candidate.tagResidentCapacityOveruse() != 0)
    return llvm::make_error<SpatialGlobalRoutingClosureFailure>(
        SpatialGlobalRoutingClosureFailureKind::TagResidentCapacityOveruse,
        "Global Action left Physical Tag table overuse");
  if (candidate.tagUnassignedCount() != 0)
    return llvm::make_error<SpatialGlobalRoutingClosureFailure>(
        SpatialGlobalRoutingClosureFailureKind::TagUnassigned,
        "Global Action left an unassigned Physical Tag");
  if (candidate.tagConflictCount() != 0)
    return llvm::make_error<SpatialGlobalRoutingClosureFailure>(
        SpatialGlobalRoutingClosureFailureKind::TagConflict,
        "Global Action left a conflicting Physical Tag");
  return llvm::Error::success();
}

} // namespace

llvm::Error
SpatialGlobalRoutingClosureScratch::run(SpatialCandidateState &candidate,
                                        SpatialPnrWorkLedgerView workLedger,
                                        ExecutionControlView executionControl) {
  if (llvm::Error error =
          actionExecutor_.prepare(candidate, workLedger, executionControl))
    return error;

  const SpatialMappingAction action =
      SpatialTransportRoutingAction{SpatialGlobalRoutingAction{}};
  auto probe = actionExecutor_.probe(
      candidate, action, SpatialActionExecutionContext::FinalClosure);
  if (!probe)
    return probe.takeError();
  if (llvm::Error error = verifyRoutingClosure(candidate)) {
    if (llvm::Error discardError = probe->discard())
      return llvm::joinErrors(std::move(error), std::move(discardError));
    return error;
  }
  if (llvm::Error error = probe->commit())
    return error;

  if (llvm::Error error = candidate.verifyCachedState())
    return error;
  if (llvm::Error error = verifyRoutingClosure(candidate))
    return error;
  auto rebuilt = candidate.problem().objectiveProgram().evaluate(candidate);
  if (!rebuilt)
    return rebuilt.takeError();
  if (!llvm::equal(rebuilt->codes(),
                   actionExecutor_.currentObjective().codes()))
    return closureError(
        "incremental objective disagrees with full owner recomputation");
  return llvm::Error::success();
}

std::size_t SpatialGlobalRoutingClosureScratch::retainedStorageBytes() const {
  return actionExecutor_.retainedStorageBytes();
}

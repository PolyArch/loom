#include "PnR/SpatialGlobalRoutingClosure.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <system_error>
#include <utility>

using namespace loom::pnr;

namespace {

llvm::Error closureError(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial global routing closure: %s", message.str().c_str());
}

llvm::Error verifyRoutingClosure(const SpatialCandidateState &candidate) {
  if (candidate.unroutedObligationCount() != 0)
    return closureError("Global Action left an unrouted sink obligation");
  if (candidate.routeCapacityOveruse() != 0)
    return closureError("Global Action left route-resource overuse");
  if (candidate.tagUnassignedCount() != 0)
    return closureError("Global Action left an unassigned Physical Tag");
  if (candidate.tagConflictCount() != 0)
    return closureError("Global Action left a conflicting Physical Tag");
  return llvm::Error::success();
}

} // namespace

llvm::Error
SpatialGlobalRoutingClosureScratch::run(SpatialCandidateState &candidate) {
  if (llvm::Error error = candidate.verify())
    return error;
  if (llvm::Error error = actionExecutor_.prepare(candidate))
    return error;

  const SpatialMappingAction action =
      SpatialTransportRoutingAction{SpatialGlobalRoutingAction{}};
  auto probe = actionExecutor_.probe(candidate, action);
  if (!probe)
    return probe.takeError();
  if (llvm::Error error = verifyRoutingClosure(candidate)) {
    if (llvm::Error discardError = probe->discard())
      return llvm::joinErrors(std::move(error), std::move(discardError));
    return error;
  }
  if (llvm::Error error = probe->commit())
    return error;

  if (llvm::Error error = candidate.verify())
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

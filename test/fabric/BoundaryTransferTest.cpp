#include "Fabric/IR/BoundaryTransfer.h"
#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>

using namespace fabric;

namespace {

[[noreturn]] void fail(const std::string &message) {
  llvm::errs() << "boundary transfer: " << message << "\n";
  std::exit(1);
}

void require(bool condition, const std::string &message) {
  if (!condition)
    fail(message);
}

void requireLegs(llvm::ArrayRef<bool> observed,
                 std::initializer_list<bool> expected,
                 const std::string &label) {
  require(observed.size() == expected.size(), label + ": leg count differs");
  std::size_t leg = 0;
  for (bool signal : expected) {
    require(observed[leg] == signal,
            label + " leg " + std::to_string(leg) + " differs");
    ++leg;
  }
}

// No leg of one side transfers: a leg moves a token only when its valid and its
// ready both hold. Cross-leg equations may assert one of the two against a
// peer, so the absence of a handshake is the claim, not the absence of a
// signal.
void requireNoLegHandshakes(llvm::ArrayRef<bool> valid,
                            llvm::ArrayRef<bool> ready,
                            const std::string &label) {
  require(valid.size() == ready.size(), label + ": leg count differs");
  for (std::size_t leg = 0; leg < valid.size(); ++leg)
    require(!(valid[leg] && ready[leg]),
            label + " leg " + std::to_string(leg) + " handshaked");
}

// A two-input s2t join with both inputs valid, the output ready, and the
// transfer selected performs the one atomic transfer. Every driven and every
// derived signal is high, so both inputs are consumed and the one output is
// published in the same cycle. Without this case every expectation below would
// be an absent transfer, which a constantly inert evaluator would satisfy.
void enabledJoinFires() {
  const bool inputValid[] = {true, true};
  const bool outputReady[] = {true};
  const BoundaryTransfer transfer =
      evaluateBoundaryTransfer({inputValid, outputReady, /*match=*/true});

  require(transfer.fire, "enabled join did not fire");
  requireLegs(transfer.inputReady, {true, true}, "enabled join input ready");
  requireLegs(transfer.outputValid, {true}, "enabled join output valid");
}

// A two-input s2t join whose tag leg is not valid consumes neither input. This
// is the counterexample spelled by docs/spec-fabric-boundary.md: the data leg
// is valid and the output is ready, yet data.ready stays low. The tag leg's
// ready does rise against the valid data peer, and consumes nothing because
// that leg carries no valid token.
void partialValidJoinDoesNotFire() {
  const bool inputValid[] = {true, false};
  const bool outputReady[] = {true};
  const BoundaryTransfer transfer =
      evaluateBoundaryTransfer({inputValid, outputReady, /*match=*/true});

  require(!transfer.fire, "partial-valid join fired");
  requireLegs(transfer.inputReady, {false, true},
              "partial-valid join input ready");
  requireLegs(transfer.outputValid, {false}, "partial-valid join output valid");
  requireNoLegHandshakes(inputValid, transfer.inputReady,
                         "partial-valid join input");
}

// A split t2s fork whose tag output is not ready retires nothing. The data
// output valid drops because its peer is stalled, while the tag output valid
// stays asserted against the ready data peer: the cross-ready equations assert
// valid independently of the joint transfer, which is the event that must not
// occur.
void partialReadyForkDoesNotFire() {
  const bool inputValid[] = {true};
  const bool outputReady[] = {true, false};
  const BoundaryTransfer transfer =
      evaluateBoundaryTransfer({inputValid, outputReady, /*match=*/true});

  require(!transfer.fire, "partial-ready fork fired");
  requireLegs(transfer.inputReady, {false}, "partial-ready fork input ready");
  requireLegs(transfer.outputValid, {false, true},
              "partial-ready fork output valid");
  requireNoLegHandshakes(inputValid, transfer.inputReady,
                         "partial-ready fork input");
  requireNoLegHandshakes(transfer.outputValid, outputReady,
                         "partial-ready fork output");
}

// An unselected transfer consumes nothing and publishes nothing even when every
// leg would otherwise handshake. This fixes the inert cycle-level signals that
// a Disabled projection and a t2t lookup miss share; it does not make a miss on
// a reachable tag legal, which stays a SpatialMapping finalization failure.
void unmatchedTransferIsInert() {
  const bool inputValid[] = {true};
  const bool outputReady[] = {true};
  const BoundaryTransfer transfer =
      evaluateBoundaryTransfer({inputValid, outputReady, /*match=*/false});

  require(!transfer.fire, "unmatched transfer fired");
  requireLegs(transfer.inputReady, {false}, "unmatched transfer input ready");
  requireLegs(transfer.outputValid, {false}, "unmatched transfer output valid");
}

// The boundary owns one stateless atomic use: no state to occupy, no capacity
// to claim, one requester, and therefore no observable requester order.
void contractIsStatelessAndUnarbitrated() {
  llvm::Expected<ResourceContract> created =
      ResourceContract::create(declareBoundaryTransferContract());
  if (!created)
    fail(llvm::toString(created.takeError()));
  const ResourceContract contract = std::move(*created);

  require(contract.stateCount() == 0, "boundary contract declares a state");
  require(contract.usePatternCount() == 1,
          "boundary contract does not declare exactly one use pattern");
  require(contract.requesterCount() == 1,
          "boundary contract does not declare exactly one requester");
  require(!contract.grantPolicy(), "boundary contract declares a grant policy");

  const UsePattern transfer = contract.usePattern(boundaryTransferPattern);
  require(transfer.claims.empty(), "boundary transfer claims capacity");
  require(!transfer.commit, "boundary transfer declares a state transition");
  require(transfer.acquire == transfer.release,
          "boundary transfer does not acquire and release in one event");
  require(contract.eventOrder(transfer.timingAndProgress) ==
              llvm::ArrayRef<std::uint32_t>({0}),
          "boundary transfer timing does not preserve one atomic event");
}

} // namespace

int main() {
  enabledJoinFires();
  partialValidJoinDoesNotFire();
  partialReadyForkDoesNotFire();
  unmatchedTransferIsInert();
  contractIsStatelessAndUnarbitrated();
  return 0;
}

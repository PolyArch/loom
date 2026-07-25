#include "Fabric/IR/BoundaryTransfer.h"

#include "llvm/ADT/STLExtras.h"

#include <cassert>
#include <cstddef>

using namespace fabric;

namespace {

// The boundary's single requester: the occurrence itself. One requester can
// never contend with another, so nothing outside this declaration orders it and
// the key stays private.
constexpr RequesterKey boundaryTransferRequester{0};

bool everyLegHolds(llvm::ArrayRef<bool> signals) {
  return llvm::all_of(signals, [](bool signal) { return signal; });
}

// Every leg of one side except `leg` itself. A single-leg side has no peer, so
// the conjunction is vacuously true.
bool everyPeerHolds(llvm::ArrayRef<bool> signals, std::size_t leg) {
  for (auto [index, signal] : llvm::enumerate(signals))
    if (index != leg && !signal)
      return false;
  return true;
}

} // namespace

BoundaryTransfer
fabric::evaluateBoundaryTransfer(const BoundaryHandshake &handshake) {
  const std::size_t inputLegs = handshake.inputValid.size();
  const std::size_t outputLegs = handshake.outputReady.size();
  // The legal shapes are enumerated rather than bounded per side: the
  // configured-tag s2t, drop-tag t2s, and t2t forms are (1, 1), the two-operand
  // s2t join is (2, 1), and the split t2s fork is (1, 2). A per-side bound
  // would also admit (2, 2), which no current op shape declares.
  assert(((inputLegs == 1 && outputLegs == 1) ||
          (inputLegs == 2 && outputLegs == 1) ||
          (inputLegs == 1 && outputLegs == 2)) &&
         "fabric.boundary joins two inputs, forks two outputs, or neither");

  const bool everyInputValid = everyLegHolds(handshake.inputValid);
  const bool everyOutputReady = everyLegHolds(handshake.outputReady);

  BoundaryTransfer transfer;
  transfer.fire = everyInputValid && everyOutputReady && handshake.match;

  transfer.inputReady.reserve(inputLegs);
  for (std::size_t leg = 0; leg < inputLegs; ++leg)
    transfer.inputReady.push_back(everyOutputReady && handshake.match &&
                                  everyPeerHolds(handshake.inputValid, leg));

  transfer.outputValid.reserve(outputLegs);
  for (std::size_t leg = 0; leg < outputLegs; ++leg)
    transfer.outputValid.push_back(everyInputValid && handshake.match &&
                                   everyPeerHolds(handshake.outputReady, leg));

  return transfer;
}

ResourceContractDeclaration fabric::declareBoundaryTransferContract() {
  ResourceContractDeclaration declaration;
  declaration.requesters = {boundaryTransferRequester};
  // One eligibility condition (the rendezvous itself), one event carrying both
  // acquire and release, and one timing contract (same local-cycle delta).
  declaration.eligibilityCount = 1;
  declaration.eventCount = 1;
  declaration.timingContracts = {
      TimingContractDeclaration{TimingContractKey(0), {0}}};
  declaration.usePatterns = {UsePatternDeclaration{boundaryTransferPattern,
                                                   boundaryTransferRequester,
                                                   EligibilityKey(0),
                                                   EventKey(0),
                                                   EventKey(0),
                                                   std::nullopt,
                                                   TimingContractKey(0),
                                                   {},
                                                   {}}};
  // `states` and `grantPolicy` stay empty: the absence is the declaration.
  return declaration;
}

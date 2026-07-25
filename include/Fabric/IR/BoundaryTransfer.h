#ifndef FABRIC_IR_BOUNDARYTRANSFER_H
#define FABRIC_IR_BOUNDARYTRANSFER_H

#include "Fabric/IR/ResourceContract.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace fabric {

/// The signals driven onto one `fabric.boundary` occurrence in one local cycle.
/// The leg counts are the occurrence's immutable op shape, so the spans carry
/// them and no separate shape descriptor exists: `inputValid` holds the
/// producer valid of every input leg and `outputReady` the consumer ready of
/// every output leg, both in operand and result order.
///
/// `match` is the selection predicate the transfer is conditioned on: a `t2t`
/// input tag that resolved to exactly one selected lookup entry, and
/// unconditionally true for the forms whose Active projection carries no
/// lookup.
///
/// `match = false` says only that this occurrence transfers nothing this cycle.
/// A `Disabled` projection and a `t2t` lookup miss share that inert signal
/// projection but are not the same fact: `Disabled` is a legally unselected
/// resource, while a miss on a reachable tag is an invalid finalized Mapping
/// that SpatialMapping finalization must reject. Which tags are reachable is
/// decided there and never here, so an inert result is not evidence that a miss
/// is admissible packet-drop behavior.
struct BoundaryHandshake {
  llvm::ArrayRef<bool> inputValid;
  llvm::ArrayRef<bool> outputReady;
  bool match;
};

/// The signals one boundary occurrence derives in the same local cycle.
/// `fire` is the single atomic transfer event: on `fire` every input leg is
/// consumed exactly once and every output leg is published exactly once.
struct BoundaryTransfer {
  bool fire;
  llvm::SmallVector<bool, 2> inputReady;
  llvm::SmallVector<bool, 2> outputValid;
};

/// Evaluates the stateless atomic rendezvous of one boundary occurrence:
///
///   fire            = all(inputValid) && all(outputReady) && match
///   inputReady[i]   = all(outputReady) && all(inputValid except i) && match
///   outputValid[j]  = all(inputValid) && all(outputReady except j) && match
///
/// One leg family covers every current op form. A one-leg side has no peer, so
/// its peer conjunction is vacuously true and the equations degenerate to the
/// ordinary `out.valid = in.valid` and `in.ready = out.ready` of the
/// configured-tag `s2t`, drop-tag `t2s`, and matched `t2t` forms. Excluding a
/// leg's own signal from its derived signal is what keeps a stalled peer from
/// making a leg depend on itself.
///
/// The result is a pure function of one cycle's driven signals: the boundary
/// holds no state between calls, buffers, drains, arbitrates, and adds no
/// registered latency.
///
/// `fabric.boundary` splits or joins on exactly one side, so the only legal
/// shapes are the two-operand `s2t` join, the split `t2s` fork, and the one-to-
/// one forms. Two legs on both sides is not a current boundary shape and is a
/// caller error, as is any other leg count.
BoundaryTransfer evaluateBoundaryTransfer(const BoundaryHandshake &handshake);

/// The boundary's one atomic use. It is the only use pattern the contract
/// declares, so it is also the only key a consumer needs to read it.
inline constexpr UsePatternKey boundaryTransferPattern{0};

/// Declares the canonical `BoundaryTransfer` contract: one requester owning one
/// atomic use pattern that claims no capacity, over no declared ResourceState.
/// Acquire and release name the same event, which is the commit, publish, and
/// retire of one local-cycle delta. With no capacity dimension there is no
/// reachable contention and therefore no grant policy.
ResourceContractDeclaration declareBoundaryTransferContract();

} // namespace fabric

#endif // FABRIC_IR_BOUNDARYTRANSFER_H

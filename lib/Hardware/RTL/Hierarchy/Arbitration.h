#ifndef LOOM_LIB_HARDWARE_RTL_HIERARCHY_ARBITRATION_H
#define LOOM_LIB_HARDWARE_RTL_HIERARCHY_ARBITRATION_H

#include "Support.h"

#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace loom::hardware::rtl::hierarchy {

/// Packs one-bit values into one word with ordinal 0 at bit 0.
mlir::Value packBits(mlir::OpBuilder &builder, mlir::Location location,
                     llvm::ArrayRef<mlir::Value> lowToHigh);

/// The canonical round-robin grant of the Fabric resource contracts: the
/// first requester at or after the cursor in cyclic requester order, as a
/// one-hot-or-zero word over the packed request domain. `packed` has exactly
/// `requestCount` bits and `cursor` is an in-range `indexWidth(requestCount)`
/// position in that same order.
mlir::Value roundRobinPackedSelection(mlir::OpBuilder &builder,
                                      mlir::Location location,
                                      mlir::Value packed, unsigned requestCount,
                                      mlir::Value cursor);

mlir::Value roundRobinPackedSelection(mlir::OpBuilder &builder,
                                      mlir::Location location,
                                      llvm::ArrayRef<mlir::Value> requests,
                                      mlir::Value cursor);

std::vector<mlir::Value>
roundRobinSelection(mlir::OpBuilder &builder, mlir::Location location,
                    llvm::ArrayRef<mlir::Value> requests, mlir::Value cursor);

/// The cursor after a committed grant: the successor of the committed
/// requester, or the current cursor while nothing committed. The committed
/// word is structurally one-hot-or-zero.
mlir::Value nextCursorFromPacked(mlir::OpBuilder &builder,
                                 mlir::Location location, mlir::Value current,
                                 mlir::Value packed,
                                 std::size_t requesterCount);

mlir::Value nextCursor(mlir::OpBuilder &builder, mlir::Location location,
                       mlir::Value current, llvm::ArrayRef<mlir::Value> fired);

/// The registered grant state of one arbitration component, as
/// `Registered grant` in `docs/spec-fabric-switch.md` defines it and the
/// shared resource contract applies it to every resource with a GrantPolicy.
/// A component of more than one requester owns exactly one arbitration
/// register, the grant pointer naming the one requester that may proceed;
/// `pointed` is therefore a function of registered state alone and names
/// exactly one position of the component-local policy order every cycle. The
/// pointer's next value observes this cycle's live eligibility vector, but
/// feeds only the register's data input, so no readiness the grant gates is a
/// function of a Valid of the same cycle.
struct RegisteredGrant final {
  std::optional<circt::Backedge> next;
  std::vector<mlir::Value> pointed;
  mlir::Value oneHot;
  mlir::Value pointer;
  bool roundRobin = true;
};

/// Builds that state over `requesterCount` positions of the component-local
/// policy order, with the pointer reset at `resetPosition`. A one-requester
/// component owns no register at all: its single requester is always the
/// pointed one.
RegisteredGrant makeRegisteredGrant(mlir::OpBuilder &builder,
                                    mlir::Location location,
                                    circt::BackedgeBuilder &backedges,
                                    std::size_t requesterCount,
                                    bool roundRobin, unsigned resetPosition,
                                    mlir::Value clock, mlir::Value reset,
                                    llvm::StringRef name,
                                    const ClockResetPlan &clockReset);

/// Closes the grant pointer's next-state function over two live words with
/// one bit per component requester in the component-local policy order:
/// `eligible`, the requesters that could be served this cycle if the pointer
/// named them, and `requested`, the requesters whose token has arrived.
/// RoundRobin moves the pointer to the first eligible requester strictly after
/// it; when no requester is eligible it moves to the first requesting one, so
/// a requester waiting on its service is already pointed at when that service
/// resumes; and it holds when none requests. A pointed requester whose service
/// refuses therefore never keeps its turn from a requester that could proceed.
/// FixedPriority names the highest-priority eligible requester, else the
/// highest-priority requesting one, else holds.
void advanceRegisteredGrant(mlir::OpBuilder &builder, mlir::Location location,
                            RegisteredGrant &grant, mlir::Value eligible,
                            mlir::Value requested);

/// A round-robin grant over a requester domain together with its registered
/// cursor. A domain of at most one requester carries no cursor state.
struct StatefulSelection final {
  std::optional<circt::Backedge> next;
  mlir::Value cursor;
  std::vector<mlir::Value> selected;
};

StatefulSelection makeStatefulSelection(mlir::OpBuilder &builder,
                                        mlir::Location location,
                                        circt::BackedgeBuilder &backedges,
                                        llvm::ArrayRef<mlir::Value> requests,
                                        mlir::Value clock, mlir::Value reset,
                                        llvm::StringRef name,
                                        const ClockResetPlan &clockReset);

void advanceStatefulSelection(mlir::OpBuilder &builder, mlir::Location location,
                              StatefulSelection &selection,
                              llvm::ArrayRef<mlir::Value> fired);

/// One Fabric presentation priority over requester/evaluation positions. The
/// returned physical requester focus is projected into nested FU selectors.
mlir::Value makeResultPresentationPriority(
    mlir::OpBuilder &builder, mlir::Location location,
    circt::BackedgeBuilder &backedges,
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> eligible,
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> evaluated, mlir::Value clock,
    mlir::Value reset, llvm::StringRef name, const ClockResetPlan &clockReset);

/// Stateless complete-tuple selection in the Fabric physical requester order.
/// Overlapping lanes refuse their complete requester.
std::vector<mlir::Value> selectResultPresentation(
    mlir::OpBuilder &builder, mlir::Location location,
    llvm::ArrayRef<llvm::SmallVector<mlir::Value>> requestedLaneDestinations,
    mlir::Value priority);

} // namespace loom::hardware::rtl::hierarchy

#endif // LOOM_LIB_HARDWARE_RTL_HIERARCHY_ARBITRATION_H

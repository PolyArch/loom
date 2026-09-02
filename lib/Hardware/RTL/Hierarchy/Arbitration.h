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
/// one-hot-or-zero word over the packed request domain.
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

} // namespace loom::hardware::rtl::hierarchy

#endif // LOOM_LIB_HARDWARE_RTL_HIERARCHY_ARBITRATION_H

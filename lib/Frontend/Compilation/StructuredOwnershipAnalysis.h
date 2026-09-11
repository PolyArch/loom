#ifndef LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDOWNERSHIPANALYSIS_H
#define LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDOWNERSHIPANALYSIS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>
#include <vector>

namespace loom::frontend::detail {

std::optional<std::string>
explainCallableOwnershipRejection(mlir::LLVM::LLVMFuncOp function);

std::optional<std::string>
explainGraphStructuralOwnershipRejection(mlir::ModuleOp module,
                                         mlir::Operation *selection);

bool containsGeneralCall(mlir::Operation *selection);

struct CallableOwnershipBoundary final {
  llvm::SmallVector<mlir::LLVM::AddressOfOp, 4> addresses;
  llvm::SmallVector<mlir::LLVM::UndefOp, 4> undefs;
  llvm::SmallVector<mlir::Value, 8> inputs;
  llvm::SmallVector<mlir::Value, 1> outputs;
};

struct CallableSpatialSlice final {
  llvm::SmallVector<mlir::Operation *, 16> body;
  llvm::SmallVector<mlir::Value, 8> liveIns;
  llvm::SmallVector<mlir::Value, 1> liveOuts;
};

CallableOwnershipBoundary
deriveCallableOwnershipBoundary(mlir::LLVM::LLVMFuncOp function);

CallableSpatialSlice
deriveCallableSpatialSlice(mlir::LLVM::LLVMFuncOp function,
                           const CallableOwnershipBoundary &boundary);

std::optional<std::string>
completeMemoryServiceBoundary(llvm::ArrayRef<mlir::Operation *> selectedBody,
                              std::vector<mlir::Value> &liveIns);

/// Whether this exact selection already holds a memory access that only a
/// stored pointer serves. `completeMemoryServiceBoundary` binds such an access
/// to a projected pointer service, except under a root-relative projection
/// which marks every selected access root-relative and refuses it. Both the
/// pre-materialization admission hint and the materialization check read the
/// same unbound-access rule, so the hint cannot drift from the refusal.
bool selectionHasUnboundPointerServiceAccess(mlir::Operation *selection);

} // namespace loom::frontend::detail

#endif // LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDOWNERSHIPANALYSIS_H

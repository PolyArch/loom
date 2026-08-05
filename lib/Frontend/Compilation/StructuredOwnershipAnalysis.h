#ifndef LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDOWNERSHIPANALYSIS_H
#define LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDOWNERSHIPANALYSIS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

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
explainUnboundMemoryService(llvm::ArrayRef<mlir::Operation *> selectedBody,
                            llvm::ArrayRef<mlir::Value> liveIns);

} // namespace loom::frontend::detail

#endif // LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDOWNERSHIPANALYSIS_H

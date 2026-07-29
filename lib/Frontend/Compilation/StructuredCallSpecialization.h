#ifndef LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDCALLSPECIALIZATION_H
#define LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDCALLSPECIALIZATION_H

#include "llvm/Support/Error.h"

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class Operation;
}

namespace loom::frontend::detail {

llvm::Expected<bool>
hasUniformExactCallArgumentSpecialization(mlir::ModuleOp module,
                                          mlir::Operation *selection);

llvm::Expected<mlir::Operation *>
materializeUniformExactCallArgumentSpecialization(mlir::ModuleOp module,
                                                  mlir::Operation *selection);

} // namespace loom::frontend::detail

#endif // LOOM_LIB_FRONTEND_COMPILATION_STRUCTUREDCALLSPECIALIZATION_H

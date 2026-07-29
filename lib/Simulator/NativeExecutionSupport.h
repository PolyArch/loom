#ifndef LOOM_SIMULATOR_NATIVEEXECUTIONSUPPORT_H
#define LOOM_SIMULATOR_NATIVEEXECUTIONSUPPORT_H

#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/Support/Error.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace llvm {
class Module;
namespace orc {
class LLJIT;
}
} // namespace llvm

namespace loom::sim::detail {

llvm::Error initializeNativeTarget();

llvm::Error lowerStructuredModuleToLlvmDialect(mlir::ModuleOp module);

llvm::Expected<llvm::orc::ThreadSafeModule>
lowerStructuredModuleToLlvm(mlir::OwningOpRef<mlir::ModuleOp> module);

llvm::Error admitNativeHostModule(llvm::Module &module,
                                  const llvm::orc::LLJIT &jit);

/// Proves every execution-visible layout fact is host-equivalent before
/// retargeting only an ephemeral oracle clone.
llvm::Error retargetStructuredOracle(llvm::Module &module,
                                     const llvm::orc::LLJIT &jit);

} // namespace loom::sim::detail

#endif // LOOM_SIMULATOR_NATIVEEXECUTIONSUPPORT_H

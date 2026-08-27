#ifndef LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTBYTECODEINTERNAL_H
#define LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTBYTECODEINTERNAL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace mlir {
class MLIRContext;
class Operation;
} // namespace mlir

namespace loom::fabric::detail {

struct ParsedFabricBytecodeModule {
  std::shared_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

llvm::Expected<ParsedFabricBytecodeModule>
parseFabricBytecodeModule(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<std::vector<std::uint8_t>>
writeCanonicalFabricBytecode(mlir::Operation *operation);

/// Proves that one parsed canonical module re-serializes to exactly the stored
/// canonical bytes. Publication already established the writer fixpoint
/// (write(parse(bytes)) == bytes), and an imported module is parse(bytes), so
/// a single re-serialization is the same predicate without the publication
/// path's text normalization round trip.
llvm::Error
verifyCanonicalFabricBytecodeStability(mlir::Operation *operation,
                                       llvm::ArrayRef<std::uint8_t> canonical);

} // namespace loom::fabric::detail

#endif // LOOM_LIB_FABRIC_ARTIFACT_FABRICARTIFACTBYTECODEINTERNAL_H

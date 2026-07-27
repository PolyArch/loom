#ifndef LOOM_LIB_DATAFLOW_IR_DATAFLOW_CANONICAL_BYTECODE_INTERNAL_H
#define LOOM_LIB_DATAFLOW_IR_DATAFLOW_CANONICAL_BYTECODE_INTERNAL_H

#include "Common/Artifact.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace dataflow::detail {

struct ParsedCanonicalDataflowModule {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

llvm::Error canonicalizeDataflowPresentation(mlir::ModuleOp module);

llvm::Expected<ParsedCanonicalDataflowModule>
parseCanonicalDataflowBytecode(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<std::vector<std::uint8_t>>
writeCanonicalDataflowBytecode(mlir::ModuleOp module);

::loom::CanonicalSemanticBytes
frameCanonicalDataflowBytes(llvm::ArrayRef<std::uint8_t> bytecode);

llvm::Expected<llvm::ArrayRef<std::uint8_t>>
extractCanonicalDataflowBytecode(
    const ::loom::CanonicalSemanticBytes &canonicalBytes);

} // namespace dataflow::detail

#endif // LOOM_LIB_DATAFLOW_IR_DATAFLOW_CANONICAL_BYTECODE_INTERNAL_H

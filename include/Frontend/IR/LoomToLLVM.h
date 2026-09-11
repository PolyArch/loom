#ifndef LOOM_FRONTEND_IR_LOOMTOLLVM_H
#define LOOM_FRONTEND_IR_LOOMTOLLVM_H

namespace mlir {
class DialectRegistry;
}

namespace loom {

/// Registers the Loom dialect's `ConvertToLLVMPatternInterface`, which lowers
/// `loom.pointer_view` to the memref descriptor of its pointer. Every owner
/// that converts a Structured Program to the LLVM dialect registers it beside
/// the MemRef and Vector interfaces.
void registerConvertLoomToLLVMInterface(mlir::DialectRegistry &registry);

} // namespace loom

#endif // LOOM_FRONTEND_IR_LOOMTOLLVM_H

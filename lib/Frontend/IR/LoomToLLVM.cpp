#include "Frontend/IR/LoomToLLVM.h"

#include "Frontend/IR/LoomDialect.h"
#include "Frontend/IR/LoomOps.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace loom {
namespace {

/// A pointer view is the memref descriptor of its pointer: allocated and
/// aligned base both the pointer, offset zero, and the static shape and
/// strides of the view type.
struct PointerViewOpLowering final
    : public mlir::ConvertOpToLLVMPattern<PointerViewOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  mlir::LogicalResult
  matchAndRewrite(PointerViewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto memory = llvm::cast<mlir::MemRefType>(op.getResult().getType());
    mlir::Value descriptor = mlir::MemRefDescriptor::fromStaticShape(
        rewriter, op.getLoc(), *getTypeConverter(), memory,
        adaptor.getSource());
    rewriter.replaceOp(op, descriptor);
    return mlir::success();
  }
};

struct LoomToLLVMDialectInterface final
    : public mlir::ConvertToLLVMPatternInterface {
  explicit LoomToLLVMDialectInterface(mlir::Dialect *dialect)
      : ConvertToLLVMPatternInterface(dialect) {}

  void loadDependentDialects(mlir::MLIRContext *context) const final {
    context->loadDialect<mlir::LLVM::LLVMDialect>();
  }

  void populateConvertToLLVMConversionPatterns(
      mlir::ConversionTarget &target, mlir::LLVMTypeConverter &typeConverter,
      mlir::RewritePatternSet &patterns) const final {
    patterns.add<PointerViewOpLowering>(typeConverter);
  }
};

} // namespace

void registerConvertLoomToLLVMInterface(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *, LoomDialect *dialect) {
    dialect->addInterfaces<LoomToLLVMDialectInterface>();
  });
}

} // namespace loom

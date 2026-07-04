// Expand known library helper calls inside dataflow.graph.func bodies into
// primitive operations before PnR sees the graph. Unknown calls are left in
// place so the existing unsupported-call diagnostics remain the SSOT.

#include "Frontend/Lowering/Passes.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/SmallVector.h"

namespace {

constexpr ::llvm::StringLiteral kCmsisVecMatMultTS8 =
    "arm_nn_vec_mat_mult_t_s8";

::mlir::Value makeI32Constant(::mlir::OpBuilder &builder,
                              ::mlir::Location loc, std::int32_t value) {
  return ::mlir::arith::ConstantOp::create(
             builder, loc, builder.getI32Type(),
             builder.getIntegerAttr(builder.getI32Type(), value))
      .getResult();
}

::mlir::Value makeI64Constant(::mlir::OpBuilder &builder,
                              ::mlir::Location loc, std::int64_t value) {
  return ::mlir::arith::ConstantOp::create(
             builder, loc, builder.getI64Type(),
             builder.getIntegerAttr(builder.getI64Type(), value))
      .getResult();
}

::mlir::Value llvmGep(::mlir::OpBuilder &builder, ::mlir::Location loc,
                      ::mlir::Value base, ::mlir::Type elemTy,
                      ::mlir::Value index) {
  return ::mlir::LLVM::GEPOp::create(
             builder, loc, ::mlir::LLVM::LLVMPointerType::get(builder.getContext()),
             elemTy, base, ::mlir::ValueRange{index})
      .getResult();
}

::mlir::Value llvmLoad(::mlir::OpBuilder &builder, ::mlir::Location loc,
                       ::mlir::Type elemTy, ::mlir::Value ptr) {
  return ::mlir::LLVM::LoadOp::create(
             builder, loc, elemTy, ptr, /*alignment=*/nullptr,
             /*volatile_=*/false, /*nontemporal=*/false, /*invariant=*/false,
             /*invariantGroup=*/false, ::mlir::LLVM::AtomicOrdering::not_atomic,
             /*syncscope=*/nullptr, /*dereferenceable=*/nullptr,
             /*access_groups=*/nullptr, /*alias_scopes=*/nullptr,
             /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr)
      .getResult();
}

void llvmStore(::mlir::OpBuilder &builder, ::mlir::Location loc,
               ::mlir::Value value, ::mlir::Value ptr) {
  ::mlir::LLVM::StoreOp::create(
      builder, loc, value, ptr, /*alignment=*/nullptr,
      /*volatile_=*/false, /*nontemporal=*/false, /*invariantGroup=*/false,
      ::mlir::LLVM::AtomicOrdering::not_atomic, /*syncscope=*/nullptr,
      /*access_groups=*/nullptr, /*alias_scopes=*/nullptr,
      /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr);
}

::mlir::Value llvmSExt(::mlir::OpBuilder &builder, ::mlir::Location loc,
                       ::mlir::Value value, ::mlir::Type resultTy) {
  return ::mlir::LLVM::SExtOp::create(builder, loc, resultTy, value)
      .getResult();
}

::mlir::Value llvmTrunc(::mlir::OpBuilder &builder, ::mlir::Location loc,
                        ::mlir::Value value, ::mlir::Type resultTy) {
  return ::mlir::LLVM::TruncOp::create(
             builder, loc, resultTy, value,
             ::mlir::LLVM::IntegerOverflowFlags::none)
      .getResult();
}

::mlir::Value selectMaxI32(::mlir::OpBuilder &builder, ::mlir::Location loc,
                           ::mlir::Value lhs, ::mlir::Value rhs) {
  auto pred = ::mlir::arith::CmpIOp::create(
      builder, loc, ::mlir::arith::CmpIPredicate::sgt, lhs, rhs);
  return ::mlir::arith::SelectOp::create(builder, loc, pred, lhs, rhs)
      .getResult();
}

::mlir::Value selectMinI32(::mlir::OpBuilder &builder, ::mlir::Location loc,
                           ::mlir::Value lhs, ::mlir::Value rhs) {
  auto pred = ::mlir::arith::CmpIOp::create(
      builder, loc, ::mlir::arith::CmpIPredicate::slt, lhs, rhs);
  return ::mlir::arith::SelectOp::create(builder, loc, pred, lhs, rhs)
      .getResult();
}

::mlir::Value buildDivideByPowerOfTwo(::mlir::OpBuilder &builder,
                                      ::mlir::Location loc,
                                      ::mlir::Value dividend,
                                      ::mlir::Value exponent) {
  ::mlir::Type i32Ty = builder.getI32Type();
  ::mlir::Value zero = makeI32Constant(builder, loc, 0);
  ::mlir::Value one = makeI32Constant(builder, loc, 1);
  auto hasRightShift = ::mlir::arith::CmpIOp::create(
      builder, loc, ::mlir::arith::CmpIPredicate::sgt, exponent, zero);

  auto ifOp = ::mlir::scf::IfOp::create(
      builder, loc, ::mlir::TypeRange{i32Ty}, hasRightShift,
      /*withElseRegion=*/true);

  {
    ::mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    ::mlir::Value thirty = makeI32Constant(builder, loc, 30);
    ::mlir::Value limited = selectMinI32(builder, loc, exponent, thirty);
    ::mlir::Value maskBase =
        ::mlir::arith::ShLIOp::create(builder, loc, one, limited).getResult();
    ::mlir::Value remainderMask =
        ::mlir::arith::SubIOp::create(builder, loc, maskBase, one).getResult();
    ::mlir::Value remainder =
        ::mlir::arith::AndIOp::create(builder, loc, remainderMask, dividend)
            .getResult();
    ::mlir::Value result =
        ::mlir::arith::ShRSIOp::create(builder, loc, dividend, limited)
            .getResult();
    ::mlir::Value threshold =
        ::mlir::arith::ShRSIOp::create(builder, loc, remainderMask, one)
            .getResult();
    auto resultIsNegative = ::mlir::arith::CmpIOp::create(
        builder, loc, ::mlir::arith::CmpIPredicate::slt, result, zero);
    ::mlir::Value thresholdPlusOne =
        ::mlir::arith::AddIOp::create(builder, loc, threshold, one).getResult();
    threshold = ::mlir::arith::SelectOp::create(
                    builder, loc, resultIsNegative, thresholdPlusOne, threshold)
                    .getResult();
    auto shouldRoundUp = ::mlir::arith::CmpIOp::create(
        builder, loc, ::mlir::arith::CmpIPredicate::sgt, remainder, threshold);
    ::mlir::Value rounded =
        ::mlir::arith::AddIOp::create(builder, loc, result, one).getResult();
    ::mlir::Value selected =
        ::mlir::arith::SelectOp::create(builder, loc, shouldRoundUp, rounded,
                                       result)
            .getResult();
    ::mlir::scf::YieldOp::create(builder, loc, selected);
  }

  {
    ::mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    ::mlir::scf::YieldOp::create(builder, loc, dividend);
  }

  return ifOp.getResult(0);
}

::mlir::Value buildCmsisRequantize(::mlir::OpBuilder &builder,
                                   ::mlir::Location loc, ::mlir::Value value,
                                   ::mlir::Value multiplier,
                                   ::mlir::Value shift) {
  ::mlir::Value zero = makeI32Constant(builder, loc, 0);
  ::mlir::Value leftShift = selectMaxI32(builder, loc, shift, zero);
  ::mlir::Value negShift =
      ::mlir::arith::SubIOp::create(builder, loc, zero, shift).getResult();
  ::mlir::Value rightShift = selectMaxI32(builder, loc, negShift, zero);
  ::mlir::Value shifted =
      ::mlir::arith::ShLIOp::create(builder, loc, value, leftShift).getResult();

  ::mlir::Type i64Ty = builder.getI64Type();
  ::mlir::Value shifted64 = llvmSExt(builder, loc, shifted, i64Ty);
  ::mlir::Value multiplier64 = llvmSExt(builder, loc, multiplier, i64Ty);
  ::mlir::Value product =
      ::mlir::arith::MulIOp::create(builder, loc, shifted64, multiplier64)
          .getResult();
  ::mlir::Value rounding = makeI64Constant(builder, loc, std::int64_t{1} << 30);
  product = ::mlir::arith::AddIOp::create(builder, loc, product, rounding)
                .getResult();
  ::mlir::Value high =
      ::mlir::arith::ShRSIOp::create(builder, loc, product,
                                     makeI64Constant(builder, loc, 31))
          .getResult();
  ::mlir::Value multiplied =
      llvmTrunc(builder, loc, high, builder.getI32Type());
  return buildDivideByPowerOfTwo(builder, loc, multiplied, rightShift);
}

bool isDirectCmsisVecMatMultTS8(::mlir::LLVM::CallOp call) {
  auto callee = call.getCalleeAttr();
  return callee && callee.getValue() == kCmsisVecMatMultTS8 &&
         call->getNumOperands() == 15 && call->getNumResults() == 1 &&
         call.getResult().getType().isInteger(32);
}

bool expandCmsisVecMatMultTS8(::mlir::LLVM::CallOp call,
                              ::mlir::OpBuilder &builder) {
  if (!isDirectCmsisVecMatMultTS8(call))
    return false;

  ::mlir::Location loc = call.getLoc();
  ::llvm::SmallVector<::mlir::Value, 15> operands(call->operand_begin(),
                                                  call->operand_end());
  ::mlir::Value lhs = operands[0];
  ::mlir::Value rhs = operands[1];
  ::mlir::Value bias = operands[3];
  ::mlir::Value dst = operands[4];
  ::mlir::Value lhsOffset = operands[5];
  ::mlir::Value dstOffset = operands[6];
  ::mlir::Value dstMultiplier = operands[7];
  ::mlir::Value dstShift = operands[8];
  ::mlir::Value rhsCols = operands[9];
  ::mlir::Value rhsRows = operands[10];
  ::mlir::Value activationMin = operands[11];
  ::mlir::Value activationMax = operands[12];
  ::mlir::Value addressOffset = operands[13];
  ::mlir::Value rhsOffset = operands[14];

  ::mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(call);
  ::mlir::Type i8Ty = builder.getI8Type();
  ::mlir::Type i32Ty = builder.getI32Type();
  ::mlir::Value zero = makeI32Constant(builder, loc, 0);
  ::mlir::Value one = makeI32Constant(builder, loc, 1);

  auto rowLoopBody = [&](::mlir::OpBuilder &rowBuilder,
                         ::mlir::Location rowLoc, ::mlir::Value row,
                         ::mlir::ValueRange) {
    ::mlir::Value nullPtr =
        ::mlir::LLVM::ZeroOp::create(rowBuilder, rowLoc, bias.getType())
            .getResult();
    ::mlir::Value hasBias =
        ::mlir::LLVM::ICmpOp::create(rowBuilder, rowLoc,
                                     ::mlir::LLVM::ICmpPredicate::ne, bias,
                                     nullPtr)
            .getResult();
    auto biasIf = ::mlir::scf::IfOp::create(
        rowBuilder, rowLoc, ::mlir::TypeRange{i32Ty}, hasBias,
        /*withElseRegion=*/true);
    {
      ::mlir::OpBuilder::InsertionGuard biasGuard(rowBuilder);
      rowBuilder.setInsertionPointToStart(&biasIf.getThenRegion().front());
      ::mlir::Value biasPtr = llvmGep(rowBuilder, rowLoc, bias, i32Ty, row);
      ::mlir::Value biasValue = llvmLoad(rowBuilder, rowLoc, i32Ty, biasPtr);
      ::mlir::scf::YieldOp::create(rowBuilder, rowLoc, biasValue);
    }
    {
      ::mlir::OpBuilder::InsertionGuard biasGuard(rowBuilder);
      rowBuilder.setInsertionPointToStart(&biasIf.getElseRegion().front());
      ::mlir::scf::YieldOp::create(rowBuilder, rowLoc, zero);
    }

    auto colLoopBody = [&](::mlir::OpBuilder &colBuilder,
                           ::mlir::Location colLoc, ::mlir::Value col,
                           ::mlir::ValueRange iterArgs) {
      ::mlir::Value acc = iterArgs.front();
      ::mlir::Value lhsPtr = llvmGep(colBuilder, colLoc, lhs, i8Ty, col);
      ::mlir::Value lhsValue = llvmLoad(colBuilder, colLoc, i8Ty, lhsPtr);
      lhsValue = llvmSExt(colBuilder, colLoc, lhsValue, i32Ty);
      lhsValue =
          ::mlir::arith::AddIOp::create(colBuilder, colLoc, lhsValue, lhsOffset)
              .getResult();

      ::mlir::Value rowBase =
          ::mlir::arith::MulIOp::create(colBuilder, colLoc, row, rhsCols)
              .getResult();
      ::mlir::Value rhsIndex =
          ::mlir::arith::AddIOp::create(colBuilder, colLoc, rowBase, col)
              .getResult();
      ::mlir::Value rhsPtr = llvmGep(colBuilder, colLoc, rhs, i8Ty, rhsIndex);
      ::mlir::Value rhsValue = llvmLoad(colBuilder, colLoc, i8Ty, rhsPtr);
      rhsValue = llvmSExt(colBuilder, colLoc, rhsValue, i32Ty);
      rhsValue =
          ::mlir::arith::AddIOp::create(colBuilder, colLoc, rhsValue, rhsOffset)
              .getResult();

      ::mlir::Value product =
          ::mlir::arith::MulIOp::create(colBuilder, colLoc, lhsValue, rhsValue)
              .getResult();
      ::mlir::Value next =
          ::mlir::arith::AddIOp::create(colBuilder, colLoc, acc, product)
              .getResult();
      ::mlir::scf::YieldOp::create(colBuilder, colLoc, next);
    };
    auto colLoop =
        ::mlir::scf::ForOp::create(rowBuilder, rowLoc, zero, rhsCols, one,
                                   ::mlir::ValueRange{biasIf.getResult(0)},
                                   colLoopBody);
    ::mlir::Value acc = colLoop.getResult(0);
    acc = buildCmsisRequantize(rowBuilder, rowLoc, acc, dstMultiplier, dstShift);
    acc = ::mlir::arith::AddIOp::create(rowBuilder, rowLoc, acc, dstOffset)
              .getResult();
    acc = selectMaxI32(rowBuilder, rowLoc, acc, activationMin);
    acc = selectMinI32(rowBuilder, rowLoc, acc, activationMax);
    ::mlir::Value dstValue = llvmTrunc(rowBuilder, rowLoc, acc, i8Ty);
    ::mlir::Value dstIndex =
        ::mlir::arith::MulIOp::create(rowBuilder, rowLoc, row, addressOffset)
            .getResult();
    ::mlir::Value dstPtr = llvmGep(rowBuilder, rowLoc, dst, i8Ty, dstIndex);
    llvmStore(rowBuilder, rowLoc, dstValue, dstPtr);
    ::mlir::scf::YieldOp::create(rowBuilder, rowLoc);
  };

  ::mlir::scf::ForOp::create(builder, loc, zero, rhsRows, one,
                             ::mlir::ValueRange{}, rowLoopBody);
  ::mlir::Value status = makeI32Constant(builder, loc, 0);
  call.getResult().replaceAllUsesWith(status);
  call.erase();
  return true;
}

unsigned rewriteGraph(::dataflow::GraphFuncOp graph,
                      ::mlir::OpBuilder &builder) {
  ::llvm::SmallVector<::mlir::LLVM::CallOp, 4> calls;
  graph.walk([&](::mlir::LLVM::CallOp call) {
    if (isDirectCmsisVecMatMultTS8(call))
      calls.push_back(call);
  });
  unsigned rewrites = 0;
  for (::mlir::LLVM::CallOp call : calls)
    if (expandCmsisVecMatMultTS8(call, builder))
      ++rewrites;
  return rewrites;
}

struct LowerKnownLibraryCallsPass
    : public ::mlir::PassWrapper<LowerKnownLibraryCallsPass,
                                 ::mlir::OperationPass<::mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerKnownLibraryCallsPass)

  ::llvm::StringRef getArgument() const final {
    return "loom-lower-known-library-calls";
  }
  ::llvm::StringRef getDescription() const final {
    return "Expand known library calls inside dataflow.graph.func bodies "
           "before graph memory lowering and PnR.";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const final {
    registry.insert<::mlir::arith::ArithDialect, ::mlir::func::FuncDialect,
                    ::mlir::LLVM::LLVMDialect, ::mlir::scf::SCFDialect,
                    ::dataflow::DataflowDialect>();
  }

  void runOnOperation() final {
    ::mlir::ModuleOp module = getOperation();
    ::mlir::OpBuilder builder(&getContext());

    ::llvm::SmallVector<::dataflow::GraphFuncOp, 8> graphs;
    for (auto graph : module.getOps<::dataflow::GraphFuncOp>())
      graphs.push_back(graph);

    for (::dataflow::GraphFuncOp graph : graphs) {
      if (graph.isExternal())
        continue;
      (void)rewriteGraph(graph, builder);
    }
  }
};

} // namespace

namespace loom {
namespace lowering {

std::unique_ptr<::mlir::Pass> createLowerKnownLibraryCallsPass() {
  return std::make_unique<LowerKnownLibraryCallsPass>();
}

void registerLowerKnownLibraryCallsPass() {
  static bool once = []() {
    ::mlir::PassRegistration<LowerKnownLibraryCallsPass>();
    return true;
  }();
  (void)once;
}

} // namespace lowering
} // namespace loom

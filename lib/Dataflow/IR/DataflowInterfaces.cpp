#include "Dataflow/IR/DataflowInterfaces.h"

#include "Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

using namespace mlir;

#include "Dataflow/IR/DataflowInterfaces.cpp.inc"

namespace {

template <typename OpTy>
struct CanonicalActorModel final
    : dataflow::CanonicalDataflowActorOpInterface::ExternalModel<
          CanonicalActorModel<OpTy>, OpTy> {};

template <typename... OpTys>
void attachActorModels(MLIRContext &context) {
  (OpTys::template attachInterface<CanonicalActorModel<OpTys>>(context), ...);
}

} // namespace

void dataflow::attachCanonicalDataflowActorInterfaces(MLIRContext &context) {
  context.getOrLoadDialect<arith::ArithDialect>();
  attachActorModels<arith::ConstantOp, arith::AddFOp, arith::SubFOp,
                    arith::MulFOp, arith::DivFOp, arith::AddIOp,
                    arith::SubIOp, arith::MulIOp, arith::AndIOp,
                    arith::OrIOp, arith::XOrIOp, arith::ShLIOp,
                    arith::ShRSIOp, arith::ShRUIOp, arith::DivSIOp,
                    arith::DivUIOp, arith::RemSIOp, arith::RemUIOp,
                    arith::CmpIOp, arith::CmpFOp, arith::SelectOp,
                    arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp,
                    arith::MaxUIOp,
                    arith::IndexCastOp, arith::IndexCastUIOp,
                    arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
                    arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
                    arith::FPToUIOp>(context);

  context.getOrLoadDialect<math::MathDialect>();
  attachActorModels<math::AbsFOp, math::AbsIOp, math::SinOp, math::CosOp,
                    math::TanOp, math::SinhOp, math::CoshOp, math::TanhOp,
                    math::ExpOp, math::Exp2Op, math::ExpM1Op, math::LogOp,
                    math::Log2Op, math::Log10Op, math::Log1pOp,
                    math::FloorOp, math::CeilOp, math::RoundOp,
                    math::TruncOp, math::RoundEvenOp, math::SqrtOp,
                    math::RsqrtOp, math::ErfOp>(context);

  context.getOrLoadDialect<LLVM::LLVMDialect>();
  attachActorModels<
      LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::SDivOp, LLVM::UDivOp,
      LLVM::SRemOp, LLVM::URemOp, LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp,
      LLVM::ShlOp, LLVM::LShrOp, LLVM::AShrOp, LLVM::FAddOp, LLVM::FSubOp,
      LLVM::FMulOp, LLVM::FDivOp, LLVM::FRemOp, LLVM::FNegOp, LLVM::ICmpOp,
      LLVM::FCmpOp, LLVM::BitcastOp, LLVM::TruncOp, LLVM::ZExtOp,
      LLVM::SExtOp, LLVM::FPTruncOp, LLVM::FPExtOp, LLVM::SIToFPOp,
      LLVM::UIToFPOp, LLVM::FPToSIOp, LLVM::FPToUIOp, LLVM::SelectOp,
      LLVM::FreezeOp, LLVM::ExtractElementOp, LLVM::InsertElementOp,
      LLVM::ExtractValueOp, LLVM::InsertValueOp, LLVM::ShuffleVectorOp,
      LLVM::FshlOp, LLVM::ByteSwapOp, LLVM::UMinOp, LLVM::UMaxOp,
      LLVM::USubSat, LLVM::SMinOp, LLVM::SMaxOp,
      LLVM::CountLeadingZerosOp, LLVM::FMulAddOp, LLVM::AbsOp,
      LLVM::FAbsOp>(context);

  context.getOrLoadDialect<ub::UBDialect>();
  attachActorModels<ub::PoisonOp>(context);
}

std::optional<dataflow::CanonicalDataflowActorKind>
dataflow::classifyCanonicalDataflowActor(Operation *op) {
  if (!llvm::isa<CanonicalDataflowActorOpInterface>(op))
    return std::nullopt;
  if (llvm::isa<LoadOp, StoreOp>(op))
    return CanonicalDataflowActorKind::Memory;
  if (llvm::isa<StreamOp, CarryOp, InvariantOp, GateOp, ParallelizeOp,
                SerializeOp, ConstantOp, SyncOp, MuxOp, DemuxOp>(op))
    return CanonicalDataflowActorKind::Control;
  return CanonicalDataflowActorKind::Compute;
}

bool dataflow::isCanonicalDataflowActor(Operation *op) {
  return classifyCanonicalDataflowActor(op).has_value();
}

bool dataflow::isCanonicalDataflowActor(Operation *op,
                                        CanonicalDataflowActorKind kind) {
  return classifyCanonicalDataflowActor(op) == kind;
}

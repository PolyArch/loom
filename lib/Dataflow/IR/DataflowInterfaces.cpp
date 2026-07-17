#include "Dataflow/IR/DataflowInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/OperationSupport.h"

using namespace mlir;

#include "Dataflow/IR/DataflowInterfaces.cpp.inc"

namespace {

struct CanonicalActorModel final
    : dataflow::CanonicalDataflowActorOpInterface::FallbackModel<
          CanonicalActorModel> {};

template <size_t N>
void attachActorModels(MLIRContext *context,
                       const StringLiteral (&names)[N]) {
  for (StringLiteral name : names) {
    std::optional<RegisteredOperationName> operation =
        RegisteredOperationName::lookup(name, context);
    if (operation)
      operation->attachInterface<CanonicalActorModel>();
  }
}

} // namespace

void dataflow::attachCanonicalDataflowActorInterfaces(MLIRContext &context) {
  context.getOrLoadDialect<arith::ArithDialect>();
  static constexpr StringLiteral arithActors[] = {
      "arith.constant",     "arith.addf",       "arith.subf",
      "arith.mulf",         "arith.divf",       "arith.addi",
      "arith.subi",         "arith.muli",       "arith.andi",
      "arith.ori",          "arith.xori",       "arith.shli",
      "arith.shrsi",        "arith.shrui",      "arith.divsi",
      "arith.divui",        "arith.remsi",      "arith.remui",
      "arith.cmpi",         "arith.cmpf",       "arith.select",
      "arith.index_cast",   "arith.index_castui", "arith.extsi",
      "arith.extui",        "arith.trunci",     "arith.sitofp",
      "arith.uitofp",       "arith.fptosi",     "arith.fptoui",
  };
  attachActorModels(&context, arithActors);

  context.getOrLoadDialect<math::MathDialect>();
  static constexpr StringLiteral mathActors[] = {
      "math.absf",      "math.absi",      "math.sin",
      "math.cos",       "math.tan",       "math.sinh",
      "math.cosh",      "math.tanh",      "math.exp",
      "math.exp2",      "math.expm1",     "math.log",
      "math.log2",      "math.log10",     "math.log1p",
      "math.floor",     "math.ceil",      "math.round",
      "math.trunc",     "math.roundeven", "math.sqrt",
      "math.rsqrt",     "math.erf",
  };
  attachActorModels(&context, mathActors);

  context.getOrLoadDialect<LLVM::LLVMDialect>();
  static constexpr StringLiteral llvmActors[] = {
      "llvm.add",            "llvm.sub",           "llvm.mul",
      "llvm.sdiv",           "llvm.udiv",          "llvm.srem",
      "llvm.urem",           "llvm.and",           "llvm.or",
      "llvm.xor",            "llvm.shl",           "llvm.lshr",
      "llvm.ashr",           "llvm.fadd",          "llvm.fsub",
      "llvm.fmul",           "llvm.fdiv",          "llvm.frem",
      "llvm.fneg",           "llvm.icmp",          "llvm.fcmp",
      "llvm.bitcast",        "llvm.trunc",         "llvm.zext",
      "llvm.sext",           "llvm.fptrunc",       "llvm.fpext",
      "llvm.sitofp",         "llvm.uitofp",        "llvm.fptosi",
      "llvm.fptoui",         "llvm.select",        "llvm.freeze",
      "llvm.extractelement", "llvm.insertelement", "llvm.extractvalue",
      "llvm.insertvalue",    "llvm.shufflevector", "llvm.call_intrinsic",
      "llvm.inline_asm",     "llvm.intr.fshl",     "llvm.intr.bswap",
      "llvm.intr.umin",      "llvm.intr.umax",     "llvm.intr.usub.sat",
      "llvm.intr.smin",      "llvm.intr.smax",     "llvm.intr.ctlz",
      "llvm.intr.fmuladd",   "llvm.intr.abs",      "llvm.intr.fabs",
  };
  attachActorModels(&context, llvmActors);

  context.getOrLoadDialect<ub::UBDialect>();
  static constexpr StringLiteral ubActors[] = {"ub.poison"};
  attachActorModels(&context, ubActors);
}

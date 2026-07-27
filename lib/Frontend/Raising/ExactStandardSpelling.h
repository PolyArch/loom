#ifndef LOOM_LIB_FRONTEND_RAISING_EXACTSTANDARDSPELLING_H
#define LOOM_LIB_FRONTEND_RAISING_EXACTSTANDARDSPELLING_H

#include "CallableRegions.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Attributes.h"

#include <optional>
#include <utility>

namespace loom {
namespace raising {

// The conditions under which a standard `arith` or `math` operation restates
// an LLVM computation exactly. Mechanical alias normalization and typed
// fmuladd materialization both have to prove the same facts, so they are
// stated once here rather than restated per pass.

// True when `type` has an exact standard counterpart: a signless integer of
// non-zero width, `index`, a float, or a fixed-shape vector of those.
//
// arith rejects zero-width and signed integers. A scalable vector's element
// count is a runtime `vscale` multiple rather than a shape, so it fails closed
// here and keeps its operations in llvm form: only once a typed structured
// transform has materialized the computation as fixed-width chunks, loops, and
// masks or tails do the resulting operations hold a fixed shape that these
// aliases accept.
inline bool isExactNumericType(::mlir::Type type) {
  if (auto vectorType = ::mlir::dyn_cast<::mlir::VectorType>(type)) {
    if (vectorType.isScalable())
      return false;
    type = vectorType.getElementType();
  }
  if (auto integerType = ::mlir::dyn_cast<::mlir::IntegerType>(type))
    return integerType.isSignless() && integerType.getWidth() > 0;
  return ::mlir::isa<::mlir::IndexType, ::mlir::FloatType>(type);
}

inline bool allExactNumericTypes(::mlir::ValueRange values) {
  for (::mlir::Value value : values) {
    if (!isExactNumericType(value.getType()))
      return false;
  }
  return true;
}

inline bool
statesDefaultDenormalEnvironment(::mlir::LLVM::DenormalFPEnvAttr env) {
  using Kind = ::mlir::LLVM::DenormalModeKind;
  return env.getDefaultOutputMode() == Kind::IEEE &&
         env.getDefaultInputMode() == Kind::IEEE &&
         env.getFloatOutputMode() == Kind::IEEE &&
         env.getFloatInputMode() == Kind::IEEE;
}

// True when `funcOp` states a floating-point policy that no standard MLIR
// operation restates.
//
// An unflagged standard floating operation means the pinned default LLVM
// floating-point environment, and neither arith nor math states an enclosing
// environment of its own. The typed attributes read below are compared against
// that default directly. A reciprocal-estimate policy names the operations the
// target may compute as an estimate plus refinement rather than exactly, so
// any such policy blocks a rewrite. The importer's generic passthrough storage
// is classified separately because it also contains unrelated LLVM function
// and code-generation attributes.
inline bool passthroughEntryStatesFloatingPolicy(::mlir::Attribute entry) {
  ::llvm::StringRef name;
  ::std::optional<::llvm::StringRef> value;
  if (auto nameAttr = ::mlir::dyn_cast<::mlir::StringAttr>(entry)) {
    name = nameAttr.getValue();
  } else if (auto pair = ::mlir::dyn_cast<::mlir::ArrayAttr>(entry);
             pair && pair.size() == 2) {
    auto nameAttr = ::mlir::dyn_cast<::mlir::StringAttr>(pair[0]);
    auto valueAttr = ::mlir::dyn_cast<::mlir::StringAttr>(pair[1]);
    if (!nameAttr || !valueAttr)
      return true;
    name = nameAttr.getValue();
    value = valueAttr.getValue();
  } else {
    return true;
  }

  // The LLVM importer places every function attribute that LLVMFuncOp does
  // not model explicitly in one passthrough array. LLVM enum attributes still
  // retain their stable native spelling there. Of those function attributes,
  // strictfp alone changes the floating execution environment; the others
  // describe effects, control, ABI, or code generation without changing the
  // meaning of an ordinary floating instruction.
  const ::llvm::Attribute::AttrKind kind =
      ::llvm::Attribute::getAttrKindFromName(name);
  if (kind != ::llvm::Attribute::None)
    return kind == ::llvm::Attribute::StrictFP;

  // These string attributes are emitted by ordinary Clang compilation but
  // are not all modeled as typed LLVMFuncOp fields by the pinned importer.
  // Keep this list closed: an unknown string attribute may carry target
  // floating semantics and therefore fails closed.
  const bool codegenOnly = ::llvm::StringSwitch<bool>(name)
                               .Cases({"min-legal-vector-width",
                                       "stack-protector-buffer-size",
                                       "target-cpu"},
                                      true)
                               .Default(false);
  if (codegenOnly)
    return false;

  // Clang's default -ffp-exception-behavior=ignore spelling. A false or
  // malformed value cannot be represented by an unconstrained arith/math op.
  if (name == "no-trapping-math")
    return !value || *value != "true";

  return true;
}

inline bool statesFloatingPolicy(::mlir::LLVM::LLVMFuncOp funcOp) {
  if (auto env = funcOp.getDenormalFpenvAttr())
    if (!statesDefaultDenormalEnvironment(env))
      return true;
  if (auto noSignedZeros = funcOp.getNoSignedZerosFpMathAttr())
    if (noSignedZeros.getValue())
      return true;
  if (auto contraction = funcOp.getFpContractAttr())
    if (contraction.getValue() != "off")
      return true;
  if (funcOp.getReciprocalEstimatesAttr())
    return true;
  if (auto passthrough = funcOp.getPassthroughAttr())
    for (::mlir::Attribute entry : passthrough)
      if (passthroughEntryStatesFloatingPolicy(entry))
        return true;
  return false;
}

// True when the enclosing callable states a floating-point environment the
// standard operation cannot restate.
inline bool enclosingFloatingPolicyBlocksRewrite(::mlir::Operation *op) {
  auto funcOp = ::mlir::dyn_cast_or_null<::mlir::LLVM::LLVMFuncOp>(
      getNearestCallableOp(op));
  return funcOp && statesFloatingPolicy(funcOp);
}

// True when every operand and the single result of `op` have an exact standard
// counterpart and, for a computation that reads or produces a floating value,
// the enclosing callable states no environment the standard operation cannot
// restate. An integer computation is independent of that environment and is
// never blocked by it.
inline bool restatesExactly(::mlir::Operation *op, bool floating) {
  if (!allExactNumericTypes(op->getOperands()))
    return false;
  if (!isExactNumericType(op->getResult(0).getType()))
    return false;
  return !floating || !enclosingFloatingPolicyBlocksRewrite(op);
}

// arith counterpart of LLVM's fast-math flags. Both enums name the same
// seven facts but assign them different bit positions, so each flag is
// mapped by name instead of being reinterpreted.
inline ::mlir::arith::FastMathFlags
exactFastMathFlags(::mlir::LLVM::FastmathFlags flags) {
  const std::pair<::mlir::LLVM::FastmathFlags, ::mlir::arith::FastMathFlags>
      equivalents[] = {
          {::mlir::LLVM::FastmathFlags::nnan,
           ::mlir::arith::FastMathFlags::nnan},
          {::mlir::LLVM::FastmathFlags::ninf,
           ::mlir::arith::FastMathFlags::ninf},
          {::mlir::LLVM::FastmathFlags::nsz, ::mlir::arith::FastMathFlags::nsz},
          {::mlir::LLVM::FastmathFlags::arcp,
           ::mlir::arith::FastMathFlags::arcp},
          {::mlir::LLVM::FastmathFlags::contract,
           ::mlir::arith::FastMathFlags::contract},
          {::mlir::LLVM::FastmathFlags::afn, ::mlir::arith::FastMathFlags::afn},
          {::mlir::LLVM::FastmathFlags::reassoc,
           ::mlir::arith::FastMathFlags::reassoc}};

  ::mlir::arith::FastMathFlags result{};
  for (auto [llvmFlag, arithFlag] : equivalents) {
    if (::mlir::LLVM::bitEnumContainsAll(flags, llvmFlag))
      result = result | arithFlag;
  }
  return result;
}

// arith counterpart of LLVM's integer overflow flags. Both enums carry
// exactly nsw and nuw with the same meaning, so nothing is lost.
inline ::mlir::arith::IntegerOverflowFlags
exactOverflowFlags(::mlir::LLVM::IntegerOverflowFlags flags) {
  ::mlir::arith::IntegerOverflowFlags result{};
  if (::mlir::LLVM::bitEnumContainsAll(flags,
                                       ::mlir::LLVM::IntegerOverflowFlags::nsw))
    result = result | ::mlir::arith::IntegerOverflowFlags::nsw;
  if (::mlir::LLVM::bitEnumContainsAll(flags,
                                       ::mlir::LLVM::IntegerOverflowFlags::nuw))
    result = result | ::mlir::arith::IntegerOverflowFlags::nuw;
  return result;
}

} // namespace raising
} // namespace loom

#endif // LOOM_LIB_FRONTEND_RAISING_EXACTSTANDARDSPELLING_H

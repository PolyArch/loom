#include "Dataflow/IR/DataflowEnums.h"

#include "llvm/ADT/StringSwitch.h"

namespace dataflow {

std::optional<StepOp> symbolizeStepOp(llvm::StringRef symbol) {
  return llvm::StringSwitch<std::optional<StepOp>>(symbol)
      .Case("+=", StepOp::AddAssign)
      .Case("-=", StepOp::SubAssign)
      .Case("*=", StepOp::MulAssign)
      .Case("/=", StepOp::DivAssign)
      .Case("<<=", StepOp::ShlAssign)
      .Case(">>=", StepOp::ShrAssign)
      .Default(std::nullopt);
}

llvm::StringRef stringifyStepOp(StepOp value) {
  switch (value) {
  case StepOp::AddAssign:
    return "+=";
  case StepOp::SubAssign:
    return "-=";
  case StepOp::MulAssign:
    return "*=";
  case StepOp::DivAssign:
    return "/=";
  case StepOp::ShlAssign:
    return "<<=";
  case StepOp::ShrAssign:
    return ">>=";
  }
  return "";
}

llvm::ArrayRef<llvm::StringRef> getStepOpSymbols() {
  static const llvm::StringRef kSymbols[] = {"+=",  "-=", "*=",
                                              "/=", "<<=", ">>="};
  return kSymbols;
}

std::optional<ContCond> symbolizeContCond(llvm::StringRef symbol) {
  return llvm::StringSwitch<std::optional<ContCond>>(symbol)
      .Case("<", ContCond::Lt)
      .Case("<=", ContCond::Le)
      .Case(">", ContCond::Gt)
      .Case(">=", ContCond::Ge)
      .Case("!=", ContCond::Ne)
      .Default(std::nullopt);
}

llvm::StringRef stringifyContCond(ContCond value) {
  switch (value) {
  case ContCond::Lt:
    return "<";
  case ContCond::Le:
    return "<=";
  case ContCond::Gt:
    return ">";
  case ContCond::Ge:
    return ">=";
  case ContCond::Ne:
    return "!=";
  }
  return "";
}

llvm::ArrayRef<llvm::StringRef> getContCondSymbols() {
  static const llvm::StringRef kSymbols[] = {"<", "<=", ">", ">=", "!="};
  return kSymbols;
}

} // namespace dataflow

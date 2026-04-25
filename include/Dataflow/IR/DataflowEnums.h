#ifndef DATAFLOW_IR_DATAFLOWENUMS_H
#define DATAFLOW_IR_DATAFLOWENUMS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace dataflow {

// Enums for finite predicate sets carried by dataflow ops. The IR text
// continues to use string symbols ("+=", "<", ...) for readability, but
// internal analysis / verification code must work with these enum values
// and only convert at parser / printer boundaries.

enum class StepOp : std::uint8_t {
  AddAssign, // "+="
  SubAssign, // "-="
  MulAssign, // "*="
  DivAssign, // "/="
  ShlAssign, // "<<="
  ShrAssign, // ">>="
};

enum class ContCond : std::uint8_t {
  Lt, // "<"
  Le, // "<="
  Gt, // ">"
  Ge, // ">="
  Ne, // "!="
};

// Boundary helpers: parse a string symbol into an enum value, render an
// enum value back to its string symbol, and enumerate the valid symbols.
std::optional<StepOp> symbolizeStepOp(llvm::StringRef symbol);
llvm::StringRef stringifyStepOp(StepOp value);
llvm::ArrayRef<llvm::StringRef> getStepOpSymbols();

std::optional<ContCond> symbolizeContCond(llvm::StringRef symbol);
llvm::StringRef stringifyContCond(ContCond value);
llvm::ArrayRef<llvm::StringRef> getContCondSymbols();

} // namespace dataflow

#endif // DATAFLOW_IR_DATAFLOWENUMS_H

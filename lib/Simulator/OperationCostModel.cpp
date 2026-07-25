#include "Simulator/OperationCostModel.h"

#include <optional>
#include <system_error>

using namespace loom::sim;

namespace {

std::optional<OperationCost>
lookupOperationCost(dataflow::OperationSchemaId schema) {
  using Schema = dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::ArithConstant:
  case Schema::ArithNegF:
  case Schema::ArithAddI:
  case Schema::ArithSubI:
  case Schema::ArithShLI:
  case Schema::ArithShRSI:
  case Schema::ArithShRUI:
  case Schema::ArithAndI:
  case Schema::ArithOrI:
  case Schema::ArithXOrI:
  case Schema::ArithMinSI:
  case Schema::ArithMaxSI:
  case Schema::ArithMinUI:
  case Schema::ArithMaxUI:
  case Schema::ArithCmpI:
  case Schema::ArithSelect:
  case Schema::ArithExtSI:
  case Schema::ArithExtUI:
  case Schema::ArithTruncI:
  case Schema::ArithIndexCast:
  case Schema::ArithIndexCastUI:
  case Schema::MathAbsF:
  case Schema::MathAbsI:
  case Schema::MathCountLeadingZeros:
  case Schema::LLVMFshl:
  case Schema::LLVMByteSwap:
  case Schema::LLVMUSubSat:
  case Schema::LLVMCountLeadingZeros:
  case Schema::LLVMAbs:
  case Schema::UBPoison:
  case Schema::DataflowStream:
  case Schema::DataflowCarry:
  case Schema::DataflowInvariant:
  case Schema::DataflowGate:
  case Schema::DataflowConstant:
  case Schema::DataflowSync:
  case Schema::DataflowParallelize:
  case Schema::DataflowSerialize:
  case Schema::DataflowPack:
  case Schema::DataflowUnpack:
    return OperationCost{1, 1};

  case Schema::ArithAddF:
  case Schema::ArithSubF:
  case Schema::ArithCmpF:
  case Schema::MathFloor:
  case Schema::MathCeil:
  case Schema::MathRound:
  case Schema::MathTrunc:
  case Schema::MathRoundEven:
  case Schema::DataflowMux:
  case Schema::DataflowDemux:
    return OperationCost{2, 2};

  case Schema::ArithMulI:
  case Schema::ArithMulF:
  case Schema::ArithSIToFP:
  case Schema::ArithUIToFP:
  case Schema::ArithFPToSI:
  case Schema::ArithFPToUI:
    return OperationCost{3, 3};

  case Schema::DataflowLoad:
  case Schema::DataflowStore:
    return OperationCost{4, 4};

  case Schema::ArithDivSI:
  case Schema::ArithDivUI:
  case Schema::ArithRemSI:
  case Schema::ArithRemUI:
  case Schema::MathFma:
  case Schema::MathSqrt:
  case Schema::MathRsqrt:
    return OperationCost{8, 8};

  case Schema::ArithDivF:
  case Schema::MathExp:
  case Schema::MathExp2:
  case Schema::MathExpM1:
  case Schema::MathLog:
  case Schema::MathLog2:
  case Schema::MathLog10:
  case Schema::MathLog1p:
    return OperationCost{12, 12};

  case Schema::MathSin:
  case Schema::MathCos:
  case Schema::MathTan:
  case Schema::MathSinh:
  case Schema::MathCosh:
  case Schema::MathTanh:
  case Schema::MathErf:
    return OperationCost{16, 16};

  default:
    return std::nullopt;
  }
}

} // namespace

bool loom::sim::hasOperationCost(dataflow::OperationSchemaId schema) {
  return lookupOperationCost(schema).has_value();
}

llvm::Expected<OperationCost>
loom::sim::estimateOperationCost(dataflow::OperationSchemaId schema) {
  std::optional<OperationCost> cost = lookupOperationCost(schema);
  if (cost)
    return *cost;
  return llvm::createStringError(
      std::errc::invalid_argument, "%s has no simulator operation cost entry",
      dataflow::operationSchemaSpelling(schema).str().c_str());
}

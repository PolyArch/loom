//===-- CpuReferenceExecutor.cpp - CPU-side DFG reference exec ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Simulator/CpuReferenceExecutor.h"

#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"

#include <cmath>

namespace loom {
namespace sim {

namespace {

/// Return the bit width of an MLIR integer or index type (0 if not integer).
unsigned getIntBitWidth(mlir::Type ty) {
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty))
    return intTy.getWidth();
  if (mlir::isa<mlir::IndexType>(ty))
    return 64;
  return 0;
}

/// Mask a value to the given bit width.
uint64_t maskToWidth(uint64_t val, unsigned width) {
  if (width >= 64)
    return val;
  return val & ((1ULL << width) - 1);
}

/// Sign-extend from the given width to 64 bits.
int64_t signExtend(uint64_t val, unsigned width) {
  if (width == 0 || width >= 64)
    return static_cast<int64_t>(val);
  uint64_t signBit = 1ULL << (width - 1);
  if (val & signBit)
    return static_cast<int64_t>(val | ~((1ULL << width) - 1));
  return static_cast<int64_t>(val);
}

/// Evaluate one token through the handshake.func body.
/// Returns false if an unsupported op is encountered.
bool evaluateOneToken(
    circt::handshake::FuncOp funcOp,
    const std::vector<uint64_t> &inputValues,
    std::vector<uint64_t> &outputValues,
    std::string &unsupportedReason) {

  // Map SSA values to runtime uint64_t values.
  llvm::DenseMap<mlir::Value, uint64_t> valueMap;

  // Map function data arguments (excluding trailing ctrl/none args).
  auto args = funcOp.getBody().getArguments();
  unsigned dataArgIdx = 0;
  for (auto arg : args) {
    if (mlir::isa<mlir::NoneType>(arg.getType()))
      continue; // Skip control tokens.
    if (dataArgIdx >= inputValues.size()) {
      unsupportedReason = "more data arguments than input values";
      return false;
    }
    unsigned width = getIntBitWidth(arg.getType());
    valueMap[arg] = width > 0 ? maskToWidth(inputValues[dataArgIdx], width)
                              : inputValues[dataArgIdx];
    dataArgIdx++;
  }

  // Walk operations in program order.
  for (auto &op : funcOp.getBody().front()) {
    llvm::StringRef opName = op.getName().getStringRef();

    // handshake.return: collect outputs.
    if (auto returnOp = mlir::dyn_cast<circt::handshake::ReturnOp>(op)) {
      for (auto operand : returnOp.getOperands()) {
        if (mlir::isa<mlir::NoneType>(operand.getType()))
          continue;
        auto it = valueMap.find(operand);
        if (it == valueMap.end()) {
          unsupportedReason = "return operand not computed";
          return false;
        }
        outputValues.push_back(it->second);
      }
      return true;
    }

    // handshake.fork: replicate input to all outputs.
    if (mlir::isa<circt::handshake::ForkOp>(op)) {
      auto it = valueMap.find(op.getOperand(0));
      if (it == valueMap.end()) {
        unsupportedReason = "fork input not computed";
        return false;
      }
      for (auto result : op.getResults())
        valueMap[result] = it->second;
      continue;
    }

    // handshake.constant: extract the value attribute.
    if (auto constOp =
            mlir::dyn_cast<circt::handshake::ConstantOp>(op)) {
      auto valAttr = constOp.getValue();
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(valAttr))
        valueMap[constOp.getResult()] = intAttr.getValue().getZExtValue();
      else
        valueMap[constOp.getResult()] = 0;
      continue;
    }

    // arith.constant: extract integer attribute.
    if (auto arithConst = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      auto valAttr = arithConst.getValue();
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(valAttr))
        valueMap[arithConst.getResult()] = intAttr.getValue().getZExtValue();
      else
        valueMap[arithConst.getResult()] = 0;
      continue;
    }

    // handshake control-flow ops that pass data through unchanged.
    // merge/mux/branch are complex, but for straight-line code:
    // - join produces a control token (none type), skip
    // - sink consumes, skip
    // - source produces, skip
    if (mlir::isa<circt::handshake::JoinOp>(op) ||
        mlir::isa<circt::handshake::SinkOp>(op) ||
        mlir::isa<circt::handshake::SourceOp>(op)) {
      // These produce none-typed tokens or consume them.
      for (auto result : op.getResults()) {
        if (mlir::isa<mlir::NoneType>(result.getType()))
          valueMap[result] = 0;
      }
      continue;
    }

    // handshake.merge: for straight-line code, forward the single active input.
    if (mlir::isa<circt::handshake::MergeOp>(op)) {
      // Forward the first input that has a value.
      for (auto operand : op.getOperands()) {
        if (mlir::isa<mlir::NoneType>(operand.getType()))
          continue;
        auto it = valueMap.find(operand);
        if (it != valueMap.end()) {
          for (auto result : op.getResults()) {
            if (!mlir::isa<mlir::NoneType>(result.getType()))
              valueMap[result] = it->second;
          }
          break;
        }
      }
      continue;
    }

    // Arithmetic binary integer operations.
    auto evalBinaryInt = [&](auto compute) -> bool {
      if (op.getNumOperands() < 2 || op.getNumResults() < 1)
        return false;
      auto itA = valueMap.find(op.getOperand(0));
      auto itB = valueMap.find(op.getOperand(1));
      if (itA == valueMap.end() || itB == valueMap.end())
        return false;
      unsigned w = getIntBitWidth(op.getResult(0).getType());
      uint64_t result = compute(itA->second, itB->second, w);
      valueMap[op.getResult(0)] = w > 0 ? maskToWidth(result, w) : result;
      return true;
    };

    // arith.addi
    if (mlir::isa<mlir::arith::AddIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned) {
            return a + b;
          })) {
        unsupportedReason = "addi operands not computed";
        return false;
      }
      continue;
    }

    // arith.subi
    if (mlir::isa<mlir::arith::SubIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned) {
            return a - b;
          })) {
        unsupportedReason = "subi operands not computed";
        return false;
      }
      continue;
    }

    // arith.muli
    if (mlir::isa<mlir::arith::MulIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned) {
            return a * b;
          })) {
        unsupportedReason = "muli operands not computed";
        return false;
      }
      continue;
    }

    // arith.divsi (signed)
    if (mlir::isa<mlir::arith::DivSIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned w) -> uint64_t {
            if (b == 0)
              return 0;
            return static_cast<uint64_t>(signExtend(a, w) / signExtend(b, w));
          })) {
        unsupportedReason = "divsi operands not computed";
        return false;
      }
      continue;
    }

    // arith.divui (unsigned)
    if (mlir::isa<mlir::arith::DivUIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned) -> uint64_t {
            return b == 0 ? 0 : a / b;
          })) {
        unsupportedReason = "divui operands not computed";
        return false;
      }
      continue;
    }

    // arith.remsi (signed remainder)
    if (mlir::isa<mlir::arith::RemSIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned w) -> uint64_t {
            if (b == 0)
              return 0;
            return static_cast<uint64_t>(signExtend(a, w) % signExtend(b, w));
          })) {
        unsupportedReason = "remsi operands not computed";
        return false;
      }
      continue;
    }

    // arith.andi
    if (mlir::isa<mlir::arith::AndIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned) {
            return a & b;
          })) {
        unsupportedReason = "andi operands not computed";
        return false;
      }
      continue;
    }

    // arith.ori
    if (mlir::isa<mlir::arith::OrIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned) {
            return a | b;
          })) {
        unsupportedReason = "ori operands not computed";
        return false;
      }
      continue;
    }

    // arith.xori
    if (mlir::isa<mlir::arith::XOrIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned) {
            return a ^ b;
          })) {
        unsupportedReason = "xori operands not computed";
        return false;
      }
      continue;
    }

    // arith.shli
    if (mlir::isa<mlir::arith::ShLIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned) {
            return (b < 64) ? (a << b) : 0;
          })) {
        unsupportedReason = "shli operands not computed";
        return false;
      }
      continue;
    }

    // arith.shrsi (arithmetic shift right)
    if (mlir::isa<mlir::arith::ShRSIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned w) -> uint64_t {
            if (b >= w)
              return (signExtend(a, w) < 0) ? ~0ULL : 0;
            return static_cast<uint64_t>(signExtend(a, w) >> b);
          })) {
        unsupportedReason = "shrsi operands not computed";
        return false;
      }
      continue;
    }

    // arith.shrui (logical shift right)
    if (mlir::isa<mlir::arith::ShRUIOp>(op)) {
      if (!evalBinaryInt([](uint64_t a, uint64_t b, unsigned) -> uint64_t {
            return (b < 64) ? (a >> b) : 0;
          })) {
        unsupportedReason = "shrui operands not computed";
        return false;
      }
      continue;
    }

    // arith.cmpi
    if (auto cmpOp = mlir::dyn_cast<mlir::arith::CmpIOp>(op)) {
      auto itA = valueMap.find(op.getOperand(0));
      auto itB = valueMap.find(op.getOperand(1));
      if (itA == valueMap.end() || itB == valueMap.end()) {
        unsupportedReason = "cmpi operands not computed";
        return false;
      }
      unsigned w = getIntBitWidth(op.getOperand(0).getType());
      uint64_t a = itA->second, b = itB->second;
      int64_t sa = signExtend(a, w), sb = signExtend(b, w);
      bool result = false;
      switch (cmpOp.getPredicate()) {
      case mlir::arith::CmpIPredicate::eq:  result = (a == b); break;
      case mlir::arith::CmpIPredicate::ne:  result = (a != b); break;
      case mlir::arith::CmpIPredicate::slt: result = (sa < sb); break;
      case mlir::arith::CmpIPredicate::sle: result = (sa <= sb); break;
      case mlir::arith::CmpIPredicate::sgt: result = (sa > sb); break;
      case mlir::arith::CmpIPredicate::sge: result = (sa >= sb); break;
      case mlir::arith::CmpIPredicate::ult: result = (a < b); break;
      case mlir::arith::CmpIPredicate::ule: result = (a <= b); break;
      case mlir::arith::CmpIPredicate::ugt: result = (a > b); break;
      case mlir::arith::CmpIPredicate::uge: result = (a >= b); break;
      }
      valueMap[op.getResult(0)] = result ? 1 : 0;
      continue;
    }

    // arith.select
    if (mlir::isa<mlir::arith::SelectOp>(op)) {
      if (op.getNumOperands() < 3) {
        unsupportedReason = "select: wrong operand count";
        return false;
      }
      auto itCond = valueMap.find(op.getOperand(0));
      auto itTrue = valueMap.find(op.getOperand(1));
      auto itFalse = valueMap.find(op.getOperand(2));
      if (itCond == valueMap.end() || itTrue == valueMap.end() ||
          itFalse == valueMap.end()) {
        unsupportedReason = "select operands not computed";
        return false;
      }
      valueMap[op.getResult(0)] =
          (itCond->second & 1) ? itTrue->second : itFalse->second;
      continue;
    }

    // arith.extsi / arith.extui / arith.trunci
    if (mlir::isa<mlir::arith::ExtSIOp>(op)) {
      auto it = valueMap.find(op.getOperand(0));
      if (it == valueMap.end()) {
        unsupportedReason = "extsi input not computed";
        return false;
      }
      unsigned srcW = getIntBitWidth(op.getOperand(0).getType());
      unsigned dstW = getIntBitWidth(op.getResult(0).getType());
      valueMap[op.getResult(0)] =
          maskToWidth(static_cast<uint64_t>(signExtend(it->second, srcW)), dstW);
      continue;
    }
    if (mlir::isa<mlir::arith::ExtUIOp>(op)) {
      auto it = valueMap.find(op.getOperand(0));
      if (it == valueMap.end()) {
        unsupportedReason = "extui input not computed";
        return false;
      }
      valueMap[op.getResult(0)] = it->second;
      continue;
    }
    if (mlir::isa<mlir::arith::TruncIOp>(op)) {
      auto it = valueMap.find(op.getOperand(0));
      if (it == valueMap.end()) {
        unsupportedReason = "trunci input not computed";
        return false;
      }
      unsigned dstW = getIntBitWidth(op.getResult(0).getType());
      valueMap[op.getResult(0)] = maskToWidth(it->second, dstW);
      continue;
    }

    // handshake.cond_br: for reference execution, evaluate both paths
    // but only the taken path produces values.
    if (auto condBr =
            mlir::dyn_cast<circt::handshake::ConditionalBranchOp>(op)) {
      auto itCond = valueMap.find(condBr.getConditionOperand());
      auto itData = valueMap.find(condBr.getDataOperand());
      if (itCond == valueMap.end() || itData == valueMap.end()) {
        unsupportedReason = "cond_br operands not computed";
        return false;
      }
      // true result = result(0), false result = result(1)
      if (itCond->second & 1)
        valueMap[condBr.getTrueResult()] = itData->second;
      else
        valueMap[condBr.getFalseResult()] = itData->second;
      continue;
    }

    // handshake.mux: select input based on select operand.
    if (auto muxOp = mlir::dyn_cast<circt::handshake::MuxOp>(op)) {
      auto itSel = valueMap.find(muxOp.getSelectOperand());
      if (itSel == valueMap.end()) {
        unsupportedReason = "mux select not computed";
        return false;
      }
      unsigned selIdx = static_cast<unsigned>(itSel->second);
      auto dataOperands = muxOp.getDataOperands();
      if (selIdx >= dataOperands.size()) {
        unsupportedReason = "mux select index out of range";
        return false;
      }
      auto itData = valueMap.find(dataOperands[selIdx]);
      if (itData == valueMap.end()) {
        unsupportedReason = "mux selected input not computed";
        return false;
      }
      valueMap[muxOp.getResult()] = itData->second;
      continue;
    }

    // handshake.control_merge: forward first available input + index.
    if (mlir::isa<circt::handshake::ControlMergeOp>(op)) {
      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        auto it = valueMap.find(op.getOperand(i));
        if (it != valueMap.end()) {
          if (op.getNumResults() > 0)
            valueMap[op.getResult(0)] = it->second;
          if (op.getNumResults() > 1)
            valueMap[op.getResult(1)] = i; // index output
          break;
        }
      }
      continue;
    }

    // handshake.br (unconditional): pass through.
    if (mlir::isa<circt::handshake::BranchOp>(op)) {
      if (op.getNumOperands() > 0 && op.getNumResults() > 0) {
        auto it = valueMap.find(op.getOperand(0));
        if (it != valueMap.end())
          valueMap[op.getResult(0)] = it->second;
      }
      continue;
    }

    // handshake.buffer: pass through (no timing in reference).
    if (mlir::isa<circt::handshake::BufferOp>(op)) {
      if (op.getNumOperands() > 0 && op.getNumResults() > 0) {
        auto it = valueMap.find(op.getOperand(0));
        if (it != valueMap.end())
          valueMap[op.getResult(0)] = it->second;
      }
      continue;
    }

    // Skip results that are none-typed (control tokens).
    bool allNone = true;
    for (auto result : op.getResults()) {
      if (!mlir::isa<mlir::NoneType>(result.getType())) {
        allNone = false;
        break;
      }
    }
    if (allNone && op.getNumResults() > 0) {
      for (auto result : op.getResults())
        valueMap[result] = 0;
      continue;
    }

    // Unsupported operation.
    unsupportedReason = "unsupported op: " + opName.str();
    return false;
  }

  unsupportedReason = "no handshake.return found";
  return false;
}

} // anonymous namespace

CpuRefResult cpuReferenceExecute(
    mlir::ModuleOp dfgModule,
    const std::vector<std::vector<uint64_t>> &inputs) {

  CpuRefResult result;

  // Find the handshake.func.
  circt::handshake::FuncOp funcOp;
  dfgModule.walk([&](circt::handshake::FuncOp func) {
    llvm::StringRef name = func.getName();
    bool isEsi = name.ends_with("_esi");
    if (!funcOp || (!isEsi && funcOp.getName().ends_with("_esi")))
      funcOp = func;
  });

  if (!funcOp) {
    result.unsupportedReason = "no handshake.func found";
    return result;
  }

  // Count data arguments (non-none types) and data results.
  unsigned numDataArgs = 0;
  for (auto arg : funcOp.getBody().getArguments()) {
    if (!mlir::isa<mlir::NoneType>(arg.getType()))
      numDataArgs++;
  }

  // Determine how many result ports are data (non-none).
  auto resultTypes = funcOp.getResultTypes();
  unsigned numDataResults = 0;
  for (auto ty : resultTypes) {
    if (!mlir::isa<mlir::NoneType>(ty))
      numDataResults++;
  }

  if (inputs.size() < numDataArgs) {
    result.unsupportedReason = "not enough input ports provided";
    return result;
  }

  // Determine the number of tokens per port. Use the minimum length
  // across all input ports (they should all be the same).
  size_t numTokens = 0;
  if (!inputs.empty()) {
    numTokens = inputs[0].size();
    for (unsigned i = 1; i < numDataArgs; ++i) {
      if (i < inputs.size() && inputs[i].size() < numTokens)
        numTokens = inputs[i].size();
    }
  }

  // Initialize output vectors.
  result.outputs.resize(numDataResults);

  // Evaluate one token at a time.
  for (size_t t = 0; t < numTokens; ++t) {
    std::vector<uint64_t> tokenInputs(numDataArgs);
    for (unsigned i = 0; i < numDataArgs; ++i)
      tokenInputs[i] = (i < inputs.size() && t < inputs[i].size())
                            ? inputs[i][t]
                            : 0;

    std::vector<uint64_t> tokenOutputs;
    std::string reason;
    if (!evaluateOneToken(funcOp, tokenInputs, tokenOutputs, reason)) {
      result.unsupportedReason = reason;
      return result;
    }

    // Distribute outputs to per-port vectors.
    for (unsigned p = 0; p < numDataResults && p < tokenOutputs.size(); ++p)
      result.outputs[p].push_back(tokenOutputs[p]);
  }

  result.supported = true;
  return result;
}

} // namespace sim
} // namespace loom

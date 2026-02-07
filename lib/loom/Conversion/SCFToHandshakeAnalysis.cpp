//===-- SCFToHandshakeAnalysis.cpp - Analysis helpers ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// This file implements static analysis helpers and utility functions used by
// the SCF-to-Handshake conversion. It includes source code parsing for
// handshake naming, stream loop analysis, and region/path utilities.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/SCFToHandshakeImpl.h"
#include "loom/Dialect/Dataflow/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <optional>
#include <string>

namespace loom {
namespace detail {

// --- Internal-only helpers (static) ---

static std::string trimString(llvm::StringRef text) {
  return text.trim().str();
}

static bool isIdentChar(char ch) {
  return std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '_';
}

static bool isIdentBoundary(llvm::StringRef text, size_t pos, size_t len) {
  if (pos > 0 && isIdentChar(text[pos - 1]))
    return false;
  size_t end = pos + len;
  if (end < text.size() && isIdentChar(text[end]))
    return false;
  return true;
}

static mlir::Value stripCasts(mlir::Value value) {
  while (value) {
    if (auto cast = value.getDefiningOp<mlir::arith::ExtUIOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::arith::ExtSIOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::arith::TruncIOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
      value = cast.getIn();
      continue;
    }
    break;
  }
  return value;
}

static bool getConstantInt(mlir::Value value, int64_t &out) {
  if (!value)
    return false;
  if (auto cst = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(cst.getValue())) {
      out = intAttr.getInt();
      return true;
    }
  }
  if (auto cst = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    out = cst.value();
    return true;
  }
  return false;
}

static bool isIndexLike(mlir::Type type) {
  return type && (type.isIndex() || llvm::isa<mlir::IntegerType>(type));
}

static bool sameStepInfo(const StreamStepInfo &lhs,
                         const StreamStepInfo &rhs) {
  if (lhs.stepOp != rhs.stepOp)
    return false;
  if (lhs.isConst != rhs.isConst)
    return false;
  if (lhs.isConst)
    return lhs.constant == rhs.constant;
  if (!lhs.value || !rhs.value)
    return false;
  return stripCasts(lhs.value) == stripCasts(rhs.value);
}

static bool matchStreamUpdateValue(mlir::Value value,
                                   mlir::BlockArgument inductionArg,
                                   StreamStepInfo &step) {
  if (!value)
    return false;
  value = stripCasts(value);
  if (auto addi = value.getDefiningOp<mlir::arith::AddIOp>()) {
    mlir::Value lhs = stripCasts(addi.getLhs());
    mlir::Value rhs = stripCasts(addi.getRhs());
    mlir::Value other;
    if (lhs == inductionArg)
      other = rhs;
    else if (rhs == inductionArg)
      other = lhs;
    else
      other = {};
    if (other) {
      int64_t constant = 0;
      if (getConstantInt(other, constant)) {
        step.isConst = true;
        if (constant < 0) {
          step.constant = -constant;
          step.stepOp = "-=";
        } else {
          step.constant = constant;
          step.stepOp = "+=";
        }
      } else {
        step.isConst = false;
        step.value = other;
        step.stepOp = "+=";
      }
      return true;
    }
  }
  if (auto subi = value.getDefiningOp<mlir::arith::SubIOp>()) {
    mlir::Value lhs = stripCasts(subi.getLhs());
    mlir::Value rhs = stripCasts(subi.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (getConstantInt(rhs, constant)) {
        step.isConst = true;
        if (constant < 0) {
          step.constant = -constant;
          step.stepOp = "+=";
        } else {
          step.constant = constant;
          step.stepOp = "-=";
        }
      } else {
        step.isConst = false;
        step.value = rhs;
        step.stepOp = "-=";
      }
      return true;
    }
  }
  if (auto muli = value.getDefiningOp<mlir::arith::MulIOp>()) {
    mlir::Value lhs = stripCasts(muli.getLhs());
    mlir::Value rhs = stripCasts(muli.getRhs());
    mlir::Value other;
    if (lhs == inductionArg)
      other = rhs;
    else if (rhs == inductionArg)
      other = lhs;
    else
      other = {};
    if (other) {
      int64_t constant = 0;
      if (getConstantInt(other, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = other;
      }
      step.stepOp = "*=";
      return true;
    }
  }
  if (auto divsi = value.getDefiningOp<mlir::arith::DivSIOp>()) {
    mlir::Value lhs = stripCasts(divsi.getLhs());
    mlir::Value rhs = stripCasts(divsi.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (getConstantInt(rhs, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = rhs;
      }
      step.stepOp = "/=";
      return true;
    }
  }
  if (auto divui = value.getDefiningOp<mlir::arith::DivUIOp>()) {
    mlir::Value lhs = stripCasts(divui.getLhs());
    mlir::Value rhs = stripCasts(divui.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (getConstantInt(rhs, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = rhs;
      }
      step.stepOp = "/=";
      return true;
    }
  }
  if (auto shl = value.getDefiningOp<mlir::arith::ShLIOp>()) {
    mlir::Value lhs = stripCasts(shl.getLhs());
    mlir::Value rhs = stripCasts(shl.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (getConstantInt(rhs, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = rhs;
      }
      step.stepOp = "<<=";
      return true;
    }
  }
  if (auto shrsi = value.getDefiningOp<mlir::arith::ShRSIOp>()) {
    mlir::Value lhs = stripCasts(shrsi.getLhs());
    mlir::Value rhs = stripCasts(shrsi.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (getConstantInt(rhs, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = rhs;
      }
      step.stepOp = ">>=";
      return true;
    }
  }
  if (auto shrui = value.getDefiningOp<mlir::arith::ShRUIOp>()) {
    mlir::Value lhs = stripCasts(shrui.getLhs());
    mlir::Value rhs = stripCasts(shrui.getRhs());
    if (lhs == inductionArg) {
      int64_t constant = 0;
      if (getConstantInt(rhs, constant)) {
        step.isConst = true;
        step.constant = constant;
      } else {
        step.isConst = false;
        step.value = rhs;
      }
      step.stepOp = ">>=";
      return true;
    }
  }
  return false;
}

static bool isValidStepOp(llvm::StringRef op) {
  return op == "+=" || op == "-=" || op == "*=" || op == "/=" ||
         op == "<<=" || op == ">>=";
}

static bool isValidStopCond(llvm::StringRef cond) {
  return cond == "<" || cond == "<=" || cond == ">" || cond == ">=" ||
         cond == "!=";
}

static std::optional<mlir::FileLineColLoc>
findFileLoc(mlir::Location loc) {
  if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc))
    return fileLoc;
  if (auto fused = mlir::dyn_cast<mlir::FusedLoc>(loc)) {
    for (mlir::Location nested : fused.getLocations()) {
      if (auto found = findFileLoc(nested))
        return found;
    }
  }
  if (auto nameLoc = mlir::dyn_cast<mlir::NameLoc>(loc))
    return findFileLoc(nameLoc.getChildLoc());
  if (auto callLoc = mlir::dyn_cast<mlir::CallSiteLoc>(loc)) {
    if (auto callee = findFileLoc(callLoc.getCallee()))
      return callee;
    return findFileLoc(callLoc.getCaller());
  }
  return std::nullopt;
}

static bool isDefinedIn(mlir::Operation *root, mlir::Value value) {
  if (!value)
    return false;
  if (auto *def = value.getDefiningOp())
    return root->isAncestor(def);
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Operation *owner = blockArg.getOwner()->getParentOp();
    return owner && root->isAncestor(owner);
  }
  return false;
}

static bool hasStoreToMemref(mlir::Operation *root, mlir::Value memref) {
  bool found = false;
  root->walk([&](mlir::memref::StoreOp store) {
    if (store.getMemref() == memref)
      found = true;
  });
  return found;
}

static bool isSideEffectFreeOp(mlir::Operation &op) {
  if (op.getNumRegions() != 0)
    return false;
  if (auto memEffect = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(&op))
    return memEffect.hasNoEffect();
  return false;
}

static bool isPassThroughYield(mlir::Block &block) {
  auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(block.getTerminator());
  if (!yieldOp)
    return false;
  if (yieldOp.getNumOperands() != block.getNumArguments())
    return false;
  for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
    if (stripCasts(yieldOp.getOperand(i)) != block.getArgument(i))
      return false;
  }
  for (mlir::Operation &op : block) {
    if (llvm::isa<mlir::scf::YieldOp>(op))
      continue;
    if (!isSideEffectFreeOp(op))
      return false;
    for (mlir::Value result : op.getResults()) {
      for (mlir::OpOperand &use : result.getUses()) {
        if (use.getOwner() != yieldOp)
          return false;
      }
    }
  }
  return true;
}

static bool cmpKindFromPredicate(mlir::arith::CmpIPredicate pred,
                                 llvm::StringRef &cond) {
  switch (pred) {
  case mlir::arith::CmpIPredicate::slt:
  case mlir::arith::CmpIPredicate::ult:
    cond = "<";
    return true;
  case mlir::arith::CmpIPredicate::sle:
  case mlir::arith::CmpIPredicate::ule:
    cond = "<=";
    return true;
  case mlir::arith::CmpIPredicate::sgt:
  case mlir::arith::CmpIPredicate::ugt:
    cond = ">";
    return true;
  case mlir::arith::CmpIPredicate::sge:
  case mlir::arith::CmpIPredicate::uge:
    cond = ">=";
    return true;
  case mlir::arith::CmpIPredicate::ne:
    cond = "!=";
    return true;
  default:
    return false;
  }
}

static mlir::arith::CmpIPredicate
swapPredicate(mlir::arith::CmpIPredicate pred) {
  switch (pred) {
  case mlir::arith::CmpIPredicate::eq:
  case mlir::arith::CmpIPredicate::ne:
    return pred;
  case mlir::arith::CmpIPredicate::slt:
    return mlir::arith::CmpIPredicate::sgt;
  case mlir::arith::CmpIPredicate::sle:
    return mlir::arith::CmpIPredicate::sge;
  case mlir::arith::CmpIPredicate::sgt:
    return mlir::arith::CmpIPredicate::slt;
  case mlir::arith::CmpIPredicate::sge:
    return mlir::arith::CmpIPredicate::sle;
  case mlir::arith::CmpIPredicate::ult:
    return mlir::arith::CmpIPredicate::ugt;
  case mlir::arith::CmpIPredicate::ule:
    return mlir::arith::CmpIPredicate::uge;
  case mlir::arith::CmpIPredicate::ugt:
    return mlir::arith::CmpIPredicate::ult;
  case mlir::arith::CmpIPredicate::uge:
    return mlir::arith::CmpIPredicate::ule;
  }
  return pred;
}

static llvm::SmallVector<std::string, 8>
splitParameters(llvm::StringRef params) {
  llvm::SmallVector<std::string, 8> result;
  std::string current;
  int depth = 0;
  for (char ch : params) {
    if (ch == '(' || ch == '<' || ch == '[' || ch == '{')
      ++depth;
    else if (ch == ')' || ch == '>' || ch == ']' || ch == '}')
      --depth;
    if (ch == ',' && depth == 0) {
      result.push_back(trimString(current));
      current.clear();
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty())
    result.push_back(trimString(current));
  return result;
}

static std::optional<std::string> extractLastIdentifier(llvm::StringRef text) {
  size_t end = text.size();
  while (end > 0 &&
         std::isspace(static_cast<unsigned char>(text[end - 1])))
    --end;
  if (end == 0)
    return std::nullopt;
  size_t start = end;
  while (start > 0 && isIdentChar(text[start - 1]))
    --start;
  if (start == end)
    return std::nullopt;
  return text.substr(start, end - start).str();
}

// --- Promoted functions (visible to SCFToHandshakeConvert.cpp) ---

void copyLoomAnnotations(mlir::Operation *src, mlir::Operation *dst) {
  if (!src || !dst)
    return;
  auto attr = src->getAttrOfType<mlir::ArrayAttr>("loom.annotations");
  if (!attr)
    return;
  dst->setAttr("loom.annotations", attr);
}

std::string demangleBaseName(llvm::StringRef name) {
  std::string demangled = llvm::demangle(name.str());
  std::string base = demangled.empty() ? name.str() : demangled;
  size_t paren = base.find('(');
  if (paren != std::string::npos)
    base = base.substr(0, paren);
  base = trimString(base);
  size_t space = base.find_last_of(" \t");
  if (space != std::string::npos)
    base = base.substr(space + 1);
  return base;
}

std::optional<std::string> resolveSourcePath(mlir::Location loc) {
  auto fileLoc = findFileLoc(loc);
  if (!fileLoc)
    return std::nullopt;
  std::string path = fileLoc->getFilename().str();
  if (path.empty())
    return std::nullopt;
  if (llvm::sys::fs::exists(path))
    return path;
  if (!llvm::sys::path::is_absolute(path)) {
    llvm::SmallString<256> cwd;
    if (!llvm::sys::fs::current_path(cwd)) {
      llvm::SmallString<256> candidate = cwd;
      llvm::sys::path::append(candidate, path);
      if (llvm::sys::fs::exists(candidate))
        return candidate.str().str();
    }
  }
  return std::nullopt;
}

mlir::Value castToIndex(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value) {
  if (!value)
    return {};
  if (value.getType().isIndex())
    return value;
  if (llvm::isa<mlir::IntegerType>(value.getType()))
    return mlir::arith::IndexCastOp::create(builder, loc,
                                                    builder.getIndexType(),
                                                    value);
  return {};
}

mlir::Value castIndexToType(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value value, mlir::Type targetType) {
  if (!value)
    return {};
  if (value.getType() == targetType)
    return value;
  if (value.getType().isIndex() &&
      llvm::isa<mlir::IntegerType>(targetType))
    return mlir::arith::IndexCastOp::create(builder, loc, targetType, value);
  return {};
}

std::optional<StreamWhileAttr>
getStreamWhileAttr(mlir::scf::WhileOp op) {
  auto dict = op->getAttrOfType<mlir::DictionaryAttr>("loom.stream");
  if (!dict)
    return std::nullopt;
  auto ivAttr = dict.getAs<mlir::IntegerAttr>("iv");
  auto stepAttr = dict.getAs<mlir::StringAttr>("step_op");
  auto stopAttr = dict.getAs<mlir::StringAttr>("stop_cond");
  if (!ivAttr || !stepAttr || !stopAttr)
    return std::nullopt;
  auto cmpAttr = dict.getAs<mlir::BoolAttr>("cmp_on_update");
  StreamWhileAttr info;
  info.ivIndex = ivAttr.getInt();
  info.stepOp = stepAttr.getValue();
  info.stopCond = stopAttr.getValue();
  info.cmpOnUpdate = cmpAttr ? cmpAttr.getValue() : false;
  return info;
}

mlir::LogicalResult
analyzeStreamableWhile(mlir::scf::WhileOp op, const StreamWhileAttr &attr,
                       StreamWhileOperands &operands) {
  if (attr.ivIndex < 0 ||
      attr.ivIndex >= static_cast<int64_t>(op.getNumOperands()))
    return op.emitError("invalid loom.stream iv index");
  if (!isValidStepOp(attr.stepOp))
    return op.emitError("invalid loom.stream step_op");
  if (!isValidStopCond(attr.stopCond))
    return op.emitError("invalid loom.stream stop_cond");

  if (!op.getBefore().hasOneBlock() || !op.getAfter().hasOneBlock())
    return op.emitError("expected single-block scf.while regions");

  mlir::Block &before = op.getBefore().front();
  mlir::Block &after = op.getAfter().front();

  auto conditionOp = llvm::dyn_cast<mlir::scf::ConditionOp>(
      before.getTerminator());
  if (!conditionOp)
    return op.emitError("expected scf.condition terminator");
  if (conditionOp.getArgs().size() != before.getNumArguments())
    return op.emitError("expected scf.condition arg count to match loop args");

  bool conditionArgsPassThrough = true;
  for (unsigned i = 0; i < before.getNumArguments(); ++i) {
    if (stripCasts(conditionOp.getArgs()[i]) != before.getArgument(i)) {
      conditionArgsPassThrough = false;
      break;
    }
  }

  bool beforeSideEffectFree = true;
  for (mlir::Operation &nested : before) {
    if (llvm::isa<mlir::scf::ConditionOp>(nested))
      continue;
    if (auto load = llvm::dyn_cast<mlir::memref::LoadOp>(&nested)) {
      if (!hasStoreToMemref(op.getOperation(), load.getMemref()))
        continue;
    }
    if (!isSideEffectFreeOp(nested)) {
      beforeSideEffectFree = false;
      break;
    }
  }

  bool afterPassThrough = isPassThroughYield(after);
  bool bodyInBefore = false;
  if (conditionArgsPassThrough && beforeSideEffectFree) {
    bodyInBefore = false;
  } else if (afterPassThrough) {
    bodyInBefore = true;
  } else {
    return op.emitError("cannot determine streamable while body location");
  }

  mlir::Value condValue = stripCasts(conditionOp.getCondition());
  auto cmpOp = condValue.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmpOp)
    return op.emitError("expected arith.cmpi loop condition");

  mlir::Value lhs = stripCasts(cmpOp.getLhs());
  mlir::Value rhs = stripCasts(cmpOp.getRhs());
  int64_t ivIndex = attr.ivIndex;
  mlir::BlockArgument ivArg = before.getArgument(ivIndex);
  bool ivOnLhs = false;
  bool ivOnRhs = false;
  bool cmpUsesUpdate = false;
  StreamStepInfo cmpStep;

  if (lhs == ivArg) {
    ivOnLhs = true;
  } else if (rhs == ivArg) {
    ivOnRhs = true;
  } else {
    StreamStepInfo candidate;
    if (matchStreamUpdateValue(lhs, ivArg, candidate)) {
      ivOnLhs = true;
      cmpUsesUpdate = true;
      cmpStep = candidate;
    } else if (matchStreamUpdateValue(rhs, ivArg, candidate)) {
      ivOnRhs = true;
      cmpUsesUpdate = true;
      cmpStep = candidate;
    }
  }
  if (ivOnLhs == ivOnRhs)
    return op.emitError("expected loop compare to use induction argument");

  auto pred = cmpOp.getPredicate();
  if (ivOnRhs) {
    pred = swapPredicate(pred);
    std::swap(lhs, rhs);
  }
  llvm::StringRef stopCond;
  if (!cmpKindFromPredicate(pred, stopCond))
    return op.emitError("unsupported loop comparison predicate");
  if (stopCond != attr.stopCond)
    return op.emitError("loom.stream stop_cond mismatch");

  mlir::Value boundValue = rhs;
  if (isDefinedIn(op, boundValue))
    return op.emitError("loop bound must be loop-invariant");

  auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(after.getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() != op.getNumOperands())
    return op.emitError("expected scf.yield to match loop operands");

  StreamStepInfo stepInfo;
  mlir::Value updateValue = bodyInBefore
                                ? stripCasts(conditionOp.getArgs()[ivIndex])
                                : stripCasts(yieldOp.getOperand(ivIndex));
  mlir::BlockArgument updateBase = bodyInBefore ? before.getArgument(ivIndex)
                                                : after.getArgument(ivIndex);
  if (!matchStreamUpdateValue(updateValue, updateBase, stepInfo))
    return op.emitError("cannot match induction update");
  if (cmpUsesUpdate && !sameStepInfo(cmpStep, stepInfo))
    return op.emitError("mismatched stream step in comparison");
  if (stepInfo.stepOp != attr.stepOp)
    return op.emitError("loom.stream step_op mismatch");
  if (stepInfo.isConst && stepInfo.constant == 0)
    return op.emitError("stream step must be nonzero");
  if (!stepInfo.isConst && isDefinedIn(op, stepInfo.value))
    return op.emitError("loop step must be loop-invariant");

  if (!op.getResult(ivIndex).use_empty())
    return op.emitError("induction result must be unused for stream conversion");

  operands.init = op.getOperands()[ivIndex];
  operands.stepIsConst = stepInfo.isConst;
  operands.stepConst = stepInfo.constant;
  operands.step = stepInfo.isConst ? nullptr : stepInfo.value;
  operands.bound = boundValue;
  operands.bodyInBefore = bodyInBefore;
  return mlir::success();
}

bool readFile(llvm::StringRef path, std::string &out) {
  std::ifstream file(path.str());
  if (!file)
    return false;
  out.assign((std::istreambuf_iterator<char>(file)),
             std::istreambuf_iterator<char>());
  return true;
}

bool extractFunctionSource(llvm::StringRef content,
                           llvm::StringRef funcName,
                           std::string &params,
                           std::string &body) {
  size_t pos = 0;
  while (true) {
    pos = content.find(funcName, pos);
    if (pos == std::string::npos)
      return false;
    if (!isIdentBoundary(content, pos, funcName.size())) {
      pos += funcName.size();
      continue;
    }
    size_t cursor = pos + funcName.size();
    while (cursor < content.size() &&
           std::isspace(static_cast<unsigned char>(content[cursor])))
      ++cursor;
    if (cursor >= content.size() || content[cursor] != '(') {
      pos = cursor;
      continue;
    }
    size_t paramsStart = cursor + 1;
    int depth = 1;
    ++cursor;
    while (cursor < content.size() && depth > 0) {
      char ch = content[cursor];
      if (ch == '(')
        ++depth;
      else if (ch == ')')
        --depth;
      ++cursor;
    }
    if (depth != 0)
      return false;
    size_t paramsEnd = cursor - 1;
    params = std::string(content.substr(paramsStart, paramsEnd - paramsStart));

    while (cursor < content.size() &&
           std::isspace(static_cast<unsigned char>(content[cursor])))
      ++cursor;
    if (cursor >= content.size() || content[cursor] != '{') {
      pos = cursor;
      continue;
    }
    size_t bodyStart = cursor + 1;
    int braceDepth = 1;
    ++cursor;
    while (cursor < content.size() && braceDepth > 0) {
      char ch = content[cursor];
      if (ch == '{')
        ++braceDepth;
      else if (ch == '}')
        --braceDepth;
      ++cursor;
    }
    if (braceDepth != 0)
      return false;
    size_t bodyEnd = cursor - 1;
    body = std::string(content.substr(bodyStart, bodyEnd - bodyStart));
    return true;
  }
}

llvm::SmallVector<std::string, 8>
extractParamNames(llvm::StringRef params) {
  llvm::SmallVector<std::string, 8> names;
  auto pieces = splitParameters(params);
  for (llvm::StringRef piece : pieces) {
    llvm::StringRef trimmed = piece.trim();
    if (trimmed.empty())
      continue;
    if (trimmed == "void")
      continue;
    size_t eq = trimmed.find('=');
    if (eq != std::string::npos)
      trimmed = trimmed.substr(0, eq).trim();
    auto name = extractLastIdentifier(trimmed);
    if (name && !name->empty())
      names.push_back(*name);
  }
  return names;
}

std::optional<std::string> extractReturnName(llvm::StringRef body) {
  std::optional<std::string> found;
  size_t pos = 0;
  while (pos < body.size()) {
    size_t hit = body.find("return", pos);
    if (hit == std::string::npos)
      break;
    if (!isIdentBoundary(body, hit, 6)) {
      pos = hit + 6;
      continue;
    }
    size_t cursor = hit + 6;
    while (cursor < body.size() &&
           std::isspace(static_cast<unsigned char>(body[cursor])))
      ++cursor;
    if (cursor >= body.size()) {
      pos = cursor;
      continue;
    }
    if (body[cursor] == ';') {
      pos = cursor + 1;
      continue;
    }
    size_t start = cursor;
    if (!isIdentChar(body[start])) {
      return std::nullopt;
    }
    while (cursor < body.size() && isIdentChar(body[cursor]))
      ++cursor;
    llvm::StringRef name = body.substr(start, cursor - start);
    while (cursor < body.size() &&
           std::isspace(static_cast<unsigned char>(body[cursor])))
      ++cursor;
    if (cursor >= body.size() || body[cursor] != ';')
      return std::nullopt;
    if (!found)
      found = name.str();
    else if (*found != name)
      return std::nullopt;
    pos = cursor + 1;
  }
  return found;
}

bool isLocalToRegion(mlir::Value value, mlir::Region *region) {
  if (!value || !region)
    return false;
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value))
    return arg.getOwner()->getParent() == region;
  if (auto *def = value.getDefiningOp())
    return def->getParentRegion() == region;
  return false;
}

ScfPath computeScfPath(mlir::Operation *op) {
  ScfPath path;
  mlir::Region *region = op->getParentRegion();
  mlir::Operation *parent = op->getParentOp();
  while (parent) {
    if (mlir::isa<mlir::scf::ForOp, mlir::scf::WhileOp, mlir::scf::IfOp,
                  mlir::scf::IndexSwitchOp>(parent)) {
      unsigned index = 0;
      for (mlir::Region &candidate : parent->getRegions()) {
        if (&candidate == region)
          break;
        ++index;
      }
      path.push_back(PathEntry{parent, index});
    }
    region = parent->getParentRegion();
    parent = parent->getParentOp();
  }
  std::reverse(path.begin(), path.end());
  return path;
}

} // namespace detail
} // namespace loom

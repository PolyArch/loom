//===- SCFToHandshakeLowering.cpp - SCF->Handshake lowering -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Conversion/HandshakeOptimize.h"
#include "loom/Conversion/SCFToHandshakeImpl.h"
#include "loom/Dialect/Dataflow/IR/DataflowOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
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

using loom::dataflow::CarryOp;
using loom::dataflow::GateOp;
using loom::dataflow::InvariantOp;
using loom::dataflow::StreamOp;

struct HandshakeLowering::RegionState {
  mlir::Region *region = nullptr;
  RegionState *parent = nullptr;
  llvm::DenseMap<mlir::Value, mlir::Value> valueMap;
  mlir::Value invariantCond;
  bool pendingCond = false;
  llvm::SmallVector<InvariantOp, 4> pendingInvariants;
  mlir::Value controlToken;
  bool controlPending = false;
  InvariantOp controlInvariant;
};

static void copyLoomAnnotations(mlir::Operation *src, mlir::Operation *dst) {
  if (!src || !dst)
    return;
  auto attr = src->getAttrOfType<mlir::ArrayAttr>("loom.annotations");
  if (!attr)
    return;
  dst->setAttr("loom.annotations", attr);
}

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

static std::string demangleBaseName(llvm::StringRef name) {
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

static std::optional<std::string> resolveSourcePath(mlir::Location loc) {
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

static bool readFile(llvm::StringRef path, std::string &out) {
  std::ifstream file(path.str());
  if (!file)
    return false;
  out.assign((std::istreambuf_iterator<char>(file)),
             std::istreambuf_iterator<char>());
  return true;
}

static bool extractFunctionSource(llvm::StringRef content,
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

static llvm::SmallVector<std::string, 8>
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

static std::optional<std::string> extractReturnName(llvm::StringRef body) {
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

static bool isLocalToRegion(mlir::Value value, mlir::Region *region) {
  if (!value || !region)
    return false;
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value))
    return arg.getOwner()->getParent() == region;
  if (auto *def = value.getDefiningOp())
    return def->getParentRegion() == region;
  return false;
}

static ScfPath computeScfPath(mlir::Operation *op) {
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

HandshakeLowering::HandshakeLowering(mlir::func::FuncOp func,
                                     mlir::AliasAnalysis &aa)
    : func(func), aliasAnalysis(aa), builder(func.getContext()),
      returnLoc(func.getLoc()) {}

mlir::Value HandshakeLowering::makeConstant(mlir::Location loc,
                                            mlir::Attribute value,
                                            mlir::Type type,
                                            mlir::Value ctrlToken) {
  mlir::Value ctrl = ctrlToken ? ctrlToken : getEntryToken(loc);
  mlir::OperationState constState(
      loc, circt::handshake::ConstantOp::getOperationName());
  constState.addOperands(ctrl);
  constState.addTypes(type);
  constState.addAttribute("value", value);
  mlir::Operation *created = builder.create(constState);
  return created->getResult(0);
}

mlir::Value HandshakeLowering::makeBool(mlir::Location loc, bool value) {
  return makeConstant(loc, builder.getBoolAttr(value), builder.getI1Type(),
                      getEntryToken(loc));
}

mlir::Value HandshakeLowering::makeDummyData(mlir::Location loc,
                                             mlir::Type type) {
  return builder.create<circt::handshake::SourceOp>(loc, type).getResult();
}

mlir::Value HandshakeLowering::getEntryToken(mlir::Location loc) {
  if (!entryToken)
    entryToken =
        builder.create<circt::handshake::SourceOp>(loc, builder.getNoneType())
            .getResult();
  return entryToken;
}

void HandshakeLowering::assignHandshakeNames() {
  unsigned argCount = func.getFunctionType().getNumInputs();
  unsigned resCount = func.getFunctionType().getNumResults();

  llvm::SmallVector<std::string, 8> argNames;
  llvm::SmallVector<std::string, 4> resNames;
  bool parsed = false;

  auto path = resolveSourcePath(func.getLoc());
  if (path) {
    std::string content;
    if (readFile(*path, content)) {
      std::string params;
      std::string body;
      std::string funcName = demangleBaseName(func.getName());
      if (extractFunctionSource(content, funcName, params, body)) {
        argNames = extractParamNames(params);
        if (resCount == 1) {
          if (auto retName = extractReturnName(body))
            resNames.push_back(*retName);
        }
        parsed = true;
      }
    }
  }

  if (!parsed || argNames.size() != argCount) {
    argNames.clear();
    for (unsigned i = 0; i < argCount; ++i)
      argNames.push_back("in" + std::to_string(i));
  }

  if (resCount == 1 && resNames.empty())
    resNames.push_back("out0");
  if (resCount > 1) {
    resNames.clear();
    for (unsigned i = 0; i < resCount; ++i)
      resNames.push_back("out" + std::to_string(i));
  }

  llvm::StringSet<> usedArgs;
  auto makeUnique = [](llvm::StringSet<> &used, llvm::StringRef base) {
    std::string name = base.str();
    unsigned suffix = 0;
    while (used.contains(name)) {
      name = base.str() + "_" + std::to_string(suffix++);
    }
    used.insert(name);
    return name;
  };

  llvm::SmallVector<mlir::Attribute, 8> argAttrs;
  argAttrs.reserve(argNames.size() + 1);
  for (const std::string &name : argNames) {
    std::string unique = makeUnique(usedArgs, name);
    argAttrs.push_back(builder.getStringAttr(unique));
  }
  std::string startName = makeUnique(usedArgs, "start_token");
  argAttrs.push_back(builder.getStringAttr(startName));

  llvm::StringSet<> usedRes;
  llvm::SmallVector<mlir::Attribute, 4> resAttrs;
  resAttrs.reserve(resNames.size() + 1);
  for (const std::string &name : resNames) {
    std::string unique = makeUnique(usedRes, name);
    resAttrs.push_back(builder.getStringAttr(unique));
  }
  std::string doneName = makeUnique(usedRes, "done_token");
  resAttrs.push_back(builder.getStringAttr(doneName));

  handshakeFunc->setAttr("argNames", builder.getArrayAttr(argAttrs));
  handshakeFunc->setAttr("resNames", builder.getArrayAttr(resAttrs));
}

mlir::Value HandshakeLowering::mapValue(mlir::Value value, RegionState &state,
                                        mlir::Location loc) {
  if (!value)
    return value;
  auto it = state.valueMap.find(value);
  if (it != state.valueMap.end())
    return it->second;

  if (state.parent) {
    mlir::Value outer = mapValue(value, *state.parent, loc);
    if (!state.invariantCond || !isLocalToRegion(value, state.parent->region))
      return outer;
    if (mlir::isa<mlir::BaseMemRefType>(outer.getType()))
      return outer;

    auto inv = builder.create<InvariantOp>(loc, outer.getType(),
                                           state.invariantCond, outer);
    if (state.pendingCond) {
      state.pendingInvariants.push_back(inv);
    } else {
      state.pendingCond = true;
    }
    state.valueMap[value] = inv.getO();
    return inv.getO();
  }

  return value;
}

void HandshakeLowering::updateInvariantCond(RegionState &state,
                                            mlir::Value cond) {
  if (!state.pendingCond)
    return;
  for (InvariantOp inv : state.pendingInvariants)
    inv->setOperand(0, cond);
  state.pendingInvariants.clear();
  state.pendingCond = false;
}

mlir::LogicalResult HandshakeLowering::lowerReturn(mlir::func::ReturnOp op,
                                                   RegionState &state) {
  if (sawReturn)
    return op.emitError("multiple func.return in accel function");
  sawReturn = true;
  for (mlir::Value operand : op.getOperands())
    pendingReturnValues.push_back(mapValue(operand, state, op.getLoc()));
  returnLoc = op.getLoc();
  return mlir::success();
}

mlir::LogicalResult HandshakeLowering::lowerLoad(mlir::memref::LoadOp op,
                                                 RegionState &state) {
  mlir::Location loc = op.getLoc();
  llvm::SmallVector<mlir::Value, 4> addrOperands;
  addrOperands.reserve(op.getIndices().size());
  for (mlir::Value index : op.getIndices())
    addrOperands.push_back(mapValue(index, state, loc));

  auto emitLoad = [&](mlir::Value origMemref, mlir::Value mappedMemref,
                      mlir::Value ctrlToken) {
    mlir::Value rootMemref = getMemrefRoot(mappedMemref);
    mlir::Value dummyCtrl = makeDummyData(loc, builder.getNoneType());

    llvm::SmallVector<mlir::Value, 6> operands(addrOperands.begin(),
                                               addrOperands.end());
    operands.push_back(dummyCtrl);
    operands.push_back(dummyCtrl);

    llvm::SmallVector<mlir::Type, 4> resultTypes;
    resultTypes.push_back(op.getType());
    for (mlir::Value addr : addrOperands)
      resultTypes.push_back(addr.getType());

    mlir::OperationState loadState(
        loc, circt::handshake::LoadOp::getOperationName());
    loadState.addOperands(operands);
    loadState.addTypes(resultTypes);
    auto load = mlir::cast<circt::handshake::LoadOp>(builder.create(loadState));
    copyLoomAnnotations(op, load);

    MemAccess access;
    access.origOp = op;
    access.origMemref = origMemref;
    access.memref = rootMemref;
    access.kind = AccessKind::Load;
    access.order = orderCounter++;
    access.path = computeScfPath(op);
    access.loadOp = load;
    access.controlToken = ctrlToken ? ctrlToken : getEntryToken(loc);
    memAccesses.push_back(access);

    return load.getResult(0);
  };

  if (auto selectOp = op.getMemref().getDefiningOp<mlir::arith::SelectOp>()) {
    if (mlir::isa<mlir::BaseMemRefType>(selectOp.getTrueValue().getType()) &&
        mlir::isa<mlir::BaseMemRefType>(selectOp.getFalseValue().getType())) {
      mlir::Value cond = mapValue(selectOp.getCondition(), state, loc);
      mlir::Value baseCtrl = state.controlToken ? state.controlToken
                                                : getEntryToken(loc);
      auto branch = builder.create<circt::handshake::ConditionalBranchOp>(
          loc, cond, baseCtrl);
      mlir::Value trueData =
          emitLoad(selectOp.getTrueValue(),
                   mapValue(selectOp.getTrueValue(), state, loc),
                   branch.getTrueResult());
      mlir::Value falseData =
          emitLoad(selectOp.getFalseValue(),
                   mapValue(selectOp.getFalseValue(), state, loc),
                   branch.getFalseResult());
      mlir::Value zero = makeConstant(
          loc, builder.getIndexAttr(0), builder.getIndexType(), baseCtrl);
      mlir::Value one = makeConstant(
          loc, builder.getIndexAttr(1), builder.getIndexType(), baseCtrl);
      mlir::Value select =
          builder.create<mlir::arith::SelectOp>(loc, cond, one, zero);
      auto mux = builder.create<circt::handshake::MuxOp>(
          loc, select, mlir::ValueRange{falseData, trueData});
      state.valueMap[op.getResult()] = mux.getResult();
      return mlir::success();
    }
  }

  mlir::Value mappedMemref = mapValue(op.getMemref(), state, loc);
  mlir::Value data =
      emitLoad(op.getMemref(), mappedMemref,
               state.controlToken ? state.controlToken : getEntryToken(loc));
  state.valueMap[op.getResult()] = data;
  return mlir::success();
}

mlir::LogicalResult HandshakeLowering::lowerStore(mlir::memref::StoreOp op,
                                                  RegionState &state) {
  mlir::Location loc = op.getLoc();
  llvm::SmallVector<mlir::Value, 4> addrOperands;
  addrOperands.reserve(op.getIndices().size());
  for (mlir::Value index : op.getIndices())
    addrOperands.push_back(mapValue(index, state, loc));

  mlir::Value dataValue = mapValue(op.getValue(), state, loc);

  auto emitStore = [&](mlir::Value origMemref, mlir::Value mappedMemref,
                       mlir::Value ctrlToken) {
    mlir::Value rootMemref = getMemrefRoot(mappedMemref);
    mlir::Value dummyCtrl = getEntryToken(loc);

    llvm::SmallVector<mlir::Value, 6> operands(addrOperands.begin(),
                                               addrOperands.end());
    operands.push_back(dataValue);
    operands.push_back(dummyCtrl);

    llvm::SmallVector<mlir::Type, 4> resultTypes;
    resultTypes.push_back(dataValue.getType());
    for (mlir::Value addr : addrOperands)
      resultTypes.push_back(addr.getType());

    mlir::OperationState storeState(
        loc, circt::handshake::StoreOp::getOperationName());
    storeState.addOperands(operands);
    storeState.addTypes(resultTypes);
    auto store =
        mlir::cast<circt::handshake::StoreOp>(builder.create(storeState));
    copyLoomAnnotations(op, store);

    MemAccess access;
    access.origOp = op;
    access.origMemref = origMemref;
    access.memref = rootMemref;
    access.kind = AccessKind::Store;
    access.order = orderCounter++;
    access.path = computeScfPath(op);
    access.storeOp = store;
    access.controlToken = ctrlToken ? ctrlToken : getEntryToken(loc);
    memAccesses.push_back(access);
  };

  if (auto selectOp = op.getMemref().getDefiningOp<mlir::arith::SelectOp>()) {
    if (mlir::isa<mlir::BaseMemRefType>(selectOp.getTrueValue().getType()) &&
        mlir::isa<mlir::BaseMemRefType>(selectOp.getFalseValue().getType())) {
      mlir::Value cond = mapValue(selectOp.getCondition(), state, loc);
      mlir::Value baseCtrl = state.controlToken ? state.controlToken
                                                : getEntryToken(loc);
      auto branch = builder.create<circt::handshake::ConditionalBranchOp>(
          loc, cond, baseCtrl);
      emitStore(selectOp.getTrueValue(),
                mapValue(selectOp.getTrueValue(), state, loc),
                branch.getTrueResult());
      emitStore(selectOp.getFalseValue(),
                mapValue(selectOp.getFalseValue(), state, loc),
                branch.getFalseResult());
      return mlir::success();
    }
  }

  mlir::Value mappedMemref = mapValue(op.getMemref(), state, loc);
  emitStore(op.getMemref(), mappedMemref,
            state.controlToken ? state.controlToken : getEntryToken(loc));
  return mlir::success();
}

mlir::LogicalResult HandshakeLowering::lowerFor(mlir::scf::ForOp op,
                                                RegionState &state) {
  mlir::Location loc = op.getLoc();
  mlir::Value lower = mapValue(op.getLowerBound(), state, loc);
  mlir::Value upper = mapValue(op.getUpperBound(), state, loc);
  mlir::Value step = mapValue(op.getStep(), state, loc);

  auto stream = builder.create<StreamOp>(loc, lower, step, upper);
  copyLoomAnnotations(op, stream);
  mlir::Value rawIndex = stream.getIndex();
  mlir::Value rawCond = stream.getWillContinue();
  forConds[op] = rawCond;

  auto gate = builder.create<GateOp>(
      loc, mlir::TypeRange{rawIndex.getType(), builder.getI1Type()}, rawIndex,
      rawCond);
  mlir::Value bodyIndex = gate.getIndex();
  mlir::Value bodyCond = gate.getCond();

  llvm::SmallVector<CarryOp, 4> carries;
  llvm::SmallVector<mlir::Value, 4> bodyIterValues;
  llvm::SmallVector<mlir::Value, 4> loopResults;

  auto iterOperands = op.getInitArgs();
  for (mlir::Value init : iterOperands) {
    mlir::Value initValue = mapValue(init, state, loc);
    auto carry = builder.create<CarryOp>(loc, initValue.getType(), rawCond,
                                         initValue, initValue);
    carries.push_back(carry);
    auto iterGate = builder.create<GateOp>(
        loc, mlir::TypeRange{carry.getO().getType(), builder.getI1Type()},
        carry.getO(), rawCond);
    bodyIterValues.push_back(iterGate.getIndex());
    auto branch = builder.create<circt::handshake::ConditionalBranchOp>(
        loc, rawCond, carry.getO());
    loopResults.push_back(branch.getFalseResult());
  }

  mlir::Block *bodyBlock = op.getBody();
  mlir::Region *bodyRegion = bodyBlock->getParent();
  if (!bodyRegion || !bodyRegion->hasOneBlock())
    return op.emitError("expected single-block scf.for body");

  RegionState bodyState;
  bodyState.region = bodyRegion;
  bodyState.parent = &state;
  bodyState.invariantCond = bodyCond;
  bodyState.pendingCond = false;
  mlir::Value parentCtrl =
      state.controlToken ? state.controlToken : getEntryToken(loc);
  bodyState.controlToken =
      builder
          .create<InvariantOp>(loc, parentCtrl.getType(), bodyCond, parentCtrl)
          .getO();

  bodyState.valueMap[bodyBlock->getArgument(0)] = bodyIndex;
  for (unsigned i = 0, e = bodyIterValues.size(); i < e; ++i)
    bodyState.valueMap[bodyBlock->getArgument(i + 1)] = bodyIterValues[i];

  llvm::SmallVector<mlir::Value, 4> yieldValues;
  for (mlir::Operation &nested : *bodyBlock) {
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
      for (mlir::Value operand : yield.getOperands())
        yieldValues.push_back(mapValue(operand, bodyState, yield.getLoc()));
      break;
    }
    if (mlir::failed(lowerOp(&nested, bodyState)))
      return mlir::failure();
  }

  if (yieldValues.size() != carries.size())
    return op.emitError("scf.for yield arity mismatch");

  for (unsigned i = 0, e = carries.size(); i < e; ++i)
    carries[i]->setOperand(2, yieldValues[i]);

  for (unsigned i = 0, e = loopResults.size(); i < e; ++i)
    state.valueMap[op.getResult(i)] = loopResults[i];

  updateInvariantCond(bodyState, bodyCond);
  return mlir::success();
}

mlir::LogicalResult HandshakeLowering::lowerWhile(mlir::scf::WhileOp op,
                                                  RegionState &state) {
  mlir::Location loc = op.getLoc();

  llvm::SmallVector<mlir::Value, 4> initValues;
  initValues.reserve(op.getNumOperands());
  for (mlir::Value operand : op.getOperands())
    initValues.push_back(mapValue(operand, state, loc));

  llvm::SmallVector<CarryOp, 4> carries;
  for (mlir::Value initValue : initValues) {
    mlir::Value placeholderCond = makeBool(loc, true);
    auto carry = builder.create<CarryOp>(loc, initValue.getType(),
                                         placeholderCond, initValue, initValue);
    carries.push_back(carry);
  }

  mlir::Block &beforeBlock = op.getBefore().front();
  if (beforeBlock.getNumArguments() != carries.size())
    return op.emitError("scf.while before arity mismatch");

  RegionState beforeState;
  beforeState.region = &op.getBefore();
  beforeState.parent = &state;
  beforeState.pendingCond = true;
  mlir::Value parentCtrl =
      state.controlToken ? state.controlToken : getEntryToken(loc);
  mlir::Value placeholderCond = makeBool(loc, true);
  beforeState.controlInvariant = builder.create<InvariantOp>(
      loc, parentCtrl.getType(), placeholderCond, parentCtrl);
  beforeState.controlToken = beforeState.controlInvariant.getO();

  for (unsigned i = 0, e = carries.size(); i < e; ++i)
    beforeState.valueMap[beforeBlock.getArgument(i)] = carries[i].getO();

  mlir::Value condValue;
  llvm::SmallVector<mlir::Value, 4> condArgs;
  for (mlir::Operation &nested : beforeBlock) {
    if (auto condition = mlir::dyn_cast<mlir::scf::ConditionOp>(nested)) {
      condValue = mapValue(condition.getCondition(), beforeState,
                           condition.getLoc());
      for (mlir::Value operand : condition.getArgs())
        condArgs.push_back(mapValue(operand, beforeState, condition.getLoc()));
      break;
    }
    if (mlir::failed(lowerOp(&nested, beforeState)))
      return mlir::failure();
  }

  if (!condValue)
    return op.emitError("scf.while missing condition");

  whileConds[op] = condValue;

  if (condArgs.size() != op.getNumResults())
    return op.emitError("scf.while result arity mismatch");

  updateInvariantCond(beforeState, condValue);

  llvm::SmallVector<mlir::Value, 4> afterArgs;
  llvm::SmallVector<mlir::Value, 4> exitValues;
  afterArgs.reserve(condArgs.size());
  exitValues.reserve(condArgs.size());
  for (mlir::Value value : condArgs) {
    auto branch = builder.create<circt::handshake::ConditionalBranchOp>(
        loc, condValue, value);
    afterArgs.push_back(branch.getTrueResult());
    exitValues.push_back(branch.getFalseResult());
  }

  auto gate = builder.create<GateOp>(
      loc, mlir::TypeRange{condValue.getType(), builder.getI1Type()}, condValue,
      condValue);
  mlir::Value bodyCond = gate.getCond();

  mlir::Block &afterBlock = op.getAfter().front();
  if (afterBlock.getNumArguments() != afterArgs.size())
    return op.emitError("scf.while after arity mismatch");

  RegionState afterState;
  afterState.region = &op.getAfter();
  afterState.parent = &state;
  afterState.invariantCond = bodyCond;
  afterState.pendingCond = false;
  afterState.controlToken =
      builder
          .create<InvariantOp>(loc, beforeState.controlToken.getType(), bodyCond,
                               beforeState.controlToken)
          .getO();

  for (unsigned i = 0, e = afterArgs.size(); i < e; ++i)
    afterState.valueMap[afterBlock.getArgument(i)] = afterArgs[i];

  llvm::SmallVector<mlir::Value, 4> yieldValues;
  for (mlir::Operation &nested : afterBlock) {
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
      for (mlir::Value operand : yield.getOperands())
        yieldValues.push_back(mapValue(operand, afterState, yield.getLoc()));
      break;
    }
    if (mlir::failed(lowerOp(&nested, afterState)))
      return mlir::failure();
  }

  if (yieldValues.size() != carries.size())
    return op.emitError("scf.while yield arity mismatch");

  for (unsigned i = 0, e = carries.size(); i < e; ++i)
    carries[i]->setOperand(2, yieldValues[i]);

  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i)
    state.valueMap[op.getResult(i)] = exitValues[i];

  updateInvariantCond(afterState, bodyCond);
  return mlir::success();
}

mlir::LogicalResult HandshakeLowering::lowerIf(mlir::scf::IfOp op,
                                               RegionState &state) {
  mlir::Location loc = op.getLoc();
  mlir::Value condValue = mapValue(op.getCondition(), state, loc);
  ifConds[op] = condValue;
  mlir::Value ctrlToken = state.controlToken ? state.controlToken
                                             : getEntryToken(loc);

  auto branch = builder.create<circt::handshake::ConditionalBranchOp>(
      loc, condValue, ctrlToken);
  mlir::Value thenCtrl = branch.getTrueResult();
  mlir::Value elseCtrl = branch.getFalseResult();

  mlir::Region &thenRegion = op.getThenRegion();
  RegionState thenState;
  thenState.region = &thenRegion;
  thenState.parent = &state;
  thenState.controlToken = thenCtrl;

  llvm::SmallVector<mlir::Value, 4> thenValues;
  if (!thenRegion.hasOneBlock())
    return op.emitError("expected single-block scf.if then region");
  for (mlir::Operation &nested : thenRegion.front()) {
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
      for (mlir::Value operand : yield.getOperands())
        thenValues.push_back(mapValue(operand, thenState, yield.getLoc()));
      break;
    }
    if (mlir::failed(lowerOp(&nested, thenState)))
      return mlir::failure();
  }

  llvm::SmallVector<mlir::Value, 4> elseValues;
  bool hasElse = !op.getElseRegion().empty();
  if (hasElse) {
    mlir::Region &elseRegion = op.getElseRegion();
    RegionState elseState;
    elseState.region = &elseRegion;
    elseState.parent = &state;
    elseState.controlToken = elseCtrl;
    if (!elseRegion.hasOneBlock())
      return op.emitError("expected single-block scf.if else region");
    for (mlir::Operation &nested : elseRegion.front()) {
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
        for (mlir::Value operand : yield.getOperands())
          elseValues.push_back(mapValue(operand, elseState, yield.getLoc()));
        break;
      }
      if (mlir::failed(lowerOp(&nested, elseState)))
        return mlir::failure();
    }
  }

  if (op.getNumResults() == 0)
    return mlir::success();

  if (!hasElse)
    return op.emitError("scf.if without else cannot return values");

  mlir::Value zero = makeConstant(
      loc, builder.getIndexAttr(0), builder.getIndexType(), ctrlToken);
  mlir::Value one = makeConstant(
      loc, builder.getIndexAttr(1), builder.getIndexType(), ctrlToken);
  mlir::Value select =
      builder.create<mlir::arith::SelectOp>(loc, condValue, one, zero);

  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    auto mux = builder.create<circt::handshake::MuxOp>(
        loc, select, mlir::ValueRange{elseValues[i], thenValues[i]});
    state.valueMap[op.getResult(i)] = mux.getResult();
  }

  return mlir::success();
}

mlir::LogicalResult HandshakeLowering::lowerIndexSwitch(
    mlir::scf::IndexSwitchOp op, RegionState &state) {
  mlir::Location loc = op.getLoc();
  mlir::Value indexValue = mapValue(op.getArg(), state, loc);
  switchConds[op] = indexValue;
  mlir::Value ctrlToken = state.controlToken ? state.controlToken
                                             : getEntryToken(loc);

  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> regionResults;
  llvm::SmallVector<mlir::Value, 4> caseConds;

  mlir::Value chainCtrl = ctrlToken;
  auto cases = op.getCases();
  auto caseRegions = op.getCaseRegions();

  for (auto [caseValue, caseRegion] : llvm::zip(cases, caseRegions)) {
    mlir::Value caseConst = makeConstant(
        loc, builder.getIndexAttr(caseValue), builder.getIndexType(), ctrlToken);
    mlir::Value caseCond = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, indexValue, caseConst);
    caseConds.push_back(caseCond);

    auto branch = builder.create<circt::handshake::ConditionalBranchOp>(
        loc, caseCond, chainCtrl);
    mlir::Value caseCtrl = branch.getTrueResult();
    chainCtrl = branch.getFalseResult();

    RegionState caseState;
    caseState.region = &caseRegion;
    caseState.parent = &state;
    caseState.controlToken = caseCtrl;

    if (!caseRegion.hasOneBlock())
      return op.emitError("expected single-block scf.index_switch case region");

    llvm::SmallVector<mlir::Value, 4> caseValues;
    for (mlir::Operation &nested : caseRegion.front()) {
      if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
        for (mlir::Value operand : yield.getOperands())
          caseValues.push_back(mapValue(operand, caseState, yield.getLoc()));
        break;
      }
      if (mlir::failed(lowerOp(&nested, caseState)))
        return mlir::failure();
    }

    regionResults.push_back(std::move(caseValues));
  }

  mlir::Region &defaultRegion = op.getDefaultRegion();
  RegionState defaultState;
  defaultState.region = &defaultRegion;
  defaultState.parent = &state;
  defaultState.controlToken = chainCtrl;

  if (!defaultRegion.hasOneBlock())
    return op.emitError("expected single-block scf.index_switch default region");

  llvm::SmallVector<mlir::Value, 4> defaultValues;
  for (mlir::Operation &nested : defaultRegion.front()) {
    if (auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(nested)) {
      for (mlir::Value operand : yield.getOperands())
        defaultValues.push_back(mapValue(operand, defaultState, yield.getLoc()));
      break;
    }
    if (mlir::failed(lowerOp(&nested, defaultState)))
      return mlir::failure();
  }

  regionResults.push_back(std::move(defaultValues));

  if (op.getNumResults() == 0)
    return mlir::success();

  mlir::Value select = makeConstant(
      loc, builder.getIndexAttr(cases.size()), builder.getIndexType(), ctrlToken);
  for (int64_t i = static_cast<int64_t>(caseConds.size()) - 1; i >= 0; --i) {
    mlir::Value caseIndex = makeConstant(
        loc, builder.getIndexAttr(i), builder.getIndexType(), ctrlToken);
    select = builder.create<mlir::arith::SelectOp>(
        loc, caseConds[static_cast<size_t>(i)], caseIndex, select);
  }

  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    llvm::SmallVector<mlir::Value, 4> values;
    values.reserve(regionResults.size());
    for (auto &caseValues : regionResults) {
      if (caseValues.size() != e)
        return op.emitError("scf.index_switch yield arity mismatch");
      values.push_back(caseValues[i]);
    }
    auto mux = builder.create<circt::handshake::MuxOp>(loc, select, values);
    state.valueMap[op.getResult(i)] = mux.getResult();
  }

  return mlir::success();
}

mlir::LogicalResult HandshakeLowering::lowerOp(mlir::Operation *op,
                                               RegionState &state) {
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op))
    return lowerFor(forOp, state);
  if (auto whileOp = mlir::dyn_cast<mlir::scf::WhileOp>(op))
    return lowerWhile(whileOp, state);
  if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(op))
    return lowerIf(ifOp, state);
  if (auto switchOp = mlir::dyn_cast<mlir::scf::IndexSwitchOp>(op))
    return lowerIndexSwitch(switchOp, state);
  if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op))
    return lowerReturn(ret, state);
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
    return lowerLoad(load, state);
  if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op))
    return lowerStore(store, state);
  if (auto castOp = mlir::dyn_cast<mlir::memref::CastOp>(op)) {
    if (mlir::isa<mlir::BlockArgument>(castOp.getSource())) {
      mlir::Value mapped = mapValue(castOp.getSource(), state, op->getLoc());
      state.valueMap[castOp.getResult()] = mapped;
      return mlir::success();
    }
  }
  if (auto allocaOp = mlir::dyn_cast<mlir::memref::AllocaOp>(op)) {
    llvm::SmallVector<mlir::Value, 4> dynSizes;
    for (mlir::Value size : allocaOp.getDynamicSizes())
      dynSizes.push_back(mapValue(size, state, op->getLoc()));
    auto newAlloc = builder.create<mlir::memref::AllocOp>(
        op->getLoc(), allocaOp.getType(), dynSizes,
        allocaOp.getAlignmentAttr());
    state.valueMap[allocaOp.getResult()] = newAlloc.getResult();
    return mlir::success();
  }
  if (auto allocOp = mlir::dyn_cast<mlir::memref::AllocOp>(op)) {
    llvm::SmallVector<mlir::Value, 4> dynSizes;
    for (mlir::Value size : allocOp.getDynamicSizes())
      dynSizes.push_back(mapValue(size, state, op->getLoc()));
    auto newAlloc = builder.create<mlir::memref::AllocOp>(
        op->getLoc(), allocOp.getType(), dynSizes, allocOp.getAlignmentAttr());
    state.valueMap[allocOp.getResult()] = newAlloc.getResult();
    return mlir::success();
  }
  if (auto deallocOp = mlir::dyn_cast<mlir::memref::DeallocOp>(op)) {
    mlir::Value mapped = mapValue(deallocOp.getMemref(), state, op->getLoc());
    builder.create<mlir::memref::DeallocOp>(op->getLoc(), mapped);
    return mlir::success();
  }
  if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
    llvm::SmallVector<mlir::Value, 4> args;
    for (mlir::Value operand : callOp.getOperands())
      args.push_back(mapValue(operand, state, op->getLoc()));
    auto newCall = builder.create<mlir::func::CallOp>(
        op->getLoc(), callOp.getCallee(), callOp.getResultTypes(), args);
    for (unsigned i = 0, e = callOp.getNumResults(); i < e; ++i)
      state.valueMap[callOp.getResult(i)] = newCall.getResult(i);
    return mlir::success();
  }

  if (auto constantOp = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
    mlir::Location loc = op->getLoc();
    mlir::Value ctrlToken =
        state.controlToken ? state.controlToken : getEntryToken(loc);
    mlir::Value result = makeConstant(loc, constantOp.getValue(),
                                      constantOp.getType(), ctrlToken);
    if (mlir::Operation *def = result.getDefiningOp())
      copyLoomAnnotations(op, def);
    state.valueMap[constantOp.getResult()] = result;
    return mlir::success();
  }

  if (op->getNumRegions() == 0) {
    mlir::IRMapping mapping;
    for (mlir::Value operand : op->getOperands())
      mapping.map(operand, mapValue(operand, state, op->getLoc()));
    mlir::Operation *clone = builder.clone(*op, mapping);
    copyLoomAnnotations(op, clone);
    for (unsigned i = 0, e = op->getNumResults(); i < e; ++i)
      state.valueMap[op->getResult(i)] = clone->getResult(i);
    return mlir::success();
  }

  op->emitError("unsupported op in SCF to Handshake lowering");
  return mlir::failure();
}

void HandshakeLowering::insertForks() {
  mlir::Block *block = handshakeFunc.getBodyBlock();
  llvm::SmallVector<mlir::Value, 16> values;
  for (mlir::BlockArgument arg : block->getArguments())
    values.push_back(arg);
  for (mlir::Operation &op : *block) {
    for (mlir::Value res : op.getResults())
      values.push_back(res);
  }

  for (mlir::Value value : values) {
    if (!value)
      continue;
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      if (mlir::isa<mlir::BaseMemRefType>(arg.getType()))
        continue;
    }
    if (value.use_empty() || value.hasOneUse())
      continue;

    llvm::SmallVector<mlir::OpOperand *, 4> uses;
    for (mlir::OpOperand &use : value.getUses())
      uses.push_back(&use);

    mlir::OpBuilder::InsertionGuard guard(builder);
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value))
      builder.setInsertionPointToStart(block);
    else
      builder.setInsertionPointAfter(value.getDefiningOp());

    auto fork = builder.create<circt::handshake::ForkOp>(
        value.getLoc(), value, static_cast<unsigned>(uses.size()));
    for (size_t i = 0; i < uses.size(); ++i)
      uses[i]->set(fork.getResults()[i]);
  }
}

mlir::LogicalResult HandshakeLowering::run() {
  builder.setInsertionPointAfter(func);
  auto originalType = func.getFunctionType();
  llvm::SmallVector<mlir::Type, 8> inputTypes;
  llvm::SmallVector<mlir::Type, 4> resultTypes;
  for (mlir::Type type : originalType.getInputs())
    inputTypes.push_back(type);
  for (mlir::Type type : originalType.getResults())
    resultTypes.push_back(type);
  inputTypes.push_back(builder.getI1Type());
  resultTypes.push_back(builder.getI1Type());
  auto handshakeType =
      builder.getFunctionType(inputTypes, resultTypes);

  handshakeFunc = builder.create<circt::handshake::FuncOp>(
      func.getLoc(), func.getName(), handshakeType);
  handshakeFunc.resolveArgAndResNames();
  if (auto visibility = func.getSymVisibilityAttr())
    handshakeFunc->setAttr(mlir::SymbolTable::getVisibilityAttrName(),
                           visibility);
  copyLoomAnnotations(func, handshakeFunc);
  assignHandshakeNames();

  mlir::Block *entry = new mlir::Block();
  handshakeFunc.getBody().push_back(entry);
  for (mlir::Type inputType : handshakeType.getInputs())
    entry->addArgument(inputType, func.getLoc());
  builder.setInsertionPointToStart(entry);

  if (!func.getBody().hasOneBlock())
    return func.emitError("expected single-block function body");

  RegionState state;
  state.region = &func.getBody();
  state.parent = nullptr;
  entrySignal = handshakeFunc.getArguments().back();
  entryToken = builder
                   .create<circt::handshake::JoinOp>(
                       func.getLoc(), mlir::ValueRange{entrySignal})
                   .getResult();
  state.controlToken = entryToken;

  auto newArgs = handshakeFunc.getArguments().drop_back(1);
  for (auto [oldArg, newArg] : llvm::zip(func.getArguments(), newArgs)) {
    state.valueMap[oldArg] = newArg;
  }

  mlir::Block &bodyBlock = func.getBody().front();
  for (mlir::Operation &op : bodyBlock) {
    if (mlir::failed(lowerOp(&op, state)))
      return mlir::failure();
  }

  if (!sawReturn)
    return func.emitError("missing func.return in accel function");

  finalizeMemory();
  if (mlir::failed(buildMemoryControl()))
    return mlir::failure();

  mlir::Value doneCtrl = memoryDoneToken ? memoryDoneToken : entryToken;
  if (!doneCtrl)
    doneCtrl = getEntryToken(func.getLoc());

  mlir::OperationState doneState(
      returnLoc, circt::handshake::ConstantOp::getOperationName());
  doneState.addOperands(doneCtrl);
  doneState.addTypes(builder.getI1Type());
  doneState.addAttribute("value", builder.getBoolAttr(true));
  mlir::Operation *doneOp = builder.create(doneState);
  doneSignal = doneOp->getResult(0);

  llvm::SmallVector<mlir::Value, 4> returnOperands(pendingReturnValues.begin(),
                                                   pendingReturnValues.end());
  returnOperands.push_back(doneSignal);
  builder.create<circt::handshake::ReturnOp>(returnLoc, returnOperands);

  insertForks();
  if (mlir::failed(loom::runHandshakeCleanup(handshakeFunc, builder)))
    return mlir::failure();

  func.erase();
  return mlir::success();
}

} // namespace detail
} // namespace loom

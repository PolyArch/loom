//===- FabricMemOp.cpp - Parser/printer/verifier for fabric.mem -----------===//
//
// Implements parser, printer, and verifier for fabric.mem. The op is a
// leaf-level memory tile with a `[spatial]` or `[temporal]` schedule
// predicate. It wraps `dataflow.load`/`dataflow.store` semantics into a
// single fabric-domain op that owns a Manager-side `memref_mgr` (always
// present, the first SSA operand) and exposes an optional Subordinate-side
// `memref_sub` (when present, the first SSA result).
//
// Operand layout:
//   memref_mgr,
//   per load port `i`:    addr_i, ctrl_i,
//   per store port `j`:   addr_j, data_j, ctrl_j.
//
// Result layout:
//   [memref_sub,]
//   per load port `i`:    data_i, done_i,
//   per store port `j`:   done_j.
//
// Hardware parameters in `[ ... ]`:
//   spatial:  load_group_size, store_group_size
//   temporal: load_group_size, store_group_size, tag_width, addr_table_size
//
// Software configuration in `{ ... }` (all-or-nothing):
//   addr_table = [ {base_addr, element_log2_size, [tag,] valid}, ... ]
//   mem_enable = true|false
//
// See `docs/spec-fabric-mem.md` for the full per-schedule rules.
//
//===----------------------------------------------------------------------===//

#include "Common/IndexWidth.h"
#include "Common/LoomConstants.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace fabric;

//===----------------------------------------------------------------------===//
// resolveLoomAddrBits / resolveLoomMemBusWidth
//===----------------------------------------------------------------------===//

namespace fabric {

unsigned resolveLoomAddrBits(Operation *op) {
  Operation *cur = op;
  while (cur) {
    if (auto m = dyn_cast<ModuleOp>(cur)) {
      if (auto a = m.getLoomAddrBitsAttr())
        return static_cast<unsigned>(a.getInt());
      break;
    }
    cur = cur->getParentOp();
  }
  return ::loom::getDefaultLoomAddrBits();
}

unsigned resolveLoomMemBusWidth(Operation *op) {
  Operation *cur = op;
  while (cur) {
    if (auto m = dyn_cast<ModuleOp>(cur)) {
      if (auto a = m.getLoomMemBusWidthAttr())
        return static_cast<unsigned>(a.getInt());
      break;
    }
    cur = cur->getParentOp();
  }
  return ::loom::getDefaultLoomMemBusWidth();
}

} // namespace fabric

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

// Integer log2 of a power-of-two-or-larger positive integer. Returns the
// floor of log2(x).
static unsigned floorLog2(uint64_t x) {
  if (x == 0)
    return 0;
  unsigned r = 0;
  while ((x >> r) > 1)
    ++r;
  return r;
}

} // namespace

//===----------------------------------------------------------------------===//
// fabric.mem: parser
//===----------------------------------------------------------------------===//

ParseResult MemOp::parse(OpAsmParser &parser, OperationState &result) {
  // Optional `@sym_name` immediately after the op keyword. When present
  // the parser switches to the named template form.
  StringAttr nameAttr;
  bool isNamed = succeeded(parser.parseOptionalSymbolName(
      nameAttr, ::mlir::SymbolTable::getSymbolAttrName(), result.attributes));

  // Mandatory `[<schedule>]` predicate.
  StringRef scheduleKw;
  SMLoc scheduleLoc = parser.getCurrentLocation();
  if (parser.parseLSquare() || parser.parseKeyword(&scheduleKw) ||
      parser.parseRSquare())
    return failure();
  auto sym = symbolizeSchedule(scheduleKw);
  if (!sym)
    return parser.emitError(scheduleLoc, "expected fabric mem schedule keyword "
                                         "'spatial' or 'temporal', got '")
           << scheduleKw << "'";
  result.addAttribute("schedule", ScheduleAttr::get(parser.getContext(), *sym));

  if (isNamed) {
    // Named template form: parse function-type signature
    //   `(<input-types>) -> (<result-types>)`.
    SmallVector<Type, 4> argTypes;
    if (parser.parseLParen())
      return failure();
    if (failed(parser.parseOptionalRParen())) {
      if (parser.parseTypeList(argTypes) || parser.parseRParen())
        return failure();
    }
    if (parser.parseArrow())
      return failure();
    SmallVector<Type, 4> resultTypes;
    if (succeeded(parser.parseOptionalLParen())) {
      if (failed(parser.parseOptionalRParen())) {
        if (parser.parseTypeList(resultTypes) || parser.parseRParen())
          return failure();
      }
    } else {
      Type ty;
      if (parser.parseType(ty))
        return failure();
      resultTypes.push_back(ty);
    }
    auto funcType =
        FunctionType::get(parser.getContext(), argTypes, resultTypes);
    result.addAttribute("function_type", TypeAttr::get(funcType));

    // Parse optional `[ hw_params ]` and `{ ... }`.
    if (succeeded(parser.parseOptionalLSquare())) {
      SmallVector<Attribute, 1> hwElems;
      auto parseOneHw = [&]() -> ParseResult {
        DictionaryAttr d;
        if (parser.parseAttribute(d))
          return failure();
        hwElems.push_back(d);
        return success();
      };
      if (failed(parser.parseOptionalRSquare())) {
        if (parseOneHw())
          return failure();
        while (succeeded(parser.parseOptionalComma()))
          if (parseOneHw())
            return failure();
        if (parser.parseRSquare())
          return failure();
      }
      result.addAttribute("hw_params",
                          ArrayAttr::get(parser.getContext(), hwElems));
    }
    {
      DictionaryAttr d;
      OptionalParseResult opr = parser.parseOptionalAttribute(d);
      if (opr.has_value()) {
        if (failed(*opr))
          return failure();
        // Lift `addr_table` and `mem_enable` to op attributes; reject any
        // other key in this dict (the verifier will produce a more
        // informative error if needed).
        if (auto at = d.get("addr_table"))
          result.addAttribute("addr_table", at);
        if (auto en = d.get("mem_enable"))
          result.addAttribute("mem_enable", en);
      }
    }
    return success();
  }

  // Anonymous form. Layout:
  //   mgr(%mgr)
  //     load(%la0, %lc0; %la1, %lc1; ...)
  //     store(%sa0, %sd0, %sc0; ...)
  //     [ hw_params ] { sw_configs }
  //     : (operand-types) -> (result-types)

  // mgr(%mgr).
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  SMLoc operandsLoc = parser.getCurrentLocation();
  if (parser.parseKeyword("mgr") || parser.parseLParen())
    return failure();
  OpAsmParser::UnresolvedOperand mgr;
  if (parser.parseOperand(mgr))
    return failure();
  operands.push_back(mgr);
  if (parser.parseRParen())
    return failure();

  // Optional load(...). Ports are separated by `;`; per-port operands
  // (`addr`, `ctrl`) are separated by `,`. The MLIR parser exposes no
  // primitive for `;`, so we use `parseOperandList` (returns a flat list
  // of operands separated by `,`) and an alternate per-port keyword
  // separator below. To keep round-trip clean, callers must spell each
  // port's operands as `%a, %c` and separate ports with `,` as well --
  // i.e. all load operands appear as one comma-separated list.
  if (succeeded(parser.parseOptionalKeyword("load"))) {
    SmallVector<OpAsmParser::UnresolvedOperand, 4> loadOps;
    if (parser.parseOperandList(loadOps, OpAsmParser::Delimiter::Paren))
      return failure();
    for (auto &o : loadOps)
      operands.push_back(o);
  }

  // Optional store(...). Same flat comma-separated list; each port has 3
  // operands (`addr`, `data`, `ctrl`).
  if (succeeded(parser.parseOptionalKeyword("store"))) {
    SmallVector<OpAsmParser::UnresolvedOperand, 4> storeOps;
    if (parser.parseOperandList(storeOps, OpAsmParser::Delimiter::Paren))
      return failure();
    for (auto &o : storeOps)
      operands.push_back(o);
  }

  // Optional `[ hw_params ]`.
  if (succeeded(parser.parseOptionalLSquare())) {
    SmallVector<Attribute, 1> hwElems;
    auto parseOneHw = [&]() -> ParseResult {
      DictionaryAttr d;
      if (parser.parseAttribute(d))
        return failure();
      hwElems.push_back(d);
      return success();
    };
    if (failed(parser.parseOptionalRSquare())) {
      if (parseOneHw())
        return failure();
      while (succeeded(parser.parseOptionalComma()))
        if (parseOneHw())
          return failure();
      if (parser.parseRSquare())
        return failure();
    }
    result.addAttribute("hw_params",
                        ArrayAttr::get(parser.getContext(), hwElems));
  }

  // Optional `{ sw_configs }`.
  {
    DictionaryAttr d;
    OptionalParseResult opr = parser.parseOptionalAttribute(d);
    if (opr.has_value()) {
      if (failed(*opr))
        return failure();
      if (auto at = d.get("addr_table"))
        result.addAttribute("addr_table", at);
      if (auto en = d.get("mem_enable"))
        result.addAttribute("mem_enable", en);
    }
  }

  if (parser.parseColon())
    return failure();

  SmallVector<Type, 8> sourceTypes;
  SmallVector<Type, 8> inputPortTypes;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    auto parseOneType = [&]() -> ParseResult {
      Type sourceType;
      if (parser.parseType(sourceType))
        return failure();
      Type inputPortType = sourceType;
      if (succeeded(parser.parseOptionalKeyword("to")))
        if (parser.parseType(inputPortType))
          return failure();
      sourceTypes.push_back(sourceType);
      inputPortTypes.push_back(inputPortType);
      return success();
    };
    if (parseOneType())
      return failure();
    while (succeeded(parser.parseOptionalComma()))
      if (parseOneType())
        return failure();
    if (parser.parseRParen())
      return failure();
  }
  if (parser.parseArrow())
    return failure();
  SmallVector<Type, 8> resultTypes;
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseOptionalRParen())) {
      if (parser.parseTypeList(resultTypes) || parser.parseRParen())
        return failure();
    }
  } else {
    Type ty;
    if (parser.parseType(ty))
      return failure();
    resultTypes.push_back(ty);
  }
  if (sourceTypes.size() != operands.size())
    return parser.emitError(operandsLoc,
                            "operand count does not match type list count");
  if (parser.resolveOperands(operands, sourceTypes, operandsLoc,
                             result.operands))
    return failure();
  bool anyDiffer = false;
  for (auto [sourceType, inputPortType] :
       llvm::zip(sourceTypes, inputPortTypes))
    if (sourceType != inputPortType) {
      anyDiffer = true;
      break;
    }
  if (anyDiffer) {
    result.getOrAddProperties<Properties>().setInnerInputTypes(inputPortTypes);
  }
  result.addTypes(resultTypes);
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.mem: printer
//===----------------------------------------------------------------------===//

void MemOp::print(OpAsmPrinter &p) {
  bool isNamed = static_cast<bool>(getSymNameAttr());
  if (isNamed) {
    p << ' ';
    p.printSymbolName(getSymNameAttr().getValue());
  }
  p << " [" << stringifySchedule(getSchedule()) << "]";

  if (isNamed) {
    FunctionType ft;
    if (auto fta = getFunctionTypeAttr())
      ft = cast<FunctionType>(fta.getValue());
    p << " (";
    if (ft)
      llvm::interleaveComma(ft.getInputs(), p);
    p << ") -> ";
    if (ft && ft.getNumResults() == 1) {
      p << ft.getResult(0);
    } else {
      p << '(';
      if (ft)
        llvm::interleaveComma(ft.getResults(), p);
      p << ')';
    }
  } else {
    // Anonymous form: print mgr(%mgr) load(...) store(...).
    auto inputs = getInputs();
    p << " mgr(" << inputs[0] << ")";

    // Determine load_group_size and store_group_size from hw_params if
    // present; otherwise from operand count assuming no load ports.
    unsigned loadN = 0, storeN = 0;
    if (auto hp = getHwParamsAttr()) {
      if (hp.size() == 1) {
        if (auto d = dyn_cast<DictionaryAttr>(hp[0])) {
          if (auto la = d.get("load_group_size"))
            if (auto i = dyn_cast<IntegerAttr>(la))
              loadN = static_cast<unsigned>(i.getInt());
          if (auto sa = d.get("store_group_size"))
            if (auto i = dyn_cast<IntegerAttr>(sa))
              storeN = static_cast<unsigned>(i.getInt());
        }
      }
    }

    // load(...): all per-port operands flat in a single comma-separated
    // list. Each port contributes 2 operands (addr, ctrl).
    if (loadN > 0) {
      p << " load(";
      for (unsigned i = 0; i < loadN; ++i) {
        if (i)
          p << ", ";
        unsigned base = 1 + 2 * i;
        if (base + 1 < inputs.size())
          p << inputs[base] << ", " << inputs[base + 1];
      }
      p << ")";
    }

    // store(...): each port contributes 3 operands (addr, data, ctrl).
    if (storeN > 0) {
      p << " store(";
      unsigned storeBase = 1 + 2 * loadN;
      for (unsigned j = 0; j < storeN; ++j) {
        if (j)
          p << ", ";
        unsigned base = storeBase + 3 * j;
        if (base + 2 < inputs.size())
          p << inputs[base] << ", " << inputs[base + 1] << ", "
            << inputs[base + 2];
      }
      p << ")";
    }
  }

  // hw_params in `[ ... ]`.
  if (auto hp = getHwParamsAttr()) {
    p << ' ' << '[';
    llvm::interleaveComma(hp, p, [&](Attribute a) { p.printAttribute(a); });
    p << ']';
  }

  // sw_configs in `{ ... }` -- assemble from addr_table/mem_enable.
  ArrayAttr at = getAddrTableAttr();
  BoolAttr en = getMemEnableAttr();
  if (at || en) {
    p << " {";
    bool first = true;
    if (at) {
      if (!first)
        p << ", ";
      first = false;
      p << "addr_table = ";
      p.printAttribute(at);
    }
    if (en) {
      if (!first)
        p << ", ";
      first = false;
      p << "mem_enable = ";
      p.printAttribute(en);
    }
    p << "}";
  }

  if (!isNamed) {
    ArrayRef<Type> innerTypes = getInnerInputTypes();
    SmallVector<Type, 8> inputPortTypes;
    inputPortTypes.reserve(getInputs().size());
    if (!innerTypes.empty() && innerTypes.size() == getInputs().size()) {
      inputPortTypes.append(innerTypes.begin(), innerTypes.end());
    } else {
      for (Value input : getInputs())
        inputPortTypes.push_back(input.getType());
    }
    p << " : (";
    llvm::interleaveComma(llvm::zip(getInputs(), inputPortTypes), p,
                          [&](auto pair) {
                            Value input;
                            Type inputPortType;
                            std::tie(input, inputPortType) = pair;
                            Type sourceType = input.getType();
                            p << sourceType;
                            if (inputPortType && inputPortType != sourceType)
                              p << " to " << inputPortType;
                          });
    p << ") -> ";
    auto rTypes = getResultTypes();
    if (rTypes.size() == 1) {
      p << rTypes.front();
    } else {
      p << '(';
      llvm::interleaveComma(rTypes, p);
      p << ')';
    }
  }
}

bool MemOp::isOptionalSymbol() { return true; }

//===----------------------------------------------------------------------===//
// fabric.mem: verifier helpers
//===----------------------------------------------------------------------===//

namespace {

static LogicalResult
collectAnonymousInputPortTypes(MemOp op,
                               SmallVectorImpl<Type> &inputPortTypes) {
  ArrayRef<Type> innerTypes = op.getInnerInputTypes();
  if (!innerTypes.empty()) {
    inputPortTypes.append(innerTypes.begin(), innerTypes.end());
  } else {
    for (Value input : op.getInputs())
      inputPortTypes.push_back(input.getType());
  }

  for (auto [i, pair] :
       llvm::enumerate(llvm::zip(op.getInputs(), inputPortTypes))) {
    Value input;
    Type inputPortType;
    std::tie(input, inputPortType) = pair;
    Type sourceType = input.getType();
    if (isa<MemRefType>(sourceType) || isa<MemRefType>(inputPortType)) {
      if (sourceType != inputPortType)
        return op.emitOpError("incoming connection operand #")
               << i
               << ": memref capabilities cannot use the 'to "
                  "<destination-type>' clause; memref types must match "
                  "exactly";
      continue;
    }
    if (!haveSameFabricModulePortKind(sourceType, inputPortType))
      return op.emitOpError("incoming connection operand #")
             << i << " source type " << sourceType
             << " and destination port type " << inputPortType
             << " must share the same fabric kind (bits or bits_tag)";
  }
  return success();
}

// Decode hw_params length-1-array-of-dict pattern. Returns the dict on
// success.
static LogicalResult readHwParams(MemOp op, DictionaryAttr &outDict) {
  auto hp = op.getHwParamsAttr();
  if (!hp)
    return op.emitOpError("requires 'hw_params' with 'load_group_size' and "
                          "'store_group_size'");
  if (hp.size() != 1)
    return op.emitOpError(
               "'hw_params' must be a length-1 array wrapping a dictionary, "
               "got length ")
           << hp.size();
  auto d = dyn_cast<DictionaryAttr>(hp[0]);
  if (!d)
    return op.emitOpError("'hw_params' inner element must be a DictionaryAttr");
  outDict = d;
  return success();
}

static LogicalResult readSizeKey(MemOp op, DictionaryAttr d, StringRef key,
                                 int64_t &out, int64_t minVal) {
  auto a = d.get(key);
  if (!a)
    return op.emitOpError("'hw_params' missing required key '") << key << "'";
  auto i = dyn_cast<IntegerAttr>(a);
  if (!i)
    return op.emitOpError("'") << key << "' must be an IntegerAttr";
  int64_t v = i.getValue().getSExtValue();
  if (v < minVal)
    return op.emitOpError("'")
           << key << "' must be >= " << minVal << ", got " << v;
  out = v;
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// fabric.mem: verifier
//===----------------------------------------------------------------------===//

LogicalResult MemOp::verify() {
  if (failed(verifyInnerInputTypesProperty(getOperation(), getInputs(),
                                           getInnerInputTypes())))
    return failure();

  // Form selection.
  bool isNamed = static_cast<bool>(getSymNameAttr());
  SmallVector<Type, 8> inTys, outTys;
  if (isNamed) {
    if (!getInputs().empty())
      return emitOpError(
                 "named fabric.mem template must have zero SSA operands; got ")
             << getInputs().size();
    if (!getResultTypes().empty())
      return emitOpError(
                 "named fabric.mem template must have zero SSA results; got ")
             << getResultTypes().size();
    if (!getInnerInputTypes().empty())
      return emitOpError("named fabric.mem template must not carry '")
             << kInnerInputTypesPropertyName << "'";
    auto fta = getFunctionTypeAttr();
    if (!fta)
      return emitOpError(
          "named fabric.mem template requires a 'function_type' attribute");
    auto ft = dyn_cast<FunctionType>(fta.getValue());
    if (!ft)
      return emitOpError("'function_type' must be a FunctionType");
    inTys.assign(ft.getInputs().begin(), ft.getInputs().end());
    outTys.assign(ft.getResults().begin(), ft.getResults().end());
  } else {
    if (getFunctionTypeAttr())
      return emitOpError(
          "anonymous fabric.mem must not carry a 'function_type' attribute");
    if (failed(collectAnonymousInputPortTypes(*this, inTys)))
      return failure();
    for (Type t : getResultTypes())
      outTys.push_back(t);
  }

  // hw_params -> load_group_size / store_group_size + temporal-only keys.
  DictionaryAttr hwDict;
  if (failed(readHwParams(*this, hwDict)))
    return failure();
  int64_t loadN = 0, storeN = 0;
  if (failed(readSizeKey(*this, hwDict, "load_group_size", loadN, 0)))
    return failure();
  if (failed(readSizeKey(*this, hwDict, "store_group_size", storeN, 0)))
    return failure();
  if (loadN + storeN < 1)
    return emitOpError("load_group_size + store_group_size must be >= 1 (got "
                       "load_group_size = ")
           << loadN << ", store_group_size = " << storeN << ")";

  bool isTemporal = (getSchedule() == Schedule::Temporal);
  int64_t tagWidth = 0, addrTableSize = 0;
  if (isTemporal) {
    if (failed(readSizeKey(*this, hwDict, "tag_width", tagWidth, 1)))
      return failure();
    if (failed(readSizeKey(*this, hwDict, "addr_table_size", addrTableSize, 1)))
      return failure();
  } else {
    if (hwDict.get("tag_width"))
      return emitOpError(
          "spatial fabric.mem must not carry temporal-only attribute "
          "'tag_width'");
    if (hwDict.get("addr_table_size"))
      return emitOpError(
          "spatial fabric.mem must not carry temporal-only attribute "
          "'addr_table_size'");
  }

  // Operand count: 1 + 2*loadN + 3*storeN.
  uint64_t expectedOps = 1u + 2u * (uint64_t)loadN + 3u * (uint64_t)storeN;
  if ((uint64_t)inTys.size() != expectedOps)
    return emitOpError("expected ")
           << expectedOps << " operand types (1 memref_mgr + 2*" << loadN
           << " load + 3*" << storeN << " store), got " << inTys.size();

  // memref_mgr type.
  auto mgrMemref = dyn_cast<MemRefType>(inTys[0]);
  if (!mgrMemref)
    return emitOpError("first operand 'memref_mgr' must be a memref type, got ")
           << inTys[0];
  auto mgrElem = dyn_cast<BitsType>(mgrMemref.getElementType());
  if (!mgrElem)
    return emitOpError(
               "memref_mgr element type must be '!fabric.bits<W>', got ")
           << mgrMemref.getElementType();
  unsigned wMgr = mgrElem.getWidth();

  // memref_sub: optional first result if it's a memref.
  bool hasSub = false;
  unsigned subIdx = 0; // result index where load/store result list starts.
  if (!outTys.empty() && isa<MemRefType>(outTys[0])) {
    hasSub = true;
    auto subMemref = cast<MemRefType>(outTys[0]);
    auto subElem = dyn_cast<BitsType>(subMemref.getElementType());
    if (!subElem)
      return emitOpError(
                 "memref_sub element type must be '!fabric.bits<W_sub>', got ")
             << subMemref.getElementType();
    subIdx = 1;
  }

  uint64_t expectedRes =
      (uint64_t)(hasSub ? 1 : 0) + 2u * (uint64_t)loadN + (uint64_t)storeN;
  if ((uint64_t)outTys.size() != expectedRes)
    return emitOpError("expected ")
           << expectedRes << " result types ("
           << (hasSub ? "1 memref_sub + " : "") << "2*" << loadN << " load + "
           << storeN << " store), got " << outTys.size();

  // Per-port type checks.
  unsigned indexW = ::loom::getIndexWidth();
  unsigned T = (unsigned)tagWidth;

  auto mkExpectAddr = [&](unsigned width) -> Type {
    if (isTemporal)
      return BitsTagType::get(getContext(), width, T);
    return BitsType::get(getContext(), width);
  };
  auto mkExpectCtrl = [&]() -> Type { return mkExpectAddr(0); };
  auto mkExpectData = [&]() -> Type { return mkExpectAddr(wMgr); };

  Type expectAddr = mkExpectAddr(indexW);
  Type expectCtrl = mkExpectCtrl();
  Type expectData = mkExpectData();

  StringRef portKindMsg =
      isTemporal ? "temporal fabric.mem requires '!fabric.bits_tag<W, T>' ports"
                 : "spatial fabric.mem requires '!fabric.bits<W>' ports";

  // Load ports (operands).
  for (int64_t i = 0; i < loadN; ++i) {
    unsigned base = 1 + 2 * (unsigned)i;
    Type a = inTys[base];
    Type c = inTys[base + 1];
    if (a != expectAddr)
      return emitOpError("schedule mismatch with port kind: ")
             << portKindMsg << "; load port #" << i << " addr type " << a
             << " (expected " << expectAddr << ")";
    if (c != expectCtrl)
      return emitOpError("schedule mismatch with port kind: ")
             << portKindMsg << "; load port #" << i << " ctrl type " << c
             << " (expected " << expectCtrl << ")";
  }
  // Store ports (operands).
  for (int64_t j = 0; j < storeN; ++j) {
    unsigned base = 1 + 2 * (unsigned)loadN + 3 * (unsigned)j;
    Type a = inTys[base];
    Type d = inTys[base + 1];
    Type c = inTys[base + 2];
    if (a != expectAddr)
      return emitOpError("schedule mismatch with port kind: ")
             << portKindMsg << "; store port #" << j << " addr type " << a
             << " (expected " << expectAddr << ")";
    if (d != expectData)
      return emitOpError(
                 "store data port width mismatch with memref_mgr element "
                 "width; store port #")
             << j << " data type " << d << " (expected " << expectData
             << " from memref_mgr element width " << wMgr << ")";
    if (c != expectCtrl)
      return emitOpError("schedule mismatch with port kind: ")
             << portKindMsg << "; store port #" << j << " ctrl type " << c
             << " (expected " << expectCtrl << ")";
  }

  // Load ports (results).
  for (int64_t i = 0; i < loadN; ++i) {
    unsigned base = subIdx + 2 * (unsigned)i;
    Type d = outTys[base];
    Type dn = outTys[base + 1];
    if (d != expectData)
      return emitOpError(
                 "load data port width mismatch with memref_mgr element "
                 "width; load port #")
             << i << " data type " << d << " (expected " << expectData
             << " from memref_mgr element width " << wMgr << ")";
    if (dn != expectCtrl)
      return emitOpError("schedule mismatch with port kind: ")
             << portKindMsg << "; load port #" << i << " done type " << dn
             << " (expected " << expectCtrl << ")";
  }
  // Store ports (results: done only).
  for (int64_t j = 0; j < storeN; ++j) {
    unsigned idx = subIdx + 2 * (unsigned)loadN + (unsigned)j;
    Type dn = outTys[idx];
    if (dn != expectCtrl)
      return emitOpError("schedule mismatch with port kind: ")
             << portKindMsg << "; store port #" << j << " done type " << dn
             << " (expected " << expectCtrl << ")";
  }

  // sw_configs: all-or-nothing on (addr_table, mem_enable).
  ArrayAttr addrTable = getAddrTableAttr();
  BoolAttr memEnable = getMemEnableAttr();
  if ((bool)addrTable != (bool)memEnable) {
    if (addrTable && !memEnable)
      return emitOpError(
          "all-or-nothing violation: 'addr_table' is present but "
          "'mem_enable' is missing");
    return emitOpError("all-or-nothing violation: 'mem_enable' is present but "
                       "'addr_table' is missing");
  }
  if (!addrTable)
    return success();

  // Programmed branch.
  unsigned addrBits = ::fabric::resolveLoomAddrBits(*this);
  unsigned busWidth = ::fabric::resolveLoomMemBusWidth(*this);
  unsigned maxLog2 = floorLog2((uint64_t)busWidth / 8u);

  uint64_t expectedEntries =
      isTemporal ? (uint64_t)addrTableSize : (uint64_t)(loadN + storeN);
  if ((uint64_t)addrTable.size() != expectedEntries)
    return emitOpError("'addr_table' length ")
           << addrTable.size() << " must equal "
           << (isTemporal ? "addr_table_size ("
                          : "load_group_size + store_group_size (")
           << expectedEntries << ")";

  llvm::DenseSet<uint64_t> seenValidTags;
  for (size_t i = 0; i < addrTable.size(); ++i) {
    auto entry = dyn_cast<DictionaryAttr>(addrTable[i]);
    if (!entry)
      return emitOpError("'addr_table' entry #")
             << i << " must be a DictionaryAttr";

    auto baseAttr = entry.get("base_addr");
    auto elsAttr = entry.get("element_log2_size");
    auto validAttr = entry.get("valid");
    if (!baseAttr || !elsAttr || !validAttr)
      return emitOpError("'addr_table' entry #")
             << i
             << " must have keys 'base_addr', 'element_log2_size', and 'valid'";

    auto baseInt = dyn_cast<IntegerAttr>(baseAttr);
    if (!baseInt)
      return emitOpError("'addr_table' entry #")
             << i << " 'base_addr' must be an IntegerAttr";
    auto baseTy = dyn_cast<IntegerType>(baseInt.getType());
    if (!baseTy)
      return emitOpError("'addr_table' entry #")
             << i << " 'base_addr' must have IntegerType";
    if (baseTy.getWidth() != addrBits)
      return emitOpError("'base_addr' integer width ")
             << baseTy.getWidth() << " must equal loom_addr_bits (" << addrBits
             << ") (entry #" << i << ")";

    auto elsInt = dyn_cast<IntegerAttr>(elsAttr);
    if (!elsInt)
      return emitOpError("'addr_table' entry #")
             << i << " 'element_log2_size' must be an IntegerAttr";
    auto elsTy = dyn_cast<IntegerType>(elsInt.getType());
    if (!elsTy)
      return emitOpError("'addr_table' entry #")
             << i << " 'element_log2_size' must have IntegerType";
    if (elsTy.getWidth() != 4)
      return emitOpError("'element_log2_size' integer width ")
             << elsTy.getWidth() << " must equal 4 (entry #" << i << ")";
    // element_log2_size is treated as an unsigned 4-bit value.
    uint64_t elsVal = elsInt.getValue().getZExtValue();
    if (elsVal > maxLog2)
      return emitOpError("'element_log2_size' value ")
             << elsVal << " exceeds log2(loom_mem_bus_width / 8) = " << maxLog2
             << " (entry #" << i << ")";

    auto validBool = dyn_cast<BoolAttr>(validAttr);
    if (!validBool)
      return emitOpError("'addr_table' entry #")
             << i << " 'valid' must be a BoolAttr";

    if (isTemporal) {
      auto tagAttr = entry.get("tag");
      if (!tagAttr)
        return emitOpError("'addr_table' entry #")
               << i << " temporal mode requires 'tag' key";
      auto tagInt = dyn_cast<IntegerAttr>(tagAttr);
      if (!tagInt)
        return emitOpError("'addr_table' entry #")
               << i << " 'tag' must be an IntegerAttr";
      auto tagTy = dyn_cast<IntegerType>(tagInt.getType());
      if (!tagTy)
        return emitOpError("'addr_table' entry #")
               << i << " 'tag' must have IntegerType";
      if (tagTy.getWidth() != T)
        return emitOpError("'tag' integer width ")
               << tagTy.getWidth() << " must equal tag_width " << T
               << " (entry #" << i << ")";
      if (validBool.getValue()) {
        uint64_t key = tagInt.getValue().getZExtValue();
        if (!seenValidTags.insert(key).second)
          return emitOpError("temporal duplicate valid tag value ") << key;
      }
    } else {
      if (entry.get("tag"))
        return emitOpError("'addr_table' entry #")
               << i << " spatial mode must not carry 'tag' (temporal-only key)";
    }
  }

  return success();
}

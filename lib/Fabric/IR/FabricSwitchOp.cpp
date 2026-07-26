//===- FabricSwitchOp.cpp - Parser/printer/verifier for fabric.switch -----===//
//
// Implements parser, printer, and verifier for fabric.switch. The op is a
// leaf-level routing crossbar with a `[spatial]` or `[temporal]` schedule
// predicate, hardware parameters (connectivity_table, plus route_table_size
// for temporal) in `[ ... ]`, and software configuration (route_table,
// switch_enable) in `{ ... }`.
//
// Two disjoint syntactic forms exist:
//
//   Anonymous:  variadic SSA inputs + variadic SSA results.
//   Named:      zero SSA operands and zero SSA results; signature lives in
//               the `function_type` attribute (template, instantiated via
//               `fabric.instantiate`).
//
// See `docs/spec-fabric-switch.md` for the full per-schedule rules.
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/TemporalSwitchResourceContract.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <system_error>
#include <vector>

using namespace mlir;
using namespace fabric;

namespace {

LogicalResult
verifyRequesterSequence(llvm::function_ref<InFlightDiagnostic()> emitError,
                        DenseI64ArrayAttr requesters,
                        std::optional<std::uint64_t> reset) {
  if (!requesters || requesters.empty())
    return emitError() << "switch grant policy requires a non-empty requester "
                          "sequence";
  llvm::DenseSet<std::uint64_t> seen;
  for (std::int64_t requester : requesters.asArrayRef()) {
    if (requester < 0 || static_cast<std::uint64_t>(requester) >
                             std::numeric_limits<std::uint32_t>::max())
      return emitError() << "switch requester ordinal is outside u32";
    if (!seen.insert(static_cast<std::uint64_t>(requester)).second)
      return emitError() << "switch grant policy repeats requester "
                         << requester;
  }
  if (reset && !seen.contains(*reset))
    return emitError() << "round-robin reset requester is absent from its "
                          "requester cycle";
  return success();
}

} // namespace

LogicalResult SwitchFixedPriorityAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    DenseI64ArrayAttr requesterOrder) {
  return verifyRequesterSequence(emitError, requesterOrder, std::nullopt);
}

LogicalResult
SwitchRoundRobinAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                             DenseI64ArrayAttr requesterCycle,
                             std::uint64_t resetRequester) {
  if (resetRequester > std::numeric_limits<std::uint32_t>::max())
    return emitError() << "round-robin reset requester is outside u32";
  return verifyRequesterSequence(emitError, requesterCycle, resetRequester);
}

//===----------------------------------------------------------------------===//
// fabric.switch: parser
//===----------------------------------------------------------------------===//

ParseResult SwitchOp::parse(OpAsmParser &parser, OperationState &result) {
  // Optional `@sym_name` immediately after the op keyword. When present
  // the parser switches to the template form (no SSA operands/results).
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
    return parser.emitError(scheduleLoc,
                            "expected fabric switch schedule keyword "
                            "'spatial' or 'temporal', got '")
           << scheduleKw << "'";
  result.addAttribute("schedule", ScheduleAttr::get(parser.getContext(), *sym));

  if (isNamed) {
    // `(<input-types>) -> (<result-types>)` builds function_type.
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
  } else {
    // Anonymous form: SSA operand list, then `-> result-types`, then types
    // are taken from the trailing functional-type signature.
    SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
    SMLoc operandsLoc = parser.getCurrentLocation();
    OpAsmParser::UnresolvedOperand first;
    OptionalParseResult firstParse = parser.parseOptionalOperand(first);
    if (firstParse.has_value()) {
      if (failed(*firstParse))
        return failure();
      operands.push_back(first);
      while (succeeded(parser.parseOptionalComma())) {
        OpAsmParser::UnresolvedOperand op;
        if (parser.parseOperand(op))
          return failure();
        operands.push_back(op);
      }
    }

    // Parse optional `[ ... ]` hw_params.
    if (succeeded(parser.parseOptionalLSquare())) {
      // Re-parse the bracketed array attribute. We've already consumed `[`,
      // so read the inner array elements as DictionaryAttr's separated by
      // commas, then `]`.
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

    // Parse optional `{ ... }` sw_configs.
    {
      DictionaryAttr d;
      OptionalParseResult opr = parser.parseOptionalAttribute(d);
      if (opr.has_value()) {
        if (failed(*opr))
          return failure();
        result.addAttribute("sw_configs", d);
      }
    }

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
      return failure();

    if (parser.parseColon())
      return failure();
    SmallVector<Type, 4> sourceTypes;
    SmallVector<Type, 4> inputPortTypes;
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
      result.getOrAddProperties<Properties>().setInnerInputTypes(
          inputPortTypes);
    }
    result.addTypes(resultTypes);
    return success();
  }

  // Named form: parse optional `[ hw_params ]` and `{ sw_configs }`.
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
      result.addAttribute("sw_configs", d);
    }
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// fabric.switch: printer
//===----------------------------------------------------------------------===//

void SwitchOp::print(OpAsmPrinter &p) {
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
    p << ' ';
    llvm::interleaveComma(getInputs(), p, [&](Value v) { p << v; });
  }

  // hw_params in `[ ... ]`.
  if (auto hp = getHwParamsAttr()) {
    p << ' ' << '[';
    llvm::interleaveComma(hp, p, [&](Attribute a) { p.printAttribute(a); });
    p << ']';
  }

  // sw_configs in `{ ... }`.
  if (auto sw = getSwConfigsAttr()) {
    p << ' ';
    p.printAttribute(sw);
  }

  SmallVector<StringRef, 6> elided{"schedule",      "sym_name",
                                   "function_type", "inner_input_types",
                                   "hw_params",     "sw_configs"};
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(), elided);

  if (!isNamed) {
    ArrayRef<Type> innerTypes = getInnerInputTypes();
    SmallVector<Type, 4> inputPortTypes;
    inputPortTypes.reserve(getInputs().size());
    if (!innerTypes.empty() && innerTypes.size() == getInputs().size()) {
      inputPortTypes.append(innerTypes.begin(), innerTypes.end());
    } else {
      for (Value input : getInputs())
        inputPortTypes.push_back(input.getType());
    }
    p << " : (";
    llvm::interleaveComma(
        llvm::zip(getInputs(), inputPortTypes), p, [&](auto pair) {
          Value input;
          Type inputPortType;
          std::tie(input, inputPortType) = pair;
          p << input.getType();
          if (inputPortType && inputPortType != input.getType())
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

bool SwitchOp::isOptionalSymbol() { return true; }

//===----------------------------------------------------------------------===//
// fabric.switch: verifier helpers
//===----------------------------------------------------------------------===//

namespace {

// Reads connectivity_table from hw_params (length-1 array of dictionary).
// On success returns the ArrayAttr of L StringAttrs. The hwDict output
// argument receives the underlying DictionaryAttr (caller may inspect
// further keys, e.g. route_table_size).
static LogicalResult readHwParamsDict(SwitchOp op, DictionaryAttr &outDict) {
  auto hp = op.getHwParamsAttr();
  if (!hp)
    return op.emitOpError("requires 'hw_params' with 'connectivity_table'");
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

// Validate connectivity_table: ArrayAttr of L StringAttrs of length K of
// '0'/'1' characters with each row having at least one '1' and each column
// having at least one '1'. Returns the parsed table on success.
static LogicalResult
verifyConnectivityTable(SwitchOp op, DictionaryAttr hwDict, unsigned K,
                        unsigned L, SmallVectorImpl<StringRef> &outRows) {
  auto attr = hwDict.get("connectivity_table");
  if (!attr)
    return op.emitOpError("requires 'hw_params' with 'connectivity_table'");
  auto arr = dyn_cast<ArrayAttr>(attr);
  if (!arr)
    return op.emitOpError("'connectivity_table' must be an ArrayAttr");
  if (arr.size() != L)
    return op.emitOpError("'connectivity_table' length ")
           << arr.size() << " must equal L (" << L << ")";
  outRows.clear();
  outRows.reserve(L);
  for (size_t j = 0; j < arr.size(); ++j) {
    auto s = dyn_cast<StringAttr>(arr[j]);
    if (!s)
      return op.emitOpError("'connectivity_table' row #")
             << j << " must be a StringAttr";
    StringRef row = s.getValue();
    if (row.size() != K)
      return op.emitOpError("'connectivity_table' row #")
             << j << " length " << row.size() << " must equal K (" << K << ")";
    bool sawOne = false;
    for (char c : row) {
      if (c != '0' && c != '1')
        return op.emitOpError("'connectivity_table' row #")
               << j << " contains non-'0'/'1' character";
      if (c == '1')
        sawOne = true;
    }
    if (!sawOne)
      return op.emitOpError("'connectivity_table' row #")
             << j
             << " must have at least one '1' (each output needs at least one "
                "physical input source)";
    outRows.push_back(row);
  }
  // Per-column: at least one '1' across the L rows.
  // Bit-string convention: MSB on the left. Column index k maps to
  // string position (K - 1 - k). Per-column existence does not depend
  // on the convention; we just check each character position k_str in
  // [0, K).
  for (unsigned kStr = 0; kStr < K; ++kStr) {
    bool sawOne = false;
    for (StringRef row : outRows)
      if (row[kStr] == '1') {
        sawOne = true;
        break;
      }
    // Translate string index to column (input-port) index for the
    // diagnostic. MSB-on-left -> input port index = K - 1 - kStr.
    unsigned colIdx = K - 1 - kStr;
    if (!sawOne)
      return op.emitOpError("'connectivity_table' column #")
             << colIdx
             << " must have at least one '1' (each input needs at least one "
                "physical destination)";
  }
  return success();
}

// Count the '1's in a connectivity row.
static unsigned popcountOnes(StringRef row) {
  unsigned n = 0;
  for (char c : row)
    if (c == '1')
      ++n;
  return n;
}

static LogicalResult
collectAnonymousInputPortTypes(SwitchOp op,
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
    if (isa<MemRefType>(sourceType) || isa<MemRefType>(inputPortType))
      return op.emitOpError("incoming connection operand #")
             << i
             << ": memref capabilities cannot use the 'to "
                "<destination-type>' clause or serve as switch transport "
                "ports";
    if (!haveSameFabricModulePortKind(sourceType, inputPortType))
      return op.emitOpError("incoming connection operand #")
             << i << " source type " << sourceType
             << " and destination port type " << inputPortType
             << " must share the same fabric kind (bits or bits_tag)";
  }
  return success();
}

// Verify a single route_sel ArrayAttr against the connectivity rows. Each
// row j has length popcount(connRow[j]); each row has at most one '1',
// and each '1' bit position must align with a '1' in connectivity_table[j]
// (at the same character index, since route_sel uses MSB-on-left over the
// physically-connected subset).
//
// Mapping: the bit at position p (counted from the right, i.e. bit p of
// the route_sel string at character index (popcount-1-p)) selects the
// p-th connected input on row j. A '1' bit position counts the connected
// inputs from the right of the connectivity row (low-index-first); we
// require that the route_sel string is the same length as popcount(connRow)
// and that any '1' in route_sel maps back to a '1' in connectivity row at
// the corresponding shifted character index.
static LogicalResult verifyRouteTableRows(SwitchOp op, ArrayAttr routeArr,
                                          ArrayRef<StringRef> connRows,
                                          StringRef diagPrefix = StringRef()) {
  if (routeArr.size() != connRows.size())
    return op.emitOpError(diagPrefix)
           << "'route_table' length " << routeArr.size() << " must equal L ("
           << connRows.size() << ")";
  for (size_t j = 0; j < routeArr.size(); ++j) {
    auto s = dyn_cast<StringAttr>(routeArr[j]);
    if (!s)
      return op.emitOpError(diagPrefix)
             << "'route_table' row #" << j << " must be a StringAttr";
    StringRef row = s.getValue();
    unsigned ones = popcountOnes(connRows[j]);
    if (row.size() != ones)
      return op.emitOpError(diagPrefix)
             << "'route_table' row #" << j << " length " << row.size()
             << " must equal '1'-count of connectivity_table row #" << j << " ("
             << ones << ")";
    unsigned routeOnes = 0;
    int onePos = -1;
    for (size_t i = 0; i < row.size(); ++i) {
      char c = row[i];
      if (c != '0' && c != '1')
        return op.emitOpError(diagPrefix) << "'route_table' row #" << j
                                          << " contains non-'0'/'1' character";
      if (c == '1') {
        ++routeOnes;
        onePos = (int)i;
      }
    }
    if (routeOnes > 1)
      return op.emitOpError(diagPrefix)
             << "spatial route_table row has '1' count > 1 (row #" << j << ")";
    if (routeOnes == 1) {
      // Bit position p (right-counted) = (row.size() - 1 - onePos).
      unsigned pRight = (unsigned)((int)row.size() - 1 - onePos);
      // Walk the connectivity row right-to-left, counting '1's; the
      // pRight-th '1' encountered marks the connectivity character index
      // we expect to be '1' (it is, by construction), and its bit-position
      // (right-counted) is what we report on errors.
      // Per the spec: "route_sel bit position only legal where
      // connectivity has '1' at the corresponding position". If pRight
      // exceeds the number of '1's in conn row, that's already caught by
      // the length check above. Compute the connectivity-row character
      // index this bit corresponds to:
      unsigned seen = 0;
      int connStrIdx = -1;
      for (int ci = (int)connRows[j].size() - 1; ci >= 0; --ci) {
        if (connRows[j][ci] == '1') {
          if (seen == pRight) {
            connStrIdx = ci;
            break;
          }
          ++seen;
        }
      }
      // Sanity: by construction connStrIdx must be valid (length check).
      // The "bit position" reported in diagnostics is the input-port
      // index = (K - 1 - connStrIdx).
      if (connStrIdx < 0 || connRows[j][connStrIdx] != '1') {
        unsigned portIdx = connStrIdx >= 0 ? (unsigned)connRows[j].size() - 1 -
                                                 (unsigned)connStrIdx
                                           : pRight;
        return op.emitOpError(diagPrefix)
               << "'route_table' row #" << j << " selects bit position "
               << portIdx << " but connectivity_table row #" << j
               << " has '0' there";
      }
    }
  }
  return success();
}

// Read the boundary types per form (anonymous: SSA operands/results;
// named: function_type). Returns K, L, and the operand/result type lists.
static LogicalResult collectShape(SwitchOp op, unsigned &K, unsigned &L,
                                  SmallVectorImpl<Type> &inTys,
                                  SmallVectorImpl<Type> &outTys,
                                  bool &isNamed) {
  isNamed = static_cast<bool>(op.getSymNameAttr());
  if (isNamed) {
    if (!op.getInnerInputTypes().empty())
      return op.emitOpError("named fabric.switch template must not carry '")
             << kInnerInputTypesPropertyName << "'";
    if (!op.getInputs().empty())
      return op.emitOpError(
                 "named fabric.switch template must have zero SSA operands; "
                 "got ")
             << op.getInputs().size();
    if (!op.getResultTypes().empty())
      return op.emitOpError(
                 "named fabric.switch template must have zero SSA results; "
                 "got ")
             << op.getResultTypes().size();
    auto fta = op.getFunctionTypeAttr();
    if (!fta)
      return op.emitOpError(
          "named fabric.switch template requires a 'function_type' attribute");
    auto ft = dyn_cast<FunctionType>(fta.getValue());
    if (!ft)
      return op.emitOpError("'function_type' attribute must be a FunctionType");
    inTys.assign(ft.getInputs().begin(), ft.getInputs().end());
    outTys.assign(ft.getResults().begin(), ft.getResults().end());
  } else {
    if (op.getFunctionTypeAttr())
      return op.emitOpError(
          "anonymous fabric.switch must not carry a 'function_type' "
          "attribute");
    if (failed(collectAnonymousInputPortTypes(op, inTys)))
      return failure();
    for (Type t : op.getResultTypes())
      outTys.push_back(t);
  }
  K = (unsigned)inTys.size();
  L = (unsigned)outTys.size();
  if (K == 0)
    return op.emitOpError("requires at least 1 input port (K >= 1)");
  if (L == 0)
    return op.emitOpError("requires at least 1 output port (L >= 1)");
  return success();
}

// Spatial: all ports must be !fabric.bits<W> with uniform W.
static LogicalResult verifySpatialPorts(SwitchOp op, ArrayRef<Type> inTys,
                                        ArrayRef<Type> outTys, unsigned &W) {
  auto firstBits = dyn_cast<BitsType>(inTys[0]);
  if (!firstBits) {
    if (isa<BitsTagType>(inTys[0]))
      return op.emitOpError(
                 "schedule mismatch with port kind: spatial fabric.switch "
                 "requires '!fabric.bits<W>' ports; input #0 has type ")
             << inTys[0];
    return op.emitOpError(
               "schedule mismatch with port kind: spatial fabric.switch "
               "requires '!fabric.bits<W>' ports; input #0 has type ")
           << inTys[0];
  }
  W = firstBits.getWidth();
  for (auto [i, t] : llvm::enumerate(inTys)) {
    auto b = dyn_cast<BitsType>(t);
    if (!b) {
      if (isa<BitsTagType>(t))
        return op.emitOpError(
                   "schedule mismatch with port kind: spatial fabric.switch "
                   "requires '!fabric.bits<W>' ports; input #")
               << i << " has type " << t;
      return op.emitOpError(
                 "schedule mismatch with port kind: spatial fabric.switch "
                 "requires '!fabric.bits<W>' ports; input #")
             << i << " has type " << t;
    }
    if (b.getWidth() != W)
      return op.emitOpError(
                 "requires uniform 'bits<W>' on all switch ports; input #")
             << i << " has type " << t << " (expected '!fabric.bits<" << W
             << ">')";
  }
  for (auto [i, t] : llvm::enumerate(outTys)) {
    auto b = dyn_cast<BitsType>(t);
    if (!b) {
      if (isa<BitsTagType>(t))
        return op.emitOpError(
                   "schedule mismatch with port kind: spatial fabric.switch "
                   "requires '!fabric.bits<W>' ports; output #")
               << i << " has type " << t;
      return op.emitOpError(
                 "schedule mismatch with port kind: spatial fabric.switch "
                 "requires '!fabric.bits<W>' ports; output #")
             << i << " has type " << t;
    }
    if (b.getWidth() != W)
      return op.emitOpError(
                 "requires uniform 'bits<W>' on all switch ports; output #")
             << i << " has type " << t << " (expected '!fabric.bits<" << W
             << ">')";
  }
  return success();
}

// Temporal: all ports must be !fabric.bits_tag<W, T> with uniform (W, T).
static LogicalResult verifyTemporalPorts(SwitchOp op, ArrayRef<Type> inTys,
                                         ArrayRef<Type> outTys, unsigned &W,
                                         unsigned &T) {
  auto firstTag = dyn_cast<BitsTagType>(inTys[0]);
  if (!firstTag)
    return op.emitOpError(
               "schedule mismatch with port kind: temporal fabric.switch "
               "requires '!fabric.bits_tag<W, T>' ports; input #0 has type ")
           << inTys[0];
  W = firstTag.getWidth();
  T = firstTag.getTagWidth();
  for (auto [i, t] : llvm::enumerate(inTys)) {
    auto tag = dyn_cast<BitsTagType>(t);
    if (!tag)
      return op.emitOpError(
                 "schedule mismatch with port kind: temporal fabric.switch "
                 "requires '!fabric.bits_tag<W, T>' ports; input #")
             << i << " has type " << t;
    if (tag.getWidth() != W || tag.getTagWidth() != T)
      return op.emitOpError(
                 "requires uniform 'bits_tag<W, T>' on all switch ports; "
                 "input #")
             << i << " has type " << t << " (expected '!fabric.bits_tag<" << W
             << ", " << T << ">')";
  }
  for (auto [i, t] : llvm::enumerate(outTys)) {
    auto tag = dyn_cast<BitsTagType>(t);
    if (!tag)
      return op.emitOpError(
                 "schedule mismatch with port kind: temporal fabric.switch "
                 "requires '!fabric.bits_tag<W, T>' ports; output #")
             << i << " has type " << t;
    if (tag.getWidth() != W || tag.getTagWidth() != T)
      return op.emitOpError(
                 "requires uniform 'bits_tag<W, T>' on all switch ports; "
                 "output #")
             << i << " has type " << t << " (expected '!fabric.bits_tag<" << W
             << ", " << T << ">')";
  }
  return success();
}

// Reject any temporal-only key on a spatial switch.
static LogicalResult verifySpatialNoTemporalKeys(SwitchOp op,
                                                 DictionaryAttr hwDict) {
  if (hwDict.get("route_table_size"))
    return op.emitOpError(
        "spatial fabric.switch must not carry temporal-only attribute "
        "'route_table_size'");
  if (hwDict.get(kSwitchGrantPolicyParameterName))
    return op.emitOpError(
               "spatial fabric.switch must not carry temporal-only attribute '")
           << kSwitchGrantPolicyParameterName << "'";
  return success();
}

// Verify the all-or-nothing rule on (route_table, switch_enable).
static LogicalResult verifyAllOrNothing(SwitchOp op, DictionaryAttr swDict,
                                        bool &programmed) {
  Attribute rt = swDict ? swDict.get("route_table") : Attribute();
  Attribute en = swDict ? swDict.get("switch_enable") : Attribute();
  if (!rt && !en) {
    programmed = false;
    return success();
  }
  if (rt && !en)
    return op.emitOpError(
        "all-or-nothing violation: 'route_table' is present but "
        "'switch_enable' is missing");
  if (en && !rt)
    return op.emitOpError(
        "all-or-nothing violation: 'switch_enable' is present but "
        "'route_table' is missing");
  if (!isa<BoolAttr>(en))
    return op.emitOpError("'switch_enable' must be a BoolAttr");
  programmed = true;
  return success();
}

} // namespace

llvm::Expected<TemporalSwitchResourceContract>
fabric::deriveTemporalSwitchResourceContract(SwitchOp operation) {
  auto invalid = [](const llvm::Twine &message)
      -> llvm::Expected<TemporalSwitchResourceContract> {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "invalid temporal fabric.switch resource projection: " + message);
  };
  if (operation.getSchedule() != Schedule::Temporal)
    return invalid("operation is not temporal");

  std::uint64_t inputCount = operation.getNumOperands();
  std::uint64_t outputCount = operation.getNumResults();
  if (operation.getSymNameAttr()) {
    auto functionType = operation.getFunctionTypeAttr();
    auto type = functionType ? dyn_cast<FunctionType>(functionType.getValue())
                             : FunctionType();
    if (!type)
      return invalid("named operation has no function type");
    inputCount = type.getNumInputs();
    outputCount = type.getNumResults();
  }
  if (inputCount == 0 || outputCount == 0 ||
      inputCount > std::numeric_limits<std::uint32_t>::max() ||
      outputCount > std::numeric_limits<std::uint32_t>::max())
    return invalid("port domain is empty or exceeds u32");

  ArrayAttr parameters = operation.getHwParamsAttr();
  auto hardware = parameters && parameters.size() == 1
                      ? dyn_cast<DictionaryAttr>(parameters[0])
                      : DictionaryAttr();
  if (!hardware)
    return invalid("hardware parameters are malformed");
  auto connectivity =
      dyn_cast_or_null<ArrayAttr>(hardware.get("connectivity_table"));
  if (!connectivity || connectivity.size() != outputCount)
    return invalid("connectivity does not cover the output domain");

  std::vector<std::vector<std::uint32_t>> sourcesByOutput;
  sourcesByOutput.reserve(connectivity.size());
  for (Attribute rowAttribute : connectivity) {
    auto row = dyn_cast<StringAttr>(rowAttribute);
    if (!row || row.getValue().size() != inputCount)
      return invalid("connectivity row is malformed");
    std::vector<std::uint32_t> sources;
    for (std::uint32_t input = 0; input != inputCount; ++input) {
      const char selected = row.getValue()[inputCount - 1 - input];
      if (selected != '0' && selected != '1')
        return invalid("connectivity row has a non-binary entry");
      if (selected == '1')
        sources.push_back(input);
    }
    sourcesByOutput.push_back(std::move(sources));
  }

  std::optional<TemporalSwitchGrantPolicy> policy;
  Attribute policyAttribute = hardware.get(kSwitchGrantPolicyParameterName);
  auto decodeRequesters = [&](DenseI64ArrayAttr values)
      -> llvm::Expected<std::vector<std::uint32_t>> {
    std::vector<std::uint32_t> result;
    result.reserve(values.size());
    for (std::int64_t requester : values.asArrayRef()) {
      if (requester < 0 || static_cast<std::uint64_t>(requester) >
                               std::numeric_limits<std::uint32_t>::max())
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "switch requester is outside u32");
      result.push_back(static_cast<std::uint32_t>(requester));
    }
    return result;
  };
  if (auto fixed = dyn_cast_or_null<SwitchFixedPriorityAttr>(policyAttribute)) {
    auto requesters = decodeRequesters(fixed.getRequesterOrder());
    if (!requesters)
      return requesters.takeError();
    policy = TemporalSwitchFixedPriority{std::move(*requesters)};
  } else if (auto roundRobin =
                 dyn_cast_or_null<SwitchRoundRobinAttr>(policyAttribute)) {
    auto requesters = decodeRequesters(roundRobin.getRequesterCycle());
    if (!requesters)
      return requesters.takeError();
    if (roundRobin.getResetRequester() >
        std::numeric_limits<std::uint32_t>::max())
      return invalid("round-robin reset requester is outside u32");
    policy = TemporalSwitchRoundRobin{
        std::move(*requesters),
        static_cast<std::uint32_t>(roundRobin.getResetRequester())};
  } else if (policyAttribute) {
    return invalid("grant policy has an unknown typed variant");
  }

  return TemporalSwitchResourceContract::create(
      {static_cast<std::uint32_t>(inputCount),
       static_cast<std::uint32_t>(outputCount), std::move(sourcesByOutput),
       std::move(policy)});
}

//===----------------------------------------------------------------------===//
// fabric.switch: verifier
//===----------------------------------------------------------------------===//

LogicalResult SwitchOp::verify() {
  if (failed(verifyInnerInputTypesProperty(getOperation(), getInputs(),
                                           getInnerInputTypes())))
    return failure();

  unsigned K = 0, L = 0;
  bool isNamed = false;
  SmallVector<Type, 4> inTys, outTys;
  if (failed(collectShape(*this, K, L, inTys, outTys, isNamed)))
    return failure();

  // Read hw_params (mandatory).
  DictionaryAttr hwDict;
  if (failed(readHwParamsDict(*this, hwDict)))
    return failure();

  // Schedule-keyed port type-kind checks + uniform width(s).
  unsigned W = 0, T = 0;
  if (getSchedule() == Schedule::Spatial) {
    if (failed(verifySpatialPorts(*this, inTys, outTys, W)))
      return failure();
    if (failed(verifySpatialNoTemporalKeys(*this, hwDict)))
      return failure();
  } else {
    if (failed(verifyTemporalPorts(*this, inTys, outTys, W, T)))
      return failure();
  }

  // connectivity_table.
  SmallVector<StringRef, 4> connRows;
  if (failed(verifyConnectivityTable(*this, hwDict, K, L, connRows)))
    return failure();

  // route_table_size: required for temporal, forbidden for spatial.
  uint64_t routeTableSize = 0;
  if (getSchedule() == Schedule::Temporal) {
    auto rtsAttr = hwDict.get("route_table_size");
    if (!rtsAttr)
      return emitOpError(
          "temporal fabric.switch requires 'route_table_size' attribute");
    auto rtsInt = dyn_cast<IntegerAttr>(rtsAttr);
    if (!rtsInt)
      return emitOpError("'route_table_size' must be an IntegerAttr");
    int64_t rts = rtsInt.getValue().getSExtValue();
    if (rts < 1)
      return emitOpError("'route_table_size' must be >= 1, got ") << rts;
    routeTableSize = (uint64_t)rts;
  }

  // sw_configs (optional). Apply all-or-nothing rule.
  DictionaryAttr swDict = getSwConfigsAttr();
  bool programmed = false;
  if (failed(verifyAllOrNothing(*this, swDict, programmed)))
    return failure();
  if (!programmed)
    return success();

  // Programmed branch.
  if (getSchedule() == Schedule::Spatial) {
    auto rtAttr = swDict.get("route_table");
    auto rtArr = dyn_cast<ArrayAttr>(rtAttr);
    if (!rtArr)
      return emitOpError("'route_table' must be an ArrayAttr");
    if (failed(verifyRouteTableRows(*this, rtArr, connRows)))
      return failure();
  } else {
    // Temporal: route_table is an ArrayAttr of route_table_size dicts.
    auto rtAttr = swDict.get("route_table");
    auto rtArr = dyn_cast<ArrayAttr>(rtAttr);
    if (!rtArr)
      return emitOpError("'route_table' must be an ArrayAttr");
    if (rtArr.size() != routeTableSize)
      return emitOpError("'route_table' length ")
             << rtArr.size() << " must equal 'route_table_size' ("
             << routeTableSize << ")";
    llvm::DenseSet<uint64_t> seenValidTags;
    for (size_t i = 0; i < rtArr.size(); ++i) {
      auto entry = dyn_cast<DictionaryAttr>(rtArr[i]);
      if (!entry)
        return emitOpError("'route_table' entry #")
               << i << " must be a DictionaryAttr";
      auto routeSelAttr = entry.get("route_sel");
      auto tagAttr = entry.get("tag");
      auto validAttr = entry.get("valid");
      if (!routeSelAttr || !tagAttr || !validAttr)
        return emitOpError("'route_table' entry #")
               << i << " must have keys 'route_sel', 'tag', and 'valid'";
      auto routeSelArr = dyn_cast<ArrayAttr>(routeSelAttr);
      if (!routeSelArr)
        return emitOpError("'route_table' entry #")
               << i << " 'route_sel' must be an ArrayAttr";
      // Reuse spatial-row verifier for route_sel.
      if (failed(verifyRouteTableRows(*this, routeSelArr, connRows)))
        return failure();

      auto tagInt = dyn_cast<IntegerAttr>(tagAttr);
      if (!tagInt)
        return emitOpError("'route_table' entry #")
               << i << " 'tag' must be an IntegerAttr";
      auto tagTy = dyn_cast<IntegerType>(tagInt.getType());
      if (!tagTy)
        return emitOpError("'route_table' entry #")
               << i << " 'tag' must have IntegerType";
      if (tagTy.getWidth() != T)
        return emitOpError("'tag' integer width ")
               << tagTy.getWidth() << " must equal port tag-width " << T
               << " (entry #" << i << ")";
      auto validBool = dyn_cast<BoolAttr>(validAttr);
      if (!validBool)
        return emitOpError("'route_table' entry #")
               << i << " 'valid' must be a BoolAttr";

      uint64_t tagVal = tagInt.getValue().getZExtValue();
      if (validBool.getValue()) {
        if (!seenValidTags.insert(tagVal).second)
          return emitOpError("temporal duplicate valid tag value ") << tagVal;
      }
    }
  }
  return success();
}

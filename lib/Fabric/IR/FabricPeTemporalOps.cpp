//===- FabricPeTemporalOps.cpp - Verifier for fabric.pe [temporal] -------===//
//
// Implements the temporal-schedule branch of fabric.pe: the eight hardware
// parameters, the three software-configuration attributes (pe_enable,
// instruction_mem, per_fu_sw_configs), and the per-instruction-entry
// validation. Spatial-side rules and the parser/printer for fabric.pe are
// in FabricOps.cpp.
//
// The temporal PE boundary is uniformly !fabric.bits_tag<W, T>. Inner FUs
// still operate on un-tagged !fabric.bits<W>; the tag is stripped at the
// PE boundary via the existing 'to <inner-type>' clause on the PE operand
// list (anonymous form).
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace fabric;

namespace fabric {

// Forward declaration. Defined below.
LogicalResult verifyPeTemporal(PeOp op);
LogicalResult verifyPeSpatialNoTemporalAttrs(PeOp op);

} // namespace fabric

namespace {

// A temporal-only attribute name; used both to reject unexpected presence
// on spatial PEs and to drive the per-attribute checks on temporal PEs.
struct TemporalAttrName {
  StringRef name;
};

static const TemporalAttrName kTemporalAttrs[] = {
    {"tag_width"},          {"num_instruction"}, {"num_reg_fifo"},
    {"reg_fifo_depth"},     {"reg_fifo_ports"},  {"fu_config_mode"},
    {"operand_buffer_mode"},{"operand_buffer_size"},
    {"pe_enable"},          {"instruction_mem"}, {"per_fu_sw_configs"},
};

// Get an integer-typed attribute as int64, or std::nullopt if the attr
// is missing or not an IntegerAttr.
static std::optional<int64_t> getOptInt(Operation *op, StringRef name) {
  auto a = op->getAttrOfType<IntegerAttr>(name);
  if (!a)
    return std::nullopt;
  return a.getInt();
}

// Compute log2Ceil(n). Returns 0 when n <= 1 (matches MLIR's llvm::Log2_64_Ceil
// semantics for n == 1).
static unsigned log2Ceil(uint64_t n) {
  if (n <= 1)
    return 0;
  return llvm::Log2_64_Ceil(n);
}

// Required keys per operand_sel / result_sel entry. Returns the missing
// key name on failure, empty StringRef on success.
static StringRef checkSelEntryKeys(DictionaryAttr d, bool isOperand) {
  static const StringRef kCommon[] = {"tag", "is_port", "discard", "disconnect"};
  for (StringRef k : kCommon)
    if (!d.get(k))
      return k;
  if (isOperand) {
    if (!d.get("src_sel"))
      return StringRef("src_sel");
  } else {
    if (!d.get("dst_sel"))
      return StringRef("dst_sel");
  }
  return StringRef();
}

// Pull a BoolAttr value; returns std::nullopt if missing/non-bool.
static std::optional<bool> getBool(DictionaryAttr d, StringRef key) {
  auto a = d.get(key);
  if (!a)
    return std::nullopt;
  auto b = dyn_cast<BoolAttr>(a);
  if (!b)
    return std::nullopt;
  return b.getValue();
}

// Pull an integer attribute from a Dictionary as int64. Treats the stored
// APInt as unsigned (boolean and small bitwidth integer attributes default
// to signless storage; getInt() sign-extends, which would yield -1 for an
// i1 true). Returns the unsigned interpretation.
static std::optional<int64_t> getInt(DictionaryAttr d, StringRef key) {
  auto a = d.get(key);
  if (!a)
    return std::nullopt;
  auto i = dyn_cast<IntegerAttr>(a);
  if (!i)
    return std::nullopt;
  return static_cast<int64_t>(i.getValue().getZExtValue());
}

// Verify a single operand_sel / result_sel entry. `selBound` is the upper
// bound (exclusive) for src_sel/dst_sel when is_port is true (= K or L).
// `numRegFifo` is num_reg_fifo (the bound when is_port is false).
static LogicalResult verifySelEntry(PeOp op, DictionaryAttr d, unsigned instIdx,
                                    unsigned entryIdx, bool isOperand,
                                    unsigned selBound, unsigned numRegFifo) {
  StringRef kind = isOperand ? "operand_sel" : "result_sel";
  StringRef selKey = isOperand ? "src_sel" : "dst_sel";

  StringRef missing = checkSelEntryKeys(d, isOperand);
  if (!missing.empty())
    return op.emitOpError("instruction[")
           << instIdx << "] " << kind << "[" << entryIdx
           << "] is missing required key '" << missing << "'";

  auto isPort = getBool(d, "is_port");
  if (!isPort)
    return op.emitOpError("instruction[")
           << instIdx << "] " << kind << "[" << entryIdx
           << "] 'is_port' must be a BoolAttr";
  auto discard = getBool(d, "discard");
  if (!discard)
    return op.emitOpError("instruction[")
           << instIdx << "] " << kind << "[" << entryIdx
           << "] 'discard' must be a BoolAttr";
  auto disconnect = getBool(d, "disconnect");
  if (!disconnect)
    return op.emitOpError("instruction[")
           << instIdx << "] " << kind << "[" << entryIdx
           << "] 'disconnect' must be a BoolAttr";
  if (*discard && *disconnect)
    return op.emitOpError("instruction[")
           << instIdx << "] " << kind << "[" << entryIdx
           << "] cannot have both 'discard' and 'disconnect' true";

  auto sel = getInt(d, selKey);
  if (!sel)
    return op.emitOpError("instruction[")
           << instIdx << "] " << kind << "[" << entryIdx << "] '" << selKey
           << "' must be an IntegerAttr";
  if (*sel < 0)
    return op.emitOpError("instruction[")
           << instIdx << "] " << kind << "[" << entryIdx << "] '" << selKey
           << "' must be >= 0";

  if (*isPort) {
    if (static_cast<unsigned>(*sel) >= selBound)
      return op.emitOpError("instruction[")
             << instIdx << "] " << kind << "[" << entryIdx << "] '" << selKey
             << "' (" << *sel << ") must be < "
             << (isOperand ? "K (" : "L (") << selBound << ")";
  } else {
    if (numRegFifo == 0)
      return op.emitOpError("instruction[")
             << instIdx << "] " << kind << "[" << entryIdx
             << "] uses 'is_port' = false but 'num_reg_fifo' is 0";
    if (static_cast<unsigned>(*sel) >= numRegFifo)
      return op.emitOpError("instruction[")
             << instIdx << "] " << kind << "[" << entryIdx << "] '" << selKey
             << "' (" << *sel << ") must be < num_reg_fifo (" << numRegFifo
             << ")";
  }

  // Tag presence and integer-attr type are checked but the tag bit-width is
  // not strictly enforced here; the configuration generator emits a
  // T-bit-wide field. Only require that the tag is an IntegerAttr.
  auto tag = d.get("tag");
  if (!tag || !isa<IntegerAttr>(tag))
    return op.emitOpError("instruction[")
           << instIdx << "] " << kind << "[" << entryIdx
           << "] 'tag' must be an IntegerAttr";
  return success();
}

// String constants for the two temporal-PE keyword attributes.
static constexpr StringRef kFuCfgPerInstr = "per_instruction_fu_config";
static constexpr StringRef kFuCfgPerFu = "per_fu_config";
static constexpr StringRef kBufPerInstr = "per_instruction";
static constexpr StringRef kBufPerInputPort = "per_input_port";
static constexpr StringRef kBufAllFuShare = "all_fu_share";

static LogicalResult
verifyInstructionEntry(PeOp op, unsigned instIdx, DictionaryAttr d,
                       unsigned numFu, unsigned K, unsigned L,
                       unsigned maxFuInputs, unsigned maxFuOutputs,
                       unsigned numRegFifo, StringRef fuCfgMode) {
  // enable
  auto enable = getBool(d, "enable");
  if (!enable)
    return op.emitOpError("instruction[")
           << instIdx << "] is missing 'enable' (BoolAttr)";

  // opcode: present, < num_fu.
  auto opcode = getInt(d, "opcode");
  if (!opcode)
    return op.emitOpError("instruction[")
           << instIdx << "] is missing 'opcode' (IntegerAttr)";
  if (*opcode < 0 || static_cast<uint64_t>(*opcode) >= numFu)
    return op.emitOpError("instruction[")
           << instIdx << "] 'opcode' (" << *opcode << ") must be < num_fu ("
           << numFu << ")";

  // operand_sel: array, length == max_fu_inputs.
  auto opSelArr = dyn_cast_or_null<ArrayAttr>(d.get("operand_sel"));
  if (!opSelArr)
    return op.emitOpError("instruction[")
           << instIdx << "] 'operand_sel' must be an ArrayAttr";
  if (opSelArr.size() != maxFuInputs)
    return op.emitOpError("instruction[")
           << instIdx << "] 'operand_sel' length (" << opSelArr.size()
           << ") must equal max_fu_inputs (" << maxFuInputs << ")";
  for (unsigned i = 0; i < opSelArr.size(); ++i) {
    auto entry = dyn_cast<DictionaryAttr>(opSelArr[i]);
    if (!entry)
      return op.emitOpError("instruction[")
             << instIdx << "] operand_sel[" << i
             << "] must be a DictionaryAttr";
    if (failed(verifySelEntry(op, entry, instIdx, i, /*isOperand=*/true, K,
                              numRegFifo)))
      return failure();
  }

  // result_sel: array, length == max_fu_outputs.
  auto resSelArr = dyn_cast_or_null<ArrayAttr>(d.get("result_sel"));
  if (!resSelArr)
    return op.emitOpError("instruction[")
           << instIdx << "] 'result_sel' must be an ArrayAttr";
  if (resSelArr.size() != maxFuOutputs)
    return op.emitOpError("instruction[")
           << instIdx << "] 'result_sel' length (" << resSelArr.size()
           << ") must equal max_fu_outputs (" << maxFuOutputs << ")";
  for (unsigned i = 0; i < resSelArr.size(); ++i) {
    auto entry = dyn_cast<DictionaryAttr>(resSelArr[i]);
    if (!entry)
      return op.emitOpError("instruction[")
             << instIdx << "] result_sel[" << i
             << "] must be a DictionaryAttr";
    if (failed(verifySelEntry(op, entry, instIdx, i, /*isOperand=*/false, L,
                              numRegFifo)))
      return failure();
  }

  // fu_sw_configs (per_instruction_fu_config only).
  if (fuCfgMode == kFuCfgPerInstr) {
    auto fucfg = d.get("fu_sw_configs");
    if (!fucfg)
      return op.emitOpError("instruction[")
             << instIdx
             << "] is missing 'fu_sw_configs' (required for "
                "'per_instruction_fu_config')";
    if (!isa<DictionaryAttr>(fucfg))
      return op.emitOpError("instruction[")
             << instIdx << "] 'fu_sw_configs' must be a DictionaryAttr";
  } else {
    if (d.get("fu_sw_configs"))
      return op.emitOpError("instruction[")
             << instIdx
             << "] must not carry 'fu_sw_configs' when 'fu_config_mode' is "
                "'per_fu_config'";
  }
  // Suppress unused-warning when only opcode is consumed.
  (void)log2Ceil;
  return success();
}

// Compute (K, L, numFu, maxFuInputs, maxFuOutputs) for the temporal PE.
// Also returns the body's compute count (FUs + instantiates) via numFu;
// fails if the body is empty.
static LogicalResult collectTemporalShape(PeOp op, unsigned &K, unsigned &L,
                                          unsigned &numFu,
                                          unsigned &maxFuInputs,
                                          unsigned &maxFuOutputs) {
  Block &entry = op.getBody().front();
  bool isNamed = static_cast<bool>(op.getSymNameAttr());

  if (isNamed) {
    auto fta = op.getFunctionTypeAttr();
    if (!fta)
      return op.emitOpError(
          "named fabric.pe template requires a 'function_type' attribute");
    auto ft = dyn_cast<FunctionType>(fta.getValue());
    if (!ft)
      return op.emitOpError("'function_type' attribute must be a FunctionType");
    K = ft.getNumInputs();
    L = ft.getNumResults();
  } else {
    K = entry.getNumArguments();
    L = op.getOutputs().size();
  }

  numFu = 0;
  maxFuInputs = 0;
  maxFuOutputs = 0;
  for (Operation &op2 : entry) {
    if (auto inst = dyn_cast<InstantiateOp>(op2)) {
      ++numFu;
      // Best-effort shape: assume 0 inputs / 0 outputs unless we resolve
      // the callee. The instantiate verifier already validates the
      // callee shape; for max_fu_inputs / max_fu_outputs we use the
      // visible operand and result counts here.
      maxFuInputs = std::max(maxFuInputs, (unsigned)inst.getInputs().size());
      maxFuOutputs = std::max(maxFuOutputs, (unsigned)inst.getOutputs().size());
      continue;
    }
    if (isa<YieldOp>(op2))
      continue;
    auto fu = dyn_cast<FuOp>(op2);
    if (!fu)
      continue; // body whitelist enforced separately
    ++numFu;
    unsigned ins, outs;
    if (fu.getSymNameAttr()) {
      auto fta = fu.getFunctionTypeAttr();
      if (!fta)
        continue;
      auto ft = dyn_cast<FunctionType>(fta.getValue());
      if (!ft)
        continue;
      ins = ft.getNumInputs();
      outs = ft.getNumResults();
    } else {
      ins = fu.getInputs().size();
      outs = fu.getOutputs().size();
    }
    maxFuInputs = std::max(maxFuInputs, ins);
    maxFuOutputs = std::max(maxFuOutputs, outs);
  }
  return success();
}

// Body whitelist + boundary type checks for the temporal PE. Returns W
// and T extracted from the first PE port; sets isNamed.
static LogicalResult
verifyTemporalBoundaryAndBody(PeOp op, unsigned &W, unsigned &T,
                              bool &isNamed) {
  isNamed = static_cast<bool>(op.getSymNameAttr());
  Block &entry = op.getBody().front();

  SmallVector<Type, 4> declaredIns;
  SmallVector<Type, 4> declaredOuts;

  if (isNamed) {
    if (!op.getInputs().empty())
      return op.emitOpError(
                 "named fabric.pe template must have zero SSA operands; got ")
             << op.getInputs().size();
    if (!op.getResultTypes().empty())
      return op.emitOpError(
                 "named fabric.pe template must have zero SSA results; got ")
             << op.getResultTypes().size();
    auto fta = op.getFunctionTypeAttr();
    if (!fta)
      return op.emitOpError(
          "named fabric.pe template requires a 'function_type' attribute");
    auto ft = dyn_cast<FunctionType>(fta.getValue());
    if (!ft)
      return op.emitOpError("'function_type' attribute must be a FunctionType");
    declaredIns.assign(ft.getInputs().begin(), ft.getInputs().end());
    declaredOuts.assign(ft.getResults().begin(), ft.getResults().end());

    // Block-arg types must equal declaredIns (named form).
    if (entry.getNumArguments() != declaredIns.size())
      return op.emitOpError("entry block argument count (")
             << entry.getNumArguments()
             << ") must match declared input count (" << declaredIns.size()
             << ")";
    for (auto [i, pair] :
         llvm::enumerate(llvm::zip(entry.getArguments(), declaredIns))) {
      BlockArgument bb;
      Type t;
      std::tie(bb, t) = pair;
      if (bb.getType() != t)
        return op.emitOpError("entry block argument #")
               << i << " type " << bb.getType()
               << " must equal declared input type " << t;
    }
  } else {
    if (op.getFunctionTypeAttr())
      return op.emitOpError(
          "anonymous fabric.pe must not carry a 'function_type' attribute");

    if (entry.getNumArguments() != op.getInputs().size())
      return op.emitOpError("region entry block argument count (")
             << entry.getNumArguments() << ") must equal operand count ("
             << op.getInputs().size() << ")";
    // For temporal PE, the SSA outer types are the boundary types, NOT the
    // block-arg types (because the anonymous-form 'to' clause may strip the
    // tag at the FU side).
    for (Type t : op.getInputs().getTypes())
      declaredIns.push_back(t);
    for (Type t : op.getOutputs().getTypes())
      declaredOuts.push_back(t);
  }

  if (declaredIns.empty())
    return op.emitOpError("requires at least 1 input port (K >= 1)");
  if (declaredOuts.empty())
    return op.emitOpError("requires at least 1 output port (L >= 1)");

  // PE boundary type: must be uniformly !fabric.bits_tag<W, T>.
  auto firstTag = dyn_cast<BitsTagType>(declaredIns[0]);
  if (!firstTag)
    return op.emitOpError(
               "temporal fabric.pe boundary type must be "
               "'!fabric.bits_tag<W, T>'; PE input #0 has type ")
           << declaredIns[0];
  W = firstTag.getWidth();
  T = firstTag.getTagWidth();
  for (auto [i, t] : llvm::enumerate(declaredIns)) {
    auto tag = dyn_cast<BitsTagType>(t);
    if (!tag)
      return op.emitOpError(
                 "temporal fabric.pe boundary type must be "
                 "'!fabric.bits_tag<W, T>'; PE input #")
             << i << " has type " << t;
    if (tag.getWidth() != W || tag.getTagWidth() != T)
      return op.emitOpError(
                 "requires uniform 'bits_tag<W, T>' on all PE ports; PE "
                 "input #")
             << i << " has type " << t << " (expected '!fabric.bits_tag<" << W
             << ", " << T << ">')";
  }
  for (auto [i, t] : llvm::enumerate(declaredOuts)) {
    auto tag = dyn_cast<BitsTagType>(t);
    if (!tag)
      return op.emitOpError(
                 "temporal fabric.pe boundary type must be "
                 "'!fabric.bits_tag<W, T>'; PE output #")
             << i << " has type " << t;
    if (tag.getWidth() != W || tag.getTagWidth() != T)
      return op.emitOpError(
                 "requires uniform 'bits_tag<W, T>' on all PE ports; PE "
                 "output #")
             << i << " has type " << t << " (expected '!fabric.bits_tag<" << W
             << ", " << T << ">')";
  }

  // Anonymous-form 'to' clause relaxation: outer type bits_tag<W,T> with
  // inner block-arg type bits<W> is allowed (tag stripping). Otherwise
  // outer must equal inner.
  if (!isNamed) {
    for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
      Type outerTy = op.getInputs()[i].getType();
      Type innerTy = arg.getType();
      if (outerTy == innerTy)
        continue;
      auto outerTag = dyn_cast<BitsTagType>(outerTy);
      auto innerBits = dyn_cast<BitsType>(innerTy);
      if (outerTag && innerBits && outerTag.getWidth() == innerBits.getWidth())
        continue;
      return op.emitOpError("operand #")
             << i << " outer type " << outerTy
             << " and PE block-arg inner type " << innerTy
             << " are not a valid temporal-PE boundary pair (allowed: equal "
                "types or 'bits_tag<W, T> to bits<W>')";
    }
  }

  // Body whitelist: fabric.fu, fabric.instantiate; named form additionally
  // permits fabric.yield.
  unsigned numCompute = 0;
  for (Operation &op2 : entry) {
    if (isa<InstantiateOp>(op2)) {
      ++numCompute;
      continue;
    }
    if (isa<YieldOp>(op2)) {
      if (!isNamed)
        return op2.emitOpError(
            "fabric.yield is not allowed in an anonymous fabric.pe body");
      continue;
    }
    auto fu = dyn_cast<FuOp>(op2);
    if (!fu)
      return op2.emitOpError(
                 "'fabric.pe' op body may only contain "
                 "fabric.fu and fabric.instantiate; got '")
             << op2.getName().getStringRef() << "'";
    ++numCompute;

    // Per-FU shape constraints (use K and L bounds).
    unsigned fuNumIns, fuNumOuts;
    SmallVector<Type, 4> fuIns, fuOuts;
    if (fu.getSymNameAttr()) {
      auto fta = fu.getFunctionTypeAttr();
      if (!fta)
        continue;
      auto ft = dyn_cast<FunctionType>(fta.getValue());
      if (!ft)
        continue;
      fuNumIns = ft.getNumInputs();
      fuNumOuts = ft.getNumResults();
      fuIns.assign(ft.getInputs().begin(), ft.getInputs().end());
      fuOuts.assign(ft.getResults().begin(), ft.getResults().end());
    } else {
      fuNumIns = fu.getInputs().size();
      fuNumOuts = fu.getOutputs().size();
      for (Type t : fu.getInputs().getTypes())
        fuIns.push_back(t);
      for (Type t : fu.getOutputs().getTypes())
        fuOuts.push_back(t);
    }
    if (fuNumIns > declaredIns.size())
      return fu.emitOpError("inner fabric.fu has ")
             << fuNumIns << " inputs which exceeds fabric.pe input count K="
             << declaredIns.size();
    if (fuNumOuts > declaredOuts.size())
      return fu.emitOpError("inner fabric.fu has ")
             << fuNumOuts
             << " outputs which exceeds fabric.pe output count L="
             << declaredOuts.size();
    // FU outer input ports may be bits<W> (un-tagged) or bits_tag<W, T>
    // (un-stripped, when wired directly from the named-form PE block-arg
    // and stripped at the FU's own 'to' clause). Outputs are strict
    // bits<W>: the FU's hardware emits un-tagged data and the temporal
    // PE re-tags at its boundary.
    for (auto [i, t] : llvm::enumerate(fuIns)) {
      unsigned w = 0;
      bool ok = false;
      if (auto bw = dyn_cast<BitsType>(t)) {
        w = bw.getWidth();
        ok = true;
      } else if (auto tg = dyn_cast<BitsTagType>(t)) {
        w = tg.getWidth();
        ok = (tg.getTagWidth() == T);
      }
      if (!ok || w != W)
        return fu.emitOpError(
                   "inner fabric.fu boundary width must equal fabric.pe "
                   "data width W=")
               << W << "; FU input #" << i << " has type " << t;
    }
    for (auto [i, t] : llvm::enumerate(fuOuts)) {
      auto bw = dyn_cast<BitsType>(t);
      if (!bw || bw.getWidth() != W)
        return fu.emitOpError(
                   "inner fabric.fu boundary width must equal fabric.pe "
                   "data width W=")
               << W << "; FU output #" << i << " has type " << t;
    }
  }
  if (numCompute < 1)
    return op.emitOpError(
        "body requires at least one fabric.fu or fabric.instantiate");

  // Named form: body must end with fabric.yield matching declared results.
  if (isNamed) {
    if (entry.empty() || !isa<YieldOp>(entry.back()))
      return op.emitOpError(
          "named fabric.pe body must terminate with fabric.yield");
    auto yield = cast<YieldOp>(entry.back());
    if (yield.getValues().size() != declaredOuts.size())
      return op.emitOpError("yield value count (")
             << yield.getValues().size()
             << ") must match declared result count (" << declaredOuts.size()
             << ")";
    for (auto [i, pair] :
         llvm::enumerate(llvm::zip(yield.getValues(), declaredOuts))) {
      Value v;
      Type t;
      std::tie(v, t) = pair;
      if (v.getType() != t)
        return op.emitOpError("yield value #")
               << i << " type " << v.getType()
               << " must equal declared result type " << t;
    }
  }
  return success();
}

static LogicalResult verifyTemporalHwParams(PeOp op, unsigned T) {
  // 1. tag_width: required, >= 1, and must equal the boundary T.
  auto tagWidth = getOptInt(op, "tag_width");
  if (!tagWidth)
    return op.emitOpError("temporal fabric.pe requires 'tag_width' attribute");
  if (*tagWidth < 1)
    return op.emitOpError("'tag_width' must be >= 1, got ") << *tagWidth;
  if (static_cast<unsigned>(*tagWidth) != T)
    return op.emitOpError("'tag_width' attribute (")
           << *tagWidth << ") must equal PE boundary tag width T (" << T << ")";

  // 2. num_instruction: required, >= 1.
  auto numInst = getOptInt(op, "num_instruction");
  if (!numInst)
    return op.emitOpError(
        "temporal fabric.pe requires 'num_instruction' attribute");
  if (*numInst < 1)
    return op.emitOpError("'num_instruction' must be >= 1, got ") << *numInst;

  // 3. num_reg_fifo: optional, defaults to 0.
  auto numRegFifo = getOptInt(op, "num_reg_fifo");
  int64_t nrf = numRegFifo.value_or(0);
  if (nrf < 0)
    return op.emitOpError("'num_reg_fifo' must be >= 0, got ") << nrf;

  // 4. reg_fifo_depth: required iff num_reg_fifo > 0.
  auto regDepth = getOptInt(op, "reg_fifo_depth");
  if (nrf > 0) {
    if (!regDepth)
      return op.emitOpError("'reg_fifo_depth' is required when 'num_reg_fifo' "
                            "> 0");
    if (*regDepth < 1)
      return op.emitOpError("'reg_fifo_depth' must be >= 1, got ") << *regDepth;
  } else {
    if (regDepth && *regDepth != 0)
      return op.emitOpError(
          "'reg_fifo_depth' must be absent (or 0) when 'num_reg_fifo' is 0");
  }

  // 5. reg_fifo_ports: optional, defaults to 1; must be 1 or 2; ignored when
  //    num_reg_fifo == 0.
  auto regPorts = getOptInt(op, "reg_fifo_ports");
  if (regPorts) {
    if (*regPorts != 1 && *regPorts != 2)
      return op.emitOpError("'reg_fifo_ports' must be 1 or 2, got ")
             << *regPorts;
  }

  // 6. fu_config_mode: required, must be one of the known keywords.
  auto fuCfgAttr = op.getFuConfigModeAttr();
  if (!fuCfgAttr)
    return op.emitOpError(
        "temporal fabric.pe requires 'fu_config_mode' attribute");
  StringRef fuCfgMode = fuCfgAttr.getValue();
  if (fuCfgMode != kFuCfgPerInstr && fuCfgMode != kFuCfgPerFu)
    return op.emitOpError("'fu_config_mode' must be one of '")
           << kFuCfgPerInstr << "' or '" << kFuCfgPerFu << "', got '"
           << fuCfgMode << "'";

  // 7. operand_buffer_mode: required, must be one of the known keywords.
  auto bufModeAttr = op.getOperandBufferModeAttr();
  if (!bufModeAttr)
    return op.emitOpError(
        "temporal fabric.pe requires 'operand_buffer_mode' attribute");
  StringRef bufMode = bufModeAttr.getValue();
  if (bufMode != kBufPerInstr && bufMode != kBufPerInputPort &&
      bufMode != kBufAllFuShare)
    return op.emitOpError("'operand_buffer_mode' must be one of '")
           << kBufPerInstr << "', '" << kBufPerInputPort << "', or '"
           << kBufAllFuShare << "', got '" << bufMode << "'";

  // 8. operand_buffer_size: required iff mode != per_instruction.
  auto bufSize = getOptInt(op, "operand_buffer_size");
  if (bufMode == kBufPerInstr) {
    if (bufSize)
      return op.emitOpError(
          "'operand_buffer_size' must be absent when 'operand_buffer_mode' is "
          "'per_instruction'");
  } else {
    if (!bufSize)
      return op.emitOpError(
          "'operand_buffer_size' is required when 'operand_buffer_mode' is "
          "not 'per_instruction'");
    if (*bufSize < 1)
      return op.emitOpError("'operand_buffer_size' must be >= 1, got ")
             << *bufSize;
  }
  return success();
}

// Verify the trio (pe_enable, instruction_mem, per_fu_sw_configs).
static LogicalResult verifyTemporalSwConfigs(PeOp op, unsigned numFu,
                                             unsigned K, unsigned L,
                                             unsigned maxFuInputs,
                                             unsigned maxFuOutputs) {
  bool hasEnable = static_cast<bool>(op.getPeEnableAttr());
  bool hasInstMem = static_cast<bool>(op.getInstructionMemAttr());
  StringRef fuCfgMode = op.getFuConfigModeAttr().getValue();
  bool needsPerFu = (fuCfgMode == kFuCfgPerFu);
  bool hasPerFu = static_cast<bool>(op.getPerFuSwConfigsAttr());

  if (!hasEnable && !hasInstMem && !hasPerFu) {
    // Fully unprogrammed (hardware-only): nothing more to check here.
    return success();
  }

  // All-or-nothing rule. The "trio" is { pe_enable, instruction_mem, and
  // (when needsPerFu) per_fu_sw_configs }. When fu_config_mode is
  // 'per_instruction_fu_config', per_fu_sw_configs must be absent.
  if (hasEnable && !hasInstMem)
    return op.emitOpError(
        "all-or-nothing violation: 'pe_enable' is present but "
        "'instruction_mem' is missing");
  if (hasInstMem && !hasEnable)
    return op.emitOpError(
        "all-or-nothing violation: 'instruction_mem' is present but "
        "'pe_enable' is missing");
  if (needsPerFu) {
    if (hasInstMem && !hasPerFu)
      return op.emitOpError(
          "all-or-nothing violation: 'instruction_mem' is present but "
          "'per_fu_sw_configs' is missing (required by 'per_fu_config' "
          "mode)");
    if (hasPerFu && !hasInstMem)
      return op.emitOpError(
          "all-or-nothing violation: 'per_fu_sw_configs' is present but "
          "'instruction_mem' is missing");
  } else {
    if (hasPerFu)
      return op.emitOpError(
          "'per_fu_sw_configs' must be absent when 'fu_config_mode' is "
          "'per_instruction_fu_config'");
  }

  // From here, the PE is "programmed": validate instruction_mem and
  // per_fu_sw_configs.
  auto instArr = op.getInstructionMemAttr();
  auto numInst = op.getNumInstructionAttr().getInt();
  if (instArr.size() != static_cast<size_t>(numInst))
    return op.emitOpError("'instruction_mem' length (")
           << instArr.size() << ") must equal 'num_instruction' (" << numInst
           << ")";
  for (unsigned i = 0; i < instArr.size(); ++i) {
    auto entry = dyn_cast<DictionaryAttr>(instArr[i]);
    if (!entry)
      return op.emitOpError("instruction[")
             << i << "] must be a DictionaryAttr";
    if (failed(verifyInstructionEntry(op, i, entry, numFu, K, L, maxFuInputs,
                                      maxFuOutputs,
                                      op.getNumRegFifoAttr()
                                          ? op.getNumRegFifoAttr().getInt()
                                          : 0,
                                      fuCfgMode)))
      return failure();
  }

  if (needsPerFu) {
    auto pfArr = op.getPerFuSwConfigsAttr();
    if (pfArr.size() != numFu)
      return op.emitOpError("'per_fu_sw_configs' length (")
             << pfArr.size() << ") must equal num_fu (" << numFu << ")";
    for (unsigned i = 0; i < pfArr.size(); ++i) {
      if (!isa<DictionaryAttr>(pfArr[i]))
        return op.emitOpError("'per_fu_sw_configs'[")
               << i << "] must be a DictionaryAttr";
    }
  }
  return success();
}

} // namespace

namespace fabric {

LogicalResult verifyPeTemporal(PeOp op) {
  unsigned W = 0, T = 0;
  bool isNamed = false;
  if (failed(verifyTemporalBoundaryAndBody(op, W, T, isNamed)))
    return failure();
  if (failed(verifyTemporalHwParams(op, T)))
    return failure();

  unsigned K = 0, L = 0, numFu = 0, maxFuInputs = 0, maxFuOutputs = 0;
  if (failed(collectTemporalShape(op, K, L, numFu, maxFuInputs, maxFuOutputs)))
    return failure();
  if (numFu == 0)
    return op.emitOpError(
        "body requires at least one fabric.fu or fabric.instantiate");

  if (failed(verifyTemporalSwConfigs(op, numFu, K, L, maxFuInputs,
                                     maxFuOutputs)))
    return failure();
  return success();
}

LogicalResult verifyPeSpatialNoTemporalAttrs(PeOp op) {
  for (const auto &t : kTemporalAttrs) {
    if (op->getAttr(t.name))
      return op.emitOpError("spatial fabric.pe must not carry temporal-only "
                            "attribute '")
             << t.name << "'";
  }
  return success();
}

} // namespace fabric

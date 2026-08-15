//===- FabricPeTemporalOps.cpp - Verifier for fabric.pe [temporal] -------===//
//
// Implements the temporal-schedule branch of fabric.pe: its typed hardware
// parameters, boundary shape, and body constraints. Spatial-side rules and
// the parser/printer for fabric.pe are in FabricOps.cpp.
//
// The temporal PE boundary is uniformly !fabric.bits_tag<W, T>. Inner FUs
// still operate on un-tagged !fabric.bits<W>; the tag is stripped at the
// PE boundary via the existing 'to <inner-type>' clause on the PE operand
// list (anonymous form).
//
//===----------------------------------------------------------------------===//

#include "Fabric/IR/FabricOps.h"

#include "Fabric/IR/Crosspoint.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"
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
    {"operand_buffer_mode"}, {"operand_buffer_size"},
};

static constexpr StringRef kSelectedConfigurationAttrs[] = {
    "pe_enable", "instruction_mem", "per_fu_sw_configs"};

static LogicalResult verifyNoSelectedConfigurationAttrs(PeOp op) {
  for (StringRef name : kSelectedConfigurationAttrs)
    if (op->hasAttr(name))
      return op.emitOpError("selected configuration attribute '")
             << name
             << "' is forbidden; ConfigurationImage owns selected "
                "instruction state";
  return success();
}

// Get an integer-typed attribute as int64, or std::nullopt if the attr
// is missing or not an IntegerAttr.
static std::optional<int64_t> getOptInt(Operation *op, StringRef name) {
  auto a = op->getAttrOfType<IntegerAttr>(name);
  if (!a)
    return std::nullopt;
  return a.getInt();
}

// Body whitelist + boundary type checks for the temporal PE. Returns T, K,
// and L extracted from the declared boundary.
static LogicalResult
verifyTemporalBoundaryAndBody(PeOp op, unsigned &T, unsigned &K,
                              unsigned &L) {
  const bool isNamed = static_cast<bool>(op.getSymNameAttr());
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

    // Named-temporal PE: entry block args must be bits<F> with F <= W,
    // where W is the bits-data part of the corresponding port type
    // bits_tag<W, T>. The check is run after the per-port bits_tag check
    // below.
    if (entry.getNumArguments() != declaredIns.size())
      return op.emitOpError("entry block argument count (")
             << entry.getNumArguments()
             << ") must match declared input count (" << declaredIns.size()
             << ")";
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
  K = declaredIns.size();
  L = declaredOuts.size();

  // PE boundary type: must be uniformly !fabric.bits_tag<W, T>.
  auto firstTag = dyn_cast<BitsTagType>(declaredIns[0]);
  if (!firstTag)
    return op.emitOpError(
               "temporal fabric.pe boundary type must be "
               "'!fabric.bits_tag<W, T>'; PE input #0 has type ")
           << declaredIns[0];
  const unsigned W = firstTag.getWidth();
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

  // Inner block-arg types are auto-tag-stripped to !fabric.bits<W'> with
  // W' <= W (the bits-data part of the corresponding port type). Both
  // anonymous and named forms apply the same rule; the anonymous form
  // additionally exposes an explicit 'to bits<F>' override (F <= W) for
  // narrower inner widths. The implicit default is W' = W.
  for (auto [i, arg] : llvm::enumerate(entry.getArguments())) {
    Type innerTy = arg.getType();
    auto innerBits = dyn_cast<BitsType>(innerTy);
    if (!innerBits) {
      if (isNamed)
        return op.emitOpError("named PE entry block arg #")
               << i << " type " << innerTy
               << " is bits_tag (forbidden); entry block args must be "
                  "!fabric.bits<W'> with W' <= port bits-data-width "
                  "(implicit tag-strip at the temporal-PE boundary)";
      return op.emitOpError("anonymous PE block-arg #")
             << i << " inner type " << innerTy
             << " must be !fabric.bits<W'> (the temporal-PE boundary "
                "auto-strips the tag; bits_tag inner types are forbidden)";
    }
    auto portTag = cast<BitsTagType>(declaredIns[i]);
    if (innerBits.getWidth() > portTag.getWidth()) {
      if (isNamed)
        return op.emitOpError("named PE entry block arg #")
               << i << " bits-width " << innerBits.getWidth()
               << " > port bits-data-width " << portTag.getWidth()
               << " (truncation only narrows)";
      return op.emitOpError("anonymous PE inner block arg #")
             << i << " width (" << innerBits.getWidth()
             << ") > outer bits part width (" << portTag.getWidth()
             << ") (truncation only narrows)";
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
    // FU outer input ports are strict !fabric.bits<F> with F <= W (the
    // tag is stripped at the PE-to-FU boundary). FU output ports are
    // strict !fabric.bits<W>: the temporal-PE result_sel.tag reattaches
    // the tag at the PE boundary.
    for (auto [i, t] : llvm::enumerate(fuIns)) {
      auto bw = dyn_cast<BitsType>(t);
      if (!bw)
        return fu.emitOpError(
                   "inner fabric.fu input port must be !fabric.bits<F>; "
                   "FU input #")
               << i << " has type " << t;
      if (bw.getWidth() > W)
        return fu.emitOpError("inner fabric.fu input #")
               << i << " width " << bw.getWidth()
               << " exceeds fabric.pe data width W=" << W;
    }
    for (auto [i, t] : llvm::enumerate(fuOuts)) {
      auto bw = dyn_cast<BitsType>(t);
      if (!bw || bw.getWidth() != W)
        return fu.emitOpError(
                   "inner fabric.fu output port width must equal "
                   "fabric.pe data width W=")
               << W << "; FU output #" << i << " has type " << t;
    }
  }
  if (numCompute < 1)
    return op.emitOpError(
        "body requires at least one fabric.fu or fabric.instantiate");

  // Named form: body closes with a zero-operand signature terminator. The
  // result ports (including their tags, reattached at the PE boundary by
  // the active ResultSelection) are owned by `function_type` alone; the
  // terminator restates them neither as values nor as declared types.
  if (isNamed) {
    YieldOp yield;
    if (!entry.empty())
      yield = dyn_cast<YieldOp>(entry.back());
    if (!yield || !yield.getValues().empty())
      return op.emitOpError(
          "named fabric.pe body must terminate with a zero-operand "
          "fabric.yield; 'function_type' alone owns the PE result ports");
    if (yield->hasAttr("declared_types"))
      return op.emitOpError(
          "named fabric.pe terminator must not carry a 'declared_types' "
          "attribute; 'function_type' alone owns the PE result ports");
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

  // 6. fu_config_mode: required typed enum.
  if (!op.getFuConfigModeAttr())
    return op.emitOpError(
        "temporal fabric.pe requires 'fu_config_mode' attribute");

  // 7. operand_buffer_mode: required typed enum.
  if (!op.getOperandBufferModeAttr())
    return op.emitOpError(
        "temporal fabric.pe requires 'operand_buffer_mode' attribute");

  // 8. operand_buffer_size: required and positive in every mode. It counts
  //    entries per mode-derived allocation unit, so a dedicated queue has no
  //    implicit depth and no mode carries a default.
  auto bufSize = getOptInt(op, "operand_buffer_size");
  if (!bufSize)
    return op.emitOpError("temporal fabric.pe requires 'operand_buffer_size' "
                          "in every 'operand_buffer_mode'");
  if (*bufSize < 1)
    return op.emitOpError("'operand_buffer_size' must be >= 1, got ")
           << *bufSize;
  return success();
}


} // namespace

namespace fabric {

LogicalResult verifyPeTemporal(PeOp op) {
  if (failed(verifyNoSelectedConfigurationAttrs(op)))
    return failure();

  unsigned T = 0, K = 0, L = 0;
  if (failed(verifyTemporalBoundaryAndBody(op, T, K, L)))
    return failure();
  if (failed(verifyTemporalHwParams(op, T)))
    return failure();

  auto crosspoints = validatedPeBoundaryCrosspointCount(K, L);
  if (!crosspoints)
    return op.emitOpError(llvm::toString(crosspoints.takeError()));
  if (*crosspoints > kPeCrosspointWarningThreshold)
    mlir::emitWarning(op.getLoc())
        << "fabric.pe boundary selectors have " << *crosspoints
        << " crosspoints; values above " << kPeCrosspointWarningThreshold
        << " may be implementation-inefficient";
  return success();
}

LogicalResult verifyPeSpatialNoTemporalAttrs(PeOp op) {
  if (failed(verifyNoSelectedConfigurationAttrs(op)))
    return failure();
  for (const auto &t : kTemporalAttrs) {
    if (op->getAttr(t.name))
      return op.emitOpError("spatial fabric.pe must not carry temporal-only "
                            "attribute '")
             << t.name << "'";
  }
  return success();
}

} // namespace fabric

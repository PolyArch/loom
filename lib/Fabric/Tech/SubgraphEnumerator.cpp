#include "Fabric/Tech/SubgraphEnumerator.h"

#include "Common/IndexWidth.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/Tech/SubgraphMatcher.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace fabric {
namespace {

using ::mlir::ArrayAttr;
using ::mlir::Attribute;
using ::mlir::Block;
using ::mlir::DictionaryAttr;
using ::mlir::FlatSymbolRefAttr;
using ::mlir::FloatType;
using ::mlir::FunctionType;
using ::mlir::IntegerType;
using ::mlir::Location;
using ::mlir::MLIRContext;
using ::mlir::NamedAttribute;
using ::mlir::NoneType;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::OperationState;
using ::mlir::SmallVector;
using ::mlir::StringAttr;
using ::mlir::StringRef;
using ::mlir::Type;
using ::mlir::Value;

//===----------------------------------------------------------------------===//
// Op flavor: how to lift fabric.bits<N> port widths to software types per op.
//===----------------------------------------------------------------------===//

enum class Flavor : uint8_t {
  IntArith,               // 2-int-in 1-int-out, no extra attrs
  IntCmp,                 // 2-int-in 1-i1-out, requires "predicate" sw_config
  FloatArith,             // 2-float-in 1-float-out, no extra attrs
  FloatCmp,               // 2-float-in 1-i1-out, requires "predicate" sw_config
  IntToFloat,             // 1-int-in 1-float-out
  FloatToInt,             // 1-float-in 1-int-out
  FloatUnary,             // 1-float-in 1-float-out (math.*)
  DataflowStreamFlavor,   // 3-int-in (T) 2-out (T,i1)
  DataflowCarryGateInv,   // dataflow.carry/invariant/gate, polymorphic
                          // shared port treated as integer
  DataflowConstantFlavor, // dataflow.constant: 1 none-in, 1 typed-out,
                          // const_hex_value sw_config materialized into
                          // an IntegerAttr (or FloatAttr when the result
                          // port is float-flavored).
  ArithSelect,            // arith.select: i1 sel + 2 data of type T -> T.
                          // Strict-SSA eager-evaluation semantics; not
                          // interchangeable with dataflow.mux.
  VariadicSyncFlavor,     // dataflow.sync: M-port hardware unit
                          // configured by a length-M bitmask whose
                          // popcount N picks the live ports.
  VariadicMuxFlavor,      // dataflow.mux: 1 sel + M data-port hardware
                          // unit; bitmask of length M with N ones
                          // picks the live data inputs.
  VariadicDemuxFlavor,    // dataflow.demux: 1 sel + 1 data-in + M
                          // data-out hardware unit; bitmask of length
                          // M with N ones picks the live outputs.
};

static bool isVariadicFlavor(Flavor f) {
  return f == Flavor::VariadicSyncFlavor || f == Flavor::VariadicMuxFlavor ||
         f == Flavor::VariadicDemuxFlavor;
}

static const llvm::StringMap<Flavor> &opFlavors() {
  static const llvm::StringMap<Flavor> m = []() {
    llvm::StringMap<Flavor> r;
    auto put = [&](StringRef n, Flavor f) { r.insert({n, f}); };

    // Integer arith
    for (StringRef n :
         {"arith.addi", "arith.subi", "arith.muli", "arith.divsi",
          "arith.divui", "arith.remsi", "arith.remui", "arith.shli",
          "arith.shrsi", "arith.shrui", "arith.andi", "arith.ori", "arith.xori",
          "arith.minsi", "arith.maxsi", "arith.minui", "arith.maxui"})
      put(n, Flavor::IntArith);
    put("arith.cmpi", Flavor::IntCmp);

    // Float arith
    for (StringRef n : {"arith.addf", "arith.subf", "arith.mulf", "arith.divf",
                        "arith.remf", "arith.minimumf", "arith.maximumf"})
      put(n, Flavor::FloatArith);
    put("arith.cmpf", Flavor::FloatCmp);

    // Int<->Float casts
    put("arith.sitofp", Flavor::IntToFloat);
    put("arith.uitofp", Flavor::IntToFloat);
    put("arith.fptosi", Flavor::FloatToInt);
    put("arith.fptoui", Flavor::FloatToInt);

    // Math unary
    for (StringRef n :
         {"math.sin",   "math.cos",       "math.tan",  "math.sinh",
          "math.cosh",  "math.tanh",      "math.exp",  "math.exp2",
          "math.expm1", "math.log",       "math.log2", "math.log10",
          "math.log1p", "math.floor",     "math.ceil", "math.round",
          "math.trunc", "math.roundeven", "math.sqrt", "math.rsqrt",
          "math.absf",  "llvm.intr.fabs", "math.erf"})
      put(n, Flavor::FloatUnary);
    put("math.absi", Flavor::IntArith);

    // Dataflow
    put("dataflow.stream", Flavor::DataflowStreamFlavor);
    put("dataflow.carry", Flavor::DataflowCarryGateInv);
    put("dataflow.invariant", Flavor::DataflowCarryGateInv);
    put("dataflow.gate", Flavor::DataflowCarryGateInv);
    put("dataflow.constant", Flavor::DataflowConstantFlavor);

    // Variadic dataflow ops: structural counts depend on the bitmask
    // sw_config picked by the enumerator. The enumerator iterates all
    // legal bitmasks (popcount in [1, M], capped at M <= 8) and emits
    // one materialized template per bitmask (post-dedup).
    put("dataflow.sync", Flavor::VariadicSyncFlavor);
    put("dataflow.mux", Flavor::VariadicMuxFlavor);
    put("dataflow.demux", Flavor::VariadicDemuxFlavor);

    // arith.select: fixed schema (i1 sel, T data, T data) -> T. Eager
    // strict-SSA semantics, distinct from dataflow.mux.
    put("arith.select", Flavor::ArithSelect);

    return r;
  }();
  return m;
}

static bool isMaterializable(StringRef opSym) {
  return opFlavors().count(opSym);
}

//===----------------------------------------------------------------------===//
// Type lifting helpers
//===----------------------------------------------------------------------===//

static Type intLifted(unsigned w, MLIRContext *ctx) {
  if (w == 0)
    return NoneType::get(ctx);
  return IntegerType::get(ctx, w);
}

static Type floatLifted(unsigned w, MLIRContext *ctx) {
  switch (w) {
  case 16:
    return ::mlir::Float16Type::get(ctx);
  case 32:
    return ::mlir::Float32Type::get(ctx);
  case 64:
    return ::mlir::Float64Type::get(ctx);
  case 80:
    return ::mlir::Float80Type::get(ctx);
  case 128:
    return ::mlir::Float128Type::get(ctx);
  default:
    return nullptr; // unsupported
  }
}

// Coarse "lift kind" for a bits<N> port: do we make it integer-like or
// floating-point? `Unknown` is listed first so that a default-constructed
// PortLift (returned by `DenseMap::lookup` for an absent key) compares
// equal to `Unknown` rather than to `Int`; without that, the propagation
// loops below silently treat unmapped values as "already Int" and
// short-circuit the float-lift inference path.
enum class PortLift : uint8_t { Unknown, Int, Float };

// Lift a bits<N> port width to the sw type expected for `flavor` at port
// position. `isOutput` tells us whether the port is on the result side.
// Returns nullptr on an unsupported width (e.g. float of an unusual width).
static Type liftFor(Flavor flavor, unsigned w, bool isOutput, unsigned portIdx,
                    MLIRContext *ctx) {
  switch (flavor) {
  case Flavor::IntArith:
    return intLifted(w, ctx);
  case Flavor::IntCmp:
    return isOutput ? IntegerType::get(ctx, 1) : intLifted(w, ctx);
  case Flavor::FloatArith:
    return floatLifted(w, ctx);
  case Flavor::FloatCmp:
    return isOutput ? IntegerType::get(ctx, 1) : floatLifted(w, ctx);
  case Flavor::IntToFloat:
    return isOutput ? floatLifted(w, ctx) : intLifted(w, ctx);
  case Flavor::FloatToInt:
    return isOutput ? intLifted(w, ctx) : floatLifted(w, ctx);
  case Flavor::FloatUnary:
    return floatLifted(w, ctx);
  case Flavor::DataflowStreamFlavor:
    if (isOutput && portIdx == 1)
      return IntegerType::get(ctx, 1); // rwc
    return intLifted(w, ctx);
  case Flavor::DataflowCarryGateInv: {
    // Heuristic: i1-typed ports map to i1, otherwise iN.
    if (w == 1)
      return IntegerType::get(ctx, 1);
    return intLifted(w, ctx);
  }
  case Flavor::DataflowConstantFlavor:
    // Input is the none-typed control token (bits<0> -> none).
    if (!isOutput)
      return NoneType::get(ctx);
    return intLifted(w, ctx);
  case Flavor::ArithSelect:
    // (i1 sel, T data, T data) -> T. Sel is fixed i1; the data ports
    // share T which we treat as integer here (the lift propagation
    // step refines it across mux/demux when relevant).
    if (!isOutput && portIdx == 0)
      return IntegerType::get(ctx, 1);
    return intLifted(w, ctx);
  case Flavor::VariadicSyncFlavor:
    // Variadic sync: every port is a data port whose lift kind comes
    // from neighboring ops via computePortLiftMap. Default to int.
    return intLifted(w, ctx);
  case Flavor::VariadicMuxFlavor:
    // The hardware-side sel port is at input index 0. Whether the
    // materialized sel block-arg ends up as i1 or index is decided
    // per-config based on the active count N (see variadicSelType
    // helpers below). At the FU lift level we just expose it as i1
    // when bits==1 and as the matching integer width otherwise.
    if (!isOutput && portIdx == 0) {
      // Preserve the hardware sel port width; per-config materialization
      // overrides the block-arg type to i1/index based on active N.
      if (w == 1)
        return IntegerType::get(ctx, 1);
      return intLifted(w, ctx);
    }
    return intLifted(w, ctx);
  case Flavor::VariadicDemuxFlavor:
    // sel at input index 0, data at input index 1; outputs are data.
    if (!isOutput && portIdx == 0) {
      if (w == 1)
        return IntegerType::get(ctx, 1);
      return intLifted(w, ctx);
    }
    return intLifted(w, ctx);
  }
  return nullptr;
}

// Per-port lift kind for a fabric.op flavor.
static PortLift portLiftKind(Flavor f, bool isOutput, unsigned idx) {
  switch (f) {
  case Flavor::IntArith:
  case Flavor::IntCmp:
  case Flavor::DataflowStreamFlavor:
  case Flavor::DataflowCarryGateInv:
  case Flavor::DataflowConstantFlavor:
    return PortLift::Int;
  case Flavor::FloatArith:
  case Flavor::FloatUnary:
    return PortLift::Float;
  case Flavor::FloatCmp:
    return isOutput ? PortLift::Int : PortLift::Float;
  case Flavor::IntToFloat:
    return isOutput ? PortLift::Float : PortLift::Int;
  case Flavor::FloatToInt:
    return isOutput ? PortLift::Int : PortLift::Float;
  case Flavor::ArithSelect:
    // sel is integer (i1); data ports are polymorphic. Leave the
    // data ports unknown so the propagation pass can fix them from
    // neighboring ops, while pinning sel as Int.
    return (!isOutput && idx == 0) ? PortLift::Int : PortLift::Unknown;
  case Flavor::VariadicSyncFlavor:
    return PortLift::Unknown;
  case Flavor::VariadicMuxFlavor:
    // Sel is integer; data ports are polymorphic.
    return (!isOutput && idx == 0) ? PortLift::Int : PortLift::Unknown;
  case Flavor::VariadicDemuxFlavor:
    return (!isOutput && idx == 0) ? PortLift::Int : PortLift::Unknown;
  }
  return PortLift::Unknown;
}

// Compute a per-fabric-SSA-value lift kind by tracing through fabric.op /
// fabric.mux / fabric.demux. Mux and demux preserve flavor; fabric.ops
// fix flavor via portLiftKind based on their op_list[0]'s flavor (for
// flavor-determination, all op_list members must share the same shape and
// thus the same lift kind per port).
static llvm::DenseMap<Value, PortLift> computePortLiftMap(FuOp fu) {
  llvm::DenseMap<Value, PortLift> m;
  Block &body = fu.getBody().front();

  auto setIfStronger = [&](Value v, PortLift k, bool &changed) {
    if (k == PortLift::Unknown)
      return;
    auto it = m.find(v);
    if (it == m.end()) {
      m[v] = k;
      changed = true;
    }
  };

  // Seed: every fabric.op's input/output gets its flavor-driven lift kind.
  for (Operation &op : body.without_terminator()) {
    if (auto fop = ::mlir::dyn_cast<::fabric::OpOp>(&op)) {
      if (fop.getOpList().empty())
        continue;
      auto sym = ::mlir::cast<FlatSymbolRefAttr>(fop.getOpList()[0]).getValue();
      Flavor f = opFlavors().lookup(sym);
      bool dummy = false;
      for (auto [i, in] : llvm::enumerate(fop.getInputs()))
        setIfStronger(in, portLiftKind(f, false, i), dummy);
      for (auto [i, out] : llvm::enumerate(fop.getOutputs()))
        setIfStronger(out, portLiftKind(f, true, i), dummy);
    }
  }

  // Propagate lift kinds across variadic data-only port-groups inside
  // fabric.ops. For dataflow.sync the pairs (input #i, output #i) share
  // a kind; for dataflow.mux all data inputs (input #1..#N-1) and the
  // single output share one data kind; for dataflow.demux the data
  // input (input #1) and every output share one data kind. Sel ports
  // (input #0 of mux/demux) are independent and pinned to Int.
  auto firstKnown = [&](::llvm::ArrayRef<Value> vs) {
    for (Value v : vs) {
      auto k = m.lookup(v);
      if (k != PortLift::Unknown)
        return k;
    }
    return PortLift::Unknown;
  };
  bool seedChanged = true;
  while (seedChanged) {
    seedChanged = false;
    for (Operation &op : body.without_terminator()) {
      auto fop = ::mlir::dyn_cast<::fabric::OpOp>(&op);
      if (!fop)
        continue;
      ArrayAttr opList = fop.getOpList();
      if (opList.empty())
        continue;
      auto sym = ::mlir::cast<FlatSymbolRefAttr>(opList[0]).getValue();
      Flavor f = opFlavors().lookup(sym);
      if (!isVariadicFlavor(f))
        continue;
      auto inputs = fop.getInputs();
      auto outputs = fop.getOutputs();
      if (f == Flavor::VariadicSyncFlavor) {
        unsigned n = std::min<unsigned>(inputs.size(), outputs.size());
        for (unsigned i = 0; i < n; ++i) {
          PortLift k = m.lookup(inputs[i]);
          if (k == PortLift::Unknown)
            k = m.lookup(outputs[i]);
          if (k == PortLift::Unknown)
            continue;
          setIfStronger(inputs[i], k, seedChanged);
          setIfStronger(outputs[i], k, seedChanged);
        }
      } else if (f == Flavor::VariadicMuxFlavor) {
        // Data ports: inputs[1..] and outputs[0].
        ::llvm::SmallVector<Value, 8> dataPorts;
        for (unsigned i = 1; i < inputs.size(); ++i)
          dataPorts.push_back(inputs[i]);
        for (Value v : outputs)
          dataPorts.push_back(v);
        PortLift k = firstKnown(dataPorts);
        if (k == PortLift::Unknown)
          continue;
        for (Value v : dataPorts)
          setIfStronger(v, k, seedChanged);
      } else if (f == Flavor::VariadicDemuxFlavor) {
        ::llvm::SmallVector<Value, 8> dataPorts;
        if (inputs.size() >= 2)
          dataPorts.push_back(inputs[1]);
        for (Value v : outputs)
          dataPorts.push_back(v);
        PortLift k = firstKnown(dataPorts);
        if (k == PortLift::Unknown)
          continue;
        for (Value v : dataPorts)
          setIfStronger(v, k, seedChanged);
      }
    }
  }

  // Iterate to propagate through mux / demux until fixed point.
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : body.without_terminator()) {
      if (auto mux = ::mlir::dyn_cast<::fabric::MuxOp>(&op)) {
        PortLift k = m.lookup(mux.getOutput());
        if (k == PortLift::Unknown)
          for (Value v : mux.getInputs()) {
            auto kk = m.lookup(v);
            if (kk != PortLift::Unknown) {
              k = kk;
              break;
            }
          }
        if (k == PortLift::Unknown)
          continue;
        setIfStronger(mux.getOutput(), k, changed);
        for (Value v : mux.getInputs())
          setIfStronger(v, k, changed);
      } else if (auto dem = ::mlir::dyn_cast<::fabric::DemuxOp>(&op)) {
        PortLift k = m.lookup(dem.getInput());
        if (k == PortLift::Unknown)
          for (Value v : dem.getOutputs()) {
            auto kk = m.lookup(v);
            if (kk != PortLift::Unknown) {
              k = kk;
              break;
            }
          }
        if (k == PortLift::Unknown)
          continue;
        setIfStronger(dem.getInput(), k, changed);
        for (Value v : dem.getOutputs())
          setIfStronger(v, k, changed);
      }
    }
  }
  return m;
}

// Lift a fabric.bits<N> to the appropriate software type given the lift kind.
static Type liftWith(unsigned w, PortLift kind, MLIRContext *ctx) {
  if (kind == PortLift::Float) {
    Type t = floatLifted(w, ctx);
    if (t)
      return t;
  }
  return intLifted(w, ctx);
}

//===----------------------------------------------------------------------===//
// Choice axes: each axis is one configurable knob over a fabric op.
//===----------------------------------------------------------------------===//

struct ChoiceAxis {
  Operation *fabricOp; // owning fabric.op / mux / demux
  std::string key;     // "op_sel" | hw_params key | "sel"
  SmallVector<Attribute> values;
};

// Parse a hex literal (with or without "0x" prefix) into an attribute
// matching `resultTy`. Returns nullptr on invalid input or unsupported
// result type.
static Attribute parseConstHex(StringRef hex, Type resultTy, MLIRContext *ctx) {
  StringRef body = hex;
  if (body.starts_with("0x") || body.starts_with("0X"))
    body = body.substr(2);
  if (body.empty())
    return nullptr;
  uint64_t v = 0;
  if (body.getAsInteger(16, v))
    return nullptr;
  if (auto intTy = ::mlir::dyn_cast<IntegerType>(resultTy)) {
    return ::mlir::IntegerAttr::get(intTy, ::llvm::APInt(intTy.getWidth(), v));
  }
  if (auto floatTy = ::mlir::dyn_cast<::mlir::FloatType>(resultTy)) {
    ::llvm::APInt bits(floatTy.getWidth(), v);
    ::llvm::APFloat fp(floatTy.getFloatSemantics(), bits);
    return ::mlir::FloatAttr::get(floatTy, fp);
  }
  return nullptr;
}

// Maximum number of hardware ports allowed for any variadic op. Bitmask
// cardinality is 2^M - 1 (the all-zero mask is illegal); we cap M at 8 so
// each variadic op contributes at most 255 axis values to the Cartesian
// product. Larger M is rejected by the enumerator with a diagnostic.
static constexpr unsigned kVariadicMaxM = 8;

// Hardware port count M for a variadic fabric.op:
//   * VariadicSync   : numInputs == numOutputs == M
//   * VariadicMux    : numInputs == M + 1 (1 sel + M data); numOutputs == 1
//   * VariadicDemux  : numInputs == 2 (1 sel + 1 data); numOutputs == M
static unsigned variadicM(Flavor f, ::fabric::OpOp fop) {
  switch (f) {
  case Flavor::VariadicSyncFlavor:
    return fop.getInputs().size();
  case Flavor::VariadicMuxFlavor:
    return fop.getInputs().size() > 0 ? fop.getInputs().size() - 1 : 0;
  case Flavor::VariadicDemuxFlavor:
    return fop.getOutputs().size();
  default:
    return 0;
  }
}

// Logical sel type for a materialized dataflow.mux/demux with N data
// ports, mirroring the dataflow op verifier:
//   * N == 2 -> i1
//   * N >= 3 -> index
// N == 1 is not legal for dataflow.mux/demux (the dataflow op verifier
// rejects fewer than 2 fan-in/out); the enumerator skips such configs
// upstream.
static Type variadicMuxLogicalSelType(unsigned N, MLIRContext *ctx) {
  if (N == 2)
    return IntegerType::get(ctx, 1);
  return ::mlir::IndexType::get(ctx);
}

// Population count of a bitmask string ("0"/"1" only). Caller validates
// the alphabet.
static unsigned popcountBitmask(StringRef s) {
  unsigned n = 0;
  for (char c : s)
    if (c == '1')
      ++n;
  return n;
}

// Convert a raw sw_config attribute value (typically StringAttr) into the
// MLIR attribute the materialized sw op expects.
static Attribute toSwAttr(StringRef opSym, StringRef key, Attribute raw,
                          MLIRContext *ctx) {
  if (opSym == "arith.cmpi" && key == "predicate") {
    auto str = ::mlir::dyn_cast<StringAttr>(raw);
    if (!str)
      return nullptr;
    auto pred = ::mlir::arith::symbolizeCmpIPredicate(str.getValue());
    if (!pred)
      return nullptr;
    return ::mlir::arith::CmpIPredicateAttr::get(ctx, *pred);
  }
  if (opSym == "arith.cmpf" && key == "predicate") {
    auto str = ::mlir::dyn_cast<StringAttr>(raw);
    if (!str)
      return nullptr;
    auto pred = ::mlir::arith::symbolizeCmpFPredicate(str.getValue());
    if (!pred)
      return nullptr;
    return ::mlir::arith::CmpFPredicateAttr::get(ctx, *pred);
  }
  return raw; // pass through (StringAttr / IntegerAttr / etc.)
}

//===----------------------------------------------------------------------===//
// Materialization
//===----------------------------------------------------------------------===//

using ValueMap = llvm::DenseMap<Value, Value>;

// Returns (sel, discard, disconnect) decoded from a chosen sw_config dict.
static std::tuple<unsigned, bool, bool>
decodeMuxLikeMode(const llvm::StringMap<Attribute> &chosen) {
  auto sel = ::mlir::cast<::mlir::IntegerAttr>(chosen.lookup("sel")).getInt();
  bool discard =
      ::mlir::cast<::mlir::BoolAttr>(chosen.lookup("discard")).getValue();
  bool disconnect =
      ::mlir::cast<::mlir::BoolAttr>(chosen.lookup("disconnect")).getValue();
  return {(unsigned)sel, discard, disconnect};
}

// Returns the bitmask string of a variadic fabric.op from the chosen
// sw_configs. Returns empty StringRef when no bitmask was chosen (the
// caller treats this as "not configured / cannot fire").
static StringRef getChosenBitmask(
    Operation *fop,
    const llvm::DenseMap<Operation *, llvm::StringMap<Attribute>> &chosenByOp) {
  auto it = chosenByOp.find(fop);
  if (it == chosenByOp.end())
    return {};
  auto bmIt = it->second.find("bitmask");
  if (bmIt == it->second.end())
    return {};
  if (auto str = ::mlir::dyn_cast<StringAttr>(bmIt->second))
    return str.getValue();
  return {};
}

// Returns true if input port `portIdx` of variadic fabric.op `fop`
// (Flavor `f`) is selected by `bitmask`. For sync, inputs[i] is active
// iff bitmask[i] == '1'. For mux, input #0 (sel) is always active when
// the op fires; input #(1+i) is a data port active iff bitmask[i] == '1'.
// For demux, input #0 (sel) and input #1 (data) are always active when
// the op fires.
static bool variadicInputActive(Flavor f, unsigned portIdx, StringRef bm) {
  if (bm.empty())
    return false;
  switch (f) {
  case Flavor::VariadicSyncFlavor:
    return portIdx < bm.size() && bm[portIdx] == '1';
  case Flavor::VariadicMuxFlavor:
    if (portIdx == 0)
      return true; // sel
    if (portIdx - 1 < bm.size())
      return bm[portIdx - 1] == '1';
    return false;
  case Flavor::VariadicDemuxFlavor:
    return portIdx <= 1; // sel + data both always active when firing
  default:
    return false;
  }
}

// Returns true if output port `portIdx` of variadic fabric.op `fop`
// (Flavor `f`) is alive when `bm` is the chosen bitmask.
static bool variadicOutputActive(Flavor f, unsigned portIdx, StringRef bm) {
  if (bm.empty())
    return false;
  switch (f) {
  case Flavor::VariadicSyncFlavor:
  case Flavor::VariadicDemuxFlavor:
    return portIdx < bm.size() && bm[portIdx] == '1';
  case Flavor::VariadicMuxFlavor:
    return portIdx == 0; // single output, always live when firing
  default:
    return false;
  }
}

// Returns the Flavor of `fop` if it is a variadic fabric.op, otherwise
// Flavor::IntArith (caller checks isVariadicFlavor first).
static Flavor flavorOfFabricOp(::fabric::OpOp fop) {
  if (fop.getOpList().empty())
    return Flavor::IntArith;
  auto sym = ::mlir::cast<FlatSymbolRefAttr>(fop.getOpList()[0]).getValue();
  return opFlavors().lookup(sym);
}

// Whether the use port at `idx` of `user` is "blocking" for `v` in this
// configuration: the user's hardware-side ready signal would remain low,
// so the value's producer cannot complete its broadcast to that consumer
// unless an alternative drain (discard mode on another fanout branch, a
// different firing consumer, etc.) is provided.
//
// "Non-blocking" (a.k.a. dormant) uses are uses that the configurable
// fabric drains transparently: their ready is tied high so they cannot
// stall the producer. Those uses do not consume `v` in the materialized
// subgraph either; they only exist for routing flexibility.
//
// Non-blocking cases:
//   * fabric.OpOp that does not fire in this configuration. The fabric
//     configures unused op modules with their input ready signals tied
//     high, draining whatever broadcasts arrive. They consume nothing.
//   * fabric.OpOp that fires but whose `idx` is masked off by a variadic
//     bitmask (dataflow.{sync,mux,demux}). The masked-off physical port
//     is unused for this bitmask; its ready is tied high.
//   * fabric.MuxOp that fires (any operand index, including the
//     non-selected ports). A firing fabric.mux drains every input port:
//     the selected one propagates, the others are accepted and discarded.
//     This is the fix for the fanout-to-distinct-muxes bug.
//   * fabric.DemuxOp that fires at any non-data operand index (only
//     `idx == 0` is the data port; demux has no other operand kinds, so
//     this case is dead weight in practice).
//   * fabric.MuxOp / fabric.DemuxOp in disconnect mode: the input ready
//     is held low by definition, so this is actually blocking. Listed
//     here as a reminder that `disconnect` is treated as "blocking with
//     no completion path", which is what causes the analyzer to drop
//     configs that disconnect a value with no other consumer.
//
// Blocking cases:
//   * fabric.yield (the FU's terminator): always blocking, since the FU
//     output port must complete the handshake outward.
//   * fabric.OpOp that fires AND `idx` is an active operand index for
//     this configuration: the hardware actually consumes the value.
//   * fabric.MuxOp that fires AND not in disconnect mode AND `idx` is the
//     selected input port: the mux propagates this value to its output.
//     Note: non-selected ports of a firing mux are drained, hence
//     non-blocking (see above).
//   * fabric.MuxOp / fabric.DemuxOp that does NOT fire and is not in
//     disconnect mode: the hardware ready remains low (the unit has no
//     downstream consumer demanding its output), so the fanout deadlocks.
//     Treat as blocking so the analyzer rejects such configs (the user
//     must explicitly request `discard` or a side drain).
//   * fabric.DemuxOp that fires AND not in disconnect mode AND `idx == 0`
//     (its single data input): consumed by the demux.
//
// Note on the legacy name `useIsActive`: the function returns true when
// the use is "active in the consume-or-drain sense" (i.e. the producer's
// broadcast can complete via this use). The alive-shrink in
// `analyzeConfig` uses `allUsesActive` to test whether every use of a
// value can complete its handshake; if any use is blocking-and-stuck,
// the value cannot stay alive. Treating drained-but-not-consuming uses
// (firing mux at non-selected port, non-firing fabric.op) as active is
// the central fix for the fanout-broadcast deadlock false positive.
static bool useIsActive(
    Operation *user, unsigned idx, const llvm::DenseSet<Operation *> &fires,
    const llvm::DenseMap<Operation *, llvm::StringMap<Attribute>> &chosenByOp) {
  if (::mlir::isa<::fabric::YieldOp>(user))
    return true;
  if (::mlir::isa<::fabric::MuxOp>(user)) {
    if (!fires.count(user))
      return false;
    auto [sel, discard, disconnect] =
        decodeMuxLikeMode(chosenByOp.lookup(user));
    (void)sel;
    (void)discard;
    if (disconnect)
      return false;
    // A firing fabric.mux drains every input port: the selected port
    // propagates the value, the non-selected ports complete their
    // handshakes by accepting and discarding the data.
    return true;
  }
  if (::mlir::isa<::fabric::DemuxOp>(user)) {
    if (!fires.count(user))
      return false;
    auto [sel, discard, disconnect] =
        decodeMuxLikeMode(chosenByOp.lookup(user));
    (void)sel;
    (void)discard;
    if (disconnect)
      return false;
    return idx == 0;
  }
  return true;
}

static bool allUsesActive(
    Value v, const llvm::DenseSet<Operation *> &fires,
    const llvm::DenseMap<Operation *, llvm::StringMap<Attribute>> &chosenByOp) {
  for (::mlir::OpOperand &use : v.getUses())
    if (!useIsActive(use.getOwner(), use.getOperandNumber(), fires, chosenByOp))
      return false;
  return true;
}

static bool opCanFire(
    Operation *op, const llvm::DenseSet<Value> &alive,
    const llvm::DenseSet<Value> &demanded,
    const llvm::DenseMap<Operation *, llvm::StringMap<Attribute>> &chosenByOp) {
  if (auto fop = ::mlir::dyn_cast<::fabric::OpOp>(op)) {
    Flavor f = flavorOfFabricOp(fop);
    if (isVariadicFlavor(f)) {
      StringRef bm = getChosenBitmask(op, chosenByOp);
      if (bm.empty())
        return false;
      if (popcountBitmask(bm) == 0)
        return false;
      // Every active input must be alive.
      for (auto [i, in] : llvm::enumerate(fop.getInputs())) {
        if (!variadicInputActive(f, i, bm))
          continue;
        if (!alive.count(in))
          return false;
      }
      // At least one active output must be demanded.
      bool any = false;
      for (auto [i, out] : llvm::enumerate(fop.getOutputs())) {
        if (!variadicOutputActive(f, i, bm))
          continue;
        if (demanded.count(out)) {
          any = true;
          break;
        }
      }
      return any;
    }
    for (Value in : fop.getInputs())
      if (!alive.count(in))
        return false;
    bool any = false;
    for (Value out : fop.getOutputs())
      if (demanded.count(out)) {
        any = true;
        break;
      }
    return any;
  }
  if (auto m = ::mlir::dyn_cast<::fabric::MuxOp>(op)) {
    auto [sel, discard, disconnect] = decodeMuxLikeMode(chosenByOp.lookup(op));
    if (disconnect)
      return false;
    if (!alive.count(m.getInputs()[sel]))
      return false;
    if (discard)
      return true;
    return demanded.count(m.getOutput());
  }
  if (auto d = ::mlir::dyn_cast<::fabric::DemuxOp>(op)) {
    auto [sel, discard, disconnect] = decodeMuxLikeMode(chosenByOp.lookup(op));
    if (disconnect)
      return false;
    if (!alive.count(d.getInput()))
      return false;
    if (discard)
      return true;
    return demanded.count(d.getOutputs()[sel]);
  }
  return false;
}

// Result of analyzing one FU configuration: which fabric ops fire and
// which fabric SSA values are live (carry a real handshake) under the
// chosen sw_configs. `liveYieldIndices` lists the FU yield positions that
// remain live in this config, in original program order.
struct ConfigAnalysis {
  llvm::DenseSet<Value> demanded;
  llvm::DenseSet<Value> alive;
  llvm::DenseSet<Operation *> fires;
  SmallVector<unsigned, 4> liveYieldIndices;
};

// Run the (demanded, alive, fires) fixed-point and validate the chosen
// configuration. Returns std::nullopt when the config is invalid (e.g. a
// discard-mode mux input is dead). Yield positions whose value is not
// alive in this config are reported via `liveYieldIndices` (they are
// silently omitted; the materialized subgraph signature shrinks
// accordingly), but never cause failure on their own.
static std::optional<ConfigAnalysis> analyzeConfig(
    FuOp fu,
    const llvm::DenseMap<Operation *, llvm::StringMap<Attribute>> &chosenByOp) {
  Block &fuBody = fu.getBody().front();
  auto yieldOp = ::mlir::cast<::fabric::YieldOp>(fuBody.getTerminator());

  // 1. Backward demand propagation.
  llvm::DenseSet<Value> demanded;
  for (Value y : yieldOp.getValues())
    demanded.insert(y);
  for (Operation &op : fuBody.without_terminator()) {
    if (auto m = ::mlir::dyn_cast<::fabric::MuxOp>(&op)) {
      auto [sel, discard, disconnect] =
          decodeMuxLikeMode(chosenByOp.lookup(&op));
      if (discard && !disconnect)
        demanded.insert(m.getInputs()[sel]);
    } else if (auto d = ::mlir::dyn_cast<::fabric::DemuxOp>(&op)) {
      auto [sel, discard, disconnect] =
          decodeMuxLikeMode(chosenByOp.lookup(&op));
      (void)sel;
      if (discard && !disconnect)
        demanded.insert(d.getInput());
    }
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : fuBody.without_terminator()) {
      if (auto fop = ::mlir::dyn_cast<::fabric::OpOp>(&op)) {
        Flavor f = flavorOfFabricOp(fop);
        if (isVariadicFlavor(f)) {
          StringRef bm = getChosenBitmask(&op, chosenByOp);
          if (bm.empty() || popcountBitmask(bm) == 0)
            continue;
          // An active output being demanded propagates demand to the
          // active inputs of the variadic op.
          bool anyOut = false;
          for (auto [i, out] : llvm::enumerate(fop.getOutputs())) {
            if (!variadicOutputActive(f, i, bm))
              continue;
            if (demanded.count(out)) {
              anyOut = true;
              break;
            }
          }
          if (!anyOut)
            continue;
          for (auto [i, in] : llvm::enumerate(fop.getInputs())) {
            if (!variadicInputActive(f, i, bm))
              continue;
            if (demanded.insert(in).second)
              changed = true;
          }
          continue;
        }
        bool anyOut = false;
        for (Value out : fop.getOutputs())
          if (demanded.count(out)) {
            anyOut = true;
            break;
          }
        if (!anyOut)
          continue;
        for (Value in : fop.getInputs())
          if (demanded.insert(in).second)
            changed = true;
      } else if (auto m = ::mlir::dyn_cast<::fabric::MuxOp>(&op)) {
        auto [sel, discard, disconnect] =
            decodeMuxLikeMode(chosenByOp.lookup(&op));
        if (discard || disconnect)
          continue;
        if (demanded.count(m.getOutput()))
          if (demanded.insert(m.getInputs()[sel]).second)
            changed = true;
      } else if (auto d = ::mlir::dyn_cast<::fabric::DemuxOp>(&op)) {
        auto [sel, discard, disconnect] =
            decodeMuxLikeMode(chosenByOp.lookup(&op));
        if (discard || disconnect)
          continue;
        if (demanded.count(d.getOutputs()[sel]))
          if (demanded.insert(d.getInput()).second)
            changed = true;
      }
    }
  }

  // 2. Monotonic shrink fixed-point on alive / fires.
  llvm::DenseSet<Value> alive(demanded.begin(), demanded.end());
  llvm::DenseSet<Operation *> fires;
  changed = true;
  while (changed) {
    changed = false;
    llvm::DenseSet<Operation *> newFires;
    for (Operation &op : fuBody.without_terminator())
      if (opCanFire(&op, alive, demanded, chosenByOp))
        newFires.insert(&op);
    if (newFires != fires) {
      fires = std::move(newFires);
      changed = true;
    }
    llvm::DenseSet<Value> newAlive;
    for (Value v : alive) {
      bool producerOk = false;
      if (::mlir::isa<::mlir::BlockArgument>(v)) {
        producerOk = demanded.count(v);
      } else if (Operation *def = v.getDefiningOp()) {
        if (!fires.count(def)) {
          producerOk = false;
        } else if (auto vfop = ::mlir::dyn_cast<::fabric::OpOp>(def);
                   vfop && isVariadicFlavor(flavorOfFabricOp(vfop))) {
          // Variadic fabric.op only produces values on its active outputs.
          Flavor f = flavorOfFabricOp(vfop);
          StringRef bm = getChosenBitmask(def, chosenByOp);
          unsigned outIdx = 0;
          for (auto [i, out] : llvm::enumerate(vfop.getOutputs())) {
            if (out == v) {
              outIdx = i;
              break;
            }
          }
          producerOk = variadicOutputActive(f, outIdx, bm);
        } else if (auto dem = ::mlir::dyn_cast<::fabric::DemuxOp>(def)) {
          // A fabric.demux is a 1-of-N selector: in any given config only
          // outputs[sel] carries a value (and even that only when not in
          // discard mode). The other outputs do not exist on the wire.
          auto [sel, discard, disconnect] =
              decodeMuxLikeMode(chosenByOp.lookup(def));
          (void)disconnect;
          if (discard) {
            producerOk = false;
          } else {
            producerOk = (v == dem.getOutputs()[sel]);
          }
        } else if (auto m = ::mlir::dyn_cast<::fabric::MuxOp>(def)) {
          // A fabric.mux that fires in discard mode drains its selected
          // input and produces no output value.
          auto [sel, discard, disconnect] =
              decodeMuxLikeMode(chosenByOp.lookup(def));
          (void)sel;
          (void)disconnect;
          producerOk = !discard;
        } else {
          producerOk = true;
        }
      }
      if (!producerOk)
        continue;
      if (!allUsesActive(v, fires, chosenByOp))
        continue;
      newAlive.insert(v);
    }
    if (newAlive != alive) {
      alive = std::move(newAlive);
      changed = true;
    }
  }

  // 3. Validate. The opCanFire / alive shrink fixed-point above already
  // computes a self-consistent (fires, alive) pair: any op that ended up
  // in `fires` had every active operand index pointing at an alive value
  // when last reconsidered. We re-check that invariant here as a defense
  // in depth, walking only firing ops and only their *active* operand
  // indices: for variadic ops we restrict to bitmask-active ports; for
  // fabric.mux/demux we restrict to the chosen-sel input (and, in
  // discard mode, the selected drain input). Non-firing internal ops are
  // ignored on purpose: their fabric inputs may have shrunk out of
  // `alive` legitimately (the canonical Bug A shape: a demux output that
  // a downstream consumer demanded but the bitmask masked off; that
  // downstream consumer is excluded from `fires`, so its dead operands
  // do not matter). Yield positions whose value is not alive are simply
  // dropped from the materialized signature via `liveYieldIndices`; they
  // never cause a rejection here.
  llvm::DenseSet<Value> yieldVals;
  for (Value y : yieldOp.getValues())
    yieldVals.insert(y);
  for (Operation *op : fires) {
    if (auto fop = ::mlir::dyn_cast<::fabric::OpOp>(op)) {
      Flavor f = flavorOfFabricOp(fop);
      if (isVariadicFlavor(f)) {
        StringRef bm = getChosenBitmask(op, chosenByOp);
        for (auto [i, in] : llvm::enumerate(fop.getInputs())) {
          if (!variadicInputActive(f, i, bm))
            continue;
          if (!alive.count(in))
            return std::nullopt;
        }
        continue;
      }
      for (Value in : fop.getInputs())
        if (!alive.count(in))
          return std::nullopt;
      continue;
    }
    if (auto m = ::mlir::dyn_cast<::fabric::MuxOp>(op)) {
      auto [sel, discard, disconnect] =
          decodeMuxLikeMode(chosenByOp.lookup(op));
      (void)disconnect;
      // Both normal-mode and discard-mode firing fabric.mux read the
      // selected input; only the unselected ports are passive drains
      // and need not be alive.
      if (!alive.count(m.getInputs()[sel]))
        return std::nullopt;
      (void)discard;
      continue;
    }
    if (auto d = ::mlir::dyn_cast<::fabric::DemuxOp>(op)) {
      auto [sel, discard, disconnect] =
          decodeMuxLikeMode(chosenByOp.lookup(op));
      (void)sel;
      (void)disconnect;
      (void)discard;
      // fabric.demux has a single data input (idx 0); it is always
      // active for any firing demux (normal or discard).
      if (!alive.count(d.getInput()))
        return std::nullopt;
      continue;
    }
  }

  ConfigAnalysis a;
  a.demanded = std::move(demanded);
  a.alive = std::move(alive);
  a.fires = std::move(fires);
  for (auto [i, y] : llvm::enumerate(yieldOp.getValues()))
    if (a.alive.count(y))
      a.liveYieldIndices.push_back(static_cast<unsigned>(i));
  return a;
}

// Compute the set of fabric SSA values that the materialized software
// body actually reads (transitively through pass-through fabric.mux /
// fabric.demux chains). The set excludes values consumed only by:
//   * non-firing ops (those don't materialize at all),
//   * the unselected ports of a firing fabric.mux (drained, not consumed),
//   * a firing fabric.mux/demux operating in discard mode (the selected
//     input is drained at the hardware level but no value flows into the
//     software graph).
//
// A block argument is software-live iff it appears in the returned set.
// This is the central enabler for the discard-input dedup invariant: a
// block-arg whose only role is to feed a discard-mode mux/demux (drain
// path) does not propagate into the materialized subgraph signature.
static llvm::DenseSet<Value> computeSoftwareLiveValues(
    FuOp fu,
    const llvm::DenseMap<Operation *, llvm::StringMap<Attribute>> &chosenByOp,
    const ConfigAnalysis &analysis) {
  Block &fuBody = fu.getBody().front();
  auto yieldOp = ::mlir::cast<::fabric::YieldOp>(fuBody.getTerminator());

  llvm::DenseSet<Value> swNeeded;

  auto addIfAlive = [&](Value v, bool &changed) {
    if (!analysis.alive.count(v))
      return;
    if (swNeeded.insert(v).second)
      changed = true;
  };

  // Seed with the live yield values: anything the dataflow.yield reads is
  // by definition software-live.
  for (unsigned idx : analysis.liveYieldIndices) {
    Value y = yieldOp.getValues()[idx];
    bool dummy = false;
    addIfAlive(y, dummy);
  }

  // Seed with operand values consumed by firing software-materialized
  // ops at their active operand indices. This is exactly the set of
  // operands the materialized op will read in pass A of
  // materializeBodyForConfig (mux/demux pass-through edges and discard
  // modes are handled below via the back-propagation loop).
  for (Operation *op : analysis.fires) {
    if (auto fop = ::mlir::dyn_cast<::fabric::OpOp>(op)) {
      Flavor f = flavorOfFabricOp(fop);
      if (isVariadicFlavor(f)) {
        StringRef bm = getChosenBitmask(op, chosenByOp);
        for (auto [i, in] : llvm::enumerate(fop.getInputs())) {
          if (!variadicInputActive(f, i, bm))
            continue;
          bool dummy = false;
          addIfAlive(in, dummy);
        }
        continue;
      }
      for (Value in : fop.getInputs()) {
        bool dummy = false;
        addIfAlive(in, dummy);
      }
    }
    // Note: firing fabric.mux/demux do not contribute direct seeds here.
    // They are handled by the back-propagation loop below: if the mux/
    // demux output is in `swNeeded` (because some downstream firing op
    // reads it), back-propagate to the selected input. A discard-mode
    // mux/demux produces no output value, so its selected input never
    // becomes software-live this way -- which is the design intent.
  }

  // Back-propagate through firing pass-through fabric.mux/demux chains
  // until fixed point: if a mux/demux output is software-needed, the
  // selected input becomes software-needed too (because the materialized
  // body wires the output to the input directly via valueMap pass-
  // through). Discard-mode mux/demux are skipped: they produce no value,
  // so their selected input is a hardware-only drain.
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &op : fuBody.without_terminator()) {
      if (auto m = ::mlir::dyn_cast<::fabric::MuxOp>(&op)) {
        if (!analysis.fires.count(&op))
          continue;
        auto [sel, discard, disconnect] =
            decodeMuxLikeMode(chosenByOp.lookup(&op));
        (void)disconnect;
        if (discard)
          continue;
        if (swNeeded.count(m.getOutput()))
          addIfAlive(m.getInputs()[sel], changed);
      } else if (auto d = ::mlir::dyn_cast<::fabric::DemuxOp>(&op)) {
        if (!analysis.fires.count(&op))
          continue;
        auto [sel, discard, disconnect] =
            decodeMuxLikeMode(chosenByOp.lookup(&op));
        (void)disconnect;
        if (discard)
          continue;
        if (swNeeded.count(d.getOutputs()[sel]))
          addIfAlive(d.getInput(), changed);
      }
    }
  }

  return swNeeded;
}

// Materialize the sw subgraph body for a configuration whose analysis has
// already succeeded. `liveInputIndices` lists the FU input port positions
// that remain live in this config, in ascending order; `subBlockArgs` is
// the matching list of sw block arguments (same length). Returns the
// materialized SSA values for the live yield positions in the same order
// as `analysis.liveYieldIndices`, or std::nullopt when an inner sw op
// cannot be built (e.g. unsupported const_hex_value width).
//
// Implementation: the FU body is a graph region (RegionKind::Graph) and may
// contain back-edges, so a single textual walk that resolves operands
// eagerly cannot work. We use a two-pass scheme:
//
//   Pass A: walk firing fabric.op ops in textual order. Build the
//           materialized sw op for each, but if any operand lookup misses
//           in `valueMap`, synthesize a placeholder Value (an
//           `unrealized_conversion_cast` with no inputs and the operand's
//           expected sw type) and use that. Each placeholder is recorded
//           in `placeholders[fuValue]`. Outputs of the sw op are recorded
//           in `valueMap`. fabric.mux/fabric.demux that fire in
//           pass-through mode short-circuit by setting
//           `valueMap[output] = valueMap[input]` (or a placeholder if the
//           input is missing); discard / disconnect modes contribute no
//           value. fabric.mux/fabric.demux pass-throughs are recorded in
//           `passThroughEdges` so we can resolve them after Pass A.
//
//   Pass B: for each pass-through edge, chain-resolve so that
//           `valueMap[output]` ultimately points at a real sw value (not
//           a placeholder). Fixed-point iteration handles back-edges
//           through pass-through chains.
//
//   Pass C: for every placeholder we created, find the corresponding real
//           sw value via `valueMap` (or by chasing a pass-through chain)
//           and `replaceAllUsesWith` it. Erase the placeholder. After
//           Pass C every materialized sw op has its real operands wired,
//           and `valueMap` no longer references any placeholder.
//
//   Pass D: build the dataflow.yield using the now-real `valueMap` entries
//           for live yield positions. (Done by the caller via the returned
//           value list.)
static std::optional<SmallVector<Value, 4>> materializeBodyForConfig(
    FuOp fu, ::llvm::ArrayRef<unsigned> liveInputIndices,
    ::mlir::ValueRange subBlockArgs, OpBuilder &builder,
    const llvm::DenseMap<Operation *, llvm::StringMap<Attribute>> &chosenByOp,
    const ConfigAnalysis &analysis, const llvm::DenseSet<Value> &swLive) {
  Block &fuBody = fu.getBody().front();
  MLIRContext *ctx = fu.getContext();
  auto yieldOp = ::mlir::cast<::fabric::YieldOp>(fuBody.getTerminator());

  ValueMap valueMap;
  for (auto [pos, fuIdx] : llvm::enumerate(liveInputIndices)) {
    Value fuArg = fuBody.getArgument(fuIdx);
    valueMap[fuArg] = subBlockArgs[pos];
  }

  // fabric SSA values that we synthesized a placeholder for, mapped to the
  // placeholder Value. Pass C rewrites uses of these placeholders to the
  // real sw value once it lands in `valueMap`.
  llvm::DenseMap<Value, Value> placeholders;
  // Op handles for the placeholders, so we can erase them after Pass C.
  llvm::SmallVector<Operation *, 4> placeholderOps;
  // fabric.mux / fabric.demux pass-through edges: (output, selected input).
  // Resolved in Pass B by chaining `valueMap[input]` into
  // `valueMap[output]`.
  llvm::SmallVector<std::pair<Value, Value>, 4> passThroughEdges;

  // Synthesize an unrealized_conversion_cast placeholder of type `ty` for
  // fabric value `fuVal`. Idempotent: subsequent lookups for the same
  // `fuVal` return the same placeholder.
  auto getOrPlaceholder = [&](Value fuVal, Type ty) -> Value {
    auto it = valueMap.find(fuVal);
    if (it != valueMap.end())
      return it->second;
    auto pIt = placeholders.find(fuVal);
    if (pIt != placeholders.end())
      return pIt->second;
    OperationState ph(fuVal.getLoc(),
                      ::mlir::UnrealizedConversionCastOp::getOperationName());
    ph.addTypes(ty);
    Operation *phOp = builder.create(ph);
    placeholderOps.push_back(phOp);
    Value v = phOp->getResult(0);
    placeholders[fuVal] = v;
    valueMap[fuVal] = v;
    return v;
  };

  // Pass A: walk firing ops in textual order and build materialized sw ops.
  for (Operation &op : fuBody.without_terminator()) {
    if (auto fop = ::mlir::dyn_cast<::fabric::OpOp>(&op)) {
      if (!analysis.fires.count(&op))
        continue; // op doesn't fire in this config
      const auto &chosen = chosenByOp.lookup(&op);
      ArrayAttr opList = fop.getOpList();
      StringRef sym;
      auto opSelIt = chosen.find("op_sel");
      if (opSelIt != chosen.end())
        sym = ::mlir::cast<StringAttr>(opSelIt->second).getValue();
      else
        sym = ::mlir::cast<FlatSymbolRefAttr>(opList[0]).getValue();
      Flavor flavor = opFlavors().lookup(sym);

      // Variadic dataflow.{sync,mux,demux} have a fundamentally different
      // realization: only the bitmask-active subset of fabric ports
      // become operands/results of the materialized dataflow op, and the
      // sel port (mux/demux) gets a logical width derived from N rather
      // than the hardware-wide port width.
      if (isVariadicFlavor(flavor)) {
        StringRef bm =
            chosen.lookup("bitmask")
                ? ::mlir::cast<StringAttr>(chosen.lookup("bitmask")).getValue()
                : StringRef{};
        if (bm.empty())
          return std::nullopt;
        unsigned N = popcountBitmask(bm);
        if (N == 0)
          return std::nullopt;

        if (flavor == Flavor::VariadicSyncFlavor) {
          // dataflow.sync: collect active inputs in port order; both
          // input and output port-pairs share a width per port.
          SmallVector<Value, 4> swInputs;
          SmallVector<Type, 4> swResultTypes;
          for (auto [i, in] : llvm::enumerate(fop.getInputs())) {
            if (!variadicInputActive(flavor, i, bm))
              continue;
            unsigned w = ::mlir::cast<BitsType>(in.getType()).getWidth();
            Type inTy = liftFor(flavor, w, /*isOutput=*/false, i, ctx);
            if (!inTy)
              return std::nullopt;
            swInputs.push_back(getOrPlaceholder(in, inTy));
          }
          for (auto [i, t] : llvm::enumerate(fop.getResultTypes())) {
            if (!variadicOutputActive(flavor, i, bm))
              continue;
            unsigned w = ::mlir::cast<BitsType>(t).getWidth();
            Type ty = liftFor(flavor, w, /*isOutput=*/true, i, ctx);
            if (!ty)
              return std::nullopt;
            swResultTypes.push_back(ty);
          }
          if (swInputs.size() != swResultTypes.size())
            return std::nullopt;
          OperationState state(fop.getLoc(), sym);
          state.addOperands(swInputs);
          state.addTypes(swResultTypes);
          Operation *swOp = builder.create(state);
          unsigned outPos = 0;
          for (auto [i, fuOut] : llvm::enumerate(fop.getOutputs())) {
            if (!variadicOutputActive(flavor, i, bm))
              continue;
            if (analysis.alive.count(fuOut))
              valueMap[fuOut] = swOp->getResult(outPos);
            ++outPos;
          }
          continue;
        }
        if (flavor == Flavor::VariadicMuxFlavor) {
          // The materialized dataflow.mux's data-input count is the
          // bitmask popcount N; the sel input is the FU's input #0
          // remapped to the logical sel type (i1 for N==2, index for
          // N>=3). N==1 is rejected upstream because the dataflow.mux
          // verifier requires at least 2 data inputs.
          if (N < 2)
            return std::nullopt;
          // Sel block-arg must already exist in valueMap with its
          // hardware width; we cast/convert downstream by re-using the
          // value as-is when its type already matches the required
          // logical sel type, otherwise we synthesize a no-op cast to
          // the logical type. In practice the FU lift step now produces
          // the logical sel type directly when possible (see the lift
          // override applied in enumerateFuSubgraphs); this guard
          // catches mismatches.
          Value selFuVal = fop.getInputs()[0];
          auto selIt = valueMap.find(selFuVal);
          if (selIt == valueMap.end())
            return std::nullopt;
          Value selSw = selIt->second;
          Type wantSelTy = variadicMuxLogicalSelType(N, ctx);
          if (selSw.getType() != wantSelTy)
            return std::nullopt;

          SmallVector<Value, 4> swInputs;
          swInputs.push_back(selSw);
          for (auto [i, in] : llvm::enumerate(fop.getInputs())) {
            if (i == 0)
              continue;
            if (!variadicInputActive(flavor, i, bm))
              continue;
            unsigned w = ::mlir::cast<BitsType>(in.getType()).getWidth();
            Type inTy = liftFor(flavor, w, /*isOutput=*/false, i, ctx);
            if (!inTy)
              return std::nullopt;
            swInputs.push_back(getOrPlaceholder(in, inTy));
          }
          // Output: single port whose type comes from the fabric output.
          unsigned outW =
              ::mlir::cast<BitsType>(fop.getResultTypes()[0]).getWidth();
          Type outTy = liftFor(flavor, outW, /*isOutput=*/true, 0, ctx);
          if (!outTy)
            return std::nullopt;
          OperationState state(fop.getLoc(), sym);
          state.addOperands(swInputs);
          state.addTypes(outTy);
          Operation *swOp = builder.create(state);
          if (analysis.alive.count(fop.getOutputs()[0]))
            valueMap[fop.getOutputs()[0]] = swOp->getResult(0);
          continue;
        }
        if (flavor == Flavor::VariadicDemuxFlavor) {
          if (N < 2)
            return std::nullopt;
          Value selFuVal = fop.getInputs()[0];
          auto selIt = valueMap.find(selFuVal);
          if (selIt == valueMap.end())
            return std::nullopt;
          Value selSw = selIt->second;
          Type wantSelTy = variadicMuxLogicalSelType(N, ctx);
          if (selSw.getType() != wantSelTy)
            return std::nullopt;

          Value dataFuVal = fop.getInputs()[1];
          unsigned dataW =
              ::mlir::cast<BitsType>(dataFuVal.getType()).getWidth();
          Type dataTy = liftFor(flavor, dataW, /*isOutput=*/false, 1, ctx);
          if (!dataTy)
            return std::nullopt;
          Value dataSw = getOrPlaceholder(dataFuVal, dataTy);

          SmallVector<Type, 4> swResultTypes;
          for (auto [i, t] : llvm::enumerate(fop.getResultTypes())) {
            if (!variadicOutputActive(flavor, i, bm))
              continue;
            unsigned w = ::mlir::cast<BitsType>(t).getWidth();
            Type ty = liftFor(flavor, w, /*isOutput=*/true, i, ctx);
            if (!ty)
              return std::nullopt;
            swResultTypes.push_back(ty);
          }
          OperationState state(fop.getLoc(), sym);
          state.addOperands({selSw, dataSw});
          state.addTypes(swResultTypes);
          Operation *swOp = builder.create(state);
          unsigned outPos = 0;
          for (auto [i, fuOut] : llvm::enumerate(fop.getOutputs())) {
            if (!variadicOutputActive(flavor, i, bm))
              continue;
            if (analysis.alive.count(fuOut))
              valueMap[fuOut] = swOp->getResult(outPos);
            ++outPos;
          }
          continue;
        }
      }

      SmallVector<Value, 4> swInputs;
      swInputs.reserve(fop.getInputs().size());
      for (auto [i, in] : llvm::enumerate(fop.getInputs())) {
        unsigned w = ::mlir::cast<BitsType>(in.getType()).getWidth();
        Type inTy = liftFor(flavor, w, /*isOutput=*/false, i, ctx);
        if (!inTy)
          return std::nullopt;
        swInputs.push_back(getOrPlaceholder(in, inTy));
      }
      SmallVector<Type, 2> swResultTypes;
      swResultTypes.reserve(fop.getOutputs().size());
      for (auto [i, t] : llvm::enumerate(fop.getResultTypes())) {
        unsigned w = ::mlir::cast<BitsType>(t).getWidth();
        Type ty = liftFor(flavor, w, /*isOutput=*/true, i, ctx);
        if (!ty)
          return std::nullopt;
        swResultTypes.push_back(ty);
      }
      OperationState state(fop.getLoc(), sym);
      state.addOperands(swInputs);
      state.addTypes(swResultTypes);
      for (auto &kv : chosen) {
        if (kv.getKey() == "op_sel")
          continue;
        if (sym == "dataflow.constant" && kv.getKey() == "const_hex_value") {
          auto str = ::mlir::dyn_cast<StringAttr>(kv.getValue());
          if (!str)
            return std::nullopt;
          if (swResultTypes.empty())
            return std::nullopt;
          Attribute v = parseConstHex(str.getValue(), swResultTypes[0], ctx);
          if (!v)
            return std::nullopt;
          state.addAttribute("const_value", v);
          continue;
        }
        Attribute conv = toSwAttr(sym, kv.getKey(), kv.getValue(), ctx);
        if (!conv)
          return std::nullopt;
        state.addAttribute(kv.getKey(), conv);
      }
      Operation *swOp = builder.create(state);
      for (auto [fuOut, swOut] :
           llvm::zip(fop.getOutputs(), swOp->getResults()))
        if (analysis.alive.count(fuOut))
          valueMap[fuOut] = swOut;
      continue;
    }
    if (auto m = ::mlir::dyn_cast<::fabric::MuxOp>(&op)) {
      if (!analysis.fires.count(&op))
        continue;
      auto [sel, discard, disconnect] =
          decodeMuxLikeMode(chosenByOp.lookup(&op));
      (void)disconnect;
      if (discard)
        continue; // drains input, no output produced
      // Only record a pass-through edge when the output is software-
      // needed; otherwise the chain leads nowhere and would create a
      // stranded placeholder for a block-arg that Fix B intentionally
      // excluded from the subgraph signature.
      if (swLive.count(m.getOutput()))
        passThroughEdges.emplace_back(m.getOutput(), m.getInputs()[sel]);
      continue;
    }
    if (auto d = ::mlir::dyn_cast<::fabric::DemuxOp>(&op)) {
      if (!analysis.fires.count(&op))
        continue;
      auto [sel, discard, disconnect] =
          decodeMuxLikeMode(chosenByOp.lookup(&op));
      (void)disconnect;
      if (discard)
        continue;
      if (swLive.count(d.getOutputs()[sel]))
        passThroughEdges.emplace_back(d.getOutputs()[sel], d.getInput());
      continue;
    }
  }

  // Pass B: chain-resolve pass-through mux/demux edges. After Pass A,
  // every fabric value reachable from a firing op either has a real sw
  // value in `valueMap` (op output), a placeholder in `valueMap` (back-
  // edge that was used before its producer was visited), or no entry yet
  // (pass-through output whose input was a back-edge). This loop seeds
  // pass-through outputs and iterates until no more progress is made.
  {
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto [out, in] : passThroughEdges) {
        auto inIt = valueMap.find(in);
        Value src;
        if (inIt != valueMap.end()) {
          src = inIt->second;
        } else {
          // Input still missing: synthesize a placeholder so users have
          // something to wire to. The expected sw type matches the input
          // fabric width with no specific flavor knowledge; use intLifted
          // as the safe default. Pass C will RAUW it once the producer
          // lands a real value.
          unsigned w = ::mlir::cast<BitsType>(in.getType()).getWidth();
          src = getOrPlaceholder(in, intLifted(w, ctx));
        }
        auto outIt = valueMap.find(out);
        if (outIt == valueMap.end() || outIt->second != src) {
          valueMap[out] = src;
          changed = true;
        }
      }
    }
  }

  // Pass C: rewire placeholders to their real sw values. After Pass A and
  // Pass B, every producer of a placeholder fabric value has stored its
  // real sw value in `valueMap[fuVal]`. Walk the placeholder map and RAUW
  // each placeholder Value to that real value, then erase the placeholder
  // op. valueMap entries that still pointed at the placeholder are
  // updated to the real value first so downstream lookups (e.g. yields)
  // see only real sw values.
  for (auto &kv : placeholders) {
    Value fuVal = kv.first;
    Value placeholder = kv.second;
    auto it = valueMap.find(fuVal);
    if (it == valueMap.end() || it->second == placeholder) {
      // No real value ever materialized for this fabric value. This is
      // unreachable for a well-formed config (any placeholder we created
      // was driven by a use from a firing op, whose alive analysis
      // guaranteed an alive producer), but bail out cleanly.
      return std::nullopt;
    }
    Value real = it->second;
    placeholder.replaceAllUsesWith(real);
    // Pass-through outputs that adopted this placeholder via Pass B keep
    // a stale valueMap entry; rewrite them so yields see real values.
    for (auto &vm : valueMap)
      if (vm.second == placeholder)
        vm.second = real;
  }
  for (Operation *phOp : placeholderOps)
    phOp->erase();

  // Pass D: collect live yield values for the caller. Pass-through chains
  // and placeholder rewrites have already settled valueMap, so a direct
  // lookup is sufficient.
  SmallVector<Value, 4> liveYields;
  liveYields.reserve(analysis.liveYieldIndices.size());
  for (unsigned idx : analysis.liveYieldIndices) {
    Value y = yieldOp.getValues()[idx];
    auto it = valueMap.find(y);
    if (it == valueMap.end())
      return std::nullopt;
    liveYields.push_back(it->second);
  }
  return liveYields;
}

static std::string
describeChoice(::llvm::ArrayRef<ChoiceAxis> axes,
               ::llvm::ArrayRef<unsigned> choices,
               llvm::DenseMap<Operation *, unsigned> &nthOp,
               llvm::DenseMap<Operation *, unsigned> &nthMux,
               llvm::DenseMap<Operation *, unsigned> &nthDemux) {
  // Group choices by fabric op for readability. Mode axes (key == "_mode")
  // hold a full DictionaryAttr; we unpack into individual key=val entries.
  llvm::DenseMap<Operation *, std::string> perOp;
  auto appendAttrEntry = [](std::string &slot, ::llvm::StringRef key,
                            Attribute v) {
    if (!slot.empty())
      slot += ",";
    // Check BoolAttr before IntegerAttr because in MLIR BoolAttr is a
    // specialized i1 IntegerAttr.
    if (auto bA = ::mlir::dyn_cast<::mlir::BoolAttr>(v)) {
      slot += key.str() + (bA.getValue() ? "=true" : "=false");
    } else if (auto str = ::mlir::dyn_cast<StringAttr>(v)) {
      slot += key.str() + "=" + str.getValue().str();
    } else if (auto iA = ::mlir::dyn_cast<::mlir::IntegerAttr>(v)) {
      llvm::raw_string_ostream os(slot);
      os << key << "=" << iA.getInt();
    } else {
      slot += key.str() + "=<attr>";
    }
  };
  for (auto [i, axis] : llvm::enumerate(axes)) {
    Attribute v = axis.values[choices[i]];
    std::string &slot = perOp[axis.fabricOp];
    if (axis.key == "_mode") {
      auto dict = ::mlir::cast<DictionaryAttr>(v);
      // Render in canonical order: sel, discard, disconnect.
      auto sel = dict.get("sel");
      auto disc = dict.get("discard");
      auto dis = dict.get("disconnect");
      if (sel)
        appendAttrEntry(slot, "sel", sel);
      if (disc)
        appendAttrEntry(slot, "discard", disc);
      if (dis)
        appendAttrEntry(slot, "disconnect", dis);
    } else {
      appendAttrEntry(slot, axis.key, v);
    }
  }

  std::string s;
  llvm::raw_string_ostream os(s);
  bool first = true;
  if (axes.empty())
    return s;
  // Iterate fabric ops in deterministic body order to preserve readability.
  Block &body = ::mlir::cast<FuOp>(axes.front().fabricOp->getParentOp())
                    .getBody()
                    .front();
  for (Operation &op : body.without_terminator()) {
    auto it = perOp.find(&op);
    if (it == perOp.end())
      continue;
    if (!first)
      os << "; ";
    first = false;
    if (::mlir::isa<::fabric::OpOp>(&op))
      os << "op#" << nthOp[&op] << "{" << it->second << "}";
    else if (::mlir::isa<::fabric::MuxOp>(&op))
      os << "mux#" << nthMux[&op] << "{" << it->second << "}";
    else if (::mlir::isa<::fabric::DemuxOp>(&op))
      os << "demux#" << nthDemux[&op] << "{" << it->second << "}";
  }
  return s;
}

} // namespace

::llvm::SmallVector<FuSubgraphCandidate>
enumerateFuSubgraphs(FuOp fu, ::mlir::ModuleOp module,
                     ::llvm::StringRef baseName,
                     ::llvm::StringRef *unsupported) {
  ::llvm::SmallVector<FuSubgraphCandidate> results;
  MLIRContext *ctx = fu.getContext();
  Block &fuBody = fu.getBody().front();

  // 1. Validate every fabric.op uses only materializable sw symbols.
  // 2. Build choice axes for the entire FU body.
  SmallVector<ChoiceAxis> axes;
  // Per-op ordinal (for human-readable description).
  llvm::DenseMap<Operation *, unsigned> nthOp, nthMux, nthDemux;
  unsigned countOp = 0, countMux = 0, countDemux = 0;
  for (Operation &op : fuBody.without_terminator()) {
    if (auto fop = ::mlir::dyn_cast<OpOp>(&op)) {
      nthOp[&op] = countOp++;
      ArrayAttr opList = fop.getOpList();
      for (Attribute a : opList) {
        StringRef name = ::mlir::cast<FlatSymbolRefAttr>(a).getValue();
        if (!isMaterializable(name)) {
          if (unsupported)
            *unsupported = name;
          return results;
        }
      }
      if (opList.size() > 1) {
        ChoiceAxis axis;
        axis.fabricOp = &op;
        axis.key = "op_sel";
        for (Attribute a : opList) {
          StringRef sym = ::mlir::cast<FlatSymbolRefAttr>(a).getValue();
          axis.values.push_back(StringAttr::get(ctx, sym));
        }
        axes.push_back(std::move(axis));
      }
      Flavor primaryFlavor = opFlavors().lookup(
          ::mlir::cast<FlatSymbolRefAttr>(opList[0]).getValue());
      bool variadic = isVariadicFlavor(primaryFlavor);

      // Collect the hw_params allowed-set for "bitmask" if any. When
      // present, the enumerator iterates only those bitmask values; when
      // absent, every length-M non-zero bitmask is iterated.
      ::llvm::SmallVector<StringRef, 8> bitmaskAllowed;
      bool bitmaskRestricted = false;
      if (auto hp = fop.getHwParamsAttr()) {
        if (hp.size() == 1) {
          if (auto dict = ::mlir::dyn_cast<DictionaryAttr>(hp[0])) {
            for (NamedAttribute na : dict) {
              auto arr = ::mlir::dyn_cast<ArrayAttr>(na.getValue());
              if (!arr)
                continue;
              StringRef key = na.getName().getValue();
              if (variadic && key == "bitmask") {
                bitmaskRestricted = true;
                for (Attribute v : arr) {
                  if (auto s = ::mlir::dyn_cast<StringAttr>(v))
                    bitmaskAllowed.push_back(s.getValue());
                }
                continue;
              }
              ChoiceAxis axis;
              axis.fabricOp = &op;
              axis.key = key.str();
              axis.values.assign(arr.begin(), arr.end());
              axes.push_back(std::move(axis));
            }
          }
        }
      }

      if (variadic) {
        unsigned M = variadicM(primaryFlavor, fop);
        if (M == 0) {
          if (unsupported)
            *unsupported =
                ::mlir::cast<FlatSymbolRefAttr>(opList[0]).getValue();
          return results;
        }
        if (M > kVariadicMaxM) {
          fop.emitWarning("variadic fabric.op port count M=")
              << M << " exceeds the enumerator's hard cap (" << kVariadicMaxM
              << "); skipping this FU";
          if (unsupported)
            *unsupported =
                ::mlir::cast<FlatSymbolRefAttr>(opList[0]).getValue();
          return results;
        }
        ChoiceAxis axis;
        axis.fabricOp = &op;
        axis.key = "bitmask";
        if (bitmaskRestricted) {
          for (StringRef bm : bitmaskAllowed) {
            if (bm.size() != M)
              continue;
            bool ok = true;
            for (char c : bm)
              if (c != '0' && c != '1') {
                ok = false;
                break;
              }
            if (!ok)
              continue;
            if (popcountBitmask(bm) == 0)
              continue;
            axis.values.push_back(StringAttr::get(ctx, bm));
          }
        } else {
          for (uint64_t mask = 1, end = (1ull << M); mask < end; ++mask) {
            std::string s(M, '0');
            for (unsigned b = 0; b < M; ++b)
              if (mask & (1ull << b))
                s[b] = '1';
            axis.values.push_back(StringAttr::get(ctx, s));
          }
        }
        if (axis.values.empty())
          continue; // no legal bitmask -> this FU yields no template
        axes.push_back(std::move(axis));
      }
    } else if (auto m = ::mlir::dyn_cast<MuxOp>(&op)) {
      nthMux[&op] = countMux++;
      // Encode each (sel, discard, disconnect) triple as one full
      // DictionaryAttr in this single axis. Three modes per port plus a
      // disconnect mode: normal sel=i, discard sel=i, disconnect (sel=0).
      ChoiceAxis axis;
      axis.fabricOp = &op;
      axis.key = "_mode";
      auto i32 = IntegerType::get(ctx, 32);
      auto trueA = ::mlir::BoolAttr::get(ctx, true);
      auto falseA = ::mlir::BoolAttr::get(ctx, false);
      auto buildModeDict = [&](unsigned sel, bool discard, bool disconnect) {
        ::llvm::SmallVector<NamedAttribute, 3> e = {
            NamedAttribute(StringAttr::get(ctx, "sel"),
                           ::mlir::IntegerAttr::get(i32, (int64_t)sel)),
            NamedAttribute(StringAttr::get(ctx, "discard"),
                           discard ? trueA : falseA),
            NamedAttribute(StringAttr::get(ctx, "disconnect"),
                           disconnect ? trueA : falseA)};
        return DictionaryAttr::get(ctx, e);
      };
      unsigned N = m.getInputs().size();
      for (unsigned i = 0; i < N; ++i)
        axis.values.push_back(buildModeDict(i, false, false));
      for (unsigned i = 0; i < N; ++i)
        axis.values.push_back(buildModeDict(i, true, false));
      axis.values.push_back(buildModeDict(0, false, true));
      axes.push_back(std::move(axis));
    } else if (auto d = ::mlir::dyn_cast<DemuxOp>(&op)) {
      nthDemux[&op] = countDemux++;
      ChoiceAxis axis;
      axis.fabricOp = &op;
      axis.key = "_mode";
      auto i32 = IntegerType::get(ctx, 32);
      auto trueA = ::mlir::BoolAttr::get(ctx, true);
      auto falseA = ::mlir::BoolAttr::get(ctx, false);
      auto buildModeDict = [&](unsigned sel, bool discard, bool disconnect) {
        ::llvm::SmallVector<NamedAttribute, 3> e = {
            NamedAttribute(StringAttr::get(ctx, "sel"),
                           ::mlir::IntegerAttr::get(i32, (int64_t)sel)),
            NamedAttribute(StringAttr::get(ctx, "discard"),
                           discard ? trueA : falseA),
            NamedAttribute(StringAttr::get(ctx, "disconnect"),
                           disconnect ? trueA : falseA)};
        return DictionaryAttr::get(ctx, e);
      };
      unsigned N = d.getOutputs().size();
      for (unsigned i = 0; i < N; ++i)
        axis.values.push_back(buildModeDict(i, false, false));
      for (unsigned i = 0; i < N; ++i)
        axis.values.push_back(buildModeDict(i, true, false));
      axis.values.push_back(buildModeDict(0, false, true));
      axes.push_back(std::move(axis));
    }
  }

  // Total Cartesian product cardinality.
  uint64_t total = 1;
  for (const ChoiceAxis &a : axes)
    total *= (a.values.empty() ? 1u : a.values.size());

  // Lifted FU input/output types: drive bits<N> -> iN / fN / i1 / none using
  // a flavor-trace through the FU body so float-flavored ops get f-typed
  // sw ports. The full lifted vectors below cover every physical FU port;
  // each per-config materialization narrows them down to the live ports.
  // For inputs, lift the inner block-arg width (which is what the body op
  // actually consumes after FU-boundary high-bit truncation), not the
  // outer operand width. For outputs, lift the yielded source value width:
  // FU/PE boundary widening is a hardware wrapper detail, while the
  // enumerated software subgraph must expose the value type produced by
  // the selected software ops.
  auto liftMap = computePortLiftMap(fu);
  SmallVector<Type, 4> fullSwInputTypes;
  for (auto [i, arg] : llvm::enumerate(fuBody.getArguments())) {
    PortLift k = liftMap.lookup(arg);
    fullSwInputTypes.push_back(
        liftWith(::mlir::cast<BitsType>(arg.getType()).getWidth(), k, ctx));
  }
  SmallVector<Type, 4> fullSwOutputTypes;
  auto yieldOp = ::mlir::cast<::fabric::YieldOp>(fuBody.getTerminator());
  for (Value yielded : yieldOp.getValues()) {
    PortLift k = liftMap.lookup(yielded);
    fullSwOutputTypes.push_back(
        liftWith(::mlir::cast<BitsType>(yielded.getType()).getWidth(), k, ctx));
  }

  Location loc = fu.getLoc();
  OpBuilder modBuilder(module.getBody(), module.getBody()->end());

  for (uint64_t configId = 0; configId < total; ++configId) {
    SmallVector<unsigned, 8> choices(axes.size(), 0);
    uint64_t v = configId;
    for (size_t i = 0; i < axes.size(); ++i) {
      unsigned step = axes[i].values.empty() ? 1u : axes[i].values.size();
      choices[i] = v % step;
      v /= step;
    }

    // Group chosen attributes by fabric op. Axes with key == "_mode"
    // contribute a full DictionaryAttr that is unpacked into individual
    // entries; other axes contribute a single (key, value) pair.
    llvm::DenseMap<Operation *, llvm::StringMap<Attribute>> chosenByOp;
    for (auto [i, axis] : llvm::enumerate(axes)) {
      Attribute v = axis.values[choices[i]];
      if (axis.key == "_mode") {
        auto dict = ::mlir::cast<DictionaryAttr>(v);
        for (NamedAttribute na : dict)
          chosenByOp[axis.fabricOp][na.getName().getValue()] = na.getValue();
      } else {
        chosenByOp[axis.fabricOp][axis.key] = v;
      }
    }

    // Analyze first so we know which FU yield positions remain live in
    // this configuration. The materialized subgraph signature only
    // exposes the live yields; configs with zero live yields are dropped
    // entirely (no 0-result subgraph is emitted).
    auto analysis = analyzeConfig(fu, chosenByOp);
    if (!analysis)
      continue;
    if (analysis->liveYieldIndices.empty())
      continue;

    // Compute which FU input ports remain software-live in this
    // configuration. A block-arg is software-live iff some firing op in
    // the FU body reads it on an active operand index AND that read is
    // not a fabric.mux/demux operating in discard mode. Pass-through
    // chains are followed transitively. Block-args whose only role is to
    // feed a discard-mode mux/demux are NOT software-live: at the
    // hardware level the discard unit drains them, but no value flows
    // into the materialized software subgraph, so including them would
    // pad the subgraph signature with unused parameters and spuriously
    // distinguish otherwise isomorphic templates.
    auto swLive = computeSoftwareLiveValues(fu, chosenByOp, *analysis);
    SmallVector<unsigned, 4> liveInputIndices;
    for (unsigned i = 0, e = fuBody.getNumArguments(); i < e; ++i)
      if (swLive.count(fuBody.getArgument(i)))
        liveInputIndices.push_back(i);

    // Per-config sel-port type override: when a variadic mux/demux fires
    // in this configuration, its sel input's logical type depends on N
    // (the bitmask popcount). The FU input port that feeds that sel
    // must therefore expose i1 (N==2) or index (N>=3) in the
    // materialized subgraph signature, irrespective of the hardware-side
    // bits<...> width. We compute that override here and apply it to the
    // live input types.
    llvm::DenseMap<unsigned, Type> selOverrideByFuArg;
    for (Operation &fopOp : fuBody.without_terminator()) {
      auto fop = ::mlir::dyn_cast<::fabric::OpOp>(&fopOp);
      if (!fop)
        continue;
      Flavor f = flavorOfFabricOp(fop);
      if (f != Flavor::VariadicMuxFlavor && f != Flavor::VariadicDemuxFlavor)
        continue;
      if (!analysis->fires.count(&fopOp))
        continue;
      StringRef bm = getChosenBitmask(&fopOp, chosenByOp);
      unsigned N = popcountBitmask(bm);
      if (N < 2)
        continue;
      Value selInput = fop.getInputs()[0];
      auto ba = ::mlir::dyn_cast<::mlir::BlockArgument>(selInput);
      if (!ba)
        continue; // sel comes from another fabric op; pass-through
      selOverrideByFuArg[ba.getArgNumber()] = variadicMuxLogicalSelType(N, ctx);
    }

    SmallVector<Type, 4> liveInputTypes;
    liveInputTypes.reserve(liveInputIndices.size());
    for (unsigned idx : liveInputIndices) {
      auto it = selOverrideByFuArg.find(idx);
      if (it != selOverrideByFuArg.end())
        liveInputTypes.push_back(it->second);
      else
        liveInputTypes.push_back(fullSwInputTypes[idx]);
    }

    SmallVector<Type, 4> liveOutputTypes;
    liveOutputTypes.reserve(analysis->liveYieldIndices.size());
    for (unsigned idx : analysis->liveYieldIndices)
      liveOutputTypes.push_back(fullSwOutputTypes[idx]);
    auto funcType = FunctionType::get(ctx, liveInputTypes, liveOutputTypes);

    // Build wrapper func with the per-config signature so that, if body
    // materialization fails, we can erase it cleanly.
    std::string fname = (baseName + "_" + std::to_string(results.size())).str();
    auto func = ::mlir::func::FuncOp::create(modBuilder, loc, fname, funcType);
    func.setPrivate();
    Block *funcBody = func.addEntryBlock();
    OpBuilder funcBuilder(funcBody, funcBody->end());

    SmallVector<Value, 4> outerOperands(funcBody->args_begin(),
                                        funcBody->args_end());
    OperationState state(loc, ::dataflow::SubgraphOp::getOperationName());
    state.addOperands(outerOperands);
    state.addTypes(liveOutputTypes);
    ::mlir::Region *body = state.addRegion();
    Block *bodyBlock = new Block();
    body->push_back(bodyBlock);
    SmallVector<Location, 4> argLocs(liveInputTypes.size(), loc);
    bodyBlock->addArguments(liveInputTypes, argLocs);

    auto subgraph =
        ::mlir::cast<::dataflow::SubgraphOp>(funcBuilder.create(state));

    OpBuilder bodyBuilder(bodyBlock, bodyBlock->end());
    auto yields = materializeBodyForConfig(
        fu, liveInputIndices, bodyBlock->getArguments(), bodyBuilder,
        chosenByOp, *analysis, swLive);
    if (!yields) {
      func.erase();
      continue;
    }
    ::dataflow::YieldOp::create(bodyBuilder, loc, *yields);
    ::mlir::func::ReturnOp::create(funcBuilder, loc, subgraph.getResults());

    {
      ::mlir::ScopedDiagnosticHandler capture(
          ctx, [](::mlir::Diagnostic &) { return ::mlir::success(); });
      if (::mlir::failed(::mlir::verify(func))) {
        func.erase();
        continue;
      }
    }

    FuSubgraphCandidate cand;
    cand.wrapper = func;
    cand.subgraph = subgraph;
    cand.configDescription =
        describeChoice(axes, choices, nthOp, nthMux, nthDemux);

    // Materialize per-fabric-op sw_configs as DictionaryAttrs.
    for (auto &kv : chosenByOp) {
      ::llvm::SmallVector<NamedAttribute, 3> entries;
      for (auto &kv2 : kv.second)
        entries.push_back(
            NamedAttribute(StringAttr::get(ctx, kv2.getKey()), kv2.getValue()));
      cand.swConfigsByOp[kv.first] = DictionaryAttr::get(ctx, entries);
    }
    results.push_back(std::move(cand));
  }

  // Dedup pass over the produced candidates. Two configurations that map
  // to graph-isomorphic subgraphs (after mux/demux reduction has already
  // been applied implicitly by materialization) are considered equivalent
  // effective configurations. We keep the FIRST occurrence in
  // configuration-id order (lexicographically smallest sw_configs choice)
  // and erase the wrapper for every later isomorphic clone. This makes
  // multi-mux / multi-demux FUs collapse to one template per distinct
  // effective compute, instead of one per Cartesian-product knob tuple.
  //
  // Design principle: if a fabric.fu produces many software-isomorphic
  // templates under different sw_configs, the FU design itself is
  // flawed -- distinct sw_configs should map to distinct software
  // functions. The dedup deliberately keeps a single representative per
  // isomorphism class; the dropped configs are a smell, not a feature.
  // Tightening the per-config signature (see software-live block-arg
  // computation upstream) shrinks isomorphism-class boundaries and
  // therefore tends to reveal those redundant configs as duplicates.
  ::llvm::SmallVector<FuSubgraphCandidate> deduped;
  deduped.reserve(results.size());
  for (auto &c : results) {
    bool dup = false;
    for (auto &kept : deduped) {
      if (subgraphsIsomorphic(c.subgraph, kept.subgraph)) {
        dup = true;
        break;
      }
    }
    if (dup)
      c.wrapper.erase();
    else
      deduped.push_back(std::move(c));
  }
  // Renumber wrapper names so the post-dedup names are contiguous and
  // stable across runs.
  for (auto [i, c] : ::llvm::enumerate(deduped)) {
    std::string fname = (baseName + "_" + std::to_string(i)).str();
    c.wrapper.setName(fname);
  }
  return deduped;
}

} // namespace fabric

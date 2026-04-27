#include "Fabric/Tech/SubgraphEnumerator.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/Tech/SubgraphMatcher.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
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
  IntArith,    // 2-int-in 1-int-out, no extra attrs
  IntCmp,      // 2-int-in 1-i1-out, requires "predicate" sw_config
  FloatArith,  // 2-float-in 1-float-out, no extra attrs
  FloatCmp,    // 2-float-in 1-i1-out, requires "predicate" sw_config
  IntToFloat,  // 1-int-in 1-float-out
  FloatToInt,  // 1-float-in 1-int-out
  FloatUnary,  // 1-float-in 1-float-out (math.*)
  DataflowStreamFlavor,    // 3-int-in (T) 2-out (T,i1)
  DataflowCarryGateInv,    // dataflow.carry/invariant/gate, polymorphic
                           // shared port treated as integer
  DataflowConstantFlavor,  // dataflow.constant: 1 none-in, 1 typed-out,
                           // const_hex_value sw_config materialized into
                           // an IntegerAttr (or FloatAttr when the result
                           // port is float-flavored).
};

static const llvm::StringMap<Flavor> &opFlavors() {
  static const llvm::StringMap<Flavor> m = []() {
    llvm::StringMap<Flavor> r;
    auto put = [&](StringRef n, Flavor f) { r.insert({n, f}); };

    // Integer arith
    for (StringRef n : {"arith.addi", "arith.subi", "arith.muli",
                        "arith.divsi", "arith.divui", "arith.remsi",
                        "arith.remui", "arith.shli", "arith.shrsi",
                        "arith.shrui", "arith.andi", "arith.ori",
                        "arith.xori", "arith.minsi", "arith.maxsi",
                        "arith.minui", "arith.maxui"})
      put(n, Flavor::IntArith);
    put("arith.cmpi", Flavor::IntCmp);

    // Float arith
    for (StringRef n : {"arith.addf", "arith.subf", "arith.mulf",
                        "arith.divf", "arith.remf",
                        "arith.minimumf", "arith.maximumf"})
      put(n, Flavor::FloatArith);
    put("arith.cmpf", Flavor::FloatCmp);

    // Int<->Float casts
    put("arith.sitofp", Flavor::IntToFloat);
    put("arith.uitofp", Flavor::IntToFloat);
    put("arith.fptosi", Flavor::FloatToInt);
    put("arith.fptoui", Flavor::FloatToInt);

    // Math unary
    for (StringRef n : {"math.sin", "math.cos", "math.tan", "math.sinh",
                        "math.cosh", "math.tanh", "math.exp", "math.exp2",
                        "math.expm1", "math.log", "math.log2", "math.log10",
                        "math.log1p", "math.floor", "math.ceil", "math.round",
                        "math.trunc", "math.roundeven", "math.sqrt",
                        "math.rsqrt", "math.absf", "math.erf"})
      put(n, Flavor::FloatUnary);
    put("math.absi", Flavor::IntArith);

    // Dataflow
    put("dataflow.stream", Flavor::DataflowStreamFlavor);
    put("dataflow.carry", Flavor::DataflowCarryGateInv);
    put("dataflow.invariant", Flavor::DataflowCarryGateInv);
    put("dataflow.gate", Flavor::DataflowCarryGateInv);
    put("dataflow.constant", Flavor::DataflowConstantFlavor);

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
// floating-point? `Unknown` means no inference available; default to Int.
enum class PortLift : uint8_t { Int, Float, Unknown };

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
            if (kk != PortLift::Unknown) { k = kk; break; }
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
            if (kk != PortLift::Unknown) { k = kk; break; }
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
static Attribute parseConstHex(StringRef hex, Type resultTy,
                               MLIRContext *ctx) {
  StringRef body = hex;
  if (body.starts_with("0x") || body.starts_with("0X"))
    body = body.substr(2);
  if (body.empty())
    return nullptr;
  uint64_t v = 0;
  if (body.getAsInteger(16, v))
    return nullptr;
  if (auto intTy = ::mlir::dyn_cast<IntegerType>(resultTy)) {
    return ::mlir::IntegerAttr::get(intTy,
                                     ::llvm::APInt(intTy.getWidth(), v));
  }
  if (auto floatTy = ::mlir::dyn_cast<::mlir::FloatType>(resultTy)) {
    ::llvm::APInt bits(floatTy.getWidth(), v);
    ::llvm::APFloat fp(floatTy.getFloatSemantics(), bits);
    return ::mlir::FloatAttr::get(floatTy, fp);
  }
  return nullptr;
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

// Whether the use port at `idx` of `user` is "active" (its handshake ready
// can go high) under the chosen modes.
static bool useIsActive(Operation *user, unsigned idx,
                         const llvm::DenseSet<Operation *> &fires,
                         const llvm::DenseMap<Operation *,
                                               llvm::StringMap<Attribute>>
                             &chosenByOp) {
  if (::mlir::isa<::fabric::YieldOp>(user))
    return true;
  if (::mlir::isa<::fabric::OpOp>(user))
    return fires.count(user);
  if (::mlir::isa<::fabric::MuxOp>(user)) {
    if (!fires.count(user))
      return false;
    auto [sel, discard, disconnect] =
        decodeMuxLikeMode(chosenByOp.lookup(user));
    (void)discard;
    if (disconnect)
      return false;
    return idx == sel;
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

static bool allUsesActive(Value v,
                           const llvm::DenseSet<Operation *> &fires,
                           const llvm::DenseMap<Operation *,
                                                 llvm::StringMap<Attribute>>
                               &chosenByOp) {
  for (::mlir::OpOperand &use : v.getUses())
    if (!useIsActive(use.getOwner(), use.getOperandNumber(), fires,
                      chosenByOp))
      return false;
  return true;
}

static bool opCanFire(Operation *op,
                       const llvm::DenseSet<Value> &alive,
                       const llvm::DenseSet<Value> &demanded,
                       const llvm::DenseMap<Operation *,
                                             llvm::StringMap<Attribute>>
                           &chosenByOp) {
  if (auto fop = ::mlir::dyn_cast<::fabric::OpOp>(op)) {
    for (Value in : fop.getInputs())
      if (!alive.count(in))
        return false;
    bool any = false;
    for (Value out : fop.getOutputs())
      if (demanded.count(out)) { any = true; break; }
    return any;
  }
  if (auto m = ::mlir::dyn_cast<::fabric::MuxOp>(op)) {
    auto [sel, discard, disconnect] =
        decodeMuxLikeMode(chosenByOp.lookup(op));
    if (disconnect)
      return false;
    if (!alive.count(m.getInputs()[sel]))
      return false;
    if (discard)
      return true;
    return demanded.count(m.getOutput());
  }
  if (auto d = ::mlir::dyn_cast<::fabric::DemuxOp>(op)) {
    auto [sel, discard, disconnect] =
        decodeMuxLikeMode(chosenByOp.lookup(op));
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
    const llvm::DenseMap<Operation *, llvm::StringMap<Attribute>>
        &chosenByOp) {
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
        bool anyOut = false;
        for (Value out : fop.getOutputs())
          if (demanded.count(out)) { anyOut = true; break; }
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

  // 3. Validate. Yield positions are allowed to be inactive in this
  // config (e.g. an unselected fabric.demux output): such positions are
  // simply dropped from the materialized subgraph signature. Every other
  // demanded value (e.g. a discard-mode mux input) must be alive.
  llvm::DenseSet<Value> yieldVals;
  for (Value y : yieldOp.getValues())
    yieldVals.insert(y);
  for (Value v : demanded)
    if (!alive.count(v) && !yieldVals.count(v))
      return std::nullopt;

  ConfigAnalysis a;
  a.demanded = std::move(demanded);
  a.alive = std::move(alive);
  a.fires = std::move(fires);
  for (auto [i, y] : llvm::enumerate(yieldOp.getValues()))
    if (a.alive.count(y))
      a.liveYieldIndices.push_back(static_cast<unsigned>(i));
  return a;
}

// Materialize the sw subgraph body for a configuration whose analysis has
// already succeeded. `liveInputIndices` lists the FU input port positions
// that remain live in this config, in ascending order; `subBlockArgs` is
// the matching list of sw block arguments (same length). Returns the
// materialized SSA values for the live yield positions in the same order
// as `analysis.liveYieldIndices`, or std::nullopt when an inner sw op
// cannot be built (e.g. unsupported const_hex_value width).
static std::optional<SmallVector<Value, 4>> materializeBodyForConfig(
    FuOp fu, ::llvm::ArrayRef<unsigned> liveInputIndices,
    ::mlir::ValueRange subBlockArgs, OpBuilder &builder,
    const llvm::DenseMap<Operation *, llvm::StringMap<Attribute>> &chosenByOp,
    const ConfigAnalysis &analysis) {
  Block &fuBody = fu.getBody().front();
  MLIRContext *ctx = fu.getContext();
  auto yieldOp = ::mlir::cast<::fabric::YieldOp>(fuBody.getTerminator());

  ValueMap valueMap;
  for (auto [pos, fuIdx] : llvm::enumerate(liveInputIndices)) {
    Value fuArg = fuBody.getArgument(fuIdx);
    valueMap[fuArg] = subBlockArgs[pos];
  }

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

      SmallVector<Value, 4> swInputs;
      swInputs.reserve(fop.getInputs().size());
      for (Value in : fop.getInputs()) {
        auto it = valueMap.find(in);
        if (it == valueMap.end())
          return std::nullopt;
        swInputs.push_back(it->second);
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
      auto it = valueMap.find(m.getInputs()[sel]);
      if (it != valueMap.end() && analysis.alive.count(m.getOutput()))
        valueMap[m.getOutput()] = it->second;
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
      auto it = valueMap.find(d.getInput());
      if (it != valueMap.end() && analysis.alive.count(d.getOutputs()[sel]))
        valueMap[d.getOutputs()[sel]] = it->second;
      continue;
    }
  }

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

static std::string describeChoice(::llvm::ArrayRef<ChoiceAxis> axes,
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
      if (auto hp = fop.getHwParamsAttr()) {
        if (hp.size() == 1) {
          if (auto dict = ::mlir::dyn_cast<DictionaryAttr>(hp[0])) {
            for (NamedAttribute na : dict) {
              auto arr = ::mlir::dyn_cast<ArrayAttr>(na.getValue());
              if (!arr)
                continue;
              ChoiceAxis axis;
              axis.fabricOp = &op;
              axis.key = na.getName().getValue().str();
              axis.values.assign(arr.begin(), arr.end());
              axes.push_back(std::move(axis));
            }
          }
        }
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
  auto liftMap = computePortLiftMap(fu);
  SmallVector<Type, 4> fullSwInputTypes;
  for (auto [i, t] : llvm::enumerate(fu.getInputs().getTypes())) {
    PortLift k = liftMap.lookup(fuBody.getArgument(i));
    fullSwInputTypes.push_back(
        liftWith(::mlir::cast<BitsType>(t).getWidth(), k, ctx));
  }
  SmallVector<Type, 4> fullSwOutputTypes;
  auto yieldOp = ::mlir::cast<::fabric::YieldOp>(fuBody.getTerminator());
  for (auto [i, t] : llvm::enumerate(fu.getResultTypes())) {
    PortLift k = liftMap.lookup(yieldOp.getValues()[i]);
    fullSwOutputTypes.push_back(
        liftWith(::mlir::cast<BitsType>(t).getWidth(), k, ctx));
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

    // Compute which FU input ports remain live in this configuration. An
    // FU input port is live iff its corresponding block argument is in the
    // alive set. Iterate in ascending FU-input-port order so the resulting
    // subgraph signature is deterministic.
    SmallVector<unsigned, 4> liveInputIndices;
    for (unsigned i = 0, e = fuBody.getNumArguments(); i < e; ++i)
      if (analysis->alive.count(fuBody.getArgument(i)))
        liveInputIndices.push_back(i);

    SmallVector<Type, 4> liveInputTypes;
    liveInputTypes.reserve(liveInputIndices.size());
    for (unsigned idx : liveInputIndices)
      liveInputTypes.push_back(fullSwInputTypes[idx]);

    SmallVector<Type, 4> liveOutputTypes;
    liveOutputTypes.reserve(analysis->liveYieldIndices.size());
    for (unsigned idx : analysis->liveYieldIndices)
      liveOutputTypes.push_back(fullSwOutputTypes[idx]);
    auto funcType = FunctionType::get(ctx, liveInputTypes, liveOutputTypes);

    // Build wrapper func with the per-config signature so that, if body
    // materialization fails, we can erase it cleanly.
    std::string fname = (baseName + "_" + std::to_string(results.size())).str();
    auto func = modBuilder.create<::mlir::func::FuncOp>(loc, fname, funcType);
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
    auto yields = materializeBodyForConfig(fu, liveInputIndices,
                                           bodyBlock->getArguments(),
                                           bodyBuilder, chosenByOp,
                                           *analysis);
    if (!yields) {
      func.erase();
      continue;
    }
    bodyBuilder.create<::dataflow::YieldOp>(loc, *yields);
    funcBuilder.create<::mlir::func::ReturnOp>(loc, subgraph.getResults());

    FuSubgraphCandidate cand;
    cand.wrapper = func;
    cand.subgraph = subgraph;
    cand.configDescription =
        describeChoice(axes, choices, nthOp, nthMux, nthDemux);

    // Materialize per-fabric-op sw_configs as DictionaryAttrs.
    for (auto &kv : chosenByOp) {
      ::llvm::SmallVector<NamedAttribute, 3> entries;
      for (auto &kv2 : kv.second)
        entries.push_back(NamedAttribute(StringAttr::get(ctx, kv2.getKey()),
                                          kv2.getValue()));
      cand.swConfigsByOp[kv.first] =
          DictionaryAttr::get(ctx, entries);
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

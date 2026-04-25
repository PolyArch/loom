#include "Fabric/Tech/SubgraphEnumerator.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricTypes.h"
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
  DataflowStreamFlavor, // 3-int-in (T) 2-out (T,i1), step_op + cont_cond
  DataflowCarryGateInv, // dataflow.carry/invariant/gate, polymorphic; treat
                        // shared port as integer for v2 (it carries a
                        // generic value; loom v2 instantiates it as iN)
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

    return r;
  }();
  return m;
}

static bool isV2Materializable(StringRef opSym) {
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

// Build the body of one configuration. `subBlockArgs` is the list of
// dataflow.subgraph block arguments. For each fabric op encountered we
// look up its chosen sw_config attributes from `chosenByOp`. Returns the
// mapped yield values on success or std::nullopt if any required path is
// dead.
static std::optional<SmallVector<Value, 4>>
buildBodyForConfig(FuOp fu, ::mlir::ValueRange subBlockArgs,
                   OpBuilder &builder,
                   const llvm::DenseMap<Operation *,
                                        llvm::StringMap<Attribute>> &chosenByOp) {
  ValueMap valueMap;
  Block &fuBody = fu.getBody().front();
  MLIRContext *ctx = fu.getContext();

  for (auto [fuArg, subArg] :
       llvm::zip(fuBody.getArguments(), subBlockArgs))
    valueMap[fuArg] = subArg;

  for (Operation &op : fuBody.without_terminator()) {
    if (auto fop = ::mlir::dyn_cast<::fabric::OpOp>(&op)) {
      // Determine chosen op symbol.
      const auto &chosen = chosenByOp.lookup(&op);
      ArrayAttr opList = fop.getOpList();
      StringRef sym;
      auto opSelIt = chosen.find("op_sel");
      if (opSelIt != chosen.end()) {
        sym = ::mlir::cast<StringAttr>(opSelIt->second).getValue();
      } else {
        sym = ::mlir::cast<FlatSymbolRefAttr>(opList[0]).getValue();
      }

      Flavor flavor = opFlavors().lookup(sym);

      // Collect sw inputs; bail out on dead input by leaving outputs dead.
      bool anyDead = false;
      SmallVector<Value, 4> swInputs;
      swInputs.reserve(fop.getInputs().size());
      for (Value in : fop.getInputs()) {
        auto it = valueMap.find(in);
        if (it == valueMap.end()) {
          anyDead = true;
          break;
        }
        swInputs.push_back(it->second);
      }
      if (anyDead)
        continue;

      // Build sw result types using flavor-aware lifting.
      SmallVector<Type, 2> swResultTypes;
      swResultTypes.reserve(fop.getOutputs().size());
      for (auto [i, t] : llvm::enumerate(fop.getResultTypes())) {
        unsigned w = ::mlir::cast<BitsType>(t).getWidth();
        Type ty = liftFor(flavor, w, /*isOutput=*/true, i, ctx);
        if (!ty)
          return std::nullopt; // unsupported width for flavor
        swResultTypes.push_back(ty);
      }

      // Build the sw op.
      OperationState state(fop.getLoc(), sym);
      state.addOperands(swInputs);
      state.addTypes(swResultTypes);
      // Inject attributes from chosen sw_configs (excluding op_sel, which
      // is consumed locally).
      for (auto &kv : chosen) {
        if (kv.getKey() == "op_sel")
          continue;
        Attribute conv = toSwAttr(sym, kv.getKey(), kv.getValue(), ctx);
        if (!conv)
          return std::nullopt;
        state.addAttribute(kv.getKey(), conv);
      }
      Operation *swOp = builder.create(state);
      for (auto [fuOut, swOut] :
           llvm::zip(fop.getOutputs(), swOp->getResults()))
        valueMap[fuOut] = swOut;
      continue;
    }
    if (auto m = ::mlir::dyn_cast<::fabric::MuxOp>(&op)) {
      const auto &chosen = chosenByOp.lookup(&op);
      auto sel =
          ::mlir::cast<::mlir::IntegerAttr>(chosen.lookup("sel")).getInt();
      Value src = m.getInputs()[sel];
      auto it = valueMap.find(src);
      if (it != valueMap.end())
        valueMap[m.getOutput()] = it->second;
      continue;
    }
    if (auto d = ::mlir::dyn_cast<::fabric::DemuxOp>(&op)) {
      const auto &chosen = chosenByOp.lookup(&op);
      auto sel =
          ::mlir::cast<::mlir::IntegerAttr>(chosen.lookup("sel")).getInt();
      auto it = valueMap.find(d.getInput());
      Value liveSrc = (it != valueMap.end()) ? it->second : Value();
      for (unsigned k = 0; k < d.getOutputs().size(); ++k) {
        if (k == (unsigned)sel && liveSrc)
          valueMap[d.getOutputs()[k]] = liveSrc;
      }
      continue;
    }
  }

  auto yieldOp = ::mlir::cast<::fabric::YieldOp>(fuBody.getTerminator());
  SmallVector<Value, 4> yields;
  yields.reserve(yieldOp.getValues().size());
  for (Value y : yieldOp.getValues()) {
    auto it = valueMap.find(y);
    if (it == valueMap.end())
      return std::nullopt;
    yields.push_back(it->second);
  }
  return yields;
}

static std::string describeChoice(::llvm::ArrayRef<ChoiceAxis> axes,
                                  ::llvm::ArrayRef<unsigned> choices,
                                  llvm::DenseMap<Operation *, unsigned> &nthOp,
                                  llvm::DenseMap<Operation *, unsigned> &nthMux,
                                  llvm::DenseMap<Operation *, unsigned> &nthDemux) {
  // Group choices by fabric op for readability.
  llvm::DenseMap<Operation *, std::string> perOp;
  for (auto [i, axis] : llvm::enumerate(axes)) {
    Attribute v = axis.values[choices[i]];
    std::string &slot = perOp[axis.fabricOp];
    if (!slot.empty())
      slot += ",";
    if (auto str = ::mlir::dyn_cast<StringAttr>(v)) {
      slot += axis.key + "=" + str.getValue().str();
    } else if (auto i = ::mlir::dyn_cast<::mlir::IntegerAttr>(v)) {
      llvm::raw_string_ostream os(slot);
      // append - we already prepended above; use += instead.
      os << axis.key << "=" << i.getInt();
    } else {
      slot += axis.key + "=<attr>";
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

  // 1. Validate every fabric.op uses only v2-materializable sw symbols.
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
        if (!isV2Materializable(name)) {
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
      ChoiceAxis axis;
      axis.fabricOp = &op;
      axis.key = "sel";
      for (unsigned i = 0; i < m.getInputs().size(); ++i)
        axis.values.push_back(::mlir::IntegerAttr::get(
            IntegerType::get(ctx, 32), (int64_t)i));
      axes.push_back(std::move(axis));
    } else if (auto d = ::mlir::dyn_cast<DemuxOp>(&op)) {
      nthDemux[&op] = countDemux++;
      ChoiceAxis axis;
      axis.fabricOp = &op;
      axis.key = "sel";
      for (unsigned i = 0; i < d.getOutputs().size(); ++i)
        axis.values.push_back(::mlir::IntegerAttr::get(
            IntegerType::get(ctx, 32), (int64_t)i));
      axes.push_back(std::move(axis));
    }
  }

  // Total Cartesian product cardinality.
  uint64_t total = 1;
  for (const ChoiceAxis &a : axes)
    total *= (a.values.empty() ? 1u : a.values.size());

  // Lifted FU input/output types: drive bits<N> -> iN / fN / i1 / none using
  // a flavor-trace through the FU body so float-flavored ops get f-typed
  // sw ports.
  auto liftMap = computePortLiftMap(fu);
  SmallVector<Type, 4> swInputTypes;
  for (auto [i, t] : llvm::enumerate(fu.getInputs().getTypes())) {
    PortLift k = liftMap.lookup(fuBody.getArgument(i));
    swInputTypes.push_back(
        liftWith(::mlir::cast<BitsType>(t).getWidth(), k, ctx));
  }
  SmallVector<Type, 4> swOutputTypes;
  auto yieldOp = ::mlir::cast<::fabric::YieldOp>(fuBody.getTerminator());
  for (auto [i, t] : llvm::enumerate(fu.getResultTypes())) {
    PortLift k = liftMap.lookup(yieldOp.getValues()[i]);
    swOutputTypes.push_back(
        liftWith(::mlir::cast<BitsType>(t).getWidth(), k, ctx));
  }

  Location loc = fu.getLoc();
  OpBuilder modBuilder(module.getBody(), module.getBody()->end());
  auto funcType = FunctionType::get(ctx, swInputTypes, swOutputTypes);

  for (uint64_t configId = 0; configId < total; ++configId) {
    SmallVector<unsigned, 8> choices(axes.size(), 0);
    uint64_t v = configId;
    for (size_t i = 0; i < axes.size(); ++i) {
      unsigned step = axes[i].values.empty() ? 1u : axes[i].values.size();
      choices[i] = v % step;
      v /= step;
    }

    // Group chosen attributes by fabric op.
    llvm::DenseMap<Operation *, llvm::StringMap<Attribute>> chosenByOp;
    for (auto [i, axis] : llvm::enumerate(axes))
      chosenByOp[axis.fabricOp][axis.key] = axis.values[choices[i]];

    // Build wrapper func first so that if body materialization fails we can
    // erase it cleanly.
    std::string fname = (baseName + "_" + std::to_string(results.size())).str();
    auto func = modBuilder.create<::mlir::func::FuncOp>(loc, fname, funcType);
    func.setPrivate();
    Block *funcBody = func.addEntryBlock();
    OpBuilder funcBuilder(funcBody, funcBody->end());

    SmallVector<Value, 4> outerOperands(funcBody->args_begin(),
                                         funcBody->args_end());
    OperationState state(loc, ::dataflow::SubgraphOp::getOperationName());
    state.addOperands(outerOperands);
    state.addTypes(swOutputTypes);
    ::mlir::Region *body = state.addRegion();
    Block *bodyBlock = new Block();
    body->push_back(bodyBlock);
    SmallVector<Location, 4> argLocs(swInputTypes.size(), loc);
    bodyBlock->addArguments(swInputTypes, argLocs);

    auto subgraph =
        ::mlir::cast<::dataflow::SubgraphOp>(funcBuilder.create(state));

    OpBuilder bodyBuilder(bodyBlock, bodyBlock->end());
    auto yields = buildBodyForConfig(fu, bodyBlock->getArguments(),
                                     bodyBuilder, chosenByOp);
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

  return results;
}

} // namespace fabric

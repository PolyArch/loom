#include "Fabric/Tech/SubgraphEnumerator.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

namespace fabric {
namespace {

using ::mlir::ArrayAttr;
using ::mlir::Block;
using ::mlir::FlatSymbolRefAttr;
using ::mlir::FunctionType;
using ::mlir::IntegerType;
using ::mlir::Location;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::NoneType;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::OperationState;
using ::mlir::SmallVector;
using ::mlir::StringRef;
using ::mlir::Type;
using ::mlir::Value;

// V1 op support: pure integer 2-in-1-out ops with no extra attributes.
static bool isV1MaterializableOp(StringRef name) {
  static const llvm::StringSet<> set = {
      "arith.addi",  "arith.subi",  "arith.muli",  "arith.divsi",
      "arith.divui", "arith.remsi", "arith.remui", "arith.shli",
      "arith.shrsi", "arith.shrui", "arith.andi",  "arith.ori",
      "arith.xori",  "arith.minsi", "arith.maxsi", "arith.minui",
      "arith.maxui",
  };
  return set.contains(name);
}

static Type bitsToSwType(BitsType bt, MLIRContext *ctx) {
  unsigned w = bt.getWidth();
  if (w == 0)
    return NoneType::get(ctx);
  return IntegerType::get(ctx, w);
}

using ValueMap = llvm::DenseMap<Value, Value>;

// Materialize body for one configuration. `subBlockArgs` is the list of
// dataflow.subgraph's entry block arguments (already lifted to sw types).
// Returns the mapped yield values on success or std::nullopt if a required
// path is dead.
static std::optional<SmallVector<Value, 4>>
buildBodyForConfig(FuOp fu, ::mlir::ValueRange subBlockArgs,
                   OpBuilder &builder,
                   ::llvm::ArrayRef<Operation *> configurableOps,
                   ::llvm::ArrayRef<unsigned> choices) {
  ValueMap valueMap;
  Block &fuBody = fu.getBody().front();

  for (auto [fuArg, subArg] :
       llvm::zip(fuBody.getArguments(), subBlockArgs))
    valueMap[fuArg] = subArg;

  unsigned configIdx = 0;
  for (Operation &op : fuBody.without_terminator()) {
    if (auto fop = ::mlir::dyn_cast<::fabric::OpOp>(&op)) {
      // If any input is dead in this config, the fabric.op exists in the
      // hardware but never fires - we leave its outputs unmapped (dead) and
      // proceed. This is what makes configs like Example 2's demux.sel=0
      // valid: addi has a dead input but mul reaches mux through demux #0.
      bool anyDeadInput = false;
      SmallVector<Value, 4> swInputs;
      swInputs.reserve(fop.getInputs().size());
      for (Value in : fop.getInputs()) {
        auto it = valueMap.find(in);
        if (it == valueMap.end()) {
          anyDeadInput = true;
          break;
        }
        swInputs.push_back(it->second);
      }
      ArrayAttr opList = fop.getOpList();
      auto sym =
          ::mlir::cast<FlatSymbolRefAttr>(opList[choices[configIdx++]])
              .getValue();
      if (anyDeadInput) {
        // Outputs remain unmapped -> dead.
        continue;
      }

      SmallVector<Type, 2> swResultTypes;
      swResultTypes.reserve(fop.getOutputs().size());
      for (Type t : fop.getResultTypes())
        swResultTypes.push_back(
            bitsToSwType(::mlir::cast<BitsType>(t), fu.getContext()));

      OperationState state(fop.getLoc(), sym);
      state.addOperands(swInputs);
      state.addTypes(swResultTypes);
      Operation *swOp = builder.create(state);
      for (auto [fuOut, swOut] :
           llvm::zip(fop.getOutputs(), swOp->getResults()))
        valueMap[fuOut] = swOut;
      continue;
    }
    if (auto m = ::mlir::dyn_cast<::fabric::MuxOp>(&op)) {
      unsigned sel = choices[configIdx++];
      Value src = m.getInputs()[sel];
      auto it = valueMap.find(src);
      if (it != valueMap.end())
        valueMap[m.getOutput()] = it->second;
      continue;
    }
    if (auto d = ::mlir::dyn_cast<::fabric::DemuxOp>(&op)) {
      unsigned sel = choices[configIdx++];
      auto it = valueMap.find(d.getInput());
      Value liveSrc = (it != valueMap.end()) ? it->second : Value();
      for (unsigned k = 0; k < d.getOutputs().size(); ++k) {
        if (k == sel && liveSrc)
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

static std::string describeConfig(::llvm::ArrayRef<Operation *> configurableOps,
                                  ::llvm::ArrayRef<unsigned> choices) {
  std::string s;
  llvm::raw_string_ostream os(s);
  unsigned opIdx = 0, muxIdx = 0, demuxIdx = 0, configIdx = 0;
  bool first = true;
  for (Operation *op : configurableOps) {
    if (!first)
      os << "; ";
    first = false;
    unsigned ch = choices[configIdx++];
    if (auto fop = ::mlir::dyn_cast<::fabric::OpOp>(op)) {
      auto sym =
          ::mlir::cast<FlatSymbolRefAttr>(fop.getOpList()[ch]).getValue();
      os << "op#" << opIdx++ << "=" << sym;
    } else if (::mlir::isa<::fabric::MuxOp>(op)) {
      os << "mux#" << muxIdx++ << ".sel=" << ch;
    } else if (::mlir::isa<::fabric::DemuxOp>(op)) {
      os << "demux#" << demuxIdx++ << ".sel=" << ch;
    }
  }
  return s;
}

} // namespace

llvm::SmallVector<FuSubgraphCandidate>
enumerateFuSubgraphs(FuOp fu, ::mlir::ModuleOp module,
                     ::llvm::StringRef baseName,
                     ::llvm::StringRef *unsupported) {
  llvm::SmallVector<FuSubgraphCandidate> results;
  ::mlir::MLIRContext *ctx = fu.getContext();
  Block &fuBody = fu.getBody().front();

  SmallVector<Operation *> configurableOps;
  SmallVector<unsigned> choiceCounts;
  for (Operation &op : fuBody.without_terminator()) {
    if (auto fop = ::mlir::dyn_cast<OpOp>(&op)) {
      for (::mlir::Attribute a : fop.getOpList()) {
        StringRef name = ::mlir::cast<FlatSymbolRefAttr>(a).getValue();
        if (!isV1MaterializableOp(name)) {
          if (unsupported)
            *unsupported = name;
          return results;
        }
      }
      configurableOps.push_back(&op);
      choiceCounts.push_back(fop.getOpList().size());
    } else if (auto m = ::mlir::dyn_cast<MuxOp>(&op)) {
      configurableOps.push_back(&op);
      choiceCounts.push_back(m.getInputs().size());
    } else if (auto d = ::mlir::dyn_cast<DemuxOp>(&op)) {
      configurableOps.push_back(&op);
      choiceCounts.push_back(d.getOutputs().size());
    }
  }

  uint64_t total = 1;
  for (unsigned c : choiceCounts)
    total *= (c == 0 ? 1u : c);

  SmallVector<Type, 4> swInputTypes;
  for (Type t : fu.getInputs().getTypes())
    swInputTypes.push_back(bitsToSwType(::mlir::cast<BitsType>(t), ctx));
  SmallVector<Type, 4> swOutputTypes;
  for (Type t : fu.getResultTypes())
    swOutputTypes.push_back(bitsToSwType(::mlir::cast<BitsType>(t), ctx));

  Location loc = fu.getLoc();
  OpBuilder modBuilder(module.getBody(), module.getBody()->end());
  auto funcType = FunctionType::get(ctx, swInputTypes, swOutputTypes);

  for (uint64_t configId = 0; configId < total; ++configId) {
    SmallVector<unsigned, 4> choices;
    uint64_t v = configId;
    choices.reserve(choiceCounts.size());
    for (unsigned c : choiceCounts) {
      unsigned step = (c == 0 ? 1u : c);
      choices.push_back(v % step);
      v /= step;
    }

    // Build the wrapper func.
    std::string fname = (baseName + "_" + std::to_string(results.size())).str();
    auto func = modBuilder.create<::mlir::func::FuncOp>(loc, fname, funcType);
    func.setPrivate();
    Block *funcBody = func.addEntryBlock();
    OpBuilder funcBuilder(funcBody, funcBody->end());

    // Build the dataflow.subgraph with the func args as outer operands.
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
                                     bodyBuilder, configurableOps, choices);
    if (!yields) {
      // Drop this configuration; tear down the func wrapper.
      func.erase();
      // Re-build the next func name index using results.size() (unchanged).
      continue;
    }
    bodyBuilder.create<::dataflow::YieldOp>(loc, *yields);

    // func.return propagating subgraph results.
    funcBuilder.create<::mlir::func::ReturnOp>(loc, subgraph.getResults());

    // Rename the func now that we know its real index. (We initially used
    // results.size() which is correct only after pushing; but since dropping
    // bumps it back, the name we assigned is still consistent at insert time.)

    FuSubgraphCandidate cand;
    cand.wrapper = func;
    cand.subgraph = subgraph;
    cand.configDescription = describeConfig(configurableOps, choices);
    results.push_back(std::move(cand));
  }

  return results;
}

} // namespace fabric

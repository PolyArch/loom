// MCS strategy: cost-prioritized MCES skeleton synthesis across an input
// group.
//
// The primary path generates local MCES candidates first: a lock-step
// candidate for isomorphic DAGs and a positional pure-DAG shared-prefix
// candidate with at most one per-input tail behind fabric.demux/fabric.mux
// hardware. Candidate FUs are accepted only after MLIR verification and
// CoverageVerifier prove that enumerating the FU covers every input.
//
// Stateful graph-region inputs are handed to the existing Tier-C-aware
// compatibility path so accumulator coverage remains stable while the
// pure-DAG MCES path handles divergent tails that Incremental cannot
// express. Both paths poll the timeout budget, count generated
// candidates against candidate_cap, and rank successful candidates by
// CostModel.
//
// Spec source: `docs/spec-generalize-subgraphs-to-fu.md`, sections
// "Strategy: mcs" and "Acceptance criteria (mcs)".

#include "Fabric/Tech/Synthesizer/MCS.h"

#include "Common/HwShareGroup.h"
#include "Common/IndexWidth.h"
#include "Common/SynthConfig.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/Tech/Synthesizer/Alignment.h"
#include "Fabric/Tech/Synthesizer/Anchor.h"
#include "Fabric/Tech/Synthesizer/CostModel.h"
#include "Fabric/Tech/Synthesizer/CoverageVerifier.h"
#include "Fabric/Tech/Synthesizer/HwParams.h"
#include "Fabric/Tech/Synthesizer/Incremental.h"
#include "Fabric/Tech/Synthesizer/Parallel.h"
#include "Fabric/Tech/Synthesizer/Synthesizer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <utility>

namespace loom::fabric::tech {

namespace {

using Clock = std::chrono::steady_clock;

bool noteDeadline(std::atomic<bool> &deadlineExpired,
                  Clock::time_point deadline) {
  if (Clock::now() < deadline)
    return false;
  deadlineExpired.store(true, std::memory_order_relaxed);
  return true;
}

//===----------------------------------------------------------------------===//
// Per-branch record carried back from the parallel worker. Mirrors the
// IncrementalRandom handoff layout: wrapper text + cost + failure
// metadata so the caller can reparse / rank without reaching back into
// the worker's defunct sub-context.
//===----------------------------------------------------------------------===//

struct BranchHandoff {
  std::size_t branchIndex = 0;
  bool succeeded = false;
  // Empty unless `succeeded` is true.
  std::string wrapperIR;
  // Cost computed in the sub-context (cost is a pure function over a
  // wrapper FU, so the value is independent of context identity).
  double cost = 0.0;
  // Failure metadata for branches that did not succeed.
  SynthFailureReason failureReason = SynthFailureReason::None;
  ::llvm::SmallVector<::std::string, 4> notes;
  CoverageReport coverage;
  // True iff this branch was skipped because the global deadline had
  // already elapsed. Used to distinguish `timeout` from inner
  // strategy failures when classifying a no-success outcome.
  bool deadlineSkipped = false;
};

struct CandidateRecord {
  std::size_t candidateIndex = 0;
  double cost = 0.0;
  bool localWrapper = true;
  ::mlir::OwningOpRef<::fabric::ModuleOp> wrapper;
  std::string wrapperIR;
  ::llvm::SmallVector<::std::string, 4> notes;
  CoverageReport coverage;
};

//===----------------------------------------------------------------------===//
// FU lookup helper. Mirrors the same one used by Incremental and
// IncrementalRandom; kept local to avoid linking against either.
//===----------------------------------------------------------------------===//

::fabric::FuOp innerFuOf(::fabric::ModuleOp wrapper) {
  if (!wrapper)
    return {};
  ::fabric::FuOp found;
  wrapper.walk([&](::fabric::FuOp fu) {
    if (!found)
      found = fu;
  });
  return found;
}

CoverageReport verifyDetachedWrapper(::fabric::ModuleOp wrapper,
                                      const SynthInputs &inputs,
                                      const ::loom::SynthConfig &cfg) {
  CoverageReport report;
  if (!wrapper)
    return report;
  ::mlir::Location loc = ::mlir::UnknownLoc::get(wrapper.getContext());
  ::mlir::OwningOpRef<::mlir::ModuleOp> host(
      ::mlir::ModuleOp::create(loc));
  ::mlir::OpBuilder hostBuilder(host->getBodyRegion());
  ::mlir::Operation *clonedWrapper =
      hostBuilder.clone(*wrapper.getOperation());
  CoverageVerifier verifier(cfg);
  return verifier.verify(
      innerFuOf(::mlir::cast<::fabric::ModuleOp>(clonedWrapper)),
      inputs.subgraphs);
}

std::optional<CandidateRecord>
scoreLocalCandidate(::mlir::OwningOpRef<::fabric::ModuleOp> wrapper,
                    const SynthInputs &inputs,
                    const ::loom::SynthConfig &cfg,
                    std::size_t candidateIndex,
                    ::llvm::ArrayRef<::std::string> notes,
                    std::atomic<bool> &deadlineExpired,
                    Clock::time_point deadline) {
  if (!wrapper)
    return std::nullopt;
  if (noteDeadline(deadlineExpired, deadline))
    return std::nullopt;
  if (::mlir::failed(::mlir::verify(wrapper.get())))
    return std::nullopt;
  if (noteDeadline(deadlineExpired, deadline))
    return std::nullopt;
  CoverageReport coverage = verifyDetachedWrapper(wrapper.get(), inputs, cfg);
  if (noteDeadline(deadlineExpired, deadline))
    return std::nullopt;
  if (!coverage.allCovered())
    return std::nullopt;
  ::fabric::FuOp fu = innerFuOf(wrapper.get());
  if (!fu)
    return std::nullopt;
  CostModel cm(cfg);
  CandidateRecord record;
  record.candidateIndex = candidateIndex;
  record.cost = cm.evaluate(fu);
  record.localWrapper = true;
  record.wrapper = std::move(wrapper);
  record.notes.assign(notes.begin(), notes.end());
  record.coverage = std::move(coverage);
  return record;
}

//===----------------------------------------------------------------------===//
// Set up a self-contained sub-MLIRContext. Sub-contexts inherit no
// dialects from their parent so we explicitly load the ones the
// Incremental strategy and the lit-test inputs require.
//===----------------------------------------------------------------------===//

void loadStandardDialects(::mlir::MLIRContext &ctx) {
  ::mlir::DialectRegistry registry;
  registry.insert<::mlir::func::FuncDialect, ::dataflow::DataflowDialect,
                  ::fabric::FabricDialect, ::mlir::arith::ArithDialect,
                  ::mlir::math::MathDialect>();
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
}

//===----------------------------------------------------------------------===//
// Real DAG MCES candidate for shared-prefix / divergent-tail inputs.
//===----------------------------------------------------------------------===//

unsigned bitWidthOf(::mlir::Type t) {
  if (auto i = ::llvm::dyn_cast<::mlir::IntegerType>(t))
    return i.getWidth();
  if (auto f = ::llvm::dyn_cast<::mlir::FloatType>(t))
    return f.getWidth();
  if (::llvm::isa<::mlir::IndexType>(t))
    return ::loom::getIndexWidth();
  if (::llvm::isa<::mlir::NoneType>(t))
    return 0;
  return 0;
}

::mlir::ArrayAttr sortedOpList(::mlir::MLIRContext *ctx,
                               const ::std::set<::std::string> &names) {
  ::llvm::SmallVector<::mlir::Attribute, 4> attrs;
  attrs.reserve(names.size());
  for (const ::std::string &name : names)
    attrs.push_back(::mlir::FlatSymbolRefAttr::get(ctx, name));
  return ::mlir::ArrayAttr::get(ctx, attrs);
}

bool opNamesCompatible(::llvm::StringRef a, ::llvm::StringRef b) {
  if (a == b)
    return true;
  return ::loom::common::sameShareGroup(a, b);
}

::fabric::OpOp emitFabricOp(::mlir::OpBuilder &builder,
                            ::mlir::Location loc,
                            ::mlir::ArrayAttr opList,
                            ::mlir::ArrayAttr hwParams,
                            ::mlir::ValueRange operands,
                            ::mlir::TypeRange resultTypes) {
  ::mlir::OperationState st(loc, ::fabric::OpOp::getOperationName());
  st.addOperands(operands);
  st.addTypes(resultTypes);
  st.addAttribute("op_list", opList);
  if (hwParams)
    st.addAttribute("hw_params", hwParams);
  return ::mlir::cast<::fabric::OpOp>(builder.create(st));
}

::fabric::MuxOp emitMux(::mlir::OpBuilder &builder, ::mlir::Location loc,
                        ::mlir::ValueRange arms, ::mlir::Type type) {
  ::mlir::OperationState st(loc, ::fabric::MuxOp::getOperationName());
  st.addOperands(arms);
  st.addTypes({type});
  return ::mlir::cast<::fabric::MuxOp>(builder.create(st));
}

::fabric::DemuxOp emitDemux(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            ::mlir::Value input, unsigned outputs,
                            ::mlir::Type type) {
  ::mlir::OperationState st(loc, ::fabric::DemuxOp::getOperationName());
  st.addOperands({input});
  st.addTypes(::llvm::SmallVector<::mlir::Type, 4>(outputs, type));
  return ::mlir::cast<::fabric::DemuxOp>(builder.create(st));
}

::std::string wrapperName(::llvm::StringRef groupName) {
  ::std::string out = "fu_";
  for (char c : groupName) {
    bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '_';
    out.push_back(ok ? c : '_');
  }
  return out;
}

struct WrapperShell {
  ::mlir::OwningOpRef<::fabric::ModuleOp> wrapper;
  ::fabric::FuOp fu;
  ::mlir::Block *fuEntry = nullptr;
  ::llvm::SmallVector<::mlir::Type, 4> outerResultTypes;
  ::mlir::OpBuilder bodyBuilder;

  WrapperShell(::mlir::MLIRContext *ctx)
      : bodyBuilder(ctx) {}
};

::std::optional<WrapperShell>
buildWrapperShell(::mlir::MLIRContext *ctx, ::llvm::StringRef groupName,
                  ::llvm::ArrayRef<::mlir::Type> innerInputs,
                  ::llvm::ArrayRef<::mlir::Type> innerOutputs) {
  if (!ctx)
    return std::nullopt;
  ::mlir::Location loc = ::mlir::UnknownLoc::get(ctx);
  unsigned uniformW = 0;
  for (::mlir::Type t : innerInputs)
    if (auto bits = ::llvm::dyn_cast<::fabric::BitsType>(t))
      uniformW = std::max(uniformW, bits.getWidth());
  for (::mlir::Type t : innerOutputs)
    if (auto bits = ::llvm::dyn_cast<::fabric::BitsType>(t))
      uniformW = std::max(uniformW, bits.getWidth());
  ::mlir::Type uniformBits = ::fabric::BitsType::get(ctx, uniformW);
  ::llvm::SmallVector<::mlir::Type, 4> outerInputs(innerInputs.size(),
                                                   uniformBits);
  ::llvm::SmallVector<::mlir::Type, 4> outerOutputs(innerOutputs.size(),
                                                    uniformBits);

  WrapperShell shell(ctx);

  auto moduleType = ::mlir::FunctionType::get(ctx, outerInputs,
                                              ::mlir::TypeRange());
  ::mlir::OperationState moduleState(loc,
                                     ::fabric::ModuleOp::getOperationName());
  moduleState.addAttribute("sym_name",
                           ::mlir::StringAttr::get(ctx, wrapperName(groupName)));
  moduleState.addAttribute("function_type",
                           ::mlir::TypeAttr::get(moduleType));
  ::mlir::Region *moduleRegion = moduleState.addRegion();
  auto *moduleEntry = new ::mlir::Block();
  moduleRegion->push_back(moduleEntry);
  moduleEntry->addArguments(
      outerInputs, ::llvm::SmallVector<::mlir::Location, 4>(outerInputs.size(),
                                                            loc));
  ::mlir::OpBuilder topBuilder(ctx);
  auto module = ::mlir::cast<::fabric::ModuleOp>(topBuilder.create(moduleState));

  ::mlir::OpBuilder moduleBuilder(moduleEntry, moduleEntry->end());
  ::llvm::SmallVector<::mlir::Type, 4> peResults(outerOutputs.begin(),
                                                 outerOutputs.end());
  if (peResults.empty())
    peResults.push_back(uniformBits);
  ::mlir::OperationState peState(loc, ::fabric::PeOp::getOperationName());
  peState.addOperands(::mlir::ValueRange(moduleEntry->getArguments()));
  peState.addTypes(peResults);
  peState.addAttribute(
      "schedule",
      ::fabric::ScheduleAttr::get(ctx, ::fabric::Schedule::Spatial));
  ::mlir::Region *peRegion = peState.addRegion();
  auto *peEntry = new ::mlir::Block();
  peRegion->push_back(peEntry);
  peEntry->addArguments(
      outerInputs, ::llvm::SmallVector<::mlir::Location, 4>(outerInputs.size(),
                                                            loc));
  auto pe = ::mlir::cast<::fabric::PeOp>(moduleBuilder.create(peState));
  (void)pe;

  ::mlir::OpBuilder peBuilder(peEntry, peEntry->end());
  ::mlir::OperationState fuState(loc, ::fabric::FuOp::getOperationName());
  fuState.addOperands(::mlir::ValueRange(peEntry->getArguments()));
  fuState.addTypes(outerOutputs);
  ::mlir::Region *fuRegion = fuState.addRegion();
  auto *fuEntry = new ::mlir::Block();
  fuRegion->push_back(fuEntry);
  fuEntry->addArguments(
      innerInputs, ::llvm::SmallVector<::mlir::Location, 4>(innerInputs.size(),
                                                            loc));
  auto fu = ::mlir::cast<::fabric::FuOp>(peBuilder.create(fuState));

  ::mlir::OperationState moduleYieldState(
      loc, ::fabric::YieldOp::getOperationName());
  moduleBuilder.create(moduleYieldState);

  shell.wrapper = ::mlir::OwningOpRef<::fabric::ModuleOp>(module);
  shell.fu = fu;
  shell.fuEntry = fuEntry;
  shell.outerResultTypes.assign(outerOutputs.begin(), outerOutputs.end());
  shell.bodyBuilder = ::mlir::OpBuilder(fuEntry, fuEntry->end());
  return shell;
}

bool isStateHeadName(::llvm::StringRef name) {
  return name == "dataflow.carry" || name == "dataflow.gate" ||
         name == "dataflow.invariant";
}

bool hasStateOrBackEdge(::dataflow::SubgraphOp sg) {
  if (!sg || !backEdges(sg).empty())
    return true;
  for (::mlir::Operation &op : sg.getBody().front().without_terminator())
    if (isStateHeadName(op.getName().getStringRef()))
      return true;
  return false;
}

::llvm::SmallVector<::mlir::Operation *, 4>
bodyOps(::dataflow::SubgraphOp sg) {
  ::llvm::SmallVector<::mlir::Operation *, 4> ops;
  if (!sg)
    return ops;
  for (::mlir::Operation &op : sg.getBody().front().without_terminator())
    ops.push_back(&op);
  return ops;
}

::std::optional<unsigned> blockArgIndex(::mlir::Value v) {
  if (auto arg = ::llvm::dyn_cast<::mlir::BlockArgument>(v))
    return arg.getArgNumber();
  return std::nullopt;
}

struct DagOperand {
  enum class Kind { BlockArg, SkeletonResult };
  Kind kind = Kind::BlockArg;
  unsigned argIndex = 0;
  unsigned nodeIndex = 0;
  unsigned resultIndex = 0;
};

struct SkeletonNodePlan {
  ::llvm::SmallVector<::mlir::Operation *, 4> peers;
  ::llvm::SmallVector<DagOperand, 4> operands;
};

struct InputArmPlan {
  ::mlir::Operation *tailOp = nullptr;
  DagOperand directYield;
  unsigned yieldResultIndex = 0;
  ::llvm::SmallVector<DagOperand, 4> operands;
};

struct SharedTailPlan {
  ::llvm::SmallVector<SkeletonNodePlan, 4> skeletonNodes;
  ::llvm::SmallVector<InputArmPlan, 4> arms;
};

::std::optional<::llvm::SmallVector<::mlir::Type, 2>>
liftedResultTypes(::mlir::MLIRContext *ctx,
                  ::llvm::ArrayRef<::mlir::Operation *> ops) {
  if (ops.empty() || !ops.front())
    return std::nullopt;
  unsigned n = ops.front()->getNumResults();
  ::llvm::SmallVector<unsigned, 2> widths;
  widths.reserve(n);
  for (unsigned i = 0; i < n; ++i) {
    unsigned bw = bitWidthOf(ops.front()->getResult(i).getType());
    widths.push_back(bw);
  }
  for (::mlir::Operation *op : ops) {
    if (!op || op->getNumResults() != n)
      return std::nullopt;
    for (unsigned i = 0; i < n; ++i)
      if (bitWidthOf(op->getResult(i).getType()) != widths[i])
        return std::nullopt;
  }
  ::llvm::SmallVector<::mlir::Type, 2> out;
  out.reserve(n);
  for (unsigned bw : widths)
    out.push_back(::fabric::BitsType::get(ctx, bw));
  return out;
}

::std::optional<DagOperand>
describeCommonOperand(::llvm::ArrayRef<::mlir::Operation *> peers,
                      unsigned operandIdx,
                      ::llvm::ArrayRef<::std::map<::mlir::Operation *,
                                                  unsigned>> skeletonByInput) {
  if (peers.empty() || peers.size() != skeletonByInput.size())
    return std::nullopt;

  DagOperand out;
  bool initialized = false;
  for (auto [inputIdx, op] : ::llvm::enumerate(peers)) {
    if (!op || operandIdx >= op->getNumOperands())
      return std::nullopt;
    DagOperand cur;
    ::mlir::Value operand = op->getOperand(operandIdx);
    if (auto idx = blockArgIndex(operand)) {
      cur.kind = DagOperand::Kind::BlockArg;
      cur.argIndex = *idx;
    } else if (auto opRes = ::llvm::dyn_cast<::mlir::OpResult>(operand)) {
      auto found = skeletonByInput[inputIdx].find(opRes.getOwner());
      if (found == skeletonByInput[inputIdx].end())
        return std::nullopt;
      cur.kind = DagOperand::Kind::SkeletonResult;
      cur.nodeIndex = found->second;
      cur.resultIndex = opRes.getResultNumber();
    } else {
      return std::nullopt;
    }

    if (!initialized) {
      out = cur;
      initialized = true;
      continue;
    }
    if (out.kind != cur.kind || out.argIndex != cur.argIndex ||
        out.nodeIndex != cur.nodeIndex || out.resultIndex != cur.resultIndex)
      return std::nullopt;
  }
  return out;
}

::std::optional<DagOperand>
describeInputOperand(
    ::mlir::Value operand,
    const ::std::map<::mlir::Operation *, unsigned> &skeletonByInput) {
  DagOperand out;
  if (auto idx = blockArgIndex(operand)) {
    out.kind = DagOperand::Kind::BlockArg;
    out.argIndex = *idx;
    return out;
  }
  if (auto opRes = ::llvm::dyn_cast<::mlir::OpResult>(operand)) {
    auto found = skeletonByInput.find(opRes.getOwner());
    if (found == skeletonByInput.end())
      return std::nullopt;
    out.kind = DagOperand::Kind::SkeletonResult;
    out.nodeIndex = found->second;
    out.resultIndex = opRes.getResultNumber();
    return out;
  }
  return std::nullopt;
}

::std::optional<SharedTailPlan>
detectSharedPrefixTail(::llvm::ArrayRef<::dataflow::SubgraphOp> sgs,
                       std::atomic<bool> &deadlineExpired,
                       Clock::time_point deadline) {
  if (sgs.empty())
    return std::nullopt;
  SharedTailPlan plan;
  plan.arms.reserve(sgs.size());

  unsigned yieldArity = 0;
  ::llvm::SmallVector<::llvm::SmallVector<::mlir::Operation *, 4>, 4>
      opsByInput;
  opsByInput.reserve(sgs.size());
  for (unsigned i = 0, e = static_cast<unsigned>(sgs.size()); i < e; ++i) {
    if (noteDeadline(deadlineExpired, deadline))
      return std::nullopt;
    ::dataflow::SubgraphOp sg = sgs[i];
    if (!sg || hasStateOrBackEdge(sg))
      return std::nullopt;
    ::mlir::Block &body = sg.getBody().front();
    ::mlir::Operation *yield = body.getTerminator();
    if (!yield)
      return std::nullopt;
    if (i == 0)
      yieldArity = yield->getNumOperands();
    if (yield->getNumOperands() != yieldArity || yieldArity != 1)
      return std::nullopt;

    auto ops = bodyOps(sg);
    if (ops.empty())
      return std::nullopt;
    opsByInput.push_back(std::move(ops));
  }

  ::llvm::SmallVector<::std::map<::mlir::Operation *, unsigned>, 4>
      skeletonByInput(sgs.size());
  for (unsigned pos = 0;; ++pos) {
    if (noteDeadline(deadlineExpired, deadline))
      return std::nullopt;
    if (::llvm::any_of(opsByInput, [pos](const auto &ops) {
          return pos >= ops.size();
        }))
      break;

    ::llvm::SmallVector<::mlir::Operation *, 4> peers;
    peers.reserve(sgs.size());
    for (const auto &ops : opsByInput)
      peers.push_back(ops[pos]);

    ::llvm::StringRef firstName = peers.front()->getName().getStringRef();
    if (!::llvm::all_of(peers, [firstName](::mlir::Operation *op) {
          return opNamesCompatible(firstName, op->getName().getStringRef());
        }))
      break;
    if (!liftedResultTypes(sgs.front()->getContext(), peers).has_value())
      break;

    unsigned arity = peers.front()->getNumOperands();
    if (!::llvm::all_of(peers, [arity](::mlir::Operation *op) {
          return op->getNumOperands() == arity;
        }))
      break;

    SkeletonNodePlan node;
    node.peers = std::move(peers);
    node.operands.reserve(arity);
    bool operandsOk = true;
    for (unsigned operandIdx = 0; operandIdx < arity; ++operandIdx) {
      auto operand =
          describeCommonOperand(node.peers, operandIdx, skeletonByInput);
      if (!operand.has_value()) {
        operandsOk = false;
        break;
      }
      node.operands.push_back(*operand);
    }
    if (!operandsOk)
      break;

    unsigned nodeIndex = static_cast<unsigned>(plan.skeletonNodes.size());
    for (auto [inputIdx, op] : ::llvm::enumerate(node.peers))
      skeletonByInput[inputIdx][op] = nodeIndex;
    plan.skeletonNodes.push_back(std::move(node));
  }

  if (plan.skeletonNodes.empty())
    return std::nullopt;

  for (auto indexed : ::llvm::enumerate(sgs)) {
    if (noteDeadline(deadlineExpired, deadline))
      return std::nullopt;
    unsigned inputIdx = static_cast<unsigned>(indexed.index());
    ::dataflow::SubgraphOp sg = indexed.value();
    ::mlir::Operation *yield = sg.getBody().front().getTerminator();
    auto yieldResult =
        ::llvm::dyn_cast<::mlir::OpResult>(yield->getOperand(0));
    if (!yieldResult)
      return std::nullopt;

    InputArmPlan arm;
    arm.yieldResultIndex = yieldResult.getResultNumber();
    auto direct =
        describeInputOperand(yield->getOperand(0), skeletonByInput[inputIdx]);
    if (direct.has_value()) {
      arm.directYield = *direct;
      plan.arms.push_back(std::move(arm));
      continue;
    }

    const auto &ops = opsByInput[inputIdx];
    if (plan.skeletonNodes.size() >= ops.size() ||
        yieldResult.getOwner() != ops[plan.skeletonNodes.size()])
      return std::nullopt;

    arm.tailOp = yieldResult.getOwner();
    arm.operands.reserve(arm.tailOp->getNumOperands());
    for (::mlir::Value operand : arm.tailOp->getOperands()) {
      auto described = describeInputOperand(operand, skeletonByInput[inputIdx]);
      if (!described.has_value())
        return std::nullopt;
      arm.operands.push_back(*described);
    }
    plan.arms.push_back(std::move(arm));
  }

  return plan;
}

using SkeletonOutputs =
    ::llvm::SmallVector<::llvm::SmallVector<::mlir::Value, 2>, 4>;

::std::optional<::mlir::Value>
resolveDagValue(const DagOperand &operand, ::mlir::Block *fuEntry,
                const SkeletonOutputs &skeletonOutputs) {
  if (operand.kind == DagOperand::Kind::BlockArg) {
    if (!fuEntry || operand.argIndex >= fuEntry->getNumArguments())
      return std::nullopt;
    return fuEntry->getArgument(operand.argIndex);
  }
  if (operand.nodeIndex >= skeletonOutputs.size() ||
      operand.resultIndex >= skeletonOutputs[operand.nodeIndex].size())
    return std::nullopt;
  return skeletonOutputs[operand.nodeIndex][operand.resultIndex];
}

using DemuxedSkeletonValues =
    ::std::map<::std::pair<unsigned, unsigned>,
               ::llvm::SmallVector<::mlir::Value, 4>>;

::std::optional<::mlir::Value>
resolveArmValue(const DagOperand &operand, unsigned armIndex,
                ::mlir::Block *fuEntry, const SkeletonOutputs &skeletonOutputs,
                const DemuxedSkeletonValues &demuxed) {
  if (operand.kind == DagOperand::Kind::BlockArg)
    return resolveDagValue(operand, fuEntry, skeletonOutputs);
  auto key = ::std::make_pair(operand.nodeIndex, operand.resultIndex);
  auto found = demuxed.find(key);
  if (found != demuxed.end()) {
    if (armIndex >= found->second.size())
      return std::nullopt;
    return found->second[armIndex];
  }
  return resolveDagValue(operand, fuEntry, skeletonOutputs);
}

void collectSkeletonRefs(const DagOperand &operand,
                         ::std::set<::std::pair<unsigned, unsigned>> &refs) {
  if (operand.kind == DagOperand::Kind::SkeletonResult)
    refs.insert({operand.nodeIndex, operand.resultIndex});
}

::std::optional<SynthResult>
tryRealDagMces(const SynthInputs &inputs, std::atomic<bool> &deadlineExpired,
               Clock::time_point deadline) {
  auto plan = detectSharedPrefixTail(inputs.subgraphs, deadlineExpired,
                                     deadline);
  if (!plan.has_value())
    return std::nullopt;
  ::mlir::MLIRContext *ctx = inputs.context;
  if (!ctx)
    return std::nullopt;
  auto ports = collectWrapperPorts(inputs.subgraphs, ctx);
  if (!ports.has_value())
    return std::nullopt;
  if (ports->outputs.size() != 1)
    return std::nullopt;
  auto shell = buildWrapperShell(ctx, inputs.groupName, ports->inputs,
                                ports->outputs);
  if (!shell.has_value())
    return std::nullopt;

  ::mlir::Location loc = ::mlir::UnknownLoc::get(ctx);
  ::mlir::OpBuilder &builder = shell->bodyBuilder;

  SkeletonOutputs skeletonOutputs;
  skeletonOutputs.reserve(plan->skeletonNodes.size());
  for (const SkeletonNodePlan &node : plan->skeletonNodes) {
    if (noteDeadline(deadlineExpired, deadline))
      return std::nullopt;
    ::std::set<::std::string> names;
    for (::mlir::Operation *op : node.peers)
      names.insert(op->getName().getStringRef().str());
    if (names.empty())
      return std::nullopt;
    auto resultTypes = liftedResultTypes(ctx, node.peers);
    if (!resultTypes.has_value())
      return std::nullopt;

    ::llvm::SmallVector<::mlir::Value, 4> operands;
    operands.reserve(node.operands.size());
    for (const DagOperand &operand : node.operands) {
      auto value = resolveDagValue(operand, shell->fuEntry, skeletonOutputs);
      if (!value.has_value())
        return std::nullopt;
      operands.push_back(*value);
    }
    ::mlir::ArrayAttr hw =
        buildHwParamsUnion(ctx, *names.begin(), node.peers);
    auto emitted = emitFabricOp(builder, loc, sortedOpList(ctx, names), hw,
                                operands, *resultTypes);
    ::llvm::SmallVector<::mlir::Value, 2> outputs;
    outputs.assign(emitted.getOutputs().begin(), emitted.getOutputs().end());
    skeletonOutputs.push_back(std::move(outputs));
  }

  bool anyTail = false;
  for (const InputArmPlan &arm : plan->arms)
    anyTail |= arm.tailOp != nullptr;

  DemuxedSkeletonValues demuxed;
  if (anyTail) {
    ::std::set<::std::pair<unsigned, unsigned>> refs;
    for (const InputArmPlan &arm : plan->arms) {
      if (noteDeadline(deadlineExpired, deadline))
        return std::nullopt;
      if (arm.tailOp) {
        for (const DagOperand &operand : arm.operands)
          collectSkeletonRefs(operand, refs);
      } else {
        collectSkeletonRefs(arm.directYield, refs);
      }
    }
    for (auto key : refs) {
      if (noteDeadline(deadlineExpired, deadline))
        return std::nullopt;
      if (key.first >= skeletonOutputs.size() ||
          key.second >= skeletonOutputs[key.first].size())
        return std::nullopt;
      ::mlir::Value value = skeletonOutputs[key.first][key.second];
      auto bits = ::llvm::dyn_cast<::fabric::BitsType>(value.getType());
      if (!bits)
        return std::nullopt;
      auto demux = emitDemux(builder, loc, value,
                             static_cast<unsigned>(plan->arms.size()), bits);
      ::llvm::SmallVector<::mlir::Value, 4> outputs;
      outputs.assign(demux.getOutputs().begin(), demux.getOutputs().end());
      demuxed[key] = std::move(outputs);
    }
  }

  ::llvm::SmallVector<::mlir::Value, 4> yieldArms;
  yieldArms.reserve(anyTail ? plan->arms.size() : 1);
  for (auto [i, arm] : ::llvm::enumerate(plan->arms)) {
    if (noteDeadline(deadlineExpired, deadline))
      return std::nullopt;
    if (!arm.tailOp) {
      auto value = resolveArmValue(arm.directYield, i, shell->fuEntry,
                                   skeletonOutputs, demuxed);
      if (!value.has_value())
        return std::nullopt;
      yieldArms.push_back(*value);
      continue;
    }

    ::llvm::SmallVector<::mlir::Value, 4> tailOperands;
    tailOperands.reserve(arm.operands.size());
    for (const DagOperand &operand : arm.operands) {
      auto value = resolveArmValue(operand, i, shell->fuEntry,
                                   skeletonOutputs, demuxed);
      if (!value.has_value())
        return std::nullopt;
      tailOperands.push_back(*value);
    }
    ::std::set<::std::string> tailNames{
        arm.tailOp->getName().getStringRef().str()};
    ::mlir::Operation *tailPeer = arm.tailOp;
    auto tailHw = buildHwParamsUnion(
        ctx, arm.tailOp->getName().getStringRef(),
        ::llvm::ArrayRef<::mlir::Operation *>(&tailPeer, 1));
    auto tailTypes = liftedResultTypes(
        ctx, ::llvm::ArrayRef<::mlir::Operation *>(&tailPeer, 1));
    if (!tailTypes.has_value() || arm.yieldResultIndex >= tailTypes->size())
      return std::nullopt;
    auto tail = emitFabricOp(builder, loc, sortedOpList(ctx, tailNames), tailHw,
                             tailOperands, *tailTypes);
    yieldArms.push_back(tail.getOutputs()[arm.yieldResultIndex]);
  }

  if (!anyTail && !yieldArms.empty()) {
    ::mlir::Value first = yieldArms.front();
    bool allSame = ::llvm::all_of(yieldArms, [first](::mlir::Value value) {
      return value == first;
    });
    if (allSame)
      yieldArms.resize(1);
  }

  if (yieldArms.empty())
    return std::nullopt;
  if (yieldArms.size() > 1) {
    auto bits = ::llvm::dyn_cast<::fabric::BitsType>(yieldArms.front().getType());
    if (!bits)
      return std::nullopt;
    for (::mlir::Value arm : yieldArms)
      if (arm.getType() != yieldArms.front().getType())
        return std::nullopt;
  }

  ::mlir::Value yieldValue = yieldArms.front();
  if (yieldArms.size() > 1)
    yieldValue = emitMux(builder, loc, yieldArms, ports->outputs[0]).getOutput();

  ::mlir::OperationState yieldState(loc,
                                    ::fabric::YieldOp::getOperationName());
  yieldState.addOperands({yieldValue});
  if (!shell->outerResultTypes.empty() &&
      yieldValue.getType() != shell->outerResultTypes[0]) {
    ::llvm::SmallVector<::mlir::Attribute, 1> declared{
        ::mlir::TypeAttr::get(shell->outerResultTypes[0])};
    yieldState.addAttribute("declared_types",
                            ::mlir::ArrayAttr::get(ctx, declared));
  }
  builder.create(yieldState);

  if (::mlir::failed(::mlir::verify(shell->wrapper.get())))
    return std::nullopt;

  SynthResult result;
  result.wrapper = std::move(shell->wrapper);
  result.notes.push_back("mcs: emitted pure-DAG shared-prefix candidate");
  return result;
}

//===----------------------------------------------------------------------===//
// Compatibility branch-ordering generation. The branch space is the union of:
//   * one anchor-rooted ordering per input subgraph (input_i first,
//     remaining inputs in stable index order).
//   * `mcs.branch_workers` seeded random permutations (mt19937_64 over
//     `incrementalRandomSeed`).
//===----------------------------------------------------------------------===//

using BranchVec = ::llvm::SmallVector<unsigned, 8>;

::llvm::SmallVector<BranchVec, 8>
buildBranches(std::size_t n, unsigned plannedBranches, uint64_t seed) {
  ::llvm::SmallVector<BranchVec, 8> branches;
  branches.reserve(plannedBranches);
  BranchVec base;
  base.reserve(n);
  for (unsigned i = 0; i < n; ++i)
    base.push_back(i);

  // Anchor branches: input_i first, remaining inputs in stable order.
  for (unsigned anchor = 0; anchor < n && branches.size() < plannedBranches;
       ++anchor) {
    BranchVec perm;
    perm.reserve(n);
    perm.push_back(anchor);
    for (unsigned i = 0; i < n; ++i)
      if (i != anchor)
        perm.push_back(i);
    branches.push_back(std::move(perm));
  }

  // Random branches: seeded permutations using a single PRNG so the
  // branch sequence is stable across runs on the same seed and input
  // size.
  std::mt19937_64 rng(seed);
  while (branches.size() < plannedBranches) {
    BranchVec perm = base;
    std::shuffle(perm.begin(), perm.end(), rng);
    branches.push_back(std::move(perm));
  }
  return branches;
}

//===----------------------------------------------------------------------===//
// Execute one branch in a fresh sub-context. Honours the shared
// deadline through the `deadlineExpired` flag; when set, the branch
// short-circuits with `deadlineSkipped=true` so the caller can
// classify the no-success outcome as `timeout`.
//===----------------------------------------------------------------------===//

BranchHandoff
runOneBranch(const ::loom::SynthConfig &cfg, ::llvm::StringRef groupName,
             ::llvm::ArrayRef<::dataflow::SubgraphOp> base,
             const BranchVec &perm, std::size_t branchIndex,
             const std::atomic<bool> &deadlineExpired) {
  BranchHandoff out;
  out.branchIndex = branchIndex;

  if (deadlineExpired.load(std::memory_order_relaxed)) {
    out.deadlineSkipped = true;
    out.failureReason = SynthFailureReason::Timeout;
    out.notes.push_back("mcs: branch skipped (deadline exceeded)");
    return out;
  }

  ::llvm::SmallVector<::dataflow::SubgraphOp, 8> permuted;
  permuted.reserve(perm.size());
  for (unsigned idx : perm) {
    if (idx >= base.size()) {
      out.failureReason = SynthFailureReason::InvalidInput;
      out.notes.push_back("mcs: branch ordering index out of range");
      return out;
    }
    permuted.push_back(base[idx]);
  }

  ::mlir::MLIRContext subContext;
  loadStandardDialects(subContext);

  // Each branch needs an independent inner cfg copy; SynthConfig is
  // plain data so a value copy is enough.
  ::loom::SynthConfig subCfg = cfg;

  IncrementalSynthesizer inner(subCfg);
  SynthInputs subInputs{groupName,
                        ::llvm::ArrayRef<::dataflow::SubgraphOp>(permuted),
                        subCfg, &subContext};
  SynthResult res = inner.run(subInputs);

  out.failureReason = res.failureReason;
  out.notes = std::move(res.notes);
  out.coverage = std::move(res.coverage);

  if (res.success() && res.wrapper) {
    CostModel cm(subCfg);
    if (auto fu = innerFuOf(res.wrapper.get()))
      out.cost = cm.evaluate(fu);

    // Serialize the winning wrapper to text so the outer caller can
    // reparse it under its own context.
    ::llvm::raw_string_ostream os(out.wrapperIR);
    ::mlir::OpPrintingFlags flags;
    flags.useLocalScope().assumeVerified();
    res.wrapper->print(os, flags);
    os.flush();
    out.succeeded = true;
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Failure classification when no branch succeeded. Priority order:
//   1. If any branch was deadline-skipped (or every branch reported
//      `timeout`), report `timeout` -- the user's deadline killed us.
//   2. Otherwise pick the most common inner failure reason; ties
//      broken by the lowest branch index that produced that reason.
// `resource_exhausted` is reported at the call site when the remaining
// candidate budget cannot admit the compatibility branch set.
//===----------------------------------------------------------------------===//

SynthFailureReason
classifyFailure(::llvm::ArrayRef<BranchHandoff> handoffs) {
  if (handoffs.empty())
    return SynthFailureReason::Timeout;
  bool anyDeadline = false;
  unsigned counts[256] = {0};
  std::size_t firstIdx[256];
  for (auto &slot : firstIdx)
    slot = static_cast<std::size_t>(-1);
  unsigned totalFailures = 0;
  for (const BranchHandoff &h : handoffs) {
    if (h.succeeded)
      continue;
    if (h.deadlineSkipped)
      anyDeadline = true;
    auto code = static_cast<uint8_t>(h.failureReason);
    if (counts[code] == 0)
      firstIdx[code] = h.branchIndex;
    ++counts[code];
    ++totalFailures;
  }
  if (anyDeadline)
    return SynthFailureReason::Timeout;
  if (totalFailures == 0)
    return SynthFailureReason::Timeout;
  unsigned bestCount = 0;
  std::size_t bestIdx = static_cast<std::size_t>(-1);
  uint8_t bestCode = static_cast<uint8_t>(SynthFailureReason::TopologyMismatch);
  for (unsigned code = 0; code < 256; ++code) {
    if (counts[code] == 0)
      continue;
    if (counts[code] > bestCount ||
        (counts[code] == bestCount && firstIdx[code] < bestIdx)) {
      bestCount = counts[code];
      bestIdx = firstIdx[code];
      bestCode = static_cast<uint8_t>(code);
    }
  }
  return static_cast<SynthFailureReason>(bestCode);
}

} // namespace

//===----------------------------------------------------------------------===//
// MCSSynthesizer.
//===----------------------------------------------------------------------===//

MCSSynthesizer::MCSSynthesizer(const ::loom::SynthConfig &c) : cfg(c) {}

SynthResult MCSSynthesizer::run(const SynthInputs &inputs) {
  SynthResult result;

  if (inputs.subgraphs.empty()) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back("mcs: no input subgraphs in synth group");
    return result;
  }
  if (!inputs.context) {
    result.failureReason = SynthFailureReason::InvalidInput;
    result.notes.push_back("mcs: missing scratch MLIRContext");
    return result;
  }

  // Pre-flight: deadline of 0 means the strategy is not allowed any
  // wall-time budget. Report `timeout` without launching any branch
  // so the failure is deterministic.
  if (cfg.mcsTimeoutSec == 0) {
    result.failureReason = SynthFailureReason::Timeout;
    result.notes.push_back("mcs: timeout_sec=0 disables synthesis");
    return result;
  }

  Clock::time_point deadline =
      Clock::now() + std::chrono::seconds(cfg.mcsTimeoutSec);
  std::atomic<bool> deadlineExpired{false};

  ::std::vector<CandidateRecord> candidates;
  std::size_t generatedCandidates = 0;
  bool capExceeded = false;

  auto addLocalCandidate =
      [&](::mlir::OwningOpRef<::fabric::ModuleOp> wrapper,
          ::llvm::ArrayRef<::std::string> notes) {
        if (!wrapper)
          return;
        if (generatedCandidates >= cfg.mcsCandidateCap) {
          capExceeded = true;
          return;
        }
        std::size_t idx = generatedCandidates++;
        auto scored = scoreLocalCandidate(std::move(wrapper), inputs, cfg, idx,
                                          notes, deadlineExpired, deadline);
        if (scored)
          candidates.push_back(std::move(*scored));
      };

  if (!noteDeadline(deadlineExpired, deadline)) {
    AnchorSynthesizer anchor(cfg);
    SynthResult anchorResult = anchor.run(inputs);
    if (anchorResult.success() && anchorResult.wrapper) {
      ::llvm::SmallVector<::std::string, 4> notes;
      notes.push_back("mcs: emitted lock-step MCES candidate");
      for (auto &n : anchorResult.notes)
        notes.push_back(std::move(n));
      addLocalCandidate(std::move(anchorResult.wrapper), notes);
    }
  }

  if (!capExceeded && !noteDeadline(deadlineExpired, deadline) &&
      generatedCandidates < cfg.mcsCandidateCap) {
    if (auto real = tryRealDagMces(inputs, deadlineExpired, deadline))
      addLocalCandidate(std::move(real->wrapper), real->notes);
  }

  if (capExceeded) {
    result.failureReason = SynthFailureReason::ResourceExhausted;
    result.notes.push_back("mcs: candidate_cap reached during MCES search");
    return result;
  }
  if (deadlineExpired.load(std::memory_order_relaxed) && candidates.empty()) {
    result.failureReason = SynthFailureReason::Timeout;
    result.notes.push_back("mcs: deadline exceeded during MCES search");
    return result;
  }

  // Plan compatibility branch count. The branch space is bounded by:
  //   * remaining `mcs.candidate_cap` budget when no local MCES candidate
  //     can be selected yet.
  //   * `mcs.branch_workers` (random restarts -- a tunable widening of
  //     the anchor branch space).
  //   * 16 * num_inputs (an internal upper bound that keeps wall time
  //     finite on small inputs even when the user passes a very large cap).
  std::size_t numInputs = inputs.subgraphs.size();
  std::size_t anchorBranches = numInputs;
  std::size_t randomBranches =
      cfg.mcsBranchWorkers > 0 ? cfg.mcsBranchWorkers : 1u;
  std::size_t internalCap = 16 * numInputs > 0 ? 16 * numInputs : 16;
  std::size_t plannedBranches = anchorBranches + randomBranches;
  if (plannedBranches > internalCap)
    plannedBranches = internalCap;

  std::size_t remainingBudget =
      generatedCandidates < cfg.mcsCandidateCap
          ? cfg.mcsCandidateCap - generatedCandidates
          : 0;
  if (plannedBranches > remainingBudget)
    plannedBranches = remainingBudget;
  bool runCompatibility = plannedBranches > 0 &&
                          !deadlineExpired.load(std::memory_order_relaxed);

  // Build the branch orderings up front from a single seeded PRNG so
  // the sequence is stable across runs on the same seed/input size.
  auto branches = runCompatibility
                      ? buildBranches(numInputs, plannedBranches,
                                      cfg.incrementalRandomSeed)
                      : ::llvm::SmallVector<BranchVec, 8>();

  // Dispatch compatibility branches in parallel. `mcs.branch_workers`
  // is validated as at least one by SynthConfig parsing.
  unsigned workers = cfg.mcsBranchWorkers;
  WorkerPool pool(workers);

  struct WorkerJob {
    std::size_t branchIndex = 0;
    const BranchVec *perm = nullptr;
  };
  ::llvm::SmallVector<WorkerJob, 8> jobs;
  jobs.reserve(branches.size());
  for (std::size_t i = 0; i < branches.size(); ++i)
    jobs.push_back(WorkerJob{i, &branches[i]});

  ::llvm::StringRef groupName = inputs.groupName;
  ::llvm::ArrayRef<::dataflow::SubgraphOp> base = inputs.subgraphs;
  const ::loom::SynthConfig &cfgRef = cfg;

  // The pass already runs each group on a worker thread, so a wall-clock
  // deadline check inside the closure is enough for compatibility
  // branches. The worst-case overshoot is one inner Incremental run.
  ::llvm::SmallVector<BranchHandoff, 8> handoffs;
  if (runCompatibility) {
    handoffs = pool.parallelMap<WorkerJob, BranchHandoff>(
        ::llvm::ArrayRef<WorkerJob>(jobs),
        [&cfgRef, groupName, base, &deadlineExpired,
         deadline](const WorkerJob &job) {
          // Refresh the deadline flag before launching this branch's
          // inner synthesis so late-starting branches that pile up after
          // the wall-clock has already passed are skipped cleanly.
          if (Clock::now() >= deadline)
            deadlineExpired.store(true, std::memory_order_relaxed);
          return runOneBranch(cfgRef, groupName, base, *job.perm,
                              job.branchIndex, deadlineExpired);
        });

    for (const BranchHandoff &h : handoffs) {
      if (!h.succeeded)
        continue;
      CandidateRecord record;
      record.candidateIndex = generatedCandidates + h.branchIndex;
      record.cost = h.cost;
      record.localWrapper = false;
      record.wrapperIR = h.wrapperIR;
      record.notes = h.notes;
      record.coverage = h.coverage;
      candidates.push_back(std::move(record));
    }
    generatedCandidates += branches.size();
  }

  if (candidates.empty()) {
    if (deadlineExpired.load(std::memory_order_relaxed)) {
      result.failureReason = SynthFailureReason::Timeout;
    } else if (generatedCandidates >= cfg.mcsCandidateCap) {
      result.failureReason = SynthFailureReason::ResourceExhausted;
    } else {
      result.failureReason = classifyFailure(handoffs);
    }
    result.notes.push_back("mcs: no candidate branch produced a legal FU");
    for (const BranchHandoff &h : handoffs) {
      if (h.succeeded)
        continue;
      for (const ::std::string &n : h.notes)
        result.notes.push_back(n);
    }
    return result;
  }

  std::stable_sort(candidates.begin(), candidates.end(),
                   [](const CandidateRecord &a, const CandidateRecord &b) {
                     if (a.cost != b.cost)
                       return a.cost < b.cost;
                     return a.candidateIndex < b.candidateIndex;
                   });
  CandidateRecord &best = candidates.front();

  if (best.localWrapper) {
    result.wrapper = std::move(best.wrapper);
  } else {
    // Reparse the winning wrapper into the outer scratch context.
    ::mlir::OwningOpRef<::mlir::ModuleOp> parsed =
        ::mlir::parseSourceString<::mlir::ModuleOp>(best.wrapperIR,
                                                    inputs.context);
    if (!parsed) {
      result.failureReason = SynthFailureReason::VerifierFailed;
      result.notes.push_back(
          "mcs: failed to reparse winning wrapper into outer scratch "
          "MLIRContext");
      return result;
    }
    ::fabric::ModuleOp reHomed;
    for (auto fn : parsed->getOps<::fabric::ModuleOp>()) {
      reHomed = fn;
      break;
    }
    if (!reHomed) {
      result.failureReason = SynthFailureReason::VerifierFailed;
      result.notes.push_back(
          "mcs: reparsed module contained no fabric.module");
      return result;
    }
    reHomed->remove();
    result.wrapper = ::mlir::OwningOpRef<::fabric::ModuleOp>(reHomed);
  }
  result.coverage = best.coverage;
  for (const ::std::string &n : best.notes)
    result.notes.push_back(n);
  ::std::string winnerNote;
  {
    ::llvm::raw_string_ostream os(winnerNote);
    os << "mcs: chose candidate " << best.candidateIndex << " of "
       << generatedCandidates << " (cost=" << best.cost << ")";
  }
  result.notes.push_back(std::move(winnerNote));
  return result;
}

} // namespace loom::fabric::tech

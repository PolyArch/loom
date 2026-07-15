#include "Fabric/Tech/Synthesizer/McesMaterializer.h"

#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/Tech/Synthesizer/Anchor.h"
#include "Fabric/Tech/Synthesizer/CostModel.h"
#include "Fabric/Tech/Synthesizer/CoverageVerifier.h"
#include "Fabric/Tech/Synthesizer/HwParams.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::fabric::tech {

namespace {

using Clock = std::chrono::steady_clock;

bool deadlineReached(Clock::time_point deadline) {
  return Clock::now() >= deadline;
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

::mlir::ArrayAttr sortedOpList(::mlir::MLIRContext *ctx,
                               const ::std::set<::std::string> &names) {
  ::llvm::SmallVector<::mlir::Attribute, 4> attrs;
  attrs.reserve(names.size());
  for (const ::std::string &name : names)
    attrs.push_back(::mlir::FlatSymbolRefAttr::get(ctx, name));
  return ::mlir::ArrayAttr::get(ctx, attrs);
}

::fabric::OpOp emitFabricOp(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            ::mlir::ArrayAttr opList,
                            ::mlir::ArrayAttr hwParams,
                            ::mlir::ValueRange operands,
                            ::mlir::TypeRange resultTypes) {
  ::mlir::OperationState state(loc, ::fabric::OpOp::getOperationName());
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttribute("op_list", opList);
  if (hwParams)
    state.addAttribute("hw_params", hwParams);
  return ::mlir::cast<::fabric::OpOp>(builder.create(state));
}

::fabric::MuxOp emitMux(::mlir::OpBuilder &builder, ::mlir::Location loc,
                        ::mlir::ValueRange arms, ::mlir::Type type) {
  ::mlir::OperationState state(loc, ::fabric::MuxOp::getOperationName());
  state.addOperands(arms);
  state.addTypes({type});
  return ::mlir::cast<::fabric::MuxOp>(builder.create(state));
}

::fabric::DemuxOp emitDemux(::mlir::OpBuilder &builder, ::mlir::Location loc,
                            ::mlir::Value input, unsigned outputs,
                            ::mlir::Type type) {
  ::mlir::OperationState state(loc, ::fabric::DemuxOp::getOperationName());
  state.addOperands({input});
  state.addTypes(::llvm::SmallVector<::mlir::Type, 4>(outputs, type));
  return ::mlir::cast<::fabric::DemuxOp>(builder.create(state));
}

struct WrapperShell {
  ::mlir::OwningOpRef<::fabric::ModuleOp> wrapper;
  ::fabric::FuOp fu;
  ::mlir::Block *fuEntry = nullptr;
  ::llvm::SmallVector<::mlir::Type, 4> outerResultTypes;
  ::mlir::OpBuilder bodyBuilder;

  explicit WrapperShell(::mlir::MLIRContext *ctx) : bodyBuilder(ctx) {}
};

::std::optional<WrapperShell>
buildWrapperShell(::mlir::MLIRContext *ctx, ::llvm::StringRef groupName,
                  ::llvm::ArrayRef<::mlir::Type> innerInputs,
                  ::llvm::ArrayRef<::mlir::Type> innerOutputs) {
  if (!ctx)
    return std::nullopt;

  ::mlir::Location loc = ::mlir::UnknownLoc::get(ctx);
  unsigned uniformWidth = 0;
  for (::mlir::Type type : innerInputs)
    if (auto bits = ::llvm::dyn_cast<::fabric::BitsType>(type))
      uniformWidth = std::max(uniformWidth, bits.getWidth());
  for (::mlir::Type type : innerOutputs)
    if (auto bits = ::llvm::dyn_cast<::fabric::BitsType>(type))
      uniformWidth = std::max(uniformWidth, bits.getWidth());

  ::mlir::Type uniformBits = ::fabric::BitsType::get(ctx, uniformWidth);
  ::llvm::SmallVector<::mlir::Type, 4> outerInputs(innerInputs.size(),
                                                   uniformBits);
  ::llvm::SmallVector<::mlir::Type, 4> outerOutputs(innerOutputs.size(),
                                                    uniformBits);

  WrapperShell shell(ctx);

  auto moduleType =
      ::mlir::FunctionType::get(ctx, outerInputs, ::mlir::TypeRange());
  ::mlir::OperationState moduleState(loc,
                                     ::fabric::ModuleOp::getOperationName());
  moduleState.addAttribute(
      "sym_name", ::mlir::StringAttr::get(ctx, wrapperName(groupName)));
  moduleState.addAttribute("function_type", ::mlir::TypeAttr::get(moduleType));
  ::mlir::Region *moduleRegion = moduleState.addRegion();
  auto *moduleEntry = new ::mlir::Block();
  moduleRegion->push_back(moduleEntry);
  moduleEntry->addArguments(
      outerInputs,
      ::llvm::SmallVector<::mlir::Location, 4>(outerInputs.size(), loc));

  ::mlir::OpBuilder topBuilder(ctx);
  auto module =
      ::mlir::cast<::fabric::ModuleOp>(topBuilder.create(moduleState));

  ::mlir::OpBuilder moduleBuilder(moduleEntry, moduleEntry->end());
  ::llvm::SmallVector<::mlir::Type, 4> peResults(outerOutputs.begin(),
                                                 outerOutputs.end());
  if (peResults.empty())
    peResults.push_back(uniformBits);

  ::mlir::OperationState peState(loc, ::fabric::PeOp::getOperationName());
  peState.addOperands(::mlir::ValueRange(moduleEntry->getArguments()));
  peState.addTypes(peResults);
  peState.addAttribute("schedule", ::fabric::ScheduleAttr::get(
                                       ctx, ::fabric::Schedule::Spatial));
  ::mlir::Region *peRegion = peState.addRegion();
  auto *peEntry = new ::mlir::Block();
  peRegion->push_back(peEntry);
  peEntry->addArguments(outerInputs, ::llvm::SmallVector<::mlir::Location, 4>(
                                         outerInputs.size(), loc));
  moduleBuilder.create(peState);

  ::mlir::OpBuilder peBuilder(peEntry, peEntry->end());
  ::mlir::OperationState fuState(loc, ::fabric::FuOp::getOperationName());
  fuState.addOperands(::mlir::ValueRange(peEntry->getArguments()));
  fuState.addTypes(outerOutputs);
  ::mlir::Region *fuRegion = fuState.addRegion();
  auto *fuEntry = new ::mlir::Block();
  fuRegion->push_back(fuEntry);
  fuEntry->addArguments(innerInputs, ::llvm::SmallVector<::mlir::Location, 4>(
                                         innerInputs.size(), loc));
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
                                     const SynthInputs &inputs) {
  CoverageReport report;
  if (!wrapper)
    return report;
  ::mlir::Location loc = ::mlir::UnknownLoc::get(wrapper.getContext());
  ::mlir::OwningOpRef<::mlir::ModuleOp> host(::mlir::ModuleOp::create(loc));
  ::mlir::OpBuilder hostBuilder(host->getBodyRegion());
  ::mlir::Operation *clonedWrapper = hostBuilder.clone(*wrapper.getOperation());
  CoverageVerifier verifier(inputs.config);
  return verifier.verify(
      innerFuOf(::mlir::cast<::fabric::ModuleOp>(clonedWrapper)),
      inputs.subgraphs);
}

::std::optional<::llvm::SmallVector<::mlir::Type, 2>>
liftedResultTypes(::mlir::MLIRContext *ctx,
                  ::llvm::ArrayRef<::mlir::Operation *> ops) {
  if (ops.empty() || !ops.front())
    return std::nullopt;
  unsigned resultCount = ops.front()->getNumResults();
  ::llvm::SmallVector<unsigned, 2> widths;
  widths.reserve(resultCount);
  for (unsigned i = 0; i < resultCount; ++i)
    widths.push_back(bitWidthOfMcsType(ops.front()->getResult(i).getType()));
  for (::mlir::Operation *op : ops) {
    if (!op || op->getNumResults() != resultCount)
      return std::nullopt;
    for (unsigned i = 0; i < resultCount; ++i) {
      if (bitWidthOfMcsType(op->getResult(i).getType()) != widths[i])
        return std::nullopt;
    }
  }

  ::llvm::SmallVector<::mlir::Type, 2> out;
  out.reserve(resultCount);
  for (unsigned width : widths)
    out.push_back(::fabric::BitsType::get(ctx, width));
  return out;
}

using SharedResultKey = ::std::pair<unsigned, unsigned>;
using PrivateNodeKey = ::std::pair<unsigned, unsigned>;

enum class CanonicalSourceKind : uint8_t { None, BlockArgument, SharedResult };

struct CanonicalSource {
  CanonicalSourceKind kind = CanonicalSourceKind::None;
  unsigned index = 0;
  unsigned resultIndex = 0;

  bool operator==(const CanonicalSource &other) const {
    return kind == other.kind && index == other.index &&
           resultIndex == other.resultIndex;
  }

  bool operator<(const CanonicalSource &other) const {
    if (kind != other.kind)
      return kind < other.kind;
    if (index != other.index)
      return index < other.index;
    return resultIndex < other.resultIndex;
  }
};

struct EmissionState {
  EmissionState(const SynthInputs &inputs, ::llvm::ArrayRef<McsGraph> graphs,
                const McesCandidate &candidate, WrapperShell &shell,
                ::mlir::MLIRContext *ctx, ::mlir::OpBuilder *builder,
                ::mlir::Location loc)
      : inputs(inputs), graphs(graphs), candidate(candidate), shell(shell),
        ctx(ctx), builder(builder), loc(loc) {}

  const SynthInputs &inputs;
  ::llvm::ArrayRef<McsGraph> graphs;
  const McesCandidate &candidate;
  WrapperShell &shell;
  ::mlir::MLIRContext *ctx = nullptr;
  ::mlir::OpBuilder *builder = nullptr;
  ::mlir::Location loc;
  ::std::vector<::std::vector<int>> sharedIdByGraphNode;
  ::std::vector<::std::vector<int>> blockArgToBaseArg;
  ::llvm::SmallVector<::llvm::SmallVector<::mlir::Value, 2>, 4> sharedOutputs;
  ::std::map<CanonicalSource, ::llvm::SmallVector<::mlir::Value, 4>>
      demuxedSources;
  ::std::map<SharedResultKey, ::mlir::Value> sharedPlaceholders;
  ::llvm::SmallVector<::mlir::Operation *, 4> placeholderOps;
  ::std::map<PrivateNodeKey, ::llvm::SmallVector<::mlir::Value, 2>>
      privateOutputs;
  ::std::set<PrivateNodeKey> activePrivateNodes;
};

::std::optional<unsigned> sharedIdForNode(const EmissionState &state,
                                          unsigned graphIndex,
                                          unsigned nodeIndex) {
  if (graphIndex >= state.sharedIdByGraphNode.size() ||
      nodeIndex >= state.sharedIdByGraphNode[graphIndex].size())
    return std::nullopt;
  int sharedId = state.sharedIdByGraphNode[graphIndex][nodeIndex];
  if (sharedId < 0)
    return std::nullopt;
  return static_cast<unsigned>(sharedId);
}

::std::optional<::mlir::Value> valueFromSharedOutput(const EmissionState &state,
                                                     unsigned sharedId,
                                                     unsigned resultIndex) {
  if (sharedId >= state.sharedOutputs.size() ||
      resultIndex >= state.sharedOutputs[sharedId].size())
    return std::nullopt;
  return state.sharedOutputs[sharedId][resultIndex];
}

::std::optional<::mlir::Type> sharedResultType(const EmissionState &state,
                                               unsigned sharedId,
                                               unsigned resultIndex) {
  if (sharedId >= state.candidate.sharedNodes.size() || state.graphs.empty())
    return std::nullopt;
  const McesSharedNode &shared = state.candidate.sharedNodes[sharedId];
  if (shared.nodeIndexByGraph.empty())
    return std::nullopt;
  unsigned nodeIndex = shared.nodeIndexByGraph.front();
  if (nodeIndex >= state.graphs.front().nodes.size())
    return std::nullopt;
  const McsNode &node = state.graphs.front().nodes[nodeIndex];
  if (resultIndex >= node.resultWidths.size())
    return std::nullopt;
  return ::fabric::BitsType::get(state.ctx, node.resultWidths[resultIndex]);
}

unsigned bitWidthOfFabricOrMcsType(::mlir::Type type) {
  if (auto bits = ::llvm::dyn_cast<::fabric::BitsType>(type))
    return bits.getWidth();
  return bitWidthOfMcsType(type);
}

::std::optional<::mlir::Value> sharedOutputOrPlaceholder(EmissionState &state,
                                                         unsigned sharedId,
                                                         unsigned resultIndex) {
  if (auto value = valueFromSharedOutput(state, sharedId, resultIndex))
    return value;

  SharedResultKey key{sharedId, resultIndex};
  auto existing = state.sharedPlaceholders.find(key);
  if (existing != state.sharedPlaceholders.end())
    return existing->second;

  auto type = sharedResultType(state, sharedId, resultIndex);
  if (!type.has_value())
    return std::nullopt;

  ::mlir::OperationState placeholder(
      state.loc, ::mlir::UnrealizedConversionCastOp::getOperationName());
  placeholder.addTypes({*type});
  placeholder.addAttribute("loom.mces.placeholder",
                           ::mlir::UnitAttr::get(state.ctx));
  ::mlir::Operation *op = state.builder->create(placeholder);
  state.placeholderOps.push_back(op);
  ::mlir::Value value = op->getResult(0);
  state.sharedPlaceholders[key] = value;
  return value;
}

bool resolveSharedPlaceholders(EmissionState &state) {
  for (auto &entry : state.sharedPlaceholders) {
    auto real =
        valueFromSharedOutput(state, entry.first.first, entry.first.second);
    if (!real.has_value() || *real == entry.second)
      return false;
    entry.second.replaceAllUsesWith(*real);
  }
  for (::mlir::Operation *op : state.placeholderOps)
    op->erase();
  state.placeholderOps.clear();
  state.sharedPlaceholders.clear();
  return true;
}

::std::optional<unsigned> mappedBlockArgumentIndex(const EmissionState &state,
                                                   unsigned graphIndex,
                                                   unsigned argIndex) {
  if (graphIndex >= state.blockArgToBaseArg.size() ||
      argIndex >= state.blockArgToBaseArg[graphIndex].size())
    return std::nullopt;
  int mapped = state.blockArgToBaseArg[graphIndex][argIndex];
  if (mapped < 0)
    return std::nullopt;
  return static_cast<unsigned>(mapped);
}

CanonicalSource canonicalSource(const EmissionState &state, unsigned graphIndex,
                                McsValueRef source) {
  if (source.kind == McsValueKind::BlockArgument) {
    auto mapped = mappedBlockArgumentIndex(state, graphIndex, source.argIndex);
    if (!mapped.has_value())
      return {};
    return {CanonicalSourceKind::BlockArgument, *mapped, 0};
  }
  auto sharedId = sharedIdForNode(state, graphIndex, source.nodeIndex);
  if (!sharedId.has_value())
    return {};
  return {CanonicalSourceKind::SharedResult, *sharedId, source.resultIndex};
}

bool collectRoutedSourcesForArm(EmissionState &state, unsigned graphIndex,
                                McsValueRef source,
                                ::std::set<CanonicalSource> &refs,
                                ::std::set<PrivateNodeKey> &seenPrivate) {
  CanonicalSource canonical = canonicalSource(state, graphIndex, source);
  if (source.kind == McsValueKind::BlockArgument) {
    if (canonical.kind == CanonicalSourceKind::None)
      return false;
    refs.insert(canonical);
    return true;
  }
  if (canonical.kind == CanonicalSourceKind::SharedResult) {
    refs.insert(canonical);
    return true;
  }

  if (graphIndex >= state.graphs.size() ||
      source.nodeIndex >= state.graphs[graphIndex].nodes.size())
    return false;
  PrivateNodeKey key{graphIndex, source.nodeIndex};
  if (!seenPrivate.insert(key).second)
    return true;
  const McsNode &node = state.graphs[graphIndex].nodes[source.nodeIndex];
  for (const McsOperand &operand : node.operands)
    if (!collectRoutedSourcesForArm(state, graphIndex, operand.source, refs,
                                    seenPrivate))
      return false;
  return true;
}

::std::optional<::mlir::Value> canonicalSourceValue(EmissionState &state,
                                                    CanonicalSource source) {
  if (source.kind == CanonicalSourceKind::BlockArgument) {
    if (!state.shell.fuEntry ||
        source.index >= state.shell.fuEntry->getNumArguments())
      return std::nullopt;
    return state.shell.fuEntry->getArgument(source.index);
  }
  if (source.kind == CanonicalSourceKind::SharedResult)
    return sharedOutputOrPlaceholder(state, source.index, source.resultIndex);
  return std::nullopt;
}

::std::optional<::llvm::SmallVector<::mlir::Value, 4>>
getOrCreateDemux(EmissionState &state, CanonicalSource source) {
  auto found = state.demuxedSources.find(source);
  if (found != state.demuxedSources.end())
    return found->second;

  auto input = canonicalSourceValue(state, source);
  if (!input.has_value())
    return std::nullopt;

  ::llvm::SmallVector<::mlir::Value, 4> outputs;
  if (state.graphs.size() < 2) {
    outputs.push_back(*input);
    state.demuxedSources[source] = outputs;
    return outputs;
  }

  auto bits = ::llvm::dyn_cast<::fabric::BitsType>(input->getType());
  if (!bits)
    return std::nullopt;
  auto demux = emitDemux(*state.builder, state.loc, *input,
                         static_cast<unsigned>(state.graphs.size()), bits);
  outputs.assign(demux.getOutputs().begin(), demux.getOutputs().end());
  state.demuxedSources[source] = outputs;
  return outputs;
}

::std::optional<::llvm::SmallVector<::mlir::Value, 2>>
materializePrivateNode(EmissionState &state, unsigned graphIndex,
                       unsigned nodeIndex);

::std::optional<::mlir::Value>
resolveArmValue(EmissionState &state, unsigned graphIndex, McsValueRef source) {
  CanonicalSource canonical = canonicalSource(state, graphIndex, source);
  if (canonical.kind != CanonicalSourceKind::None) {
    auto demuxed = state.demuxedSources.find(canonical);
    if (demuxed != state.demuxedSources.end()) {
      if (graphIndex >= demuxed->second.size())
        return std::nullopt;
      return demuxed->second[graphIndex];
    }
    return canonicalSourceValue(state, canonical);
  }
  if (source.kind == McsValueKind::BlockArgument)
    return std::nullopt;

  auto outputs = materializePrivateNode(state, graphIndex, source.nodeIndex);
  if (!outputs.has_value() || source.resultIndex >= outputs->size())
    return std::nullopt;
  return (*outputs)[source.resultIndex];
}

::std::optional<::llvm::SmallVector<::mlir::Value, 2>>
materializePrivateNode(EmissionState &state, unsigned graphIndex,
                       unsigned nodeIndex) {
  PrivateNodeKey key{graphIndex, nodeIndex};
  auto cached = state.privateOutputs.find(key);
  if (cached != state.privateOutputs.end())
    return cached->second;
  if (!state.activePrivateNodes.insert(key).second)
    return std::nullopt;
  if (graphIndex >= state.graphs.size() ||
      nodeIndex >= state.graphs[graphIndex].nodes.size())
    return std::nullopt;

  const McsNode &node = state.graphs[graphIndex].nodes[nodeIndex];
  ::llvm::SmallVector<::mlir::Value, 4> operands;
  operands.reserve(node.operands.size());
  for (const McsOperand &operand : node.operands) {
    auto value = resolveArmValue(state, graphIndex, operand.source);
    if (!value.has_value())
      return std::nullopt;
    operands.push_back(*value);
  }

  ::mlir::Operation *peer = node.op;
  auto resultTypes = liftedResultTypes(
      state.ctx, ::llvm::ArrayRef<::mlir::Operation *>(&peer, 1));
  if (!resultTypes.has_value())
    return std::nullopt;

  ::std::set<::std::string> names{node.opName.str()};
  ::mlir::ArrayAttr hw = buildHwParamsUnion(
      state.ctx, node.opName, ::llvm::ArrayRef<::mlir::Operation *>(&peer, 1));
  auto emitted =
      emitFabricOp(*state.builder, state.loc, sortedOpList(state.ctx, names),
                   hw, operands, *resultTypes);
  ::llvm::SmallVector<::mlir::Value, 2> outputs;
  outputs.assign(emitted.getOutputs().begin(), emitted.getOutputs().end());
  state.privateOutputs[key] = outputs;
  state.activePrivateNodes.erase(key);
  return outputs;
}

bool directSourceCompatibleForPermutation(const EmissionState &state,
                                          unsigned graphIndex,
                                          McsValueRef baseSource,
                                          McsValueRef candidateSource) {
  CanonicalSource base = canonicalSource(state, 0, baseSource);
  CanonicalSource candidate =
      canonicalSource(state, graphIndex, candidateSource);
  return base.kind != CanonicalSourceKind::None && base == candidate;
}

::llvm::SmallVector<unsigned, 4>
operandPermutationForSharedNode(const EmissionState &state,
                                const McesSharedNode &shared,
                                unsigned graphIndex) {
  ::llvm::SmallVector<unsigned, 4> identity;
  if (graphIndex >= state.graphs.size() || shared.nodeIndexByGraph.empty() ||
      shared.nodeIndexByGraph.front() >= state.graphs.front().nodes.size() ||
      graphIndex >= shared.nodeIndexByGraph.size() ||
      shared.nodeIndexByGraph[graphIndex] >=
          state.graphs[graphIndex].nodes.size())
    return identity;

  const McsNode &base =
      state.graphs.front().nodes[shared.nodeIndexByGraph.front()];
  const McsNode &node =
      state.graphs[graphIndex].nodes[shared.nodeIndexByGraph[graphIndex]];
  unsigned operandCount = base.operands.size();
  identity.reserve(operandCount);
  for (unsigned i = 0; i < operandCount; ++i)
    identity.push_back(i);
  if (graphIndex == 0 || !base.commutative || !node.commutative ||
      node.operands.size() != operandCount)
    return identity;

  ::llvm::SmallVector<unsigned, 4> perm = identity;
  do {
    bool ok = true;
    for (unsigned i = 0; i < operandCount; ++i) {
      if (base.operands[i].width != node.operands[perm[i]].width ||
          !directSourceCompatibleForPermutation(
              state, graphIndex, base.operands[i].source,
              node.operands[perm[i]].source)) {
        ok = false;
        break;
      }
    }
    if (ok)
      return perm;
  } while (std::next_permutation(perm.begin(), perm.end()));
  return identity;
}

::std::optional<::mlir::Value>
resolveDirectCommonValue(EmissionState &state,
                         ::llvm::ArrayRef<McsValueRef> sources) {
  if (sources.empty() || sources.size() != state.graphs.size())
    return std::nullopt;
  CanonicalSource first = canonicalSource(state, 0, sources.front());
  if (first.kind == CanonicalSourceKind::None)
    return std::nullopt;
  for (unsigned graphIndex = 1; graphIndex < sources.size(); ++graphIndex)
    if (!(canonicalSource(state, graphIndex, sources[graphIndex]) == first))
      return std::nullopt;
  return canonicalSourceValue(state, first);
}

::std::optional<::mlir::Value>
buildAdapterValue(EmissionState &state, ::llvm::ArrayRef<McsValueRef> sources) {
  if (sources.empty() || sources.size() != state.graphs.size())
    return std::nullopt;
  if (auto direct = resolveDirectCommonValue(state, sources))
    return direct;

  ::std::set<CanonicalSource> routedSources;
  for (auto indexed : ::llvm::enumerate(sources)) {
    ::std::set<PrivateNodeKey> seenPrivate;
    if (!collectRoutedSourcesForArm(
            state, static_cast<unsigned>(indexed.index()), indexed.value(),
            routedSources, seenPrivate))
      return std::nullopt;
  }
  for (CanonicalSource source : routedSources)
    if (!getOrCreateDemux(state, source).has_value())
      return std::nullopt;

  ::llvm::SmallVector<::mlir::Value, 4> arms;
  arms.reserve(sources.size());
  for (auto indexed : ::llvm::enumerate(sources)) {
    auto value = resolveArmValue(state, static_cast<unsigned>(indexed.index()),
                                 indexed.value());
    if (!value.has_value())
      return std::nullopt;
    arms.push_back(*value);
  }
  if (arms.empty())
    return std::nullopt;

  ::mlir::Type type = arms.front().getType();
  for (::mlir::Value arm : arms)
    if (arm.getType() != type)
      return std::nullopt;
  if (::llvm::all_of(arms, [first = arms.front()](::mlir::Value value) {
        return value == first;
      }))
    return arms.front();
  if (arms.size() < 2)
    return arms.front();
  return emitMux(*state.builder, state.loc, arms, type).getOutput();
}

void erasePlaceholderScaffolding(::fabric::FuOp fu) {
  ::llvm::SmallVector<::mlir::Operation *, 4> toErase;
  fu.walk([&](::mlir::Operation *op) {
    if (op->hasAttr("loom.mces.placeholder"))
      toErase.push_back(op);
  });
  for (::mlir::Operation *op : toErase)
    op->erase();
}

bool buildSharedMaps(EmissionState &state) {
  state.sharedIdByGraphNode.clear();
  state.sharedIdByGraphNode.reserve(state.graphs.size());
  for (const McsGraph &graph : state.graphs)
    state.sharedIdByGraphNode.push_back(
        ::std::vector<int>(graph.nodes.size(), -1));
  state.sharedOutputs.resize(state.candidate.sharedNodes.size());

  for (auto indexed : ::llvm::enumerate(state.candidate.sharedNodes)) {
    const McesSharedNode &shared = indexed.value();
    if (shared.id != indexed.index())
      return false;
    if (shared.nodeIndexByGraph.size() != state.graphs.size())
      return false;
    for (auto nodeIndexed : ::llvm::enumerate(shared.nodeIndexByGraph)) {
      unsigned graphIndex = static_cast<unsigned>(nodeIndexed.index());
      unsigned nodeIndex = nodeIndexed.value();
      if (graphIndex >= state.graphs.size() ||
          nodeIndex >= state.graphs[graphIndex].nodes.size())
        return false;
      int &slot = state.sharedIdByGraphNode[graphIndex][nodeIndex];
      if (slot >= 0)
        return false;
      slot = static_cast<int>(shared.id);
    }
  }
  return true;
}

bool addBlockArgConstraint(EmissionState &state, unsigned graphIndex,
                           unsigned graphArg, unsigned baseArg) {
  if (graphIndex >= state.graphs.size() ||
      graphIndex >= state.blockArgToBaseArg.size() ||
      graphArg >= state.blockArgToBaseArg[graphIndex].size() ||
      !state.shell.fuEntry || baseArg >= state.shell.fuEntry->getNumArguments())
    return false;

  const McsGraph &graph = state.graphs[graphIndex];
  if (graphArg >= graph.blockArgTypes.size())
    return false;
  if (bitWidthOfMcsType(graph.blockArgTypes[graphArg]) !=
      bitWidthOfFabricOrMcsType(
          state.shell.fuEntry->getArgument(baseArg).getType()))
    return false;

  int &slot = state.blockArgToBaseArg[graphIndex][graphArg];
  if (slot >= 0)
    return slot == static_cast<int>(baseArg);
  slot = static_cast<int>(baseArg);
  return true;
}

bool addSourceBlockArgConstraint(EmissionState &state, unsigned graphIndex,
                                 McsValueRef baseSource,
                                 McsValueRef graphSource) {
  if (baseSource.kind == McsValueKind::BlockArgument &&
      graphSource.kind == McsValueKind::BlockArgument)
    return addBlockArgConstraint(state, graphIndex, graphSource.argIndex,
                                 baseSource.argIndex);
  return true;
}

bool addSharedNodeBlockArgConstraints(EmissionState &state,
                                      const McesSharedNode &shared,
                                      unsigned graphIndex) {
  if (graphIndex == 0)
    return true;
  if (state.graphs.empty() || shared.nodeIndexByGraph.empty() ||
      shared.nodeIndexByGraph.front() >= state.graphs.front().nodes.size() ||
      graphIndex >= state.graphs.size() ||
      graphIndex >= shared.nodeIndexByGraph.size() ||
      shared.nodeIndexByGraph[graphIndex] >=
          state.graphs[graphIndex].nodes.size())
    return false;

  const McsNode &base =
      state.graphs.front().nodes[shared.nodeIndexByGraph.front()];
  const McsNode &node =
      state.graphs[graphIndex].nodes[shared.nodeIndexByGraph[graphIndex]];
  if (base.operands.size() != node.operands.size())
    return false;

  ::llvm::SmallVector<unsigned, 4> identity;
  identity.reserve(base.operands.size());
  for (unsigned i = 0, e = static_cast<unsigned>(base.operands.size()); i < e;
       ++i)
    identity.push_back(i);

  auto scorePermutation =
      [&](::llvm::ArrayRef<unsigned> permutation) -> ::std::optional<unsigned> {
    auto saved = state.blockArgToBaseArg[graphIndex];
    unsigned score = 0;
    for (unsigned operandIndex = 0,
                  e = static_cast<unsigned>(base.operands.size());
         operandIndex < e; ++operandIndex) {
      unsigned graphOperandIndex = operandIndex;
      if (operandIndex < permutation.size())
        graphOperandIndex = permutation[operandIndex];
      if (graphOperandIndex >= node.operands.size()) {
        state.blockArgToBaseArg[graphIndex] = std::move(saved);
        return std::nullopt;
      }
      McsValueRef baseSource = base.operands[operandIndex].source;
      McsValueRef graphSource = node.operands[graphOperandIndex].source;
      if (baseSource.kind == McsValueKind::BlockArgument &&
          graphSource.kind == McsValueKind::BlockArgument &&
          baseSource.argIndex != graphSource.argIndex)
        ++score;
      if (!addSourceBlockArgConstraint(state, graphIndex, baseSource,
                                       graphSource)) {
        state.blockArgToBaseArg[graphIndex] = std::move(saved);
        return std::nullopt;
      }
    }
    state.blockArgToBaseArg[graphIndex] = std::move(saved);
    return score;
  };

  ::llvm::SmallVector<unsigned, 4> permutation = identity;
  if (base.commutative && node.commutative) {
    unsigned bestScore = std::numeric_limits<unsigned>::max();
    ::llvm::SmallVector<unsigned, 4> candidate = identity;
    do {
      auto score = scorePermutation(candidate);
      if (score.has_value() && *score < bestScore) {
        bestScore = *score;
        permutation = candidate;
      }
    } while (std::next_permutation(candidate.begin(), candidate.end()));
    if (bestScore == std::numeric_limits<unsigned>::max())
      return false;
  }

  for (unsigned operandIndex = 0,
                e = static_cast<unsigned>(base.operands.size());
       operandIndex < e; ++operandIndex) {
    unsigned graphOperandIndex = operandIndex;
    if (operandIndex < permutation.size())
      graphOperandIndex = permutation[operandIndex];
    if (graphOperandIndex >= node.operands.size())
      return false;
    if (!addSourceBlockArgConstraint(state, graphIndex,
                                     base.operands[operandIndex].source,
                                     node.operands[graphOperandIndex].source))
      return false;
  }
  return true;
}

bool completeBlockArgMap(EmissionState &state, unsigned graphIndex) {
  if (state.graphs.empty() || graphIndex >= state.graphs.size() ||
      graphIndex >= state.blockArgToBaseArg.size())
    return false;
  const McsGraph &graph = state.graphs[graphIndex];
  if (!state.shell.fuEntry)
    return false;
  unsigned wrapperArgCount = state.shell.fuEntry->getNumArguments();

  ::std::set<unsigned> used;
  for (int mapped : state.blockArgToBaseArg[graphIndex]) {
    if (mapped < 0)
      continue;
    unsigned baseArg = static_cast<unsigned>(mapped);
    if (baseArg >= wrapperArgCount || !used.insert(baseArg).second)
      return false;
  }

  for (unsigned graphArg = 0,
                e = static_cast<unsigned>(graph.blockArgTypes.size());
       graphArg < e; ++graphArg) {
    if (state.blockArgToBaseArg[graphIndex][graphArg] >= 0)
      continue;

    auto canMapTo = [&](unsigned baseArg) {
      return !used.count(baseArg) &&
             bitWidthOfMcsType(graph.blockArgTypes[graphArg]) ==
                 bitWidthOfFabricOrMcsType(
                     state.shell.fuEntry->getArgument(baseArg).getType());
    };

    unsigned chosen = wrapperArgCount;
    if (graphArg < wrapperArgCount && canMapTo(graphArg))
      chosen = graphArg;
    else {
      for (unsigned baseArg = 0, baseCount = wrapperArgCount;
           baseArg < baseCount; ++baseArg) {
        if (canMapTo(baseArg)) {
          chosen = baseArg;
          break;
        }
      }
    }
    if (chosen >= wrapperArgCount)
      return false;
    state.blockArgToBaseArg[graphIndex][graphArg] = static_cast<int>(chosen);
    used.insert(chosen);
  }
  return true;
}

bool buildBlockArgMaps(EmissionState &state) {
  if (state.graphs.empty())
    return false;
  state.blockArgToBaseArg.clear();
  state.blockArgToBaseArg.reserve(state.graphs.size());
  for (const McsGraph &graph : state.graphs)
    state.blockArgToBaseArg.push_back(
        ::std::vector<int>(graph.blockArgTypes.size(), -1));

  for (unsigned argIndex = 0, argCount = static_cast<unsigned>(
                                  state.graphs.front().blockArgTypes.size());
       argIndex < argCount; ++argIndex)
    if (!addBlockArgConstraint(state, 0, argIndex, argIndex))
      return false;

  for (const McesSharedNode &shared : state.candidate.sharedNodes)
    for (unsigned graphIndex = 1,
                  graphCount = static_cast<unsigned>(state.graphs.size());
         graphIndex < graphCount; ++graphIndex)
      if (!addSharedNodeBlockArgConstraints(state, shared, graphIndex))
        return false;

  for (const McsGraph &graph : state.graphs) {
    unsigned graphIndex = graph.inputIndex;
    if (graphIndex >= state.graphs.size())
      return false;
    if (graph.yieldSources.size() != state.graphs.front().yieldSources.size())
      return false;
    for (unsigned yieldIndex = 0,
                  yieldCount = static_cast<unsigned>(graph.yieldSources.size());
         yieldIndex < yieldCount; ++yieldIndex)
      if (!addSourceBlockArgConstraint(
              state, graphIndex, state.graphs.front().yieldSources[yieldIndex],
              graph.yieldSources[yieldIndex]))
        return false;
  }

  for (unsigned graphIndex = 0,
                graphCount = static_cast<unsigned>(state.graphs.size());
       graphIndex < graphCount; ++graphIndex)
    if (!completeBlockArgMap(state, graphIndex))
      return false;
  return true;
}

::std::optional<WrapperPorts>
collectGraphWrapperPorts(::mlir::MLIRContext *ctx,
                         ::llvm::ArrayRef<McsGraph> graphs) {
  if (!ctx || graphs.empty())
    return std::nullopt;
  const McsGraph &base = graphs.front();
  WrapperPorts ports;
  ::llvm::SmallVector<::mlir::Type, 4> inputTypes(base.blockArgTypes.begin(),
                                                  base.blockArgTypes.end());
  for (const McsGraph &graph : graphs) {
    for (auto indexed : ::llvm::enumerate(graph.blockArgTypes)) {
      if (indexed.index() >= inputTypes.size())
        inputTypes.push_back(indexed.value());
    }
  }

  ports.inputs.reserve(inputTypes.size());
  for (::mlir::Type type : inputTypes) {
    unsigned width = bitWidthOfMcsType(type);
    ports.inputs.push_back(::fabric::BitsType::get(ctx, width));
  }

  ports.outputs.reserve(base.yieldSources.size());
  for (McsValueRef source : base.yieldSources) {
    ::std::optional<unsigned> width;
    if (source.kind == McsValueKind::BlockArgument) {
      if (source.argIndex >= base.blockArgTypes.size())
        return std::nullopt;
      width = bitWidthOfMcsType(base.blockArgTypes[source.argIndex]);
    } else {
      if (source.nodeIndex >= base.nodes.size() ||
          source.resultIndex >=
              base.nodes[source.nodeIndex].resultWidths.size())
        return std::nullopt;
      width = base.nodes[source.nodeIndex].resultWidths[source.resultIndex];
    }
    ports.outputs.push_back(::fabric::BitsType::get(ctx, *width));
  }
  return ports;
}

::std::optional<::mlir::OwningOpRef<::fabric::ModuleOp>>
materializeOne(const SynthInputs &inputs, ::llvm::ArrayRef<McsGraph> graphs,
               const McesCandidate &candidate) {
  if (!inputs.context || graphs.empty() || candidate.sharedNodes.empty())
    return std::nullopt;

  auto ports = collectGraphWrapperPorts(inputs.context, graphs);
  if (!ports.has_value())
    return std::nullopt;
  auto shell = buildWrapperShell(inputs.context, inputs.groupName,
                                 ports->inputs, ports->outputs);
  if (!shell.has_value())
    return std::nullopt;

  EmissionState state(inputs, graphs, candidate, *shell, inputs.context,
                      &shell->bodyBuilder,
                      ::mlir::UnknownLoc::get(inputs.context));
  if (!buildSharedMaps(state))
    return std::nullopt;
  if (!buildBlockArgMaps(state))
    return std::nullopt;

  for (const McesSharedNode &shared : candidate.sharedNodes) {
    ::llvm::SmallVector<const McsNode *, 4> nodes;
    ::llvm::SmallVector<::mlir::Operation *, 4> peers;
    nodes.reserve(graphs.size());
    peers.reserve(graphs.size());
    for (auto indexed : ::llvm::enumerate(shared.nodeIndexByGraph)) {
      unsigned graphIndex = static_cast<unsigned>(indexed.index());
      const McsNode &node = graphs[graphIndex].nodes[indexed.value()];
      nodes.push_back(&node);
      peers.push_back(node.op);
    }
    if (nodes.empty())
      return std::nullopt;

    unsigned operandCount = nodes.front()->operands.size();
    if (!::llvm::all_of(nodes, [operandCount](const McsNode *node) {
          return node->operands.size() == operandCount;
        }))
      return std::nullopt;

    ::llvm::SmallVector<::mlir::Value, 4> operands;
    operands.reserve(operandCount);
    ::llvm::SmallVector<::llvm::SmallVector<unsigned, 4>, 4> operandPerms;
    operandPerms.reserve(nodes.size());
    for (unsigned graphIndex = 0, e = static_cast<unsigned>(nodes.size());
         graphIndex < e; ++graphIndex)
      operandPerms.push_back(
          operandPermutationForSharedNode(state, shared, graphIndex));
    for (unsigned operandIndex = 0; operandIndex < operandCount;
         ++operandIndex) {
      ::llvm::SmallVector<McsValueRef, 4> sources;
      sources.reserve(nodes.size());
      for (unsigned graphIndex = 0, e = static_cast<unsigned>(nodes.size());
           graphIndex < e; ++graphIndex) {
        unsigned sourceOperandIndex = operandIndex;
        if (graphIndex < operandPerms.size() &&
            operandIndex < operandPerms[graphIndex].size())
          sourceOperandIndex = operandPerms[graphIndex][operandIndex];
        sources.push_back(
            nodes[graphIndex]->operands[sourceOperandIndex].source);
      }
      auto value = buildAdapterValue(state, sources);
      if (!value.has_value())
        return std::nullopt;
      operands.push_back(*value);
    }

    auto resultTypes = liftedResultTypes(inputs.context, peers);
    if (!resultTypes.has_value())
      return std::nullopt;

    ::std::set<::std::string> names;
    for (const McsNode *node : nodes)
      names.insert(node->opName.str());
    if (names.empty())
      return std::nullopt;
    ::mlir::ArrayAttr hw =
        buildHwParamsUnion(inputs.context, *names.begin(), peers);
    auto emitted = emitFabricOp(*state.builder, state.loc,
                                sortedOpList(inputs.context, names), hw,
                                operands, *resultTypes);
    ::llvm::SmallVector<::mlir::Value, 2> outputs;
    outputs.assign(emitted.getOutputs().begin(), emitted.getOutputs().end());
    state.sharedOutputs[shared.id] = std::move(outputs);
  }

  if (!resolveSharedPlaceholders(state))
    return std::nullopt;

  if (ports->outputs.size() != graphs.front().yieldSources.size())
    return std::nullopt;
  ::llvm::SmallVector<::mlir::Value, 4> yieldValues;
  yieldValues.reserve(ports->outputs.size());
  for (unsigned yieldIndex = 0; yieldIndex < ports->outputs.size();
       ++yieldIndex) {
    ::llvm::SmallVector<McsValueRef, 4> sources;
    sources.reserve(graphs.size());
    for (const McsGraph &graph : graphs) {
      if (yieldIndex >= graph.yieldSources.size())
        return std::nullopt;
      sources.push_back(graph.yieldSources[yieldIndex]);
    }
    auto value = buildAdapterValue(state, sources);
    if (!value.has_value())
      return std::nullopt;
    yieldValues.push_back(*value);
  }

  ::mlir::OperationState yieldState(state.loc,
                                    ::fabric::YieldOp::getOperationName());
  yieldState.addOperands(yieldValues);
  bool needsDeclaredTypes =
      yieldValues.size() == shell->outerResultTypes.size();
  if (needsDeclaredTypes) {
    needsDeclaredTypes = false;
    for (auto pair : ::llvm::zip(yieldValues, shell->outerResultTypes))
      if (std::get<0>(pair).getType() != std::get<1>(pair))
        needsDeclaredTypes = true;
  }
  if (needsDeclaredTypes) {
    ::llvm::SmallVector<::mlir::Attribute, 4> declared;
    declared.reserve(shell->outerResultTypes.size());
    for (::mlir::Type type : shell->outerResultTypes)
      declared.push_back(::mlir::TypeAttr::get(type));
    yieldState.addAttribute("declared_types",
                            ::mlir::ArrayAttr::get(inputs.context, declared));
  }
  state.builder->create(yieldState);

  erasePlaceholderScaffolding(shell->fu);
  if (::mlir::failed(::mlir::verify(shell->wrapper.get())))
    return std::nullopt;

  return std::move(shell->wrapper);
}

} // namespace

::llvm::SmallVector<McesMaterializedCandidate, 4>
McesMaterializer::materializeExactCoverCandidates(
    const SynthInputs &inputs,
    ::llvm::ArrayRef<McesCandidate> candidates) const {
  McsGraphBuildResult graphResult = buildMcsGraphs(inputs.subgraphs);
  if (!graphResult.success())
    return {};
  return materializeExactCoverCandidates(inputs, graphResult.graphs,
                                         candidates);
}

::llvm::SmallVector<McesMaterializedCandidate, 4>
McesMaterializer::materializeExactCoverCandidates(
    const SynthInputs &inputs, ::llvm::ArrayRef<McsGraph> graphs,
    ::llvm::ArrayRef<McesCandidate> candidates) const {
  return materializeExactCoverCandidates(inputs, graphs, candidates,
                                         Clock::time_point::max());
}

::llvm::SmallVector<McesMaterializedCandidate, 4>
McesMaterializer::materializeExactCoverCandidates(
    const SynthInputs &inputs, ::llvm::ArrayRef<McsGraph> graphs,
    ::llvm::ArrayRef<McesCandidate> candidates,
    Clock::time_point deadline) const {
  ::llvm::SmallVector<McesMaterializedCandidate, 4> out;
  if (!inputs.context || graphs.empty())
    return out;

  for (auto indexed : ::llvm::enumerate(candidates)) {
    if (deadlineReached(deadline))
      break;
    auto wrapper = materializeOne(inputs, graphs, indexed.value());
    if (!wrapper.has_value() || !*wrapper)
      continue;
    if (deadlineReached(deadline))
      break;

    CoverageReport coverage = verifyDetachedWrapper(wrapper->get(), inputs);
    if (deadlineReached(deadline))
      break;
    if (!coverage.allCovered())
      continue;

    ::fabric::FuOp fu = innerFuOf(wrapper->get());
    if (!fu)
      continue;

    CostModel costModel(inputs.config);
    McesMaterializedCandidate materialized;
    materialized.candidateIndex = indexed.index();
    materialized.cost = costModel.evaluate(fu);
    materialized.wrapper = std::move(*wrapper);
    materialized.coverage = std::move(coverage);
    materialized.notes.push_back("mces: materialized exact-cover candidate");
    if (!indexed.value().debugLabel.empty())
      materialized.notes.push_back(indexed.value().debugLabel);
    out.push_back(std::move(materialized));
  }

  return out;
}

} // namespace loom::fabric::tech

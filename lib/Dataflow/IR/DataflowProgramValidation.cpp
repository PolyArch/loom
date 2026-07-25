#include "Dataflow/IR/DataflowGraphValidation.h"
#include "Dataflow/IR/DataflowThreadCompletion.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Errc.h"

namespace {

llvm::Error programError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::invalid_argument, message.str());
}

bool containsChannelType(mlir::Type type) {
  return type
      .walk<mlir::WalkOrder::PreOrder>([](mlir::Type nested) {
        return llvm::isa<dataflow::ChannelType>(nested)
                   ? mlir::WalkResult::interrupt()
                   : mlir::WalkResult::advance();
      })
      .wasInterrupted();
}

unsigned getThreadRank(dataflow::ThreadOp thread) {
  if (!thread || thread.isExternal())
    return 0;
  mlir::Block &entry = thread.getBody().front();
  return entry.getNumArguments() - thread.getFunctionType().getNumInputs() - 1;
}

std::optional<int64_t> getConstantIndex(mlir::Value value) {
  mlir::APInt constant;
  if (!mlir::matchPattern(value, mlir::m_ConstantInt(&constant)))
    return std::nullopt;
  return constant.getSExtValue();
}

bool isThreadLaunchBodyOperand(dataflow::ThreadLaunchOp launch,
                               mlir::OpOperand &use) {
  return use.getOperandNumber() < launch.getBodyOperands().size();
}

using dataflow::ChannelEndpointBinding;
using dataflow::ChannelRelation;

llvm::Error verifyRootUses(mlir::Value root) {
  auto argument = llvm::dyn_cast<mlir::BlockArgument>(root);
  if (!argument)
    return programError(
        "channel values must be external block arguments in finalized "
        "programs");
  for (mlir::OpOperand &use : root.getUses()) {
    auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(use.getOwner());
    if (!launch || !isThreadLaunchBodyOperand(launch, use))
      return programError(
          "channel value may only bind dataflow.thread.launch body operands");
  }
  return llvm::Error::success();
}

llvm::Error verifyChannelUseSurface(mlir::ModuleOp module) {
  llvm::Error error = llvm::Error::success();
  module.walk([&](mlir::Operation *op) {
    if (error)
      return mlir::WalkResult::interrupt();
    if (llvm::any_of(op->getResultTypes(), containsChannelType)) {
      error = programError(
          llvm::Twine("finalized program contains channel producer '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }

    for (mlir::OpOperand &operand : op->getOpOperands()) {
      if (!containsChannelType(operand.get().getType()))
        continue;
      if (auto launch = llvm::dyn_cast<dataflow::ThreadLaunchOp>(op)) {
        if (isThreadLaunchBodyOperand(launch, operand))
          continue;
      }
      if (auto launch = llvm::dyn_cast<dataflow::GraphLaunchOp>(op)) {
        unsigned number = operand.getOperandNumber();
        unsigned streamInputBegin =
            launch.getDependencies().size() + launch.getValueInputs().size();
        unsigned streamInputEnd =
            streamInputBegin + launch.getStreamInputs().size();
        unsigned streamOutputBegin =
            streamInputEnd + launch.getMemoryInputs().size();
        unsigned streamOutputEnd =
            streamOutputBegin + launch.getStreamOutputs().size();
        if ((number >= streamInputBegin && number < streamInputEnd) ||
            (number >= streamOutputBegin && number < streamOutputEnd))
          continue;
      }
      if (llvm::isa<dataflow::ChannelSendOp, dataflow::ChannelReceiveOp>(op) &&
          operand.getOperandNumber() == 0)
        continue;
      error = programError(
          llvm::Twine("channel operand is not a permitted binding of '") +
          op->getName().getStringRef() + "'");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return error;
}

llvm::Error collectThreadArgumentBindings(dataflow::ThreadLaunchOp launch,
                                          dataflow::ThreadOp thread,
                                          unsigned argumentIndex,
                                          mlir::Value root,
                                          ChannelRelation &relation) {
  mlir::BlockArgument argument =
      thread.getBody().front().getArgument(argumentIndex);

  for (mlir::OpOperand &use : argument.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (auto graphLaunch = llvm::dyn_cast<dataflow::GraphLaunchOp>(owner)) {
      unsigned number = use.getOperandNumber();
      unsigned streamInputBegin = graphLaunch.getDependencies().size() +
                                  graphLaunch.getValueInputs().size();
      unsigned streamInputEnd =
          streamInputBegin + graphLaunch.getStreamInputs().size();
      unsigned streamOutputBegin =
          streamInputEnd + graphLaunch.getMemoryInputs().size();
      unsigned streamOutputEnd =
          streamOutputBegin + graphLaunch.getStreamOutputs().size();
      if (number >= streamInputBegin && number < streamInputEnd) {
        unsigned bindingIndex = number - streamInputBegin;
        relation.consumers.push_back(ChannelEndpointBinding{
            launch, thread, argumentIndex, graphLaunch, bindingIndex,
            llvm::cast<mlir::AffineMapAttr>(
                graphLaunch.getSourceMaps()[bindingIndex])
                .getValue()});
        continue;
      }
      if (number >= streamOutputBegin && number < streamOutputEnd) {
        relation.producers.push_back(
            ChannelEndpointBinding{launch, thread, argumentIndex, graphLaunch,
                                   number - streamOutputBegin, std::nullopt});
        continue;
      }
    }
    if (llvm::isa<dataflow::ChannelSendOp>(owner) &&
        use.getOperandNumber() == 0) {
      relation.producers.push_back(ChannelEndpointBinding{
          launch, thread, argumentIndex, owner, std::nullopt, std::nullopt});
      continue;
    }
    if (llvm::isa<dataflow::ChannelReceiveOp>(owner) &&
        use.getOperandNumber() == 0) {
      relation.consumers.push_back(ChannelEndpointBinding{
          launch, thread, argumentIndex, owner, std::nullopt, std::nullopt});
      continue;
    }
    return programError(llvm::Twine("thread channel argument #") +
                        llvm::Twine(argumentIndex) + " of @" +
                        thread.getSymName() + " has an unsupported use by '" +
                        owner->getName().getStringRef() + "'");
  }
  return llvm::Error::success();
}

llvm::Error verifyLaunchRank(dataflow::ThreadLaunchOp launch,
                             dataflow::ThreadOp thread) {
  unsigned rank = getThreadRank(thread);
  if (launch.getGridUpperBounds().size() == rank)
    return llvm::Error::success();
  return programError(llvm::Twine("dataflow.thread.launch @") +
                      thread.getSymName() + " supplies " +
                      llvm::Twine(launch.getGridUpperBounds().size()) +
                      " grid bounds for thread rank " + llvm::Twine(rank));
}

bool isIdentityBound(mlir::AffineExpr expression, mlir::ValueRange consumer,
                     mlir::Value producer) {
  auto dim = llvm::dyn_cast<mlir::AffineDimExpr>(expression);
  return dim && dim.getPosition() < consumer.size() &&
         consumer[dim.getPosition()] == producer;
}

llvm::Error verifySourceMapBounds(const ChannelEndpointBinding &consumer,
                                  const ChannelEndpointBinding &producer,
                                  mlir::AffineMap sourceMap) {
  dataflow::ThreadLaunchOp consumerLaunch = consumer.rootLaunch;
  dataflow::ThreadLaunchOp producerLaunch = producer.rootLaunch;
  mlir::ValueRange consumerBounds = consumerLaunch.getGridUpperBounds();
  mlir::ValueRange producerBounds = producerLaunch.getGridUpperBounds();
  llvm::SmallVector<std::optional<int64_t>, 4> lowerBounds(
      consumerBounds.size(), 0);
  llvm::SmallVector<std::optional<int64_t>, 4> upperBounds;
  upperBounds.reserve(consumerBounds.size());
  bool emptyDomain = false;
  for (mlir::Value bound : consumerBounds) {
    std::optional<int64_t> constant = getConstantIndex(bound);
    if (constant && *constant <= 0)
      emptyDomain = true;
    upperBounds.push_back(constant ? std::optional<int64_t>(*constant - 1)
                                   : std::nullopt);
  }
  if (emptyDomain)
    return llvm::Error::success();

  for (auto [index, expression] : llvm::enumerate(sourceMap.getResults())) {
    std::optional<int64_t> producerUpper =
        getConstantIndex(producerBounds[index]);
    std::optional<int64_t> lower = mlir::getBoundForAffineExpr(
        expression, sourceMap.getNumDims(), sourceMap.getNumSymbols(),
        lowerBounds, upperBounds, /*isUpper=*/false);
    std::optional<int64_t> upper = mlir::getBoundForAffineExpr(
        expression, sourceMap.getNumDims(), sourceMap.getNumSymbols(),
        lowerBounds, upperBounds, /*isUpper=*/true);
    if (producerUpper && lower && upper) {
      if (*lower < 0 || *upper >= *producerUpper)
        return programError(llvm::Twine("source_map result #") +
                            llvm::Twine(index) +
                            " is not in bounds for the producer domain");
      continue;
    }
    if (isIdentityBound(expression, consumerBounds, producerBounds[index]))
      continue;
    return programError(llvm::Twine("source_map result #") +
                        llvm::Twine(index) +
                        " cannot be proven in bounds for the producer domain");
  }
  return llvm::Error::success();
}

llvm::Error verifyTopology(const ChannelRelation &relation) {
  if (relation.producers.empty())
    return programError("channel topology has no producer binding");

  const ChannelEndpointBinding &producer = relation.producers.front();
  for (const ChannelEndpointBinding &site :
       llvm::ArrayRef(relation.producers).drop_front())
    if (site.rootLaunch != producer.rootLaunch ||
        site.thread != producer.thread ||
        site.threadArgumentOrdinal != producer.threadArgumentOrdinal)
      return programError("channel topology has multiple producer bindings");
  if (relation.consumers.empty())
    return programError("channel topology has no consumer binding");
  unsigned producerRank = getThreadRank(producer.thread);
  for (const ChannelEndpointBinding &consumer : relation.consumers) {
    unsigned consumerRank = getThreadRank(consumer.thread);
    if (!consumer.sourceMap) {
      if (producerRank != 0 || consumerRank != 0)
        return programError(
            "ranked direct channel receive requires an explicit stream "
            "input binding with source_map");
      continue;
    }
    mlir::AffineMap sourceMap = *consumer.sourceMap;
    if (sourceMap.getNumDims() != consumerRank)
      return programError(llvm::Twine("source_map domain rank ") +
                          llvm::Twine(sourceMap.getNumDims()) +
                          " does not match consumer rank " +
                          llvm::Twine(consumerRank));
    if (sourceMap.getNumResults() != producerRank)
      return programError(llvm::Twine("source_map result rank ") +
                          llvm::Twine(sourceMap.getNumResults()) +
                          " does not match producer rank " +
                          llvm::Twine(producerRank));
    if (llvm::Error error =
            verifySourceMapBounds(consumer, producer, sourceMap))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error verifyChannelTopology(mlir::ModuleOp module) {
  if (llvm::Error error = verifyChannelUseSurface(module))
    return error;

  // Rank agreement is a whole-program launch rule; verify it before the shared
  // per-channel relation pass.
  llvm::Error error = llvm::Error::success();
  mlir::SymbolTableCollection symbols;
  module.walk([&](dataflow::ThreadLaunchOp launch) {
    if (error)
      return mlir::WalkResult::interrupt();
    if (auto thread = symbols.lookupNearestSymbolFrom<dataflow::ThreadOp>(
            launch, launch.getCalleeAttr()))
      if (llvm::Error rankError = verifyLaunchRank(launch, thread)) {
        error = std::move(rankError);
        return mlir::WalkResult::interrupt();
      }
    return mlir::WalkResult::advance();
  });
  if (error)
    return error;

  return dataflow::forEachHostChannelRelation(
      module, [](mlir::Value, const ChannelRelation &relation) {
        return verifyTopology(relation);
      });
}

// The one per-thread ownership index for stored-program graph completion
// events. Op verifiers guarantee that every launch is transitively inside
// exactly one thread definition, so the innermost enclosing thread is a
// total, unambiguous owner. Thread retirement frontiers and graph.wait
// coverage both read this index instead of walking for launches themselves.
struct ThreadOwnershipIndex {
  llvm::SmallVector<dataflow::ThreadOp, 4> threads;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value, 4>>
      launchCompletions;

  llvm::ArrayRef<mlir::Value> completionsOf(dataflow::ThreadOp thread) const {
    auto found = launchCompletions.find(thread);
    if (found == launchCompletions.end())
      return {};
    return found->second;
  }
};

ThreadOwnershipIndex indexThreadOwnership(mlir::ModuleOp module) {
  ThreadOwnershipIndex ownership;
  module.walk([&](dataflow::ThreadOp thread) {
    if (!thread.isExternal())
      ownership.threads.push_back(thread);
  });
  module.walk([&](dataflow::GraphLaunchOp launch) {
    ownership.launchCompletions[launch->getParentOfType<dataflow::ThreadOp>()]
        .push_back(launch.getDone());
  });
  return ownership;
}

llvm::Error
verifyThreadCompletionFrontiers(const ThreadOwnershipIndex &ownership) {
  for (dataflow::ThreadOp thread : ownership.threads) {
    auto yield = llvm::cast<dataflow::ThreadYieldOp>(
        thread.getBody().front().getTerminator());
    mlir::ValueRange frontier = yield.getCompletionFrontier();
    dataflow::ThreadCompletionCoverageAnalysis coverage;

    llvm::DenseSet<mlir::Value> seen;
    for (mlir::Value event : frontier)
      if (!seen.insert(event).second)
        return programError(llvm::Twine("thread @") + thread.getSymName() +
                            " has a duplicate completion frontier event");

    for (unsigned first = 0; first < frontier.size(); ++first) {
      for (unsigned second = first + 1; second < frontier.size(); ++second) {
        if (coverage.covers(frontier[first], frontier[second]) ||
            coverage.covers(frontier[second], frontier[first]))
          return programError(
              llvm::Twine("thread @") + thread.getSymName() +
              " has a causally redundant completion frontier event");
      }
    }

    llvm::ArrayRef<mlir::Value> graphLaunchCompletions =
        ownership.completionsOf(thread);

    for (mlir::Value completion : graphLaunchCompletions)
      if (!llvm::any_of(frontier, [&](mlir::Value terminal) {
            return coverage.covers(terminal, completion);
          }))
        return programError(
            llvm::Twine("thread @") + thread.getSymName() +
            " has graph launch completion not covered by its completion "
            "frontier");

    for (unsigned index = 0; index < frontier.size(); ++index)
      if (!coverage.isFrontierMemberNecessary(frontier, index,
                                              graphLaunchCompletions))
        return programError(
            llvm::Twine("thread @") + thread.getSymName() +
            " has a completion frontier event unnecessary for graph launch "
            "coverage");
  }
  return llvm::Error::success();
}

llvm::Error verifyGraphWaitFrontiers(mlir::ModuleOp module,
                                     const ThreadOwnershipIndex &ownership) {
  llvm::SmallVector<dataflow::GraphWaitOp, 2> waits;
  module.walk([&](dataflow::GraphWaitOp wait) { waits.push_back(wait); });

  dataflow::ThreadCompletionCoverageAnalysis coverage;
  for (dataflow::GraphWaitOp wait : waits) {
    llvm::ArrayRef<mlir::Value> completions =
        ownership.completionsOf(wait->getParentOfType<dataflow::ThreadOp>());
    mlir::ValueRange frontier = wait.getCompletionFrontier();
    for (unsigned index = 0; index < frontier.size(); ++index) {
      mlir::Value event = frontier[index];
      if (!llvm::any_of(completions, [&](mlir::Value completion) {
            return coverage.covers(event, completion);
          }))
        return programError(llvm::Twine("dataflow.graph.wait operand #") +
                            llvm::Twine(index) +
                            " does not cover any graph launch completion "
                            "event");
    }
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<dataflow::ChannelRelation>
dataflow::computeChannelRelation(mlir::Value hostChannel) {
  if (llvm::Error error = verifyRootUses(hostChannel))
    return std::move(error);
  ChannelRelation relation;
  mlir::SymbolTableCollection symbols;
  for (mlir::OpOperand &use : hostChannel.getUses()) {
    auto launch = llvm::cast<dataflow::ThreadLaunchOp>(use.getOwner());
    auto thread = symbols.lookupNearestSymbolFrom<dataflow::ThreadOp>(
        launch, launch.getCalleeAttr());
    if (!thread)
      continue;
    if (llvm::Error error = collectThreadArgumentBindings(
            launch, thread, use.getOperandNumber(), hostChannel, relation))
      return std::move(error);
  }
  return relation;
}

llvm::Error dataflow::forEachHostChannelRelation(
    mlir::ModuleOp module,
    llvm::function_ref<llvm::Error(mlir::Value, const ChannelRelation &)>
        callback) {
  llvm::SmallVector<mlir::Value> hostChannels;
  llvm::DenseSet<mlir::Value> seen;
  module.walk([&](mlir::Operation *op) {
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region) {
        mlir::Operation *owner = block.getParentOp();
        if (llvm::isa<dataflow::ThreadOp>(owner) ||
            owner->getParentOfType<dataflow::ThreadOp>())
          continue;
        for (mlir::BlockArgument argument : block.getArguments())
          if (llvm::isa<dataflow::ChannelType>(argument.getType()) &&
              seen.insert(argument).second)
            hostChannels.push_back(argument);
      }
  });
  for (mlir::Value hostChannel : hostChannels) {
    llvm::Expected<ChannelRelation> relation =
        dataflow::computeChannelRelation(hostChannel);
    if (!relation)
      return relation.takeError();
    if (llvm::Error error = callback(hostChannel, *relation))
      return error;
  }
  return llvm::Error::success();
}

llvm::Error dataflow::validateFinalizedProgram(mlir::ModuleOp module) {
  if (!module)
    return programError("finalized program must be a module");
  bool hasSpatialCandidate = false;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "loom.spatial_region")
      return mlir::WalkResult::advance();
    hasSpatialCandidate = true;
    return mlir::WalkResult::interrupt();
  });
  if (hasSpatialCandidate)
    return programError(
        "finalized program contains temporary loom.spatial_region");
  ThreadOwnershipIndex ownership = indexThreadOwnership(module);
  if (llvm::Error error = verifyThreadCompletionFrontiers(ownership))
    return error;
  if (llvm::Error error = verifyGraphWaitFrontiers(module, ownership))
    return error;
  if (llvm::Error error = verifyChannelTopology(module))
    return error;

  llvm::Error error = llvm::Error::success();
  module.walk([&](GraphOp graph) {
    if (error)
      return mlir::WalkResult::interrupt();
    error = validateFinalizedGraph(graph);
    return error ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
  });
  return error;
}

#include "Dataflow/IR/DataflowGraphValidation.h"

#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
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

struct ProducerBinding {
  dataflow::ThreadLaunchOp launch;
  dataflow::ThreadOp thread;
};

struct ConsumerBinding {
  dataflow::ThreadLaunchOp launch;
  dataflow::ThreadOp thread;
  std::optional<mlir::AffineMap> sourceMap;
};

struct ChannelTopology {
  llvm::SmallVector<ProducerBinding, 1> producers;
  llvm::SmallVector<ConsumerBinding, 2> consumers;
};

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
        unsigned streamInputBegin = launch.getDependencies().size() +
                                    launch.getValueInputs().size();
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
      if (llvm::isa<dataflow::ChannelSendOp,
                    dataflow::ChannelReceiveOp>(op) &&
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

llvm::Error collectThreadArgumentBindings(
    dataflow::ThreadLaunchOp launch, dataflow::ThreadOp thread,
    unsigned argumentIndex, mlir::Value root, ChannelTopology &topology) {
  mlir::BlockArgument argument =
      thread.getBody().front().getArgument(argumentIndex);
  bool hasDirectProducer = false;
  bool hasDirectConsumer = false;

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
        topology.consumers.push_back(ConsumerBinding{
            launch, thread,
            llvm::cast<mlir::AffineMapAttr>(
                graphLaunch.getSourceMaps()[bindingIndex])
                .getValue()});
        continue;
      }
      if (number >= streamOutputBegin && number < streamOutputEnd) {
        topology.producers.push_back({launch, thread});
        continue;
      }
    }
    if (llvm::isa<dataflow::ChannelSendOp>(owner) &&
        use.getOperandNumber() == 0) {
      hasDirectProducer = true;
      continue;
    }
    if (llvm::isa<dataflow::ChannelReceiveOp>(owner) &&
        use.getOperandNumber() == 0) {
      hasDirectConsumer = true;
      continue;
    }
    return programError(
        llvm::Twine("thread channel argument #") +
        llvm::Twine(argumentIndex) + " of @" + thread.getSymName() +
        " has an unsupported use by '" + owner->getName().getStringRef() +
        "'");
  }

  if (hasDirectProducer)
    topology.producers.push_back({launch, thread});
  if (hasDirectConsumer)
    topology.consumers.push_back({launch, thread, std::nullopt});
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

llvm::Error verifySourceMapBounds(ConsumerBinding &consumer,
                                  ProducerBinding &producer,
                                  mlir::AffineMap sourceMap) {
  mlir::ValueRange consumerBounds = consumer.launch.getGridUpperBounds();
  mlir::ValueRange producerBounds = producer.launch.getGridUpperBounds();
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

llvm::Error verifyTopology(ChannelTopology &topology) {
  if (topology.producers.empty())
    return programError("channel topology has no producer binding");
  if (topology.producers.size() != 1)
    return programError("channel topology has multiple producer bindings");
  if (topology.consumers.empty())
    return programError("channel topology has no consumer binding");

  ProducerBinding &producer = topology.producers.front();
  unsigned producerRank = getThreadRank(producer.thread);
  for (ConsumerBinding &consumer : topology.consumers) {
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

  mlir::SymbolTableCollection symbols;
  llvm::DenseMap<mlir::Value, ChannelTopology> topologies;
  llvm::SmallVector<mlir::Value> topologyOrder;
  llvm::Error error = llvm::Error::success();
  module.walk([&](mlir::Operation *op) {
    for (mlir::Region &region : op->getRegions()) {
      for (mlir::Block &block : region) {
        mlir::Operation *owner = block.getParentOp();
        if (llvm::isa<dataflow::ThreadOp>(owner) ||
            owner->getParentOfType<dataflow::ThreadOp>())
          continue;
        for (mlir::BlockArgument argument : block.getArguments()) {
          if (!llvm::isa<dataflow::ChannelType>(argument.getType()))
            continue;
          if (llvm::Error rootError = verifyRootUses(argument)) {
            error = std::move(rootError);
            return mlir::WalkResult::interrupt();
          }
          if (topologies.try_emplace(argument).second)
            topologyOrder.push_back(argument);
        }
      }
    }
    return mlir::WalkResult::advance();
  });
  if (error)
    return error;

  module.walk([&](dataflow::ThreadLaunchOp launch) {
    if (error)
      return mlir::WalkResult::interrupt();
    auto thread = symbols.lookupNearestSymbolFrom<dataflow::ThreadOp>(
        launch, launch.getCalleeAttr());
    if (!thread)
      return mlir::WalkResult::advance();
    if (llvm::Error rankError = verifyLaunchRank(launch, thread)) {
      error = std::move(rankError);
      return mlir::WalkResult::interrupt();
    }
    for (auto [index, operand] : llvm::enumerate(launch.getBodyOperands())) {
      if (!llvm::isa<dataflow::ChannelType>(operand.getType()))
        continue;
      auto topology = topologies.find(operand);
      if (topology == topologies.end()) {
        error = programError(
            "channel values must be external block arguments in finalized "
            "programs");
        return mlir::WalkResult::interrupt();
      }
      if (llvm::Error bindingError = collectThreadArgumentBindings(
              launch, thread, index, operand, topology->second)) {
        error = std::move(bindingError);
        return mlir::WalkResult::interrupt();
      }
    }
    return mlir::WalkResult::advance();
  });
  if (error)
    return error;
  for (mlir::Value root : topologyOrder)
    if (llvm::Error topologyError =
            verifyTopology(topologies.find(root)->second))
      return topologyError;
  return llvm::Error::success();
}

} // namespace

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

#include "Arbitration.h"
#include "Components.h"

#include "Common/InvocationDiagnosticLog.h"
#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Fabric/Identity/FabricTemporalPeConfiguration.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::hardware::rtl::hierarchy {
namespace {

struct TemporalPeMaterializationMetrics final {
  std::uint64_t beginOperations = 0;
  std::uint64_t decodeAndPoolOperations = 0;
  std::uint64_t ingressOperations = 0;
  std::uint64_t dispatchOperations = 0;
  std::uint64_t childOperations = 0;
  std::uint64_t resultProjectionOperations = 0;
  std::uint64_t resultArbitrationOperations = 0;
};

std::uint64_t blockOperationCount(mlir::OpBuilder &builder) {
  return static_cast<std::uint64_t>(
      std::distance(builder.getBlock()->begin(), builder.getBlock()->end()));
}

mlir::Value zero(mlir::OpBuilder &builder, mlir::Location location,
                 unsigned width) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, 0));
}

mlir::Value constant(mlir::OpBuilder &builder, mlir::Location location,
                     unsigned width, std::uint64_t value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, value));
}

mlir::Value equals(mlir::OpBuilder &builder, mlir::Location location,
                   mlir::Value value, std::uint64_t expected) {
  const unsigned width =
      mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
  return circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::eq, value,
      constant(builder, location, width, expected), true);
}

mlir::Value extract(mlir::OpBuilder &builder, mlir::Location location,
                    mlir::Value value, std::uint64_t offset,
                    std::uint64_t width) {
  if (width == 0)
    return zero(builder, location, 1);
  return circt::comb::ExtractOp::create(builder, location, value, offset,
                                        width);
}

mlir::Value selectValue(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::Value selector,
                        llvm::ArrayRef<mlir::Value> values) {
  assert(!values.empty() && "selection domain must not be empty");
  mlir::Value result = values.front();
  for (std::uint64_t ordinal = 1; ordinal != values.size(); ++ordinal)
    result = circt::comb::MuxOp::create(
        builder, location, equals(builder, location, selector, ordinal),
        values[ordinal], result, true);
  return result;
}

mlir::Value indexedValue(mlir::OpBuilder &builder, mlir::Location location,
                         mlir::Value index,
                         llvm::ArrayRef<mlir::Value> lowToHigh) {
  assert(!lowToHigh.empty() && "indexed domain must not be empty");
  if (lowToHigh.size() == 1)
    return lowToHigh.front();
  llvm::SmallVector<mlir::Value> highToLow(llvm::reverse(lowToHigh));
  mlir::Value array =
      circt::hw::ArrayCreateOp::create(builder, location, highToLow);
  return circt::hw::ArrayGetOp::create(builder, location, array, index);
}

mlir::Value splatBit(mlir::OpBuilder &builder, mlir::Location location,
                     mlir::Value value, std::size_t count) {
  assert(count != 0 && "splat domain must not be empty");
  if (count == 1)
    return value;
  return circt::comb::MuxOp::create(
      builder, location, value,
      circt::hw::ConstantOp::create(builder, location,
                                    llvm::APInt::getAllOnes(count)),
      circt::hw::ConstantOp::create(builder, location, llvm::APInt(count, 0)),
      true);
}

std::string queuePort(llvm::StringRef prefix, std::uint32_t queue,
                      llvm::StringRef suffix) {
  return prefix.str() + "_" + std::to_string(queue) + suffix.str();
}

struct TokenPoolModule final {
  circt::hw::HWModuleOp module;
  std::uint32_t queueCount = 0;
  unsigned payloadWidth = 0;
};

std::string tokenPoolModuleName(std::uint32_t queueCount, std::uint32_t depth,
                                unsigned payloadWidth, bool singlePort,
                                bool fullReplacement, bool exposeNearFull,
                                bool asynchronousReset) {
  return "loom_token_pool_q" + std::to_string(queueCount) + "_d" +
         std::to_string(depth) + "_w" + std::to_string(payloadWidth) +
         "_single" + std::to_string(singlePort) + "_replace" +
         std::to_string(fullReplacement) + "_near" +
         std::to_string(exposeNearFull) + "_async" +
         std::to_string(asynchronousReset);
}

llvm::Error verifyTokenPoolModuleContract(
    circt::hw::HWModuleOp module,
    llvm::ArrayRef<circt::hw::PortInfo> expectedInputs,
    llvm::ArrayRef<circt::hw::PortInfo> expectedOutputs) {
  if (!module.getParametersAttr().empty())
    return invalid("canonical token-pool module has parameters");
  const circt::hw::ModulePortInfo expected(expectedInputs, expectedOutputs);
  const auto actual = module.getPortList();
  if (actual.size() != expected.size())
    return invalid("canonical token-pool module changed its port count");
  for (const auto &[lhs, rhs] : llvm::zip_equal(actual, expected))
    if (lhs.getName() != rhs.getName() || lhs.type != rhs.type ||
        lhs.dir != rhs.dir)
      return invalid("canonical token-pool module changed its port contract");
  return llvm::Error::success();
}

llvm::Expected<TokenPoolModule>
buildTokenPoolModule(mlir::OpBuilder &builder, mlir::Location location,
                     std::uint32_t queueCount, std::uint32_t depth,
                     unsigned payloadWidth, bool singlePort,
                     bool fullReplacement, bool exposeNearFull,
                     const ClockResetPlan &clockReset) {
  if (queueCount == 0 || depth == 0)
    return invalid("token pool requires nonempty queue and entry domains");
  if (singlePort && fullReplacement)
    return invalid("single-port token pool cannot replace a full entry");

  llvm::SmallVector<circt::hw::PortInfo, 32> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 32> outputs;
  inputs.push_back({{builder.getStringAttr("clock"),
                     circt::seq::ClockType::get(builder.getContext()),
                     circt::hw::ModulePort::Direction::Input}});
  inputs.push_back({{builder.getStringAttr("reset"), builder.getI1Type(),
                     circt::hw::ModulePort::Direction::Input}});
  for (std::uint32_t queue = 0; queue != queueCount; ++queue) {
    if (payloadWidth != 0)
      inputs.push_back(
          {{builder.getStringAttr(queuePort("enqueue", queue, "_data")),
            builder.getIntegerType(payloadWidth),
            circt::hw::ModulePort::Direction::Input}});
    inputs.push_back(
        {{builder.getStringAttr(queuePort("enqueue", queue, "_commit")),
          builder.getI1Type(), circt::hw::ModulePort::Direction::Input}});
    outputs.push_back(
        {{builder.getStringAttr(queuePort("enqueue", queue, "_ready")),
          builder.getI1Type(), circt::hw::ModulePort::Direction::Output}});
    if (payloadWidth != 0)
      outputs.push_back(
          {{builder.getStringAttr(queuePort("dequeue", queue, "_data")),
            builder.getIntegerType(payloadWidth),
            circt::hw::ModulePort::Direction::Output}});
    outputs.push_back(
        {{builder.getStringAttr(queuePort("dequeue", queue, "_valid")),
          builder.getI1Type(), circt::hw::ModulePort::Direction::Output}});
    inputs.push_back(
        {{builder.getStringAttr(queuePort("dequeue", queue, "_ready")),
          builder.getI1Type(), circt::hw::ModulePort::Direction::Input}});
  }
  if (exposeNearFull)
    outputs.push_back({{builder.getStringAttr("near_full"), builder.getI1Type(),
                        circt::hw::ModulePort::Direction::Output}});

  mlir::Block *insertionBlock = builder.getInsertionBlock();
  auto container =
      insertionBlock
          ? llvm::dyn_cast<mlir::ModuleOp>(insertionBlock->getParentOp())
          : mlir::ModuleOp{};
  if (!container)
    return invalid("token-pool definition has no builtin.module owner");
  const std::string name = tokenPoolModuleName(
      queueCount, depth, payloadWidth, singlePort, fullReplacement,
      exposeNearFull, clockReset.asynchronousReset);
  if (mlir::Operation *existing = container.lookupSymbol(name)) {
    auto module = llvm::dyn_cast<circt::hw::HWModuleOp>(existing);
    if (!module)
      return invalid("canonical token-pool symbol names another definition");
    if (llvm::Error error =
            verifyTokenPoolModuleContract(module, inputs, outputs))
      return std::move(error);
    return TokenPoolModule{module, queueCount, payloadWidth};
  }

  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr(name),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        const unsigned occupancyWidth = indexWidth(depth + 1);
        const unsigned queueWidth = indexWidth(queueCount);
        circt::Backedge occupancyNext =
            backedges.get(bodyBuilder.getIntegerType(occupancyWidth));
        mlir::Value occupancy = createRegister(
            bodyBuilder, location, occupancyNext, accessor.getInput("clock"),
            accessor.getInput("reset"), llvm::APInt(occupancyWidth, 0),
            "occupancy_reg", clockReset.asynchronousReset);

        struct Bank final {
          unsigned width = 0;
          std::vector<circt::Backedge> next;
          std::vector<mlir::Value> current;
        };
        const auto makeBank = [&](unsigned width,
                                  llvm::StringRef prefix) -> Bank {
          Bank bank;
          bank.width = width;
          if (width == 0)
            return bank;
          bank.next.resize(depth);
          bank.current.resize(depth);
          for (std::uint32_t slot = 0; slot != depth; ++slot) {
            bank.next[slot] = backedges.get(bodyBuilder.getIntegerType(width));
            bank.current[slot] = createRegister(
                bodyBuilder, location, bank.next[slot],
                accessor.getInput("clock"), accessor.getInput("reset"),
                llvm::APInt(width, 0),
                prefix.str() + "_" + std::to_string(slot) + "_reg",
                clockReset.asynchronousReset);
          }
          return bank;
        };
        Bank payload = makeBank(payloadWidth, "payload");
        Bank queueIds =
            queueCount == 1 ? Bank{} : makeBank(queueWidth, "queue");

        std::vector<std::vector<mlir::Value>> headSlots(
            queueCount, std::vector<mlir::Value>(
                            depth, bitConstant(bodyBuilder, location, false)));
        std::vector<mlir::Value> headValid(
            queueCount, bitConstant(bodyBuilder, location, false));
        if (queueCount == 1) {
          mlir::Value occupied = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::ult,
              constant(bodyBuilder, location, occupancyWidth, 0), occupancy,
              true);
          headSlots[0][0] = occupied;
          headValid[0] = occupied;
          if (payloadWidth != 0)
            accessor.setOutput(queuePort("dequeue", 0, "_data"),
                               payload.current.front());
          accessor.setOutput(queuePort("dequeue", 0, "_valid"), occupied);
        } else {
          for (std::uint32_t queue = 0; queue != queueCount; ++queue) {
            mlir::Value earlier = bitConstant(bodyBuilder, location, false);
            mlir::Value headData =
                payloadWidth == 0 ? mlir::Value{}
                                  : zero(bodyBuilder, location, payloadWidth);
            for (std::uint32_t slot = 0; slot != depth; ++slot) {
              mlir::Value occupied = circt::comb::ICmpOp::create(
                  bodyBuilder, location, circt::comb::ICmpPredicate::ult,
                  constant(bodyBuilder, location, occupancyWidth, slot),
                  occupancy, true);
              mlir::Value matches =
                  andValues(bodyBuilder, location,
                            {occupied, equals(bodyBuilder, location,
                                              queueIds.current[slot], queue)});
              headSlots[queue][slot] =
                  andValues(bodyBuilder, location,
                            {matches, circt::comb::createOrFoldNot(
                                          bodyBuilder, location, earlier)});
              earlier = circt::comb::OrOp::create(bodyBuilder, location,
                                                  earlier, matches);
              if (payloadWidth != 0)
                headData = circt::comb::MuxOp::create(
                    bodyBuilder, location, headSlots[queue][slot],
                    payload.current[slot], headData, true);
            }
            headValid[queue] = earlier;
            if (payloadWidth != 0)
              accessor.setOutput(queuePort("dequeue", queue, "_data"),
                                 headData);
            accessor.setOutput(queuePort("dequeue", queue, "_valid"), earlier);
          }
        }

        std::vector<mlir::Value> dequeueRequests;
        dequeueRequests.reserve(queueCount);
        for (std::uint32_t queue = 0; queue != queueCount; ++queue)
          dequeueRequests.push_back(andValues(
              bodyBuilder, location,
              {headValid[queue],
               accessor.getInput(queuePort("dequeue", queue, "_ready"))}));
        std::vector<mlir::Value> dequeueSelected(
            queueCount, bitConstant(bodyBuilder, location, false));
        mlir::Value dequeueReserved = bitConstant(bodyBuilder, location, false);
        for (std::uint32_t queue = 0; queue != queueCount; ++queue) {
          dequeueSelected[queue] =
              andValues(bodyBuilder, location,
                        {dequeueRequests[queue],
                         circt::comb::createOrFoldNot(bodyBuilder, location,
                                                      dequeueReserved)});
          dequeueReserved = circt::comb::OrOp::create(
              bodyBuilder, location, dequeueReserved, dequeueRequests[queue]);
        }
        mlir::Value dequeue = orValues(bodyBuilder, location, dequeueSelected);

        mlir::Value full = circt::comb::ICmpOp::create(
            bodyBuilder, location, circt::comb::ICmpPredicate::eq, occupancy,
            constant(bodyBuilder, location, occupancyWidth, depth), true);
        if (exposeNearFull) {
          mlir::Value nearFull = bitConstant(bodyBuilder, location, true);
          if (depth != 1) {
            mlir::Value belowNearFull = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::ult,
                occupancy,
                constant(bodyBuilder, location, occupancyWidth, depth - 1),
                true);
            nearFull = circt::comb::createOrFoldNot(bodyBuilder, location,
                                                    belowNearFull);
          }
          accessor.setOutput("near_full", nearFull);
        }
        mlir::Value canEnqueue =
            circt::comb::createOrFoldNot(bodyBuilder, location, full);
        if (singlePort)
          canEnqueue = andValues(
              bodyBuilder, location,
              {canEnqueue,
               circt::comb::createOrFoldNot(bodyBuilder, location, dequeue)});
        else if (fullReplacement)
          canEnqueue = circt::comb::OrOp::create(bodyBuilder, location,
                                                 canEnqueue, dequeue);
        // Enqueue readiness is the unit's cycle-start capacity alone. The
        // owning PE grants the unit's single enqueue service among its
        // requesters and commits at most one queue per cycle, so readiness
        // never observes a requester's own valid.
        std::vector<mlir::Value> enqueueFired(queueCount);
        for (std::uint32_t queue = 0; queue != queueCount; ++queue) {
          accessor.setOutput(queuePort("enqueue", queue, "_ready"), canEnqueue);
          enqueueFired[queue] =
              accessor.getInput(queuePort("enqueue", queue, "_commit"));
        }
        mlir::Value enqueue = orValues(bodyBuilder, location, enqueueFired);

        mlir::Value one = constant(bodyBuilder, location, occupancyWidth, 1);
        mlir::Value occupancyAfterDequeue = circt::comb::MuxOp::create(
            bodyBuilder, location, dequeue,
            circt::comb::SubOp::create(bodyBuilder, location, occupancy, one,
                                       true),
            occupancy, true);
        mlir::Value nextOccupancy = circt::comb::MuxOp::create(
            bodyBuilder, location, enqueue,
            circt::comb::AddOp::create(bodyBuilder, location,
                                       occupancyAfterDequeue, one, true),
            occupancyAfterDequeue, true);
        occupancyNext.setValue(nextOccupancy);

        std::vector<mlir::Value> removedSlot(
            depth, bitConstant(bodyBuilder, location, false));
        if (queueCount == 1) {
          removedSlot[0] = dequeueSelected[0];
        } else {
          for (std::uint32_t queue = 0; queue != queueCount; ++queue)
            for (std::uint32_t slot = 0; slot != depth; ++slot)
              removedSlot[slot] = circt::comb::OrOp::create(
                  bodyBuilder, location, removedSlot[slot],
                  andValues(bodyBuilder, location,
                            {dequeueSelected[queue], headSlots[queue][slot]}));
        }

        mlir::Value selectedPayload =
            payloadWidth == 0 ? mlir::Value{}
                              : zero(bodyBuilder, location, payloadWidth);
        mlir::Value selectedQueue = zero(bodyBuilder, location, queueWidth);
        for (std::uint32_t queue = 0; queue != queueCount; ++queue) {
          if (payloadWidth != 0)
            selectedPayload = circt::comb::MuxOp::create(
                bodyBuilder, location, enqueueFired[queue],
                accessor.getInput(queuePort("enqueue", queue, "_data")),
                selectedPayload, true);
          selectedQueue = circt::comb::MuxOp::create(
              bodyBuilder, location, enqueueFired[queue],
              constant(bodyBuilder, location, queueWidth, queue), selectedQueue,
              true);
        }

        const auto updateBank = [&](Bank &bank, mlir::Value appended) {
          if (bank.width == 0)
            return;
          mlir::Value shift = bitConstant(bodyBuilder, location, false);
          for (std::uint32_t slot = 0; slot != depth; ++slot) {
            if (queueCount == 1)
              shift = dequeue;
            else
              shift = circt::comb::OrOp::create(bodyBuilder, location, shift,
                                                removedSlot[slot]);
            mlir::Value shifted = slot + 1 < depth
                                      ? bank.current[slot + 1]
                                      : zero(bodyBuilder, location, bank.width);
            mlir::Value compacted = circt::comb::MuxOp::create(
                bodyBuilder, location,
                andValues(bodyBuilder, location, {dequeue, shift}), shifted,
                bank.current[slot], true);
            mlir::Value appendHere =
                andValues(bodyBuilder, location,
                          {enqueue, equals(bodyBuilder, location,
                                           occupancyAfterDequeue, slot)});
            bank.next[slot].setValue(circt::comb::MuxOp::create(
                bodyBuilder, location, appendHere, appended, compacted, true));
          }
        };
        updateBank(payload, selectedPayload);
        updateBank(queueIds, selectedQueue);
        if (materializationError)
          backedges.abandon();
      });
  if (materializationError)
    return invalid(*materializationError);
  return TokenPoolModule{module, queueCount, payloadWidth};
}

const FuModule *findFu(llvm::ArrayRef<FuModule> modules,
                       fabric::FabricFuOccurrenceRef reference) {
  const FuModule *result = nullptr;
  for (const FuModule &module : modules)
    if (module.reference == reference) {
      if (result)
        return nullptr;
      result = &module;
    }
  return result;
}

llvm::Error addCommonInputs(mlir::OpBuilder &builder, mlir::Location location,
                            circt::hw::HWModulePortAccessor &accessor,
                            const ConfigurationBundlePlan &parent,
                            const ConfigurationBundlePlan &childConfiguration,
                            circt::hw::HWModuleOp child,
                            std::map<std::string, mlir::Value> &inputs) {
  inputs.emplace("clock", accessor.getInput("clock"));
  inputs.emplace("reset", accessor.getInput("reset"));
  return addConfigurationInstanceInput(builder, location, accessor, parent,
                                       childConfiguration, child, inputs);
}

struct LogicalQueuePlan final {
  std::uint32_t context = 0;
  std::uint32_t fu = 0;
  std::uint32_t input = 0;
  std::uint32_t unit = 0;
  std::uint32_t unitQueue = 0;
};

struct AllocationUnitPlan final {
  std::vector<std::uint32_t> queues;
  TokenPoolModule pool;
  mlir::Value nearFull;
};

struct SelectorSignals final {
  mlir::Value route;
  mlir::Value discard;
  mlir::Value target;
  mlir::Value tag;
};

struct InstructionRowSignals final {
  mlir::Value active;
  mlir::Value selectedFu;
  std::vector<SelectorSignals> operands;
  std::vector<SelectorSignals> results;
};

SelectorSignals decodeSelector(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value field,
                               std::uint64_t offset, std::uint32_t targetWidth,
                               std::uint32_t tagWidth) {
  mlir::Value kind = extract(builder, location, field, offset, 2);
  return SelectorSignals{
      equals(builder, location, kind,
             static_cast<std::uint32_t>(
                 fabric::FabricTemporalPeSelectorKind::Route)),
      equals(builder, location, kind,
             static_cast<std::uint32_t>(
                 fabric::FabricTemporalPeSelectorKind::Discard)),
      extract(builder, location, field, offset + 2, targetWidth),
      extract(builder, location, field, offset + 2 + targetWidth, tagWidth)};
}

struct QueueRuntime final {
  circt::Backedge dequeueReady;
  circt::Backedge enqueueCommit;
  std::optional<mlir::Value> data;
  mlir::Value valid;
  mlir::Value enqueueReady;
};

struct FuOutputRuntime final {
  std::uint32_t fu = 0;
  std::uint32_t output = 0;
  const EndpointPlan *endpoint = nullptr;
  std::optional<mlir::Value> data;
  mlir::Value context;
  mlir::Value valid;
  circt::Backedge ready;
  circt::Backedge offered;
};

struct ResultRouteSignals final {
  mlir::Value active;
  mlir::Value route;
  mlir::Value discard;
  mlir::Value target;
  mlir::Value tag;
};

mlir::Value packToken(mlir::OpBuilder &builder, mlir::Location location,
                      std::optional<mlir::Value> data, mlir::Value tag) {
  if (!data)
    return tag;
  return circt::comb::ConcatOp::create(builder, location,
                                       llvm::ArrayRef<mlir::Value>{tag, *data});
}

} // namespace

llvm::Expected<PeModule> buildTemporalPeModule(
    mlir::OpBuilder &builder, mlir::Location location,
    fabric::SpatialCoreOccurrenceRef spatialCore,
    const fabric::FabricArtifactView &fabric,
    const ConfigurationABI &configurationAbi,
    const ConfigurationTransportLayout &transportLayout,
    llvm::ArrayRef<FuModule> fuModules, const ClockResetPlan &clockReset,
    fabric::FabricPeOccurrenceRef pe, llvm::StringRef materializationKey) {
  if (fabric.peSchedule(pe) != ::fabric::Schedule::Temporal)
    return invalid("Temporal PE lowering received a non-Temporal PE");
  auto endpoints = deriveEndpointPlans(
      builder, fabric, fabric::FabricTransportEndpointOwnerRef::of(pe));
  if (!endpoints)
    return endpoints.takeError();
  auto schema = fabric.temporalPeConfigurationSchema(pe);
  if (!schema)
    return schema.takeError();
  const auto &layout = schema->layout();
  auto decoder = prepareFieldDecoder(spatialCore, schema->field(),
                                     configurationAbi, transportLayout);
  if (!decoder)
    return decoder.takeError();
  if (decoder->encodedBitCount != layout.carrierBitCount)
    return invalid("Temporal PE carrier width differs from its typed schema");

  std::vector<const EndpointPlan *> inputEndpoints(layout.inputPortCount);
  std::vector<const EndpointPlan *> outputEndpoints(layout.outputPortCount);
  for (const EndpointPlan &endpoint : *endpoints) {
    auto &slot = endpoint.direction == fabric::FabricPortDirection::Input
                     ? inputEndpoints[endpoint.localOrdinal]
                     : outputEndpoints[endpoint.localOrdinal];
    if (slot)
      return invalid("Temporal PE endpoint ordinal is duplicated");
    slot = &endpoint;
  }
  if (llvm::is_contained(inputEndpoints, nullptr) ||
      llvm::is_contained(outputEndpoints, nullptr))
    return invalid("Temporal PE endpoint domain is not dense");
  const unsigned payloadWidth =
      inputEndpoints.front()->dataPath.payloadWidthBits;
  const unsigned tagWidth = inputEndpoints.front()->dataPath.tagWidthBits;
  if (tagWidth != layout.tagWidthBits)
    return invalid("Temporal PE tag width differs from its configuration");
  for (const EndpointPlan &endpoint : *endpoints)
    if (endpoint.dataPath.kind != ::fabric::DataPathKind::BitsTag ||
        endpoint.dataPath.payloadWidthBits != payloadWidth ||
        endpoint.dataPath.tagWidthBits != tagWidth)
      return invalid("Temporal PE boundary does not have one uniform shape");

  const unsigned contextWidth = indexWidth(layout.contextCount);
  std::vector<const FuModule *> children;
  children.reserve(layout.fus.size());
  for (const auto &shape : layout.fus) {
    const FuModule *child = findFu(fuModules, shape.fu);
    if (!child || child->contextWidthBits != contextWidth)
      return invalid("Temporal PE child FU has no compatible context port");
    children.push_back(child);
  }

  const auto mode = fabric.peOperandBufferMode(pe);
  if (!mode)
    return invalid("Temporal PE has no operand-buffer mode");
  const std::uint32_t operandDepth = fabric.peOperandBufferSize(pe);
  if (operandDepth == 0)
    return invalid("Temporal PE has zero operand-buffer depth");

  std::vector<std::uint32_t> fuInputCounts;
  fuInputCounts.reserve(layout.fus.size());
  for (const auto &shape : layout.fus)
    fuInputCounts.push_back(shape.inputCount);
  auto operandResources = ::fabric::TemporalOperandBufferContract::create(
      {pe, layout.contextCount, fuInputCounts, *mode, operandDepth});
  if (!operandResources)
    return invalid(llvm::toString(operandResources.takeError()));

  std::vector<LogicalQueuePlan> queues;
  queues.reserve(operandResources->logicalQueues().size());
  std::vector<std::vector<std::vector<std::uint32_t>>> queueOf(
      layout.contextCount,
      std::vector<std::vector<std::uint32_t>>(layout.fus.size()));
  for (std::uint32_t context = 0; context != layout.contextCount; ++context)
    for (std::uint32_t fu = 0; fu != layout.fus.size(); ++fu)
      queueOf[context][fu].resize(layout.fus[fu].inputCount);
  for (auto indexed : llvm::enumerate(operandResources->logicalQueues())) {
    const ::fabric::LogicalOperandQueueKey &key = indexed.value();
    if (key.context.pe != pe || key.context.ordinal >= layout.contextCount ||
        key.fuOccurrence >= layout.fus.size() ||
        key.fuInput >= layout.fus[key.fuOccurrence].inputCount)
      return invalid("Temporal PE operand contract has a foreign QueueKey");
    const std::uint32_t queue = static_cast<std::uint32_t>(indexed.index());
    const std::uint32_t unit = operandResources->allocationUnitOf(queue);
    queueOf[key.context.ordinal][key.fuOccurrence][key.fuInput] = queue;
    queues.push_back({static_cast<std::uint32_t>(key.context.ordinal),
                      static_cast<std::uint32_t>(key.fuOccurrence),
                      static_cast<std::uint32_t>(key.fuInput), unit, 0});
  }
  std::vector<AllocationUnitPlan> units(
      operandResources->allocationUnitCount());
  for (std::uint32_t unit = 0; unit != units.size(); ++unit)
    for (std::uint32_t queue : operandResources->queuesOf(unit)) {
      queues[queue].unitQueue = units[unit].queues.size();
      units[unit].queues.push_back(queue);
    }
  std::map<std::uint32_t, TokenPoolModule> operandPoolDefinitions;
  for (std::uint32_t unit = 0; unit != units.size(); ++unit) {
    const std::uint32_t queueCount = units[unit].queues.size();
    auto definition = operandPoolDefinitions.find(queueCount);
    if (definition == operandPoolDefinitions.end()) {
      auto pool =
          buildTokenPoolModule(builder, location, queueCount, operandDepth,
                               payloadWidth, false, false, true, clockReset);
      if (!pool)
        return pool.takeError();
      definition =
          operandPoolDefinitions.emplace(queueCount, std::move(*pool)).first;
    }
    units[unit].pool = definition->second;
  }

  const std::uint32_t fifoDepth = fabric.peRegisterFifoDepth(pe);
  const std::uint32_t fifoPorts = fabric.peRegisterFifoPorts(pe);
  auto peResources = ::fabric::TemporalPeResourceContract::create(
      {pe, layout.contextCount, fuInputCounts, *mode, operandDepth,
       layout.registerFifoCount, fifoDepth, fifoPorts});
  if (!peResources)
    return invalid(llvm::toString(peResources.takeError()));
  std::vector<TokenPoolModule> fifoPools;
  fifoPools.reserve(layout.registerFifoCount);
  if (layout.registerFifoCount != 0) {
    auto pool = buildTokenPoolModule(builder, location, 1, fifoDepth,
                                     payloadWidth + tagWidth, fifoPorts == 1,
                                     fifoPorts == 2, false, clockReset);
    if (!pool)
      return pool.takeError();
    fifoPools.assign(layout.registerFifoCount, *pool);
  }

  llvm::SmallVector<circt::hw::PortInfo, 32> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 32> outputs;
  std::vector<ConfigurationBundlePlan> childConfigurations;
  childConfigurations.reserve(children.size());
  for (const FuModule *child : children)
    childConfigurations.push_back(child->configuration);
  auto configuration = deriveConfigurationBundlePlan(
      llvm::ArrayRef<FieldDecoderPlan>(&*decoder, 1), childConfigurations);
  if (!configuration)
    return configuration.takeError();
  appendClockResetAndConfigurationPorts(builder, *configuration, inputs);
  for (const EndpointPlan &endpoint : *endpoints)
    appendEndpointPorts(inputs, outputs, endpoint);

  std::uint64_t fuInputCount = 0;
  std::uint64_t fuOutputCount = 0;
  std::uint64_t squaredFuInputCount = 0;
  for (const auto &shape : layout.fus) {
    fuInputCount += shape.inputCount;
    fuOutputCount += shape.outputCount;
    squaredFuInputCount += std::uint64_t(shape.inputCount) * shape.inputCount;
  }
  TemporalPeMaterializationMetrics metrics;

  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_temporal_pe_" + std::to_string(pe.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        metrics.beginOperations = blockOperationCount(bodyBuilder);
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        ConfigurationBundleSignals configurationValues =
            configurationBundleSignals(accessor, *configuration);
        mlir::Value field = decodeFieldSignal(bodyBuilder, location,
                                              configurationValues, *decoder);
        mlir::Value enabled = selectedBit(bodyBuilder, location, field, 0);
        std::vector<InstructionRowSignals> rows;
        rows.reserve(layout.contextCount);
        for (std::uint32_t context = 0; context != layout.contextCount;
             ++context) {
          InstructionRowSignals row;
          row.active =
              andValues(bodyBuilder, location,
                        {enabled, selectedBit(bodyBuilder, location, field,
                                              layout.rowOffset(context))});
          row.selectedFu = extract(bodyBuilder, location, field,
                                   layout.selectedFuOffset(context),
                                   layout.selectedFuBitCount);
          for (std::uint32_t input = 0; input != layout.maximumFuInputCount;
               ++input)
            row.operands.push_back(decodeSelector(
                bodyBuilder, location, field,
                layout.operandSelectionOffset(context, input),
                layout.inputTargetBitCount, layout.tagWidthBits));
          for (std::uint32_t output = 0; output != layout.maximumFuOutputCount;
               ++output)
            row.results.push_back(decodeSelector(
                bodyBuilder, location, field,
                layout.resultSelectionOffset(context, output),
                layout.outputTargetBitCount, layout.tagWidthBits));
          rows.push_back(std::move(row));
        }

        // Operand targets name the input ports first and then the register
        // FIFOs; without a FIFO bank every target is an input port and the
        // port count need not fit the target width.
        std::vector<std::vector<mlir::Value>> operandTargetsExternal(
            layout.contextCount,
            std::vector<mlir::Value>(layout.maximumFuInputCount));
        for (std::uint32_t context = 0; context != layout.contextCount;
             ++context)
          for (std::uint32_t input = 0; input != layout.maximumFuInputCount;
               ++input) {
            if (layout.registerFifoCount == 0) {
              operandTargetsExternal[context][input] =
                  bitConstant(bodyBuilder, location, true);
              continue;
            }
            mlir::Value target = rows[context].operands[input].target;
            const unsigned width =
                mlir::cast<mlir::IntegerType>(target.getType()).getWidth();
            operandTargetsExternal[context][input] =
                circt::comb::ICmpOp::create(
                    bodyBuilder, location, circt::comb::ICmpPredicate::ult,
                    target,
                    constant(bodyBuilder, location, width,
                             layout.inputPortCount),
                    true);
          }

        std::vector<llvm::SmallVector<mlir::Value>> inputMatchTerms(
            layout.inputPortCount);
        std::vector<QueueRuntime> queueRuntime(queues.size());
        std::vector<std::vector<mlir::Value>> queueInputMatches(
            queues.size(), std::vector<mlir::Value>(layout.inputPortCount));
        for (auto [unitOrdinal, unit] : llvm::enumerate(units)) {
          std::map<std::string, mlir::Value> instanceInputs;
          instanceInputs.emplace("clock", accessor.getInput("clock"));
          instanceInputs.emplace("reset", accessor.getInput("reset"));
          for (std::uint32_t local = 0; local != unit.queues.size(); ++local) {
            const std::uint32_t queueOrdinal = unit.queues[local];
            const LogicalQueuePlan &queue = queues[queueOrdinal];
            const InstructionRowSignals &row = rows[queue.context];
            const SelectorSignals &selector = row.operands[queue.input];
            mlir::Value rowSelectsFu =
                andValues(bodyBuilder, location,
                          {row.active, equals(bodyBuilder, location,
                                              row.selectedFu, queue.fu)});
            mlir::Value enqueueData =
                payloadWidth == 0 ? mlir::Value{}
                                  : zero(bodyBuilder, location, payloadWidth);
            for (std::uint32_t port = 0; port != inputEndpoints.size();
                 ++port) {
              mlir::Value targetMatches =
                  equals(bodyBuilder, location, selector.target, port);
              mlir::Value tagMatches = circt::comb::ICmpOp::create(
                  bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                  selector.tag,
                  accessor.getInput(inputEndpoints[port]->tag->getName()),
                  true);
              mlir::Value routeMatches = andValues(
                  bodyBuilder, location,
                  {rowSelectsFu, selector.route, targetMatches, tagMatches});
              mlir::Value discardMatches = andValues(
                  bodyBuilder, location,
                  {rowSelectsFu, selector.discard, targetMatches, tagMatches});
              queueInputMatches[queueOrdinal][port] = routeMatches;
              inputMatchTerms[port].push_back(discardMatches);
              if (payloadWidth != 0)
                enqueueData = circt::comb::MuxOp::create(
                    bodyBuilder, location, routeMatches,
                    accessor.getInput(inputEndpoints[port]->data->getName()),
                    enqueueData, true);
            }
            if (payloadWidth != 0)
              instanceInputs.emplace(queuePort("enqueue", local, "_data"),
                                     enqueueData);
            queueRuntime[queueOrdinal].enqueueCommit =
                backedges.get(bodyBuilder.getI1Type());
            instanceInputs.emplace(queuePort("enqueue", local, "_commit"),
                                   queueRuntime[queueOrdinal].enqueueCommit);
            queueRuntime[queueOrdinal].dequeueReady =
                backedges.get(bodyBuilder.getI1Type());
            instanceInputs.emplace(queuePort("dequeue", local, "_ready"),
                                   queueRuntime[queueOrdinal].dequeueReady);
          }
          auto instance = instantiateModule(
              bodyBuilder, location, unit.pool.module,
              "operand_pool_" + std::to_string(unitOrdinal), instanceInputs);
          if (!instance) {
            materializationError = llvm::toString(instance.takeError());
            backedges.abandon();
            return;
          }
          unit.nearFull = instance->at("near_full");
          for (std::uint32_t local = 0; local != unit.queues.size(); ++local) {
            QueueRuntime &runtime = queueRuntime[unit.queues[local]];
            if (payloadWidth != 0)
              runtime.data = instance->at(queuePort("dequeue", local, "_data"));
            runtime.valid = instance->at(queuePort("dequeue", local, "_valid"));
            runtime.enqueueReady =
                instance->at(queuePort("enqueue", local, "_ready"));
          }
        }

        metrics.decodeAndPoolOperations =
            blockOperationCount(bodyBuilder) - metrics.beginOperations;
        // Ingress admission follows the Fabric operand-buffer contract: a
        // boundary token is accepted only when every logical queue it matches
        // has cycle-start capacity and holds its allocation unit's single
        // enqueue service. Readiness observes configuration, tags, cycle-start
        // queue state, and the competing requesters of a shared unit; it never
        // observes the port's own valid, so an atomic upstream fanout cannot
        // close a combinational loop through this boundary.
        std::vector<std::vector<mlir::Value>> queueArrivals(
            queues.size(), std::vector<mlir::Value>(layout.inputPortCount));
        for (std::uint32_t queue = 0; queue != queues.size(); ++queue)
          for (std::uint32_t port = 0; port != layout.inputPortCount; ++port)
            queueArrivals[queue][port] = andValues(
                bodyBuilder, location,
                {queueInputMatches[queue][port],
                 accessor.getInput(inputEndpoints[port]->valid.getName())});

        // Each port's transaction is classified from its own match set and
        // the cycle-start queue heads: it completes a context/FU tuple, it
        // complements a partial tuple whose occupied role is near full, or it
        // is ordinary.
        std::vector<mlir::Value> portCompletes(
            layout.inputPortCount, bitConstant(bodyBuilder, location, false));
        std::vector<mlir::Value> portNearFullComplement(
            layout.inputPortCount, bitConstant(bodyBuilder, location, false));
        for (std::uint32_t context = 0; context != layout.contextCount;
             ++context)
          for (std::uint32_t fu = 0; fu != layout.fus.size(); ++fu) {
            const std::uint32_t inputCount = layout.fus[fu].inputCount;
            llvm::SmallVector<mlir::Value> roleUses;
            llvm::SmallVector<mlir::Value> headPresent;
            llvm::SmallVector<mlir::Value> occupiedNearFull;
            for (std::uint32_t input = 0; input != inputCount; ++input) {
              const std::uint32_t queue = queueOf[context][fu][input];
              roleUses.push_back(andValues(
                  bodyBuilder, location,
                  {rows[context].active,
                   equals(bodyBuilder, location, rows[context].selectedFu, fu),
                   rows[context].operands[input].route,
                   operandTargetsExternal[context][input]}));
              headPresent.push_back(queueRuntime[queue].valid);
              occupiedNearFull.push_back(
                  andValues(bodyBuilder, location,
                            {roleUses.back(), headPresent.back(),
                             units[queues[queue].unit].nearFull}));
            }
            const mlir::Value anyOccupiedNearFull =
                orValues(bodyBuilder, location, occupiedNearFull);
            for (std::uint32_t port = 0; port != layout.inputPortCount;
                 ++port) {
              llvm::SmallVector<mlir::Value> fills;
              llvm::SmallVector<mlir::Value> present;
              for (std::uint32_t input = 0; input != inputCount; ++input) {
                const mlir::Value matched =
                    queueInputMatches[queueOf[context][fu][input]][port];
                fills.push_back(andValues(
                    bodyBuilder, location,
                    {matched, circt::comb::createOrFoldNot(
                                  bodyBuilder, location, headPresent[input])}));
                present.push_back(orValues(
                    bodyBuilder, location,
                    {circt::comb::createOrFoldNot(bodyBuilder, location,
                                                  roleUses[input]),
                     headPresent[input], matched}));
              }
              const mlir::Value fillsMissingRole =
                  orValues(bodyBuilder, location, fills);
              const mlir::Value completeAfterIngress =
                  andValues(bodyBuilder, location, present);
              portCompletes[port] = orValues(
                  bodyBuilder, location,
                  {portCompletes[port],
                   andValues(bodyBuilder, location,
                             {fillsMissingRole, completeAfterIngress})});
              portNearFullComplement[port] = orValues(
                  bodyBuilder, location,
                  {portNearFullComplement[port],
                   andValues(bodyBuilder, location,
                             {fillsMissingRole,
                              circt::comb::createOrFoldNot(
                                  bodyBuilder, location, completeAfterIngress),
                              anyOccupiedNearFull})});
            }
          }
        std::vector<mlir::Value> portRank(layout.inputPortCount);
        for (std::uint32_t port = 0; port != layout.inputPortCount; ++port)
          portRank[port] = packBits(
              bodyBuilder, location,
              {andValues(bodyBuilder, location,
                         {circt::comb::createOrFoldNot(bodyBuilder, location,
                                                       portCompletes[port]),
                          portNearFullComplement[port]}),
               portCompletes[port]});

        // A shared allocation unit grants its enqueue service to the
        // highest-class requesting queue, then round-robin over the unit's
        // canonical queue order from a cursor that advances only on a
        // committed enqueue. A queue's grant never observes its own request, so
        // the boundary ready stays independent of that port's valid. A
        // dedicated unit has at most one requester and carries no policy.
        std::vector<mlir::Value> queueGrant(
            queues.size(), bitConstant(bodyBuilder, location, true));
        for (auto [unitOrdinal, unit] : llvm::enumerate(units)) {
          const std::size_t memberCount = unit.queues.size();
          if (memberCount < 2)
            continue;
          std::vector<mlir::Value> active(memberCount);
          std::vector<mlir::Value> rank(memberCount);
          for (std::size_t member = 0; member != memberCount; ++member) {
            const std::uint32_t queue = unit.queues[member];
            active[member] =
                orValues(bodyBuilder, location, queueArrivals[queue]);
            mlir::Value memberRank = zero(bodyBuilder, location, 2);
            for (std::uint32_t port = 0; port != layout.inputPortCount;
                 ++port)
              memberRank = circt::comb::MuxOp::create(
                  bodyBuilder, location, queueInputMatches[queue][port],
                  portRank[port], memberRank, true);
            rank[member] = memberRank;
          }
          const unsigned cursorWidth = indexWidth(memberCount);
          circt::Backedge cursorNext =
              backedges.get(bodyBuilder.getIntegerType(cursorWidth));
          mlir::Value cursor = createRegister(
              bodyBuilder, location, cursorNext, accessor.getInput("clock"),
              accessor.getInput("reset"), llvm::APInt(cursorWidth, 0),
              "operand_unit_" + std::to_string(unitOrdinal) +
                  "_enqueue_cursor_reg",
              clockReset.asynchronousReset);
          for (std::size_t member = 0; member != memberCount; ++member) {
            llvm::SmallVector<mlir::Value> higher;
            llvm::SmallVector<mlir::Value> candidates;
            for (std::size_t other = 0; other != memberCount; ++other) {
              if (other == member) {
                candidates.push_back(bitConstant(bodyBuilder, location, true));
                continue;
              }
              higher.push_back(andValues(
                  bodyBuilder, location,
                  {active[other],
                   circt::comb::ICmpOp::create(
                       bodyBuilder, location, circt::comb::ICmpPredicate::ugt,
                       rank[other], rank[member], true)}));
              candidates.push_back(andValues(
                  bodyBuilder, location,
                  {active[other],
                   circt::comb::ICmpOp::create(
                       bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                       rank[other], rank[member], true)}));
            }
            mlir::Value selected = roundRobinPackedSelection(
                bodyBuilder, location, candidates, cursor);
            queueGrant[unit.queues[member]] = andValues(
                bodyBuilder, location,
                {circt::comb::createOrFoldNot(
                     bodyBuilder, location,
                     orValues(bodyBuilder, location, higher)),
                 circt::comb::ExtractOp::create(bodyBuilder, location,
                                                selected, member, 1)});
          }
          llvm::SmallVector<mlir::Value> committedMembers;
          for (std::uint32_t queue : unit.queues)
            committedMembers.push_back(queueRuntime[queue].enqueueCommit);
          cursorNext.setValue(
              nextCursor(bodyBuilder, location, cursor, committedMembers));
        }

        // ready = any_match AND AND(!match[i] OR queue_ready[i]);
        // fire = input_valid AND ready; enqueue[i] = fire AND match[i].
        std::vector<mlir::Value> queueReady;
        queueReady.reserve(queues.size());
        for (std::uint32_t queue = 0; queue != queues.size(); ++queue)
          queueReady.push_back(
              andValues(bodyBuilder, location,
                        {queueRuntime[queue].enqueueReady, queueGrant[queue]}));
        mlir::Value queueReadyPacked =
            packBits(bodyBuilder, location, queueReady);
        const llvm::APInt zeroQueueMask(queues.size(), 0);
        std::vector<mlir::Value> inputReady(layout.inputPortCount);
        llvm::SmallVector<mlir::Value> committedByPort;
        committedByPort.reserve(layout.inputPortCount);
        for (std::uint32_t port = 0; port != layout.inputPortCount; ++port) {
          llvm::SmallVector<mlir::Value> matches;
          matches.reserve(queues.size());
          for (std::uint32_t queue = 0; queue != queues.size(); ++queue)
            matches.push_back(queueInputMatches[queue][port]);
          mlir::Value matchesPacked = packBits(bodyBuilder, location, matches);
          mlir::Value anyRouteMatch = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::ne,
              matchesPacked,
              circt::hw::ConstantOp::create(bodyBuilder, location,
                                            zeroQueueMask),
              true);
          mlir::Value blocked = circt::comb::AndOp::create(
              bodyBuilder, location, matchesPacked,
              circt::comb::createOrFoldNot(bodyBuilder, location,
                                           queueReadyPacked),
              true);
          mlir::Value unblocked = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::eq, blocked,
              circt::hw::ConstantOp::create(bodyBuilder, location,
                                            zeroQueueMask),
              true);
          inputReady[port] = andValues(
              bodyBuilder, location,
              {orValues(bodyBuilder, location,
                        {anyRouteMatch,
                         orValues(bodyBuilder, location,
                                  inputMatchTerms[port])}),
               unblocked});
          mlir::Value fired = andValues(
              bodyBuilder, location,
              {inputReady[port],
               accessor.getInput(inputEndpoints[port]->valid.getName())});
          committedByPort.push_back(circt::comb::AndOp::create(
              bodyBuilder, location, matchesPacked,
              splatBit(bodyBuilder, location, fired, queues.size()), true));
        }
        mlir::Value committed =
            orValues(bodyBuilder, location, committedByPort);
        for (std::uint32_t queue = 0; queue != queues.size(); ++queue)
          queueRuntime[queue].enqueueCommit.setValue(
              circt::comb::ExtractOp::create(bodyBuilder, location, committed,
                                             queue, 1));

        metrics.ingressOperations = blockOperationCount(bodyBuilder) -
                                    metrics.beginOperations -
                                    metrics.decodeAndPoolOperations;
        const unsigned fifoPayloadWidth = payloadWidth + tagWidth;
        std::vector<circt::Backedge> fifoHeadPayload;
        std::vector<circt::Backedge> fifoHeadValid;
        std::vector<mlir::Value> fifoHeadData;
        std::vector<mlir::Value> fifoHeadTag;
        fifoHeadPayload.reserve(layout.registerFifoCount);
        fifoHeadValid.reserve(layout.registerFifoCount);
        fifoHeadData.reserve(layout.registerFifoCount);
        fifoHeadTag.reserve(layout.registerFifoCount);
        for (std::uint32_t fifo = 0; fifo != layout.registerFifoCount; ++fifo) {
          fifoHeadPayload.push_back(
              backedges.get(bodyBuilder.getIntegerType(fifoPayloadWidth)));
          fifoHeadValid.push_back(backedges.get(bodyBuilder.getI1Type()));
          if (payloadWidth != 0)
            fifoHeadData.push_back(extract(bodyBuilder, location,
                                           fifoHeadPayload.back(), 0,
                                           payloadWidth));
          fifoHeadTag.push_back(extract(bodyBuilder, location,
                                        fifoHeadPayload.back(), payloadWidth,
                                        tagWidth));
        }

        const unsigned idleFuWidth = indexWidth(layout.fus.size());
        std::vector<std::vector<mlir::Value>> contextEligible(
            layout.fus.size(), std::vector<mlir::Value>(layout.contextCount));
        for (std::uint32_t fu = 0; fu != layout.fus.size(); ++fu)
          for (std::uint32_t context = 0; context != layout.contextCount;
               ++context) {
            const InstructionRowSignals &row = rows[context];
            contextEligible[fu][context] =
                andValues(bodyBuilder, location,
                          {row.active,
                           equals(bodyBuilder, location, row.selectedFu, fu)});
          }

        std::vector<std::vector<mlir::Value>> contextSelected(
            layout.fus.size(), std::vector<mlir::Value>(
                                   layout.contextCount,
                                   bitConstant(bodyBuilder, location, false)));
        // The context-evaluation service is one shared unit for the PE or
        // one independently rotating unit per FU. Each unit reports the
        // cycle in which its grant restarts a pass over the eligible
        // candidates and, when shared, the FU it grants this cycle; the
        // result-egress idle presentation below follows those passes.
        const std::uint32_t dispatchUnitCount = peResources->dispatchUnitCount();
        std::vector<mlir::Value> unitPassStart(dispatchUnitCount);
        std::vector<std::optional<std::uint32_t>> unitOfFu(layout.fus.size());
        mlir::Value sharedDispatchFu;
        for (std::uint32_t unit = 0; unit != dispatchUnitCount; ++unit) {
          std::vector<mlir::Value> requests;
          const auto candidates = peResources->dispatchCandidatesOf(unit);
          requests.reserve(candidates.size());
          for (std::uint32_t candidateOrdinal : candidates) {
            const auto &candidate =
                peResources->dispatchCandidates()[candidateOrdinal];
            requests.push_back(contextEligible[candidate.fuOccurrence]
                                              [candidate.context.ordinal]);
          }
          StatefulSelection selection = makeStatefulSelection(
              bodyBuilder, location, backedges, requests,
              accessor.getInput("clock"), accessor.getInput("reset"),
              "dispatch_unit_" + std::to_string(unit) + "_cursor_reg",
              clockReset);
          const unsigned cursorWidth =
              mlir::cast<mlir::IntegerType>(selection.cursor.getType())
                  .getWidth();
          llvm::SmallVector<mlir::Value> wrapped;
          mlir::Value grantedFu = zero(bodyBuilder, location, idleFuWidth);
          bool singleFu = true;
          for (auto [request, candidateOrdinal] : llvm::enumerate(candidates)) {
            const auto &candidate =
                peResources->dispatchCandidates()[candidateOrdinal];
            contextSelected[candidate.fuOccurrence][candidate.context.ordinal] =
                selection.selected[request];
            wrapped.push_back(andValues(
                bodyBuilder, location,
                {selection.selected[request],
                 circt::comb::ICmpOp::create(
                     bodyBuilder, location, circt::comb::ICmpPredicate::ult,
                     constant(bodyBuilder, location, cursorWidth, request),
                     selection.cursor, true)}));
            grantedFu = circt::comb::MuxOp::create(
                bodyBuilder, location, selection.selected[request],
                constant(bodyBuilder, location, idleFuWidth,
                         candidate.fuOccurrence),
                grantedFu, true);
            singleFu = singleFu && candidate.fuOccurrence ==
                                       peResources->dispatchCandidates()
                                           [candidates.front()]
                                               .fuOccurrence;
          }
          // A pass restarts when the grant searched from ordinal 0 (the
          // cursor wrapped) or wrapped around the end of the domain.
          unitPassStart[unit] = andValues(
              bodyBuilder, location,
              {orValues(bodyBuilder, location, selection.selected),
               orValues(bodyBuilder, location,
                        {equals(bodyBuilder, location, selection.cursor, 0),
                         orValues(bodyBuilder, location, wrapped)})});
          if (singleFu && !candidates.empty())
            unitOfFu[peResources->dispatchCandidates()[candidates.front()]
                         .fuOccurrence] = unit;
          if (dispatchUnitCount == 1)
            sharedDispatchFu = grantedFu;
          advanceStatefulSelection(bodyBuilder, location, selection,
                                   selection.selected);
        }

        std::vector<mlir::Value> selectedContext(layout.fus.size());
        for (std::uint32_t fu = 0; fu != layout.fus.size(); ++fu) {
          mlir::Value value = zero(bodyBuilder, location, contextWidth);
          for (std::uint32_t context = 0; context != layout.contextCount;
               ++context)
            value = circt::comb::MuxOp::create(
                bodyBuilder, location, contextSelected[fu][context],
                constant(bodyBuilder, location, contextWidth, context), value,
                true);
          selectedContext[fu] = value;
        }

        mlir::Value packedFifoRequesterSelected;
        mlir::Value packedFifoRequesterKind;
        if (layout.registerFifoCount != 0) {
          llvm::SmallVector<mlir::Value> selectedDomain;
          llvm::SmallVector<mlir::Value> kindDomain;
          for (std::uint32_t fu = 0; fu != layout.fus.size(); ++fu)
            for (std::uint32_t context = 0; context != layout.contextCount;
                 ++context)
              for (std::uint32_t input = 0; input != layout.fus[fu].inputCount;
                   ++input) {
                const SelectorSignals &selector = rows[context].operands[input];
                for (bool discard : {false, true}) {
                  selectedDomain.push_back(contextSelected[fu][context]);
                  kindDomain.push_back(discard ? selector.discard
                                               : selector.route);
                }
              }
          packedFifoRequesterSelected =
              packBits(bodyBuilder, location, selectedDomain);
          packedFifoRequesterKind = packBits(bodyBuilder, location, kindDomain);
        }

        std::vector<mlir::Value> queueGranted(
            queues.size(), bitConstant(bodyBuilder, location, false));
        for (std::uint32_t unit = 0; unit != units.size(); ++unit) {
          std::vector<mlir::Value> requests;
          requests.reserve(units[unit].queues.size());
          for (std::uint32_t queueOrdinal : units[unit].queues) {
            const LogicalQueuePlan &queue = queues[queueOrdinal];
            const SelectorSignals &selector =
                rows[queue.context].operands[queue.input];
            requests.push_back(andValues(
                bodyBuilder, location,
                {contextSelected[queue.fu][queue.context], selector.route,
                 operandTargetsExternal[queue.context][queue.input],
                 queueRuntime[queueOrdinal].valid}));
          }
          StatefulSelection selection = makeStatefulSelection(
              bodyBuilder, location, backedges, requests,
              accessor.getInput("clock"), accessor.getInput("reset"),
              "operand_unit_" + std::to_string(unit) + "_read_cursor_reg",
              clockReset);
          for (std::uint32_t local = 0; local != units[unit].queues.size();
               ++local)
            queueGranted[units[unit].queues[local]] = selection.selected[local];
          advanceStatefulSelection(bodyBuilder, location, selection,
                                   selection.selected);
        }

        struct FifoReadCandidate final {
          std::uint32_t fifo = 0;
          std::uint32_t fu = 0;
          std::uint32_t context = 0;
          std::uint32_t input = 0;
          bool discard = false;
          std::size_t selectionOrdinal = 0;
        };
        std::vector<FifoReadCandidate> fifoReadCandidates;
        std::vector<mlir::Value> fifoReadSelected(layout.registerFifoCount);
        for (std::uint32_t fifo = 0; fifo != layout.registerFifoCount; ++fifo) {
          std::vector<std::vector<mlir::Value>> selectorMatches(
              layout.contextCount,
              std::vector<mlir::Value>(layout.maximumFuInputCount));
          for (std::uint32_t context = 0; context != layout.contextCount;
               ++context)
            for (std::uint32_t input = 0; input != layout.maximumFuInputCount;
                 ++input) {
              const SelectorSignals &selector = rows[context].operands[input];
              mlir::Value targetMatches =
                  equals(bodyBuilder, location, selector.target,
                         layout.inputPortCount + fifo);
              mlir::Value tagMatches = circt::comb::ICmpOp::create(
                  bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                  selector.tag, fifoHeadTag[fifo], true);
              selectorMatches[context][input] =
                  andValues(bodyBuilder, location,
                            {targetMatches, fifoHeadValid[fifo], tagMatches});
            }
          std::vector<mlir::Value> matchDomain;
          for (std::uint32_t fu = 0; fu != layout.fus.size(); ++fu)
            for (std::uint32_t context = 0; context != layout.contextCount;
                 ++context)
              for (std::uint32_t input = 0; input != layout.fus[fu].inputCount;
                   ++input) {
                for (bool discard : {false, true}) {
                  const std::size_t selectionOrdinal = matchDomain.size();
                  matchDomain.push_back(selectorMatches[context][input]);
                  fifoReadCandidates.push_back(
                      {fifo, fu, context, input, discard, selectionOrdinal});
                }
              }
          const unsigned requestCount = matchDomain.size();
          mlir::Value packedRequests =
              andValues(bodyBuilder, location,
                        {packedFifoRequesterSelected,
                         packBits(bodyBuilder, location, matchDomain),
                         packedFifoRequesterKind});
          const unsigned width = indexWidth(requestCount);
          circt::Backedge next =
              backedges.get(bodyBuilder.getIntegerType(width));
          mlir::Value cursor = createRegister(
              bodyBuilder, location, next, accessor.getInput("clock"),
              accessor.getInput("reset"), llvm::APInt(width, 0),
              "register_fifo_" + std::to_string(fifo) + "_read_cursor_reg",
              clockReset.asynchronousReset);
          fifoReadSelected[fifo] = roundRobinPackedSelection(
              bodyBuilder, location, packedRequests, requestCount, cursor);
          next.setValue(nextCursorFromPacked(bodyBuilder, location, cursor,
                                             fifoReadSelected[fifo],
                                             requestCount));
        }

        std::vector<std::vector<std::vector<std::vector<mlir::Value>>>>
            fifoReadGranted(layout.fus.size());
        for (std::uint32_t fu = 0; fu != layout.fus.size(); ++fu) {
          fifoReadGranted[fu].resize(layout.contextCount);
          for (std::uint32_t context = 0; context != layout.contextCount;
               ++context) {
            fifoReadGranted[fu][context].resize(layout.fus[fu].inputCount);
            for (std::uint32_t input = 0; input != layout.fus[fu].inputCount;
                 ++input)
              fifoReadGranted[fu][context][input].resize(
                  layout.registerFifoCount);
          }
        }
        for (const FifoReadCandidate &candidate : fifoReadCandidates)
          if (!candidate.discard)
            fifoReadGranted[candidate.fu][candidate.context][candidate.input]
                           [candidate.fifo] = circt::comb::ExtractOp::create(
                               bodyBuilder, location,
                               fifoReadSelected[candidate.fifo],
                               candidate.selectionOrdinal, 1);

        std::vector<std::vector<mlir::Value>> fuInputReady(layout.fus.size());
        std::vector<FuOutputRuntime> fuOutputs;
        metrics.dispatchOperations =
            blockOperationCount(bodyBuilder) - metrics.beginOperations -
            metrics.decodeAndPoolOperations - metrics.ingressOperations;
        for (std::uint32_t fu = 0; fu != layout.fus.size(); ++fu) {
          const FuModule &child = *children[fu];
          std::map<std::string, mlir::Value> instanceInputs;
          if (llvm::Error error = addCommonInputs(
                  bodyBuilder, location, accessor, *configuration,
                  child.configuration, child.module, instanceInputs)) {
            materializationError = llvm::toString(std::move(error));
            backedges.abandon();
            return;
          }
          fuInputReady[fu].resize(layout.fus[fu].inputCount);
          std::vector<circt::Backedge> outputReady(layout.fus[fu].outputCount);
          std::vector<circt::Backedge> outputOffered(
              layout.fus[fu].outputCount);
          for (const EndpointPlan &endpoint : child.endpoints) {
            if (endpoint.direction == fabric::FabricPortDirection::Output) {
              outputReady[endpoint.localOrdinal] =
                  backedges.get(bodyBuilder.getI1Type());
              instanceInputs.emplace(endpoint.ready.getName().str(),
                                     outputReady[endpoint.localOrdinal]);
              outputOffered[endpoint.localOrdinal] =
                  backedges.get(bodyBuilder.getI1Type());
              instanceInputs.emplace(
                  "output_" + std::to_string(endpoint.localOrdinal) +
                      "_offered",
                  outputOffered[endpoint.localOrdinal]);
              continue;
            }
            llvm::SmallVector<mlir::Value> contextValid;
            llvm::SmallVector<mlir::Value> contextData;
            contextValid.reserve(layout.contextCount);
            contextData.reserve(layout.contextCount);
            for (std::uint32_t context = 0; context != layout.contextCount;
                 ++context) {
              const SelectorSignals &selector =
                  rows[context].operands[endpoint.localOrdinal];
              const std::uint32_t queue =
                  queueOf[context][fu][endpoint.localOrdinal];
              mlir::Value queuePath = andValues(
                  bodyBuilder, location,
                  {queueGranted[queue],
                   operandTargetsExternal[context][endpoint.localOrdinal],
                   queueRuntime[queue].valid});

              const std::uint64_t targetDomainSize =
                  std::uint64_t{1} << layout.inputTargetBitCount;
              llvm::SmallVector<mlir::Value> targetGrants(
                  targetDomainSize, bitConstant(bodyBuilder, location, false));
              for (std::uint32_t fifo = 0; fifo != layout.registerFifoCount;
                   ++fifo)
                targetGrants[layout.inputPortCount + fifo] =
                    fifoReadGranted[fu][context][endpoint.localOrdinal][fifo];
              mlir::Value fifoGrant = indexedValue(
                  bodyBuilder, location, selector.target, targetGrants);
              contextValid.push_back(
                  orValues(bodyBuilder, location, {queuePath, fifoGrant}));
              if (payloadWidth != 0) {
                llvm::SmallVector<mlir::Value> targetData(
                    targetDomainSize,
                    zero(bodyBuilder, location, payloadWidth));
                for (std::uint32_t fifo = 0; fifo != layout.registerFifoCount;
                     ++fifo)
                  targetData[layout.inputPortCount + fifo] = fifoHeadData[fifo];
                mlir::Value selectedFifoData = indexedValue(
                    bodyBuilder, location, selector.target, targetData);
                contextData.push_back(circt::comb::MuxOp::create(
                    bodyBuilder, location, fifoGrant, selectedFifoData,
                    *queueRuntime[queue].data, true));
              }
            }
            mlir::Value valid = indexedValue(bodyBuilder, location,
                                             selectedContext[fu], contextValid);
            mlir::Value data =
                payloadWidth == 0
                    ? mlir::Value{}
                    : indexedValue(bodyBuilder, location, selectedContext[fu],
                                   contextData);
            ForwardTransportSignals source{
                valid,
                endpoint.data ? std::optional<mlir::Value>{data} : std::nullopt,
                std::nullopt};
            auto adapted = adaptForwardTransportSignals(
                bodyBuilder, location,
                ::fabric::DataPathType{::fabric::DataPathKind::Bits,
                                       payloadWidth, 0},
                endpoint.dataPath, source);
            if (!adapted) {
              materializationError = llvm::toString(adapted.takeError());
              backedges.abandon();
              return;
            }
            if (endpoint.data)
              instanceInputs.emplace(endpoint.data->getName().str(),
                                     *adapted->payload);
            instanceInputs.emplace(endpoint.valid.getName().str(),
                                   adapted->valid);
          }
          instanceInputs.emplace(dispatchContextPortName.str(),
                                 selectedContext[fu]);

          auto instance = instantiateModule(
              bodyBuilder, location, child.module,
              "fu_" + std::to_string(child.reference.id()), instanceInputs);
          if (!instance) {
            materializationError = llvm::toString(instance.takeError());
            backedges.abandon();
            return;
          }
          for (const EndpointPlan &endpoint : child.endpoints) {
            if (endpoint.direction == fabric::FabricPortDirection::Input) {
              fuInputReady[fu][endpoint.localOrdinal] =
                  instance->at(endpoint.ready.getName().str());
              continue;
            }
            FuOutputRuntime output;
            output.fu = fu;
            output.output = endpoint.localOrdinal;
            output.endpoint = &endpoint;
            if (endpoint.data)
              output.data = instance->at(endpoint.data->getName().str());
            output.context = instance->at(
                "output_" + std::to_string(endpoint.localOrdinal) + "_context");
            output.valid = instance->at(endpoint.valid.getName().str());
            output.ready = std::move(outputReady[endpoint.localOrdinal]);
            output.offered = std::move(outputOffered[endpoint.localOrdinal]);
            fuOutputs.push_back(std::move(output));
          }
        }

        for (std::uint32_t queue = 0; queue != queues.size(); ++queue) {
          const LogicalQueuePlan &plan = queues[queue];
          queueRuntime[queue].dequeueReady.setValue(andValues(
              bodyBuilder, location,
              {queueGranted[queue], fuInputReady[plan.fu][plan.input]}));
        }

        std::vector<mlir::Value> fifoReadReady(layout.registerFifoCount);
        for (std::uint32_t fifo = 0; fifo != layout.registerFifoCount; ++fifo) {
          llvm::SmallVector<mlir::Value> commitEligible;
          for (const FifoReadCandidate &candidate : fifoReadCandidates)
            if (candidate.fifo == fifo)
              commitEligible.push_back(
                  candidate.discard
                      ? bitConstant(bodyBuilder, location, true)
                      : fuInputReady[candidate.fu][candidate.input]);
          mlir::Value eligible =
              packBits(bodyBuilder, location, commitEligible);
          mlir::Value committed = circt::comb::AndOp::create(
              bodyBuilder, location, fifoReadSelected[fifo], eligible, true);
          fifoReadReady[fifo] = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::ne, committed,
              circt::hw::ConstantOp::create(
                  bodyBuilder, location, llvm::APInt(commitEligible.size(), 0)),
              true);
        }

        metrics.childOperations =
            blockOperationCount(bodyBuilder) - metrics.beginOperations -
            metrics.decodeAndPoolOperations - metrics.ingressOperations -
            metrics.dispatchOperations;
        std::vector<ResultRouteSignals> resultRoutes;
        resultRoutes.reserve(fuOutputs.size());
        for (const FuOutputRuntime &output : fuOutputs) {
          std::vector<mlir::Value> active;
          std::vector<mlir::Value> route;
          std::vector<mlir::Value> discard;
          std::vector<mlir::Value> target;
          std::vector<mlir::Value> tag;
          active.reserve(layout.contextCount);
          route.reserve(layout.contextCount);
          discard.reserve(layout.contextCount);
          target.reserve(layout.contextCount);
          tag.reserve(layout.contextCount);
          for (std::uint32_t context = 0; context != layout.contextCount;
               ++context) {
            const InstructionRowSignals &row = rows[context];
            const SelectorSignals &selector = row.results[output.output];
            active.push_back(
                andValues(bodyBuilder, location,
                          {row.active, equals(bodyBuilder, location,
                                              row.selectedFu, output.fu)}));
            route.push_back(selector.route);
            discard.push_back(selector.discard);
            target.push_back(selector.target);
            tag.push_back(selector.tag);
          }
          mlir::Value selectedActive =
              selectValue(bodyBuilder, location, output.context, active);
          resultRoutes.push_back(
              {selectedActive,
               andValues(bodyBuilder, location,
                         {selectedActive, selectValue(bodyBuilder, location,
                                                      output.context, route)}),
               andValues(
                   bodyBuilder, location,
                   {selectedActive, selectValue(bodyBuilder, location,
                                                output.context, discard)}),
               selectValue(bodyBuilder, location, output.context, target),
               selectValue(bodyBuilder, location, output.context, tag)});
        }

        std::vector<std::optional<mlir::Value>> adaptedFuOutputData(
            fuOutputs.size());
        if (payloadWidth != 0)
          for (std::size_t candidate = 0; candidate != fuOutputs.size();
               ++candidate) {
            const FuOutputRuntime &output = fuOutputs[candidate];
            auto adapted = adaptForwardTransportSignals(
                bodyBuilder, location, output.endpoint->dataPath,
                ::fabric::DataPathType{::fabric::DataPathKind::Bits,
                                       payloadWidth, 0},
                ForwardTransportSignals{output.valid, output.data,
                                        std::nullopt});
            if (!adapted) {
              materializationError = llvm::toString(adapted.takeError());
              backedges.abandon();
              return;
            }
            adaptedFuOutputData[candidate] = *adapted->payload;
          }

        metrics.resultProjectionOperations =
            blockOperationCount(bodyBuilder) - metrics.beginOperations -
            metrics.decodeAndPoolOperations - metrics.ingressOperations -
            metrics.dispatchOperations - metrics.childOperations;
        // Result egress: valid FU outputs whose resident row routes them to a
        // port or register FIFO are offered by the canonical round-robin
        // policy; the cursor advances past every offered requester, accepted
        // or refused, so a result whose downstream is not ready for its tag
        // cannot hold the port against the other valid results (the offer
        // rotation of the per-tag virtual channel discipline), and every FU
        // output learns through its offered signal when the PE presents it.
        // Readiness must be observable before valid, because an operation
        // that publishes several results atomically asserts each result's
        // valid only once its peers are ready. While no valid requester holds
        // a port, the port presents one idle candidate and its configured
        // tag, so downstream tag-dispatched readiness applies to that
        // candidate. The idle candidate is the first routed output of one
        // idle FU, shared by every port of the PE so the several results of one
        // FU's operation are presented on their ports in the same cycle. The
        // idle FU follows the context-evaluation service: under a shared
        // service it is the FU granted this cycle; under per-FU services a
        // pointer holds each FU that has eligible rows for one complete pass
        // of its dispatch rotation, from one pass start to the next, so every
        // resident context of every FU is presented on its ports while it is
        // the dispatch context. A free-running pointer could never align with
        // a rotation whose period shares a factor with the FU count.
        std::vector<llvm::SmallVector<mlir::Value>> fuOutputReadyTerms(
            fuOutputs.size());
        std::vector<llvm::SmallVector<mlir::Value>> fuOutputOfferedTerms(
            fuOutputs.size());
        mlir::Value idleFu = zero(bodyBuilder, location, idleFuWidth);
        if (layout.fus.size() > 1 && sharedDispatchFu) {
          idleFu = sharedDispatchFu;
        } else if (layout.fus.size() > 1) {
          circt::Backedge idleFuNext =
              backedges.get(bodyBuilder.getIntegerType(idleFuWidth));
          idleFu = createRegister(bodyBuilder, location, idleFuNext,
                                  accessor.getInput("clock"),
                                  accessor.getInput("reset"),
                                  llvm::APInt(idleFuWidth, 0),
                                  "result_idle_fu_cursor_reg",
                                  clockReset.asynchronousReset);
          circt::Backedge passSeenNext = backedges.get(bodyBuilder.getI1Type());
          mlir::Value passSeen = createRegister(
              bodyBuilder, location, passSeenNext, accessor.getInput("clock"),
              accessor.getInput("reset"), llvm::APInt(1, 0),
              "result_idle_fu_pass_seen_reg", clockReset.asynchronousReset);
          llvm::SmallVector<mlir::Value> fuPassStart;
          llvm::SmallVector<mlir::Value> fuEligible;
          for (std::uint32_t fu = 0; fu != layout.fus.size(); ++fu) {
            fuPassStart.push_back(unitOfFu[fu]
                                      ? unitPassStart[*unitOfFu[fu]]
                                      : bitConstant(bodyBuilder, location, false));
            fuEligible.push_back(
                orValues(bodyBuilder, location, contextEligible[fu]));
          }
          mlir::Value currentPassStart =
              indexedValue(bodyBuilder, location, idleFu, fuPassStart);
          mlir::Value currentEligible =
              indexedValue(bodyBuilder, location, idleFu, fuEligible);
          mlir::Value advance = orValues(
              bodyBuilder, location,
              {circt::comb::createOrFoldNot(bodyBuilder, location,
                                            currentEligible),
               andValues(bodyBuilder, location, {currentPassStart, passSeen})});
          passSeenNext.setValue(andValues(
              bodyBuilder, location,
              {circt::comb::createOrFoldNot(bodyBuilder, location, advance),
               orValues(bodyBuilder, location, {passSeen, currentPassStart})}));
          mlir::Value successor = circt::comb::MuxOp::create(
              bodyBuilder, location,
              equals(bodyBuilder, location, idleFu, layout.fus.size() - 1),
              zero(bodyBuilder, location, idleFuWidth),
              circt::comb::AddOp::create(
                  bodyBuilder, location, idleFu,
                  constant(bodyBuilder, location, idleFuWidth, 1), true),
              true);
          idleFuNext.setValue(circt::comb::MuxOp::create(
              bodyBuilder, location, advance, successor, idleFu, true));
        }
        const auto presentedCandidates =
            [&](llvm::ArrayRef<mlir::Value> requests,
                llvm::ArrayRef<mlir::Value> candidates,
                const StatefulSelection &selection) {
              mlir::Value anyRequest =
                  orValues(bodyBuilder, location, requests);
              mlir::Value taken = bitConstant(bodyBuilder, location, false);
              std::vector<mlir::Value> presented;
              presented.reserve(candidates.size());
              for (std::size_t candidate = 0; candidate != candidates.size();
                   ++candidate) {
                mlir::Value idle = andValues(
                    bodyBuilder, location,
                    {candidates[candidate],
                     equals(bodyBuilder, location, idleFu,
                            fuOutputs[candidate].fu),
                     circt::comb::createOrFoldNot(bodyBuilder, location,
                                                  taken)});
                taken = orValues(bodyBuilder, location, {taken, idle});
                presented.push_back(circt::comb::MuxOp::create(
                    bodyBuilder, location, anyRequest,
                    selection.selected[candidate], idle, true));
              }
              return presented;
            };
        for (std::uint32_t outputPort = 0; outputPort != outputEndpoints.size();
             ++outputPort) {
          std::vector<mlir::Value> requests;
          std::vector<mlir::Value> candidates;
          std::vector<std::optional<mlir::Value>> data;
          requests.reserve(fuOutputs.size());
          candidates.reserve(fuOutputs.size());
          data.reserve(fuOutputs.size());
          for (std::size_t candidate = 0; candidate != fuOutputs.size();
               ++candidate) {
            const FuOutputRuntime &output = fuOutputs[candidate];
            candidates.push_back(andValues(
                bodyBuilder, location,
                {resultRoutes[candidate].route,
                 equals(bodyBuilder, location, resultRoutes[candidate].target,
                        outputPort)}));
            requests.push_back(andValues(bodyBuilder, location,
                                         {output.valid, candidates.back()}));
            data.push_back(adaptedFuOutputData[candidate]);
          }
          StatefulSelection selection = makeStatefulSelection(
              bodyBuilder, location, backedges, requests,
              accessor.getInput("clock"), accessor.getInput("reset"),
              "output_" + std::to_string(outputPort) + "_cursor_reg",
              clockReset);
          const std::vector<mlir::Value> presented =
              presentedCandidates(requests, candidates, selection);
          mlir::Value outputData =
              payloadWidth == 0 ? mlir::Value{}
                                : zero(bodyBuilder, location, payloadWidth);
          mlir::Value outputTag = zero(bodyBuilder, location, tagWidth);
          for (std::size_t candidate = 0; candidate != fuOutputs.size();
               ++candidate) {
            if (payloadWidth != 0)
              outputData = circt::comb::MuxOp::create(
                  bodyBuilder, location, presented[candidate],
                  *data[candidate], outputData, true);
            outputTag = circt::comb::MuxOp::create(
                bodyBuilder, location, presented[candidate],
                resultRoutes[candidate].tag, outputTag, true);
            mlir::Value portReady = accessor.getInput(
                outputEndpoints[outputPort]->ready.getName());
            fuOutputReadyTerms[candidate].push_back(andValues(
                bodyBuilder, location, {presented[candidate], portReady}));
            fuOutputOfferedTerms[candidate].push_back(presented[candidate]);
          }
          if (payloadWidth != 0)
            accessor.setOutput(outputEndpoints[outputPort]->data->getName(),
                               outputData);
          accessor.setOutput(outputEndpoints[outputPort]->tag->getName(),
                             outputTag);
          accessor.setOutput(
              outputEndpoints[outputPort]->valid.getName(),
              orValues(bodyBuilder, location, selection.selected));
          advanceStatefulSelection(bodyBuilder, location, selection,
                                   selection.selected);
        }

        for (std::size_t candidate = 0; candidate != fuOutputs.size();
             ++candidate)
          fuOutputReadyTerms[candidate].push_back(
              resultRoutes[candidate].discard);

        for (std::uint32_t fifo = 0; fifo != layout.registerFifoCount; ++fifo) {
          std::vector<mlir::Value> requests;
          std::vector<mlir::Value> candidates;
          requests.reserve(fuOutputs.size());
          candidates.reserve(fuOutputs.size());
          for (std::size_t candidate = 0; candidate != fuOutputs.size();
               ++candidate) {
            candidates.push_back(andValues(
                bodyBuilder, location,
                {resultRoutes[candidate].route,
                 equals(bodyBuilder, location, resultRoutes[candidate].target,
                        layout.outputPortCount + fifo)}));
            requests.push_back(
                andValues(bodyBuilder, location,
                          {fuOutputs[candidate].valid, candidates.back()}));
          }
          StatefulSelection selection = makeStatefulSelection(
              bodyBuilder, location, backedges, requests,
              accessor.getInput("clock"), accessor.getInput("reset"),
              "register_fifo_" + std::to_string(fifo) + "_write_cursor_reg",
              clockReset);
          const std::vector<mlir::Value> presented =
              presentedCandidates(requests, candidates, selection);
          mlir::Value writePayload =
              zero(bodyBuilder, location, fifoPayloadWidth);
          for (std::size_t candidate = 0; candidate != fuOutputs.size();
               ++candidate) {
            mlir::Value packed =
                packToken(bodyBuilder, location,
                          payloadWidth == 0 ? std::optional<mlir::Value>{}
                                            : adaptedFuOutputData[candidate],
                          resultRoutes[candidate].tag);
            writePayload = circt::comb::MuxOp::create(
                bodyBuilder, location, selection.selected[candidate], packed,
                writePayload, true);
          }
          std::map<std::string, mlir::Value> instanceInputs;
          instanceInputs.emplace("clock", accessor.getInput("clock"));
          instanceInputs.emplace("reset", accessor.getInput("reset"));
          instanceInputs.emplace(queuePort("enqueue", 0, "_data"),
                                 writePayload);
          circt::Backedge writeCommit = backedges.get(bodyBuilder.getI1Type());
          instanceInputs.emplace(queuePort("enqueue", 0, "_commit"),
                                 writeCommit);
          instanceInputs.emplace(queuePort("dequeue", 0, "_ready"),
                                 fifoReadReady[fifo]);
          auto instance = instantiateModule(
              bodyBuilder, location, fifoPools[fifo].module,
              "register_fifo_" + std::to_string(fifo), instanceInputs);
          if (!instance) {
            materializationError = llvm::toString(instance.takeError());
            backedges.abandon();
            return;
          }
          mlir::Value enqueueReady =
              instance->at(queuePort("enqueue", 0, "_ready"));
          fifoHeadPayload[fifo].setValue(
              instance->at(queuePort("dequeue", 0, "_data")));
          fifoHeadValid[fifo].setValue(
              instance->at(queuePort("dequeue", 0, "_valid")));
          std::vector<mlir::Value> fired;
          fired.reserve(fuOutputs.size());
          for (std::size_t candidate = 0; candidate != fuOutputs.size();
               ++candidate) {
            fired.push_back(
                andValues(bodyBuilder, location,
                          {selection.selected[candidate], enqueueReady}));
            fuOutputReadyTerms[candidate].push_back(andValues(
                bodyBuilder, location, {presented[candidate], enqueueReady}));
            fuOutputOfferedTerms[candidate].push_back(presented[candidate]);
          }
          writeCommit.setValue(orValues(bodyBuilder, location, fired));
          advanceStatefulSelection(bodyBuilder, location, selection,
                                   selection.selected);
        }

        for (std::size_t candidate = 0; candidate != fuOutputs.size();
             ++candidate) {
          fuOutputs[candidate].ready.setValue(
              orValues(bodyBuilder, location, fuOutputReadyTerms[candidate]));
          fuOutputs[candidate].offered.setValue(orValues(
              bodyBuilder, location, fuOutputOfferedTerms[candidate]));
        }

        metrics.resultArbitrationOperations =
            blockOperationCount(bodyBuilder) - metrics.beginOperations -
            metrics.decodeAndPoolOperations - metrics.ingressOperations -
            metrics.dispatchOperations - metrics.childOperations -
            metrics.resultProjectionOperations;
        for (const EndpointPlan *input : inputEndpoints)
          accessor.setOutput(input->ready.getName(),
                             inputReady[input->localOrdinal]);
      });
  if (materializationError)
    return invalid(*materializationError);
  if (invocationDiagnosticEnabled(DiagnosticVerbosity::Summary)) {
    const std::uint64_t actualOperations =
        metrics.decodeAndPoolOperations + metrics.ingressOperations +
        metrics.dispatchOperations + metrics.childOperations +
        metrics.resultProjectionOperations +
        metrics.resultArbitrationOperations;
    emitInvocationDiagnostic(
        DiagnosticVerbosity::Summary,
        InvocationDiagnosticStage::HardwareConfiguration,
        InvocationDiagnosticEvent::Statistics, [&] {
          return llvm::json::Value(llvm::json::Object{
              {"statistics_kind", "rtl_temporal_pe_materialization_shape"},
              {"materialization_key", materializationKey.str()},
              {"pe_ordinal", pe.id()},
              {"context_count", layout.contextCount},
              {"fu_count", layout.fus.size()},
              {"fu_input_count", fuInputCount},
              {"squared_fu_input_count", squaredFuInputCount},
              {"fu_output_count", fuOutputCount},
              {"logical_queue_count", queues.size()},
              {"allocation_unit_count", units.size()},
              {"input_port_count", layout.inputPortCount},
              {"output_port_count", layout.outputPortCount},
              {"register_fifo_count", layout.registerFifoCount},
              {"operand_depth", operandDepth},
              {"configuration_port_count", configuration->empty() ? 0 : 1},
              {"configuration_bundle_member_count",
               configuration->words.size()},
              {"predicted_repeated_tuple_input_predicates",
               std::uint64_t(layout.contextCount) * squaredFuInputCount *
                   layout.inputPortCount},
              {"predicted_canonical_tuple_input_predicates",
               std::uint64_t(layout.contextCount) * fuInputCount *
                   layout.inputPortCount},
              {"predicted_queue_port_predicates",
               std::uint64_t(queues.size()) * layout.inputPortCount},
              {"predicted_fifo_read_candidates",
               std::uint64_t(layout.registerFifoCount) * layout.contextCount *
                   fuInputCount * 2},
              {"predicted_fifo_selector_matches",
               std::uint64_t(layout.registerFifoCount) * layout.contextCount *
                   layout.maximumFuInputCount},
              {"predicted_fifo_candidate_common_terms",
               std::uint64_t(layout.registerFifoCount) * layout.contextCount *
                   fuInputCount},
              {"predicted_child_input_paths",
               std::uint64_t(layout.contextCount) * fuInputCount *
                   (layout.inputPortCount + layout.registerFifoCount)},
              {"predicted_result_context_projections",
               fuOutputCount * layout.contextCount},
              {"predicted_output_arbitration_candidates",
               fuOutputCount *
                   (layout.outputPortCount + layout.registerFifoCount)},
              {"actual_decode_and_pool_operations",
               metrics.decodeAndPoolOperations},
              {"actual_ingress_operations", metrics.ingressOperations},
              {"actual_dispatch_operations", metrics.dispatchOperations},
              {"actual_child_operations", metrics.childOperations},
              {"actual_result_projection_operations",
               metrics.resultProjectionOperations},
              {"actual_result_arbitration_operations",
               metrics.resultArbitrationOperations},
              {"actual_module_body_operations", actualOperations}});
        });
  }
  return PeModule{pe, module, std::move(*endpoints), std::move(*configuration)};
}

} // namespace loom::hardware::rtl::hierarchy

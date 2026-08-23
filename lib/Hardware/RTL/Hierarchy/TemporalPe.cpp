#include "Components.h"

#include "Fabric/IR/TemporalPeResourceContract.h"
#include "Fabric/Identity/FabricTemporalPeConfiguration.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::hardware::rtl::hierarchy {
namespace {

unsigned indexWidth(std::uint64_t count) {
  return std::max(1U, llvm::Log2_64_Ceil(std::max<std::uint64_t>(count, 1)));
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

std::vector<mlir::Value>
roundRobinSelection(mlir::OpBuilder &builder, mlir::Location location,
                    llvm::ArrayRef<mlir::Value> requests, mlir::Value cursor) {
  std::vector<mlir::Value> selected(requests.size(),
                                    bitConstant(builder, location, false));
  if (requests.empty())
    return selected;
  for (std::size_t start = 0; start != requests.size(); ++start) {
    mlir::Value cursorIs = equals(builder, location, cursor, start);
    mlir::Value reserved = bitConstant(builder, location, false);
    for (std::size_t offset = 0; offset != requests.size(); ++offset) {
      const std::size_t requester = (start + offset) % requests.size();
      mlir::Value grant = andValues(
          builder, location,
          {cursorIs, requests[requester],
           circt::comb::createOrFoldNot(builder, location, reserved)});
      selected[requester] = circt::comb::OrOp::create(
          builder, location, selected[requester], grant);
      reserved = circt::comb::OrOp::create(builder, location, reserved,
                                           requests[requester]);
    }
  }
  return selected;
}

mlir::Value nextCursor(mlir::OpBuilder &builder, mlir::Location location,
                       mlir::Value current, llvm::ArrayRef<mlir::Value> fired) {
  const unsigned width =
      mlir::cast<mlir::IntegerType>(current.getType()).getWidth();
  mlir::Value next = current;
  for (std::size_t requester = 0; requester != fired.size(); ++requester)
    next = circt::comb::MuxOp::create(
        builder, location, fired[requester],
        constant(builder, location, width, (requester + 1) % fired.size()),
        next, true);
  return next;
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

llvm::Expected<TokenPoolModule>
buildTokenPoolModule(mlir::OpBuilder &builder, mlir::Location location,
                     llvm::StringRef name, std::uint32_t queueCount,
                     std::uint32_t depth, unsigned payloadWidth,
                     bool singlePort, bool fullReplacement, bool exposeNearFull,
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
        {{builder.getStringAttr(queuePort("enqueue", queue, "_valid")),
          builder.getI1Type(), circt::hw::ModulePort::Direction::Input}});
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
            mlir::Value matches = occupied;
            if (queueCount != 1)
              matches =
                  andValues(bodyBuilder, location,
                            {occupied, equals(bodyBuilder, location,
                                              queueIds.current[slot], queue)});
            headSlots[queue][slot] = andValues(
                bodyBuilder, location,
                {matches,
                 circt::comb::createOrFoldNot(bodyBuilder, location, earlier)});
            earlier = circt::comb::OrOp::create(bodyBuilder, location, earlier,
                                                matches);
            if (payloadWidth != 0)
              headData = circt::comb::MuxOp::create(
                  bodyBuilder, location, headSlots[queue][slot],
                  payload.current[slot], headData, true);
          }
          headValid[queue] = earlier;
          if (payloadWidth != 0)
            accessor.setOutput(queuePort("dequeue", queue, "_data"), headData);
          accessor.setOutput(queuePort("dequeue", queue, "_valid"), earlier);
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

        std::vector<mlir::Value> enqueueRequests;
        enqueueRequests.reserve(queueCount);
        for (std::uint32_t queue = 0; queue != queueCount; ++queue)
          enqueueRequests.push_back(
              accessor.getInput(queuePort("enqueue", queue, "_valid")));
        const unsigned cursorWidth = indexWidth(queueCount);
        circt::Backedge cursorNext =
            backedges.get(bodyBuilder.getIntegerType(cursorWidth));
        mlir::Value cursor = createRegister(
            bodyBuilder, location, cursorNext, accessor.getInput("clock"),
            accessor.getInput("reset"), llvm::APInt(cursorWidth, 0),
            "enqueue_cursor_reg", clockReset.asynchronousReset);
        std::vector<mlir::Value> enqueueSelected =
            roundRobinSelection(bodyBuilder, location, enqueueRequests, cursor);
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
        std::vector<mlir::Value> enqueueFired(queueCount);
        for (std::uint32_t queue = 0; queue != queueCount; ++queue) {
          mlir::Value granted = andValues(bodyBuilder, location,
                                          {enqueueSelected[queue], canEnqueue});
          accessor.setOutput(queuePort("enqueue", queue, "_ready"), granted);
          enqueueFired[queue] = andValues(
              bodyBuilder, location,
              {granted,
               accessor.getInput(queuePort("enqueue", queue, "_commit"))});
        }
        mlir::Value enqueue = orValues(bodyBuilder, location, enqueueFired);
        cursorNext.setValue(
            nextCursor(bodyBuilder, location, cursor, enqueueFired));

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
        for (std::uint32_t queue = 0; queue != queueCount; ++queue)
          for (std::uint32_t slot = 0; slot != depth; ++slot)
            removedSlot[slot] = circt::comb::OrOp::create(
                bodyBuilder, location, removedSlot[slot],
                andValues(bodyBuilder, location,
                          {dequeueSelected[queue], headSlots[queue][slot]}));

        mlir::Value selectedPayload =
            payloadWidth == 0 ? mlir::Value{}
                              : zero(bodyBuilder, location, payloadWidth);
        mlir::Value selectedQueue = zero(bodyBuilder, location, queueWidth);
        for (std::uint32_t queue = 0; queue != queueCount; ++queue) {
          if (payloadWidth != 0)
            selectedPayload = circt::comb::MuxOp::create(
                bodyBuilder, location, enqueueSelected[queue],
                accessor.getInput(queuePort("enqueue", queue, "_data")),
                selectedPayload, true);
          selectedQueue = circt::comb::MuxOp::create(
              bodyBuilder, location, enqueueSelected[queue],
              constant(bodyBuilder, location, queueWidth, queue), selectedQueue,
              true);
        }

        const auto updateBank = [&](Bank &bank, mlir::Value appended) {
          if (bank.width == 0)
            return;
          mlir::Value shift = bitConstant(bodyBuilder, location, false);
          for (std::uint32_t slot = 0; slot != depth; ++slot) {
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

void addCommonInputs(circt::hw::HWModulePortAccessor &accessor,
                     const ConfigurationTransportLayout &transportLayout,
                     std::map<std::string, mlir::Value> &inputs) {
  inputs.emplace("clock", accessor.getInput("clock"));
  inputs.emplace("reset", accessor.getInput("reset"));
  for (std::size_t ordinal = 0; ordinal != transportLayout.units.size();
       ++ordinal)
    inputs.emplace(configurationPortName(ordinal),
                   accessor.getInput(configurationPortName(ordinal)));
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
  circt::Backedge enqueueAdmission;
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
};

struct ResultRouteSignals final {
  mlir::Value active;
  mlir::Value route;
  mlir::Value discard;
  mlir::Value target;
  mlir::Value tag;
};

struct StatefulSelection final {
  circt::Backedge next;
  mlir::Value cursor;
  std::vector<mlir::Value> selected;
};

StatefulSelection makeStatefulSelection(mlir::OpBuilder &builder,
                                        mlir::Location location,
                                        circt::BackedgeBuilder &backedges,
                                        llvm::ArrayRef<mlir::Value> requests,
                                        mlir::Value clock, mlir::Value reset,
                                        llvm::StringRef name,
                                        const ClockResetPlan &clockReset) {
  const unsigned width = indexWidth(requests.size());
  circt::Backedge next = backedges.get(builder.getIntegerType(width));
  mlir::Value cursor =
      createRegister(builder, location, next, clock, reset,
                     llvm::APInt(width, 0), name, clockReset.asynchronousReset);
  return StatefulSelection{
      std::move(next), cursor,
      roundRobinSelection(builder, location, requests, cursor)};
}

mlir::Value packToken(mlir::OpBuilder &builder, mlir::Location location,
                      std::optional<mlir::Value> data, mlir::Value tag) {
  if (!data)
    return tag;
  return circt::comb::ConcatOp::create(builder, location,
                                       llvm::ArrayRef<mlir::Value>{tag, *data});
}

} // namespace

llvm::Expected<PeModule>
buildTemporalPeModule(mlir::OpBuilder &builder, mlir::Location location,
                      fabric::SpatialCoreOccurrenceRef spatialCore,
                      const fabric::FabricArtifactView &fabric,
                      const ConfigurationABI &configurationAbi,
                      const ConfigurationTransportLayout &transportLayout,
                      llvm::ArrayRef<FuModule> fuModules,
                      const ClockResetPlan &clockReset,
                      fabric::FabricPeOccurrenceRef pe) {
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
  for (std::uint32_t unit = 0; unit != units.size(); ++unit) {
    auto pool =
        buildTokenPoolModule(builder, location,
                             "loom_temporal_pe_" + std::to_string(pe.id()) +
                                 "_operand_pool_" + std::to_string(unit),
                             units[unit].queues.size(), operandDepth,
                             payloadWidth, false, false, true, clockReset);
    if (!pool)
      return pool.takeError();
    units[unit].pool = std::move(*pool);
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
  for (std::uint32_t fifo = 0; fifo != layout.registerFifoCount; ++fifo) {
    auto pool =
        buildTokenPoolModule(builder, location,
                             "loom_temporal_pe_" + std::to_string(pe.id()) +
                                 "_register_fifo_" + std::to_string(fifo),
                             1, fifoDepth, payloadWidth + tagWidth,
                             fifoPorts == 1, fifoPorts == 2, false, clockReset);
    if (!pool)
      return pool.takeError();
    fifoPools.push_back(std::move(*pool));
  }

  llvm::SmallVector<circt::hw::PortInfo, 32> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 32> outputs;
  appendClockResetAndConfigurationPorts(builder, configurationAbi,
                                        transportLayout, inputs);
  for (const EndpointPlan &endpoint : *endpoints)
    appendEndpointPorts(inputs, outputs, endpoint);

  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_temporal_pe_" + std::to_string(pe.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        mlir::Value field =
            decodeFieldSignal(bodyBuilder, location, accessor, *decoder);
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
            llvm::SmallVector<mlir::Value> enqueueValidTerms;
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
              enqueueValidTerms.push_back(andValues(
                  bodyBuilder, location,
                  {routeMatches,
                   accessor.getInput(inputEndpoints[port]->valid.getName())}));
              if (payloadWidth != 0)
                enqueueData = circt::comb::MuxOp::create(
                    bodyBuilder, location, routeMatches,
                    accessor.getInput(inputEndpoints[port]->data->getName()),
                    enqueueData, true);
            }
            if (payloadWidth != 0)
              instanceInputs.emplace(queuePort("enqueue", local, "_data"),
                                     enqueueData);
            queueRuntime[queueOrdinal].enqueueAdmission =
                backedges.get(bodyBuilder.getI1Type());
            instanceInputs.emplace(
                queuePort("enqueue", local, "_valid"),
                andValues(bodyBuilder, location,
                          {orValues(bodyBuilder, location, enqueueValidTerms),
                           queueRuntime[queueOrdinal].enqueueAdmission}));
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

        // Prefer an ingress transaction that fills a missing role and makes a
        // context/FU tuple complete. Priority belongs to the whole physical
        // ingress transaction: every QueueKey matched by that ingress remains
        // one atomic fanout group.
        std::vector<mlir::Value> queuePairReady(queues.size());
        std::vector<mlir::Value> queueNearFullComplement(queues.size());
        std::vector<std::vector<mlir::Value>> queueCompletingArrivals(
            queues.size(), std::vector<mlir::Value>(layout.inputPortCount));
        for (std::uint32_t queueOrdinal = 0; queueOrdinal != queues.size();
             ++queueOrdinal) {
          const LogicalQueuePlan &queue = queues[queueOrdinal];
          llvm::SmallVector<mlir::Value> requiredInputs;
          llvm::SmallVector<mlir::Value> missingRoleArrivals;
          llvm::SmallVector<mlir::Value> nearFullOccupiedRoles;
          for (std::uint32_t input = 0;
               input != layout.fus[queue.fu].inputCount; ++input) {
            const std::uint32_t requiredQueue =
                queueOf[queue.context][queue.fu][input];
            const SelectorSignals &requiredSelector =
                rows[queue.context].operands[input];
            llvm::SmallVector<mlir::Value> externalTargets;
            for (std::uint32_t port = 0; port != inputEndpoints.size(); ++port)
              externalTargets.push_back(
                  equals(bodyBuilder, location, requiredSelector.target, port));
            const mlir::Value roleUsesOperandQueue =
                andValues(bodyBuilder, location,
                          {rows[queue.context].active,
                           equals(bodyBuilder, location,
                                  rows[queue.context].selectedFu, queue.fu),
                           requiredSelector.route,
                           orValues(bodyBuilder, location, externalTargets)});
            llvm::SmallVector<mlir::Value> arrivals;
            for (std::uint32_t port = 0; port != inputEndpoints.size();
                 ++port) {
              const mlir::Value arrival = andValues(
                  bodyBuilder, location,
                  {queueInputMatches[requiredQueue][port],
                   accessor.getInput(inputEndpoints[port]->valid.getName())});
              arrivals.push_back(arrival);
              queueCompletingArrivals[requiredQueue][port] =
                  andValues(bodyBuilder, location,
                            {arrival, circt::comb::createOrFoldNot(
                                          bodyBuilder, location,
                                          queueRuntime[requiredQueue].valid)});
              missingRoleArrivals.push_back(
                  queueCompletingArrivals[requiredQueue][port]);
            }
            requiredInputs.push_back(
                orValues(bodyBuilder, location,
                         {circt::comb::createOrFoldNot(bodyBuilder, location,
                                                       roleUsesOperandQueue),
                          queueRuntime[requiredQueue].valid,
                          orValues(bodyBuilder, location, arrivals)}));
            nearFullOccupiedRoles.push_back(andValues(
                bodyBuilder, location,
                {roleUsesOperandQueue, queueRuntime[requiredQueue].valid,
                 units[queues[requiredQueue].unit].nearFull}));
          }
          const mlir::Value fillsMissingRole =
              orValues(bodyBuilder, location, missingRoleArrivals);
          queuePairReady[queueOrdinal] =
              andValues(bodyBuilder, location,
                        {andValues(bodyBuilder, location, requiredInputs),
                         fillsMissingRole});
          queueNearFullComplement[queueOrdinal] = andValues(
              bodyBuilder, location,
              {fillsMissingRole,
               orValues(bodyBuilder, location, nearFullOccupiedRoles),
               circt::comb::createOrFoldNot(bodyBuilder, location,
                                            queuePairReady[queueOrdinal])});
        }
        std::vector<std::vector<mlir::Value>> queuePortPreferred(
            queues.size(), std::vector<mlir::Value>(layout.inputPortCount));
        for (std::uint32_t queueOrdinal = 0; queueOrdinal != queues.size();
             ++queueOrdinal) {
          const LogicalQueuePlan &queue = queues[queueOrdinal];
          for (std::uint32_t port = 0; port != layout.inputPortCount; ++port) {
            llvm::SmallVector<mlir::Value> completingRoles;
            for (std::uint32_t input = 0;
                 input != layout.fus[queue.fu].inputCount; ++input)
              completingRoles.push_back(
                  queueCompletingArrivals[queueOf[queue.context][queue.fu]
                                                 [input]][port]);
            const mlir::Value completesTuple =
                andValues(bodyBuilder, location,
                          {queuePairReady[queueOrdinal],
                           orValues(bodyBuilder, location, completingRoles)});
            const mlir::Value nearFullComplement =
                andValues(bodyBuilder, location,
                          {queueNearFullComplement[queueOrdinal],
                           orValues(bodyBuilder, location, completingRoles)});
            queuePortPreferred[queueOrdinal][port] = orValues(
                bodyBuilder, location,
                {completesTuple,
                 andValues(
                     bodyBuilder, location,
                     {circt::comb::createOrFoldNot(
                          bodyBuilder, location, queuePairReady[queueOrdinal]),
                      orValues(
                          bodyBuilder, location,
                          {nearFullComplement,
                           circt::comb::createOrFoldNot(
                               bodyBuilder, location,
                               queueNearFullComplement[queueOrdinal])})})});
          }
        }
        std::vector<mlir::Value> portPriorityAdmitted(layout.inputPortCount);
        for (std::uint32_t port = 0; port != layout.inputPortCount; ++port) {
          llvm::SmallVector<mlir::Value> blocked;
          for (std::uint32_t queue = 0; queue != queues.size(); ++queue)
            blocked.push_back(andValues(
                bodyBuilder, location,
                {queueInputMatches[queue][port],
                 circt::comb::createOrFoldNot(
                     bodyBuilder, location, queuePortPreferred[queue][port])}));
          portPriorityAdmitted[port] = circt::comb::createOrFoldNot(
              bodyBuilder, location, orValues(bodyBuilder, location, blocked));
        }
        std::vector<std::vector<mlir::Value>> admissionMatches(
            queues.size(), std::vector<mlir::Value>(layout.inputPortCount));
        std::vector<llvm::SmallVector<mlir::Value>> admissionMatchTerms(
            layout.inputPortCount);
        std::vector<llvm::SmallVector<mlir::Value>> admissionBlockedTerms(
            layout.inputPortCount);
        for (std::uint32_t queueOrdinal = 0; queueOrdinal != queues.size();
             ++queueOrdinal)
          for (std::uint32_t port = 0; port != inputEndpoints.size(); ++port) {
            const mlir::Value allowed =
                andValues(bodyBuilder, location,
                          {queueInputMatches[queueOrdinal][port],
                           portPriorityAdmitted[port]});
            admissionMatches[queueOrdinal][port] = allowed;
            admissionMatchTerms[port].push_back(allowed);
            admissionBlockedTerms[port].push_back(andValues(
                bodyBuilder, location,
                {allowed, circt::comb::createOrFoldNot(
                              bodyBuilder, location,
                              queueRuntime[queueOrdinal].enqueueReady)}));
          }
        std::vector<mlir::Value> inputReady(layout.inputPortCount);
        for (std::uint32_t port = 0; port != layout.inputPortCount; ++port)
          inputReady[port] = andValues(
              bodyBuilder, location,
              {portPriorityAdmitted[port],
               orValues(
                   bodyBuilder, location,
                   {orValues(bodyBuilder, location, admissionMatchTerms[port]),
                    orValues(bodyBuilder, location, inputMatchTerms[port])}),
               circt::comb::createOrFoldNot(
                   bodyBuilder, location,
                   orValues(bodyBuilder, location,
                            admissionBlockedTerms[port]))});
        for (std::uint32_t queue = 0; queue != queues.size(); ++queue)
          queueRuntime[queue].enqueueAdmission.setValue(
              orValues(bodyBuilder, location, admissionMatches[queue]));
        for (std::uint32_t queue = 0; queue != queues.size(); ++queue) {
          llvm::SmallVector<mlir::Value> commitTerms;
          for (std::uint32_t port = 0; port != inputEndpoints.size(); ++port)
            commitTerms.push_back(andValues(
                bodyBuilder, location,
                {admissionMatches[queue][port], inputReady[port],
                 accessor.getInput(inputEndpoints[port]->valid.getName())}));
          queueRuntime[queue].enqueueCommit.setValue(
              orValues(bodyBuilder, location, commitTerms));
        }

        const unsigned fifoPayloadWidth = payloadWidth + tagWidth;
        std::vector<circt::Backedge> fifoHeadPayload;
        std::vector<circt::Backedge> fifoHeadValid;
        fifoHeadPayload.reserve(layout.registerFifoCount);
        fifoHeadValid.reserve(layout.registerFifoCount);
        for (std::uint32_t fifo = 0; fifo != layout.registerFifoCount; ++fifo) {
          fifoHeadPayload.push_back(
              backedges.get(bodyBuilder.getIntegerType(fifoPayloadWidth)));
          fifoHeadValid.push_back(backedges.get(bodyBuilder.getI1Type()));
        }
        const auto fifoData = [&](std::uint32_t fifo) -> mlir::Value {
          if (payloadWidth == 0)
            return {};
          return extract(bodyBuilder, location, fifoHeadPayload[fifo], 0,
                         payloadWidth);
        };
        const auto fifoTag = [&](std::uint32_t fifo) {
          return extract(bodyBuilder, location, fifoHeadPayload[fifo],
                         payloadWidth, tagWidth);
        };

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
        for (std::uint32_t unit = 0; unit != peResources->dispatchUnitCount();
             ++unit) {
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
          for (auto [request, candidateOrdinal] : llvm::enumerate(candidates)) {
            const auto &candidate =
                peResources->dispatchCandidates()[candidateOrdinal];
            contextSelected[candidate.fuOccurrence][candidate.context.ordinal] =
                selection.selected[request];
          }
          selection.next.setValue(nextCursor(
              bodyBuilder, location, selection.cursor, selection.selected));
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
            llvm::SmallVector<mlir::Value> portTargets;
            for (std::uint32_t port = 0; port != layout.inputPortCount; ++port)
              portTargets.push_back(
                  equals(bodyBuilder, location, selector.target, port));
            requests.push_back(andValues(
                bodyBuilder, location,
                {contextSelected[queue.fu][queue.context], selector.route,
                 orValues(bodyBuilder, location, portTargets),
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
          selection.next.setValue(nextCursor(
              bodyBuilder, location, selection.cursor, selection.selected));
        }

        struct FifoReadCandidate final {
          std::uint32_t fifo = 0;
          std::uint32_t fu = 0;
          std::uint32_t context = 0;
          std::uint32_t input = 0;
          bool discard = false;
          mlir::Value request;
          mlir::Value granted;
        };
        std::vector<FifoReadCandidate> fifoReadCandidates;
        for (std::uint32_t fifo = 0; fifo != layout.registerFifoCount; ++fifo) {
          std::vector<std::size_t> candidates;
          std::vector<mlir::Value> requests;
          for (std::uint32_t fu = 0; fu != layout.fus.size(); ++fu)
            for (std::uint32_t context = 0; context != layout.contextCount;
                 ++context)
              for (std::uint32_t input = 0; input != layout.fus[fu].inputCount;
                   ++input) {
                const SelectorSignals &selector = rows[context].operands[input];
                mlir::Value targetMatches =
                    equals(bodyBuilder, location, selector.target,
                           layout.inputPortCount + fifo);
                mlir::Value tagMatches = circt::comb::ICmpOp::create(
                    bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                    selector.tag, fifoTag(fifo), true);
                for (bool discard : {false, true}) {
                  mlir::Value kind =
                      discard ? selector.discard : selector.route;
                  mlir::Value request = andValues(
                      bodyBuilder, location,
                      {contextSelected[fu][context], kind, targetMatches,
                       fifoHeadValid[fifo], tagMatches});
                  candidates.push_back(fifoReadCandidates.size());
                  requests.push_back(request);
                  fifoReadCandidates.push_back(
                      {fifo, fu, context, input, discard, request,
                       bitConstant(bodyBuilder, location, false)});
                }
              }
          StatefulSelection selection = makeStatefulSelection(
              bodyBuilder, location, backedges, requests,
              accessor.getInput("clock"), accessor.getInput("reset"),
              "register_fifo_" + std::to_string(fifo) + "_read_cursor_reg",
              clockReset);
          for (std::size_t ordinal = 0; ordinal != candidates.size(); ++ordinal)
            fifoReadCandidates[candidates[ordinal]].granted =
                selection.selected[ordinal];
          selection.next.setValue(nextCursor(
              bodyBuilder, location, selection.cursor, selection.selected));
        }

        std::vector<std::vector<mlir::Value>> fuInputReady(layout.fus.size());
        std::vector<FuOutputRuntime> fuOutputs;
        for (std::uint32_t fu = 0; fu != layout.fus.size(); ++fu) {
          const FuModule &child = *children[fu];
          std::map<std::string, mlir::Value> instanceInputs;
          addCommonInputs(accessor, transportLayout, instanceInputs);
          fuInputReady[fu].resize(layout.fus[fu].inputCount);
          std::vector<circt::Backedge> outputReady(layout.fus[fu].outputCount);
          for (const EndpointPlan &endpoint : child.endpoints) {
            if (endpoint.direction == fabric::FabricPortDirection::Output) {
              outputReady[endpoint.localOrdinal] =
                  backedges.get(bodyBuilder.getI1Type());
              instanceInputs.emplace(endpoint.ready.getName().str(),
                                     outputReady[endpoint.localOrdinal]);
              continue;
            }
            mlir::Value data = endpoint.data
                                   ? zero(bodyBuilder, location, payloadWidth)
                                   : mlir::Value{};
            llvm::SmallVector<mlir::Value> validTerms;
            mlir::Value contextValue =
                zero(bodyBuilder, location, contextWidth);
            for (std::uint32_t context = 0; context != layout.contextCount;
                 ++context) {
              const SelectorSignals &selector =
                  rows[context].operands[endpoint.localOrdinal];
              const std::uint32_t queue =
                  queueOf[context][fu][endpoint.localOrdinal];
              llvm::SmallVector<mlir::Value> portTargets;
              for (std::uint32_t port = 0; port != layout.inputPortCount;
                   ++port)
                portTargets.push_back(
                    equals(bodyBuilder, location, selector.target, port));
              mlir::Value queuePath =
                  andValues(bodyBuilder, location,
                            {queueGranted[queue],
                             orValues(bodyBuilder, location, portTargets),
                             queueRuntime[queue].valid});
              mlir::Value sourceValid = queuePath;
              mlir::Value sourceData =
                  payloadWidth == 0 ? mlir::Value{} : *queueRuntime[queue].data;
              for (const FifoReadCandidate &candidate : fifoReadCandidates)
                if (!candidate.discard && candidate.fu == fu &&
                    candidate.context == context &&
                    candidate.input == endpoint.localOrdinal) {
                  sourceValid = orValues(bodyBuilder, location,
                                         {sourceValid, candidate.granted});
                  if (payloadWidth != 0)
                    sourceData = circt::comb::MuxOp::create(
                        bodyBuilder, location, candidate.granted,
                        fifoData(candidate.fifo), sourceData, true);
                }
              mlir::Value valid = andValues(
                  bodyBuilder, location,
                  {contextSelected[fu][context], selector.route, sourceValid});
              validTerms.push_back(valid);
              if (payloadWidth != 0)
                data = circt::comb::MuxOp::create(bodyBuilder, location, valid,
                                                  sourceData, data, true);
              contextValue = circt::comb::MuxOp::create(
                  bodyBuilder, location, contextSelected[fu][context],
                  constant(bodyBuilder, location, contextWidth, context),
                  contextValue, true);
            }
            ForwardTransportSignals source{
                orValues(bodyBuilder, location, validTerms),
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
            instanceInputs.emplace(
                "input_" + std::to_string(endpoint.localOrdinal) + "_context",
                contextValue);
          }

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
            fuOutputs.push_back(std::move(output));
          }
        }

        for (std::uint32_t queue = 0; queue != queues.size(); ++queue) {
          const LogicalQueuePlan &plan = queues[queue];
          queueRuntime[queue].dequeueReady.setValue(andValues(
              bodyBuilder, location,
              {queueGranted[queue], fuInputReady[plan.fu][plan.input]}));
        }

        std::vector<llvm::SmallVector<mlir::Value>> fifoReadReadyTerms(
            layout.registerFifoCount);
        for (const FifoReadCandidate &candidate : fifoReadCandidates)
          fifoReadReadyTerms[candidate.fifo].push_back(
              candidate.discard
                  ? candidate.granted
                  : andValues(bodyBuilder, location,
                              {candidate.granted,
                               fuInputReady[candidate.fu][candidate.input]}));

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

        std::vector<llvm::SmallVector<mlir::Value>> fuOutputReadyTerms(
            fuOutputs.size());
        for (std::uint32_t outputPort = 0; outputPort != outputEndpoints.size();
             ++outputPort) {
          std::vector<mlir::Value> requests;
          std::vector<std::optional<mlir::Value>> data;
          requests.reserve(fuOutputs.size());
          data.reserve(fuOutputs.size());
          for (std::size_t candidate = 0; candidate != fuOutputs.size();
               ++candidate) {
            const FuOutputRuntime &output = fuOutputs[candidate];
            requests.push_back(andValues(
                bodyBuilder, location,
                {output.valid, resultRoutes[candidate].route,
                 equals(bodyBuilder, location, resultRoutes[candidate].target,
                        outputPort)}));
            ForwardTransportSignals source{output.valid, output.data,
                                           std::nullopt};
            auto adapted = adaptForwardTransportSignals(
                bodyBuilder, location, output.endpoint->dataPath,
                ::fabric::DataPathType{::fabric::DataPathKind::Bits,
                                       payloadWidth, 0},
                source);
            if (!adapted) {
              materializationError = llvm::toString(adapted.takeError());
              backedges.abandon();
              return;
            }
            data.push_back(adapted->payload);
          }
          StatefulSelection selection = makeStatefulSelection(
              bodyBuilder, location, backedges, requests,
              accessor.getInput("clock"), accessor.getInput("reset"),
              "output_" + std::to_string(outputPort) + "_cursor_reg",
              clockReset);
          mlir::Value outputData =
              payloadWidth == 0 ? mlir::Value{}
                                : zero(bodyBuilder, location, payloadWidth);
          mlir::Value outputTag = zero(bodyBuilder, location, tagWidth);
          std::vector<mlir::Value> fired;
          fired.reserve(fuOutputs.size());
          for (std::size_t candidate = 0; candidate != fuOutputs.size();
               ++candidate) {
            if (payloadWidth != 0)
              outputData = circt::comb::MuxOp::create(
                  bodyBuilder, location, selection.selected[candidate],
                  *data[candidate], outputData, true);
            outputTag = circt::comb::MuxOp::create(
                bodyBuilder, location, selection.selected[candidate],
                resultRoutes[candidate].tag, outputTag, true);
            mlir::Value fire =
                andValues(bodyBuilder, location,
                          {selection.selected[candidate],
                           accessor.getInput(
                               outputEndpoints[outputPort]->ready.getName())});
            fired.push_back(fire);
            fuOutputReadyTerms[candidate].push_back(fire);
          }
          if (payloadWidth != 0)
            accessor.setOutput(outputEndpoints[outputPort]->data->getName(),
                               outputData);
          accessor.setOutput(outputEndpoints[outputPort]->tag->getName(),
                             outputTag);
          accessor.setOutput(
              outputEndpoints[outputPort]->valid.getName(),
              orValues(bodyBuilder, location, selection.selected));
          selection.next.setValue(
              nextCursor(bodyBuilder, location, selection.cursor, fired));
        }

        for (std::size_t candidate = 0; candidate != fuOutputs.size();
             ++candidate)
          fuOutputReadyTerms[candidate].push_back(
              resultRoutes[candidate].discard);

        for (std::uint32_t fifo = 0; fifo != layout.registerFifoCount; ++fifo) {
          std::vector<mlir::Value> requests;
          requests.reserve(fuOutputs.size());
          for (std::size_t candidate = 0; candidate != fuOutputs.size();
               ++candidate)
            requests.push_back(andValues(
                bodyBuilder, location,
                {fuOutputs[candidate].valid, resultRoutes[candidate].route,
                 equals(bodyBuilder, location, resultRoutes[candidate].target,
                        layout.outputPortCount + fifo)}));
          StatefulSelection selection = makeStatefulSelection(
              bodyBuilder, location, backedges, requests,
              accessor.getInput("clock"), accessor.getInput("reset"),
              "register_fifo_" + std::to_string(fifo) + "_write_cursor_reg",
              clockReset);
          mlir::Value writePayload =
              zero(bodyBuilder, location, fifoPayloadWidth);
          for (std::size_t candidate = 0; candidate != fuOutputs.size();
               ++candidate) {
            mlir::Value data;
            if (payloadWidth != 0) {
              auto adapted = adaptForwardTransportSignals(
                  bodyBuilder, location,
                  fuOutputs[candidate].endpoint->dataPath,
                  ::fabric::DataPathType{::fabric::DataPathKind::Bits,
                                         payloadWidth, 0},
                  ForwardTransportSignals{fuOutputs[candidate].valid,
                                          fuOutputs[candidate].data,
                                          std::nullopt});
              if (!adapted) {
                materializationError = llvm::toString(adapted.takeError());
                backedges.abandon();
                return;
              }
              data = *adapted->payload;
            }
            mlir::Value packed =
                packToken(bodyBuilder, location,
                          payloadWidth == 0 ? std::optional<mlir::Value>{}
                                            : std::optional<mlir::Value>{data},
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
          instanceInputs.emplace(
              queuePort("enqueue", 0, "_valid"),
              orValues(bodyBuilder, location, selection.selected));
          instanceInputs.emplace(queuePort("enqueue", 0, "_commit"),
                                 bitConstant(bodyBuilder, location, true));
          instanceInputs.emplace(
              queuePort("dequeue", 0, "_ready"),
              orValues(bodyBuilder, location, fifoReadReadyTerms[fifo]));
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
            mlir::Value fire =
                andValues(bodyBuilder, location,
                          {selection.selected[candidate], enqueueReady});
            fired.push_back(fire);
            fuOutputReadyTerms[candidate].push_back(fire);
          }
          selection.next.setValue(
              nextCursor(bodyBuilder, location, selection.cursor, fired));
        }

        for (std::size_t candidate = 0; candidate != fuOutputs.size();
             ++candidate)
          fuOutputs[candidate].ready.setValue(
              orValues(bodyBuilder, location, fuOutputReadyTerms[candidate]));

        for (const EndpointPlan *input : inputEndpoints)
          accessor.setOutput(input->ready.getName(),
                             inputReady[input->localOrdinal]);
      });
  if (materializationError)
    return invalid(*materializationError);
  return PeModule{pe, module, std::move(*endpoints)};
}

} // namespace loom::hardware::rtl::hierarchy

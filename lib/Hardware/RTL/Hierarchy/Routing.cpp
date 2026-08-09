#include "Components.h"

#include "Fabric/IR/FabricOps.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::rtl::hierarchy {
namespace {

const EndpointPlan *findEndpoint(llvm::ArrayRef<EndpointPlan> endpoints,
                                 fabric::FabricPortDirection direction,
                                 fabric::FabricOrdinal ordinal) {
  const EndpointPlan *result = nullptr;
  for (const EndpointPlan &endpoint : endpoints)
    if (endpoint.direction == direction && endpoint.localOrdinal == ordinal) {
      if (result)
        return nullptr;
      result = &endpoint;
    }
  return result;
}

void appendComponentPorts(mlir::OpBuilder &builder,
                          const ConfigurationABI &configurationAbi,
                          llvm::ArrayRef<EndpointPlan> endpoints,
                          llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
                          llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs,
                          bool stateful = false) {
  if (stateful)
    appendClockResetAndConfigurationPorts(builder, configurationAbi, inputs);
  else
    for (const ProgrammingUnit &unit : configurationAbi.programmingUnits())
      inputs.push_back(circt::hw::PortInfo{
          {builder.getStringAttr(configurationPortName(unit.id)),
           builder.getIntegerType(static_cast<unsigned>(unit.payloadBitCount)),
           circt::hw::ModulePort::Direction::Input}});
  for (const EndpointPlan &endpoint : endpoints)
    appendEndpointPorts(inputs, outputs, endpoint);
}

llvm::Expected<std::uint64_t>
singleSelectedBit(llvm::ArrayRef<std::uint8_t> bytes, std::uint64_t bitCount) {
  std::optional<std::uint64_t> selected;
  for (std::uint64_t bit = 0; bit < bitCount; ++bit)
    if (((bytes[static_cast<std::size_t>(bit / 8)] >> (bit % 8)) & 1U) != 0) {
      if (selected)
        return invalid("one traversal carrier selects multiple bits");
      selected = bit;
    }
  if (!selected)
    return invalid("one traversal carrier selects no bit");
  return *selected;
}

struct SwitchRoute final {
  const EndpointPlan *input = nullptr;
  const EndpointPlan *output = nullptr;
  std::uint64_t configurationBit = 0;
};

llvm::Expected<SwitchModule>
buildSpatialSwitchModule(mlir::OpBuilder &builder, mlir::Location location,
                         fabric::SpatialCoreOccurrenceRef spatialCore,
                         const fabric::FabricArtifactView &fabric,
                         const ConfigurationABI &configurationAbi,
                         fabric::FabricSwitchOccurrenceRef sw) {
  if (fabric.switchSchedule(sw) != ::fabric::Schedule::Spatial)
    return unsupported("Temporal switch hierarchy lowering is not implemented");
  auto endpoints = deriveEndpointPlans(
      builder, fabric, fabric::FabricTransportEndpointOwnerRef::of(sw));
  if (!endpoints)
    return endpoints.takeError();
  const fabric::FabricSemanticConfigFieldRef field{
      fabric::FabricConfigurationOwnerRef(
          fabric::FabricInventoryOwnerRef::of(sw)),
      0};
  auto decoder = prepareFieldDecoder(spatialCore, field, configurationAbi);
  if (!decoder)
    return decoder.takeError();
  auto relation = fabric.semanticFieldRelation(
      field, *const_cast<mlir::Operation *>(fabric.canonicalOperation())
                  ->getContext());
  if (!relation)
    return relation.takeError();
  if (relation->kind() != fabric::FabricSemanticFieldRelationKind::Direct ||
      relation->directEncodedBitCount() != decoder->encodedBitCount)
    return invalid("Spatial switch field is not its exact direct carrier");

  std::vector<SwitchRoute> routes;
  for (const fabric::FabricPhysicalTraversalView &traversal :
       fabric.physicalTraversals()) {
    if (traversal.reference.kind() !=
        fabric::FabricPhysicalTraversalKind::SwitchTraversal)
      continue;
    const auto &payload = std::get<fabric::FabricSwitchTraversalPayload>(
        traversal.reference.payload);
    if (payload.owner != sw)
      continue;
    const EndpointPlan *input = findEndpoint(
        *endpoints, fabric::FabricPortDirection::Input, payload.input);
    const EndpointPlan *output = findEndpoint(
        *endpoints, fabric::FabricPortDirection::Output, payload.output);
    if (!input || !output)
      return invalid("switch traversal names an absent endpoint");
    auto semantic = fabric::encodeSpatialSwitchConfiguration(
        fabric, field, {traversal.reference});
    if (!semantic)
      return semantic.takeError();
    auto bit = singleSelectedBit(semantic->bytes(), decoder->encodedBitCount);
    if (!bit)
      return bit.takeError();
    routes.push_back({input, output, *bit});
  }
  if (routes.empty())
    return invalid("Spatial switch has no admitted traversal");

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  appendComponentPorts(builder, configurationAbi, *endpoints, inputs, outputs);
  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_spatial_switch_" + std::to_string(sw.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value fieldSignal =
            decodeFieldSignal(bodyBuilder, location, accessor, *decoder);
        std::vector<mlir::Value> selected;
        selected.reserve(routes.size());
        for (const SwitchRoute &route : routes)
          selected.push_back(selectedBit(bodyBuilder, location, fieldSignal,
                                         route.configurationBit));

        for (const EndpointPlan &inputEndpoint : *endpoints) {
          if (inputEndpoint.direction != fabric::FabricPortDirection::Input)
            continue;
          llvm::SmallVector<mlir::Value> allSelectedReady;
          llvm::SmallVector<mlir::Value> anySelected;
          for (auto [index, route] : llvm::enumerate(routes)) {
            if (route.input != &inputEndpoint)
              continue;
            anySelected.push_back(selected[index]);
            allSelectedReady.push_back(circt::comb::OrOp::create(
                bodyBuilder, location,
                circt::comb::createOrFoldNot(bodyBuilder, location,
                                             selected[index]),
                accessor.getInput(route.output->ready.getName())));
          }
          accessor.setOutput(
              inputEndpoint.ready.getName(),
              andValues(bodyBuilder, location,
                        {orValues(bodyBuilder, location, anySelected),
                         andValues(bodyBuilder, location, allSelectedReady)}));
        }

        for (const EndpointPlan &outputEndpoint : *endpoints) {
          if (outputEndpoint.direction != fabric::FabricPortDirection::Output)
            continue;
          mlir::Value data =
              outputEndpoint.data
                  ? circt::hw::ConstantOp::create(
                        bodyBuilder, location,
                        llvm::APInt(outputEndpoint.dataPath.payloadWidthBits,
                                    0))
                  : mlir::Value{};
          mlir::Value tag =
              outputEndpoint.tag
                  ? circt::hw::ConstantOp::create(
                        bodyBuilder, location,
                        llvm::APInt(outputEndpoint.dataPath.tagWidthBits, 0))
                  : mlir::Value{};
          llvm::SmallVector<mlir::Value> validTerms;
          for (auto [index, route] : llvm::enumerate(routes)) {
            if (route.output != &outputEndpoint)
              continue;
            llvm::SmallVector<mlir::Value> peerReady;
            for (auto [peerIndex, peer] : llvm::enumerate(routes)) {
              if (peer.input != route.input || peer.output == route.output)
                continue;
              peerReady.push_back(circt::comb::OrOp::create(
                  bodyBuilder, location,
                  circt::comb::createOrFoldNot(bodyBuilder, location,
                                               selected[peerIndex]),
                  accessor.getInput(peer.output->ready.getName())));
            }
            validTerms.push_back(
                andValues(bodyBuilder, location,
                          {selected[index],
                           accessor.getInput(route.input->valid.getName()),
                           andValues(bodyBuilder, location, peerReady)}));
            auto adapted = adaptForwardTransportSignals(
                bodyBuilder, location, route.input->dataPath,
                outputEndpoint.dataPath,
                ForwardTransportSignals{
                    accessor.getInput(route.input->valid.getName()),
                    route.input->data
                        ? std::optional<mlir::Value>{accessor.getInput(
                              route.input->data->getName())}
                        : std::nullopt,
                    route.input->tag
                        ? std::optional<mlir::Value>{accessor.getInput(
                              route.input->tag->getName())}
                        : std::nullopt});
            if (!adapted) {
              materializationError = llvm::toString(adapted.takeError());
              return;
            }
            if (outputEndpoint.data)
              data = circt::comb::MuxOp::create(bodyBuilder, location,
                                                selected[index],
                                                *adapted->payload, data, true);
            if (outputEndpoint.tag)
              tag = circt::comb::MuxOp::create(bodyBuilder, location,
                                               selected[index], *adapted->tag,
                                               tag, true);
          }
          if (outputEndpoint.data)
            accessor.setOutput(outputEndpoint.data->getName(), data);
          if (outputEndpoint.tag)
            accessor.setOutput(outputEndpoint.tag->getName(), tag);
          accessor.setOutput(outputEndpoint.valid.getName(),
                             orValues(bodyBuilder, location, validTerms));
        }
      });
  if (materializationError)
    return invalid(*materializationError);
  return SwitchModule{sw, module, std::move(*endpoints)};
}

unsigned counterWidth(std::uint64_t bound) {
  return std::max(1U, llvm::Log2_64_Ceil(bound));
}

mlir::Value incrementModulo(mlir::OpBuilder &builder, mlir::Location location,
                            mlir::Value value, std::uint64_t modulus) {
  const unsigned width =
      mlir::cast<mlir::IntegerType>(value.getType()).getWidth();
  mlir::Value one =
      circt::hw::ConstantOp::create(builder, location, llvm::APInt(width, 1));
  mlir::Value zero =
      circt::hw::ConstantOp::create(builder, location, llvm::APInt(width, 0));
  mlir::Value last = circt::hw::ConstantOp::create(
      builder, location, llvm::APInt(width, modulus - 1));
  mlir::Value wraps = circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::eq, value, last, true);
  mlir::Value incremented =
      circt::comb::AddOp::create(builder, location, value, one, true);
  return circt::comb::MuxOp::create(builder, location, wraps, zero, incremented,
                                    true);
}

llvm::Expected<FifoModule>
buildFifoModule(mlir::OpBuilder &builder, mlir::Location location,
                fabric::SpatialCoreOccurrenceRef spatialCore,
                const fabric::FabricArtifactView &fabric,
                const ConfigurationABI &configurationAbi,
                const ClockResetPlan &clockReset,
                fabric::FabricFifoOccurrenceRef fifo) {
  auto canonical = findCanonicalEntityOperation(fabric, fifo.id());
  if (!canonical)
    return canonical.takeError();
  auto operation = mlir::dyn_cast<::fabric::FifoOp>(*canonical);
  if (!operation)
    return invalid("FIFO occurrence entity does not name fabric.fifo");
  const std::uint64_t depth = operation.getMaxDepth();
  if (depth == 0)
    return invalid("FIFO has zero physical depth");
  auto endpoints = deriveEndpointPlans(
      builder, fabric, fabric::FabricTransportEndpointOwnerRef::of(fifo));
  if (!endpoints)
    return endpoints.takeError();
  const EndpointPlan *input =
      findEndpoint(*endpoints, fabric::FabricPortDirection::Input, 0);
  const EndpointPlan *output =
      findEndpoint(*endpoints, fabric::FabricPortDirection::Output, 0);
  if (!input || !output || endpoints->size() != 2)
    return invalid("FIFO endpoint inventory is not one-in/one-out");
  const fabric::FabricSemanticConfigFieldRef field{
      fabric::FabricConfigurationOwnerRef(
          fabric::FabricInventoryOwnerRef::of(fifo)),
      0};
  auto prepared = prepareFiniteField(spatialCore, field, configurationAbi);
  if (!prepared)
    return prepared.takeError();
  auto bufferedSemantic = fabric::encodeFabricFifoConfiguration(
      fabric, field, fabric::FabricFifoTraversalMode::Buffered);
  if (!bufferedSemantic)
    return bufferedSemantic.takeError();
  auto bufferedCode =
      physicalCode(*prepared->second, bufferedSemantic->bytes());
  if (!bufferedCode)
    return bufferedCode.takeError();
  std::optional<llvm::APInt> bypassCode;
  if (operation.getBypassable()) {
    auto bypassSemantic = fabric::encodeFabricFifoConfiguration(
        fabric, field, fabric::FabricFifoTraversalMode::Bypass);
    if (!bypassSemantic)
      return bypassSemantic.takeError();
    auto code = physicalCode(*prepared->second, bypassSemantic->bytes());
    if (!code)
      return code.takeError();
    bypassCode = std::move(*code);
  }

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  appendComponentPorts(builder, configurationAbi, *endpoints, inputs, outputs,
                       true);
  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_fabric_fifo_" + std::to_string(fifo.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value fieldSignal =
            decodeFieldSignal(bodyBuilder, location, accessor, prepared->first);
        mlir::Value buffered =
            matchesCode(bodyBuilder, location, fieldSignal, *bufferedCode);
        mlir::Value bypass =
            bypassCode
                ? matchesCode(bodyBuilder, location, fieldSignal, *bypassCode)
                : bitConstant(bodyBuilder, location, false);
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        const unsigned pointerBits = counterWidth(depth);
        const unsigned occupancyBits = counterWidth(depth + 1);
        circt::Backedge headNext =
            backedges.get(bodyBuilder.getIntegerType(pointerBits));
        circt::Backedge tailNext =
            backedges.get(bodyBuilder.getIntegerType(pointerBits));
        circt::Backedge occupancyNext =
            backedges.get(bodyBuilder.getIntegerType(occupancyBits));
        mlir::Value head = createRegister(
            bodyBuilder, location, headNext, accessor.getInput("clock"),
            accessor.getInput("reset"), llvm::APInt(pointerBits, 0), "head_reg",
            clockReset.asynchronousReset);
        mlir::Value tail = createRegister(
            bodyBuilder, location, tailNext, accessor.getInput("clock"),
            accessor.getInput("reset"), llvm::APInt(pointerBits, 0), "tail_reg",
            clockReset.asynchronousReset);
        mlir::Value occupancy = createRegister(
            bodyBuilder, location, occupancyNext, accessor.getInput("clock"),
            accessor.getInput("reset"), llvm::APInt(occupancyBits, 0),
            "occupancy_reg", clockReset.asynchronousReset);
        mlir::Value zeroOccupancy = circt::hw::ConstantOp::create(
            bodyBuilder, location, llvm::APInt(occupancyBits, 0));
        mlir::Value fullOccupancy = circt::hw::ConstantOp::create(
            bodyBuilder, location, llvm::APInt(occupancyBits, depth));
        mlir::Value empty = circt::comb::ICmpOp::create(
            bodyBuilder, location, circt::comb::ICmpPredicate::eq, occupancy,
            zeroOccupancy, true);
        mlir::Value full = circt::comb::ICmpOp::create(
            bodyBuilder, location, circt::comb::ICmpPredicate::eq, occupancy,
            fullOccupancy, true);
        mlir::Value bufferedInputReady =
            andValues(bodyBuilder, location,
                      {buffered, circt::comb::createOrFoldNot(bodyBuilder,
                                                              location, full)});
        mlir::Value bufferedOutputValid = andValues(
            bodyBuilder, location,
            {buffered,
             circt::comb::createOrFoldNot(bodyBuilder, location, empty)});
        mlir::Value enqueue = andValues(
            bodyBuilder, location,
            {bufferedInputReady, accessor.getInput(input->valid.getName())});
        mlir::Value dequeue = andValues(
            bodyBuilder, location,
            {bufferedOutputValid, accessor.getInput(output->ready.getName())});

        struct StorageBank final {
          std::vector<circt::Backedge> next;
          std::vector<mlir::Value> current;
          unsigned width = 0;
        };
        const auto makeBank = [&](unsigned width,
                                  llvm::StringRef name) -> StorageBank {
          StorageBank bank;
          bank.width = width;
          if (width == 0)
            return bank;
          bank.next.resize(depth);
          bank.current.resize(depth);
          for (std::uint64_t slot = 0; slot < depth; ++slot) {
            bank.next[slot] = backedges.get(bodyBuilder.getIntegerType(width));
            bank.current[slot] = createRegister(
                bodyBuilder, location, bank.next[slot],
                accessor.getInput("clock"), accessor.getInput("reset"),
                llvm::APInt(width, 0),
                name.str() + "_" + std::to_string(slot) + "_reg",
                clockReset.asynchronousReset);
          }
          return bank;
        };
        StorageBank dataBank =
            makeBank(output->dataPath.payloadWidthBits, "data");
        StorageBank tagBank = makeBank(output->dataPath.tagWidthBits, "tag");
        const auto writeBank = [&](StorageBank &bank,
                                   std::optional<mlir::Value> source) {
          if (bank.width == 0)
            return;
          if (!source) {
            materializationError = "FIFO storage source is absent";
            return;
          }
          for (std::uint64_t slot = 0; slot < depth; ++slot) {
            mlir::Value slotValue = circt::hw::ConstantOp::create(
                bodyBuilder, location, llvm::APInt(pointerBits, slot));
            mlir::Value selected = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq, tail,
                slotValue, true);
            mlir::Value write =
                andValues(bodyBuilder, location, {enqueue, selected});
            bank.next[slot].setValue(
                circt::comb::MuxOp::create(bodyBuilder, location, write,
                                           *source, bank.current[slot], true));
          }
        };
        auto adaptedInput = adaptForwardTransportSignals(
            bodyBuilder, location, input->dataPath, output->dataPath,
            ForwardTransportSignals{
                accessor.getInput(input->valid.getName()),
                input->data ? std::optional<mlir::Value>{accessor.getInput(
                                  input->data->getName())}
                            : std::nullopt,
                input->tag ? std::optional<mlir::Value>{accessor.getInput(
                                 input->tag->getName())}
                           : std::nullopt});
        if (!adaptedInput) {
          materializationError = llvm::toString(adaptedInput.takeError());
          backedges.abandon();
          return;
        }
        writeBank(dataBank, adaptedInput->payload);
        writeBank(tagBank, adaptedInput->tag);
        if (materializationError) {
          backedges.abandon();
          return;
        }
        const auto readBank = [&](const StorageBank &bank) -> mlir::Value {
          if (bank.width == 0)
            return {};
          mlir::Value value = circt::hw::ConstantOp::create(
              bodyBuilder, location, llvm::APInt(bank.width, 0));
          for (std::uint64_t slot = 0; slot < depth; ++slot) {
            mlir::Value slotValue = circt::hw::ConstantOp::create(
                bodyBuilder, location, llvm::APInt(pointerBits, slot));
            mlir::Value selected = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq, head,
                slotValue, true);
            value = circt::comb::MuxOp::create(bodyBuilder, location, selected,
                                               bank.current[slot], value, true);
          }
          return value;
        };
        mlir::Value bufferedData = readBank(dataBank);
        mlir::Value bufferedTag = readBank(tagBank);
        if (output->data)
          accessor.setOutput(output->data->getName(),
                             circt::comb::MuxOp::create(
                                 bodyBuilder, location, bypass,
                                 *adaptedInput->payload, bufferedData, true));
        if (output->tag)
          accessor.setOutput(output->tag->getName(),
                             circt::comb::MuxOp::create(
                                 bodyBuilder, location, bypass,
                                 *adaptedInput->tag, bufferedTag, true));
        accessor.setOutput(
            output->valid.getName(),
            circt::comb::OrOp::create(
                bodyBuilder, location, bufferedOutputValid,
                andValues(
                    bodyBuilder, location,
                    {bypass, accessor.getInput(input->valid.getName())})));
        accessor.setOutput(
            input->ready.getName(),
            circt::comb::OrOp::create(
                bodyBuilder, location, bufferedInputReady,
                andValues(
                    bodyBuilder, location,
                    {bypass, accessor.getInput(output->ready.getName())})));

        mlir::Value incrementHead =
            incrementModulo(bodyBuilder, location, head, depth);
        mlir::Value incrementTail =
            incrementModulo(bodyBuilder, location, tail, depth);
        headNext.setValue(circt::comb::MuxOp::create(
            bodyBuilder, location, dequeue, incrementHead, head, true));
        tailNext.setValue(circt::comb::MuxOp::create(
            bodyBuilder, location, enqueue, incrementTail, tail, true));
        mlir::Value oneOccupancy = circt::hw::ConstantOp::create(
            bodyBuilder, location, llvm::APInt(occupancyBits, 1));
        mlir::Value incrementOccupancy = circt::comb::AddOp::create(
            bodyBuilder, location, occupancy, oneOccupancy, true);
        mlir::Value decrementOccupancy = circt::comb::SubOp::create(
            bodyBuilder, location, occupancy, oneOccupancy, true);
        mlir::Value enqueueOnly = andValues(
            bodyBuilder, location,
            {enqueue,
             circt::comb::createOrFoldNot(bodyBuilder, location, dequeue)});
        mlir::Value dequeueOnly = andValues(
            bodyBuilder, location,
            {dequeue,
             circt::comb::createOrFoldNot(bodyBuilder, location, enqueue)});
        mlir::Value occupancyAfterDequeue =
            circt::comb::MuxOp::create(bodyBuilder, location, dequeueOnly,
                                       decrementOccupancy, occupancy, true);
        occupancyNext.setValue(circt::comb::MuxOp::create(
            bodyBuilder, location, enqueueOnly, incrementOccupancy,
            occupancyAfterDequeue, true));
      });
  if (materializationError)
    return invalid(*materializationError);
  return FifoModule{fifo, module, std::move(*endpoints)};
}

llvm::Expected<BoundaryModule>
buildBoundaryModule(mlir::OpBuilder &builder, mlir::Location location,
                    fabric::SpatialCoreOccurrenceRef spatialCore,
                    const fabric::FabricArtifactView &fabric,
                    const ConfigurationABI &configurationAbi,
                    fabric::FabricBoundaryOccurrenceRef boundary) {
  auto canonical = findCanonicalEntityOperation(fabric, boundary.id());
  if (!canonical)
    return canonical.takeError();
  auto operation = mlir::dyn_cast<::fabric::BoundaryOp>(*canonical);
  if (!operation)
    return invalid("boundary occurrence entity does not name fabric.boundary");
  if (operation.getDirection() != ::fabric::BoundaryDirection::S2t ||
      operation.getNumOperands() != 2 || operation.getNumResults() != 1)
    return unsupported(
        "hierarchical boundary lowering currently supports two-input s2t");
  auto endpoints = deriveEndpointPlans(
      builder, fabric, fabric::FabricTransportEndpointOwnerRef::of(boundary));
  if (!endpoints)
    return endpoints.takeError();
  const EndpointPlan *dataInput =
      findEndpoint(*endpoints, fabric::FabricPortDirection::Input, 0);
  const EndpointPlan *tagInput =
      findEndpoint(*endpoints, fabric::FabricPortDirection::Input, 1);
  const EndpointPlan *output =
      findEndpoint(*endpoints, fabric::FabricPortDirection::Output, 0);
  if (!dataInput || !tagInput || !output || endpoints->size() != 3 ||
      !output->tag || !tagInput->data)
    return invalid("s2t boundary endpoint inventory is incomplete");
  const fabric::FabricSemanticConfigFieldRef field{
      fabric::FabricConfigurationOwnerRef(
          fabric::FabricInventoryOwnerRef::of(boundary)),
      0};
  auto prepared = prepareFiniteField(spatialCore, field, configurationAbi);
  if (!prepared)
    return prepared.takeError();
  auto relation = fabric.semanticFieldRelation(
      field, *const_cast<mlir::Operation *>(fabric.canonicalOperation())
                  ->getContext());
  if (!relation)
    return relation.takeError();
  if (relation->finiteDomain().size() != 2)
    return invalid("two-input s2t boundary has the wrong finite domain");
  auto activeCode =
      physicalCode(*prepared->second, relation->finiteDomain().back().bytes());
  if (!activeCode)
    return activeCode.takeError();

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  appendComponentPorts(builder, configurationAbi, *endpoints, inputs, outputs);
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_fabric_boundary_" +
                            std::to_string(boundary.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value fieldSignal =
            decodeFieldSignal(bodyBuilder, location, accessor, prepared->first);
        mlir::Value active =
            matchesCode(bodyBuilder, location, fieldSignal, *activeCode);
        mlir::Value dataValid = accessor.getInput(dataInput->valid.getName());
        mlir::Value tagValid = accessor.getInput(tagInput->valid.getName());
        mlir::Value outputReady = accessor.getInput(output->ready.getName());
        if (output->data)
          accessor.setOutput(output->data->getName(),
                             accessor.getInput(dataInput->data->getName()));
        accessor.setOutput(output->tag->getName(),
                           accessor.getInput(tagInput->data->getName()));
        accessor.setOutput(
            output->valid.getName(),
            andValues(bodyBuilder, location, {active, dataValid, tagValid}));
        accessor.setOutput(
            dataInput->ready.getName(),
            andValues(bodyBuilder, location, {active, outputReady, tagValid}));
        accessor.setOutput(
            tagInput->ready.getName(),
            andValues(bodyBuilder, location, {active, outputReady, dataValid}));
      });
  return BoundaryModule{boundary, module, std::move(*endpoints)};
}

} // namespace

llvm::Expected<std::vector<SwitchModule>>
buildSwitchModules(mlir::OpBuilder &builder, mlir::Location location,
                   fabric::SpatialCoreOccurrenceRef spatialCore,
                   const fabric::FabricArtifactView &fabric,
                   const ConfigurationABI &configurationAbi) {
  std::vector<SwitchModule> result;
  result.reserve(fabric.switchOccurrences().size());
  for (fabric::FabricSwitchOccurrenceRef sw : fabric.switchOccurrences()) {
    auto module = buildSpatialSwitchModule(builder, location, spatialCore,
                                           fabric, configurationAbi, sw);
    if (!module)
      return module.takeError();
    result.push_back(std::move(*module));
  }
  return result;
}

llvm::Expected<std::vector<FifoModule>>
buildFifoModules(mlir::OpBuilder &builder, mlir::Location location,
                 fabric::SpatialCoreOccurrenceRef spatialCore,
                 const fabric::FabricArtifactView &fabric,
                 const ConfigurationABI &configurationAbi,
                 const ClockResetPlan &clockReset) {
  std::vector<FifoModule> result;
  result.reserve(fabric.fifoOccurrences().size());
  for (fabric::FabricFifoOccurrenceRef fifo : fabric.fifoOccurrences()) {
    auto module = buildFifoModule(builder, location, spatialCore, fabric,
                                  configurationAbi, clockReset, fifo);
    if (!module)
      return module.takeError();
    result.push_back(std::move(*module));
  }
  return result;
}

llvm::Expected<std::vector<BoundaryModule>>
buildBoundaryModules(mlir::OpBuilder &builder, mlir::Location location,
                     fabric::SpatialCoreOccurrenceRef spatialCore,
                     const fabric::FabricArtifactView &fabric,
                     const ConfigurationABI &configurationAbi) {
  std::vector<BoundaryModule> result;
  result.reserve(fabric.boundaryOccurrences().size());
  for (fabric::FabricBoundaryOccurrenceRef boundary :
       fabric.boundaryOccurrences()) {
    auto module = buildBoundaryModule(builder, location, spatialCore, fabric,
                                      configurationAbi, boundary);
    if (!module)
      return module.takeError();
    result.push_back(std::move(*module));
  }
  return result;
}

} // namespace loom::hardware::rtl::hierarchy

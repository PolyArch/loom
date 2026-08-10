#include "Components.h"

#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
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
                          const ConfigurationTransportLayout &transportLayout,
                          llvm::ArrayRef<EndpointPlan> endpoints,
                          llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
                          llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs,
                          bool stateful = false) {
  if (stateful)
    appendClockResetAndConfigurationPorts(builder, configurationAbi,
                                          transportLayout, inputs);
  else {
    for (auto [ordinal, transportUnit] :
         llvm::enumerate(transportLayout.units)) {
      const ProgrammingUnit *unit = configurationAbi.findProgrammingUnit(
          transportUnit.programmingUnit.unitId);
      if (!unit)
        continue;
      inputs.push_back(circt::hw::PortInfo{
          {builder.getStringAttr(configurationPortName(ordinal)),
           builder.getIntegerType(static_cast<unsigned>(unit->payloadBitCount)),
           circt::hw::ModulePort::Direction::Input}});
    }
  }
  for (const EndpointPlan &endpoint : endpoints)
    appendEndpointPorts(inputs, outputs, endpoint);
}

struct SwitchRoute final {
  const EndpointPlan *input = nullptr;
  const EndpointPlan *output = nullptr;
  fabric::FabricOrdinal inputOrdinal = 0;
  fabric::FabricOrdinal outputOrdinal = 0;
  std::uint64_t configurationBit = 0;
};

struct SwitchArbitrationComponent final {
  std::vector<unsigned> inputs;
  std::vector<unsigned> outputs;
  std::vector<unsigned> requesterOrder;
  std::optional<unsigned> roundRobinResetPosition;
};

std::vector<unsigned> switchComponents(unsigned inputCount,
                                       unsigned outputCount,
                                       llvm::ArrayRef<SwitchRoute> routes) {
  std::vector<unsigned> parent(inputCount + outputCount);
  for (unsigned index = 0; index != parent.size(); ++index)
    parent[index] = index;
  const auto root = [&](unsigned value) {
    while (parent[value] != value) {
      parent[value] = parent[parent[value]];
      value = parent[value];
    }
    return value;
  };
  for (const SwitchRoute &route : routes) {
    unsigned input = static_cast<unsigned>(route.inputOrdinal);
    unsigned output = inputCount + static_cast<unsigned>(route.outputOrdinal);
    input = root(input);
    output = root(output);
    if (input != output)
      parent[output] = input;
  }
  std::vector<unsigned> result(parent.size());
  for (unsigned index = 0; index != parent.size(); ++index)
    result[index] = root(index);
  return result;
}

unsigned counterWidth(std::uint64_t bound);

llvm::Expected<SwitchModule>
buildSwitchModule(mlir::OpBuilder &builder, mlir::Location location,
                  fabric::SpatialCoreOccurrenceRef spatialCore,
                  const fabric::FabricArtifactView &fabric,
                  const ConfigurationABI &configurationAbi,
                  const ConfigurationTransportLayout &transportLayout,
                  const ClockResetPlan &clockReset,
                  fabric::FabricSwitchOccurrenceRef sw) {
  const std::optional<::fabric::Schedule> schedule = fabric.switchSchedule(sw);
  if (!schedule)
    return invalid("switch has no exact schedule");
  auto endpoints = deriveEndpointPlans(
      builder, fabric, fabric::FabricTransportEndpointOwnerRef::of(sw));
  if (!endpoints)
    return endpoints.takeError();
  const fabric::FabricSemanticConfigFieldRef field{
      fabric::FabricConfigurationOwnerRef(
          fabric::FabricInventoryOwnerRef::of(sw)),
      0};
  auto decoder = prepareFieldDecoder(spatialCore, field, configurationAbi,
                                     transportLayout);
  if (!decoder)
    return decoder.takeError();
  auto relation = fabric.semanticFieldRelation(
      field, *const_cast<mlir::Operation *>(fabric.canonicalOperation())
                  ->getContext());
  if (!relation)
    return relation.takeError();
  if (relation->kind() != fabric::FabricSemanticFieldRelationKind::Direct ||
      relation->directEncodedBitCount() != decoder->encodedBitCount)
    return invalid("switch field is not its exact direct carrier");

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
    routes.push_back(
        {input, output, payload.input, payload.output, std::uint64_t(0)});
  }
  if (routes.empty())
    return invalid("switch has no admitted traversal");
  llvm::sort(routes, [](const SwitchRoute &lhs, const SwitchRoute &rhs) {
    return std::tie(lhs.outputOrdinal, lhs.inputOrdinal) <
           std::tie(rhs.outputOrdinal, rhs.inputOrdinal);
  });
  for (auto [ordinal, route] : llvm::enumerate(routes))
    route.configurationBit = ordinal;

  unsigned inputCount = 0;
  unsigned outputCount = 0;
  std::vector<const EndpointPlan *> inputEndpoints;
  std::vector<const EndpointPlan *> outputEndpoints;
  for (const EndpointPlan &endpoint : *endpoints) {
    if (endpoint.direction == fabric::FabricPortDirection::Input)
      inputCount = std::max(inputCount,
                            static_cast<unsigned>(endpoint.localOrdinal + 1));
    else
      outputCount = std::max(outputCount,
                             static_cast<unsigned>(endpoint.localOrdinal + 1));
  }
  inputEndpoints.resize(inputCount);
  outputEndpoints.resize(outputCount);
  for (const EndpointPlan &endpoint : *endpoints) {
    auto &slot = endpoint.direction == fabric::FabricPortDirection::Input
                     ? inputEndpoints[endpoint.localOrdinal]
                     : outputEndpoints[endpoint.localOrdinal];
    if (slot)
      return invalid("switch endpoint ordinal is duplicated");
    slot = &endpoint;
  }
  if (llvm::is_contained(inputEndpoints, nullptr) ||
      llvm::is_contained(outputEndpoints, nullptr))
    return invalid("switch endpoint domain is not dense");

  const std::vector<unsigned> componentRoots =
      switchComponents(inputCount, outputCount, routes);
  std::map<unsigned, SwitchArbitrationComponent> componentsByRoot;
  for (unsigned input = 0; input != inputCount; ++input)
    componentsByRoot[componentRoots[input]].inputs.push_back(input);
  for (unsigned output = 0; output != outputCount; ++output)
    componentsByRoot[componentRoots[inputCount + output]].outputs.push_back(
        output);

  const ::fabric::ResourceContract *contract =
      fabric.resourceContract(fabric::FabricInventoryOwnerRef::of(sw));
  if (!contract)
    return invalid("switch has no exact ResourceContract");
  const std::optional<::fabric::GrantPolicyView> policy =
      contract->grantPolicy();
  std::vector<unsigned> fullOrder;
  std::optional<unsigned> resetRequester;
  bool roundRobin = false;
  if (policy) {
    if (const auto *fixed =
            std::get_if<::fabric::FixedPriorityView>(&*policy)) {
      for (::fabric::RequesterKey requester : fixed->requesterOrder())
        fullOrder.push_back(requester.ordinal());
    } else {
      const auto &typed = std::get<::fabric::RoundRobinView>(*policy);
      roundRobin = true;
      for (::fabric::RequesterKey requester : typed.requesterCycle())
        fullOrder.push_back(requester.ordinal());
      resetRequester = typed.resetCursor().ordinal();
    }
  }
  std::vector<SwitchArbitrationComponent> components;
  components.reserve(componentsByRoot.size());
  for (auto &[root, component] : componentsByRoot) {
    (void)root;
    if (policy) {
      for (unsigned requester : fullOrder)
        if (llvm::is_contained(component.inputs, requester))
          component.requesterOrder.push_back(requester);
      if (component.requesterOrder.size() != component.inputs.size())
        return invalid("switch GrantPolicy omits a requester");
      if (roundRobin) {
        const auto reset = llvm::find(fullOrder, *resetRequester);
        if (reset == fullOrder.end())
          return invalid("switch reset requester is outside its policy");
        const unsigned resetOrdinal =
            static_cast<unsigned>(reset - fullOrder.begin());
        unsigned bestDistance = std::numeric_limits<unsigned>::max();
        unsigned bestPosition = 0;
        for (auto [position, requester] :
             llvm::enumerate(component.requesterOrder)) {
          const auto found = llvm::find(fullOrder, requester);
          const unsigned ordinal =
              static_cast<unsigned>(found - fullOrder.begin());
          const unsigned distance =
              ordinal >= resetOrdinal
                  ? ordinal - resetOrdinal
                  : static_cast<unsigned>(fullOrder.size()) - resetOrdinal +
                        ordinal;
          if (distance < bestDistance) {
            bestDistance = distance;
            bestPosition = static_cast<unsigned>(position);
          }
        }
        component.roundRobinResetPosition = bestPosition;
      }
    } else {
      component.requesterOrder = component.inputs;
      if (*schedule == ::fabric::Schedule::Temporal &&
          component.inputs.size() != 1)
        return invalid("contending switch component has no GrantPolicy");
    }
    components.push_back(std::move(component));
  }

  std::uint64_t temporalEntryCount = 0;
  std::uint64_t temporalTagWidth = 0;
  std::uint64_t temporalEntryWidth = 0;
  if (*schedule == ::fabric::Schedule::Temporal) {
    temporalEntryCount = fabric.switchRouteTableSize(sw);
    temporalTagWidth = inputEndpoints.front()->dataPath.tagWidthBits;
    temporalEntryWidth = 1 + temporalTagWidth + routes.size();
    if (temporalEntryCount == 0 || temporalTagWidth == 0 ||
        temporalEntryCount > UINT64_MAX / temporalEntryWidth ||
        temporalEntryCount * temporalEntryWidth != decoder->encodedBitCount)
      return invalid("Temporal switch direct carrier has the wrong shape");
  } else if (*schedule == ::fabric::Schedule::Spatial) {
    if (routes.size() != decoder->encodedBitCount)
      return invalid("Spatial switch direct carrier has the wrong shape");
  } else {
    return invalid("switch schedule is outside the closed domain");
  }

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  appendComponentPorts(builder, configurationAbi, transportLayout, *endpoints,
                       inputs, outputs, roundRobin);
  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_fabric_switch_" + std::to_string(sw.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value fieldSignal =
            decodeFieldSignal(bodyBuilder, location, accessor, *decoder);
        std::vector<std::vector<mlir::Value>> requestedRoute(
            inputCount,
            std::vector<mlir::Value>(
                outputCount, bitConstant(bodyBuilder, location, false)));
        for (const SwitchRoute &route : routes) {
          mlir::Value selected = bitConstant(bodyBuilder, location, false);
          if (*schedule == ::fabric::Schedule::Spatial) {
            selected = selectedBit(bodyBuilder, location, fieldSignal,
                                   route.configurationBit);
          } else {
            for (std::uint64_t entry = 0; entry != temporalEntryCount;
                 ++entry) {
              const std::uint64_t base = entry * temporalEntryWidth;
              mlir::Value valid =
                  selectedBit(bodyBuilder, location, fieldSignal, base);
              mlir::Value tag = circt::comb::ExtractOp::create(
                  bodyBuilder, location, fieldSignal, base + 1,
                  temporalTagWidth);
              mlir::Value matches = circt::comb::ICmpOp::create(
                  bodyBuilder, location, circt::comb::ICmpPredicate::eq, tag,
                  accessor.getInput(route.input->tag->getName()), true);
              mlir::Value crosspoint = selectedBit(
                  bodyBuilder, location, fieldSignal,
                  base + 1 + temporalTagWidth + route.configurationBit);
              selected = circt::comb::OrOp::create(
                  bodyBuilder, location, selected,
                  andValues(bodyBuilder, location,
                            {valid, matches, crosspoint}));
            }
          }
          requestedRoute[route.inputOrdinal][route.outputOrdinal] = selected;
        }

        std::vector<mlir::Value> requested(inputCount);
        for (unsigned input = 0; input != inputCount; ++input)
          requested[input] = andValues(
              bodyBuilder, location,
              {accessor.getInput(inputEndpoints[input]->valid.getName()),
               orValues(bodyBuilder, location, requestedRoute[input])});

        std::vector<mlir::Value> selectedInput(
            inputCount, bitConstant(bodyBuilder, location, false));
        std::optional<circt::BackedgeBuilder> backedges;
        if (roundRobin)
          backedges.emplace(bodyBuilder, location);
        for (const SwitchArbitrationComponent &component : components) {
          const auto deriveSelection = [&](llvm::ArrayRef<unsigned> order) {
            std::vector<mlir::Value> selected(
                inputCount, bitConstant(bodyBuilder, location, false));
            std::vector<mlir::Value> reserved(
                outputCount, bitConstant(bodyBuilder, location, false));
            for (unsigned input : order) {
              llvm::SmallVector<mlir::Value> conflicts;
              for (unsigned output : component.outputs)
                conflicts.push_back(andValues(
                    bodyBuilder, location,
                    {requestedRoute[input][output], reserved[output]}));
              selected[input] =
                  andValues(bodyBuilder, location,
                            {requested[input],
                             circt::comb::createOrFoldNot(
                                 bodyBuilder, location,
                                 orValues(bodyBuilder, location, conflicts))});
              for (unsigned output : component.outputs)
                reserved[output] = circt::comb::OrOp::create(
                    bodyBuilder, location, reserved[output],
                    andValues(
                        bodyBuilder, location,
                        {selected[input], requestedRoute[input][output]}));
            }
            return selected;
          };

          if (!component.roundRobinResetPosition) {
            std::vector<mlir::Value> selected =
                deriveSelection(component.requesterOrder);
            for (unsigned input : component.inputs)
              selectedInput[input] = selected[input];
            continue;
          }

          const unsigned cursorWidth =
              counterWidth(component.requesterOrder.size());
          circt::Backedge cursorNext =
              backedges->get(bodyBuilder.getIntegerType(cursorWidth));
          mlir::Value cursor = createRegister(
              bodyBuilder, location, cursorNext, accessor.getInput("clock"),
              accessor.getInput("reset"),
              llvm::APInt(cursorWidth, *component.roundRobinResetPosition),
              "round_robin_cursor_" + std::to_string(component.inputs.front()) +
                  "_reg",
              clockReset.asynchronousReset);
          mlir::Value nextCursor = cursor;
          for (unsigned start = 0; start != component.requesterOrder.size();
               ++start) {
            std::vector<unsigned> order;
            order.reserve(component.requesterOrder.size());
            for (unsigned offset = 0; offset != component.requesterOrder.size();
                 ++offset)
              order.push_back(
                  component.requesterOrder[(start + offset) %
                                           component.requesterOrder.size()]);
            std::vector<mlir::Value> selected = deriveSelection(order);
            mlir::Value cursorIs = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq, cursor,
                circt::hw::ConstantOp::create(bodyBuilder, location,
                                              llvm::APInt(cursorWidth, start)),
                true);
            mlir::Value candidateNext = circt::hw::ConstantOp::create(
                bodyBuilder, location, llvm::APInt(cursorWidth, start));
            for (unsigned offset = 0; offset != order.size(); ++offset) {
              const unsigned input = order[offset];
              selectedInput[input] = circt::comb::OrOp::create(
                  bodyBuilder, location, selectedInput[input],
                  andValues(bodyBuilder, location,
                            {cursorIs, selected[input]}));
              llvm::SmallVector<mlir::Value> allReady;
              for (unsigned output : component.outputs)
                allReady.push_back(circt::comb::OrOp::create(
                    bodyBuilder, location,
                    circt::comb::createOrFoldNot(bodyBuilder, location,
                                                 requestedRoute[input][output]),
                    accessor.getInput(
                        outputEndpoints[output]->ready.getName())));
              mlir::Value fire =
                  andValues(bodyBuilder, location,
                            {selected[input],
                             andValues(bodyBuilder, location, allReady)});
              const unsigned next = (start + offset + 1) % order.size();
              candidateNext = circt::comb::MuxOp::create(
                  bodyBuilder, location, fire,
                  circt::hw::ConstantOp::create(bodyBuilder, location,
                                                llvm::APInt(cursorWidth, next)),
                  candidateNext, true);
            }
            nextCursor =
                circt::comb::MuxOp::create(bodyBuilder, location, cursorIs,
                                           candidateNext, nextCursor, true);
          }
          cursorNext.setValue(nextCursor);
        }

        for (unsigned input = 0; input != inputCount; ++input) {
          llvm::SmallVector<mlir::Value> allReady;
          for (unsigned output = 0; output != outputCount; ++output)
            allReady.push_back(circt::comb::OrOp::create(
                bodyBuilder, location,
                circt::comb::createOrFoldNot(bodyBuilder, location,
                                             requestedRoute[input][output]),
                accessor.getInput(outputEndpoints[output]->ready.getName())));
          accessor.setOutput(
              inputEndpoints[input]->ready.getName(),
              andValues(bodyBuilder, location,
                        {selectedInput[input],
                         andValues(bodyBuilder, location, allReady)}));
        }

        for (unsigned output = 0; output != outputCount; ++output) {
          const EndpointPlan &outputEndpoint = *outputEndpoints[output];
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
          for (unsigned input = 0; input != inputCount; ++input) {
            const EndpointPlan &inputEndpoint = *inputEndpoints[input];
            mlir::Value selected = andValues(
                bodyBuilder, location,
                {selectedInput[input], requestedRoute[input][output]});
            llvm::SmallVector<mlir::Value> peerReady;
            for (unsigned peer = 0; peer != outputCount; ++peer) {
              if (peer == output)
                continue;
              peerReady.push_back(circt::comb::OrOp::create(
                  bodyBuilder, location,
                  circt::comb::createOrFoldNot(bodyBuilder, location,
                                               requestedRoute[input][peer]),
                  accessor.getInput(outputEndpoints[peer]->ready.getName())));
            }
            validTerms.push_back(andValues(
                bodyBuilder, location,
                {selected, andValues(bodyBuilder, location, peerReady)}));
            auto adapted = adaptForwardTransportSignals(
                bodyBuilder, location, inputEndpoint.dataPath,
                outputEndpoint.dataPath,
                ForwardTransportSignals{
                    accessor.getInput(inputEndpoint.valid.getName()),
                    inputEndpoint.data
                        ? std::optional<mlir::Value>{accessor.getInput(
                              inputEndpoint.data->getName())}
                        : std::nullopt,
                    inputEndpoint.tag
                        ? std::optional<mlir::Value>{accessor.getInput(
                              inputEndpoint.tag->getName())}
                        : std::nullopt});
            if (!adapted) {
              materializationError = llvm::toString(adapted.takeError());
              if (backedges)
                backedges->abandon();
              return;
            }
            if (outputEndpoint.data)
              data = circt::comb::MuxOp::create(bodyBuilder, location, selected,
                                                *adapted->payload, data, true);
            if (outputEndpoint.tag)
              tag = circt::comb::MuxOp::create(bodyBuilder, location, selected,
                                               *adapted->tag, tag, true);
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
                const ConfigurationTransportLayout &transportLayout,
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
  auto prepared =
      prepareFiniteField(spatialCore, field, configurationAbi, transportLayout);
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
  appendComponentPorts(builder, configurationAbi, transportLayout, *endpoints,
                       inputs, outputs, true);
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
                    const ConfigurationTransportLayout &transportLayout,
                    fabric::FabricBoundaryOccurrenceRef boundary) {
  auto canonical = findCanonicalEntityOperation(fabric, boundary.id());
  if (!canonical)
    return canonical.takeError();
  auto operation = mlir::dyn_cast<::fabric::BoundaryOp>(*canonical);
  if (!operation)
    return invalid("boundary occurrence entity does not name fabric.boundary");
  auto endpoints = deriveEndpointPlans(
      builder, fabric, fabric::FabricTransportEndpointOwnerRef::of(boundary));
  if (!endpoints)
    return endpoints.takeError();
  const fabric::FabricSemanticConfigFieldRef field{
      fabric::FabricConfigurationOwnerRef(
          fabric::FabricInventoryOwnerRef::of(boundary)),
      0};
  auto relation = fabric.semanticFieldRelation(
      field, *const_cast<mlir::Operation *>(fabric.canonicalOperation())
                  ->getContext());
  if (!relation)
    return relation.takeError();
  std::optional<FieldDecoderPlan> decoder;
  std::optional<llvm::APInt> finiteActiveCode;
  if (relation->kind() == fabric::FabricSemanticFieldRelationKind::Finite) {
    auto prepared = prepareFiniteField(spatialCore, field, configurationAbi,
                                       transportLayout);
    if (!prepared)
      return prepared.takeError();
    if (relation->finiteDomain().size() != 2)
      return invalid("boundary activation field has the wrong finite domain");
    auto active = physicalCode(*prepared->second,
                               relation->finiteDomain().back().bytes());
    if (!active)
      return active.takeError();
    decoder = std::move(prepared->first);
    finiteActiveCode = std::move(*active);
  } else if (relation->kind() ==
             fabric::FabricSemanticFieldRelationKind::Direct) {
    auto prepared = prepareFieldDecoder(spatialCore, field, configurationAbi,
                                        transportLayout);
    if (!prepared)
      return prepared.takeError();
    if (relation->directEncodedBitCount() != prepared->encodedBitCount)
      return invalid("boundary field is not its exact direct carrier");
    decoder = std::move(*prepared);
  } else {
    return invalid("boundary has no exact semantic configuration field");
  }

  const EndpointPlan *input =
      findEndpoint(*endpoints, fabric::FabricPortDirection::Input, 0);
  const EndpointPlan *secondInput =
      findEndpoint(*endpoints, fabric::FabricPortDirection::Input, 1);
  const EndpointPlan *output =
      findEndpoint(*endpoints, fabric::FabricPortDirection::Output, 0);
  const EndpointPlan *secondOutput =
      findEndpoint(*endpoints, fabric::FabricPortDirection::Output, 1);
  if (!input || !output)
    return invalid("boundary endpoint inventory is incomplete");

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  appendComponentPorts(builder, configurationAbi, transportLayout, *endpoints,
                       inputs, outputs);
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_fabric_boundary_" +
                            std::to_string(boundary.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value fieldSignal =
            decodeFieldSignal(bodyBuilder, location, accessor, *decoder);
        mlir::Value active =
            finiteActiveCode
                ? matchesCode(bodyBuilder, location, fieldSignal,
                              *finiteActiveCode)
                : selectedBit(bodyBuilder, location, fieldSignal, 0);

        switch (operation.getDirection()) {
        case ::fabric::BoundaryDirection::S2t: {
          if (!output->data || !output->tag || !input->data || secondOutput)
            return;
          accessor.setOutput(output->data->getName(),
                             accessor.getInput(input->data->getName()));
          if (secondInput) {
            if (!secondInput->data || secondInput->tag ||
                endpoints->size() != 3)
              return;
            accessor.setOutput(output->tag->getName(),
                               accessor.getInput(secondInput->data->getName()));
            mlir::Value dataValid = accessor.getInput(input->valid.getName());
            mlir::Value tagValid =
                accessor.getInput(secondInput->valid.getName());
            mlir::Value ready = accessor.getInput(output->ready.getName());
            accessor.setOutput(output->valid.getName(),
                               andValues(bodyBuilder, location,
                                         {active, dataValid, tagValid}));
            accessor.setOutput(
                input->ready.getName(),
                andValues(bodyBuilder, location, {active, ready, tagValid}));
            accessor.setOutput(
                secondInput->ready.getName(),
                andValues(bodyBuilder, location, {active, ready, dataValid}));
            break;
          }
          if (relation->kind() !=
                  fabric::FabricSemanticFieldRelationKind::Direct ||
              decoder->encodedBitCount != 1 + output->dataPath.tagWidthBits ||
              endpoints->size() != 2)
            return;
          accessor.setOutput(
              output->tag->getName(),
              circt::comb::ExtractOp::create(bodyBuilder, location, fieldSignal,
                                             1, output->dataPath.tagWidthBits));
          accessor.setOutput(
              output->valid.getName(),
              andValues(bodyBuilder, location,
                        {active, accessor.getInput(input->valid.getName())}));
          accessor.setOutput(
              input->ready.getName(),
              andValues(bodyBuilder, location,
                        {active, accessor.getInput(output->ready.getName())}));
          break;
        }
        case ::fabric::BoundaryDirection::T2t: {
          if (!input->data || !input->tag || !output->data || !output->tag ||
              secondInput || secondOutput || endpoints->size() != 2)
            return;
          const std::uint64_t inputTagWidth = input->dataPath.tagWidthBits;
          const std::uint64_t outputTagWidth = output->dataPath.tagWidthBits;
          const std::uint64_t rowCount =
              fabric.boundaryLookupTableSize(boundary);
          const std::uint64_t rowWidth = 1 + inputTagWidth + outputTagWidth;
          if (rowCount == 0 || rowCount > UINT64_MAX / rowWidth ||
              rowCount * rowWidth != decoder->encodedBitCount)
            return;
          mlir::Value match = bitConstant(bodyBuilder, location, false);
          mlir::Value remapped = circt::hw::ConstantOp::create(
              bodyBuilder, location, llvm::APInt(outputTagWidth, 0));
          for (std::uint64_t row = 0; row != rowCount; ++row) {
            const std::uint64_t base = row * rowWidth;
            mlir::Value valid =
                selectedBit(bodyBuilder, location, fieldSignal, base);
            mlir::Value sourceTag = circt::comb::ExtractOp::create(
                bodyBuilder, location, fieldSignal, base + 1, inputTagWidth);
            mlir::Value rowMatch = andValues(
                bodyBuilder, location,
                {valid, circt::comb::ICmpOp::create(
                            bodyBuilder, location,
                            circt::comb::ICmpPredicate::eq, sourceTag,
                            accessor.getInput(input->tag->getName()), true)});
            mlir::Value destinationTag = circt::comb::ExtractOp::create(
                bodyBuilder, location, fieldSignal, base + 1 + inputTagWidth,
                outputTagWidth);
            remapped =
                circt::comb::MuxOp::create(bodyBuilder, location, rowMatch,
                                           destinationTag, remapped, true);
            match = circt::comb::OrOp::create(bodyBuilder, location, match,
                                              rowMatch);
          }
          accessor.setOutput(output->data->getName(),
                             accessor.getInput(input->data->getName()));
          accessor.setOutput(output->tag->getName(), remapped);
          accessor.setOutput(
              output->valid.getName(),
              andValues(bodyBuilder, location,
                        {match, accessor.getInput(input->valid.getName())}));
          accessor.setOutput(
              input->ready.getName(),
              andValues(bodyBuilder, location,
                        {match, accessor.getInput(output->ready.getName())}));
          break;
        }
        case ::fabric::BoundaryDirection::T2s: {
          if (!input->data || !input->tag || !output->data || secondInput)
            return;
          accessor.setOutput(output->data->getName(),
                             accessor.getInput(input->data->getName()));
          if (secondOutput) {
            if (!secondOutput->data || secondOutput->tag ||
                endpoints->size() != 3)
              return;
            accessor.setOutput(secondOutput->data->getName(),
                               accessor.getInput(input->tag->getName()));
            mlir::Value inputValid = accessor.getInput(input->valid.getName());
            mlir::Value dataReady = accessor.getInput(output->ready.getName());
            mlir::Value tagReady =
                accessor.getInput(secondOutput->ready.getName());
            accessor.setOutput(output->valid.getName(),
                               andValues(bodyBuilder, location,
                                         {active, inputValid, tagReady}));
            accessor.setOutput(secondOutput->valid.getName(),
                               andValues(bodyBuilder, location,
                                         {active, inputValid, dataReady}));
            accessor.setOutput(input->ready.getName(),
                               andValues(bodyBuilder, location,
                                         {active, dataReady, tagReady}));
            break;
          }
          if (endpoints->size() != 2)
            return;
          accessor.setOutput(
              output->valid.getName(),
              andValues(bodyBuilder, location,
                        {active, accessor.getInput(input->valid.getName())}));
          accessor.setOutput(
              input->ready.getName(),
              andValues(bodyBuilder, location,
                        {active, accessor.getInput(output->ready.getName())}));
          break;
        }
        }
      });
  return BoundaryModule{boundary, module, std::move(*endpoints)};
}

} // namespace

llvm::Expected<std::vector<SwitchModule>>
buildSwitchModules(mlir::OpBuilder &builder, mlir::Location location,
                   fabric::SpatialCoreOccurrenceRef spatialCore,
                   const fabric::FabricArtifactView &fabric,
                   const ConfigurationABI &configurationAbi,
                   const ConfigurationTransportLayout &transportLayout,
                   const ClockResetPlan &clockReset) {
  std::vector<SwitchModule> result;
  result.reserve(fabric.switchOccurrences().size());
  for (fabric::FabricSwitchOccurrenceRef sw : fabric.switchOccurrences()) {
    auto module =
        buildSwitchModule(builder, location, spatialCore, fabric,
                          configurationAbi, transportLayout, clockReset, sw);
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
                 const ConfigurationTransportLayout &transportLayout,
                 const ClockResetPlan &clockReset) {
  std::vector<FifoModule> result;
  result.reserve(fabric.fifoOccurrences().size());
  for (fabric::FabricFifoOccurrenceRef fifo : fabric.fifoOccurrences()) {
    auto module =
        buildFifoModule(builder, location, spatialCore, fabric,
                        configurationAbi, transportLayout, clockReset, fifo);
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
                     const ConfigurationABI &configurationAbi,
                     const ConfigurationTransportLayout &transportLayout) {
  std::vector<BoundaryModule> result;
  result.reserve(fabric.boundaryOccurrences().size());
  for (fabric::FabricBoundaryOccurrenceRef boundary :
       fabric.boundaryOccurrences()) {
    auto module =
        buildBoundaryModule(builder, location, spatialCore, fabric,
                            configurationAbi, transportLayout, boundary);
    if (!module)
      return module.takeError();
    result.push_back(std::move(*module));
  }
  return result;
}

} // namespace loom::hardware::rtl::hierarchy

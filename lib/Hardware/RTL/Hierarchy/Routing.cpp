#include "Components.h"

#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/Identity/FabricRefImport.h"
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

llvm::Expected<ConfigurationBundlePlan> appendComponentPorts(
    mlir::OpBuilder &builder, llvm::ArrayRef<FieldDecoderPlan> decoders,
    llvm::ArrayRef<EndpointPlan> endpoints,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs,
    bool stateful = false,
    const FieldDecoderPlan *decodedConfiguration = nullptr) {
  auto configuration = deriveConfigurationBundlePlan(decoders);
  if (!configuration)
    return configuration.takeError();
  if (stateful) {
    const ConfigurationBundlePlan empty;
    appendClockResetAndConfigurationPorts(
        builder, decodedConfiguration ? empty : *configuration, inputs);
  } else if (!configuration->empty() && !decodedConfiguration) {
    inputs.push_back(circt::hw::PortInfo{
        {builder.getStringAttr(configurationBundlePortName),
         configurationBundleType(builder.getContext(), *configuration),
         circt::hw::ModulePort::Direction::Input}});
  }
  if (decodedConfiguration)
    inputs.push_back(circt::hw::PortInfo{
        {builder.getStringAttr(configurationValuePortName),
         builder.getIntegerType(
             static_cast<unsigned>(decodedConfiguration->encodedBitCount)),
         circt::hw::ModulePort::Direction::Input}});
  for (const EndpointPlan &endpoint : endpoints)
    appendEndpointPorts(inputs, outputs, endpoint);
  return std::move(*configuration);
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
mlir::Value incrementModulo(mlir::OpBuilder &builder, mlir::Location location,
                            mlir::Value value, std::uint64_t modulus);
void appendKeyU64(std::vector<std::uint8_t> &key, std::uint64_t value);
void appendKeyDataPath(std::vector<std::uint8_t> &key,
                       ::fabric::DataPathType path);

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
  auto configuration = appendComponentPorts(
      builder, llvm::ArrayRef<FieldDecoderPlan>(&*decoder, 1), *endpoints,
      inputs, outputs, roundRobin, &*decoder);
  if (!configuration)
    return configuration.takeError();
  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_fabric_switch_" + std::to_string(sw.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value fieldSignal =
            accessor.getInput(configurationValuePortName);
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
        std::vector<mlir::Value> configuredRequest(inputCount);
        for (unsigned input = 0; input != inputCount; ++input) {
          configuredRequest[input] =
              orValues(bodyBuilder, location, requestedRoute[input]);
          requested[input] = andValues(
              bodyBuilder, location,
              {accessor.getInput(inputEndpoints[input]->valid.getName()),
               configuredRequest[input]});
        }

        std::vector<mlir::Value> selectedInput(
            inputCount, bitConstant(bodyBuilder, location, false));
        std::vector<mlir::Value> admissibleInput(
            inputCount, bitConstant(bodyBuilder, location, false));
        std::optional<circt::BackedgeBuilder> backedges;
        if (roundRobin || *schedule == ::fabric::Schedule::Temporal)
          backedges.emplace(bodyBuilder, location);
        for (const SwitchArbitrationComponent &component : components) {
          struct ArbitrationSelection final {
            std::vector<mlir::Value> selected;
            std::vector<mlir::Value> admissible;
          };
          const auto deriveSelection = [&](llvm::ArrayRef<unsigned> order) {
            ArbitrationSelection result{
                std::vector<mlir::Value>(
                    inputCount, bitConstant(bodyBuilder, location, false)),
                std::vector<mlir::Value>(
                    inputCount, bitConstant(bodyBuilder, location, false))};
            std::vector<mlir::Value> reserved(
                outputCount, bitConstant(bodyBuilder, location, false));
            for (unsigned input : order) {
              llvm::SmallVector<mlir::Value> conflicts;
              for (unsigned output : component.outputs)
                conflicts.push_back(andValues(
                    bodyBuilder, location,
                    {requestedRoute[input][output], reserved[output]}));
              mlir::Value conflictFree = circt::comb::createOrFoldNot(
                  bodyBuilder, location,
                  orValues(bodyBuilder, location, conflicts));
              result.admissible[input] = andValues(
                  bodyBuilder, location,
                  {configuredRequest[input], conflictFree});
              result.selected[input] = andValues(
                  bodyBuilder, location, {requested[input], conflictFree});
              for (unsigned output : component.outputs)
                reserved[output] = circt::comb::OrOp::create(
                    bodyBuilder, location, reserved[output],
                    andValues(
                        bodyBuilder, location,
                        {result.selected[input],
                         requestedRoute[input][output]}));
            }
            return result;
          };

          if (!component.roundRobinResetPosition) {
            ArbitrationSelection selection =
                deriveSelection(component.requesterOrder);
            for (unsigned input : component.inputs) {
              selectedInput[input] = selection.selected[input];
              admissibleInput[input] = selection.admissible[input];
            }
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
            ArbitrationSelection selection = deriveSelection(order);
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
                            {cursorIs, selection.selected[input]}));
              admissibleInput[input] = circt::comb::OrOp::create(
                  bodyBuilder, location, admissibleInput[input],
                  andValues(bodyBuilder, location,
                            {cursorIs, selection.admissible[input]}));
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
                            {selection.selected[input],
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

        // A temporal switch presents the tag of idle candidates so the
        // downstream readiness of a row is observable before the token's
        // valid arrives: an atomic upstream fanout asserts valid on one output
        // only after every peer output is ready, so readiness must never wait
        // for valid. Valid requesters are presented by the grant policy; among
        // idle candidates whose selected outputs overlap, a free-running
        // rotation presents one at a time and never changes the grant order,
        // while candidates whose selected outputs no other candidate claims
        // are presented together. An input is ready only while it is
        // presented on every output it routes to, so a row that contends with
        // no other row is always presented and its readiness reflects only
        // its outputs' readiness, never the port's own valid. Only another
        // input's grant excludes a candidate: the grant is exclusive per
        // output, so an output this input holds is not held by another.
        std::vector<mlir::Value> presentedInput = selectedInput;
        if (*schedule == ::fabric::Schedule::Temporal) {
          std::vector<mlir::Value> held(
              outputCount, bitConstant(bodyBuilder, location, false));
          for (unsigned input = 0; input != inputCount; ++input)
            for (unsigned output = 0; output != outputCount; ++output)
              held[output] = circt::comb::OrOp::create(
                  bodyBuilder, location, held[output],
                  andValues(bodyBuilder, location,
                            {selectedInput[input],
                             requestedRoute[input][output]}));
          for (const SwitchArbitrationComponent &component : components) {
            llvm::SmallVector<mlir::Value> candidates;
            for (unsigned input : component.inputs) {
              llvm::SmallVector<mlir::Value> free;
              for (unsigned output : component.outputs)
                free.push_back(orValues(
                    bodyBuilder, location,
                    {circt::comb::createOrFoldNot(
                         bodyBuilder, location, requestedRoute[input][output]),
                     circt::comb::createOrFoldNot(bodyBuilder, location,
                                                  held[output]),
                     selectedInput[input]}));
              candidates.push_back(
                  andValues(bodyBuilder, location,
                            {configuredRequest[input],
                             andValues(bodyBuilder, location, free)}));
            }
            const unsigned candidateCount =
                static_cast<unsigned>(component.inputs.size());
            std::vector<mlir::Value> idleSelected(
                candidateCount, bitConstant(bodyBuilder, location, false));
            if (candidateCount == 1) {
              idleSelected.front() = candidates.front();
            } else {
              const unsigned pointerWidth = counterWidth(candidateCount);
              circt::Backedge pointerNext =
                  backedges->get(bodyBuilder.getIntegerType(pointerWidth));
              mlir::Value pointer = createRegister(
                  bodyBuilder, location, pointerNext,
                  accessor.getInput("clock"), accessor.getInput("reset"),
                  llvm::APInt(pointerWidth, 0),
                  "idle_presentation_" +
                      std::to_string(component.inputs.front()) + "_reg",
                  clockReset.asynchronousReset);
              pointerNext.setValue(incrementModulo(bodyBuilder, location,
                                                   pointer, candidateCount));
              for (unsigned start = 0; start != candidateCount; ++start) {
                mlir::Value pointerIs = circt::comb::ICmpOp::create(
                    bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                    pointer,
                    circt::hw::ConstantOp::create(
                        bodyBuilder, location,
                        llvm::APInt(pointerWidth, start)),
                    true);
                std::vector<mlir::Value> claimed(
                    component.outputs.size(),
                    bitConstant(bodyBuilder, location, false));
                for (unsigned offset = 0; offset != candidateCount; ++offset) {
                  const unsigned position = (start + offset) % candidateCount;
                  const unsigned input = component.inputs[position];
                  llvm::SmallVector<mlir::Value> contention;
                  for (auto [index, output] :
                       llvm::enumerate(component.outputs))
                    contention.push_back(andValues(
                        bodyBuilder, location,
                        {requestedRoute[input][output], claimed[index]}));
                  mlir::Value presented = andValues(
                      bodyBuilder, location,
                      {pointerIs, candidates[position],
                       circt::comb::createOrFoldNot(
                           bodyBuilder, location,
                           orValues(bodyBuilder, location, contention))});
                  idleSelected[position] = circt::comb::OrOp::create(
                      bodyBuilder, location, idleSelected[position], presented);
                  for (auto [index, output] :
                       llvm::enumerate(component.outputs))
                    claimed[index] = circt::comb::OrOp::create(
                        bodyBuilder, location, claimed[index],
                        andValues(bodyBuilder, location,
                                  {presented, requestedRoute[input][output]}));
                }
              }
            }
            for (auto [position, input] : llvm::enumerate(component.inputs))
              presentedInput[input] = circt::comb::OrOp::create(
                  bodyBuilder, location, selectedInput[input],
                  idleSelected[position]);
          }
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
                        {*schedule == ::fabric::Schedule::Temporal
                             ? presentedInput[input]
                             : admissibleInput[input],
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
            mlir::Value presented = andValues(
                bodyBuilder, location,
                {presentedInput[input], requestedRoute[input][output]});
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
              data = circt::comb::MuxOp::create(bodyBuilder, location,
                                                presented, *adapted->payload,
                                                data, true);
            if (outputEndpoint.tag)
              tag = circt::comb::MuxOp::create(bodyBuilder, location, presented,
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
  std::vector<std::uint8_t> implementationKey;
  appendKeyU64(implementationKey, 1);
  appendKeyU64(implementationKey, static_cast<std::uint32_t>(*schedule));
  appendKeyU64(implementationKey, decoder->encodedBitCount);
  appendKeyU64(implementationKey, clockReset.asynchronousReset);
  appendKeyU64(implementationKey, endpoints->size());
  for (const EndpointPlan &endpoint : *endpoints) {
    appendKeyU64(implementationKey,
                 static_cast<std::uint32_t>(endpoint.direction));
    appendKeyU64(implementationKey, endpoint.localOrdinal);
    appendKeyDataPath(implementationKey, endpoint.dataPath);
  }
  appendKeyU64(implementationKey, routes.size());
  for (const SwitchRoute &route : routes) {
    appendKeyU64(implementationKey, route.inputOrdinal);
    appendKeyU64(implementationKey, route.outputOrdinal);
    appendKeyU64(implementationKey, route.configurationBit);
  }
  appendKeyU64(implementationKey, components.size());
  for (const SwitchArbitrationComponent &component : components) {
    appendKeyU64(implementationKey, component.inputs.size());
    for (unsigned input : component.inputs)
      appendKeyU64(implementationKey, input);
    appendKeyU64(implementationKey, component.outputs.size());
    for (unsigned output : component.outputs)
      appendKeyU64(implementationKey, output);
    appendKeyU64(implementationKey, component.requesterOrder.size());
    for (unsigned requester : component.requesterOrder)
      appendKeyU64(implementationKey, requester);
    appendKeyU64(implementationKey,
                 component.roundRobinResetPosition.has_value());
    if (component.roundRobinResetPosition)
      appendKeyU64(implementationKey, *component.roundRobinResetPosition);
  }
  appendKeyU64(implementationKey, temporalEntryCount);
  appendKeyU64(implementationKey, temporalTagWidth);
  appendKeyU64(implementationKey, temporalEntryWidth);
  return SwitchModule{sw,
                      module,
                      std::move(*endpoints),
                      std::move(implementationKey),
                      std::move(*configuration),
                      std::move(*decoder)};
}

unsigned counterWidth(std::uint64_t bound) {
  return std::max(1U, llvm::Log2_64_Ceil(bound));
}

void appendKeyU64(std::vector<std::uint8_t> &key, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    key.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendKeyDataPath(std::vector<std::uint8_t> &key,
                       ::fabric::DataPathType path) {
  appendKeyU64(key, static_cast<std::uint32_t>(path.kind));
  appendKeyU64(key, path.payloadWidthBits);
  appendKeyU64(key, path.tagWidthBits);
}

void appendKeyApInt(std::vector<std::uint8_t> &key, const llvm::APInt &value) {
  appendKeyU64(key, value.getBitWidth());
  for (unsigned bit = 0; bit < value.getBitWidth(); bit += 8)
    key.push_back(static_cast<std::uint8_t>(
        value.extractBitsAsZExtValue(std::min(8U, value.getBitWidth() - bit),
                                     bit)));
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
  const bool virtualChannel =
      fabric.fifoQueueDiscipline(fifo).value_or(
          ::fabric::FifoQueueDiscipline::StrictFifo) ==
      ::fabric::FifoQueueDiscipline::PerTagVirtualChannel;
  if (virtualChannel &&
      (operation.getBypassable() || output->dataPath.tagWidthBits == 0))
    return invalid("virtual-channel FIFO must be tagged and non-bypassable");
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
  auto configuration = appendComponentPorts(
      builder, llvm::ArrayRef<FieldDecoderPlan>(&prepared->first, 1),
      *endpoints, inputs, outputs, true, &prepared->first);
  if (!configuration)
    return configuration.takeError();
  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_fabric_fifo_" + std::to_string(fifo.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value fieldSignal =
            accessor.getInput(configurationValuePortName);
        mlir::Value buffered =
            matchesCode(bodyBuilder, location, fieldSignal, *bufferedCode);
        mlir::Value bypass =
            bypassCode
                ? matchesCode(bodyBuilder, location, fieldSignal, *bypassCode)
                : bitConstant(bodyBuilder, location, false);
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        const unsigned pointerBits = counterWidth(depth);
        const unsigned occupancyBits = counterWidth(depth + 1);
        const unsigned tagWidthBits = output->dataPath.tagWidthBits;
        const auto integerConstant = [&](unsigned width, std::uint64_t value) {
          return circt::hw::ConstantOp::create(bodyBuilder, location,
                                               llvm::APInt(width, value));
        };
        circt::Backedge occupancyNext =
            backedges.get(bodyBuilder.getIntegerType(occupancyBits));
        // A strict FIFO moves a head pointer on dequeue. The virtual-channel
        // discipline keeps resident entries compacted toward slot zero and
        // instead moves a cursor over Physical Tag values; see below.
        std::optional<circt::Backedge> headNext;
        std::optional<circt::Backedge> tailNext;
        mlir::Value head;
        mlir::Value tail;
        if (!virtualChannel) {
          headNext = backedges.get(bodyBuilder.getIntegerType(pointerBits));
          tailNext = backedges.get(bodyBuilder.getIntegerType(pointerBits));
          head = createRegister(bodyBuilder, location, *headNext,
                                accessor.getInput("clock"),
                                accessor.getInput("reset"),
                                llvm::APInt(pointerBits, 0), "head_reg",
                                clockReset.asynchronousReset);
          tail = createRegister(bodyBuilder, location, *tailNext,
                                accessor.getInput("clock"),
                                accessor.getInput("reset"),
                                llvm::APInt(pointerBits, 0), "tail_reg",
                                clockReset.asynchronousReset);
        }
        mlir::Value occupancy = createRegister(
            bodyBuilder, location, occupancyNext, accessor.getInput("clock"),
            accessor.getInput("reset"), llvm::APInt(occupancyBits, 0),
            "occupancy_reg", clockReset.asynchronousReset);
        std::optional<circt::Backedge> offerCursorNext;
        mlir::Value offerCursor;
        if (virtualChannel) {
          offerCursorNext =
              backedges.get(bodyBuilder.getIntegerType(tagWidthBits));
          offerCursor = createRegister(
              bodyBuilder, location, *offerCursorNext, accessor.getInput("clock"),
              accessor.getInput("reset"), llvm::APInt(tagWidthBits, 0),
              "offer_cursor_reg", clockReset.asynchronousReset);
        }
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
        // The virtual-channel discipline presents the head of exactly one
        // non-empty channel per cycle. Resident entries occupy slots
        // [0, occupancy), so slot order is arrival order. Minimizing the
        // wrapped distance (tag - cursor) over occupied slots selects the
        // smallest resident tag value at or after the cursor and wraps once
        // to the lowest resident value. A balanced stable tournament keeps
        // the arrival-oldest slot of that channel on distance ties.
        mlir::Value selectedSlot;
        if (virtualChannel) {
          struct OfferCandidate final {
            mlir::Value valid;
            mlir::Value distance;
            mlir::Value slot;
          };
          std::vector<OfferCandidate> level;
          level.reserve(depth);
          for (std::uint64_t slot = 0; slot != depth; ++slot) {
            mlir::Value occupied = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::ult,
                integerConstant(occupancyBits, slot), occupancy, true);
            mlir::Value distance = circt::comb::SubOp::create(
                bodyBuilder, location, tagBank.current[slot], offerCursor, true);
            level.push_back({occupied, distance,
                             integerConstant(pointerBits, slot)});
          }
          while (level.size() != 1) {
            std::vector<OfferCandidate> next;
            next.reserve((level.size() + 1) / 2);
            for (std::size_t index = 0; index < level.size(); index += 2) {
              if (index + 1 == level.size()) {
                next.push_back(level[index]);
                continue;
              }
              const OfferCandidate &older = level[index];
              const OfferCandidate &newer = level[index + 1];
              mlir::Value nearer = circt::comb::ICmpOp::create(
                  bodyBuilder, location, circt::comb::ICmpPredicate::ult,
                  newer.distance, older.distance, true);
              mlir::Value newerWins = andValues(
                  bodyBuilder, location,
                  {newer.valid,
                   orValues(bodyBuilder, location,
                            {circt::comb::createOrFoldNot(
                                 bodyBuilder, location, older.valid),
                             nearer})});
              next.push_back(
                  {orValues(bodyBuilder, location,
                            {older.valid, newer.valid}),
                   circt::comb::MuxOp::create(
                       bodyBuilder, location, newerWins, newer.distance,
                       older.distance, true),
                   circt::comb::MuxOp::create(bodyBuilder, location, newerWins,
                                              newer.slot, older.slot, true)});
            }
            level = std::move(next);
          }
          selectedSlot = level.front().slot;
        }
        // A grant removes the presented slot. In the virtual-channel
        // discipline the hole closes toward the tail: every slot at or after
        // the granted slot takes its successor's content, and an enqueue in
        // the same cycle lands at the post-dequeue append position.
        const auto writeBank = [&](StorageBank &bank,
                                   std::optional<mlir::Value> source,
                                   mlir::Value appendPosition,
                                   mlir::Value grantedSlot) {
          if (bank.width == 0)
            return;
          if (!source) {
            materializationError = "FIFO storage source is absent";
            return;
          }
          const unsigned positionBits = mlir::cast<mlir::IntegerType>(
                                            appendPosition.getType())
                                            .getWidth();
          for (std::uint64_t slot = 0; slot < depth; ++slot) {
            mlir::Value slotValue =
                integerConstant(positionBits, slot);
            mlir::Value appendHere = andValues(
                bodyBuilder, location,
                {enqueue, circt::comb::ICmpOp::create(
                              bodyBuilder, location,
                              circt::comb::ICmpPredicate::eq, appendPosition,
                              slotValue, true)});
            mlir::Value next = bank.current[slot];
            if (grantedSlot) {
              mlir::Value shifts = circt::comb::ICmpOp::create(
                  bodyBuilder, location, circt::comb::ICmpPredicate::ule,
                  grantedSlot, integerConstant(pointerBits, slot), true);
              mlir::Value successor = slot + 1 != depth
                                          ? bank.current[slot + 1]
                                          : bank.current[slot];
              next = circt::comb::MuxOp::create(
                  bodyBuilder, location,
                  andValues(bodyBuilder, location, {dequeue, shifts}), successor,
                  next, true);
            }
            bank.next[slot].setValue(circt::comb::MuxOp::create(
                bodyBuilder, location, appendHere, *source, next, true));
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
        mlir::Value readPointer = head;
        mlir::Value appendPosition = tail;
        mlir::Value grantedSlot;
        if (virtualChannel) {
          readPointer = selectedSlot;
          grantedSlot = selectedSlot;
          // An enqueue in a dequeue cycle lands at the position the closing
          // hole leaves behind, one below the pre-dequeue occupancy.
          mlir::Value decrementedOccupancy = circt::comb::SubOp::create(
              bodyBuilder, location, occupancy,
              integerConstant(occupancyBits, 1), true);
          appendPosition = circt::comb::MuxOp::create(
              bodyBuilder, location, dequeue, decrementedOccupancy, occupancy,
              true);
        }
        writeBank(dataBank, adaptedInput->payload, appendPosition,
                  grantedSlot);
        writeBank(tagBank, adaptedInput->tag, appendPosition, grantedSlot);
        if (materializationError) {
          backedges.abandon();
          return;
        }
        const auto readBank = [&](const StorageBank &bank,
                                  mlir::Value pointer) -> mlir::Value {
          if (bank.width == 0)
            return {};
          llvm::SmallVector<mlir::Value> highToLow;
          highToLow.reserve(bank.current.size());
          for (mlir::Value value : llvm::reverse(bank.current))
            highToLow.push_back(value);
          mlir::Value entries =
              circt::hw::ArrayCreateOp::create(bodyBuilder, location,
                                               highToLow);
          return circt::hw::ArrayGetOp::create(bodyBuilder, location, entries,
                                               pointer);
        };
        mlir::Value bufferedData = readBank(dataBank, readPointer);
        mlir::Value bufferedTag = readBank(tagBank, readPointer);
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

        if (!virtualChannel) {
          mlir::Value incrementHead =
              incrementModulo(bodyBuilder, location, head, depth);
          mlir::Value incrementTail =
              incrementModulo(bodyBuilder, location, tail, depth);
          headNext->setValue(circt::comb::MuxOp::create(
              bodyBuilder, location, dequeue, incrementHead, head, true));
          tailNext->setValue(circt::comb::MuxOp::create(
              bodyBuilder, location, enqueue, incrementTail, tail, true));
        } else {
          // A grant and a refused offer (valid && !ready) share one cursor
          // rule: move past the presented channel so the next cycle presents
          // the next non-empty channel in canonical ascending tag order. The
          // tag-width add wraps to the zero value past the highest tag value.
          mlir::Value presentedRefused = andValues(
              bodyBuilder, location,
              {bufferedOutputValid,
               circt::comb::createOrFoldNot(bodyBuilder, location,
                                            accessor.getInput(
                                                output->ready.getName()))});
          mlir::Value cursorAdvances =
              orValues(bodyBuilder, location, {dequeue, presentedRefused});
          mlir::Value successor = circt::comb::AddOp::create(
              bodyBuilder, location, bufferedTag,
              integerConstant(tagWidthBits, 1), true);
          offerCursorNext->setValue(circt::comb::MuxOp::create(
              bodyBuilder, location, cursorAdvances, successor, offerCursor,
              true));
        }
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
  std::vector<std::uint8_t> implementationKey;
  appendKeyU64(implementationKey, depth);
  appendKeyU64(implementationKey, operation.getBypassable());
  appendKeyU64(implementationKey, virtualChannel);
  appendKeyU64(implementationKey, clockReset.asynchronousReset);
  appendKeyDataPath(implementationKey, input->dataPath);
  appendKeyDataPath(implementationKey, output->dataPath);
  appendKeyApInt(implementationKey, *bufferedCode);
  appendKeyU64(implementationKey, bypassCode.has_value());
  if (bypassCode)
    appendKeyApInt(implementationKey, *bypassCode);
  return FifoModule{fifo,
                    module,
                    std::move(*endpoints),
                    std::move(implementationKey),
                    std::move(*configuration),
                    std::move(prepared->first)};
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
  auto configuration = appendComponentPorts(
      builder, llvm::ArrayRef<FieldDecoderPlan>(&*decoder, 1), *endpoints,
      inputs, outputs);
  if (!configuration)
    return configuration.takeError();
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_fabric_boundary_" +
                            std::to_string(boundary.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        ConfigurationBundleSignals configurationValues =
            configurationBundleSignals(accessor, *configuration);
        mlir::Value fieldSignal = decodeFieldSignal(
            bodyBuilder, location, configurationValues, *decoder);
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
  return BoundaryModule{boundary, module, std::move(*endpoints),
                        std::move(*configuration)};
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
  std::map<std::vector<std::uint8_t>, circt::hw::HWModuleOp> definitions;
  for (fabric::FabricSwitchOccurrenceRef sw : fabric.switchOccurrences()) {
    auto module =
        buildSwitchModule(builder, location, spatialCore, fabric,
                          configurationAbi, transportLayout, clockReset, sw);
    if (!module)
      return module.takeError();
    if (llvm::Error error = verifyConfigurationValuePort(
            module->module, module->configurationDecoder))
      return std::move(error);
    auto definition = definitions.find(module->implementationKey);
    if (definition == definitions.end()) {
      definitions.emplace(module->implementationKey, module->module);
    } else {
      module->module.erase();
      module->module = definition->second;
      if (llvm::Error error = verifyConfigurationValuePort(
              module->module, module->configurationDecoder))
        return std::move(error);
    }
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
  std::map<std::vector<std::uint8_t>, circt::hw::HWModuleOp> definitions;
  for (fabric::FabricFifoOccurrenceRef fifo : fabric.fifoOccurrences()) {
    auto module =
        buildFifoModule(builder, location, spatialCore, fabric,
                        configurationAbi, transportLayout, clockReset, fifo);
    if (!module)
      return module.takeError();
    if (llvm::Error error = verifyConfigurationValuePort(
            module->module, module->configurationDecoder))
      return std::move(error);
    auto definition = definitions.find(module->implementationKey);
    if (definition == definitions.end()) {
      definitions.emplace(module->implementationKey, module->module);
    } else {
      module->module.erase();
      module->module = definition->second;
      if (llvm::Error error = verifyConfigurationValuePort(
              module->module, module->configurationDecoder))
        return std::move(error);
    }
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

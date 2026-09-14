#include "Arbitration.h"
#include "Components.h"

#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/IR/SwitchResourceContract.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
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

  std::vector<std::vector<std::uint32_t>> sourcesByOutput(outputCount);
  for (const SwitchRoute &route : routes)
    sourcesByOutput[route.outputOrdinal].push_back(route.inputOrdinal);
  const ::fabric::ResourceContract *resourceContract =
      fabric.resourceContract(fabric::FabricInventoryOwnerRef::of(sw));
  if (!resourceContract)
    return invalid("switch has no finalized ResourceContract");
  auto arbitration = ::fabric::deriveSwitchArbitrationComponents(
      *schedule, inputCount, outputCount, sourcesByOutput, *resourceContract);
  if (!arbitration)
    return arbitration.takeError();
  const llvm::ArrayRef<::fabric::SwitchArbitrationComponent> components =
      *arbitration;
  const bool consumesClockAndReset = resourceContract->stateCount() != 0;

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
      inputs, outputs, consumesClockAndReset, &*decoder);
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
        std::vector<std::vector<mlir::Value>> temporalEntryMatches;
        if (*schedule == ::fabric::Schedule::Temporal) {
          temporalEntryMatches.resize(temporalEntryCount,
                                      std::vector<mlir::Value>(inputCount));
          for (std::uint64_t entry = 0; entry != temporalEntryCount; ++entry) {
            const std::uint64_t base = entry * temporalEntryWidth;
            mlir::Value valid =
                selectedBit(bodyBuilder, location, fieldSignal, base);
            mlir::Value tag = circt::comb::ExtractOp::create(
                bodyBuilder, location, fieldSignal, base + 1, temporalTagWidth);
            for (unsigned input = 0; input != inputCount; ++input)
              temporalEntryMatches[entry][input] = andValues(
                  bodyBuilder, location,
                  {valid,
                   circt::comb::ICmpOp::create(
                       bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                       tag,
                       accessor.getInput(inputEndpoints[input]->tag->getName()),
                       true)});
          }
        }
        for (const SwitchRoute &route : routes) {
          mlir::Value selected = bitConstant(bodyBuilder, location, false);
          if (*schedule == ::fabric::Schedule::Spatial) {
            selected = selectedBit(bodyBuilder, location, fieldSignal,
                                   route.configurationBit);
          } else {
            for (std::uint64_t entry = 0; entry != temporalEntryCount;
                 ++entry) {
              const std::uint64_t base = entry * temporalEntryWidth;
              mlir::Value crosspoint = selectedBit(
                  bodyBuilder, location, fieldSignal,
                  base + 1 + temporalTagWidth + route.configurationBit);
              selected = circt::comb::OrOp::create(
                  bodyBuilder, location, selected,
                  andValues(bodyBuilder, location,
                            {temporalEntryMatches[entry][route.inputOrdinal],
                             crosspoint}));
            }
          }
          requestedRoute[route.inputOrdinal][route.outputOrdinal] = selected;
        }

        std::vector<mlir::Value> requestedRouteMask(inputCount);
        std::vector<mlir::Value> routeReady(inputCount);
        std::vector<std::vector<mlir::Value>> peerRouteReady(
            inputCount, std::vector<mlir::Value>(outputCount));
        std::vector<mlir::Value> requested(inputCount);
        std::vector<mlir::Value> configuredRequest(inputCount);
        mlir::Value emptyRouteMask = circt::hw::ConstantOp::create(
            bodyBuilder, location, llvm::APInt(outputCount, 0));
        for (unsigned input = 0; input != inputCount; ++input) {
          requestedRouteMask[input] =
              packBits(bodyBuilder, location, requestedRoute[input]);
          configuredRequest[input] = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::ne,
              requestedRouteMask[input], emptyRouteMask, true);
          requested[input] = andValues(
              bodyBuilder, location,
              {accessor.getInput(inputEndpoints[input]->valid.getName()),
               configuredRequest[input]});
          std::vector<mlir::Value> readyTerms;
          readyTerms.reserve(outputCount);
          for (unsigned output = 0; output != outputCount; ++output)
            readyTerms.push_back(circt::comb::OrOp::create(
                bodyBuilder, location,
                circt::comb::createOrFoldNot(bodyBuilder, location,
                                             requestedRoute[input][output]),
                accessor.getInput(outputEndpoints[output]->ready.getName())));
          std::vector<mlir::Value> prefixReady(
              outputCount + 1, bitConstant(bodyBuilder, location, true));
          std::vector<mlir::Value> suffixReady(
              outputCount + 1, bitConstant(bodyBuilder, location, true));
          for (unsigned output = 0; output != outputCount; ++output)
            prefixReady[output + 1] =
                andValues(bodyBuilder, location,
                          {prefixReady[output], readyTerms[output]});
          for (unsigned output = outputCount; output != 0; --output)
            suffixReady[output - 1] =
                andValues(bodyBuilder, location,
                          {readyTerms[output - 1], suffixReady[output]});
          routeReady[input] = andValues(bodyBuilder, location, readyTerms);
          for (unsigned output = 0; output != outputCount; ++output)
            peerRouteReady[input][output] =
                andValues(bodyBuilder, location,
                          {prefixReady[output], suffixReady[output + 1]});
        }

        // `grantedInput` is the input's permission to transfer, ignoring its
        // own valid; `transferInput` is that permission taken by a valid
        // token. A Temporal switch decides the permission from registered
        // state alone: each multi-input physical component's grant pointer
        // names the one input that may transfer, so no ready of the switch is
        // a function of any valid of the same cycle and at most one input of
        // a component transfers per cycle. A Spatial switch owns no runtime
        // arbiter: its selected rows are statically capacity-closed, and the
        // conflict-free scan in physical requester order only withholds a
        // malformed configuration's second claim on one output.
        std::vector<mlir::Value> grantedInput(
            inputCount, bitConstant(bodyBuilder, location, false));
        std::vector<mlir::Value> transferInput(
            inputCount, bitConstant(bodyBuilder, location, false));
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        for (const ::fabric::SwitchArbitrationComponent &component :
             components) {
          if (*schedule == ::fabric::Schedule::Temporal) {
            RegisteredGrant grant = makeRegisteredGrant(
                bodyBuilder, location, backedges,
                component.requesterOrder.size(),
                component.roundRobinResetPosition.has_value(),
                component.roundRobinResetPosition.value_or(0),
                accessor.getInput("clock"), accessor.getInput("reset"),
                "switch_grant_" + std::to_string(component.inputs.front()),
                clockReset);
            // A requester of the component is an input whose resident row
            // selects an output and whose token has arrived; the pointer's
            // next value observes exactly that.
            llvm::SmallVector<mlir::Value> componentRequest;
            llvm::SmallVector<mlir::Value> fired;
            componentRequest.reserve(component.requesterOrder.size());
            fired.reserve(component.requesterOrder.size());
            for (auto [position, input] :
                 llvm::enumerate(component.requesterOrder)) {
              grantedInput[input] = andValues(
                  bodyBuilder, location,
                  {grant.pointed[position], configuredRequest[input]});
              transferInput[input] =
                  andValues(bodyBuilder, location,
                            {grant.pointed[position], requested[input]});
              componentRequest.push_back(requested[input]);
              fired.push_back(andValues(
                  bodyBuilder, location,
                  {transferInput[input], routeReady[input]}));
            }
            advanceRegisteredGrant(
                bodyBuilder, location, grant,
                packBits(bodyBuilder, location, componentRequest),
                orValues(bodyBuilder, location, fired));
            continue;
          }

          mlir::Value reserved = emptyRouteMask;
          for (unsigned input : component.requesterOrder) {
            mlir::Value conflicts = circt::comb::AndOp::create(
                bodyBuilder, location, requestedRouteMask[input], reserved,
                true);
            mlir::Value conflictFree = circt::comb::createOrFoldNot(
                bodyBuilder, location,
                circt::comb::ICmpOp::create(
                    bodyBuilder, location, circt::comb::ICmpPredicate::ne,
                    conflicts, emptyRouteMask, true));
            grantedInput[input] =
                andValues(bodyBuilder, location,
                          {configuredRequest[input], conflictFree});
            transferInput[input] = andValues(
                bodyBuilder, location, {requested[input], conflictFree});
            reserved = circt::comb::OrOp::create(
                bodyBuilder, location, reserved,
                circt::comb::MuxOp::create(
                    bodyBuilder, location, transferInput[input],
                    requestedRouteMask[input], emptyRouteMask, true),
                true);
          }
        }

        // A Temporal switch drives the granted input's payload and tag on the
        // outputs its resident row selects whether or not its valid has
        // arrived, so a downstream row's capacity term, and the readiness that
        // term gates, stay observable before the token does. A Spatial switch
        // drives the payload of the input that actually transfers.
        const std::vector<mlir::Value> &payloadSource =
            *schedule == ::fabric::Schedule::Temporal ? grantedInput
                                                      : transferInput;

        for (unsigned input = 0; input != inputCount; ++input) {
          accessor.setOutput(
              inputEndpoints[input]->ready.getName(),
              andValues(bodyBuilder, location,
                        {grantedInput[input], routeReady[input]}));
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
            mlir::Value transferred = andValues(
                bodyBuilder, location,
                {transferInput[input], requestedRoute[input][output]});
            mlir::Value sourced = andValues(
                bodyBuilder, location,
                {payloadSource[input], requestedRoute[input][output]});
            validTerms.push_back(
                andValues(bodyBuilder, location,
                          {transferred, peerRouteReady[input][output]}));
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
              backedges.abandon();
              return;
            }
            if (outputEndpoint.data)
              data = circt::comb::MuxOp::create(
                  bodyBuilder, location, sourced, *adapted->payload, data,
                  true);
            if (outputEndpoint.tag)
              tag = circt::comb::MuxOp::create(bodyBuilder, location, sourced,
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
  // The switch implementation identity; 4 is the registered grant.
  appendKeyU64(implementationKey, 4);
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
  for (const ::fabric::SwitchArbitrationComponent &component : components) {
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
  // The channels the shared pool guarantees one slot each; one is the
  // undeclared default and guarantees nothing beyond the first resident.
  const std::uint32_t reservedChannels =
      virtualChannel ? fabric.fifoReservedChannels(fifo).value_or(1) : 0;
  if (reservedChannels > depth)
    return invalid("FIFO reserves more channels than it has slots");
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
        const unsigned pointerBits = indexWidth(depth);
        const unsigned occupancyBits = indexWidth(depth + 1);
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
        // Input ready is registered cycle-start capacity. A virtual-channel
        // pool with reserved channels keeps one free slot for every
        // guaranteed channel that is neither resident nor the arriving one,
        // so a channel that runs ahead cannot starve an absent channel.
        mlir::Value admissible =
            circt::comb::createOrFoldNot(bodyBuilder, location, full);
        if (virtualChannel && reservedChannels > 1 && tagBank.width != 0 &&
            adaptedInput->tag) {
          llvm::SmallVector<mlir::Value, 8> occupied;
          for (std::uint64_t slot = 0; slot != depth; ++slot)
            occupied.push_back(circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::ult,
                integerConstant(occupancyBits, slot), occupancy, true));
          const auto sameTag = [&](mlir::Value lhs, mlir::Value rhs) {
            return circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq, lhs,
                rhs, true);
          };
          // A slot heads a distinct channel when no older occupied slot
          // carries its tag; the distinct heads count the resident channels.
          llvm::SmallVector<mlir::Value, 8> channelHeads;
          llvm::SmallVector<mlir::Value, 8> inputMatches;
          for (std::uint64_t slot = 0; slot != depth; ++slot) {
            llvm::SmallVector<mlir::Value, 8> olderSameTag;
            for (std::uint64_t older = 0; older != slot; ++older)
              olderSameTag.push_back(andValues(
                  bodyBuilder, location,
                  {occupied[older],
                   sameTag(tagBank.current[older], tagBank.current[slot])}));
            mlir::Value head = andValues(
                bodyBuilder, location,
                {occupied[slot],
                 circt::comb::createOrFoldNot(
                     bodyBuilder, location,
                     orValues(bodyBuilder, location, olderSameTag))});
            channelHeads.push_back(circt::comb::MuxOp::create(
                bodyBuilder, location, head, integerConstant(occupancyBits, 1),
                integerConstant(occupancyBits, 0), true));
            inputMatches.push_back(
                andValues(bodyBuilder, location,
                          {occupied[slot],
                           sameTag(tagBank.current[slot], *adaptedInput->tag)}));
          }
          // A balanced adder tree keeps the count logarithmic in the depth.
          std::vector<mlir::Value> countLevel(channelHeads.begin(),
                                              channelHeads.end());
          while (countLevel.size() != 1) {
            std::vector<mlir::Value> next;
            next.reserve((countLevel.size() + 1) / 2);
            for (std::size_t index = 0; index < countLevel.size(); index += 2) {
              if (index + 1 == countLevel.size())
                next.push_back(countLevel[index]);
              else
                next.push_back(circt::comb::AddOp::create(
                    bodyBuilder, location, countLevel[index],
                    countLevel[index + 1], true));
            }
            countLevel = std::move(next);
          }
          mlir::Value residentCount = countLevel.front();
          mlir::Value inputResident =
              orValues(bodyBuilder, location, inputMatches);
          mlir::Value claimed = circt::comb::AddOp::create(
              bodyBuilder, location, residentCount,
              circt::comb::MuxOp::create(bodyBuilder, location, inputResident,
                                         integerConstant(occupancyBits, 0),
                                         integerConstant(occupancyBits, 1),
                                         true),
              true);
          mlir::Value guaranteed =
              integerConstant(occupancyBits, reservedChannels);
          mlir::Value belowGuarantee = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::ult, claimed,
              guaranteed, true);
          mlir::Value reservedForOthers = circt::comb::MuxOp::create(
              bodyBuilder, location, belowGuarantee,
              circt::comb::SubOp::create(bodyBuilder, location, guaranteed,
                                         claimed, true),
              integerConstant(occupancyBits, 0), true);
          mlir::Value free = circt::comb::SubOp::create(
              bodyBuilder, location, fullOccupancy, occupancy, true);
          admissible = circt::comb::ICmpOp::create(
              bodyBuilder, location, circt::comb::ICmpPredicate::ugt, free,
              reservedForOthers, true);
        }
        mlir::Value bufferedInputReady =
            andValues(bodyBuilder, location, {buffered, admissible});
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
  appendKeyU64(implementationKey, reservedChannels);
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

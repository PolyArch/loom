#include "Arbitration.h"
#include "Components.h"

#include "Common/InvocationDiagnosticLog.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"
#include "Hardware/RTL/MaterializationDiagnostics.h"
#include "Hardware/RTL/OperationLeaf.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"

#include <map>
#include <optional>
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

const OperationShellModule *
findOperationShell(llvm::ArrayRef<OperationShellModule> shells,
                   fabric::FabricFuOccurrenceNodeRef operation) {
  const OperationShellModule *result = nullptr;
  for (const OperationShellModule &shell : shells)
    if (shell.operation.localOccurrence == operation) {
      if (result)
        return nullptr;
      result = &shell;
    }
  return result;
}

const FuModule *findFuModule(llvm::ArrayRef<FuModule> modules,
                             fabric::FabricFuOccurrenceRef fu) {
  const FuModule *result = nullptr;
  for (const FuModule &module : modules)
    if (module.reference == fu) {
      if (result)
        return nullptr;
      result = &module;
    }
  return result;
}

const OperationEndpointPlan *
findOperationEndpoint(const OperationShellModule &shell,
                      fabric::FabricPortDirection direction,
                      fabric::FabricOrdinal ordinal) {
  const OperationEndpointPlan *result = nullptr;
  for (const OperationEndpointPlan &endpoint : shell.endpoints)
    if (endpoint.direction == direction && endpoint.ordinal == ordinal) {
      if (result)
        return nullptr;
      result = &endpoint;
    }
  return result;
}

std::string fuTemplateEndpointKey(
    const fabric::FabricFuCapabilityTemplateEndpointRef &endpoint) {
  std::vector<std::uint8_t> bytes;
  if (endpoint.kind() ==
      fabric::FabricFuCapabilityTemplateEndpointKind::BoundaryPort) {
    bytes = fabric::canonicalFabricBytes(
        std::get<fabric::FabricFuTemplatePortRef>(endpoint.payload));
    bytes.insert(bytes.begin(), 0);
  } else {
    bytes = fabric::canonicalFabricBytes(
        std::get<fabric::FabricFuNodePortRef>(endpoint.payload));
    bytes.insert(bytes.begin(), 1);
  }
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Error
addCommonInstanceInputs(mlir::OpBuilder &builder, mlir::Location location,
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

mlir::Value zeroData(mlir::OpBuilder &builder, mlir::Location location,
                     unsigned width) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, 0));
}

std::string outputContextPortName(const EndpointPlan &endpoint) {
  return "output_" + std::to_string(endpoint.localOrdinal) + "_context";
}

/// The enclosing Temporal PE asserts this input while it presents the FU
/// output to one of its ports or register FIFOs, accepted or refused.

mlir::Value contextEquals(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Value context, std::uint64_t ordinal) {
  const unsigned width =
      mlir::cast<mlir::IntegerType>(context.getType()).getWidth();
  return circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::eq, context,
      circt::hw::ConstantOp::create(builder, location,
                                    llvm::APInt(width, ordinal)),
      true);
}

mlir::Value selectContextValue(mlir::OpBuilder &builder,
                               mlir::Location location, mlir::Value context,
                               llvm::ArrayRef<mlir::Value> values) {
  assert(!values.empty() && "context selection requires a value domain");
  mlir::Value selected = values.front();
  for (std::uint64_t ordinal = 1; ordinal < values.size(); ++ordinal)
    selected = circt::comb::MuxOp::create(
        builder, location, contextEquals(builder, location, context, ordinal),
        values[ordinal], selected, true);
  return selected;
}

llvm::Expected<FuModule>
buildFuModule(mlir::OpBuilder &builder, mlir::Location location,
              fabric::SpatialCoreOccurrenceRef spatialCore,
              const fabric::FabricArtifactView &fabric,
              const ConfigurationABI &configurationAbi,
              const ConfigurationTransportLayout &transportLayout,
              llvm::ArrayRef<OperationShellModule> operationShells,
              fabric::FabricFuOccurrenceRef fu,
              const ClockResetPlan &clockReset) {
  auto endpoints = deriveEndpointPlans(
      builder, fabric, fabric::FabricTransportEndpointOwnerRef::of(fu));
  if (!endpoints)
    return endpoints.takeError();
  const auto definition = fabric.fuTemplateOf(fu);
  if (!definition)
    return invalid("FU occurrence has no definition template");
  const auto templates = fabric.fuCapabilityTemplates(*definition);
  if (templates.empty())
    return invalid("FU definition has no capability template");

  const fabric::FabricSemanticConfigFieldRef field{
      fabric::FabricConfigurationOwnerRef(
          fabric::FabricInventoryOwnerRef::of(fu)),
      0};
  const auto parentPe = fabric.parentPeOf(fu);
  if (!parentPe)
    return invalid("FU occurrence has no parent PE");
  const bool temporal =
      fabric.peSchedule(*parentPe) == ::fabric::Schedule::Temporal;
  const std::uint64_t contextCount =
      temporal ? fabric.peResidentContextCount(*parentPe) : 1;
  if (contextCount == 0 || contextCount > UINT32_MAX)
    return invalid("FU context domain is outside u32");
  const std::optional<unsigned> contextWidth =
      temporal ? std::optional<unsigned>(
                     std::max(1U, llvm::Log2_64_Ceil(contextCount)))
               : std::nullopt;

  auto residencies = fabric.configurationResidencies(field);
  if (!residencies)
    return residencies.takeError();
  if (residencies->size() != 1 && residencies->size() != contextCount)
    return invalid("FU configuration residency is incomplete");
  std::vector<std::pair<FieldDecoderPlan, const FiniteCodebookEncoding *>>
      prepared;
  prepared.reserve(residencies->size());
  for (const auto &residency : *residencies) {
    auto entry = prepareFiniteField(spatialCore, field, residency,
                                    configurationAbi, transportLayout);
    if (!entry)
      return entry.takeError();
    prepared.push_back(std::move(*entry));
  }

  struct TemplatePlan final {
    std::vector<llvm::APInt> activeCodes;
    std::vector<fabric::FabricFuCapabilityTemplateEdge> edges;
  };
  std::vector<TemplatePlan> templatePlans;
  templatePlans.reserve(templates.size());
  for (auto [ordinal, record] : llvm::enumerate(templates)) {
    auto semantic = fabric::encodeFabricFuConfiguration(
        fabric, field,
        fabric::FabricFuCapabilityTemplateRef{*definition, ordinal});
    if (!semantic)
      return semantic.takeError();
    std::vector<llvm::APInt> activeCodes;
    activeCodes.reserve(prepared.size());
    for (const auto &entry : prepared) {
      auto activeCode = physicalCode(*entry.second, semantic->bytes());
      if (!activeCode)
        return activeCode.takeError();
      activeCodes.push_back(std::move(*activeCode));
    }
    auto edges = fabric::projectFabricFuCapabilityTemplateTerminalEdges(record);
    if (!edges)
      return edges.takeError();
    std::set<std::string> destinations;
    for (const auto &edge : *edges)
      if (!destinations.insert(fuTemplateEndpointKey(edge.destination)).second)
        return invalid(
            "one FU capability template drives a terminal more than once");
    templatePlans.push_back(
        TemplatePlan{std::move(activeCodes), std::move(*edges)});
  }

  std::vector<const OperationShellModule *> operations;
  for (const OperationShellModule &shell : operationShells)
    if (shell.operation.localOccurrence.fu == fu)
      operations.push_back(&shell);
  if (operations.empty())
    return invalid("FU has no operation shell");

  const auto resolveNode = [&](const fabric::FabricFuNodePortRef &port)
      -> llvm::Expected<std::pair<const OperationShellModule *,
                                  const OperationEndpointPlan *>> {
    if (port.node.node != fabric::FabricFuNodeKind::Op)
      return invalid("projected FU terminal still names a selector node");
    auto occurrence =
        fabric::deriveFabricFuOccurrenceNode(fabric, port.node, fu);
    if (!occurrence)
      return occurrence.takeError();
    const OperationShellModule *shell =
        findOperationShell(operationShells, *occurrence);
    if (!shell)
      return invalid("FU terminal operation has no unique shell");
    const OperationEndpointPlan *endpoint =
        findOperationEndpoint(*shell, port.direction, port.ordinal);
    if (!endpoint)
      return invalid("FU terminal names an absent operation port");
    return std::make_pair(shell, endpoint);
  };
  for (const TemplatePlan &plan : templatePlans)
    for (const auto &edge : plan.edges) {
      for (const auto &terminal : {edge.source, edge.destination}) {
        if (terminal.kind() ==
            fabric::FabricFuCapabilityTemplateEndpointKind::BoundaryPort) {
          const auto &port =
              std::get<fabric::FabricFuTemplatePortRef>(terminal.payload);
          if (port.fu != *definition ||
              !findEndpoint(*endpoints, port.direction, port.ordinal))
            return invalid("FU template names an absent boundary terminal");
          continue;
        }
        auto resolved = resolveNode(
            std::get<fabric::FabricFuNodePortRef>(terminal.payload));
        if (!resolved)
          return resolved.takeError();
      }
    }

  // Direct result lanes share the producer's atomic publication identity.
  // Stored lanes retain independent retirement; their source port is the
  // requester. Producing context remains a separate carried namespace field.
  std::map<std::string, std::string> sourceRequesterKeys;
  std::map<std::string, std::uint32_t> requesterOrdinals;
  std::map<std::string, bool> requesterDirect;
  for (const TemplatePlan &plan : templatePlans)
    for (const auto &edge : plan.edges) {
      if (edge.destination.kind() !=
          fabric::FabricFuCapabilityTemplateEndpointKind::BoundaryPort)
        continue;
      const std::string sourceKey = fuTemplateEndpointKey(edge.source);
      std::string requesterKey = sourceKey;
      bool direct =
          edge.source.kind() ==
          fabric::FabricFuCapabilityTemplateEndpointKind::BoundaryPort;
      if (const auto *port =
              std::get_if<fabric::FabricFuNodePortRef>(&edge.source.payload)) {
        auto source = resolveNode(*port);
        if (!source)
          return source.takeError();
        auto sourceInterface = deriveFabricOperationLeafInterface(
            *source->first->operation.capability);
        if (!sourceInterface)
          return sourceInterface.takeError();
        if (sourceInterface->hasDirectTokenPublication()) {
          direct = true;
          auto bytes = fabric::canonicalFabricBytes(
              source->first->operation.localOccurrence);
          requesterKey.assign(1, '\2');
          requesterKey.append(reinterpret_cast<const char *>(bytes.data()),
                              bytes.size());
        }
      }
      sourceRequesterKeys.emplace(sourceKey, requesterKey);
      requesterOrdinals.try_emplace(requesterKey, 0);
      requesterDirect.emplace(requesterKey, direct);
    }
  if (requesterOrdinals.empty())
    return invalid("FU has no result presentation requester");
  std::uint32_t resultRequesterCount = 0;
  std::vector<bool> resultRequesterDirectPublication;
  for (auto &[key, ordinal] : requesterOrdinals) {
    ordinal = resultRequesterCount++;
    resultRequesterDirectPublication.push_back(requesterDirect.at(key));
  }
  const unsigned requesterWidth = indexWidth(resultRequesterCount);

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  std::vector<FieldDecoderPlan> fuDecoders;
  fuDecoders.reserve(prepared.size());
  for (const auto &entry : prepared)
    fuDecoders.push_back(entry.first);
  std::vector<ConfigurationBundlePlan> operationConfigurations;
  operationConfigurations.reserve(operations.size());
  for (const OperationShellModule *operation : operations)
    operationConfigurations.push_back(operation->configuration);
  auto configuration =
      deriveConfigurationBundlePlan(fuDecoders, operationConfigurations);
  if (!configuration)
    return configuration.takeError();
  appendClockResetAndConfigurationPorts(builder, *configuration, inputs);
  for (const EndpointPlan &endpoint : *endpoints) {
    appendEndpointPorts(inputs, outputs, endpoint);
    if (endpoint.direction == fabric::FabricPortDirection::Output)
      outputs.push_back(
          {{builder.getStringAttr(fuOutputOfferPortName(endpoint)),
            builder.getI1Type(), circt::hw::ModulePort::Direction::Output}});
    if (endpoint.direction == fabric::FabricPortDirection::Output)
      outputs.push_back(
          {{builder.getStringAttr(fuOutputRequesterPortName(endpoint)),
            builder.getIntegerType(requesterWidth),
            circt::hw::ModulePort::Direction::Output}});
  }
  if (contextWidth) {
    inputs.push_back({{builder.getStringAttr(dispatchEnablePortName),
                       builder.getI1Type(),
                       circt::hw::ModulePort::Direction::Input}});
    inputs.push_back(
        {{builder.getStringAttr(resultPresentationPriorityPortName),
          builder.getIntegerType(requesterWidth),
          circt::hw::ModulePort::Direction::Input}});
    outputs.push_back({{builder.getStringAttr(resultRequesterOffersPortName),
                        builder.getIntegerType(resultRequesterCount),
                        circt::hw::ModulePort::Direction::Output}});
    inputs.push_back({{builder.getStringAttr(dispatchContextPortName),
                       builder.getIntegerType(*contextWidth),
                       circt::hw::ModulePort::Direction::Input}});
    for (const EndpointPlan &endpoint : *endpoints) {
      if (endpoint.direction != fabric::FabricPortDirection::Output)
        continue;
      outputs.push_back(
          {{builder.getStringAttr(outputContextPortName(endpoint)),
            builder.getIntegerType(*contextWidth),
            circt::hw::ModulePort::Direction::Output}});
    }
  }
  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_fabric_fu_" + std::to_string(fu.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        ConfigurationBundleSignals configurationValues =
            configurationBundleSignals(accessor, *configuration);
        std::vector<mlir::Value> fieldSignals;
        fieldSignals.reserve(prepared.size());
        for (const auto &entry : prepared)
          fieldSignals.push_back(decodeFieldSignal(
              bodyBuilder, location, configurationValues, entry.first));

        circt::BackedgeBuilder backedges(bodyBuilder, location);
        // The parent Temporal PE grants one context to this FU per clock
        // cycle. Boundary tokens belong to that context by construction; an
        // FU-internal result names the context that produced it and is
        // deliverable to another operation only while that context is the
        // granted one.
        const mlir::Value dispatchContext =
            contextWidth ? accessor.getInput(dispatchContextPortName)
                         : mlir::Value{};
        std::map<std::string, circt::Backedge> dataInput;
        std::map<std::string, circt::Backedge> validInput;
        std::map<std::string, circt::Backedge> offerInput;
        std::map<std::string, circt::Backedge> readyInput;
        std::map<std::string, mlir::Value> dataOutput;
        std::map<std::string, mlir::Value> validOutput;
        std::map<std::string, mlir::Value> offerOutput;
        std::map<std::string, mlir::Value> readyOutput;
        std::map<std::string, mlir::Value> contextOutput;
        for (const OperationShellModule *operation : operations) {
          std::map<std::string, mlir::Value> instanceInputs;
          if (llvm::Error error = addCommonInstanceInputs(
                  bodyBuilder, location, accessor, *configuration,
                  operation->configuration, operation->module,
                  instanceInputs)) {
            materializationError = llvm::toString(std::move(error));
            backedges.abandon();
            return;
          }
          if (contextWidth) {
            instanceInputs.emplace(dispatchContextPortName.str(),
                                   dispatchContext);
            instanceInputs.emplace(dispatchEnablePortName.str(),
                                   accessor.getInput(dispatchEnablePortName));
          }
          for (const OperationEndpointPlan &endpoint : operation->endpoints) {
            const fabric::FabricFuNodePortRef nodePort{
                operation->operation.capability->occurrence, endpoint.direction,
                endpoint.ordinal};
            const std::string key = fuTemplateEndpointKey(
                fabric::FabricFuCapabilityTemplateEndpointRef::nodePort(
                    nodePort));
            if (endpoint.direction == fabric::FabricPortDirection::Input) {
              if (endpoint.data) {
                auto edge = backedges.get(endpoint.data->type);
                instanceInputs.emplace(endpoint.data->getName().str(), edge);
                dataInput.emplace(key, std::move(edge));
              }
              auto edge = backedges.get(bodyBuilder.getI1Type());
              instanceInputs.emplace(endpoint.valid.getName().str(), edge);
              validInput.emplace(key, std::move(edge));
              if (endpoint.offer) {
                auto offer = backedges.get(bodyBuilder.getI1Type());
                instanceInputs.emplace(endpoint.offer->getName().str(), offer);
                offerInput.emplace(key, std::move(offer));
              }
            } else {
              auto edge = backedges.get(bodyBuilder.getI1Type());
              instanceInputs.emplace(endpoint.ready.getName().str(), edge);
              readyInput.emplace(key, std::move(edge));
            }
          }
          auto instance = instantiateModule(
              bodyBuilder, location, operation->module,
              "operation_" +
                  std::to_string(operation->operation.localOccurrence.ordinal),
              instanceInputs);
          if (!instance) {
            materializationError = llvm::toString(instance.takeError());
            backedges.abandon();
            return;
          }
          for (const OperationEndpointPlan &endpoint : operation->endpoints) {
            const fabric::FabricFuNodePortRef nodePort{
                operation->operation.capability->occurrence, endpoint.direction,
                endpoint.ordinal};
            const std::string key = fuTemplateEndpointKey(
                fabric::FabricFuCapabilityTemplateEndpointRef::nodePort(
                    nodePort));
            if (endpoint.direction == fabric::FabricPortDirection::Input) {
              readyOutput.emplace(key,
                                  instance->at(endpoint.ready.getName().str()));
            } else {
              if (endpoint.data)
                dataOutput.emplace(
                    key, instance->at(endpoint.data->getName().str()));
              validOutput.emplace(key,
                                  instance->at(endpoint.valid.getName().str()));
              offerOutput.emplace(
                  key, instance->at(endpoint.offer->getName().str()));
              if (endpoint.context)
                contextOutput.emplace(
                    key, instance->at(endpoint.context->getName().str()));
            }
          }
        }

        struct RouteRuntime final {
          std::size_t templateOrdinal = 0;
          const fabric::FabricFuCapabilityTemplateEdge *edge = nullptr;
          std::string sourceKey;
          std::string destinationKey;
          mlir::Value active;
          mlir::Value sourceValid;
          mlir::Value sourceOffer;
          mlir::Value destinationReady;
          std::optional<mlir::Value> data;
          std::optional<mlir::Value> context;
        };
        std::vector<RouteRuntime> routes;
        const auto terminalType = [&](const auto &terminal)
            -> llvm::Expected<::fabric::DataPathType> {
          if (terminal.kind() ==
              fabric::FabricFuCapabilityTemplateEndpointKind::BoundaryPort) {
            const auto &port =
                std::get<fabric::FabricFuTemplatePortRef>(terminal.payload);
            const EndpointPlan *endpoint =
                findEndpoint(*endpoints, port.direction, port.ordinal);
            if (!endpoint)
              return invalid("FU boundary terminal disappeared");
            return endpoint->dataPath;
          }
          auto resolved = resolveNode(
              std::get<fabric::FabricFuNodePortRef>(terminal.payload));
          if (!resolved)
            return resolved.takeError();
          return ::fabric::DataPathType{::fabric::DataPathKind::Bits,
                                        resolved->second->payloadWidthBits, 0};
        };
        const auto sourceSignals = [&](const auto &terminal)
            -> llvm::Expected<std::pair<ForwardTransportSignals,
                                        std::optional<mlir::Value>>> {
          const std::string key = fuTemplateEndpointKey(terminal);
          if (terminal.kind() ==
              fabric::FabricFuCapabilityTemplateEndpointKind::BoundaryPort) {
            const auto &port =
                std::get<fabric::FabricFuTemplatePortRef>(terminal.payload);
            const EndpointPlan *endpoint =
                findEndpoint(*endpoints, port.direction, port.ordinal);
            if (!endpoint ||
                endpoint->direction != fabric::FabricPortDirection::Input)
              return invalid("FU route source is not an input terminal");
            return std::make_pair(
                ForwardTransportSignals{
                    accessor.getInput(endpoint->valid.getName()),
                    endpoint->data
                        ? std::optional<mlir::Value>{accessor.getInput(
                              endpoint->data->getName())}
                        : std::nullopt,
                    endpoint->tag
                        ? std::optional<mlir::Value>{accessor.getInput(
                              endpoint->tag->getName())}
                        : std::nullopt},
                contextWidth ? std::optional<mlir::Value>{dispatchContext}
                             : std::nullopt);
          }
          const auto valid = validOutput.find(key);
          if (valid == validOutput.end())
            return invalid("FU route source operation signal is absent");
          const auto data = dataOutput.find(key);
          const auto context = contextOutput.find(key);
          return std::make_pair(
              ForwardTransportSignals{
                  valid->second,
                  data == dataOutput.end()
                      ? std::nullopt
                      : std::optional<mlir::Value>{data->second},
                  std::nullopt},
              context == contextOutput.end()
                  ? std::nullopt
                  : std::optional<mlir::Value>{context->second});
        };
        const auto destinationReady =
            [&](const auto &terminal) -> llvm::Expected<mlir::Value> {
          const std::string key = fuTemplateEndpointKey(terminal);
          if (terminal.kind() ==
              fabric::FabricFuCapabilityTemplateEndpointKind::BoundaryPort) {
            const auto &port =
                std::get<fabric::FabricFuTemplatePortRef>(terminal.payload);
            const EndpointPlan *endpoint =
                findEndpoint(*endpoints, port.direction, port.ordinal);
            if (!endpoint ||
                endpoint->direction != fabric::FabricPortDirection::Output)
              return invalid("FU route destination is not an output terminal");
            return accessor.getInput(endpoint->ready.getName());
          }
          const auto ready = readyOutput.find(key);
          if (ready == readyOutput.end())
            return invalid("FU route destination operation signal is absent");
          return ready->second;
        };

        for (auto [templateOrdinal, plan] : llvm::enumerate(templatePlans))
          for (const auto &edge : plan.edges) {
            auto source = sourceSignals(edge.source);
            auto sourceType = terminalType(edge.source);
            auto destinationType = terminalType(edge.destination);
            auto ready = destinationReady(edge.destination);
            if (!source || !sourceType || !destinationType || !ready) {
              materializationError =
                  !source       ? llvm::toString(source.takeError())
                  : !sourceType ? llvm::toString(sourceType.takeError())
                  : !destinationType
                      ? llvm::toString(destinationType.takeError())
                      : llvm::toString(ready.takeError());
              backedges.abandon();
              return;
            }
            auto adapted =
                adaptForwardTransportSignals(bodyBuilder, location, *sourceType,
                                             *destinationType, source->first);
            if (!adapted) {
              materializationError = llvm::toString(adapted.takeError());
              backedges.abandon();
              return;
            }
            std::vector<mlir::Value> contextActives;
            contextActives.reserve(prepared.size());
            for (auto [context, fieldSignal] : llvm::enumerate(fieldSignals))
              contextActives.push_back(matchesCode(bodyBuilder, location,
                                                   fieldSignal,
                                                   plan.activeCodes[context]));
            mlir::Value active = contextActives.front();
            if (source->second && contextActives.size() > 1)
              active = selectContextValue(bodyBuilder, location,
                                          *source->second, contextActives);
            // A held result may retire at the boundary independently of
            // dispatch, but a new operation input or boundary-input bypass
            // belongs to this cycle's explicit evaluation grant.
            if (contextWidth &&
                (edge.source.kind() ==
                     fabric::FabricFuCapabilityTemplateEndpointKind::
                         BoundaryPort ||
                 edge.destination.kind() !=
                     fabric::FabricFuCapabilityTemplateEndpointKind::
                         BoundaryPort))
              active = andValues(
                  bodyBuilder, location,
                  {active, accessor.getInput(dispatchEnablePortName)});
            const bool internalToken =
                edge.source.kind() !=
                    fabric::FabricFuCapabilityTemplateEndpointKind::
                        BoundaryPort &&
                edge.destination.kind() !=
                    fabric::FabricFuCapabilityTemplateEndpointKind::
                        BoundaryPort;
            if (internalToken && source->second)
              active = andValues(
                  bodyBuilder, location,
                  {active,
                   circt::comb::ICmpOp::create(
                       bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                       *source->second, dispatchContext, true)});
            routes.push_back(RouteRuntime{
                templateOrdinal, &edge, fuTemplateEndpointKey(edge.source),
                fuTemplateEndpointKey(edge.destination), active,
                source->first.valid,
                edge.source.kind() ==
                        fabric::FabricFuCapabilityTemplateEndpointKind::
                            BoundaryPort
                    ? source->first.valid
                    : offerOutput.at(fuTemplateEndpointKey(edge.source)),
                *ready, adapted->payload, source->second});
          }

        // Present a complete direct-operation tuple before forwarding any
        // readiness. Stored lanes remain independent requesters. Internal
        // edges have one active source in the selected capability template;
        // their offer projection is not gated by downstream admission.
        std::map<std::string, std::vector<std::size_t>> destinationRoutes;
        for (auto [ordinal, route] : llvm::enumerate(routes))
          destinationRoutes[route.destinationKey].push_back(ordinal);
        std::uint32_t outputCount = 0;
        for (const EndpointPlan &endpoint : *endpoints)
          outputCount +=
              endpoint.direction == fabric::FabricPortDirection::Output;
        std::vector<std::map<std::string, std::vector<mlir::Value>>>
            requesterClaims(resultRequesterCount);
        std::vector<std::optional<std::uint32_t>> routeRequester(routes.size());
        for (auto [ordinal, route] : llvm::enumerate(routes)) {
          const auto *destination =
              std::get_if<fabric::FabricFuTemplatePortRef>(
                  &route.edge->destination.payload);
          if (!destination)
            continue;
          const std::uint32_t requester =
              requesterOrdinals.at(sourceRequesterKeys.at(route.sourceKey));
          routeRequester[ordinal] = requester;
          auto [lane, inserted] = requesterClaims[requester].try_emplace(
              route.sourceKey, outputCount,
              bitConstant(bodyBuilder, location, false));
          mlir::Value &claim = lane->second[destination->ordinal];
          claim =
              orValues(bodyBuilder, location,
                       {claim, andValues(bodyBuilder, location,
                                         {route.active, route.sourceOffer})});
        }
        std::vector<llvm::SmallVector<mlir::Value>> packedClaims(
            resultRequesterCount);
        for (auto [requester, lanes] : llvm::enumerate(requesterClaims))
          for (const auto &[source, claims] : lanes)
            packedClaims[requester].push_back(
                packBits(bodyBuilder, location, claims));
        std::vector<llvm::SmallVector<mlir::Value>> requesterEligible;
        std::vector<llvm::SmallVector<mlir::Value>> requesterEvaluated;
        llvm::SmallVector<mlir::Value> rawOffers;
        for (const auto &lanes : packedClaims) {
          llvm::SmallVector<mlir::Value> offers;
          for (mlir::Value claims : lanes)
            offers.push_back(circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::ne, claims,
                zeroData(bodyBuilder, location, outputCount), true));
          rawOffers.push_back(orValues(bodyBuilder, location, offers));
          requesterEligible.push_back({rawOffers.back()});
          requesterEvaluated.push_back(
              {bitConstant(bodyBuilder, location, true)});
        }
        mlir::Value priority;
        if (contextWidth) {
          priority = accessor.getInput(resultPresentationPriorityPortName);
          accessor.setOutput(resultRequesterOffersPortName,
                             packBits(bodyBuilder, location, rawOffers));
        } else {
          priority = makeResultPresentationPriority(
              bodyBuilder, location, backedges, requesterEligible,
              requesterEvaluated, accessor.getInput("clock"),
              accessor.getInput("reset"), "result_presentation_cursor_reg",
              clockReset);
        }
        const auto presentation = selectResultPresentation(
            bodyBuilder, location, packedClaims, priority);
        std::vector<mlir::Value> routeAdmissible(routes.size());
        std::vector<mlir::Value> routeGrant(routes.size());
        std::vector<mlir::Value> routePresentsPayload(routes.size());
        std::vector<mlir::Value> routeReady(routes.size());
        for (auto [ordinal, route] : llvm::enumerate(routes)) {
          routeAdmissible[ordinal] = route.active;
          if (routeRequester[ordinal]) {
            const std::uint32_t requester = *routeRequester[ordinal];
            routeAdmissible[ordinal] = andValues(
                bodyBuilder, location, {route.active, presentation[requester]});
          }
          routeGrant[ordinal] =
              andValues(bodyBuilder, location,
                        {routeAdmissible[ordinal], route.sourceOffer});
          routePresentsPayload[ordinal] =
              destinationRoutes.at(route.destinationKey).size() == 1
                  ? route.active
                  : routeGrant[ordinal];
          routeReady[ordinal] =
              andValues(bodyBuilder, location,
                        {routeAdmissible[ordinal], route.destinationReady});
        }

        const auto sourceReady = [&](llvm::StringRef sourceKey) {
          llvm::SmallVector<mlir::Value> alternatives;
          for (std::size_t templateOrdinal = 0;
               templateOrdinal < templatePlans.size(); ++templateOrdinal) {
            llvm::SmallVector<mlir::Value> destinations;
            for (auto [ordinal, route] : llvm::enumerate(routes))
              if (route.templateOrdinal == templateOrdinal &&
                  route.sourceKey == sourceKey)
                destinations.push_back(routeReady[ordinal]);
            if (!destinations.empty())
              alternatives.push_back(
                  andValues(bodyBuilder, location, destinations));
          }
          return orValues(bodyBuilder, location, alternatives);
        };
        // A source presents its token to every destination of its selected
        // template at once; a destination publishes the granted route only
        // while the route's peers are ready, so an atomic fanout inside the
        // FU never delivers to a ready subset.
        const auto peersReady = [&](std::size_t ordinal) {
          const RouteRuntime &route = routes[ordinal];
          llvm::SmallVector<mlir::Value> peers;
          for (auto [peerOrdinal, peer] : llvm::enumerate(routes))
            if (peer.templateOrdinal == route.templateOrdinal &&
                peer.sourceKey == route.sourceKey &&
                peer.destinationKey != route.destinationKey)
              peers.push_back(routeReady[peerOrdinal]);
          return andValues(bodyBuilder, location, peers);
        };
        struct DestinationSignals final {
          mlir::Value valid;
          mlir::Value offer;
          mlir::Value requester;
          std::optional<mlir::Value> data;
          std::optional<mlir::Value> context;
        };
        const auto publishDestination =
            [&](const std::string &key, std::optional<unsigned> dataWidth,
                std::optional<mlir::Value> idleContext) {
              DestinationSignals signals;
              signals.requester =
                  zeroData(bodyBuilder, location, requesterWidth);
              llvm::SmallVector<mlir::Value> validTerms;
              llvm::SmallVector<mlir::Value> offerTerms;
              if (dataWidth)
                signals.data = zeroData(bodyBuilder, location, *dataWidth);
              signals.context = idleContext;
              const auto found = destinationRoutes.find(key);
              if (found != destinationRoutes.end())
                for (std::size_t ordinal : found->second) {
                  const RouteRuntime &route = routes[ordinal];
                  offerTerms.push_back(routeGrant[ordinal]);
                  if (routeRequester[ordinal])
                    signals.requester = circt::comb::MuxOp::create(
                        bodyBuilder, location, routeGrant[ordinal],
                        circt::hw::ConstantOp::create(
                            bodyBuilder, location,
                            llvm::APInt(requesterWidth,
                                        *routeRequester[ordinal])),
                        signals.requester, true);
                  mlir::Value peers = peersReady(ordinal);
                  validTerms.push_back(andValues(
                      bodyBuilder, location,
                      {routeGrant[ordinal], route.sourceValid, peers}));
                  if (signals.data && route.data)
                    signals.data = circt::comb::MuxOp::create(
                        bodyBuilder, location, routePresentsPayload[ordinal],
                        *route.data, *signals.data, true);
                  if (signals.context && route.context)
                    signals.context = circt::comb::MuxOp::create(
                        bodyBuilder, location, routeGrant[ordinal],
                        *route.context, *signals.context, true);
                }
              signals.valid = orValues(bodyBuilder, location, validTerms);
              signals.offer = orValues(bodyBuilder, location, offerTerms);
              return signals;
            };

        for (const EndpointPlan &endpoint : *endpoints) {
          const fabric::FabricFuTemplatePortRef port{
              *definition, endpoint.direction, endpoint.localOrdinal};
          const auto terminal =
              fabric::FabricFuCapabilityTemplateEndpointRef::boundaryPort(port);
          const std::string key = fuTemplateEndpointKey(terminal);
          if (endpoint.direction == fabric::FabricPortDirection::Input) {
            accessor.setOutput(endpoint.ready.getName(), sourceReady(key));
            continue;
          }
          // An output holding no granted result reports the dispatch
          // context, so the PE evaluates the granted row's result selectors
          // and presents egress readiness to a transparent operation before
          // it publishes.
          DestinationSignals signals = publishDestination(
              key,
              endpoint.data
                  ? std::optional<unsigned>(endpoint.dataPath.payloadWidthBits)
                  : std::nullopt,
              contextWidth ? std::optional<mlir::Value>(dispatchContext)
                           : std::nullopt);
          if (endpoint.data)
            accessor.setOutput(endpoint.data->getName(), *signals.data);
          if (endpoint.tag)
            accessor.setOutput(endpoint.tag->getName(),
                               zeroData(bodyBuilder, location,
                                        endpoint.dataPath.tagWidthBits));
          if (contextWidth)
            accessor.setOutput(outputContextPortName(endpoint),
                               *signals.context);
          accessor.setOutput(endpoint.valid.getName(), signals.valid);
          accessor.setOutput(fuOutputOfferPortName(endpoint), signals.offer);
          accessor.setOutput(fuOutputRequesterPortName(endpoint),
                             signals.requester);
        }

        for (const OperationShellModule *operation : operations)
          for (const OperationEndpointPlan &endpoint : operation->endpoints) {
            const fabric::FabricFuNodePortRef port{
                operation->operation.capability->occurrence, endpoint.direction,
                endpoint.ordinal};
            const auto terminal =
                fabric::FabricFuCapabilityTemplateEndpointRef::nodePort(port);
            const std::string key = fuTemplateEndpointKey(terminal);
            if (endpoint.direction == fabric::FabricPortDirection::Output) {
              readyInput.at(key).setValue(sourceReady(key));
              continue;
            }
            DestinationSignals signals = publishDestination(
                key,
                endpoint.data
                    ? std::optional<unsigned>(endpoint.payloadWidthBits)
                    : std::nullopt,
                std::nullopt);
            if (endpoint.data)
              dataInput.at(key).setValue(*signals.data);
            validInput.at(key).setValue(signals.valid);
            if (endpoint.offer)
              offerInput.at(key).setValue(signals.offer);
          }
      });
  if (materializationError)
    return invalid(*materializationError);
  return FuModule{fu,
                  module,
                  std::move(*endpoints),
                  contextWidth,
                  std::move(resultRequesterDirectPublication),
                  std::move(*configuration)};
}

struct InputSelectorRuntime final {
  const EndpointPlan *fuEndpoint = nullptr;
  std::map<std::string, mlir::Value> selectedByPeEndpoint;
  std::map<std::string, mlir::Value> discardByPeEndpoint;
};

struct OutputSelectorRuntime final {
  const EndpointPlan *fuEndpoint = nullptr;
  std::map<std::string, mlir::Value> selectedByPeEndpoint;
  mlir::Value discard;
};

llvm::Expected<PeModule>
buildSpatialPeModule(mlir::OpBuilder &builder, mlir::Location location,
                     fabric::SpatialCoreOccurrenceRef spatialCore,
                     const fabric::FabricArtifactView &fabric,
                     const ConfigurationABI &configurationAbi,
                     const ConfigurationTransportLayout &transportLayout,
                     llvm::ArrayRef<FuModule> fuModules,
                     fabric::FabricPeOccurrenceRef pe) {
  if (fabric.peSchedule(pe) != ::fabric::Schedule::Spatial)
    return invalid("Spatial PE lowering received a non-Spatial PE");
  auto endpoints = deriveEndpointPlans(
      builder, fabric, fabric::FabricTransportEndpointOwnerRef::of(pe));
  if (!endpoints)
    return endpoints.takeError();
  auto schema = fabric.spatialPeConfigurationSchema(pe);
  if (!schema)
    return schema.takeError();

  const fabric::FabricPeConfigurationFieldView *activationField = nullptr;
  for (const auto &descriptor : schema->fields())
    if (descriptor.kind == fabric::FabricPeConfigurationFieldKind::Activation) {
      if (activationField)
        return invalid("PE activation field is duplicated");
      activationField = &descriptor;
    }
  if (!activationField)
    return invalid("PE activation field is absent");
  auto activationPrepared =
      prepareFiniteField(spatialCore, activationField->reference,
                         configurationAbi, transportLayout);
  if (!activationPrepared)
    return activationPrepared.takeError();

  std::vector<const FuModule *> children;
  for (fabric::FabricFuOccurrenceRef fu : fabric.fuOccurrences())
    if (fabric.parentPeOf(fu) == pe) {
      const FuModule *module = findFuModule(fuModules, fu);
      if (!module)
        return invalid("PE child FU has no unique hierarchy module");
      children.push_back(module);
    }
  if (children.empty())
    return invalid("PE has no child FU hierarchy module");

  std::vector<FieldDecoderPlan> peDecoders;
  peDecoders.push_back(activationPrepared->first);
  for (const FuModule *child : children)
    for (const EndpointPlan &fuEndpoint : child->endpoints) {
      const fabric::FabricFuOccurrencePortRef port{
          child->reference, fuEndpoint.direction, fuEndpoint.localOrdinal};
      const auto descriptor =
          llvm::find_if(schema->fields(), [&](const auto &candidate) {
            return candidate.port && *candidate.port == port;
          });
      if (descriptor == schema->fields().end())
        return invalid("PE selector field is absent");
      auto prepared = prepareFiniteField(spatialCore, descriptor->reference,
                                         configurationAbi, transportLayout);
      if (!prepared)
        return prepared.takeError();
      peDecoders.push_back(std::move(prepared->first));
    }

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  std::vector<ConfigurationBundlePlan> childConfigurations;
  childConfigurations.reserve(children.size());
  for (const FuModule *child : children)
    childConfigurations.push_back(child->configuration);
  auto configuration =
      deriveConfigurationBundlePlan(peDecoders, childConfigurations);
  if (!configuration)
    return configuration.takeError();
  appendClockResetAndConfigurationPorts(builder, *configuration, inputs);
  for (const EndpointPlan &endpoint : *endpoints)
    appendEndpointPorts(inputs, outputs, endpoint);
  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_spatial_pe_" + std::to_string(pe.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        ConfigurationBundleSignals configurationValues =
            configurationBundleSignals(accessor, *configuration);
        mlir::Value activation =
            decodeFieldSignal(bodyBuilder, location, configurationValues,
                              activationPrepared->first);
        std::map<std::string, llvm::SmallVector<mlir::Value>> peReadyTerms;
        std::map<std::string, llvm::SmallVector<mlir::Value>> peValidTerms;
        std::map<std::string, mlir::Value> peData;
        for (const EndpointPlan &endpoint : *endpoints) {
          if (endpoint.direction == fabric::FabricPortDirection::Input)
            peReadyTerms.emplace(endpointKey(endpoint.endpoint),
                                 llvm::SmallVector<mlir::Value>{});
          else {
            peValidTerms.emplace(endpointKey(endpoint.endpoint),
                                 llvm::SmallVector<mlir::Value>{});
            if (endpoint.data)
              peData.emplace(endpointKey(endpoint.endpoint),
                             zeroData(bodyBuilder, location,
                                      endpoint.dataPath.payloadWidthBits));
          }
        }

        for (const FuModule *child : children) {
          auto activeSemantic =
              schema->encode(activationField->reference,
                             fabric::FabricPeActive{child->reference});
          if (!activeSemantic) {
            materializationError = llvm::toString(activeSemantic.takeError());
            return;
          }
          auto activeCode = physicalCode(*activationPrepared->second,
                                         activeSemantic->bytes());
          if (!activeCode) {
            materializationError = llvm::toString(activeCode.takeError());
            return;
          }
          mlir::Value active =
              matchesCode(bodyBuilder, location, activation, *activeCode);
          std::map<std::string, mlir::Value> instanceInputs;
          if (llvm::Error error = addCommonInstanceInputs(
                  bodyBuilder, location, accessor, *configuration,
                  child->configuration, child->module, instanceInputs)) {
            materializationError = llvm::toString(std::move(error));
            return;
          }
          std::vector<InputSelectorRuntime> inputSelectors;
          std::vector<OutputSelectorRuntime> outputSelectors;

          for (const EndpointPlan &fuEndpoint : child->endpoints) {
            const fabric::FabricFuOccurrencePortRef port{
                child->reference, fuEndpoint.direction,
                fuEndpoint.localOrdinal};
            const auto descriptor =
                llvm::find_if(schema->fields(), [&](const auto &candidate) {
                  return candidate.port && *candidate.port == port;
                });
            if (descriptor == schema->fields().end()) {
              materializationError = "PE selector field is absent";
              return;
            }
            auto prepared =
                prepareFiniteField(spatialCore, descriptor->reference,
                                   configurationAbi, transportLayout);
            if (!prepared) {
              materializationError = llvm::toString(prepared.takeError());
              return;
            }
            mlir::Value field = decodeFieldSignal(
                bodyBuilder, location, configurationValues, prepared->first);
            const auto attachments = fabric.fuOccurrencePortAttachments(port);
            if (attachments.empty()) {
              materializationError = "PE selector has no attachment domain";
              return;
            }
            if (fuEndpoint.direction == fabric::FabricPortDirection::Input) {
              mlir::Value data =
                  fuEndpoint.data
                      ? zeroData(bodyBuilder, location,
                                 fuEndpoint.dataPath.payloadWidthBits)
                      : mlir::Value{};
              llvm::SmallVector<mlir::Value> valids;
              InputSelectorRuntime runtime;
              runtime.fuEndpoint = &fuEndpoint;
              for (const auto &attachment : attachments) {
                auto routeSemantic =
                    schema->encode(descriptor->reference,
                                   fabric::FabricPeRoute{attachment.endpoint});
                if (!routeSemantic) {
                  materializationError =
                      llvm::toString(routeSemantic.takeError());
                  return;
                }
                auto routeCode =
                    physicalCode(*prepared->second, routeSemantic->bytes());
                if (!routeCode) {
                  materializationError = llvm::toString(routeCode.takeError());
                  return;
                }
                mlir::Value selected =
                    matchesCode(bodyBuilder, location, field, *routeCode);
                const auto peEndpoint = llvm::find_if(
                    *endpoints, [&](const EndpointPlan &candidate) {
                      return candidate.endpoint == attachment.endpoint;
                    });
                if (peEndpoint == endpoints->end() ||
                    peEndpoint->direction !=
                        fabric::FabricPortDirection::Input) {
                  materializationError =
                      "PE input selector names a foreign endpoint";
                  return;
                }
                if (fuEndpoint.data) {
                  auto adapted = adaptForwardTransportSignals(
                      bodyBuilder, location, peEndpoint->dataPath,
                      fuEndpoint.dataPath,
                      ForwardTransportSignals{
                          accessor.getInput(peEndpoint->valid.getName()),
                          peEndpoint->data
                              ? std::optional<mlir::Value>{accessor.getInput(
                                    peEndpoint->data->getName())}
                              : std::nullopt,
                          std::nullopt});
                  if (!adapted) {
                    materializationError = llvm::toString(adapted.takeError());
                    return;
                  }
                  data = circt::comb::MuxOp::create(bodyBuilder, location,
                                                    selected, *adapted->payload,
                                                    data, true);
                }
                valids.push_back(circt::comb::AndOp::create(
                    bodyBuilder, location, selected,
                    accessor.getInput(peEndpoint->valid.getName())));
                runtime.selectedByPeEndpoint.emplace(
                    endpointKey(peEndpoint->endpoint), selected);

                auto discardSemantic = schema->encode(
                    descriptor->reference,
                    fabric::FabricPeInputDiscard{attachment.endpoint});
                if (discardSemantic) {
                  auto discardCode =
                      physicalCode(*prepared->second, discardSemantic->bytes());
                  if (!discardCode) {
                    materializationError =
                        llvm::toString(discardCode.takeError());
                    return;
                  }
                  runtime.discardByPeEndpoint.emplace(
                      endpointKey(peEndpoint->endpoint),
                      matchesCode(bodyBuilder, location, field, *discardCode));
                } else {
                  llvm::consumeError(discardSemantic.takeError());
                }
              }
              if (fuEndpoint.data)
                instanceInputs.emplace(fuEndpoint.data->getName().str(), data);
              instanceInputs.emplace(
                  fuEndpoint.valid.getName().str(),
                  andValues(bodyBuilder, location,
                            {active, orValues(bodyBuilder, location, valids)}));
              inputSelectors.push_back(std::move(runtime));
              continue;
            }

            llvm::SmallVector<mlir::Value> readyTerms;
            OutputSelectorRuntime runtime;
            runtime.fuEndpoint = &fuEndpoint;
            for (const auto &attachment : attachments) {
              auto routeSemantic =
                  schema->encode(descriptor->reference,
                                 fabric::FabricPeRoute{attachment.endpoint});
              if (!routeSemantic) {
                materializationError =
                    llvm::toString(routeSemantic.takeError());
                return;
              }
              auto routeCode =
                  physicalCode(*prepared->second, routeSemantic->bytes());
              if (!routeCode) {
                materializationError = llvm::toString(routeCode.takeError());
                return;
              }
              mlir::Value selected =
                  matchesCode(bodyBuilder, location, field, *routeCode);
              const auto peEndpoint =
                  llvm::find_if(*endpoints, [&](const EndpointPlan &candidate) {
                    return candidate.endpoint == attachment.endpoint;
                  });
              if (peEndpoint == endpoints->end() ||
                  peEndpoint->direction !=
                      fabric::FabricPortDirection::Output) {
                materializationError =
                    "PE output selector names a foreign endpoint";
                return;
              }
              readyTerms.push_back(circt::comb::AndOp::create(
                  bodyBuilder, location, selected,
                  accessor.getInput(peEndpoint->ready.getName())));
              runtime.selectedByPeEndpoint.emplace(
                  endpointKey(peEndpoint->endpoint), selected);
            }
            auto discardSemantic = schema->encode(
                descriptor->reference, fabric::FabricPeOutputDiscard{});
            if (!discardSemantic) {
              materializationError =
                  llvm::toString(discardSemantic.takeError());
              return;
            }
            auto discardCode =
                physicalCode(*prepared->second, discardSemantic->bytes());
            if (!discardCode) {
              materializationError = llvm::toString(discardCode.takeError());
              return;
            }
            runtime.discard =
                matchesCode(bodyBuilder, location, field, *discardCode);
            readyTerms.push_back(runtime.discard);
            instanceInputs.emplace(
                fuEndpoint.ready.getName().str(),
                andValues(
                    bodyBuilder, location,
                    {active, orValues(bodyBuilder, location, readyTerms)}));
            outputSelectors.push_back(std::move(runtime));
          }

          auto instance = instantiateModule(
              bodyBuilder, location, child->module,
              "fu_" + std::to_string(child->reference.id()), instanceInputs);
          if (!instance) {
            materializationError = llvm::toString(instance.takeError());
            return;
          }
          for (const InputSelectorRuntime &selector : inputSelectors) {
            mlir::Value fuReady =
                instance->at(selector.fuEndpoint->ready.getName().str());
            for (const auto &[key, selected] : selector.selectedByPeEndpoint)
              peReadyTerms.at(key).push_back(andValues(
                  bodyBuilder, location, {active, selected, fuReady}));
            for (const auto &[key, discard] : selector.discardByPeEndpoint)
              peReadyTerms.at(key).push_back(
                  andValues(bodyBuilder, location, {active, discard}));
          }
          for (const OutputSelectorRuntime &selector : outputSelectors) {
            mlir::Value fuValid =
                instance->at(selector.fuEndpoint->valid.getName().str());
            for (const auto &[key, selected] : selector.selectedByPeEndpoint) {
              const std::string endpointSelectionKey = key;
              peValidTerms.at(key).push_back(andValues(
                  bodyBuilder, location, {active, selected, fuValid}));
              if (selector.fuEndpoint->data) {
                const auto peEndpoint = llvm::find_if(
                    *endpoints, [&](const EndpointPlan &candidate) {
                      return endpointKey(candidate.endpoint) ==
                             endpointSelectionKey;
                    });
                auto adapted = adaptForwardTransportSignals(
                    bodyBuilder, location, selector.fuEndpoint->dataPath,
                    peEndpoint->dataPath,
                    ForwardTransportSignals{
                        fuValid,
                        std::optional<mlir::Value>{instance->at(
                            selector.fuEndpoint->data->getName().str())},
                        std::nullopt});
                if (!adapted) {
                  materializationError = llvm::toString(adapted.takeError());
                  return;
                }
                peData.at(key) = circt::comb::MuxOp::create(
                    bodyBuilder, location, selected, *adapted->payload,
                    peData.at(key), true);
              }
            }
          }
        }

        for (const EndpointPlan &endpoint : *endpoints) {
          const std::string key = endpointKey(endpoint.endpoint);
          if (endpoint.direction == fabric::FabricPortDirection::Input) {
            accessor.setOutput(
                endpoint.ready.getName(),
                orValues(bodyBuilder, location, peReadyTerms.at(key)));
            continue;
          }
          if (endpoint.data)
            accessor.setOutput(endpoint.data->getName(), peData.at(key));
          accessor.setOutput(
              endpoint.valid.getName(),
              orValues(bodyBuilder, location, peValidTerms.at(key)));
        }
      });
  if (materializationError)
    return invalid(*materializationError);
  return PeModule{pe, module, std::move(*endpoints), std::move(*configuration)};
}

} // namespace

llvm::Expected<std::vector<FuModule>>
buildFuModules(mlir::OpBuilder &builder, mlir::Location location,
               fabric::SpatialCoreOccurrenceRef spatialCore,
               const fabric::FabricArtifactView &fabric,
               const ConfigurationABI &configurationAbi,
               const ConfigurationTransportLayout &transportLayout,
               llvm::ArrayRef<OperationShellModule> operationShells,
               const ClockResetPlan &clockReset) {
  std::vector<FuModule> result;
  result.reserve(fabric.fuOccurrences().size());
  for (fabric::FabricFuOccurrenceRef fu : fabric.fuOccurrences()) {
    auto module =
        buildFuModule(builder, location, spatialCore, fabric, configurationAbi,
                      transportLayout, operationShells, fu, clockReset);
    if (!module)
      return module.takeError();
    result.push_back(std::move(*module));
  }
  return result;
}

llvm::Expected<std::vector<PeModule>>
buildPeModules(mlir::OpBuilder &builder, mlir::Location location,
               fabric::SpatialCoreOccurrenceRef spatialCore,
               const fabric::FabricArtifactView &fabric,
               const ConfigurationABI &configurationAbi,
               const ConfigurationTransportLayout &transportLayout,
               llvm::ArrayRef<FuModule> fuModules,
               const ClockResetPlan &clockReset, mlir::ModuleOp container,
               llvm::StringRef materializationKey) {
  std::vector<PeModule> result;
  result.reserve(fabric.peOccurrences().size());
  for (fabric::FabricPeOccurrenceRef pe : fabric.peOccurrences()) {
    const std::string peKey =
        (llvm::Twine(materializationKey) + ":pe:" + llvm::Twine(pe.id())).str();
    const bool temporal = fabric.peSchedule(pe) == ::fabric::Schedule::Temporal;
    if (invocationDiagnosticEnabled(DiagnosticVerbosity::Summary)) {
      std::uint64_t childCount = 0;
      std::uint64_t childEndpointCount = 0;
      std::uint64_t attachmentCount = 0;
      std::uint64_t childConfigurationPortCount = 0;
      std::uint64_t childConfigurationMemberCount = 0;
      for (const FuModule &child : fuModules) {
        if (fabric.parentPeOf(child.reference) != pe)
          continue;
        ++childCount;
        childEndpointCount += child.endpoints.size();
        childConfigurationPortCount += !child.configuration.empty();
        childConfigurationMemberCount += child.configuration.words.size();
        for (const EndpointPlan &endpoint : child.endpoints) {
          const fabric::FabricFuOccurrencePortRef port{
              child.reference, endpoint.direction, endpoint.localOrdinal};
          attachmentCount += fabric.fuOccurrencePortAttachments(port).size();
        }
      }
      emitInvocationDiagnostic(
          DiagnosticVerbosity::Summary,
          InvocationDiagnosticStage::HardwareConfiguration,
          InvocationDiagnosticEvent::Statistics, [&] {
            return llvm::json::Value(llvm::json::Object{
                {"statistics_kind", "rtl_pe_materialization_shape"},
                {"materialization_key", materializationKey.str()},
                {"pe_ordinal", pe.id()},
                {"schedule", temporal ? "temporal" : "spatial"},
                {"child_fu_count", childCount},
                {"child_fu_endpoint_count", childEndpointCount},
                {"child_configuration_port_count", childConfigurationPortCount},
                {"child_configuration_bundle_member_count",
                 childConfigurationMemberCount},
                {"fu_port_attachment_count", attachmentCount}});
          });
    }
    RtlMaterializationStageTracker stage(
        temporal ? "skeleton_temporal_pe_module" : "skeleton_spatial_pe_module",
        peKey, container);
    auto module =
        temporal ? buildTemporalPeModule(builder, location, spatialCore, fabric,
                                         configurationAbi, transportLayout,
                                         fuModules, clockReset, pe, peKey)
                 : buildSpatialPeModule(builder, location, spatialCore, fabric,
                                        configurationAbi, transportLayout,
                                        fuModules, pe);
    if (!module)
      return module.takeError();
    result.push_back(std::move(*module));
    stage.finish(container);
  }
  return result;
}

} // namespace loom::hardware::rtl::hierarchy

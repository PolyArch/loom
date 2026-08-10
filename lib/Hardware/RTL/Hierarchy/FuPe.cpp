#include "Components.h"

#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"

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

void addCommonInstanceInputs(
    circt::hw::HWModulePortAccessor &accessor,
    const ConfigurationABI &configurationAbi,
    const ConfigurationTransportLayout &transportLayout,
    std::map<std::string, mlir::Value> &inputs) {
  inputs.emplace("clock", accessor.getInput("clock"));
  inputs.emplace("reset", accessor.getInput("reset"));
  (void)configurationAbi;
  for (auto [ordinal, unit] : llvm::enumerate(transportLayout.units)) {
    (void)unit;
    inputs.emplace(configurationPortName(ordinal),
                   accessor.getInput(configurationPortName(ordinal)));
  }
}

mlir::Value zeroData(mlir::OpBuilder &builder, mlir::Location location,
                     unsigned width) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, 0));
}

llvm::Expected<FuModule>
buildFuModule(mlir::OpBuilder &builder, mlir::Location location,
              fabric::SpatialCoreOccurrenceRef spatialCore,
              const fabric::FabricArtifactView &fabric,
              const ConfigurationABI &configurationAbi,
              const ConfigurationTransportLayout &transportLayout,
              llvm::ArrayRef<OperationShellModule> operationShells,
              fabric::FabricFuOccurrenceRef fu) {
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
  auto prepared =
      prepareFiniteField(spatialCore, field, configurationAbi, transportLayout);
  if (!prepared)
    return prepared.takeError();

  struct TemplatePlan final {
    llvm::APInt activeCode;
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
    auto activeCode = physicalCode(*prepared->second, semantic->bytes());
    if (!activeCode)
      return activeCode.takeError();
    auto edges = fabric::projectFabricFuCapabilityTemplateTerminalEdges(record);
    if (!edges)
      return edges.takeError();
    std::set<std::string> destinations;
    for (const auto &edge : *edges)
      if (!destinations.insert(fuTemplateEndpointKey(edge.destination)).second)
        return invalid(
            "one FU capability template drives a terminal more than once");
    templatePlans.push_back(
        TemplatePlan{std::move(*activeCode), std::move(*edges)});
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

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  appendClockResetAndConfigurationPorts(builder, configurationAbi,
                                        transportLayout, inputs);
  for (const EndpointPlan &endpoint : *endpoints)
    appendEndpointPorts(inputs, outputs, endpoint);
  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_fabric_fu_" + std::to_string(fu.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value fieldSignal =
            decodeFieldSignal(bodyBuilder, location, accessor, prepared->first);
        std::vector<mlir::Value> active;
        active.reserve(templatePlans.size());
        for (const TemplatePlan &plan : templatePlans)
          active.push_back(
              matchesCode(bodyBuilder, location, fieldSignal, plan.activeCode));

        circt::BackedgeBuilder backedges(bodyBuilder, location);
        std::map<std::string, circt::Backedge> dataInput;
        std::map<std::string, circt::Backedge> validInput;
        std::map<std::string, circt::Backedge> readyInput;
        std::map<std::string, mlir::Value> dataOutput;
        std::map<std::string, mlir::Value> validOutput;
        std::map<std::string, mlir::Value> readyOutput;
        for (const OperationShellModule *operation : operations) {
          std::map<std::string, mlir::Value> instanceInputs;
          addCommonInstanceInputs(accessor, configurationAbi, transportLayout,
                                  instanceInputs);
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
          mlir::Value destinationReady;
          std::optional<mlir::Value> data;
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
            -> llvm::Expected<ForwardTransportSignals> {
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
            return ForwardTransportSignals{
                accessor.getInput(endpoint->valid.getName()),
                endpoint->data ? std::optional<mlir::Value>{accessor.getInput(
                                     endpoint->data->getName())}
                               : std::nullopt,
                endpoint->tag ? std::optional<mlir::Value>{accessor.getInput(
                                    endpoint->tag->getName())}
                              : std::nullopt};
          }
          const auto valid = validOutput.find(key);
          if (valid == validOutput.end())
            return invalid("FU route source operation signal is absent");
          const auto data = dataOutput.find(key);
          return ForwardTransportSignals{
              valid->second,
              data == dataOutput.end()
                  ? std::nullopt
                  : std::optional<mlir::Value>{data->second},
              std::nullopt};
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
            auto adapted = adaptForwardTransportSignals(
                bodyBuilder, location, *sourceType, *destinationType, *source);
            if (!adapted) {
              materializationError = llvm::toString(adapted.takeError());
              backedges.abandon();
              return;
            }
            routes.push_back(RouteRuntime{
                templateOrdinal, &edge, fuTemplateEndpointKey(edge.source),
                fuTemplateEndpointKey(edge.destination),
                active[templateOrdinal], source->valid, *ready,
                adapted->payload});
          }

        const auto sourceReady = [&](llvm::StringRef sourceKey) {
          llvm::SmallVector<mlir::Value> alternatives;
          for (std::size_t templateOrdinal = 0;
               templateOrdinal < templatePlans.size(); ++templateOrdinal) {
            llvm::SmallVector<mlir::Value> destinations;
            for (const RouteRuntime &route : routes)
              if (route.templateOrdinal == templateOrdinal &&
                  route.sourceKey == sourceKey)
                destinations.push_back(route.destinationReady);
            if (!destinations.empty())
              alternatives.push_back(
                  andValues(bodyBuilder, location,
                            {active[templateOrdinal],
                             andValues(bodyBuilder, location, destinations)}));
          }
          return orValues(bodyBuilder, location, alternatives);
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
          llvm::SmallVector<mlir::Value> validTerms;
          mlir::Value data = endpoint.data
                                 ? zeroData(bodyBuilder, location,
                                            endpoint.dataPath.payloadWidthBits)
                                 : mlir::Value{};
          mlir::Value tag = endpoint.tag
                                ? zeroData(bodyBuilder, location,
                                           endpoint.dataPath.tagWidthBits)
                                : mlir::Value{};
          for (const RouteRuntime &route : routes) {
            if (route.destinationKey != key)
              continue;
            llvm::SmallVector<mlir::Value> peers;
            for (const RouteRuntime &peer : routes)
              if (peer.templateOrdinal == route.templateOrdinal &&
                  peer.sourceKey == route.sourceKey &&
                  peer.destinationKey != route.destinationKey)
                peers.push_back(peer.destinationReady);
            validTerms.push_back(
                andValues(bodyBuilder, location,
                          {route.active, route.sourceValid,
                           andValues(bodyBuilder, location, peers)}));
            if (endpoint.data && route.data)
              data = circt::comb::MuxOp::create(
                  bodyBuilder, location, route.active, *route.data, data, true);
          }
          if (endpoint.data)
            accessor.setOutput(endpoint.data->getName(), data);
          if (endpoint.tag)
            accessor.setOutput(endpoint.tag->getName(), tag);
          accessor.setOutput(endpoint.valid.getName(),
                             orValues(bodyBuilder, location, validTerms));
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
            llvm::SmallVector<mlir::Value> validTerms;
            mlir::Value data =
                endpoint.data
                    ? zeroData(bodyBuilder, location, endpoint.payloadWidthBits)
                    : mlir::Value{};
            for (const RouteRuntime &route : routes) {
              if (route.destinationKey != key)
                continue;
              llvm::SmallVector<mlir::Value> peers;
              for (const RouteRuntime &peer : routes)
                if (peer.templateOrdinal == route.templateOrdinal &&
                    peer.sourceKey == route.sourceKey &&
                    peer.destinationKey != route.destinationKey)
                  peers.push_back(peer.destinationReady);
              validTerms.push_back(
                  andValues(bodyBuilder, location,
                            {route.active, route.sourceValid,
                             andValues(bodyBuilder, location, peers)}));
              if (endpoint.data && route.data)
                data = circt::comb::MuxOp::create(bodyBuilder, location,
                                                  route.active, *route.data,
                                                  data, true);
            }
            if (endpoint.data)
              dataInput.at(key).setValue(data);
            validInput.at(key).setValue(
                orValues(bodyBuilder, location, validTerms));
          }
      });
  if (materializationError)
    return invalid(*materializationError);
  return FuModule{fu, module, std::move(*endpoints)};
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
    return unsupported("Temporal PE hierarchy lowering is not implemented");
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

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  appendClockResetAndConfigurationPorts(builder, configurationAbi,
                                        transportLayout, inputs);
  for (const EndpointPlan &endpoint : *endpoints)
    appendEndpointPorts(inputs, outputs, endpoint);
  std::optional<std::string> materializationError;
  auto module = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_spatial_pe_" + std::to_string(pe.id())),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value activation = decodeFieldSignal(
            bodyBuilder, location, accessor, activationPrepared->first);
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
          addCommonInstanceInputs(accessor, configurationAbi, transportLayout,
                                  instanceInputs);
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
            mlir::Value field = decodeFieldSignal(bodyBuilder, location,
                                                  accessor, prepared->first);
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
  return PeModule{pe, module, std::move(*endpoints)};
}

} // namespace

llvm::Expected<std::vector<FuModule>>
buildFuModules(mlir::OpBuilder &builder, mlir::Location location,
               fabric::SpatialCoreOccurrenceRef spatialCore,
               const fabric::FabricArtifactView &fabric,
               const ConfigurationABI &configurationAbi,
               const ConfigurationTransportLayout &transportLayout,
               llvm::ArrayRef<OperationShellModule> operationShells,
               const ClockResetPlan &) {
  std::vector<FuModule> result;
  result.reserve(fabric.fuOccurrences().size());
  for (fabric::FabricFuOccurrenceRef fu : fabric.fuOccurrences()) {
    auto module =
        buildFuModule(builder, location, spatialCore, fabric, configurationAbi,
                      transportLayout, operationShells, fu);
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
               llvm::ArrayRef<FuModule> fuModules, const ClockResetPlan &) {
  std::vector<PeModule> result;
  result.reserve(fabric.peOccurrences().size());
  for (fabric::FabricPeOccurrenceRef pe : fabric.peOccurrences()) {
    auto module =
        buildSpatialPeModule(builder, location, spatialCore, fabric,
                             configurationAbi, transportLayout, fuModules, pe);
    if (!module)
      return module.takeError();
    result.push_back(std::move(*module));
  }
  return result;
}

} // namespace loom::hardware::rtl::hierarchy

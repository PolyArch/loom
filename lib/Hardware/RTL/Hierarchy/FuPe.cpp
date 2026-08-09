#include "Components.h"

#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "circt/Dialect/Comb/CombOps.h"
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

void addCommonInstanceInputs(circt::hw::HWModulePortAccessor &accessor,
                             const ConfigurationABI &configurationAbi,
                             std::map<std::string, mlir::Value> &inputs) {
  inputs.emplace("clock", accessor.getInput("clock"));
  inputs.emplace("reset", accessor.getInput("reset"));
  for (const ProgrammingUnit &unit : configurationAbi.programmingUnits())
    inputs.emplace(configurationPortName(unit.id),
                   accessor.getInput(configurationPortName(unit.id)));
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
  if (templates.size() != 1 || templates.front().activeNodes.size() != 1 ||
      templates.front().activeNodes.front().node !=
          fabric::FabricFuNodeKind::Op)
    return unsupported(
        "hierarchical FU lowering currently requires one direct operation "
        "template");
  auto localOperation = fabric::deriveFabricFuOccurrenceNode(
      fabric, templates.front().activeNodes.front(), fu);
  if (!localOperation)
    return localOperation.takeError();
  const OperationShellModule *operation =
      findOperationShell(operationShells, *localOperation);
  if (!operation)
    return invalid("FU operation has no unique operation shell");

  auto terminalEdges =
      fabric::projectFabricFuCapabilityTemplateTerminalEdges(templates.front());
  if (!terminalEdges)
    return terminalEdges.takeError();
  std::map<fabric::FabricOrdinal, fabric::FabricOrdinal> inputBoundaryByPort;
  std::map<fabric::FabricOrdinal, fabric::FabricOrdinal> outputBoundaryByPort;
  for (const auto &edge : *terminalEdges) {
    if (edge.source.kind() ==
            fabric::FabricFuCapabilityTemplateEndpointKind::BoundaryPort &&
        edge.destination.kind() ==
            fabric::FabricFuCapabilityTemplateEndpointKind::NodePort) {
      const auto &source =
          std::get<fabric::FabricFuTemplatePortRef>(edge.source.payload);
      const auto &destination =
          std::get<fabric::FabricFuNodePortRef>(edge.destination.payload);
      if (destination.node != templates.front().activeNodes.front() ||
          source.direction != fabric::FabricPortDirection::Input ||
          destination.direction != fabric::FabricPortDirection::Input ||
          !inputBoundaryByPort.emplace(destination.ordinal, source.ordinal)
               .second)
        return invalid("FU input terminal relation is not one-to-one");
      continue;
    }
    if (edge.source.kind() ==
            fabric::FabricFuCapabilityTemplateEndpointKind::NodePort &&
        edge.destination.kind() ==
            fabric::FabricFuCapabilityTemplateEndpointKind::BoundaryPort) {
      const auto &source =
          std::get<fabric::FabricFuNodePortRef>(edge.source.payload);
      const auto &destination =
          std::get<fabric::FabricFuTemplatePortRef>(edge.destination.payload);
      if (source.node != templates.front().activeNodes.front() ||
          source.direction != fabric::FabricPortDirection::Output ||
          destination.direction != fabric::FabricPortDirection::Output ||
          !outputBoundaryByPort.emplace(source.ordinal, destination.ordinal)
               .second)
        return invalid("FU output terminal relation is not one-to-one");
      continue;
    }
    return invalid("direct FU template contains a non-terminal edge");
  }

  const fabric::FabricSemanticConfigFieldRef field{
      fabric::FabricConfigurationOwnerRef(
          fabric::FabricInventoryOwnerRef::of(fu)),
      0};
  auto prepared = prepareFiniteField(spatialCore, field, configurationAbi);
  if (!prepared)
    return prepared.takeError();
  auto semantic = fabric::encodeFabricFuConfiguration(
      fabric, field, fabric::FabricFuCapabilityTemplateRef{*definition, 0});
  if (!semantic)
    return semantic.takeError();
  auto activeCode = physicalCode(*prepared->second, semantic->bytes());
  if (!activeCode)
    return activeCode.takeError();

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  appendClockResetAndConfigurationPorts(builder, configurationAbi, inputs);
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
        mlir::Value active =
            matchesCode(bodyBuilder, location, fieldSignal, *activeCode);
        std::map<std::string, mlir::Value> instanceInputs;
        addCommonInstanceInputs(accessor, configurationAbi, instanceInputs);
        for (const OperationEndpointPlan &port : operation->endpoints) {
          if (port.direction == fabric::FabricPortDirection::Input) {
            const auto found = inputBoundaryByPort.find(port.ordinal);
            const EndpointPlan *boundary =
                found == inputBoundaryByPort.end()
                    ? nullptr
                    : findEndpoint(*endpoints,
                                   fabric::FabricPortDirection::Input,
                                   found->second);
            if (!boundary) {
              materializationError =
                  "operation input has no FU boundary source";
              return;
            }
            if (port.data) {
              auto adapted = adaptForwardTransportSignals(
                  bodyBuilder, location, boundary->dataPath,
                  ::fabric::DataPathType{::fabric::DataPathKind::Bits,
                                         port.payloadWidthBits, 0},
                  ForwardTransportSignals{
                      accessor.getInput(boundary->valid.getName()),
                      boundary->data
                          ? std::optional<mlir::Value>{accessor.getInput(
                                boundary->data->getName())}
                          : std::nullopt,
                      std::nullopt});
              if (!adapted) {
                materializationError = llvm::toString(adapted.takeError());
                return;
              }
              instanceInputs.emplace(port.data->getName().str(),
                                     *adapted->payload);
            }
            instanceInputs.emplace(
                port.valid.getName().str(),
                circt::comb::AndOp::create(
                    bodyBuilder, location, active,
                    accessor.getInput(boundary->valid.getName())));
          } else {
            const auto found = outputBoundaryByPort.find(port.ordinal);
            const EndpointPlan *boundary =
                found == outputBoundaryByPort.end()
                    ? nullptr
                    : findEndpoint(*endpoints,
                                   fabric::FabricPortDirection::Output,
                                   found->second);
            if (!boundary) {
              materializationError = "operation output has no FU boundary sink";
              return;
            }
            instanceInputs.emplace(
                port.ready.getName().str(),
                circt::comb::AndOp::create(
                    bodyBuilder, location, active,
                    accessor.getInput(boundary->ready.getName())));
          }
        }
        auto instance =
            instantiateModule(bodyBuilder, location, operation->module,
                              "operation", instanceInputs);
        if (!instance) {
          materializationError = llvm::toString(instance.takeError());
          return;
        }
        for (const OperationEndpointPlan &port : operation->endpoints) {
          if (port.direction == fabric::FabricPortDirection::Input) {
            const EndpointPlan *boundary =
                findEndpoint(*endpoints, fabric::FabricPortDirection::Input,
                             inputBoundaryByPort.at(port.ordinal));
            accessor.setOutput(boundary->ready.getName(),
                               circt::comb::AndOp::create(
                                   bodyBuilder, location, active,
                                   instance->at(port.ready.getName().str())));
            continue;
          }
          const EndpointPlan *boundary =
              findEndpoint(*endpoints, fabric::FabricPortDirection::Output,
                           outputBoundaryByPort.at(port.ordinal));
          if (boundary->data) {
            auto adapted = adaptForwardTransportSignals(
                bodyBuilder, location,
                ::fabric::DataPathType{::fabric::DataPathKind::Bits,
                                       port.payloadWidthBits, 0},
                boundary->dataPath,
                ForwardTransportSignals{
                    instance->at(port.valid.getName().str()),
                    port.data ? std::optional<mlir::Value>{instance->at(
                                    port.data->getName().str())}
                              : std::nullopt,
                    std::nullopt});
            if (!adapted) {
              materializationError = llvm::toString(adapted.takeError());
              return;
            }
            accessor.setOutput(boundary->data->getName(), *adapted->payload);
          }
          accessor.setOutput(boundary->valid.getName(),
                             circt::comb::AndOp::create(
                                 bodyBuilder, location, active,
                                 instance->at(port.valid.getName().str())));
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
  auto activationPrepared = prepareFiniteField(
      spatialCore, activationField->reference, configurationAbi);
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
  appendClockResetAndConfigurationPorts(builder, configurationAbi, inputs);
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
          addCommonInstanceInputs(accessor, configurationAbi, instanceInputs);
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
            auto prepared = prepareFiniteField(
                spatialCore, descriptor->reference, configurationAbi);
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
               llvm::ArrayRef<OperationShellModule> operationShells,
               const ClockResetPlan &) {
  std::vector<FuModule> result;
  result.reserve(fabric.fuOccurrences().size());
  for (fabric::FabricFuOccurrenceRef fu : fabric.fuOccurrences()) {
    auto module = buildFuModule(builder, location, spatialCore, fabric,
                                configurationAbi, operationShells, fu);
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
               llvm::ArrayRef<FuModule> fuModules, const ClockResetPlan &) {
  std::vector<PeModule> result;
  result.reserve(fabric.peOccurrences().size());
  for (fabric::FabricPeOccurrenceRef pe : fabric.peOccurrences()) {
    auto module = buildSpatialPeModule(builder, location, spatialCore, fabric,
                                       configurationAbi, fuModules, pe);
    if (!module)
      return module.takeError();
    result.push_back(std::move(*module));
  }
  return result;
}

} // namespace loom::hardware::rtl::hierarchy

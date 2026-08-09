#include "ModuleHierarchy.h"

#include "Components.h"
#include "OperationShell.h"
#include "Support.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/RTL/PhysicalOperation.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace loom::hardware::rtl::hierarchy {
namespace {

struct ComponentView final {
  fabric::FabricTransportEndpointOwnerRef owner;
  circt::hw::HWModuleOp module;
  llvm::ArrayRef<EndpointPlan> endpoints;
  std::string instanceName;
};

struct PendingSignal final {
  circt::Backedge edge;
  bool resolved = false;
};

struct EndpointRuntime final {
  const EndpointPlan *plan = nullptr;
  std::optional<PendingSignal> dataInput;
  std::optional<PendingSignal> tagInput;
  std::optional<PendingSignal> validInput;
  std::optional<PendingSignal> readyInput;
  std::optional<mlir::Value> dataOutput;
  std::optional<mlir::Value> tagOutput;
  mlir::Value validOutput;
  mlir::Value readyOutput;
};

std::string
boundaryKey(const fabric::FabricModuleBoundaryEndpointRef &boundary) {
  const std::vector<std::uint8_t> bytes =
      fabric::canonicalFabricBytes(boundary);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

const ModuleBoundaryTransportPortProjection *findBoundaryProjection(
    llvm::ArrayRef<ModuleBoundaryTransportPortProjection> projections,
    const fabric::FabricModuleBoundaryEndpointRef &boundary) {
  const ModuleBoundaryTransportPortProjection *result = nullptr;
  for (const auto &projection : projections)
    if (projection.boundary == boundary) {
      if (result)
        return nullptr;
      result = &projection;
    }
  return result;
}

void appendBoundaryPorts(
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs,
    const ModuleBoundaryTransportPortProjection &projection) {
  const auto append = [&](const circt::hw::PortInfo &port) {
    (port.isOutput() ? outputs : inputs).push_back(port);
  };
  if (projection.data)
    append(*projection.data);
  if (projection.tag)
    append(*projection.tag);
  append(projection.valid);
  append(projection.ready);
}

template <typename Component>
void appendComponents(llvm::ArrayRef<Component> components,
                      llvm::StringRef prefix,
                      std::vector<ComponentView> &result) {
  for (const Component &component : components)
    result.push_back(
        {fabric::FabricTransportEndpointOwnerRef::of(component.reference),
         component.module, component.endpoints,
         prefix.str() + std::to_string(component.reference.id())});
}

llvm::Expected<std::vector<ResolvedFabricPhysicalOperation>>
selectOperations(const ConfigurationABI &configurationAbi,
                 fabric::SpatialCoreOccurrenceRef spatialCore) {
  auto all = enumerateFabricPhysicalOperations(configurationAbi.fabricSystem());
  if (!all)
    return all.takeError();
  std::vector<ResolvedFabricPhysicalOperation> selected;
  for (ResolvedFabricPhysicalOperation &operation : *all) {
    if (operation.physicalOccurrence.kind() !=
        fabric::FabricPhysicalOccurrenceOwnerKind::SpatialCoreInternal)
      continue;
    const auto &internal = std::get<fabric::SpatialCoreInternalOccurrenceRef>(
        operation.physicalOccurrence.payload());
    if (internal.spatialCore == spatialCore)
      selected.push_back(std::move(operation));
  }
  return selected;
}

llvm::Error setPending(PendingSignal &pending, mlir::Value value,
                       llvm::StringRef description) {
  if (pending.resolved)
    return invalid(description + " is structurally driven more than once");
  const mlir::Value carrier = pending.edge;
  if (!value || value.getType() != carrier.getType())
    return invalid(description + " has an incompatible structural type");
  pending.edge.setValue(value);
  pending.resolved = true;
  return llvm::Error::success();
}

llvm::Expected<ForwardTransportSignals>
forwardSignals(const EndpointRuntime &runtime) {
  if (!runtime.plan ||
      runtime.plan->direction != fabric::FabricPortDirection::Output ||
      !runtime.validOutput)
    return invalid("structural source is not one output endpoint");
  return ForwardTransportSignals{runtime.validOutput, runtime.dataOutput,
                                 runtime.tagOutput};
}

llvm::Error driveForward(EndpointRuntime &runtime,
                         const ForwardTransportSignals &signals) {
  if (!runtime.plan ||
      runtime.plan->direction != fabric::FabricPortDirection::Input ||
      !runtime.validInput)
    return invalid("structural destination is not one input endpoint");
  if (runtime.dataInput) {
    if (!signals.payload)
      return invalid("structural destination payload is absent");
    if (llvm::Error error =
            setPending(*runtime.dataInput, *signals.payload, "endpoint data"))
      return error;
  } else if (signals.payload) {
    return invalid("payload reaches a zero-width structural destination");
  }
  if (runtime.tagInput) {
    if (!signals.tag)
      return invalid("structural destination tag is absent");
    if (llvm::Error error =
            setPending(*runtime.tagInput, *signals.tag, "endpoint tag"))
      return error;
  } else if (signals.tag) {
    return invalid("tag reaches an untagged structural destination");
  }
  return setPending(*runtime.validInput, signals.valid, "endpoint valid");
}

llvm::Error driveReady(EndpointRuntime &runtime, mlir::Value ready) {
  if (!runtime.plan ||
      runtime.plan->direction != fabric::FabricPortDirection::Output ||
      !runtime.readyInput)
    return invalid("structural source has no reverse ready carrier");
  return setPending(*runtime.readyInput, ready, "endpoint ready");
}

mlir::Value zero(mlir::OpBuilder &builder, mlir::Location location,
                 mlir::Type type) {
  const unsigned width = mlir::cast<mlir::IntegerType>(type).getWidth();
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(width, 0));
}

} // namespace

llvm::Expected<ModuleRootCirctSkeleton> buildModuleHierarchySkeleton(
    mlir::MLIRContext &context, fabric::SpatialCoreOccurrenceRef spatialCore,
    const ConfigurationABI &configurationAbi,
    const fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<ModuleBoundaryTransportPortProjection> projections) {
  if (!fabric.memoryOccurrences().empty() ||
      !fabric.moduleBoundaryMemoryAttachments().empty())
    return unsupported(
        "memory hierarchy lowering requires the portable memory shell");
  auto clockReset =
      prepareClockReset(configurationAbi.fabricSystem(), spatialCore);
  if (!clockReset)
    return clockReset.takeError();
  auto operations = selectOperations(configurationAbi, spatialCore);
  if (!operations)
    return operations.takeError();

  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));

  std::vector<FabricOperationLeafAssociation> associations;
  auto operationShells = buildOperationShellModules(
      builder, location, spatialCore, configurationAbi, *operations,
      associations, *clockReset);
  if (!operationShells)
    return operationShells.takeError();
  auto fuModules =
      buildFuModules(builder, location, spatialCore, fabric, configurationAbi,
                     *operationShells, *clockReset);
  if (!fuModules)
    return fuModules.takeError();
  auto peModules = buildPeModules(builder, location, spatialCore, fabric,
                                  configurationAbi, *fuModules, *clockReset);
  if (!peModules)
    return peModules.takeError();
  auto switchModules = buildSwitchModules(builder, location, spatialCore,
                                          fabric, configurationAbi);
  if (!switchModules)
    return switchModules.takeError();
  auto fifoModules = buildFifoModules(builder, location, spatialCore, fabric,
                                      configurationAbi, *clockReset);
  if (!fifoModules)
    return fifoModules.takeError();
  auto boundaryModules = buildBoundaryModules(builder, location, spatialCore,
                                              fabric, configurationAbi);
  if (!boundaryModules)
    return boundaryModules.takeError();

  std::vector<ComponentView> components;
  components.reserve(peModules->size() + switchModules->size() +
                     fifoModules->size() + boundaryModules->size());
  appendComponents(llvm::ArrayRef(*peModules), "pe_", components);
  appendComponents(llvm::ArrayRef(*switchModules), "switch_", components);
  appendComponents(llvm::ArrayRef(*fifoModules), "fifo_", components);
  appendComponents(llvm::ArrayRef(*boundaryModules), "boundary_", components);

  llvm::SmallVector<circt::hw::PortInfo, 32> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 32> outputs;
  appendClockResetAndConfigurationPorts(builder, configurationAbi, inputs);
  for (const auto &projection : projections)
    appendBoundaryPorts(inputs, outputs, projection);

  std::optional<std::string> materializationError;
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("loom_module"),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        const auto fail = [&](llvm::Error error) {
          materializationError = llvm::toString(std::move(error));
          backedges.abandon();
        };
        mlir::Value reset = accessor.getInput("reset");
        if (clockReset->activeLowReset)
          reset = circt::comb::createOrFoldNot(bodyBuilder, location, reset);
        std::map<std::string, EndpointRuntime> runtime;

        for (const ComponentView &component : components) {
          std::map<std::string, mlir::Value> instanceInputs;
          instanceInputs.emplace("clock", accessor.getInput("clock"));
          instanceInputs.emplace("reset", reset);
          for (const ProgrammingUnit &unit :
               configurationAbi.programmingUnits())
            instanceInputs.emplace(
                configurationPortName(unit.id),
                accessor.getInput(configurationPortName(unit.id)));

          std::vector<EndpointRuntime *> pendingRuntime;
          for (const EndpointPlan &endpoint : component.endpoints) {
            EndpointRuntime endpointRuntime;
            endpointRuntime.plan = &endpoint;
            if (endpoint.direction == fabric::FabricPortDirection::Input) {
              if (endpoint.data) {
                endpointRuntime.dataInput =
                    PendingSignal{backedges.get(endpoint.data->type), false};
                instanceInputs.emplace(endpoint.data->getName().str(),
                                       endpointRuntime.dataInput->edge);
              }
              if (endpoint.tag) {
                endpointRuntime.tagInput =
                    PendingSignal{backedges.get(endpoint.tag->type), false};
                instanceInputs.emplace(endpoint.tag->getName().str(),
                                       endpointRuntime.tagInput->edge);
              }
              endpointRuntime.validInput =
                  PendingSignal{backedges.get(endpoint.valid.type), false};
              instanceInputs.emplace(endpoint.valid.getName().str(),
                                     endpointRuntime.validInput->edge);
            } else {
              endpointRuntime.readyInput =
                  PendingSignal{backedges.get(endpoint.ready.type), false};
              instanceInputs.emplace(endpoint.ready.getName().str(),
                                     endpointRuntime.readyInput->edge);
            }
            const std::string key = endpointKey(endpoint.endpoint);
            auto [position, inserted] =
                runtime.emplace(key, std::move(endpointRuntime));
            if (!inserted) {
              fail(invalid("component endpoint identity is duplicated"));
              return;
            }
            pendingRuntime.push_back(&position->second);
          }

          auto instance =
              instantiateModule(bodyBuilder, location, component.module,
                                component.instanceName, instanceInputs);
          if (!instance) {
            fail(instance.takeError());
            return;
          }
          for (EndpointRuntime *endpoint : pendingRuntime) {
            if (endpoint->plan->direction ==
                fabric::FabricPortDirection::Input) {
              endpoint->readyOutput =
                  instance->at(endpoint->plan->ready.getName().str());
              continue;
            }
            if (endpoint->plan->data)
              endpoint->dataOutput =
                  instance->at(endpoint->plan->data->getName().str());
            if (endpoint->plan->tag)
              endpoint->tagOutput =
                  instance->at(endpoint->plan->tag->getName().str());
            endpoint->validOutput =
                instance->at(endpoint->plan->valid.getName().str());
          }
        }

        for (const fabric::FabricPointConnectionPayload &connection :
             fabric.pointConnections()) {
          auto source = runtime.find(endpointKey(connection.source));
          auto destination = runtime.find(endpointKey(connection.destination));
          if (source == runtime.end() || destination == runtime.end()) {
            fail(invalid("point connection names a non-top-level endpoint"));
            return;
          }
          auto forward = forwardSignals(source->second);
          if (!forward) {
            fail(forward.takeError());
            return;
          }
          auto adapted = adaptFabricPointConnectionForwardSignals(
              bodyBuilder, location, fabric, connection, *forward);
          if (!adapted) {
            fail(adapted.takeError());
            return;
          }
          if (llvm::Error error = driveForward(destination->second, *adapted)) {
            fail(std::move(error));
            return;
          }
          if (llvm::Error error =
                  driveReady(source->second, destination->second.readyOutput)) {
            fail(std::move(error));
            return;
          }
        }

        std::set<std::string> connectedBoundaries;
        for (const auto &attachment :
             fabric.moduleBoundaryTransportAttachments()) {
          const auto *projection =
              findBoundaryProjection(projections, attachment.boundary);
          auto endpoint = runtime.find(endpointKey(attachment.endpoint));
          if (!projection || endpoint == runtime.end() ||
              !connectedBoundaries.insert(boundaryKey(attachment.boundary))
                   .second) {
            fail(invalid("Module boundary attachment is not unique"));
            return;
          }
          const auto boundaryType =
              fabric.moduleBoundaryEndpointDataPath(attachment.boundary);
          if (!boundaryType) {
            fail(invalid("Module boundary attachment has no transport type"));
            return;
          }
          if (attachment.boundary.direction ==
              fabric::FabricPortDirection::Input) {
            auto adapted = adaptForwardTransportSignals(
                bodyBuilder, location, *boundaryType,
                endpoint->second.plan->dataPath,
                ForwardTransportSignals{
                    accessor.getInput(projection->valid.getName()),
                    projection->data
                        ? std::optional<mlir::Value>{accessor.getInput(
                              projection->data->getName())}
                        : std::nullopt,
                    projection->tag
                        ? std::optional<mlir::Value>{accessor.getInput(
                              projection->tag->getName())}
                        : std::nullopt});
            if (!adapted) {
              fail(adapted.takeError());
              return;
            }
            if (llvm::Error error = driveForward(endpoint->second, *adapted)) {
              fail(std::move(error));
              return;
            }
            accessor.setOutput(projection->ready.getName(),
                               endpoint->second.readyOutput);
            continue;
          }
          auto forward = forwardSignals(endpoint->second);
          if (!forward) {
            fail(forward.takeError());
            return;
          }
          auto adapted = adaptForwardTransportSignals(
              bodyBuilder, location, endpoint->second.plan->dataPath,
              *boundaryType, *forward);
          if (!adapted) {
            fail(adapted.takeError());
            return;
          }
          accessor.setOutput(projection->valid.getName(), adapted->valid);
          if (projection->data)
            accessor.setOutput(projection->data->getName(), *adapted->payload);
          if (projection->tag)
            accessor.setOutput(projection->tag->getName(), *adapted->tag);
          if (llvm::Error error =
                  driveReady(endpoint->second,
                             accessor.getInput(projection->ready.getName()))) {
            fail(std::move(error));
            return;
          }
        }

        for (const auto &passthrough :
             fabric.moduleBoundaryTransportPassthroughs()) {
          const auto *input =
              findBoundaryProjection(projections, passthrough.input);
          const auto *output =
              findBoundaryProjection(projections, passthrough.output);
          const auto inputType =
              fabric.moduleBoundaryEndpointDataPath(passthrough.input);
          const auto outputType =
              fabric.moduleBoundaryEndpointDataPath(passthrough.output);
          if (!input || !output || !inputType || !outputType ||
              !connectedBoundaries.insert(boundaryKey(passthrough.input))
                   .second ||
              !connectedBoundaries.insert(boundaryKey(passthrough.output))
                   .second) {
            fail(invalid("Module boundary passthrough is not unique"));
            return;
          }
          auto adapted = adaptForwardTransportSignals(
              bodyBuilder, location, *inputType, *outputType,
              ForwardTransportSignals{
                  accessor.getInput(input->valid.getName()),
                  input->data ? std::optional<mlir::Value>{accessor.getInput(
                                    input->data->getName())}
                              : std::nullopt,
                  input->tag ? std::optional<mlir::Value>{accessor.getInput(
                                   input->tag->getName())}
                             : std::nullopt});
          if (!adapted) {
            fail(adapted.takeError());
            return;
          }
          accessor.setOutput(output->valid.getName(), adapted->valid);
          if (output->data)
            accessor.setOutput(output->data->getName(), *adapted->payload);
          if (output->tag)
            accessor.setOutput(output->tag->getName(), *adapted->tag);
          accessor.setOutput(input->ready.getName(),
                             accessor.getInput(output->ready.getName()));
        }

        mlir::Value falseValue = bitConstant(bodyBuilder, location, false);
        for (auto &[key, endpoint] : runtime) {
          if (endpoint.dataInput && !endpoint.dataInput->resolved)
            if (llvm::Error error = setPending(
                    *endpoint.dataInput,
                    zero(bodyBuilder, location,
                         mlir::Value(endpoint.dataInput->edge).getType()),
                    "unused endpoint data")) {
              fail(std::move(error));
              return;
            }
          if (endpoint.tagInput && !endpoint.tagInput->resolved)
            if (llvm::Error error = setPending(
                    *endpoint.tagInput,
                    zero(bodyBuilder, location,
                         mlir::Value(endpoint.tagInput->edge).getType()),
                    "unused endpoint tag")) {
              fail(std::move(error));
              return;
            }
          if (endpoint.validInput && !endpoint.validInput->resolved)
            if (llvm::Error error = setPending(*endpoint.validInput, falseValue,
                                               "unused endpoint valid")) {
              fail(std::move(error));
              return;
            }
          if (endpoint.readyInput && !endpoint.readyInput->resolved)
            if (llvm::Error error = setPending(*endpoint.readyInput, falseValue,
                                               "unused endpoint ready")) {
              fail(std::move(error));
              return;
            }
        }

        for (const auto &projection : projections) {
          if (connectedBoundaries.count(boundaryKey(projection.boundary)))
            continue;
          if (projection.boundary.direction ==
              fabric::FabricPortDirection::Input) {
            accessor.setOutput(projection.ready.getName(), falseValue);
            continue;
          }
          if (projection.data)
            accessor.setOutput(
                projection.data->getName(),
                zero(bodyBuilder, location, projection.data->type));
          if (projection.tag)
            accessor.setOutput(
                projection.tag->getName(),
                zero(bodyBuilder, location, projection.tag->type));
          accessor.setOutput(projection.valid.getName(), falseValue);
        }
      });
  if (materializationError)
    return invalid(*materializationError);

  ModuleRootCirctSkeleton result{std::move(module), std::move(associations)};
  if (llvm::Error error = verifyCommonCirctSkeleton(
          *result.module, configurationAbi, result.operationLeaves))
    return std::move(error);
  return result;
}

} // namespace loom::hardware::rtl::hierarchy

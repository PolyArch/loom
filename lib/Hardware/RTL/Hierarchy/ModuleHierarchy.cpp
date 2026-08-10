#include "ModuleHierarchy.h"

#include "Components.h"
#include "ConfigurationController.h"
#include "OperationShell.h"
#include "Support.h"

#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/RTL/ConfigurationTransport.h"
#include "Hardware/RTL/PhysicalOperation.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <array>
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
  llvm::ArrayRef<MemoryEndpointPortPlan> memoryEndpoints;
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

struct MemoryPortRuntime final {
  const MemoryServicePortPlan *plan = nullptr;
  std::map<std::string, PendingSignal> driven;
  std::map<std::string, mlir::Value> observed;
};

struct MemoryRequestRuntime final {
  mlir::Value kind;
  mlir::Value address;
  mlir::Value data;
  mlir::Value mask;
  mlir::Value activeLanesKind;
  mlir::Value accessForm;
  mlir::Value addressForm;
  mlir::Value elementWidth;
  mlir::Value laneCount;
  mlir::Value addressLaneWidth;
  mlir::Value baseAddress;
  mlir::Value context;
  mlir::Value valid;
};

struct MemorySourceRuntime final {
  MemoryPortRuntime *runtime = nullptr;
  MemoryRequestRuntime request;
  mlir::Value responseReady;
};

struct MemorySinkRuntime final {
  MemoryPortRuntime *runtime = nullptr;
  mlir::Value requestReady;
  mlir::Value responseData;
  mlir::Value responseValid;
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
         component.module,
         component.endpoints,
         {},
         prefix.str() + std::to_string(component.reference.id())});
}

void appendMemoryComponents(llvm::ArrayRef<MemoryModule> components,
                            std::vector<ComponentView> &result) {
  for (const MemoryModule &component : components)
    result.push_back(
        {fabric::FabricTransportEndpointOwnerRef::of(component.reference),
         component.module, component.endpoints, component.memoryEndpoints,
         "memory_" + std::to_string(component.reference.id())});
}

std::string memoryEndpointKey(const fabric::FabricMemoryEndpointRef &endpoint) {
  const std::vector<std::uint8_t> bytes =
      fabric::canonicalFabricBytes(endpoint);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
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

mlir::Value zero(mlir::OpBuilder &builder, mlir::Location location,
                 mlir::Type type);

std::array<const circt::hw::PortInfo *, 17>
memoryPorts(const MemoryServicePortPlan &ports) {
  return {&ports.requestKind,
          &ports.requestAddress,
          &ports.requestData,
          &ports.requestMask,
          &ports.requestActiveLanesKind,
          &ports.requestAccessForm,
          &ports.requestAddressForm,
          &ports.requestElementWidth,
          &ports.requestLaneCount,
          &ports.requestAddressLaneWidth,
          &ports.requestBaseAddress,
          &ports.requestContext,
          &ports.requestValid,
          &ports.requestReady,
          &ports.responseData,
          &ports.responseValid,
          &ports.responseReady};
}

llvm::Expected<mlir::Value>
observedMemoryPort(const MemoryPortRuntime &runtime,
                   const circt::hw::PortInfo &port) {
  const auto found = runtime.observed.find(port.getName().str());
  if (found == runtime.observed.end())
    return invalid("memory service port is not observable");
  return found->second;
}

llvm::Error driveMemoryPort(MemoryPortRuntime &runtime,
                            const circt::hw::PortInfo &port,
                            mlir::Value value) {
  const auto found = runtime.driven.find(port.getName().str());
  if (found == runtime.driven.end())
    return invalid("memory service port is not structurally driven");
  return setPending(found->second, value, "memory service port");
}

llvm::Expected<MemoryRequestRuntime>
readMemoryRequest(const MemoryPortRuntime &runtime) {
  const MemoryServicePortPlan &ports = *runtime.plan;
  auto kind = observedMemoryPort(runtime, ports.requestKind);
  auto address = observedMemoryPort(runtime, ports.requestAddress);
  auto data = observedMemoryPort(runtime, ports.requestData);
  auto mask = observedMemoryPort(runtime, ports.requestMask);
  auto activeLanesKind =
      observedMemoryPort(runtime, ports.requestActiveLanesKind);
  auto accessForm = observedMemoryPort(runtime, ports.requestAccessForm);
  auto addressForm = observedMemoryPort(runtime, ports.requestAddressForm);
  auto elementWidth = observedMemoryPort(runtime, ports.requestElementWidth);
  auto laneCount = observedMemoryPort(runtime, ports.requestLaneCount);
  auto addressLaneWidth =
      observedMemoryPort(runtime, ports.requestAddressLaneWidth);
  auto baseAddress = observedMemoryPort(runtime, ports.requestBaseAddress);
  auto context = observedMemoryPort(runtime, ports.requestContext);
  auto valid = observedMemoryPort(runtime, ports.requestValid);
  if (!kind || !address || !data || !mask || !activeLanesKind || !accessForm ||
      !addressForm || !elementWidth || !laneCount || !addressLaneWidth ||
      !baseAddress || !context || !valid)
    return invalid("memory request projection is incomplete");
  return MemoryRequestRuntime{
      *kind,        *address,     *data,         *mask,      *activeLanesKind,
      *accessForm,  *addressForm, *elementWidth, *laneCount, *addressLaneWidth,
      *baseAddress, *context,     *valid};
}

llvm::Error driveMemoryRequest(MemoryPortRuntime &runtime,
                               const MemoryRequestRuntime &request) {
  const MemoryServicePortPlan &ports = *runtime.plan;
  if (llvm::Error error =
          driveMemoryPort(runtime, ports.requestKind, request.kind))
    return error;
  if (llvm::Error error =
          driveMemoryPort(runtime, ports.requestAddress, request.address))
    return error;
  if (llvm::Error error =
          driveMemoryPort(runtime, ports.requestData, request.data))
    return error;
  if (llvm::Error error =
          driveMemoryPort(runtime, ports.requestMask, request.mask))
    return error;
  if (llvm::Error error = driveMemoryPort(runtime, ports.requestActiveLanesKind,
                                          request.activeLanesKind))
    return error;
  if (llvm::Error error =
          driveMemoryPort(runtime, ports.requestAccessForm, request.accessForm))
    return error;
  if (llvm::Error error = driveMemoryPort(runtime, ports.requestAddressForm,
                                          request.addressForm))
    return error;
  if (llvm::Error error = driveMemoryPort(runtime, ports.requestElementWidth,
                                          request.elementWidth))
    return error;
  if (llvm::Error error =
          driveMemoryPort(runtime, ports.requestLaneCount, request.laneCount))
    return error;
  if (llvm::Error error = driveMemoryPort(
          runtime, ports.requestAddressLaneWidth, request.addressLaneWidth))
    return error;
  if (llvm::Error error = driveMemoryPort(runtime, ports.requestBaseAddress,
                                          request.baseAddress))
    return error;
  if (llvm::Error error =
          driveMemoryPort(runtime, ports.requestContext, request.context))
    return error;
  return driveMemoryPort(runtime, ports.requestValid, request.valid);
}

MemoryRequestRuntime zeroMemoryRequest(mlir::OpBuilder &builder,
                                       mlir::Location location,
                                       const MemoryServicePortPlan &ports) {
  return {zero(builder, location, ports.requestKind.type),
          zero(builder, location, ports.requestAddress.type),
          zero(builder, location, ports.requestData.type),
          zero(builder, location, ports.requestMask.type),
          zero(builder, location, ports.requestActiveLanesKind.type),
          zero(builder, location, ports.requestAccessForm.type),
          zero(builder, location, ports.requestAddressForm.type),
          zero(builder, location, ports.requestElementWidth.type),
          zero(builder, location, ports.requestLaneCount.type),
          zero(builder, location, ports.requestAddressLaneWidth.type),
          zero(builder, location, ports.requestBaseAddress.type),
          zero(builder, location, ports.requestContext.type),
          bitConstant(builder, location, false)};
}

MemoryRequestRuntime muxMemoryRequest(mlir::OpBuilder &builder,
                                      mlir::Location location,
                                      mlir::Value select,
                                      const MemoryRequestRuntime &selected,
                                      const MemoryRequestRuntime &fallback) {
  const auto mux = [&](mlir::Value lhs, mlir::Value rhs) {
    return mlir::Value(
        circt::comb::MuxOp::create(builder, location, select, lhs, rhs, true));
  };
  return {mux(selected.kind, fallback.kind),
          mux(selected.address, fallback.address),
          mux(selected.data, fallback.data),
          mux(selected.mask, fallback.mask),
          mux(selected.activeLanesKind, fallback.activeLanesKind),
          mux(selected.accessForm, fallback.accessForm),
          mux(selected.addressForm, fallback.addressForm),
          mux(selected.elementWidth, fallback.elementWidth),
          mux(selected.laneCount, fallback.laneCount),
          mux(selected.addressLaneWidth, fallback.addressLaneWidth),
          mux(selected.baseAddress, fallback.baseAddress),
          mux(selected.context, fallback.context),
          mux(selected.valid, fallback.valid)};
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

mlir::Value integerConstant(mlir::OpBuilder &builder, mlir::Location location,
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
      integerConstant(builder, location, width, expected), true);
}

unsigned indexWidth(std::uint64_t count) {
  return std::max(1U, llvm::Log2_64_Ceil(std::max<std::uint64_t>(count, 1)));
}

std::vector<mlir::Value>
roundRobinSelection(mlir::OpBuilder &builder, mlir::Location location,
                    llvm::ArrayRef<mlir::Value> requests, mlir::Value cursor) {
  std::vector<mlir::Value> selected(requests.size(),
                                    bitConstant(builder, location, false));
  for (std::size_t start = 0; start != requests.size(); ++start) {
    mlir::Value atStart = equals(builder, location, cursor, start);
    mlir::Value reserved = bitConstant(builder, location, false);
    for (std::size_t offset = 0; offset != requests.size(); ++offset) {
      const std::size_t requester = (start + offset) % requests.size();
      mlir::Value grant = andValues(
          builder, location,
          {atStart, requests[requester],
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
        integerConstant(builder, location, width,
                        (requester + 1) % fired.size()),
        next, true);
  return next;
}

} // namespace

llvm::Expected<ModuleRootCirctSkeleton> buildModuleHierarchySkeleton(
    mlir::MLIRContext &context, fabric::SpatialCoreOccurrenceRef spatialCore,
    const FinalizedConfigurationABI &finalizedAbi,
    const fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<ModuleBoundaryTransportPortProjection> projections) {
  const ConfigurationABI &configurationAbi = finalizedAbi.abi();
  auto transportLayout =
      derivePortableConfigurationTransportLayout(finalizedAbi, spatialCore);
  if (!transportLayout)
    return transportLayout.takeError();
  auto clockReset =
      prepareClockReset(configurationAbi.fabricSystem(), spatialCore);
  if (!clockReset)
    return clockReset.takeError();
  auto operations = selectOperations(configurationAbi, spatialCore);
  if (!operations)
    return operations.takeError();
  auto memoryServiceLayout = derivePortableMemoryServiceLayout(fabric);
  if (!memoryServiceLayout)
    return memoryServiceLayout.takeError();

  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  auto memoryBoundaryProjections =
      deriveModuleBoundaryMemoryPorts(builder, fabric, *memoryServiceLayout);
  if (!memoryBoundaryProjections)
    return memoryBoundaryProjections.takeError();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));

  std::vector<FabricOperationLeafAssociation> associations;
  auto operationShells = buildOperationShellModules(
      builder, location, spatialCore, fabric, configurationAbi,
      *transportLayout, *operations, associations, *clockReset);
  if (!operationShells)
    return operationShells.takeError();
  auto fuModules =
      buildFuModules(builder, location, spatialCore, fabric, configurationAbi,
                     *transportLayout, *operationShells, *clockReset);
  if (!fuModules)
    return fuModules.takeError();
  auto peModules =
      buildPeModules(builder, location, spatialCore, fabric, configurationAbi,
                     *transportLayout, *fuModules, *clockReset);
  if (!peModules)
    return peModules.takeError();
  auto switchModules =
      buildSwitchModules(builder, location, spatialCore, fabric,
                         configurationAbi, *transportLayout, *clockReset);
  if (!switchModules)
    return switchModules.takeError();
  auto fifoModules =
      buildFifoModules(builder, location, spatialCore, fabric, configurationAbi,
                       *transportLayout, *clockReset);
  if (!fifoModules)
    return fifoModules.takeError();
  auto boundaryModules =
      buildBoundaryModules(builder, location, spatialCore, fabric,
                           configurationAbi, *transportLayout);
  if (!boundaryModules)
    return boundaryModules.takeError();
  auto memoryModules = buildMemoryModules(
      builder, location, spatialCore, fabric, configurationAbi,
      *transportLayout, *clockReset, *memoryServiceLayout);
  if (!memoryModules)
    return memoryModules.takeError();
  auto configurationController = buildConfigurationControllerModule(
      builder, location, configurationAbi, *transportLayout, *clockReset);
  if (!configurationController)
    return configurationController.takeError();

  std::vector<ComponentView> components;
  components.reserve(peModules->size() + switchModules->size() +
                     fifoModules->size() + boundaryModules->size() +
                     memoryModules->size());
  appendComponents(llvm::ArrayRef(*peModules), "pe_", components);
  appendComponents(llvm::ArrayRef(*switchModules), "switch_", components);
  appendComponents(llvm::ArrayRef(*fifoModules), "fifo_", components);
  appendComponents(llvm::ArrayRef(*boundaryModules), "boundary_", components);
  appendMemoryComponents(*memoryModules, components);

  llvm::SmallVector<circt::hw::PortInfo, 32> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 32> outputs;
  inputs.push_back(
      circt::hw::PortInfo{{builder.getStringAttr("clock"),
                           circt::seq::ClockType::get(builder.getContext()),
                           circt::hw::ModulePort::Direction::Input}});
  inputs.push_back(
      circt::hw::PortInfo{{builder.getStringAttr("reset"), builder.getI1Type(),
                           circt::hw::ModulePort::Direction::Input}});
  appendAxiLiteConfigurationPorts(builder, inputs, outputs);
  for (const auto &projection : projections)
    appendBoundaryPorts(inputs, outputs, projection);
  for (const auto &projection : *memoryBoundaryProjections)
    appendMemoryServicePorts(inputs, outputs, projection.ports);
  std::set<std::string> portNames;
  for (const circt::hw::PortInfo &port : inputs)
    if (!portNames.insert(port.getName().str()).second)
      return invalid("Module hierarchy input port name is duplicated: " +
                     port.getName());
  for (const circt::hw::PortInfo &port : outputs)
    if (!portNames.insert(port.getName().str()).second)
      return invalid("Module hierarchy output port name is duplicated: " +
                     port.getName());

  std::optional<std::string> materializationError;
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("loom_module"),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        std::map<std::string, mlir::Type> outputTypes;
        for (const circt::hw::PortInfo &port : outputs)
          outputTypes.emplace(port.getName().str(), port.type);
        std::set<std::string> assignedOutputs;
        const auto assignOutput = [&](llvm::StringRef name, mlir::Value value) {
          const auto type = outputTypes.find(name.str());
          if (type == outputTypes.end()) {
            if (!materializationError)
              materializationError =
                  "Module hierarchy selected an unknown output port: " +
                  name.str();
            return;
          }
          if (!assignedOutputs.insert(name.str()).second) {
            if (!materializationError)
              materializationError =
                  "Module hierarchy drove an output port twice: " + name.str();
            return;
          }
          if (!value || value.getType() != type->second) {
            if (!materializationError)
              materializationError =
                  "Module hierarchy drove an output with the wrong type: " +
                  name.str();
            value = zero(bodyBuilder, location, type->second);
          }
          accessor.setOutput(name, value);
        };
        const auto completeOutputs = [&] {
          for (const circt::hw::PortInfo &port : outputs)
            if (assignedOutputs.insert(port.getName().str()).second)
              accessor.setOutput(port.getName(),
                                 zero(bodyBuilder, location, port.type));
        };
        const auto fail = [&](llvm::Error error) {
          if (!materializationError)
            materializationError = llvm::toString(std::move(error));
          else
            llvm::consumeError(std::move(error));
          backedges.abandon();
          completeOutputs();
        };
        mlir::Value reset = accessor.getInput("reset");
        if (clockReset->activeLowReset)
          reset = circt::comb::createOrFoldNot(bodyBuilder, location, reset);
        std::map<std::string, mlir::Value> controllerInputs;
        controllerInputs.emplace("clock", accessor.getInput("clock"));
        controllerInputs.emplace("reset", reset);
        for (llvm::StringRef name : {"cfg_awaddr", "cfg_awvalid", "cfg_wdata",
                                     "cfg_wstrb", "cfg_wvalid", "cfg_bready",
                                     "cfg_araddr", "cfg_arvalid", "cfg_rready"})
          controllerInputs.emplace(name.str(), accessor.getInput(name));
        auto controller = instantiateModule(
            bodyBuilder, location, configurationController->module,
            "configuration_controller", controllerInputs);
        if (!controller) {
          fail(controller.takeError());
          return;
        }
        for (llvm::StringRef name :
             {"cfg_awready", "cfg_wready", "cfg_bresp", "cfg_bvalid",
              "cfg_arready", "cfg_rdata", "cfg_rresp", "cfg_rvalid"})
          assignOutput(name, controller->at(name.str()));
        std::map<std::string, EndpointRuntime> runtime;
        std::map<std::string, MemoryPortRuntime> memoryRuntime;
        std::map<std::string, MemoryPortRuntime> memoryBoundaryRuntime;

        for (const ModuleBoundaryMemoryPortProjection &projection :
             *memoryBoundaryProjections) {
          MemoryPortRuntime boundaryRuntime;
          boundaryRuntime.plan = &projection.ports;
          for (const circt::hw::PortInfo *port :
               memoryPorts(projection.ports)) {
            if (port->isInput()) {
              boundaryRuntime.observed.emplace(
                  port->getName().str(), accessor.getInput(port->getName()));
              continue;
            }
            auto [position, inserted] = boundaryRuntime.driven.emplace(
                port->getName().str(),
                PendingSignal{backedges.get(port->type), false});
            if (!inserted) {
              fail(invalid("memory boundary port name is duplicated"));
              return;
            }
          }
          if (!memoryBoundaryRuntime
                   .emplace(boundaryKey(projection.boundary),
                            std::move(boundaryRuntime))
                   .second) {
            fail(invalid("memory boundary identity is duplicated"));
            return;
          }
        }

        for (const ComponentView &component : components) {
          std::map<std::string, mlir::Value> instanceInputs;
          instanceInputs.emplace("clock", accessor.getInput("clock"));
          instanceInputs.emplace("reset", reset);
          for (auto [ordinal, unit] : llvm::enumerate(transportLayout->units)) {
            (void)unit;
            instanceInputs.emplace(
                configurationPortName(ordinal),
                controller->at(configurationPortName(ordinal)));
          }

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

          std::vector<MemoryPortRuntime *> pendingMemoryRuntime;
          for (const MemoryEndpointPortPlan &endpoint :
               component.memoryEndpoints) {
            MemoryPortRuntime endpointRuntime;
            endpointRuntime.plan = &endpoint.ports;
            for (const circt::hw::PortInfo *port :
                 memoryPorts(endpoint.ports)) {
              if (port->isOutput())
                continue;
              auto [position, inserted] = endpointRuntime.driven.emplace(
                  port->getName().str(),
                  PendingSignal{backedges.get(port->type), false});
              if (!inserted ||
                  !instanceInputs
                       .emplace(port->getName().str(), position->second.edge)
                       .second) {
                fail(invalid("memory endpoint port name is duplicated"));
                return;
              }
            }
            auto [position, inserted] =
                memoryRuntime.emplace(memoryEndpointKey(endpoint.endpoint),
                                      std::move(endpointRuntime));
            if (!inserted) {
              fail(invalid("memory endpoint identity is duplicated"));
              return;
            }
            pendingMemoryRuntime.push_back(&position->second);
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
          for (MemoryPortRuntime *endpoint : pendingMemoryRuntime)
            for (const circt::hw::PortInfo *port : memoryPorts(*endpoint->plan))
              if (port->isOutput())
                endpoint->observed.emplace(port->getName().str(),
                                           instance->at(port->getName().str()));
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
            assignOutput(projection->ready.getName(),
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
          assignOutput(projection->valid.getName(), adapted->valid);
          if (projection->data)
            assignOutput(projection->data->getName(), *adapted->payload);
          if (projection->tag)
            assignOutput(projection->tag->getName(), *adapted->tag);
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
          assignOutput(output->valid.getName(), adapted->valid);
          if (output->data)
            assignOutput(output->data->getName(), *adapted->payload);
          if (output->tag)
            assignOutput(output->tag->getName(), *adapted->tag);
          assignOutput(input->ready.getName(),
                       accessor.getInput(output->ready.getName()));
        }

        std::map<std::string, MemorySourceRuntime> memorySources;
        std::map<std::string, MemorySinkRuntime> memorySinks;
        const auto addSource =
            [&](std::string key,
                MemoryPortRuntime &portRuntime) -> llvm::Error {
          auto request = readMemoryRequest(portRuntime);
          if (!request)
            return request.takeError();
          auto responseReady =
              observedMemoryPort(portRuntime, portRuntime.plan->responseReady);
          if (!responseReady)
            return responseReady.takeError();
          if (!memorySources
                   .emplace(std::move(key),
                            MemorySourceRuntime{&portRuntime,
                                                std::move(*request),
                                                *responseReady})
                   .second)
            return invalid("memory service source identity is duplicated");
          return llvm::Error::success();
        };
        const auto addSink =
            [&](std::string key,
                MemoryPortRuntime &portRuntime) -> llvm::Error {
          auto requestReady =
              observedMemoryPort(portRuntime, portRuntime.plan->requestReady);
          auto responseData =
              observedMemoryPort(portRuntime, portRuntime.plan->responseData);
          auto responseValid =
              observedMemoryPort(portRuntime, portRuntime.plan->responseValid);
          if (!requestReady || !responseData || !responseValid)
            return invalid("memory service sink projection is incomplete");
          if (!memorySinks
                   .emplace(std::move(key),
                            MemorySinkRuntime{&portRuntime, *requestReady,
                                              *responseData, *responseValid})
                   .second)
            return invalid("memory service sink identity is duplicated");
          return llvm::Error::success();
        };
        for (auto &[key, portRuntime] : memoryRuntime) {
          llvm::Error error = portRuntime.plan->role ==
                                      fabric::FabricMemoryEndpointRole::Manager
                                  ? addSource("endpoint:" + key, portRuntime)
                                  : addSink("endpoint:" + key, portRuntime);
          if (error) {
            fail(std::move(error));
            return;
          }
        }
        for (const ModuleBoundaryMemoryPortProjection &projection :
             *memoryBoundaryProjections) {
          auto found =
              memoryBoundaryRuntime.find(boundaryKey(projection.boundary));
          if (found == memoryBoundaryRuntime.end()) {
            fail(invalid("memory boundary runtime is absent"));
            return;
          }
          llvm::Error error =
              projection.boundary.direction ==
                      fabric::FabricPortDirection::Output
                  ? addSource("boundary:" + boundaryKey(projection.boundary),
                              found->second)
                  : addSink("boundary:" + boundaryKey(projection.boundary),
                            found->second);
          if (error) {
            fail(std::move(error));
            return;
          }
        }

        std::map<std::string, std::vector<std::string>> sinkSources;
        for (const auto &[key, sink] : memorySinks) {
          (void)sink;
          sinkSources.emplace(key, std::vector<std::string>{});
        }
        std::map<std::string, std::string> sourceSink;
        const auto connectMemory = [&](const std::string &source,
                                       const std::string &sink) -> llvm::Error {
          if (!memorySources.count(source) || !memorySinks.count(sink))
            return invalid("memory service edge has an unknown endpoint");
          if (!sourceSink.emplace(source, sink).second)
            return invalid(
                "one memory manager is connected to multiple providers");
          sinkSources[sink].push_back(source);
          return llvm::Error::success();
        };
        for (const fabric::FabricMemoryServiceConnectionPayload &connection :
             fabric.memoryServiceConnections())
          if (llvm::Error error = connectMemory(
                  "endpoint:" + memoryEndpointKey(connection.source),
                  "endpoint:" + memoryEndpointKey(connection.destination))) {
            fail(std::move(error));
            return;
          }
        for (const fabric::FabricModuleBoundaryMemoryAttachmentView
                 &attachment : fabric.moduleBoundaryMemoryAttachments()) {
          const std::string endpoint =
              "endpoint:" + memoryEndpointKey(attachment.endpoint);
          const std::string boundary =
              "boundary:" + boundaryKey(attachment.boundary);
          llvm::Error error = attachment.boundary.direction ==
                                      fabric::FabricPortDirection::Input
                                  ? connectMemory(endpoint, boundary)
                                  : connectMemory(boundary, endpoint);
          if (error) {
            fail(std::move(error));
            return;
          }
        }

        std::set<std::string> connectedMemorySources;
        for (auto &[sinkKey, sourceKeys] : sinkSources) {
          MemorySinkRuntime &sink = memorySinks.at(sinkKey);
          if (sourceKeys.empty()) {
            if (llvm::Error error = driveMemoryRequest(
                    *sink.runtime, zeroMemoryRequest(bodyBuilder, location,
                                                     *sink.runtime->plan))) {
              fail(std::move(error));
              return;
            }
            if (llvm::Error error = driveMemoryPort(
                    *sink.runtime, sink.runtime->plan->responseReady,
                    bitConstant(bodyBuilder, location, false))) {
              fail(std::move(error));
              return;
            }
            continue;
          }

          const unsigned ownerWidth = indexWidth(sourceKeys.size());
          circt::Backedge ownedNext = backedges.get(bodyBuilder.getI1Type());
          circt::Backedge acceptedNext = backedges.get(bodyBuilder.getI1Type());
          circt::Backedge ownerNext =
              backedges.get(bodyBuilder.getIntegerType(ownerWidth));
          circt::Backedge cursorNext =
              backedges.get(bodyBuilder.getIntegerType(ownerWidth));
          mlir::Value owned = createRegister(
              bodyBuilder, location, ownedNext, accessor.getInput("clock"),
              reset, llvm::APInt(1, 0), "memory_network_owned",
              clockReset->asynchronousReset);
          mlir::Value accepted = createRegister(
              bodyBuilder, location, acceptedNext, accessor.getInput("clock"),
              reset, llvm::APInt(1, 0), "memory_network_accepted",
              clockReset->asynchronousReset);
          mlir::Value owner = createRegister(
              bodyBuilder, location, ownerNext, accessor.getInput("clock"),
              reset, llvm::APInt(ownerWidth, 0), "memory_network_owner",
              clockReset->asynchronousReset);
          mlir::Value cursor = createRegister(
              bodyBuilder, location, cursorNext, accessor.getInput("clock"),
              reset, llvm::APInt(ownerWidth, 0), "memory_network_cursor",
              clockReset->asynchronousReset);
          std::vector<mlir::Value> candidates;
          for (const std::string &sourceKey : sourceKeys)
            candidates.push_back(memorySources.at(sourceKey).request.valid);
          std::vector<mlir::Value> selectedNew =
              roundRobinSelection(bodyBuilder, location, candidates, cursor);
          std::vector<mlir::Value> selected(sourceKeys.size());
          mlir::Value anySelected = bitConstant(bodyBuilder, location, false);
          MemoryRequestRuntime outgoing =
              zeroMemoryRequest(bodyBuilder, location, *sink.runtime->plan);
          for (std::size_t ordinal = 0; ordinal != sourceKeys.size();
               ++ordinal) {
            selected[ordinal] = circt::comb::MuxOp::create(
                bodyBuilder, location, owned,
                equals(bodyBuilder, location, owner, ordinal),
                selectedNew[ordinal], true);
            anySelected = circt::comb::OrOp::create(
                bodyBuilder, location, anySelected, selected[ordinal]);
            outgoing = muxMemoryRequest(
                bodyBuilder, location, selected[ordinal],
                memorySources.at(sourceKeys[ordinal]).request, outgoing);
          }
          outgoing.valid = andValues(
              bodyBuilder, location,
              {outgoing.valid,
               circt::comb::createOrFoldNot(bodyBuilder, location, accepted)});
          if (llvm::Error error = driveMemoryRequest(*sink.runtime, outgoing)) {
            fail(std::move(error));
            return;
          }
          mlir::Value requestFire = andValues(
              bodyBuilder, location, {outgoing.valid, sink.requestReady});
          std::vector<mlir::Value> fired(sourceKeys.size());
          for (std::size_t ordinal = 0; ordinal != sourceKeys.size();
               ++ordinal) {
            MemorySourceRuntime &source = memorySources.at(sourceKeys[ordinal]);
            connectedMemorySources.insert(sourceKeys[ordinal]);
            fired[ordinal] = andValues(bodyBuilder, location,
                                       {selected[ordinal], requestFire});
            if (llvm::Error error = driveMemoryPort(
                    *source.runtime, source.runtime->plan->requestReady,
                    fired[ordinal])) {
              fail(std::move(error));
              return;
            }
            mlir::Value returns = andValues(
                bodyBuilder, location,
                {owned, accepted, selected[ordinal], sink.responseValid});
            if (llvm::Error error = driveMemoryPort(
                    *source.runtime, source.runtime->plan->responseData,
                    sink.responseData)) {
              fail(std::move(error));
              return;
            }
            if (llvm::Error error = driveMemoryPort(
                    *source.runtime, source.runtime->plan->responseValid,
                    returns)) {
              fail(std::move(error));
              return;
            }
          }
          mlir::Value selectedResponseReady =
              bitConstant(bodyBuilder, location, false);
          for (std::size_t ordinal = 0; ordinal != sourceKeys.size(); ++ordinal)
            selectedResponseReady = circt::comb::MuxOp::create(
                bodyBuilder, location, selected[ordinal],
                memorySources.at(sourceKeys[ordinal]).responseReady,
                selectedResponseReady, true);
          selectedResponseReady = andValues(
              bodyBuilder, location, {owned, accepted, selectedResponseReady});
          if (llvm::Error error = driveMemoryPort(
                  *sink.runtime, sink.runtime->plan->responseReady,
                  selectedResponseReady)) {
            fail(std::move(error));
            return;
          }
          mlir::Value responseFire =
              andValues(bodyBuilder, location,
                        {sink.responseValid, selectedResponseReady});
          mlir::Value acquire = andValues(
              bodyBuilder, location,
              {circt::comb::createOrFoldNot(bodyBuilder, location, owned),
               anySelected});
          mlir::Value selectedOwner = owner;
          for (std::size_t ordinal = 0; ordinal != sourceKeys.size(); ++ordinal)
            selectedOwner = circt::comb::MuxOp::create(
                bodyBuilder, location, selectedNew[ordinal],
                integerConstant(bodyBuilder, location, ownerWidth, ordinal),
                selectedOwner, true);
          ownerNext.setValue(circt::comb::MuxOp::create(
              bodyBuilder, location, acquire, selectedOwner, owner, true));
          ownedNext.setValue(
              andValues(bodyBuilder, location,
                        {orValues(bodyBuilder, location, {owned, anySelected}),
                         circt::comb::createOrFoldNot(bodyBuilder, location,
                                                      responseFire)}));
          acceptedNext.setValue(andValues(
              bodyBuilder, location,
              {orValues(bodyBuilder, location, {accepted, requestFire}),
               circt::comb::createOrFoldNot(bodyBuilder, location,
                                            responseFire)}));
          cursorNext.setValue(nextCursor(bodyBuilder, location, cursor, fired));
        }

        for (auto &[sourceKey, source] : memorySources) {
          if (connectedMemorySources.count(sourceKey))
            continue;
          if (llvm::Error error = driveMemoryPort(
                  *source.runtime, source.runtime->plan->requestReady,
                  bitConstant(bodyBuilder, location, false))) {
            fail(std::move(error));
            return;
          }
          if (llvm::Error error = driveMemoryPort(
                  *source.runtime, source.runtime->plan->responseData,
                  zero(bodyBuilder, location,
                       source.runtime->plan->responseData.type))) {
            fail(std::move(error));
            return;
          }
          if (llvm::Error error = driveMemoryPort(
                  *source.runtime, source.runtime->plan->responseValid,
                  bitConstant(bodyBuilder, location, false))) {
            fail(std::move(error));
            return;
          }
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

        const auto closeUnusedMemoryPorts = [&](auto &runtimes) -> llvm::Error {
          for (auto &[key, memory] : runtimes) {
            (void)key;
            for (auto &[name, pending] : memory.driven) {
              (void)name;
              if (pending.resolved)
                continue;
              if (llvm::Error error =
                      setPending(pending,
                                 zero(bodyBuilder, location,
                                      mlir::Value(pending.edge).getType()),
                                 "unused memory service port"))
                return error;
            }
          }
          return llvm::Error::success();
        };
        if (llvm::Error error = closeUnusedMemoryPorts(memoryRuntime)) {
          fail(std::move(error));
          return;
        }
        if (llvm::Error error = closeUnusedMemoryPorts(memoryBoundaryRuntime)) {
          fail(std::move(error));
          return;
        }
        for (const auto &[key, memory] : memoryBoundaryRuntime) {
          (void)key;
          for (const auto &[name, pending] : memory.driven)
            assignOutput(name, pending.edge);
        }

        for (const auto &projection : projections) {
          if (connectedBoundaries.count(boundaryKey(projection.boundary)))
            continue;
          if (projection.boundary.direction ==
              fabric::FabricPortDirection::Input) {
            assignOutput(projection.ready.getName(), falseValue);
            continue;
          }
          if (projection.data)
            assignOutput(projection.data->getName(),
                         zero(bodyBuilder, location, projection.data->type));
          if (projection.tag)
            assignOutput(projection.tag->getName(),
                         zero(bodyBuilder, location, projection.tag->type));
          assignOutput(projection.valid.getName(), falseValue);
        }
        if (materializationError || assignedOutputs.size() != outputs.size()) {
          std::string missing;
          for (const circt::hw::PortInfo &port : outputs)
            if (!assignedOutputs.count(port.getName().str())) {
              if (!missing.empty())
                missing += ", ";
              missing += port.getName().str();
            }
          if (!materializationError)
            materializationError =
                "Module hierarchy did not drive output ports: " + missing;
          backedges.abandon();
          completeOutputs();
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

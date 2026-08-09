#include "OperationShell.h"

#include "Support.h"

#include "Hardware/RTL/OperationLeaf.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>

namespace loom::hardware::rtl::hierarchy {
namespace {

llvm::Expected<std::vector<OperationEndpointPlan>> deriveOperationEndpoints(
    mlir::OpBuilder &builder,
    const fabric::ResolvedFabricOpCapabilityView &capability) {
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> ports;
  ports.reserve(capability.physicalPorts.size());
  for (const auto &port : capability.physicalPorts)
    ports.push_back(&port);
  llvm::sort(ports, [](const auto *lhs, const auto *rhs) {
    return std::tie(lhs->reference.direction, lhs->reference.ordinal) <
           std::tie(rhs->reference.direction, rhs->reference.ordinal);
  });

  std::vector<OperationEndpointPlan> result;
  result.reserve(ports.size());
  std::uint64_t expectedInput = 0;
  std::uint64_t expectedOutput = 0;
  for (const auto *port : ports) {
    std::uint64_t &expected =
        port->reference.direction == fabric::FabricPortDirection::Input
            ? expectedInput
            : expectedOutput;
    if (port->reference.ordinal != expected++)
      return invalid("operation physical port ordinals are not dense");
    if (port->payloadWidthBits > mlir::IntegerType::kMaxWidth)
      return unsupported("operation physical port exceeds CIRCT capacity");
    const bool input =
        port->reference.direction == fabric::FabricPortDirection::Input;
    const std::string prefix = (input ? "input_" : "output_") +
                               std::to_string(port->reference.ordinal);
    const auto forward = input ? circt::hw::ModulePort::Direction::Input
                               : circt::hw::ModulePort::Direction::Output;
    const auto backward = input ? circt::hw::ModulePort::Direction::Output
                                : circt::hw::ModulePort::Direction::Input;
    const auto makePort = [&](llvm::StringRef suffix, mlir::Type type,
                              circt::hw::ModulePort::Direction direction) {
      return circt::hw::PortInfo{
          {builder.getStringAttr(prefix + suffix.str()), type, direction}};
    };
    std::optional<circt::hw::PortInfo> data;
    if (port->payloadWidthBits != 0)
      data = makePort("_data", builder.getIntegerType(port->payloadWidthBits),
                      forward);
    result.push_back({port->reference.direction, port->reference.ordinal,
                      port->payloadWidthBits, std::move(data),
                      makePort("_valid", builder.getI1Type(), forward),
                      makePort("_ready", builder.getI1Type(), backward)});
  }
  return result;
}

void appendOperationPorts(llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
                          llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs,
                          const OperationEndpointPlan &endpoint) {
  const auto append = [&](const circt::hw::PortInfo &port) {
    (port.isOutput() ? outputs : inputs).push_back(port);
  };
  if (endpoint.data)
    append(*endpoint.data);
  append(endpoint.valid);
  append(endpoint.ready);
}

llvm::Expected<OperationShellModule>
buildOperationShell(mlir::OpBuilder &builder, mlir::Location location,
                    fabric::SpatialCoreOccurrenceRef spatialCore,
                    const ConfigurationABI &configurationAbi,
                    const ResolvedFabricPhysicalOperation &operation,
                    std::size_t index,
                    std::vector<FabricOperationLeafAssociation> &associations,
                    const ClockResetPlan &clockReset) {
  auto interface = deriveFabricOperationLeafInterface(*operation.capability);
  if (!interface)
    return interface.takeError();
  if (interface->protocol != FabricOperationLeafProtocol::Combinational)
    return unsupported(
        "hierarchical control/stream operation shell is not implemented");
  auto endpoints = deriveOperationEndpoints(builder, *operation.capability);
  if (!endpoints)
    return endpoints.takeError();
  auto leafPorts =
      deriveFabricOperationLeafPorts(builder, operation.physicalOccurrence,
                                     *operation.capability, configurationAbi);
  if (!leafPorts)
    return leafPorts.takeError();

  auto leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(builder.getContext(),
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("loom_fabric_operation_" + std::to_string(index)),
      *leafPorts);
  associations.push_back({leaf, operation.physicalOccurrence});

  std::vector<std::pair<fabric::FabricOrdinal, FieldDecoderPlan>>
      configurationFields;
  for (const fabric::FabricSemanticConfigFieldRef &field :
       operation.capability->configurationFieldSchema) {
    auto decoder = prepareFieldDecoder(spatialCore, field, configurationAbi);
    if (!decoder)
      return decoder.takeError();
    configurationFields.emplace_back(field.ordinal, std::move(*decoder));
  }
  llvm::sort(configurationFields, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  appendClockResetAndConfigurationPorts(builder, configurationAbi, inputs);
  for (const OperationEndpointPlan &endpoint : *endpoints)
    appendOperationPorts(inputs, outputs, endpoint);

  std::optional<std::string> materializationError;
  auto shell = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_operation_shell_" + std::to_string(index)),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        std::vector<const OperationEndpointPlan *> inputEndpoints;
        std::vector<const OperationEndpointPlan *> outputEndpoints;
        for (const OperationEndpointPlan &endpoint : *endpoints)
          (endpoint.direction == fabric::FabricPortDirection::Input
               ? inputEndpoints
               : outputEndpoints)
              .push_back(&endpoint);
        if (inputEndpoints.empty() || outputEndpoints.empty()) {
          materializationError =
              "operation shell requires nonempty input and output tuples";
          backedges.abandon();
          return;
        }

        circt::Backedge tupleValidNext = backedges.get(bodyBuilder.getI1Type());
        mlir::Value tupleValid = createRegister(
            bodyBuilder, location, tupleValidNext, accessor.getInput("clock"),
            accessor.getInput("reset"), llvm::APInt(1, 0), "result_valid_reg",
            clockReset.asynchronousReset);
        std::vector<circt::Backedge> dataNext(outputEndpoints.size());
        std::vector<mlir::Value> dataRegisters(outputEndpoints.size());
        for (auto [ordinal, endpoint] : llvm::enumerate(outputEndpoints)) {
          if (endpoint->payloadWidthBits == 0)
            continue;
          dataNext[ordinal] = backedges.get(
              bodyBuilder.getIntegerType(endpoint->payloadWidthBits));
          dataRegisters[ordinal] = createRegister(
              bodyBuilder, location, dataNext[ordinal],
              accessor.getInput("clock"), accessor.getInput("reset"),
              llvm::APInt(endpoint->payloadWidthBits, 0),
              outputEndpoints.size() == 1
                  ? "result_data_reg"
                  : "result_data_" + std::to_string(ordinal) + "_reg",
              clockReset.asynchronousReset);
        }

        llvm::SmallVector<mlir::Value, 4> downstreamReady;
        llvm::SmallVector<mlir::Value, 4> heldValids(outputEndpoints.size(),
                                                     tupleValid);
        for (const OperationEndpointPlan *endpoint : outputEndpoints)
          downstreamReady.push_back(
              accessor.getInput(endpoint->ready.getName()));
        auto held = deriveAtomicResultTupleSignals(bodyBuilder, location,
                                                   heldValids, downstreamReady);
        if (!held) {
          materializationError = llvm::toString(held.takeError());
          backedges.abandon();
          return;
        }

        llvm::SmallVector<mlir::Value, 4> inputValids;
        for (const OperationEndpointPlan *endpoint : inputEndpoints)
          inputValids.push_back(accessor.getInput(endpoint->valid.getName()));
        auto readiness = deriveAtomicInputReadiness(
            bodyBuilder, location, inputValids, held->available);
        if (!readiness) {
          materializationError = llvm::toString(readiness.takeError());
          backedges.abandon();
          return;
        }
        mlir::Value accept = andValues(
            bodyBuilder, location,
            {held->available, andValues(bodyBuilder, location, inputValids)});

        std::map<std::string, mlir::Value> leafInput;
        for (const OperationEndpointPlan *endpoint : inputEndpoints)
          if (endpoint->data)
            leafInput.emplace("data_input_" + std::to_string(endpoint->ordinal),
                              accessor.getInput(endpoint->data->getName()));
        for (const auto &[ordinal, decoder] : configurationFields)
          leafInput.emplace(
              "config_" + std::to_string(ordinal),
              decodeFieldSignal(bodyBuilder, location, accessor, decoder));
        llvm::SmallVector<mlir::Value> leafOperands;
        for (const circt::hw::PortInfo &port : *leafPorts) {
          if (port.isOutput())
            continue;
          const auto found = leafInput.find(port.getName().str());
          if (found == leafInput.end()) {
            materializationError =
                "operation leaf input has no hierarchical signal";
            backedges.abandon();
            return;
          }
          leafOperands.push_back(found->second);
        }
        auto instance = circt::hw::InstanceOp::create(
            bodyBuilder, location, leaf.getOperation(), "operation",
            leafOperands);
        std::map<std::string, mlir::Value> leafOutput;
        unsigned resultOrdinal = 0;
        for (const circt::hw::PortInfo &port : *leafPorts)
          if (port.isOutput())
            leafOutput.emplace(port.getName().str(),
                               instance.getResult(resultOrdinal++));

        mlir::Value retain = circt::comb::AndOp::create(
            bodyBuilder, location, tupleValid,
            circt::comb::createOrFoldNot(bodyBuilder, location,
                                         held->released));
        tupleValidNext.setValue(
            circt::comb::OrOp::create(bodyBuilder, location, accept, retain));
        for (auto [ordinal, endpoint] : llvm::enumerate(outputEndpoints)) {
          accessor.setOutput(endpoint->valid.getName(),
                             held->publishedValids[ordinal]);
          if (!endpoint->data)
            continue;
          const auto found = leafOutput.find("data_output_" +
                                             std::to_string(endpoint->ordinal));
          if (found == leafOutput.end()) {
            materializationError =
                "operation leaf output has no hierarchical signal";
            backedges.abandon();
            return;
          }
          dataNext[ordinal].setValue(circt::comb::MuxOp::create(
              bodyBuilder, location, accept, found->second,
              dataRegisters[ordinal], true));
          accessor.setOutput(endpoint->data->getName(), dataRegisters[ordinal]);
        }
        for (auto [ordinal, endpoint] : llvm::enumerate(inputEndpoints))
          accessor.setOutput(endpoint->ready.getName(), (*readiness)[ordinal]);
      });
  if (materializationError)
    return invalid(*materializationError);
  return OperationShellModule{operation, shell, std::move(*endpoints)};
}

} // namespace

llvm::Expected<std::vector<OperationShellModule>> buildOperationShellModules(
    mlir::OpBuilder &builder, mlir::Location location,
    fabric::SpatialCoreOccurrenceRef spatialCore,
    const ConfigurationABI &configurationAbi,
    llvm::ArrayRef<ResolvedFabricPhysicalOperation> operations,
    std::vector<FabricOperationLeafAssociation> &associations,
    const ClockResetPlan &clockReset) {
  std::vector<OperationShellModule> result;
  result.reserve(operations.size());
  for (auto [index, operation] : llvm::enumerate(operations)) {
    auto shell =
        buildOperationShell(builder, location, spatialCore, configurationAbi,
                            operation, index, associations, clockReset);
    if (!shell)
      return shell.takeError();
    result.push_back(std::move(*shell));
  }
  return result;
}

} // namespace loom::hardware::rtl::hierarchy

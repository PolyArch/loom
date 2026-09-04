#include "OperationShell.h"

#include "Support.h"

#include "Hardware/RTL/OperationLeaf.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <map>
#include <string>
#include <utility>

namespace loom::hardware::rtl::hierarchy {
namespace {

llvm::Expected<std::vector<OperationEndpointPlan>> deriveOperationEndpoints(
    mlir::OpBuilder &builder,
    const fabric::ResolvedFabricOpCapabilityView &capability,
    std::optional<unsigned> contextWidth, bool directPublication) {
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
    std::optional<circt::hw::PortInfo> context;
    if (contextWidth && !input)
      context =
          makePort("_context", builder.getIntegerType(*contextWidth), forward);
    result.push_back(
        {port->reference.direction, port->reference.ordinal,
         port->payloadWidthBits, std::move(data), std::move(context),
         !input || directPublication
             ? std::optional<circt::hw::PortInfo>(
                   makePort("_offer", builder.getI1Type(), forward))
             : std::nullopt,
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
  if (endpoint.context)
    append(*endpoint.context);
  if (endpoint.offer)
    append(*endpoint.offer);
  append(endpoint.valid);
  append(endpoint.ready);
}

mlir::Value contextEquals(mlir::OpBuilder &builder, mlir::Location location,
                          mlir::Value context, std::uint64_t ordinal) {
  const unsigned width =
      mlir::cast<mlir::IntegerType>(context.getType()).getWidth();
  mlir::Value expected = circt::hw::ConstantOp::create(
      builder, location, llvm::APInt(width, ordinal));
  return circt::comb::ICmpOp::create(builder, location,
                                     circt::comb::ICmpPredicate::eq, context,
                                     expected, true);
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

llvm::Expected<OperationShellModule>
buildOperationShell(mlir::OpBuilder &builder, mlir::Location location,
                    fabric::SpatialCoreOccurrenceRef spatialCore,
                    const fabric::FabricArtifactView &fabric,
                    const ConfigurationABI &configurationAbi,
                    const ConfigurationTransportLayout &transportLayout,
                    const ResolvedFabricPhysicalOperation &operation,
                    std::size_t index,
                    std::vector<FabricOperationLeafAssociation> &associations,
                    const ClockResetPlan &clockReset) {
  if (llvm::Error error =
          validateFabricOperationStructuralContract(*operation.capability))
    return std::move(error);
  auto interface = deriveFabricOperationLeafInterface(*operation.capability);
  if (!interface)
    return interface.takeError();
  auto stateLayout =
      deriveFabricOperationLeafStateLayout(*operation.capability);
  if (!stateLayout)
    return stateLayout.takeError();
  const auto parentPe = fabric.parentPeOf(operation.localOccurrence.fu);
  if (!parentPe)
    return invalid("operation shell has no parent PE");
  const bool temporal =
      fabric.peSchedule(*parentPe) == ::fabric::Schedule::Temporal;
  const std::uint64_t contextCount =
      temporal ? fabric.peResidentContextCount(*parentPe) : 1;
  if (contextCount == 0 || contextCount > UINT32_MAX)
    return invalid("operation shell context domain is outside u32");
  const std::optional<unsigned> contextWidth =
      temporal ? std::optional<unsigned>(
                     std::max(1U, llvm::Log2_64_Ceil(contextCount)))
               : std::nullopt;
  auto endpoints =
      deriveOperationEndpoints(builder, *operation.capability, contextWidth,
                               interface->hasDirectTokenPublication());
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

  struct ConfigurationFieldPlan final {
    fabric::FabricOrdinal ordinal = 0;
    std::vector<FieldDecoderPlan> contexts;
  };
  std::vector<ConfigurationFieldPlan> configurationFields;
  for (const fabric::FabricSemanticConfigFieldRef &templateField :
       operation.capability->configurationFieldSchema) {
    const fabric::FabricSemanticConfigFieldRef occurrenceField{
        fabric::FabricConfigurationOwnerRef(
            fabric::FabricInventoryOwnerRef::of(operation.localOccurrence)),
        templateField.ordinal};
    auto residencies = fabric.configurationResidencies(occurrenceField);
    if (!residencies)
      return residencies.takeError();
    ConfigurationFieldPlan plan;
    plan.ordinal = templateField.ordinal;
    plan.contexts.reserve(residencies->size());
    for (const auto &residency : *residencies) {
      auto decoder =
          prepareFieldDecoder(spatialCore, occurrenceField, residency,
                              configurationAbi, transportLayout);
      if (!decoder)
        return decoder.takeError();
      plan.contexts.push_back(std::move(*decoder));
    }
    if (plan.contexts.size() != 1 && plan.contexts.size() != contextCount)
      return invalid("operation configuration residency is incomplete");
    configurationFields.push_back(std::move(plan));
  }
  llvm::sort(configurationFields, [](const auto &lhs, const auto &rhs) {
    return lhs.ordinal < rhs.ordinal;
  });

  llvm::SmallVector<circt::hw::PortInfo, 16> inputs;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputs;
  std::vector<FieldDecoderPlan> shellDecoders;
  for (const ConfigurationFieldPlan &field : configurationFields)
    shellDecoders.insert(shellDecoders.end(), field.contexts.begin(),
                         field.contexts.end());
  auto configuration = deriveConfigurationBundlePlan(shellDecoders);
  if (!configuration)
    return configuration.takeError();
  appendClockResetAndConfigurationPorts(builder, *configuration, inputs);
  if (contextWidth) {
    inputs.push_back({{builder.getStringAttr(dispatchContextPortName),
                       builder.getIntegerType(*contextWidth),
                       circt::hw::ModulePort::Direction::Input}});
    inputs.push_back(
        {{builder.getStringAttr(dispatchEnablePortName), builder.getI1Type(),
          circt::hw::ModulePort::Direction::Input}});
  }
  for (const OperationEndpointPlan &endpoint : *endpoints)
    appendOperationPorts(inputs, outputs, endpoint);

  std::optional<std::string> materializationError;
  auto shell = circt::hw::HWModuleOp::create(
      builder, location,
      builder.getStringAttr("loom_operation_shell_" + std::to_string(index)),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        std::optional<ConfigurationBundleSignals> configurationSignals;
        if (!configuration->empty())
          configurationSignals.emplace(
              configurationBundleSignals(accessor, *configuration));
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

        mlir::Value enabled = circt::comb::createOrFoldNot(
            bodyBuilder, location, accessor.getInput("reset"));

        // The enclosing Temporal PE grants one context to this operation's FU
        // per clock cycle. That dispatch context selects the state bank and
        // configuration slot; every operand head the FU presents belongs to
        // it, so a transition fires exactly when the heads its schema case
        // consumes are valid, whatever the other inputs hold.
        mlir::Value dispatchContext;
        mlir::Value contextInRange = bitConstant(bodyBuilder, location, true);
        if (temporal) {
          dispatchContext = accessor.getInput(dispatchContextPortName);
          if (!llvm::isPowerOf2_64(contextCount)) {
            mlir::Value bound = circt::hw::ConstantOp::create(
                bodyBuilder, location,
                llvm::APInt(*contextWidth, contextCount));
            contextInRange = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::ult,
                dispatchContext, bound, true);
          }
        }

        const bool hasResultStorage = !interface->hasDirectTokenPublication();
        circt::Backedge resultContextNext;
        mlir::Value resultContext;
        if (temporal && hasResultStorage) {
          resultContextNext =
              backedges.get(bodyBuilder.getIntegerType(*contextWidth));
          resultContext = createRegister(
              bodyBuilder, location, resultContextNext,
              accessor.getInput("clock"), accessor.getInput("reset"),
              llvm::APInt(*contextWidth, 0), "result_context_reg",
              clockReset.asynchronousReset);
        }

        circt::Backedge continuationNext;
        mlir::Value continuation = bitConstant(bodyBuilder, location, false);
        if (interface->hasOrderedProductionGroups()) {
          continuationNext = backedges.get(bodyBuilder.getI1Type());
          continuation = createRegister(
              bodyBuilder, location, continuationNext,
              accessor.getInput("clock"), accessor.getInput("reset"),
              llvm::APInt(1, 0), "operation_continuation_reg",
              clockReset.asynchronousReset);
        }

        mlir::Value executionContext = dispatchContext;
        if (temporal && interface->hasOrderedProductionGroups())
          executionContext =
              circt::comb::MuxOp::create(bodyBuilder, location, continuation,
                                         resultContext, dispatchContext, true);
        mlir::Value executionContextValid =
            temporal ? andValues(bodyBuilder, location,
                                 {contextInRange,
                                  accessor.getInput(dispatchEnablePortName)})
                     : contextInRange;
        if (interface->hasOrderedProductionGroups())
          executionContextValid = orValues(
              bodyBuilder, location, {continuation, executionContextValid});

        std::vector<circt::Backedge> stateNext;
        std::vector<mlir::Value> stateRegisters;
        mlir::Value selectedState;
        if (*stateLayout) {
          const std::uint64_t bankCount = temporal ? contextCount : 1;
          stateNext.resize(bankCount);
          stateRegisters.resize(bankCount);
          for (std::uint64_t context = 0; context < bankCount; ++context) {
            stateNext[context] = backedges.get(
                bodyBuilder.getIntegerType((*stateLayout)->encodedBitCount()));
            stateRegisters[context] = createRegister(
                bodyBuilder, location, stateNext[context],
                accessor.getInput("clock"), accessor.getInput("reset"),
                (*stateLayout)->resetValue(),
                temporal ? "operation_state_" + std::to_string(context) + "_reg"
                         : "operation_state_reg",
                clockReset.asynchronousReset);
          }
          selectedState =
              temporal ? selectContextValue(bodyBuilder, location,
                                            executionContext, stateRegisters)
                       : stateRegisters.front();
        }

        std::vector<circt::Backedge> dataNext(outputEndpoints.size());
        std::vector<mlir::Value> dataRegisters(outputEndpoints.size());
        std::vector<circt::Backedge> resultValidNext;
        std::vector<mlir::Value> resultValid;
        if (hasResultStorage) {
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
          resultValidNext.resize(outputEndpoints.size());
          resultValid.resize(outputEndpoints.size());
          for (std::size_t ordinal = 0; ordinal < outputEndpoints.size();
               ++ordinal) {
            resultValidNext[ordinal] = backedges.get(bodyBuilder.getI1Type());
            resultValid[ordinal] = createRegister(
                bodyBuilder, location, resultValidNext[ordinal],
                accessor.getInput("clock"), accessor.getInput("reset"),
                llvm::APInt(1, 0),
                outputEndpoints.size() == 1
                    ? "result_valid_reg"
                    : "result_valid_" + std::to_string(ordinal) + "_reg",
                clockReset.asynchronousReset);
          }
        }

        std::optional<ElasticResultTupleSignals> held;
        mlir::Value slotAvailable = bitConstant(bodyBuilder, location, true);
        if (hasResultStorage) {
          llvm::SmallVector<mlir::Value, 4> downstreamReady;
          for (const OperationEndpointPlan *endpoint : outputEndpoints)
            downstreamReady.push_back(
                accessor.getInput(endpoint->ready.getName()));
          auto tuple = deriveElasticResultTupleSignals(
              bodyBuilder, location, resultValid, downstreamReady);
          if (!tuple) {
            materializationError = llvm::toString(tuple.takeError());
            backedges.abandon();
            return;
          }
          held = std::move(*tuple);
          slotAvailable = held->available;
        }

        mlir::Value leafTransitionEnabled =
            andValues(bodyBuilder, location, {enabled, executionContextValid});
        if (interface->hasElasticResultStorage())
          leafTransitionEnabled = andValues(
              bodyBuilder, location, {leafTransitionEnabled, slotAvailable});
        mlir::Value acceptsInput = leafTransitionEnabled;
        if (interface->hasOrderedProductionGroups())
          acceptsInput = andValues(bodyBuilder, location,
                                   {leafTransitionEnabled,
                                    circt::comb::createOrFoldNot(
                                        bodyBuilder, location, continuation)});

        std::map<std::string, mlir::Value> leafInput;
        for (const OperationEndpointPlan *endpoint : inputEndpoints) {
          if (endpoint->data)
            leafInput.emplace("data_input_" + std::to_string(endpoint->ordinal),
                              accessor.getInput(endpoint->data->getName()));
          if (interface->hasTokenHandshake())
            leafInput.emplace(
                "valid_input_" + std::to_string(endpoint->ordinal),
                andValues(bodyBuilder, location,
                          {acceptsInput,
                           accessor.getInput(endpoint->valid.getName())}));
        }
        if (interface->hasTokenHandshake())
          for (const OperationEndpointPlan *endpoint : outputEndpoints) {
            mlir::Value ready =
                andValues(bodyBuilder, location,
                          {leafTransitionEnabled,
                           accessor.getInput(endpoint->ready.getName())});
            if (interface->hasElasticResultStorage())
              ready = leafTransitionEnabled;
            leafInput.emplace(
                "ready_output_" + std::to_string(endpoint->ordinal), ready);
          }
        if (*stateLayout)
          leafInput.emplace("state_current", selectedState);
        if (interface->hasOrderedProductionGroups())
          leafInput.emplace("continuation_current", continuation);
        for (const ConfigurationFieldPlan &field : configurationFields) {
          llvm::SmallVector<mlir::Value, 4> values;
          for (const FieldDecoderPlan &decoder : field.contexts)
            values.push_back(decodeFieldSignal(bodyBuilder, location,
                                               *configurationSignals, decoder));
          leafInput.emplace("config_" + std::to_string(field.ordinal),
                            values.size() == 1
                                ? values.front()
                                : selectContextValue(bodyBuilder, location,
                                                     executionContext, values));
        }
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

        // Project an offer through the same semantic transform before any
        // downstream admission. The transform is combinational: its state
        // outputs are unused here, and only the ordinary instance can commit
        // an operation transition. This derives case-dependent production
        // directly from the provider instead of defining a second case owner.
        std::map<std::string, mlir::Value> offeredOutput;
        if (interface->hasDirectTokenPublication()) {
          auto offerInput = leafInput;
          for (const OperationEndpointPlan *endpoint : inputEndpoints)
            offerInput["valid_input_" + std::to_string(endpoint->ordinal)] =
                andValues(bodyBuilder, location,
                          {acceptsInput,
                           accessor.getInput(endpoint->offer->getName())});
          for (const OperationEndpointPlan *endpoint : outputEndpoints)
            offerInput["ready_output_" + std::to_string(endpoint->ordinal)] =
                bitConstant(bodyBuilder, location, true);
          llvm::SmallVector<mlir::Value> offerOperands;
          for (const circt::hw::PortInfo &port : *leafPorts)
            if (!port.isOutput())
              offerOperands.push_back(offerInput.at(port.getName().str()));
          auto offerInstance = circt::hw::InstanceOp::create(
              bodyBuilder, location, leaf.getOperation(), "operation_offer",
              offerOperands);
          unsigned outputOrdinal = 0;
          for (const circt::hw::PortInfo &port : *leafPorts)
            if (port.isOutput())
              offeredOutput.emplace(port.getName().str(),
                                    offerInstance.getResult(outputOrdinal++));
        }

        if (*stateLayout) {
          mlir::Value write =
              andValues(bodyBuilder, location,
                        {leafTransitionEnabled, executionContextValid,
                         leafOutput.at("state_write")});
          for (std::uint64_t context = 0; context < stateNext.size();
               ++context) {
            mlir::Value selected =
                temporal ? contextEquals(bodyBuilder, location,
                                         executionContext, context)
                         : bitConstant(bodyBuilder, location, true);
            stateNext[context].setValue(circt::comb::MuxOp::create(
                bodyBuilder, location,
                andValues(bodyBuilder, location, {write, selected}),
                leafOutput.at("state_next"), stateRegisters[context], true));
          }
        }

        llvm::SmallVector<mlir::Value, 4> publishedValid;
        std::vector<mlir::Value> publishedData(outputEndpoints.size());
        llvm::SmallVector<mlir::Value, 4> operationInputReady;
        mlir::Value contextCapture = bitConstant(bodyBuilder, location, false);
        if (interface->protocol == FabricOperationLeafProtocol::Combinational) {
          llvm::SmallVector<mlir::Value, 4> inputValids;
          for (const OperationEndpointPlan *endpoint : inputEndpoints)
            inputValids.push_back(accessor.getInput(endpoint->valid.getName()));
          mlir::Value capacity =
              andValues(bodyBuilder, location,
                        {enabled, slotAvailable, executionContextValid});
          auto ready = deriveAtomicInputReadiness(bodyBuilder, location,
                                                  inputValids, capacity);
          if (!ready) {
            materializationError = llvm::toString(ready.takeError());
            backedges.abandon();
            return;
          }
          operationInputReady = std::move(*ready);
          mlir::Value accept = andValues(
              bodyBuilder, location,
              {capacity, andValues(bodyBuilder, location, inputValids)});
          contextCapture = accept;
          for (std::size_t ordinal = 0; ordinal < outputEndpoints.size();
               ++ordinal) {
            mlir::Value retain = andValues(
                bodyBuilder, location,
                {resultValid[ordinal],
                 circt::comb::createOrFoldNot(bodyBuilder, location,
                                              held->handoffs[ordinal])});
            resultValidNext[ordinal].setValue(
                orValues(bodyBuilder, location, {accept, retain}));
          }
          publishedValid = held->publishedValids;
          for (auto [ordinal, endpoint] : llvm::enumerate(outputEndpoints)) {
            if (!endpoint->data)
              continue;
            dataNext[ordinal].setValue(circt::comb::MuxOp::create(
                bodyBuilder, location, accept,
                leafOutput.at("data_output_" +
                              std::to_string(endpoint->ordinal)),
                dataRegisters[ordinal], true));
            publishedData[ordinal] = dataRegisters[ordinal];
          }
        } else if (interface->hasElasticResultStorage()) {
          llvm::SmallVector<mlir::Value, 4> producedValid;
          for (const OperationEndpointPlan *endpoint : outputEndpoints)
            producedValid.push_back(leafOutput.at(
                "valid_output_" + std::to_string(endpoint->ordinal)));
          mlir::Value capture =
              andValues(bodyBuilder, location,
                        {leafTransitionEnabled,
                         orValues(bodyBuilder, location, producedValid)});
          contextCapture = capture;
          if (interface->hasOrderedProductionGroups()) {
            mlir::Value capturedNonFinal =
                andValues(bodyBuilder, location,
                          {capture, circt::comb::createOrFoldNot(
                                        bodyBuilder, location,
                                        leafOutput.at("final_production"))});
            mlir::Value retainedContinuation = andValues(
                bodyBuilder, location,
                {continuation,
                 circt::comb::createOrFoldNot(bodyBuilder, location, capture)});
            continuationNext.setValue(
                orValues(bodyBuilder, location,
                         {capturedNonFinal, retainedContinuation}));
          }
          for (auto [ordinal, endpoint] : llvm::enumerate(outputEndpoints)) {
            mlir::Value retain = andValues(
                bodyBuilder, location,
                {resultValid[ordinal],
                 circt::comb::createOrFoldNot(bodyBuilder, location,
                                              held->handoffs[ordinal])});
            mlir::Value acquire = andValues(bodyBuilder, location,
                                            {capture, producedValid[ordinal]});
            resultValidNext[ordinal].setValue(
                orValues(bodyBuilder, location, {retain, acquire}));
            if (endpoint->data) {
              dataNext[ordinal].setValue(circt::comb::MuxOp::create(
                  bodyBuilder, location, capture,
                  leafOutput.at("data_output_" +
                                std::to_string(endpoint->ordinal)),
                  dataRegisters[ordinal], true));
              publishedData[ordinal] = dataRegisters[ordinal];
            }
          }
          publishedValid = held->publishedValids;
          for (const OperationEndpointPlan *endpoint : inputEndpoints)
            operationInputReady.push_back(
                andValues(bodyBuilder, location,
                          {acceptsInput,
                           leafOutput.at("ready_input_" +
                                         std::to_string(endpoint->ordinal))}));
        } else {
          for (auto [ordinal, endpoint] : llvm::enumerate(outputEndpoints)) {
            publishedValid.push_back(
                andValues(bodyBuilder, location,
                          {enabled, executionContextValid,
                           leafOutput.at("valid_output_" +
                                         std::to_string(endpoint->ordinal))}));
            if (endpoint->data)
              publishedData[ordinal] = leafOutput.at(
                  "data_output_" + std::to_string(endpoint->ordinal));
          }
          for (const OperationEndpointPlan *endpoint : inputEndpoints)
            operationInputReady.push_back(
                andValues(bodyBuilder, location,
                          {enabled, executionContextValid,
                           leafOutput.at("ready_input_" +
                                         std::to_string(endpoint->ordinal))}));
        }

        if (temporal && hasResultStorage)
          resultContextNext.setValue(circt::comb::MuxOp::create(
              bodyBuilder, location, contextCapture, executionContext,
              resultContext, true));

        for (auto [ordinal, endpoint] : llvm::enumerate(inputEndpoints))
          accessor.setOutput(endpoint->ready.getName(),
                             operationInputReady[ordinal]);
        for (auto [ordinal, endpoint] : llvm::enumerate(outputEndpoints)) {
          accessor.setOutput(endpoint->valid.getName(),
                             publishedValid[ordinal]);
          accessor.setOutput(
              endpoint->offer->getName(),
              hasResultStorage
                  ? publishedValid[ordinal]
                  : andValues(
                        bodyBuilder, location,
                        {enabled, executionContextValid,
                         offeredOutput.at("valid_output_" +
                                          std::to_string(endpoint->ordinal))}));
          if (endpoint->data)
            accessor.setOutput(endpoint->data->getName(),
                               publishedData[ordinal]);
          if (endpoint->context)
            accessor.setOutput(endpoint->context->getName(),
                               hasResultStorage ? resultContext
                                                : executionContext);
        }
      });
  if (materializationError)
    return invalid(*materializationError);
  return OperationShellModule{operation, shell, std::move(*endpoints),
                              std::move(*configuration)};
}

} // namespace

llvm::Expected<std::vector<OperationShellModule>> buildOperationShellModules(
    mlir::OpBuilder &builder, mlir::Location location,
    fabric::SpatialCoreOccurrenceRef spatialCore,
    const fabric::FabricArtifactView &fabric,
    const ConfigurationABI &configurationAbi,
    const ConfigurationTransportLayout &transportLayout,
    llvm::ArrayRef<ResolvedFabricPhysicalOperation> operations,
    std::vector<FabricOperationLeafAssociation> &associations,
    const ClockResetPlan &clockReset) {
  std::vector<OperationShellModule> result;
  result.reserve(operations.size());
  for (auto [index, operation] : llvm::enumerate(operations)) {
    auto shell = buildOperationShell(
        builder, location, spatialCore, fabric, configurationAbi,
        transportLayout, operation, index, associations, clockReset);
    if (!shell)
      return shell.takeError();
    result.push_back(std::move(*shell));
  }
  return result;
}

} // namespace loom::hardware::rtl::hierarchy

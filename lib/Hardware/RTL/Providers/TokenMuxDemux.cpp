#include "Hardware/RTL/Providers/TokenMuxDemux.h"

#include "Hardware/RTL/OperationLeaf.h"
#include "ProviderSupport.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

using PhysicalPort = fabric::ResolvedFabricOpPhysicalPortView;

struct Mode final {
  std::vector<std::uint64_t> lanes;
  unsigned selectorWidth = 0;
  unsigned payloadWidth = 0;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

struct ModeSignals final {
  std::vector<mlir::Value> inputReady;
  std::vector<mlir::Value> outputData;
  std::vector<mlir::Value> outputValid;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_token_mux_demux_invalid: " +
                                     message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

mlir::Value bitConstant(mlir::OpBuilder &builder, mlir::Location location,
                        bool value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(1, value));
}

mlir::Value andAll(mlir::OpBuilder &builder, mlir::Location location,
                   llvm::ArrayRef<mlir::Value> values) {
  if (values.empty())
    return bitConstant(builder, location, true);
  mlir::Value result = values.front();
  for (mlir::Value value : values.drop_front())
    result = circt::comb::AndOp::create(builder, location, result, value);
  return result;
}

mlir::Value orAll(mlir::OpBuilder &builder, mlir::Location location,
                  llvm::ArrayRef<mlir::Value> values) {
  if (values.empty())
    return bitConstant(builder, location, false);
  mlir::Value result = values.front();
  for (mlir::Value value : values.drop_front())
    result = circt::comb::OrOp::create(builder, location, result, value);
  return result;
}

llvm::Expected<unsigned> payloadWidth(mlir::Type type) {
  if (mlir::isa<mlir::NoneType>(type))
    return 0;
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  if (!integer || !integer.isSignless())
    return invalid("routed-token payload is not a signless integer or none");
  return integer.getWidth();
}

llvm::Expected<unsigned>
selectorWidth(const ::fabric::FiniteImplementationFamilyBehaviorPoint &point) {
  mlir::Type type = point.representativeActor.type.getInput(0);
  if (point.resolvedIndexWidth) {
    if (!mlir::isa<mlir::IndexType>(type))
      return invalid("resolved routed selector is not index typed");
    return ::fabric::getResolvedIndexBitWidth(*point.resolvedIndexWidth);
  }
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  if (!integer || !integer.isSignless())
    return invalid("routed selector is not a signless integer");
  return integer.getWidth();
}

llvm::Expected<Mode>
lowerMode(const ::fabric::FiniteImplementationFamilyBehaviorPoint &point,
          ::fabric::ImplementationFamilyId family,
          llvm::ArrayRef<const PhysicalPort *> inputs,
          llvm::ArrayRef<const PhysicalPort *> outputs,
          const FiniteCodebookEntry *codebookEntry) {
  const bool isMux = family == ::fabric::ImplementationFamilyId::TokenMux;
  const auto &actor = point.representativeActor;
  if (actor.schema != (isMux ? ::dataflow::OperationSchemaId::DataflowMux
                             : ::dataflow::OperationSchemaId::DataflowDemux))
    return invalid("Fabric returned a foreign routed-token behavior witness");

  Mode mode;
  mode.codebookEntry = codebookEntry;
  auto width = selectorWidth(point);
  if (!width)
    return width.takeError();
  mode.selectorWidth = *width;
  if (mode.selectorWidth == 0 ||
      inputs[0]->payloadWidthBits < mode.selectorWidth)
    return invalid("selector behavior exceeds its physical input width");

  if (isMux) {
    if (point.operandPorts.size() < 3 || point.operandPorts.front() != 0 ||
        point.resultPorts != std::vector<std::uint64_t>{0} ||
        actor.type.getNumInputs() != point.operandPorts.size() ||
        actor.type.getNumResults() != 1)
      return invalid("mux witness has the wrong arity or port correspondence");
    mode.lanes.assign(point.operandPorts.begin() + 1, point.operandPorts.end());
    auto activeWidth = payloadWidth(actor.type.getResult(0));
    if (!activeWidth)
      return activeWidth.takeError();
    mode.payloadWidth = *activeWidth;
    for (std::size_t lane = 0; lane != mode.lanes.size(); ++lane) {
      const std::uint64_t physical = mode.lanes[lane];
      if (physical == 0 || physical >= inputs.size() ||
          actor.type.getInput(lane + 1) != actor.type.getResult(0) ||
          inputs[physical]->payloadWidthBits < mode.payloadWidth)
        return invalid("mux witness exceeds its physical lane image");
    }
    if (outputs[0]->payloadWidthBits < mode.payloadWidth)
      return invalid("mux result exceeds its physical output width");
  } else {
    if (point.operandPorts != std::vector<std::uint64_t>({0, 1}) ||
        point.resultPorts.size() < 2 || actor.type.getNumInputs() != 2 ||
        actor.type.getNumResults() != point.resultPorts.size())
      return invalid(
          "demux witness has the wrong arity or port correspondence");
    mode.lanes = point.resultPorts;
    auto activeWidth = payloadWidth(actor.type.getInput(1));
    if (!activeWidth)
      return activeWidth.takeError();
    mode.payloadWidth = *activeWidth;
    if (inputs[1]->payloadWidthBits < mode.payloadWidth)
      return invalid("demux payload exceeds its physical input width");
    for (std::size_t lane = 0; lane != mode.lanes.size(); ++lane) {
      const std::uint64_t physical = mode.lanes[lane];
      if (physical >= outputs.size() ||
          actor.type.getResult(lane) != actor.type.getInput(1) ||
          outputs[physical]->payloadWidthBits < mode.payloadWidth)
        return invalid("demux witness exceeds its physical lane image");
    }
  }

  if (mode.lanes.size() == 2) {
    if (mode.selectorWidth != 1 || point.resolvedIndexWidth)
      return invalid("two-lane routed mode does not use its i1 selector");
  } else if (mode.selectorWidth != 32 ||
             point.resolvedIndexWidth != ::fabric::ResolvedIndexWidth::I32) {
    return invalid("wide routed mode does not use its resolved i32 selector");
  }
  if (!llvm::is_sorted(mode.lanes) ||
      std::adjacent_find(mode.lanes.begin(), mode.lanes.end()) !=
          mode.lanes.end())
    return invalid("routed-token lane image is not strictly ordered");
  return mode;
}

std::vector<mlir::Value> materializeLaneSelection(mlir::OpBuilder &builder,
                                                  mlir::Location location,
                                                  mlir::Value physicalSelector,
                                                  const Mode &mode) {
  mlir::Value selector = detail::resizeUnsigned(
      builder, location, physicalSelector, mode.selectorWidth);
  std::vector<mlir::Value> selected(mode.lanes.size());
  llvm::SmallVector<mlir::Value, 4> nonzeroSelections;
  for (std::size_t lane = 1; lane != mode.lanes.size(); ++lane) {
    mlir::Value index = circt::hw::ConstantOp::create(
        builder, location, llvm::APInt(mode.selectorWidth, lane));
    selected[lane] = circt::comb::ICmpOp::create(builder, location,
                                                 circt::comb::ICmpPredicate::eq,
                                                 selector, index, true);
    nonzeroSelections.push_back(selected[lane]);
  }
  selected[0] = circt::comb::createOrFoldNot(
      builder, location, orAll(builder, location, nonzeroSelections));
  return selected;
}

mlir::Value resizedPayload(mlir::OpBuilder &builder, mlir::Location location,
                           circt::hw::HWModulePortAccessor &accessor,
                           llvm::StringRef inputName, unsigned activeWidth,
                           unsigned outputWidth) {
  if (activeWidth == 0)
    return circt::hw::ConstantOp::create(builder, location,
                                         llvm::APInt(outputWidth, 0));
  mlir::Value active = detail::resizeUnsigned(
      builder, location, accessor.getInput(inputName), activeWidth);
  return detail::resizeUnsigned(builder, location, active, outputWidth);
}

ModeSignals materializeMuxMode(mlir::OpBuilder &builder,
                               mlir::Location location,
                               circt::hw::HWModulePortAccessor &accessor,
                               const Mode &mode,
                               llvm::ArrayRef<const PhysicalPort *> inputs,
                               llvm::ArrayRef<const PhysicalPort *> outputs) {
  mlir::Value falseValue = bitConstant(builder, location, false);
  std::vector<mlir::Value> selected = materializeLaneSelection(
      builder, location, accessor.getInput("data_input_0"), mode);
  mlir::Value selectorValid = accessor.getInput("valid_input_0");
  mlir::Value outputReady = accessor.getInput("ready_output_0");
  ModeSignals signals;
  signals.inputReady.assign(inputs.size(), falseValue);
  signals.outputValid.assign(outputs.size(), falseValue);
  if (outputs[0]->payloadWidthBits != 0)
    signals.outputData.assign(
        outputs.size(),
        mlir::Value(circt::hw::ConstantOp::create(
            builder, location, llvm::APInt(outputs[0]->payloadWidthBits, 0))));

  llvm::SmallVector<mlir::Value, 4> selectorReadyCases;
  llvm::SmallVector<mlir::Value, 4> resultValidCases;
  mlir::Value result =
      outputs[0]->payloadWidthBits == 0 ? mlir::Value() : signals.outputData[0];
  for (std::size_t lane = 0; lane != mode.lanes.size(); ++lane) {
    const unsigned physical = static_cast<unsigned>(mode.lanes[lane]);
    mlir::Value dataValid =
        accessor.getInput("valid_input_" + std::to_string(physical));
    selectorReadyCases.push_back(
        andAll(builder, location, {selected[lane], dataValid, outputReady}));
    signals.inputReady[physical] =
        andAll(builder, location, {selected[lane], selectorValid, outputReady});
    resultValidCases.push_back(
        andAll(builder, location, {selected[lane], selectorValid, dataValid}));
    if (result) {
      mlir::Value payload = resizedPayload(
          builder, location, accessor, "data_input_" + std::to_string(physical),
          mode.payloadWidth, outputs[0]->payloadWidthBits);
      result = circt::comb::MuxOp::create(builder, location, selected[lane],
                                          payload, result, true);
    }
  }
  signals.inputReady[0] = orAll(builder, location, selectorReadyCases);
  signals.outputValid[0] = orAll(builder, location, resultValidCases);
  if (result)
    signals.outputData[0] = result;
  return signals;
}

ModeSignals materializeDemuxMode(mlir::OpBuilder &builder,
                                 mlir::Location location,
                                 circt::hw::HWModulePortAccessor &accessor,
                                 const Mode &mode,
                                 llvm::ArrayRef<const PhysicalPort *> inputs,
                                 llvm::ArrayRef<const PhysicalPort *> outputs) {
  mlir::Value falseValue = bitConstant(builder, location, false);
  std::vector<mlir::Value> selected = materializeLaneSelection(
      builder, location, accessor.getInput("data_input_0"), mode);
  mlir::Value selectorValid = accessor.getInput("valid_input_0");
  mlir::Value dataValid = accessor.getInput("valid_input_1");
  ModeSignals signals;
  signals.inputReady.assign(inputs.size(), falseValue);
  signals.outputValid.assign(outputs.size(), falseValue);
  signals.outputData.resize(outputs.size());
  llvm::SmallVector<mlir::Value, 4> selectedReadyCases;
  for (std::size_t lane = 0; lane != mode.lanes.size(); ++lane) {
    const unsigned physical = static_cast<unsigned>(mode.lanes[lane]);
    mlir::Value ready =
        accessor.getInput("ready_output_" + std::to_string(physical));
    selectedReadyCases.push_back(
        circt::comb::AndOp::create(builder, location, selected[lane], ready));
    signals.outputValid[physical] =
        andAll(builder, location, {selected[lane], selectorValid, dataValid});
  }
  mlir::Value selectedReady = orAll(builder, location, selectedReadyCases);
  signals.inputReady[0] =
      circt::comb::AndOp::create(builder, location, dataValid, selectedReady);
  signals.inputReady[1] = circt::comb::AndOp::create(
      builder, location, selectorValid, selectedReady);

  for (std::size_t physical = 0; physical != outputs.size(); ++physical) {
    if (outputs[physical]->payloadWidthBits == 0)
      continue;
    mlir::Value result = circt::hw::ConstantOp::create(
        builder, location, llvm::APInt(outputs[physical]->payloadWidthBits, 0));
    const auto lane = llvm::find(mode.lanes, physical);
    if (lane != mode.lanes.end()) {
      mlir::Value payload = resizedPayload(builder, location, accessor,
                                           "data_input_1", mode.payloadWidth,
                                           outputs[physical]->payloadWidthBits);
      result = circt::comb::MuxOp::create(
          builder, location,
          selected[static_cast<std::size_t>(lane - mode.lanes.begin())],
          payload, result, true);
    }
    signals.outputData[physical] = result;
  }
  return signals;
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableTokenRouting(FabricOperationProviderRequest request,
                                ::fabric::ImplementationFamilyId family) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return unsupported(request);
  if (request.capability.implementationFamily != family)
    return invalid("provider received a different implementation family");
  const auto &descriptor = ::fabric::implementationFamily(family);
  if (::fabric::capabilityParamsSchema(
          request.capability.parameterizedCapability) !=
      descriptor.capabilityParamsSchema)
    return invalid("capability parameter schema does not match its family");
  const auto *parameters = std::get_if<::fabric::RoutedTokenParams>(
      &request.capability.parameterizedCapability);
  if (!parameters || parameters->maxPayloadBits == 0 || parameters->maxFan < 2)
    return invalid("capability has malformed routed-token parameters");
  const bool isMux = family == ::fabric::ImplementationFamilyId::TokenMux;
  const std::vector<::dataflow::OperationSchemaId> expectedSchemas{
      isMux ? ::dataflow::OperationSchemaId::DataflowMux
            : ::dataflow::OperationSchemaId::DataflowDemux};
  if (request.capability.enabledOperationSchemas != expectedSchemas)
    return invalid("capability does not contain exactly its routed schema");

  auto actualContract = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract = ::fabric::encodeResourceContractRecord(
      ::fabric::oneCycleElasticOperationResourceContract());
  if (!supportedContract)
    return supportedContract.takeError();
  if (*actualContract != *supportedContract)
    return unsupported(request);

  std::vector<const PhysicalPort *> inputs;
  std::vector<const PhysicalPort *> outputs;
  for (const PhysicalPort &port : request.capability.physicalPorts) {
    if (port.reference.direction == fabric::FabricPortDirection::Input)
      inputs.push_back(&port);
    else if (port.reference.direction == fabric::FabricPortDirection::Output)
      outputs.push_back(&port);
    else
      return invalid("capability has a physical port with unknown direction");
  }
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(inputs, byOrdinal);
  llvm::sort(outputs, byOrdinal);
  const auto contiguous = [](llvm::ArrayRef<const PhysicalPort *> ports) {
    return llvm::all_of(llvm::enumerate(ports), [](const auto &entry) {
      return entry.value()->reference.ordinal == entry.index();
    });
  };
  if (!contiguous(inputs) || !contiguous(outputs) ||
      (isMux && (inputs.size() < 3 || outputs.size() != 1)) ||
      (!isMux && (inputs.size() != 2 || outputs.size() < 2)))
    return invalid("capability does not have routed-token physical port roles");
  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto domain = relation->finiteBehaviorDomain();
  if (domain.empty())
    return invalid("Fabric returned an empty routed-token behavior domain");

  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  if (request.capability.configurationFieldSchema.empty()) {
    if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::None ||
        domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid("configuration-free routed capability is not a singleton");
  } else {
    if (relation->kind() !=
            ::fabric::FabricOpSemanticFieldRelationKind::Finite ||
        request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured routed capability is not one finite field");
    const auto fieldOrdinal =
        request.capability.configurationFieldSchema.front().ordinal;
    field = request.configurationAbi.findOperationField(request.occurrence,
                                                        fieldOrdinal);
    if (!field)
      return invalid("configured routed field is absent from ABI 2.0");
    codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("configured routed field is not a finite codebook");
    if (codebook->entries.size() != domain.size())
      return invalid("codebook does not exactly cover the routed domain");
  }

  std::set<const FiniteCodebookEntry *> usedEntries;
  std::vector<Mode> modes;
  modes.reserve(domain.size());
  for (const auto &point : domain) {
    const FiniteCodebookEntry *entry = nullptr;
    if (field) {
      if (!point.semanticConfiguration)
        return invalid("configured routed behavior has no semantic value");
      entry = detail::findFiniteCodebookEntry(
          *codebook, point.semanticConfiguration->bytes());
      if (!entry)
        return invalid("codebook omits an admitted routed behavior");
      if (!usedEntries.insert(entry).second)
        return invalid("codebook entry represents multiple routed behaviors");
    }
    auto mode = lowerMode(point, family, inputs, outputs, entry);
    if (!mode)
      return mode.takeError();
    modes.push_back(std::move(*mode));
  }
  if (codebook && usedEntries.size() != codebook->entries.size())
    return invalid("codebook contains a foreign routed behavior");

  std::size_t inactiveMode = 0;
  if (field) {
    const auto inactive = llvm::find_if(modes, [&](const Mode &mode) {
      return llvm::ArrayRef<std::uint8_t>(mode.codebookEntry->semanticValue)
          .equals(field->inactiveValue);
    });
    if (inactive == modes.end())
      return invalid("ABI inactive value is outside the routed domain");
    inactiveMode = static_cast<std::size_t>(inactive - modes.begin());
  }

  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        std::vector<ModeSignals> signals;
        signals.reserve(modes.size());
        for (const Mode &mode : modes)
          signals.push_back(
              isMux ? materializeMuxMode(bodyBuilder, location, accessor, mode,
                                         inputs, outputs)
                    : materializeDemuxMode(bodyBuilder, location, accessor,
                                           mode, inputs, outputs));

        std::vector<mlir::Value> selectedModes(modes.size());
        if (field) {
          mlir::Value configuration = accessor.getInput(
              "config_" +
              std::to_string(
                  request.capability.configurationFieldSchema.front().ordinal));
          for (std::size_t index = 0; index != modes.size(); ++index) {
            if (index == inactiveMode)
              continue;
            mlir::Value code = circt::hw::ConstantOp::create(
                bodyBuilder, location,
                detail::decodePhysicalCode(
                    modes[index].codebookEntry->physicalCode,
                    codebook->encodedBitCount));
            selectedModes[index] = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                configuration, code, true);
          }
        }

        const auto selectSignal = [&](auto projection) {
          std::vector<mlir::Value> values;
          values.reserve(signals.size());
          for (const ModeSignals &modeSignals : signals)
            values.push_back(projection(modeSignals));
          mlir::Value result = values[inactiveMode];
          for (std::size_t index = 0; index != modes.size(); ++index)
            if (index != inactiveMode && selectedModes[index])
              result = circt::comb::MuxOp::create(bodyBuilder, location,
                                                  selectedModes[index],
                                                  values[index], result, true);
          return result;
        };

        for (std::size_t input = 0; input != inputs.size(); ++input)
          accessor.setOutput("ready_input_" + std::to_string(input),
                             selectSignal([&](const ModeSignals &modeSignals) {
                               return modeSignals.inputReady[input];
                             }));
        for (std::size_t output = 0; output != outputs.size(); ++output) {
          if (outputs[output]->payloadWidthBits != 0)
            accessor.setOutput(
                "data_output_" + std::to_string(output),
                selectSignal([&](const ModeSignals &modeSignals) {
                  return modeSignals.outputData[output];
                }));
          accessor.setOutput("valid_output_" + std::to_string(output),
                             selectSignal([&](const ModeSignals &modeSignals) {
                               return modeSignals.outputValid[output];
                             }));
        }
      },
      request.leaf.getParametersAttr());
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableTokenMux(FabricOperationProviderRequest request) {
  return materializePortableTokenRouting(
      request, ::fabric::ImplementationFamilyId::TokenMux);
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableTokenDemux(FabricOperationProviderRequest request) {
  return materializePortableTokenRouting(
      request, ::fabric::ImplementationFamilyId::TokenDemux);
}

} // namespace

llvm::Error registerPortableTokenMuxDemuxProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  if (llvm::Error error =
          candidate.add({::fabric::ImplementationFamilyId::TokenMux,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableTokenMux}))
    return error;
  if (llvm::Error error =
          candidate.add({::fabric::ImplementationFamilyId::TokenDemux,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableTokenDemux}))
    return error;
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl

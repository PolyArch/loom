#include "Hardware/RTL/Providers/TokenConstantSync.h"

#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/Transport.h"
#include "ProviderSupport.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_token_constant_sync_invalid: " +
                                     message);
}

llvm::Expected<FabricOperationProviderOutput>
unsupported(const FabricOperationProviderRequest &request) {
  return llvm::make_error<FabricOperationProviderUnsupportedError>(
      request.capability.implementationFamily, request.recipe);
}

struct PhysicalPortInventory final {
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> inputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> outputs;
};

llvm::Expected<PhysicalPortInventory> derivePhysicalPortInventory(
    const fabric::ResolvedFabricOpCapabilityView &capability) {
  PhysicalPortInventory result;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       capability.physicalPorts) {
    if (port.reference.node != capability.occurrence)
      return invalid("physical port belongs to a different Fabric operation");
    if (port.reference.direction == fabric::FabricPortDirection::Input)
      result.inputs.push_back(&port);
    else if (port.reference.direction == fabric::FabricPortDirection::Output)
      result.outputs.push_back(&port);
    else
      return invalid("physical port has an unknown direction");
  }
  const auto byOrdinal = [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  };
  llvm::sort(result.inputs, byOrdinal);
  llvm::sort(result.outputs, byOrdinal);
  const auto dense =
      [](llvm::ArrayRef<const fabric::ResolvedFabricOpPhysicalPortView *>
             ports) {
        return llvm::all_of(llvm::enumerate(ports), [](const auto &entry) {
          return entry.value()->reference.ordinal == entry.index();
        });
      };
  if (!dense(result.inputs) || !dense(result.outputs))
    return invalid("physical port ordinals are not dense and unique");
  return result;
}

llvm::Expected<bool>
hasSupportedContract(const FabricOperationProviderRequest &request) {
  auto actual = ::fabric::encodeResourceContractRecord(
      request.capability.resourceStateAndTimingContract);
  if (!actual)
    return actual.takeError();
  auto supported = ::fabric::encodeResourceContractRecord(
      ::fabric::oneCycleElasticOperationResourceContract());
  if (!supported)
    return supported.takeError();
  return *actual == *supported;
}

mlir::Value bitConstant(mlir::OpBuilder &builder, mlir::Location location,
                        bool value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(1, value));
}

mlir::Value andAll(mlir::OpBuilder &builder, mlir::Location location,
                   llvm::ArrayRef<mlir::Value> values) {
  mlir::Value result = bitConstant(builder, location, true);
  for (mlir::Value value : values)
    result = circt::comb::AndOp::create(builder, location, result, value);
  return result;
}

llvm::Expected<unsigned> payloadWidth(mlir::Type type) {
  if (mlir::isa<mlir::NoneType>(type))
    return 0U;
  const auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!integer || !integer.isSignless())
    return invalid("sync relation lane is not signless integer or none");
  return integer.getWidth();
}

bool hasExactSchema(const fabric::ResolvedFabricOpCapabilityView &capability,
                    ::dataflow::OperationSchemaId schema) {
  return capability.enabledOperationSchemas ==
         std::vector<::dataflow::OperationSchemaId>{schema};
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableTokenConstant(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::TokenConstant)
    return invalid("constant provider received a different family");
  if (!std::holds_alternative<::fabric::PayloadCapacityParams>(
          request.capability.parameterizedCapability))
    return invalid("constant capability has the wrong parameter schema");
  if (!hasExactSchema(request.capability,
                      ::dataflow::OperationSchemaId::DataflowConstant))
    return invalid("constant capability does not expose its exact schema");

  auto supported = hasSupportedContract(request);
  if (!supported)
    return supported.takeError();
  if (!*supported)
    return unsupported(request);

  auto ports = derivePhysicalPortInventory(request.capability);
  if (!ports)
    return ports.takeError();
  if (ports->inputs.size() != 1 || ports->outputs.size() != 1 ||
      ports->outputs.front()->payloadWidthBits == 0)
    return unsupported(request);

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  if (relation->kind() != ::fabric::FabricOpSemanticFieldRelationKind::Direct ||
      !relation->directEncodedBitCount() ||
      request.capability.configurationFieldSchema.size() != 1)
    return invalid("constant capability has no exact direct carrier");
  const unsigned carrierWidth = *relation->directEncodedBitCount();
  const auto &semanticField =
      request.capability.configurationFieldSchema.front();
  const ConfigurationFieldEncoding *field =
      request.configurationAbi.findOperationField(request.occurrence,
                                                  semanticField.ordinal);
  if (!field)
    return invalid("constant direct carrier is absent from the ABI");
  const auto *direct =
      std::get_if<DirectBitsEncoding>(&field->semanticEncoding);
  if (!direct || direct->encodedBitCount != carrierWidth ||
      field->encodedBitCount() != carrierWidth)
    return invalid("constant ABI field does not match the direct carrier");
  if (carrierWidth > ports->outputs.front()->payloadWidthBits)
    return invalid("constant direct carrier exceeds its physical result");

  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  std::optional<std::string> materializationError;
  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value inputValid = accessor.getInput("valid_input_0");
        mlir::Value outputReady = accessor.getInput("ready_output_0");
        auto inputReady = deriveAtomicInputReadiness(bodyBuilder, location,
                                                     {inputValid}, outputReady);
        if (!inputReady) {
          materializationError = llvm::toString(inputReady.takeError());
          return;
        }
        auto outputTuple = deriveAtomicResultTupleSignals(
            bodyBuilder, location, {inputValid}, {outputReady});
        if (!outputTuple) {
          materializationError = llvm::toString(outputTuple.takeError());
          return;
        }
        accessor.setOutput("ready_input_0", inputReady->front());
        accessor.setOutput("valid_output_0",
                           outputTuple->publishedValids.front());
        accessor.setOutput(
            "data_output_0",
            detail::resizeUnsigned(
                bodyBuilder, location,
                accessor.getInput("config_" +
                                  std::to_string(semanticField.ordinal)),
                ports->outputs.front()->payloadWidthBits));
      },
      request.leaf.getParametersAttr());
  if (materializationError)
    return invalid(*materializationError);
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

struct SyncLane final {
  std::size_t physicalOrdinal = 0;
  unsigned payloadWidth = 0;
};

struct SyncMode final {
  std::vector<SyncLane> lanes;
  const FiniteCodebookEntry *codebookEntry = nullptr;
};

llvm::Expected<SyncMode>
deriveSyncMode(const ::fabric::FiniteImplementationFamilyBehaviorPoint &point,
               const PhysicalPortInventory &ports) {
  if (point.representativeActor.schema !=
      ::dataflow::OperationSchemaId::DataflowSync)
    return invalid("sync relation contains a different actor schema");
  if (point.operandPorts.empty() || point.operandPorts != point.resultPorts ||
      point.operandPorts.size() !=
          point.representativeActor.type.getNumInputs() ||
      point.resultPorts.size() !=
          point.representativeActor.type.getNumResults())
    return invalid("sync relation does not preserve one positional lane image");

  SyncMode mode;
  mode.lanes.reserve(point.operandPorts.size());
  std::optional<std::uint64_t> previous;
  for (auto [laneOrdinal, physicalOrdinal] :
       llvm::enumerate(point.operandPorts)) {
    if (physicalOrdinal >= ports.inputs.size() ||
        physicalOrdinal >= ports.outputs.size() ||
        (previous && physicalOrdinal <= *previous))
      return invalid("sync relation lane image is not ordered and unique");
    previous = physicalOrdinal;
    mlir::Type inputType = point.representativeActor.type.getInput(laneOrdinal);
    mlir::Type resultType =
        point.representativeActor.type.getResult(laneOrdinal);
    if (inputType != resultType)
      return invalid("sync relation changes a lane payload type");
    auto width = payloadWidth(inputType);
    if (!width)
      return width.takeError();
    if (*width > ports.inputs[physicalOrdinal]->payloadWidthBits ||
        *width > ports.outputs[physicalOrdinal]->payloadWidthBits)
      return invalid("sync relation lane exceeds a physical payload width");
    mode.lanes.push_back({static_cast<std::size_t>(physicalOrdinal), *width});
  }
  return mode;
}

llvm::Error validatePhysicalCode(const FiniteCodebookEntry &entry,
                                 std::uint64_t bitCount) {
  if (bitCount == 0 || bitCount > mlir::IntegerType::kMaxWidth)
    return invalid("sync codebook has an unsupported encoded width");
  const std::uint64_t byteCount = (bitCount + 7) / 8;
  if (entry.physicalCode.size() != byteCount)
    return invalid("sync codebook physical code has the wrong byte count");
  const unsigned usedFinalBits = bitCount % 8;
  if (usedFinalBits != 0 && (entry.physicalCode.back() >> usedFinalBits) != 0)
    return invalid("sync codebook physical code has nonzero padding bits");
  return llvm::Error::success();
}

struct SyncModeSignals final {
  llvm::SmallVector<mlir::Value, 4> inputReady;
  std::vector<mlir::Value> outputData;
  llvm::SmallVector<mlir::Value, 4> outputValid;
};

llvm::Expected<SyncModeSignals>
materializeSyncMode(mlir::OpBuilder &builder, mlir::Location location,
                    circt::hw::HWModulePortAccessor &accessor,
                    const SyncMode &mode, const PhysicalPortInventory &ports) {
  mlir::Value low = bitConstant(builder, location, false);
  llvm::SmallVector<mlir::Value, 4> selectedValids;
  llvm::SmallVector<mlir::Value, 4> selectedOutputReady;
  selectedValids.reserve(mode.lanes.size());
  selectedOutputReady.reserve(mode.lanes.size());
  for (const SyncLane &lane : mode.lanes) {
    selectedValids.push_back(accessor.getInput(
        "valid_input_" + std::to_string(lane.physicalOrdinal)));
    selectedOutputReady.push_back(accessor.getInput(
        "ready_output_" + std::to_string(lane.physicalOrdinal)));
  }

  auto selectedReady = deriveAtomicInputReadiness(
      builder, location, selectedValids,
      andAll(builder, location, selectedOutputReady));
  if (!selectedReady)
    return selectedReady.takeError();
  llvm::SmallVector<mlir::Value, 4> inputReady(ports.inputs.size(), low);
  for (auto [laneOrdinal, lane] : llvm::enumerate(mode.lanes))
    inputReady[lane.physicalOrdinal] = (*selectedReady)[laneOrdinal];

  mlir::Value allInputsValid = andAll(builder, location, selectedValids);
  llvm::SmallVector<mlir::Value, 4> heldValid(ports.outputs.size(), low);
  llvm::SmallVector<mlir::Value, 4> downstreamReady;
  downstreamReady.reserve(ports.outputs.size());
  for (std::size_t ordinal = 0; ordinal < ports.outputs.size(); ++ordinal)
    downstreamReady.push_back(
        accessor.getInput("ready_output_" + std::to_string(ordinal)));
  for (const SyncLane &lane : mode.lanes)
    heldValid[lane.physicalOrdinal] = allInputsValid;
  auto outputTuple = deriveAtomicResultTupleSignals(builder, location,
                                                    heldValid, downstreamReady);
  if (!outputTuple)
    return outputTuple.takeError();

  std::vector<mlir::Value> outputData(ports.outputs.size());
  for (std::size_t ordinal = 0; ordinal < ports.outputs.size(); ++ordinal) {
    const unsigned outputWidth = ports.outputs[ordinal]->payloadWidthBits;
    if (outputWidth != 0)
      outputData[ordinal] = circt::hw::ConstantOp::create(
          builder, location, llvm::APInt(outputWidth, 0));
  }
  for (const SyncLane &lane : mode.lanes) {
    const unsigned outputWidth =
        ports.outputs[lane.physicalOrdinal]->payloadWidthBits;
    if (outputWidth == 0)
      continue;
    if (lane.payloadWidth == 0) {
      outputData[lane.physicalOrdinal] = circt::hw::ConstantOp::create(
          builder, location, llvm::APInt(outputWidth, 0));
      continue;
    }
    mlir::Value value =
        accessor.getInput("data_input_" + std::to_string(lane.physicalOrdinal));
    value = detail::resizeUnsigned(builder, location, value, lane.payloadWidth);
    outputData[lane.physicalOrdinal] =
        detail::resizeUnsigned(builder, location, value, outputWidth);
  }
  return SyncModeSignals{std::move(inputReady), std::move(outputData),
                         std::move(outputTuple->publishedValids)};
}

llvm::Expected<FabricOperationProviderOutput>
materializePortableTokenSync(FabricOperationProviderRequest request) {
  if (request.recipe != BackendRecipeKey::PortableSystemVerilog)
    return invalid("provider received a non-portable recipe");
  if (request.capability.implementationFamily !=
      ::fabric::ImplementationFamilyId::TokenSync)
    return invalid("sync provider received a different family");
  if (!std::holds_alternative<::fabric::RoutedTokenParams>(
          request.capability.parameterizedCapability))
    return invalid("sync capability has the wrong parameter schema");
  if (!hasExactSchema(request.capability,
                      ::dataflow::OperationSchemaId::DataflowSync))
    return invalid("sync capability does not expose its exact schema");

  auto supported = hasSupportedContract(request);
  if (!supported)
    return supported.takeError();
  if (!*supported)
    return unsupported(request);

  auto ports = derivePhysicalPortInventory(request.capability);
  if (!ports)
    return ports.takeError();
  if (ports->inputs.empty() || ports->inputs.size() != ports->outputs.size())
    return unsupported(request);

  auto relation = request.capability.resolveSemanticFieldRelation(
      *request.leaf.getContext());
  if (!relation)
    return relation.takeError();
  const auto domain = relation->finiteBehaviorDomain();
  if (domain.empty())
    return invalid("sync relation has an empty behavior domain");

  const ConfigurationFieldEncoding *field = nullptr;
  const FiniteCodebookEncoding *codebook = nullptr;
  std::vector<SyncMode> modes;
  modes.reserve(domain.size());
  if (relation->kind() == ::fabric::FabricOpSemanticFieldRelationKind::None) {
    if (!request.capability.configurationFieldSchema.empty() ||
        domain.size() != 1 || domain.front().semanticConfiguration)
      return invalid("fieldless sync relation is not a singleton");
    auto mode = deriveSyncMode(domain.front(), *ports);
    if (!mode)
      return mode.takeError();
    modes.push_back(std::move(*mode));
  } else if (relation->kind() ==
             ::fabric::FabricOpSemanticFieldRelationKind::Finite) {
    if (request.capability.configurationFieldSchema.size() != 1)
      return invalid("configured sync relation does not own one field");
    const auto &semanticField =
        request.capability.configurationFieldSchema.front();
    field = request.configurationAbi.findOperationField(request.occurrence,
                                                        semanticField.ordinal);
    if (!field)
      return invalid("sync configuration field is absent from the ABI");
    codebook = std::get_if<FiniteCodebookEncoding>(&field->semanticEncoding);
    if (!codebook)
      return invalid("sync ABI field is not a finite codebook");
    if (codebook->entries.size() != domain.size())
      return invalid("sync codebook does not exactly cover the relation");
    for (const auto &point : domain) {
      if (!point.semanticConfiguration)
        return invalid("configured sync behavior has no semantic value");
      const FiniteCodebookEntry *entry = detail::findFiniteCodebookEntry(
          *codebook, point.semanticConfiguration->bytes());
      if (!entry)
        return invalid("sync codebook omits an admitted semantic value");
      if (llvm::Error error =
              validatePhysicalCode(*entry, codebook->encodedBitCount))
        return std::move(error);
      auto mode = deriveSyncMode(point, *ports);
      if (!mode)
        return mode.takeError();
      mode->codebookEntry = entry;
      modes.push_back(std::move(*mode));
    }
  } else {
    return invalid("sync relation has an unsupported carrier kind");
  }

  std::size_t inactiveMode = 0;
  if (field) {
    const auto inactive = llvm::find_if(modes, [&](const SyncMode &mode) {
      return llvm::ArrayRef<std::uint8_t>(mode.codebookEntry->semanticValue)
          .equals(field->inactiveValue);
    });
    if (inactive == modes.end())
      return invalid("sync ABI inactive value is outside the relation");
    inactiveMode = static_cast<std::size_t>(inactive - modes.begin());
  }

  if (llvm::Error error = verifyFabricOperationLeafPorts(
          request.leaf, request.occurrence, request.capability,
          request.configurationAbi))
    return std::move(error);

  std::optional<std::string> materializationError;
  mlir::OpBuilder builder(request.leaf.getContext());
  builder.setInsertionPoint(request.leaf);
  const mlir::Location location = request.leaf.getLoc();
  circt::hw::HWModuleOp::create(
      builder, location, request.leaf.getSymNameAttr(),
      circt::hw::ModulePortInfo(request.leaf.getPortList()),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        std::vector<SyncModeSignals> modeSignals;
        modeSignals.reserve(modes.size());
        for (const SyncMode &mode : modes) {
          auto signals = materializeSyncMode(bodyBuilder, location, accessor,
                                             mode, *ports);
          if (!signals) {
            materializationError = llvm::toString(signals.takeError());
            return;
          }
          modeSignals.push_back(std::move(*signals));
        }

        SyncModeSignals selected = modeSignals[inactiveMode];
        if (field) {
          const auto &semanticField =
              request.capability.configurationFieldSchema.front();
          mlir::Value configuration = accessor.getInput(
              "config_" + std::to_string(semanticField.ordinal));
          for (std::size_t modeIndex = 0; modeIndex < modes.size();
               ++modeIndex) {
            if (modeIndex == inactiveMode)
              continue;
            mlir::Value code = circt::hw::ConstantOp::create(
                bodyBuilder, location,
                detail::decodePhysicalCode(
                    modes[modeIndex].codebookEntry->physicalCode,
                    codebook->encodedBitCount));
            mlir::Value matches = circt::comb::ICmpOp::create(
                bodyBuilder, location, circt::comb::ICmpPredicate::eq,
                configuration, code, true);
            for (std::size_t ordinal = 0; ordinal < ports->inputs.size();
                 ++ordinal)
              selected.inputReady[ordinal] = circt::comb::MuxOp::create(
                  bodyBuilder, location, matches,
                  modeSignals[modeIndex].inputReady[ordinal],
                  selected.inputReady[ordinal], true);
            for (std::size_t ordinal = 0; ordinal < ports->outputs.size();
                 ++ordinal) {
              selected.outputValid[ordinal] = circt::comb::MuxOp::create(
                  bodyBuilder, location, matches,
                  modeSignals[modeIndex].outputValid[ordinal],
                  selected.outputValid[ordinal], true);
              if (ports->outputs[ordinal]->payloadWidthBits != 0)
                selected.outputData[ordinal] = circt::comb::MuxOp::create(
                    bodyBuilder, location, matches,
                    modeSignals[modeIndex].outputData[ordinal],
                    selected.outputData[ordinal], true);
            }
          }
        }

        for (std::size_t ordinal = 0; ordinal < ports->inputs.size(); ++ordinal)
          accessor.setOutput("ready_input_" + std::to_string(ordinal),
                             selected.inputReady[ordinal]);
        for (std::size_t ordinal = 0; ordinal < ports->outputs.size();
             ++ordinal) {
          if (ports->outputs[ordinal]->payloadWidthBits != 0)
            accessor.setOutput("data_output_" + std::to_string(ordinal),
                               selected.outputData[ordinal]);
          accessor.setOutput("valid_output_" + std::to_string(ordinal),
                             selected.outputValid[ordinal]);
        }
      },
      request.leaf.getParametersAttr());
  if (materializationError)
    return invalid(*materializationError);
  request.leaf.erase();
  return FabricOperationProviderOutput{};
}

} // namespace

llvm::Error registerPortableTokenConstantSyncProviders(
    FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  if (llvm::Error error =
          candidate.add({::fabric::ImplementationFamilyId::TokenConstant,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableTokenConstant}))
    return error;
  if (llvm::Error error =
          candidate.add({::fabric::ImplementationFamilyId::TokenSync,
                         BackendRecipeKey::PortableSystemVerilog,
                         {},
                         materializePortableTokenSync}))
    return error;
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl

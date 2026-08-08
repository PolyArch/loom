#include "Hardware/RTL/CommonSkeleton.h"

#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricFuCapabilityTemplate.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Hardware/RTL/OperationLeaf.h"
#include "Hardware/RTL/PhysicalOperation.h"
#include "Hardware/RTL/Transport.h"

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Conversion/SeqToSV.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::rtl {
char FabricStructuralLoweringUnsupportedError::ID = 0;

void FabricStructuralLoweringUnsupportedError::log(
    llvm::raw_ostream &stream) const {
  stream << "rtl_structural_lowering_unsupported: " << reason_;
}

std::error_code
FabricStructuralLoweringUnsupportedError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

namespace {

llvm::Error skeletonError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_skeleton_invalid: " + message);
}

bool isFabricOperationLeaf(circt::hw::HWModuleGeneratedOp module) {
  return module.getGeneratorKind() == fabricOperationGeneratorSchemaSymbol;
}

llvm::Expected<std::set<std::vector<std::uint8_t>>>
expectedOperationOccurrences(
    const fabric::FabricSystemRootView &system,
    std::optional<fabric::SpatialCoreOccurrenceRef> spatialCore) {
  std::set<std::vector<std::uint8_t>> result;
  auto operations = enumerateFabricPhysicalOperations(system);
  if (!operations)
    return operations.takeError();
  for (const ResolvedFabricPhysicalOperation &operation : *operations) {
    if (spatialCore) {
      const auto &internal = std::get<fabric::SpatialCoreInternalOccurrenceRef>(
          operation.physicalOccurrence.payload());
      if (internal.spatialCore != *spatialCore)
        continue;
    }
    if (!result
             .insert(fabric::canonicalFabricBytes(operation.physicalOccurrence))
             .second)
      return skeletonError(
          "Fabric operation occurrence inventory is not unique");
  }
  return result;
}

llvm::Error verifyNoUnresolvedFabricOperationLeaves(mlir::ModuleOp module) {
  bool unresolved = false;
  module.walk([&](circt::hw::HWModuleGeneratedOp leaf) {
    unresolved |= isFabricOperationLeaf(leaf);
  });
  if (unresolved)
    return skeletonError("unresolved Loom Fabric operation leaf reached "
                         "SystemVerilog export");
  return llvm::Error::success();
}

llvm::Error verifyNoUnresolvedStructuralLowering(mlir::ModuleOp module) {
  bool unresolved = false;
  module.walk([&](mlir::UnrealizedConversionCastOp) { unresolved = true; });
  if (unresolved)
    return skeletonError("unresolved structural lowering remains in CIRCT "
                         "module");
  return llvm::Error::success();
}

struct BoundaryPassthroughPlan final {
  const ModuleBoundaryTransportPortProjection *input;
  const ModuleBoundaryTransportPortProjection *output;
  ::fabric::DataPathType inputType;
  ::fabric::DataPathType outputType;
};

void appendBoundaryPorts(
    llvm::SmallVectorImpl<circt::hw::PortInfo> &inputs,
    llvm::SmallVectorImpl<circt::hw::PortInfo> &outputs,
    const ModuleBoundaryTransportPortProjection &boundary) {
  const auto append = [&](const circt::hw::PortInfo &port) {
    (port.isOutput() ? outputs : inputs).push_back(port);
  };
  if (boundary.data)
    append(*boundary.data);
  if (boundary.tag)
    append(*boundary.tag);
  append(boundary.valid);
  append(boundary.ready);
}

llvm::Error structuralUnsupported(const llvm::Twine &message) {
  return llvm::make_error<FabricStructuralLoweringUnsupportedError>(
      message.str());
}

struct BoundaryChannelPlan final {
  const ModuleBoundaryTransportPortProjection *projection = nullptr;
  fabric::FabricTransportEndpointRef endpoint;
  ::fabric::DataPathType dataPath;
};

struct FieldDecoderPlan final {
  const ProgrammingUnit *unit = nullptr;
  std::uint64_t encodedBitCount = 0;
  std::vector<std::uint64_t> destinationBits;
};

struct CodeMatchPlan final {
  fabric::FabricTransportEndpointRef endpoint;
  llvm::APInt code;
};

struct ActivationPlan final {
  FieldDecoderPlan decoder;
  llvm::APInt activeCode;
};

struct InputSelectorPlan final {
  FieldDecoderPlan decoder;
  std::vector<CodeMatchPlan> routes;
  std::vector<CodeMatchPlan> discards;
};

struct OutputSelectorPlan final {
  FieldDecoderPlan decoder;
  std::vector<CodeMatchPlan> routes;
  std::optional<llvm::APInt> discard;
};

struct ClockResetPlan final {
  bool asynchronousReset = false;
  bool activeLowReset = false;
};

struct OperationConfigurationPlan final {
  fabric::FabricOrdinal ordinal = 0;
  FieldDecoderPlan decoder;
};

struct InternalOperationPlan final {
  ResolvedFabricPhysicalOperation operation;
  FabricOperationLeafInterface interface;
  std::optional<FabricOperationLeafStateLayout> stateLayout;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> physicalInputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> physicalOutputs;
  std::vector<BoundaryChannelPlan> inputs;
  std::vector<BoundaryChannelPlan> outputs;
  ActivationPlan activation;
  std::vector<InputSelectorPlan> inputSelectors;
  std::vector<OutputSelectorPlan> outputSelectors;
  std::vector<OperationConfigurationPlan> operationConfiguration;
  std::vector<const ProgrammingUnit *> programmingUnits;
  ClockResetPlan clockReset;
};

llvm::Expected<fabric::FabricPhysicalConfigurationFieldRef>
qualifyConfigurationField(fabric::SpatialCoreOccurrenceRef spatialCore,
                          const fabric::FabricSemanticConfigFieldRef &field) {
  auto target = fabric::FabricModulePhysicalTargetRef::create(field);
  if (!target)
    return target.takeError();
  return fabric::FabricPhysicalConfigurationFieldRef::create(
      fabric::SpatialCoreInternalOccurrenceRef{spatialCore,
                                               std::move(*target)});
}

llvm::Expected<FieldDecoderPlan>
prepareFieldDecoder(const ConfigurationFieldEncoding &encoding,
                    const ConfigurationABI &configurationAbi) {
  const ProgrammingUnit *owner = nullptr;
  for (const ProgrammingUnit &unit : configurationAbi.programmingUnits())
    for (const ConfigurationFieldEncoding &candidate : unit.fields)
      if (&candidate == &encoding) {
        if (owner)
          return skeletonError(
              "configuration field has duplicate programming owners");
        owner = &unit;
      }
  if (!owner)
    return skeletonError("configuration field has no programming owner");

  const std::uint64_t width = encoding.encodedBitCount();
  if (width == 0 || width > mlir::IntegerType::kMaxWidth)
    return structuralUnsupported(
        "PE configuration field width exceeds the CIRCT support envelope");
  std::vector<std::uint64_t> destinationBits(static_cast<std::size_t>(width),
                                             UINT64_MAX);
  for (const DestinationSlice &slice : encoding.destinationSlices) {
    if (slice.sourceBitOffset > width || slice.bitCount > width ||
        slice.sourceBitOffset + slice.bitCount > width ||
        slice.destinationBitOffset > owner->payloadBitCount ||
        slice.bitCount > owner->payloadBitCount ||
        slice.destinationBitOffset + slice.bitCount > owner->payloadBitCount)
      return skeletonError("configuration destination slice is out of range");
    for (std::uint64_t bit = 0; bit < slice.bitCount; ++bit) {
      const std::size_t source =
          static_cast<std::size_t>(slice.sourceBitOffset + bit);
      if (destinationBits[source] != UINT64_MAX)
        return skeletonError(
            "configuration destination slices overlap one source bit");
      destinationBits[source] = slice.destinationBitOffset + bit;
    }
  }
  if (llvm::is_contained(destinationBits, UINT64_MAX))
    return skeletonError(
        "configuration destination slices do not cover the field");
  return FieldDecoderPlan{owner, width, std::move(destinationBits)};
}

llvm::Expected<FieldDecoderPlan>
prepareFieldDecoder(fabric::SpatialCoreOccurrenceRef spatialCore,
                    const fabric::FabricSemanticConfigFieldRef &field,
                    const ConfigurationABI &configurationAbi) {
  auto physical = qualifyConfigurationField(spatialCore, field);
  if (!physical)
    return physical.takeError();
  const ConfigurationFieldEncoding *encoding =
      configurationAbi.findField(*physical);
  if (!encoding)
    return skeletonError("PE configuration field is absent from the ABI");
  return prepareFieldDecoder(*encoding, configurationAbi);
}

llvm::Expected<llvm::APInt>
physicalCode(const FiniteCodebookEncoding &codebook,
             llvm::ArrayRef<std::uint8_t> semanticValue) {
  const FiniteCodebookEntry *entry = nullptr;
  for (const FiniteCodebookEntry &candidate : codebook.entries)
    if (llvm::ArrayRef<std::uint8_t>(candidate.semanticValue)
            .equals(semanticValue)) {
      if (entry)
        return skeletonError("PE codebook repeats one semantic value");
      entry = &candidate;
    }
  if (!entry)
    return skeletonError("PE codebook omits one semantic value");
  if (entry->physicalCode.size() <
      (codebook.encodedBitCount + std::uint64_t(7)) / std::uint64_t(8))
    return skeletonError("PE codebook physical code is truncated");
  llvm::APInt result(static_cast<unsigned>(codebook.encodedBitCount), 0);
  for (std::uint64_t bit = 0; bit < codebook.encodedBitCount; ++bit)
    if (((entry->physicalCode[static_cast<std::size_t>(bit / 8)] >> (bit % 8)) &
         1U) != 0)
      result.setBit(static_cast<unsigned>(bit));
  return result;
}

llvm::Expected<std::pair<FieldDecoderPlan, const FiniteCodebookEncoding *>>
prepareFiniteField(fabric::SpatialCoreOccurrenceRef spatialCore,
                   const fabric::FabricSemanticConfigFieldRef &field,
                   const ConfigurationABI &configurationAbi) {
  auto decoder = prepareFieldDecoder(spatialCore, field, configurationAbi);
  if (!decoder)
    return decoder.takeError();
  auto physical = qualifyConfigurationField(spatialCore, field);
  if (!physical)
    return physical.takeError();
  const auto *codebook = std::get_if<FiniteCodebookEncoding>(
      &configurationAbi.findField(*physical)->semanticEncoding);
  if (!codebook || codebook->encodedBitCount != decoder->encodedBitCount)
    return structuralUnsupported(
        "PE selector requires one exact finite ABI codebook");
  return std::make_pair(std::move(*decoder), codebook);
}

bool hasTerminalEdge(
    llvm::ArrayRef<fabric::FabricFuCapabilityTemplateEdge> edges,
    const fabric::FabricFuCapabilityTemplateEdge &expected) {
  return llvm::is_contained(edges, expected);
}

llvm::Error validateOperationCapabilityTopology(
    const fabric::FabricArtifactView &module,
    const ResolvedFabricPhysicalOperation &operation,
    fabric::FabricFuOccurrenceRef fu,
    llvm::ArrayRef<const fabric::ResolvedFabricOpPhysicalPortView *> inputs,
    llvm::ArrayRef<const fabric::ResolvedFabricOpPhysicalPortView *> outputs) {
  const auto definition = module.fuTemplateOf(fu);
  if (!definition)
    return skeletonError("internal FU has no exact definition");
  const auto templates = module.fuCapabilityTemplates(*definition);
  if (templates.size() != 1)
    return structuralUnsupported(
        "single-operation lowering requires one FU capability template");
  const fabric::FabricFuCapabilityTemplateRecord &record = templates.front();
  if (record.activeNodes.size() != 1 ||
      record.activeNodes.front() != operation.capability->occurrence)
    return structuralUnsupported(
        "operation capability template has a different active node set");
  auto edges = fabric::projectFabricFuCapabilityTemplateTerminalEdges(record);
  if (!edges)
    return edges.takeError();
  const auto boundary = [](fabric::FabricFuTemplateRef owner,
                           fabric::FabricPortDirection direction,
                           fabric::FabricOrdinal ordinal) {
    return fabric::FabricFuCapabilityTemplateEndpointRef::boundaryPort(
        {owner, direction, ordinal});
  };
  const auto node = [&](fabric::FabricPortDirection direction,
                        fabric::FabricOrdinal ordinal) {
    return fabric::FabricFuCapabilityTemplateEndpointRef::nodePort(
        {operation.capability->occurrence, direction, ordinal});
  };
  std::vector<fabric::FabricFuCapabilityTemplateEdge> expected;
  expected.reserve(inputs.size() + outputs.size());
  for (const auto *input : inputs)
    expected.push_back(
        {boundary(*definition, fabric::FabricPortDirection::Input,
                  input->reference.ordinal),
         node(fabric::FabricPortDirection::Input, input->reference.ordinal)});
  for (const auto *output : outputs)
    expected.push_back(
        {node(fabric::FabricPortDirection::Output, output->reference.ordinal),
         boundary(*definition, fabric::FabricPortDirection::Output,
                  output->reference.ordinal)});
  if (edges->size() != expected.size() ||
      !llvm::all_of(expected, [&](const auto &edge) {
        return hasTerminalEdge(*edges, edge);
      }))
    return structuralUnsupported(
        "operation capability template has a different terminal relation");
  auto local = fabric::deriveFabricFuOccurrenceNode(
      module, operation.capability->occurrence, fu);
  if (!local)
    return local.takeError();
  if (*local != operation.localOccurrence)
    return skeletonError(
        "physical operation does not match its FU capability node");
  return llvm::Error::success();
}

llvm::Expected<ClockResetPlan>
prepareClockReset(const fabric::FabricSystemRootView &system,
                  fabric::SpatialCoreOccurrenceRef spatialCore) {
  const fabric::FabricInventoryOwnerRef owner =
      fabric::FabricInventoryOwnerRef::of(spatialCore);
  const fabric::HardwareDomainContractRecord *clock = nullptr;
  const fabric::HardwareDomainContractRecord *reset = nullptr;
  std::optional<fabric::ClockDomainRef> clockReference;
  for (fabric::HardwareDomainRef domain : system.hardwareDomains()) {
    const fabric::HardwareDomainContractRecord *contract =
        system.hardwareDomainContract(domain);
    if (!contract || !llvm::is_contained(contract->members(), owner))
      continue;
    if (contract->kind() == fabric::FabricHardwareDomainKind::Clock) {
      if (clock)
        return skeletonError("SpatialCore belongs to multiple Clock domains");
      clock = contract;
      clockReference = fabric::ClockDomainRef(domain);
    } else if (contract->kind() == fabric::FabricHardwareDomainKind::Reset) {
      if (reset)
        return skeletonError("SpatialCore belongs to multiple Reset domains");
      reset = contract;
    }
  }
  if (!clock || !reset || !clockReference)
    return structuralUnsupported(
        "operation lowering requires exact Clock and Reset domains");
  if (!std::get_if<fabric::ClockDomainContractRecord>(&clock->contract()))
    return skeletonError("Clock domain carries a non-Clock contract");
  const auto *resetContract =
      std::get_if<fabric::ResetDomainContractRecord>(&reset->contract());
  if (!resetContract)
    return skeletonError("Reset domain carries a non-Reset contract");
  if (resetContract->initialState() != fabric::ResetInitialState::Asserted)
    return structuralUnsupported(
        "operation lowering requires an initially asserted Reset");
  if (resetContract->releaseLatencyCycles() != 0)
    return structuralUnsupported(
        "reset release latency requires structural synchronization support");
  const bool asynchronous =
      resetContract->assertion() == fabric::ResetTiming::Asynchronous &&
      resetContract->deassertion() == fabric::ResetTiming::Asynchronous;
  const bool synchronous =
      resetContract->assertion() == fabric::ResetTiming::Synchronous &&
      resetContract->deassertion() == fabric::ResetTiming::Synchronous;
  if (!asynchronous && !synchronous)
    return structuralUnsupported(
        "mixed Reset assertion and deassertion timing is unsupported");
  if (synchronous && resetContract->synchronousTo() != clockReference)
    return skeletonError(
        "synchronous Reset does not name the SpatialCore Clock domain");
  return ClockResetPlan{asynchronous, resetContract->polarity() ==
                                          fabric::ResetPolarity::ActiveLow};
}

const ModuleBoundaryTransportPortProjection *findProjection(
    llvm::ArrayRef<ModuleBoundaryTransportPortProjection> projections,
    const fabric::FabricModuleBoundaryEndpointRef &boundary) {
  const ModuleBoundaryTransportPortProjection *result = nullptr;
  for (const ModuleBoundaryTransportPortProjection &candidate : projections)
    if (candidate.boundary == boundary) {
      if (result)
        return nullptr;
      result = &candidate;
    }
  return result;
}

const ::fabric::ResourceContract &
supportedOperationResourceContract(::fabric::ImplementationFamilyId family) {
  switch (family) {
  case ::fabric::ImplementationFamilyId::LoopStream:
    return ::fabric::loopStreamOperationResourceContract();
  case ::fabric::ImplementationFamilyId::LoopCarry:
    return ::fabric::loopCarryOperationResourceContract();
  case ::fabric::ImplementationFamilyId::LoopInvariant:
    return ::fabric::loopInvariantOperationResourceContract();
  case ::fabric::ImplementationFamilyId::LoopGate:
    return ::fabric::loopGateOperationResourceContract();
  default:
    return ::fabric::oneCycleElasticOperationResourceContract();
  }
}

llvm::Expected<InternalOperationPlan> prepareInternalOperation(
    fabric::SpatialCoreOccurrenceRef spatialCore,
    const ConfigurationABI &configurationAbi,
    const fabric::FabricArtifactView &module,
    llvm::ArrayRef<ModuleBoundaryTransportPortProjection> projections) {
  if (module.peOccurrences().size() != 1 ||
      module.fuOccurrences().size() != 1 ||
      !module.memoryOccurrences().empty() ||
      !module.switchOccurrences().empty() ||
      !module.fifoOccurrences().empty() ||
      !module.boundaryOccurrences().empty() ||
      !module.pointConnections().empty() ||
      !module.moduleBoundaryMemoryAttachments().empty() ||
      !module.moduleBoundaryTransportPassthroughs().empty())
    return structuralUnsupported(
        "internal topology is outside the single-operation support envelope");
  const auto pe = module.peOccurrences().front();
  const auto fu = module.fuOccurrences().front();
  if (module.parentPeOf(fu) != pe)
    return structuralUnsupported("operation FU is not owned by its sole PE");

  auto operations =
      enumerateFabricPhysicalOperations(configurationAbi.fabricSystem());
  if (!operations)
    return operations.takeError();
  std::vector<ResolvedFabricPhysicalOperation> selected;
  for (ResolvedFabricPhysicalOperation &operation : *operations) {
    if (operation.physicalOccurrence.kind() !=
        fabric::FabricPhysicalOccurrenceOwnerKind::SpatialCoreInternal)
      continue;
    const auto &internal = std::get<fabric::SpatialCoreInternalOccurrenceRef>(
        operation.physicalOccurrence.payload());
    if (internal.spatialCore == spatialCore)
      selected.push_back(std::move(operation));
  }
  if (selected.size() != 1)
    return structuralUnsupported(
        "single-operation lowering requires one physical occurrence");
  ResolvedFabricPhysicalOperation operation = std::move(selected.front());
  if (operation.localOccurrence.fu != fu)
    return skeletonError("physical operation belongs to a different FU");

  auto interface = deriveFabricOperationLeafInterface(*operation.capability);
  if (!interface)
    return interface.takeError();
  auto stateLayout =
      deriveFabricOperationLeafStateLayout(*operation.capability);
  if (!stateLayout)
    return stateLayout.takeError();
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> physicalInputs;
  std::vector<const fabric::ResolvedFabricOpPhysicalPortView *> physicalOutputs;
  for (const fabric::ResolvedFabricOpPhysicalPortView &port :
       operation.capability->physicalPorts)
    (port.reference.direction == fabric::FabricPortDirection::Input
         ? physicalInputs
         : physicalOutputs)
        .push_back(&port);
  llvm::sort(physicalInputs, [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  });
  llvm::sort(physicalOutputs, [](const auto *lhs, const auto *rhs) {
    return lhs->reference.ordinal < rhs->reference.ordinal;
  });
  if (physicalInputs.empty() || physicalOutputs.empty())
    return structuralUnsupported(
        "single-operation lowering requires nonempty input and output tuples");
  const auto denseAndRepresentable =
      [](llvm::ArrayRef<const fabric::ResolvedFabricOpPhysicalPortView *>
             ports) {
        return llvm::all_of(llvm::enumerate(ports), [](const auto &entry) {
          return entry.value()->reference.ordinal == entry.index() &&
                 entry.value()->payloadWidthBits <=
                     mlir::IntegerType::kMaxWidth;
        });
      };
  if (!denseAndRepresentable(physicalInputs) ||
      !denseAndRepresentable(physicalOutputs))
    return structuralUnsupported(
        "operation physical ports are not dense CIRCT-width carriers");
  auto actualContract = ::fabric::encodeResourceContractRecord(
      operation.capability->resourceStateAndTimingContract);
  if (!actualContract)
    return actualContract.takeError();
  auto supportedContract =
      ::fabric::encodeResourceContractRecord(supportedOperationResourceContract(
          operation.capability->implementationFamily));
  if (!supportedContract)
    return supportedContract.takeError();
  if (*actualContract != *supportedContract)
    return structuralUnsupported(
        "operation resource contract does not match its structural protocol");
  if (llvm::Error error = validateOperationCapabilityTopology(
          module, operation, fu, physicalInputs, physicalOutputs))
    return std::move(error);

  const auto root = module.moduleRootTemplate();
  if (!root ||
      module.moduleBoundaryEndpointCount(
          *root, fabric::FabricPortDirection::Input) != physicalInputs.size() ||
      module.moduleBoundaryEndpointCount(*root,
                                         fabric::FabricPortDirection::Output) !=
          physicalOutputs.size() ||
      module.moduleBoundaryTransportAttachments().size() !=
          physicalInputs.size() + physicalOutputs.size() ||
      projections.size() != physicalInputs.size() + physicalOutputs.size())
    return structuralUnsupported(
        "operation Module boundary does not match its physical tuple arity");
  std::vector<BoundaryChannelPlan> inputs(physicalInputs.size());
  std::vector<BoundaryChannelPlan> outputs(physicalOutputs.size());
  const auto peOwner = fabric::FabricTransportEndpointOwnerRef::of(pe);
  for (const auto &attachment : module.moduleBoundaryTransportAttachments()) {
    if (attachment.endpoint.owner != peOwner)
      return structuralUnsupported(
          "operation boundary attachment does not terminate on its PE");
    const auto type =
        module.moduleBoundaryEndpointDataPath(attachment.boundary);
    const auto *projection = findProjection(projections, attachment.boundary);
    if (!type || type->kind != ::fabric::DataPathKind::Bits || !projection ||
        static_cast<bool>(projection->data) != (type->payloadWidthBits != 0) ||
        projection->tag)
      return structuralUnsupported(
          "operation boundary channel is not one canonical untagged token");
    if (attachment.boundary.direction == fabric::FabricPortDirection::Input) {
      if (attachment.boundary.ordinal >= inputs.size() ||
          inputs[attachment.boundary.ordinal].projection)
        return skeletonError("operation input boundary is not one-to-one");
      inputs[attachment.boundary.ordinal] = {projection, attachment.endpoint,
                                             *type};
    } else {
      if (attachment.boundary.ordinal >= outputs.size() ||
          outputs[attachment.boundary.ordinal].projection)
        return skeletonError("operation output boundary is not one-to-one");
      outputs[attachment.boundary.ordinal] = {projection, attachment.endpoint,
                                              *type};
    }
  }
  const auto incomplete = [](const BoundaryChannelPlan &channel) {
    return channel.projection == nullptr;
  };
  if (llvm::any_of(inputs, incomplete) || llvm::any_of(outputs, incomplete))
    return skeletonError("operation Module boundary is incomplete");

  auto schema = module.spatialPeConfigurationSchema(pe);
  if (!schema)
    return schema.takeError();
  if (schema->fields().size() !=
      1 + physicalInputs.size() + physicalOutputs.size())
    return structuralUnsupported(
        "operation PE configuration schema has the wrong field count");
  std::optional<ActivationPlan> activation;
  std::vector<std::optional<InputSelectorPlan>> inputSelectors(
      physicalInputs.size());
  std::vector<std::optional<OutputSelectorPlan>> outputSelectors(
      physicalOutputs.size());
  std::map<ProgrammingUnitId, const ProgrammingUnit *> programmingUnits;
  for (const fabric::FabricPeConfigurationFieldView &descriptor :
       schema->fields()) {
    auto prepared =
        prepareFiniteField(spatialCore, descriptor.reference, configurationAbi);
    if (!prepared)
      return prepared.takeError();
    FieldDecoderPlan decoder = std::move(prepared->first);
    const FiniteCodebookEncoding &codebook = *prepared->second;
    programmingUnits.emplace(decoder.unit->id, decoder.unit);
    if (descriptor.kind == fabric::FabricPeConfigurationFieldKind::Activation) {
      if (activation || descriptor.port)
        return skeletonError("PE activation field is not unique");
      auto semantic =
          schema->encode(descriptor.reference, fabric::FabricPeActive{fu});
      if (!semantic)
        return semantic.takeError();
      auto code = physicalCode(codebook, semantic->bytes());
      if (!code)
        return code.takeError();
      activation = ActivationPlan{std::move(decoder), std::move(*code)};
      continue;
    }
    if (!descriptor.port || descriptor.port->fu != fu)
      return structuralUnsupported(
          "PE selector field names a different FU occurrence");
    const auto attachments =
        module.fuOccurrencePortAttachments(*descriptor.port);
    if (attachments.empty())
      return skeletonError("FU selector port has no sealed attachment domain");
    auto domain = schema->finiteDomain(descriptor.reference);
    if (!domain)
      return domain.takeError();
    if (descriptor.kind ==
        fabric::FabricPeConfigurationFieldKind::InputSelector) {
      if (descriptor.port->direction != fabric::FabricPortDirection::Input ||
          descriptor.port->ordinal >= inputSelectors.size() ||
          inputSelectors[descriptor.port->ordinal])
        return skeletonError("PE input selector field is not one-to-one");
      InputSelectorPlan selector{std::move(decoder), {}, {}};
      for (const fabric::FabricPeConfigurationValue &value : *domain) {
        const fabric::FabricTransportEndpointRef *endpoint = nullptr;
        auto *destination = &selector.routes;
        if (const auto *route = std::get_if<fabric::FabricPeRoute>(&value))
          endpoint = &route->endpoint;
        else if (const auto *discard =
                     std::get_if<fabric::FabricPeInputDiscard>(&value)) {
          endpoint = &discard->endpoint;
          destination = &selector.discards;
        } else
          continue;
        if (!llvm::any_of(attachments,
                          [&](const auto &attachment) {
                            return attachment.endpoint == *endpoint;
                          }) ||
            !llvm::any_of(inputs, [&](const auto &input) {
              return input.endpoint == *endpoint;
            }))
          return structuralUnsupported(
              "PE input selector domain is outside its sealed attachments");
        auto semantic = schema->encode(descriptor.reference, value);
        if (!semantic)
          return semantic.takeError();
        auto code = physicalCode(codebook, semantic->bytes());
        if (!code)
          return code.takeError();
        destination->push_back({*endpoint, std::move(*code)});
      }
      if (selector.routes.size() != inputs.size() ||
          selector.discards.size() != inputs.size())
        return structuralUnsupported(
            "PE input selector codebook does not cover its endpoint domain");
      inputSelectors[descriptor.port->ordinal] = std::move(selector);
      continue;
    }
    if (descriptor.kind !=
            fabric::FabricPeConfigurationFieldKind::OutputSelector ||
        descriptor.port->direction != fabric::FabricPortDirection::Output ||
        descriptor.port->ordinal >= outputSelectors.size() ||
        outputSelectors[descriptor.port->ordinal])
      return skeletonError("PE output selector field is not one-to-one");
    OutputSelectorPlan selector{std::move(decoder), {}, std::nullopt};
    for (const fabric::FabricPeConfigurationValue &value : *domain) {
      if (const auto *route = std::get_if<fabric::FabricPeRoute>(&value)) {
        if (!llvm::any_of(attachments,
                          [&](const auto &attachment) {
                            return attachment.endpoint == route->endpoint;
                          }) ||
            !llvm::any_of(outputs, [&](const auto &output) {
              return output.endpoint == route->endpoint;
            }))
          return structuralUnsupported(
              "PE output selector domain is outside sealed attachments");
        auto semantic = schema->encode(descriptor.reference, value);
        if (!semantic)
          return semantic.takeError();
        auto code = physicalCode(codebook, semantic->bytes());
        if (!code)
          return code.takeError();
        selector.routes.push_back({route->endpoint, std::move(*code)});
      } else if (std::holds_alternative<fabric::FabricPeOutputDiscard>(value)) {
        auto semantic = schema->encode(descriptor.reference, value);
        if (!semantic)
          return semantic.takeError();
        auto code = physicalCode(codebook, semantic->bytes());
        if (!code)
          return code.takeError();
        if (selector.discard)
          return skeletonError("PE output selector repeats Discard");
        selector.discard = std::move(*code);
      }
    }
    if (selector.routes.empty() || !selector.discard)
      return structuralUnsupported(
          "PE output selector codebook is outside the support envelope");
    outputSelectors[descriptor.port->ordinal] = std::move(selector);
  }
  if (!activation ||
      llvm::any_of(inputSelectors,
                   [](const auto &selector) { return !selector; }) ||
      llvm::any_of(outputSelectors,
                   [](const auto &selector) { return !selector; }))
    return skeletonError("operation PE configuration schema is incomplete");

  std::vector<fabric::FabricSemanticConfigFieldRef> configurationFields =
      operation.capability->configurationFieldSchema;
  llvm::sort(configurationFields, [](const auto &lhs, const auto &rhs) {
    return fabric::canonicalFabricBytes(lhs) <
           fabric::canonicalFabricBytes(rhs);
  });
  std::vector<OperationConfigurationPlan> operationConfiguration;
  operationConfiguration.reserve(configurationFields.size());
  for (const auto &field : configurationFields) {
    const ConfigurationFieldEncoding *encoding =
        configurationAbi.findOperationField(operation.physicalOccurrence,
                                            field.ordinal);
    if (!encoding)
      return skeletonError(
          "operation configuration field is absent from the ABI");
    auto decoder = prepareFieldDecoder(*encoding, configurationAbi);
    if (!decoder)
      return decoder.takeError();
    programmingUnits.emplace(decoder->unit->id, decoder->unit);
    operationConfiguration.push_back({field.ordinal, std::move(*decoder)});
  }
  auto clockReset =
      prepareClockReset(configurationAbi.fabricSystem(), spatialCore);
  if (!clockReset)
    return clockReset.takeError();
  std::vector<const ProgrammingUnit *> units;
  for (const auto &[id, unit] : programmingUnits) {
    (void)id;
    if (unit->payloadBitCount == 0 ||
        unit->payloadBitCount > mlir::IntegerType::kMaxWidth)
      return structuralUnsupported(
          "programming payload width exceeds the CIRCT support envelope");
    units.push_back(unit);
  }
  std::vector<InputSelectorPlan> completeInputSelectors;
  completeInputSelectors.reserve(inputSelectors.size());
  for (auto &selector : inputSelectors)
    completeInputSelectors.push_back(std::move(*selector));
  std::vector<OutputSelectorPlan> completeOutputSelectors;
  completeOutputSelectors.reserve(outputSelectors.size());
  for (auto &selector : outputSelectors)
    completeOutputSelectors.push_back(std::move(*selector));

  return InternalOperationPlan{std::move(operation),
                               *interface,
                               std::move(*stateLayout),
                               std::move(physicalInputs),
                               std::move(physicalOutputs),
                               std::move(inputs),
                               std::move(outputs),
                               std::move(*activation),
                               std::move(completeInputSelectors),
                               std::move(completeOutputSelectors),
                               std::move(operationConfiguration),
                               std::move(units),
                               *clockReset};
}

std::string configurationPortName(ProgrammingUnitId id) {
  return "configuration_" + std::to_string(id);
}

mlir::Value bitConstant(mlir::OpBuilder &builder, mlir::Location location,
                        bool value) {
  return circt::hw::ConstantOp::create(builder, location,
                                       llvm::APInt(1, value));
}

mlir::Value andValues(mlir::OpBuilder &builder, mlir::Location location,
                      llvm::ArrayRef<mlir::Value> values) {
  mlir::Value result = bitConstant(builder, location, true);
  for (mlir::Value value : values)
    result = circt::comb::AndOp::create(builder, location, result, value);
  return result;
}

mlir::Value orValues(mlir::OpBuilder &builder, mlir::Location location,
                     llvm::ArrayRef<mlir::Value> values) {
  mlir::Value result = bitConstant(builder, location, false);
  for (mlir::Value value : values)
    result = circt::comb::OrOp::create(builder, location, result, value);
  return result;
}

mlir::Value decodeFieldSignal(mlir::OpBuilder &builder, mlir::Location location,
                              circt::hw::HWModulePortAccessor &accessor,
                              const FieldDecoderPlan &decoder) {
  mlir::Value payload =
      accessor.getInput(configurationPortName(decoder.unit->id));
  llvm::SmallVector<mlir::Value> highToLow;
  highToLow.reserve(static_cast<std::size_t>(decoder.encodedBitCount));
  for (std::uint64_t source = decoder.encodedBitCount; source > 0; --source)
    highToLow.push_back(circt::comb::ExtractOp::create(
        builder, location, payload,
        decoder.destinationBits[static_cast<std::size_t>(source - 1)], 1));
  if (highToLow.size() == 1)
    return highToLow.front();
  return circt::comb::ConcatOp::create(builder, location, highToLow);
}

mlir::Value matchesCode(mlir::OpBuilder &builder, mlir::Location location,
                        mlir::Value field, const llvm::APInt &code) {
  mlir::Value constant = circt::hw::ConstantOp::create(builder, location, code);
  return circt::comb::ICmpOp::create(
      builder, location, circt::comb::ICmpPredicate::eq, field, constant, true);
}

const BoundaryChannelPlan &
channelFor(llvm::ArrayRef<BoundaryChannelPlan> channels,
           const fabric::FabricTransportEndpointRef &endpoint) {
  const auto found = llvm::find_if(channels, [&](const auto &channel) {
    return channel.endpoint == endpoint;
  });
  assert(found != channels.end());
  return *found;
}

mlir::Value resizeUnsignedSignal(mlir::OpBuilder &builder,
                                 mlir::Location location,
                                 std::optional<mlir::Value> source,
                                 unsigned sourceWidth,
                                 unsigned destinationWidth) {
  assert(destinationWidth != 0);
  if (sourceWidth == 0)
    return circt::hw::ConstantOp::create(builder, location,
                                         llvm::APInt(destinationWidth, 0));
  assert(source && *source);
  if (sourceWidth == destinationWidth)
    return *source;
  if (sourceWidth > destinationWidth)
    return circt::comb::ExtractOp::create(builder, location, *source, 0,
                                          destinationWidth);
  mlir::Value highZeros = circt::hw::ConstantOp::create(
      builder, location, llvm::APInt(destinationWidth - sourceWidth, 0));
  return circt::comb::ConcatOp::create(
      builder, location, llvm::SmallVector<mlir::Value, 2>{highZeros, *source});
}

mlir::Value createOperationRegister(mlir::OpBuilder &builder,
                                    mlir::Location location, mlir::Value next,
                                    mlir::Value clock, mlir::Value reset,
                                    const llvm::APInt &resetValue,
                                    llvm::StringRef name,
                                    bool asynchronousReset) {
  mlir::Value resetConstant =
      circt::hw::ConstantOp::create(builder, location, resetValue);
  if (asynchronousReset)
    return circt::seq::FirRegOp::create(
        builder, location, next, clock, builder.getStringAttr(name), reset,
        resetConstant, circt::hw::InnerSymAttr{}, true);
  return circt::seq::CompRegOp::create(builder, location, next, clock, reset,
                                       resetConstant, name);
}

llvm::Expected<ModuleRootCirctSkeleton> buildInternalOperationSkeleton(
    mlir::MLIRContext &context, fabric::SpatialCoreOccurrenceRef spatialCore,
    const ConfigurationABI &configurationAbi,
    const fabric::FabricArtifactView &module,
    llvm::ArrayRef<ModuleBoundaryTransportPortProjection> projections) {
  auto plan = prepareInternalOperation(spatialCore, configurationAbi, module,
                                       projections);
  if (!plan)
    return plan.takeError();
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  auto leafPorts = deriveFabricOperationLeafPorts(
      builder, plan->operation.physicalOccurrence, *plan->operation.capability,
      configurationAbi);
  if (!leafPorts)
    return leafPorts.takeError();

  mlir::OwningOpRef<mlir::ModuleOp> result = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(result->getBody());
  circt::hw::HWGeneratorSchemaOp::create(
      builder, location, fabricOperationGeneratorSchemaSymbol,
      fabricOperationGeneratorDescriptor, builder.getArrayAttr({}));
  circt::hw::HWModuleGeneratedOp leaf = circt::hw::HWModuleGeneratedOp::create(
      builder, location,
      mlir::FlatSymbolRefAttr::get(&context,
                                   fabricOperationGeneratorSchemaSymbol),
      builder.getStringAttr("loom_fabric_operation_0"), *leafPorts);

  llvm::SmallVector<circt::hw::PortInfo, 16> inputPorts;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputPorts;
  inputPorts.push_back(circt::hw::PortInfo{
      {builder.getStringAttr("clock"), circt::seq::ClockType::get(&context),
       circt::hw::ModulePort::Direction::Input}});
  inputPorts.push_back(
      circt::hw::PortInfo{{builder.getStringAttr("reset"), builder.getI1Type(),
                           circt::hw::ModulePort::Direction::Input}});
  for (const ProgrammingUnit *unit : plan->programmingUnits)
    inputPorts.push_back(circt::hw::PortInfo{
        {builder.getStringAttr(configurationPortName(unit->id)),
         builder.getIntegerType(static_cast<unsigned>(unit->payloadBitCount)),
         circt::hw::ModulePort::Direction::Input}});
  for (const ModuleBoundaryTransportPortProjection &projection : projections)
    appendBoundaryPorts(inputPorts, outputPorts, projection);

  std::optional<std::string> materializationError;
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("loom_module"),
      circt::hw::ModulePortInfo(inputPorts, outputPorts),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        mlir::Value reset = accessor.getInput("reset");
        if (plan->clockReset.activeLowReset)
          reset = circt::comb::createOrFoldNot(bodyBuilder, location, reset);
        mlir::Value activeField = decodeFieldSignal(
            bodyBuilder, location, accessor, plan->activation.decoder);
        mlir::Value active = matchesCode(bodyBuilder, location, activeField,
                                         plan->activation.activeCode);
        mlir::Value enabled = circt::comb::AndOp::create(
            bodyBuilder, location, active,
            circt::comb::createOrFoldNot(bodyBuilder, location, reset));

        struct InputRuntime final {
          mlir::Value data;
          mlir::Value valid;
          std::vector<std::pair<const CodeMatchPlan *, mlir::Value>> routes;
          std::vector<std::pair<const CodeMatchPlan *, mlir::Value>> discards;
        };
        std::vector<InputRuntime> inputRuntime;
        inputRuntime.reserve(plan->inputSelectors.size());
        for (auto [ordinal, selector] : llvm::enumerate(plan->inputSelectors)) {
          mlir::Value field = decodeFieldSignal(bodyBuilder, location, accessor,
                                                selector.decoder);
          const unsigned physicalWidth =
              plan->physicalInputs[ordinal]->payloadWidthBits;
          mlir::Value data;
          if (physicalWidth != 0)
            data = circt::hw::ConstantOp::create(bodyBuilder, location,
                                                 llvm::APInt(physicalWidth, 0));
          llvm::SmallVector<mlir::Value> selectedValids;
          std::vector<std::pair<const CodeMatchPlan *, mlir::Value>> routes;
          for (const CodeMatchPlan &route : selector.routes) {
            mlir::Value selected =
                matchesCode(bodyBuilder, location, field, route.code);
            const BoundaryChannelPlan &channel =
                channelFor(plan->inputs, route.endpoint);
            if (physicalWidth != 0) {
              std::optional<mlir::Value> boundaryData;
              if (channel.projection->data)
                boundaryData =
                    accessor.getInput(channel.projection->data->getName());
              mlir::Value adapted = resizeUnsignedSignal(
                  bodyBuilder, location, boundaryData,
                  channel.dataPath.payloadWidthBits, physicalWidth);
              data = circt::comb::MuxOp::create(bodyBuilder, location, selected,
                                                adapted, data, true);
            }
            selectedValids.push_back(circt::comb::AndOp::create(
                bodyBuilder, location, selected,
                accessor.getInput(channel.projection->valid.getName())));
            routes.emplace_back(&route, selected);
          }
          std::vector<std::pair<const CodeMatchPlan *, mlir::Value>> discards;
          for (const CodeMatchPlan &discard : selector.discards)
            discards.emplace_back(&discard, matchesCode(bodyBuilder, location,
                                                        field, discard.code));
          inputRuntime.push_back(
              {data, orValues(bodyBuilder, location, selectedValids),
               std::move(routes), std::move(discards)});
        }

        struct OutputRuntime final {
          mlir::Value ready;
          std::vector<std::pair<const CodeMatchPlan *, mlir::Value>> routes;
        };
        std::vector<OutputRuntime> outputRuntime;
        outputRuntime.reserve(plan->outputSelectors.size());
        for (const OutputSelectorPlan &selector : plan->outputSelectors) {
          mlir::Value field = decodeFieldSignal(bodyBuilder, location, accessor,
                                                selector.decoder);
          llvm::SmallVector<mlir::Value> readyTerms;
          std::vector<std::pair<const CodeMatchPlan *, mlir::Value>> routes;
          for (const CodeMatchPlan &route : selector.routes) {
            mlir::Value selected =
                matchesCode(bodyBuilder, location, field, route.code);
            const BoundaryChannelPlan &channel =
                channelFor(plan->outputs, route.endpoint);
            readyTerms.push_back(circt::comb::AndOp::create(
                bodyBuilder, location, selected,
                accessor.getInput(channel.projection->ready.getName())));
            routes.emplace_back(&route, selected);
          }
          readyTerms.push_back(
              matchesCode(bodyBuilder, location, field, *selector.discard));
          outputRuntime.push_back(
              {andValues(
                   bodyBuilder, location,
                   {enabled, orValues(bodyBuilder, location, readyTerms)}),
               std::move(routes)});
        }

        circt::BackedgeBuilder backedges(bodyBuilder, location);
        circt::Backedge stateNext;
        mlir::Value stateRegister;
        if (plan->stateLayout) {
          stateNext = backedges.get(
              bodyBuilder.getIntegerType(plan->stateLayout->encodedBitCount()));
          stateRegister = createOperationRegister(
              bodyBuilder, location, stateNext, accessor.getInput("clock"),
              reset, plan->stateLayout->resetValue(), "operation_state_reg",
              plan->clockReset.asynchronousReset);
        }

        std::vector<circt::Backedge> resultDataNext(
            plan->physicalOutputs.size());
        std::vector<mlir::Value> resultData(plan->physicalOutputs.size());
        std::vector<circt::Backedge> resultValidNext;
        std::vector<mlir::Value> resultValid;
        circt::Backedge tupleValidNext;
        mlir::Value tupleValid;
        const bool hasResultStorage =
            !plan->interface.hasDirectTokenPublication();
        if (hasResultStorage) {
          for (auto [ordinal, output] :
               llvm::enumerate(plan->physicalOutputs)) {
            if (output->payloadWidthBits == 0)
              continue;
            resultDataNext[ordinal] = backedges.get(
                bodyBuilder.getIntegerType(output->payloadWidthBits));
            resultData[ordinal] = createOperationRegister(
                bodyBuilder, location, resultDataNext[ordinal],
                accessor.getInput("clock"), reset,
                llvm::APInt(output->payloadWidthBits, 0),
                plan->physicalOutputs.size() == 1
                    ? "result_data_reg"
                    : "result_data_" + std::to_string(ordinal) + "_reg",
                plan->clockReset.asynchronousReset);
          }
          if (plan->interface.protocol ==
              FabricOperationLeafProtocol::Combinational) {
            tupleValidNext = backedges.get(bodyBuilder.getI1Type());
            tupleValid = createOperationRegister(
                bodyBuilder, location, tupleValidNext,
                accessor.getInput("clock"), reset, llvm::APInt(1, 0),
                "result_valid_reg", plan->clockReset.asynchronousReset);
            resultValid.assign(plan->physicalOutputs.size(), tupleValid);
          } else {
            resultValidNext.resize(plan->physicalOutputs.size());
            resultValid.resize(plan->physicalOutputs.size());
            for (std::size_t ordinal = 0;
                 ordinal < plan->physicalOutputs.size(); ++ordinal) {
              resultValidNext[ordinal] = backedges.get(bodyBuilder.getI1Type());
              resultValid[ordinal] = createOperationRegister(
                  bodyBuilder, location, resultValidNext[ordinal],
                  accessor.getInput("clock"), reset, llvm::APInt(1, 0),
                  "result_valid_" + std::to_string(ordinal) + "_reg",
                  plan->clockReset.asynchronousReset);
            }
          }
        }

        std::optional<AtomicResultTupleSignals> heldTuple;
        mlir::Value slotAvailable = bitConstant(bodyBuilder, location, true);
        if (hasResultStorage) {
          llvm::SmallVector<mlir::Value, 4> downstreamReady;
          for (const OutputRuntime &output : outputRuntime)
            downstreamReady.push_back(output.ready);
          auto tuple = deriveAtomicResultTupleSignals(
              bodyBuilder, location, resultValid, downstreamReady);
          if (!tuple) {
            materializationError = llvm::toString(tuple.takeError());
            backedges.abandon();
            return;
          }
          heldTuple = std::move(*tuple);
          slotAvailable = heldTuple->available;
        }

        mlir::Value leafTransitionEnabled = enabled;
        if (plan->interface.hasElasticResultStorage())
          leafTransitionEnabled = circt::comb::AndOp::create(
              bodyBuilder, location, enabled, slotAvailable);

        std::map<std::string, mlir::Value> leafInputs;
        for (auto [ordinal, input] : llvm::enumerate(plan->physicalInputs)) {
          if (input->payloadWidthBits != 0)
            leafInputs.emplace("data_input_" + std::to_string(ordinal),
                               inputRuntime[ordinal].data);
          if (plan->interface.hasTokenHandshake())
            leafInputs.emplace("valid_input_" + std::to_string(ordinal),
                               circt::comb::AndOp::create(
                                   bodyBuilder, location, leafTransitionEnabled,
                                   inputRuntime[ordinal].valid));
        }
        if (plan->interface.hasTokenHandshake())
          for (std::size_t ordinal = 0; ordinal < plan->physicalOutputs.size();
               ++ordinal) {
            mlir::Value ready = outputRuntime[ordinal].ready;
            if (plan->interface.hasElasticResultStorage())
              ready = leafTransitionEnabled;
            leafInputs.emplace("ready_output_" + std::to_string(ordinal),
                               ready);
          }
        if (plan->stateLayout)
          leafInputs.emplace("state_current", stateRegister);
        for (const OperationConfigurationPlan &configuration :
             plan->operationConfiguration)
          leafInputs.emplace("config_" + std::to_string(configuration.ordinal),
                             decodeFieldSignal(bodyBuilder, location, accessor,
                                               configuration.decoder));

        llvm::SmallVector<mlir::Value> leafOperands;
        for (const circt::hw::PortInfo &port : *leafPorts) {
          if (port.isOutput())
            continue;
          const auto found = leafInputs.find(port.getName().str());
          if (found == leafInputs.end()) {
            materializationError =
                "derived operation leaf input has no structural signal";
            backedges.abandon();
            return;
          }
          leafOperands.push_back(found->second);
        }
        circt::hw::InstanceOp instance = circt::hw::InstanceOp::create(
            bodyBuilder, location, leaf.getOperation(), "operation",
            leafOperands);

        std::map<std::string, mlir::Value> leafOutputs;
        unsigned resultOrdinal = 0;
        for (const circt::hw::PortInfo &port : *leafPorts)
          if (port.isOutput())
            leafOutputs.emplace(port.getName().str(),
                                instance.getResult(resultOrdinal++));

        if (plan->stateLayout) {
          mlir::Value write =
              andValues(bodyBuilder, location,
                        {leafTransitionEnabled, leafOutputs.at("state_write")});
          stateNext.setValue(circt::comb::MuxOp::create(
              bodyBuilder, location, write, leafOutputs.at("state_next"),
              stateRegister, true));
        }

        llvm::SmallVector<mlir::Value, 4> publishedValid;
        std::vector<mlir::Value> publishedData(plan->physicalOutputs.size());
        llvm::SmallVector<mlir::Value, 4> operationInputReady;
        if (plan->interface.protocol ==
            FabricOperationLeafProtocol::Combinational) {
          llvm::SmallVector<mlir::Value, 4> inputValids;
          for (const InputRuntime &input : inputRuntime)
            inputValids.push_back(input.valid);
          mlir::Value capacity = circt::comb::AndOp::create(
              bodyBuilder, location, enabled, slotAvailable);
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
          mlir::Value retain = circt::comb::AndOp::create(
              bodyBuilder, location, tupleValid,
              circt::comb::createOrFoldNot(bodyBuilder, location,
                                           heldTuple->released));
          tupleValidNext.setValue(
              circt::comb::OrOp::create(bodyBuilder, location, accept, retain));
          publishedValid = heldTuple->publishedValids;
          for (auto [ordinal, output] :
               llvm::enumerate(plan->physicalOutputs)) {
            if (output->payloadWidthBits == 0)
              continue;
            resultDataNext[ordinal].setValue(circt::comb::MuxOp::create(
                bodyBuilder, location, accept,
                leafOutputs.at("data_output_" + std::to_string(ordinal)),
                resultData[ordinal], true));
            publishedData[ordinal] = resultData[ordinal];
          }
        } else if (plan->interface.hasElasticResultStorage()) {
          llvm::SmallVector<mlir::Value, 4> producedValid;
          for (std::size_t ordinal = 0; ordinal < plan->physicalOutputs.size();
               ++ordinal)
            producedValid.push_back(
                leafOutputs.at("valid_output_" + std::to_string(ordinal)));
          mlir::Value capture =
              andValues(bodyBuilder, location,
                        {leafTransitionEnabled,
                         orValues(bodyBuilder, location, producedValid)});
          for (auto [ordinal, output] :
               llvm::enumerate(plan->physicalOutputs)) {
            mlir::Value retain = circt::comb::AndOp::create(
                bodyBuilder, location, resultValid[ordinal],
                circt::comb::createOrFoldNot(bodyBuilder, location,
                                             heldTuple->released));
            mlir::Value acquire = circt::comb::AndOp::create(
                bodyBuilder, location, capture, producedValid[ordinal]);
            resultValidNext[ordinal].setValue(circt::comb::OrOp::create(
                bodyBuilder, location, retain, acquire));
            if (output->payloadWidthBits != 0) {
              resultDataNext[ordinal].setValue(circt::comb::MuxOp::create(
                  bodyBuilder, location, capture,
                  leafOutputs.at("data_output_" + std::to_string(ordinal)),
                  resultData[ordinal], true));
              publishedData[ordinal] = resultData[ordinal];
            }
          }
          publishedValid = heldTuple->publishedValids;
          for (std::size_t ordinal = 0; ordinal < plan->physicalInputs.size();
               ++ordinal)
            operationInputReady.push_back(circt::comb::AndOp::create(
                bodyBuilder, location, leafTransitionEnabled,
                leafOutputs.at("ready_input_" + std::to_string(ordinal))));
        } else {
          for (auto [ordinal, output] :
               llvm::enumerate(plan->physicalOutputs)) {
            publishedValid.push_back(circt::comb::AndOp::create(
                bodyBuilder, location, enabled,
                leafOutputs.at("valid_output_" + std::to_string(ordinal))));
            if (output->payloadWidthBits != 0)
              publishedData[ordinal] =
                  leafOutputs.at("data_output_" + std::to_string(ordinal));
          }
          for (std::size_t ordinal = 0; ordinal < plan->physicalInputs.size();
               ++ordinal)
            operationInputReady.push_back(circt::comb::AndOp::create(
                bodyBuilder, location, enabled,
                leafOutputs.at("ready_input_" + std::to_string(ordinal))));
        }

        for (std::size_t endpointIndex = 0; endpointIndex < plan->inputs.size();
             ++endpointIndex) {
          const BoundaryChannelPlan &channel = plan->inputs[endpointIndex];
          llvm::SmallVector<mlir::Value> readyTerms;
          for (std::size_t selectorIndex = 0;
               selectorIndex < inputRuntime.size(); ++selectorIndex) {
            for (const auto &[route, selected] :
                 inputRuntime[selectorIndex].routes)
              if (route->endpoint == channel.endpoint)
                readyTerms.push_back(circt::comb::AndOp::create(
                    bodyBuilder, location, selected,
                    operationInputReady[selectorIndex]));
            for (const auto &[discard, selected] :
                 inputRuntime[selectorIndex].discards)
              if (discard->endpoint == channel.endpoint)
                readyTerms.push_back(circt::comb::AndOp::create(
                    bodyBuilder, location, enabled, selected));
          }
          accessor.setOutput(channel.projection->ready.getName(),
                             orValues(bodyBuilder, location, readyTerms));
        }
        for (const BoundaryChannelPlan &channel : plan->outputs) {
          llvm::SmallVector<mlir::Value> validTerms;
          mlir::Value data;
          if (channel.dataPath.payloadWidthBits != 0)
            data = circt::hw::ConstantOp::create(
                bodyBuilder, location,
                llvm::APInt(channel.dataPath.payloadWidthBits, 0));
          for (auto [ordinal, output] : llvm::enumerate(outputRuntime)) {
            for (const auto &[route, selected] : output.routes) {
              if (route->endpoint != channel.endpoint)
                continue;
              validTerms.push_back(
                  andValues(bodyBuilder, location,
                            {enabled, selected, publishedValid[ordinal]}));
              if (channel.dataPath.payloadWidthBits != 0) {
                mlir::Value adapted = resizeUnsignedSignal(
                    bodyBuilder, location,
                    std::optional<mlir::Value>{publishedData[ordinal]},
                    plan->physicalOutputs[ordinal]->payloadWidthBits,
                    channel.dataPath.payloadWidthBits);
                data = circt::comb::MuxOp::create(
                    bodyBuilder, location, selected, adapted, data, true);
              }
            }
          }
          accessor.setOutput(channel.projection->valid.getName(),
                             orValues(bodyBuilder, location, validTerms));
          if (channel.projection->data)
            accessor.setOutput(channel.projection->data->getName(), data);
        }
      });

  if (materializationError)
    return skeletonError(*materializationError);

  ModuleRootCirctSkeleton skeleton{
      std::move(result), {{leaf, plan->operation.physicalOccurrence}}};
  if (llvm::Error error = verifyCommonCirctSkeleton(
          *skeleton.module, configurationAbi, skeleton.operationLeaves))
    return std::move(error);
  return skeleton;
}

} // namespace

llvm::Expected<ModuleRootCirctSkeleton>
buildModuleRootCirctSkeleton(mlir::MLIRContext &context,
                             fabric::SpatialCoreOccurrenceRef spatialCore,
                             const ConfigurationABI &configurationAbi) {
  auto fabricModule = resolveFabricSpatialCoreModule(
      configurationAbi.fabricSystem(), spatialCore);
  if (!fabricModule)
    return fabricModule.takeError();
  const fabric::FabricArtifactView &fabric = *fabricModule;
  const auto root = fabric.moduleRootTemplate();
  if (!root)
    return skeletonError("Module skeleton construction requires a Module "
                         "root");

  mlir::OpBuilder builder(&context);
  auto projections = deriveModuleBoundaryTransportPorts(builder, fabric);
  if (!projections)
    return projections.takeError();
  if (!fabric.moduleBoundaryTransportAttachments().empty() ||
      !fabric.pointConnections().empty() || !fabric.peOccurrences().empty() ||
      !fabric.fuOccurrences().empty() || !fabric.memoryOccurrences().empty() ||
      !fabric.switchOccurrences().empty() ||
      !fabric.fifoOccurrences().empty() ||
      !fabric.boundaryOccurrences().empty())
    return buildInternalOperationSkeleton(
        context, spatialCore, configurationAbi, fabric, *projections);

  const std::uint64_t inputCount = fabric.moduleBoundaryEndpointCount(
      *root, fabric::FabricPortDirection::Input);
  const std::uint64_t outputCount = fabric.moduleBoundaryEndpointCount(
      *root, fabric::FabricPortDirection::Output);
  if (projections->size() != inputCount + outputCount)
    return skeletonError(
        "Module boundary constructor accepts no memory-plane boundary");

  std::vector<const ModuleBoundaryTransportPortProjection *> inputs(inputCount);
  std::vector<const ModuleBoundaryTransportPortProjection *> outputs(
      outputCount);
  for (const ModuleBoundaryTransportPortProjection &projection : *projections) {
    if (projection.boundary.module != *root)
      return skeletonError("Module boundary projection names another root");
    auto &index =
        projection.boundary.direction == fabric::FabricPortDirection::Input
            ? inputs
            : outputs;
    if (projection.boundary.ordinal >= index.size() ||
        index[projection.boundary.ordinal])
      return skeletonError("Module boundary projection is not one-to-one");
    index[projection.boundary.ordinal] = &projection;
  }

  std::vector<bool> usedInputs(inputCount, false);
  std::vector<bool> usedOutputs(outputCount, false);
  std::vector<BoundaryPassthroughPlan> passthroughs;
  passthroughs.reserve(fabric.moduleBoundaryTransportPassthroughs().size());
  for (const fabric::FabricModuleBoundaryTransportPassthroughView &passthrough :
       fabric.moduleBoundaryTransportPassthroughs()) {
    if (passthrough.input.module != *root ||
        passthrough.output.module != *root ||
        passthrough.input.direction != fabric::FabricPortDirection::Input ||
        passthrough.output.direction != fabric::FabricPortDirection::Output ||
        passthrough.input.ordinal >= inputs.size() ||
        passthrough.output.ordinal >= outputs.size() ||
        !inputs[passthrough.input.ordinal] ||
        !outputs[passthrough.output.ordinal] ||
        usedInputs[passthrough.input.ordinal] ||
        usedOutputs[passthrough.output.ordinal])
      return skeletonError("Module boundary passthrough is not one-to-one");
    const auto inputType =
        fabric.moduleBoundaryEndpointDataPath(passthrough.input);
    const auto outputType =
        fabric.moduleBoundaryEndpointDataPath(passthrough.output);
    if (!inputType || !outputType)
      return skeletonError("Module boundary passthrough has no token type");
    usedInputs[passthrough.input.ordinal] = true;
    usedOutputs[passthrough.output.ordinal] = true;
    passthroughs.push_back({inputs[passthrough.input.ordinal],
                            outputs[passthrough.output.ordinal], *inputType,
                            *outputType});
  }
  if (llvm::is_contained(usedInputs, false) ||
      llvm::is_contained(usedOutputs, false))
    return skeletonError("Module boundary-only construction requires every "
                         "token port to be connected");

  llvm::SmallVector<circt::hw::PortInfo, 16> inputPorts;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputPorts;
  for (const ModuleBoundaryTransportPortProjection &projection : *projections)
    appendBoundaryPorts(inputPorts, outputPorts, projection);

  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module->getBody());
  std::optional<std::string> materializationError;
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("loom_module"),
      circt::hw::ModulePortInfo(inputPorts, outputPorts),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        for (const BoundaryPassthroughPlan &passthrough : passthroughs) {
          if (materializationError)
            return;
          ForwardTransportSignals source{
              accessor.getInput(passthrough.input->valid.getName()),
              passthrough.input->data
                  ? std::optional<mlir::Value>{accessor.getInput(
                        passthrough.input->data->getName())}
                  : std::nullopt,
              passthrough.input->tag
                  ? std::optional<mlir::Value>{accessor.getInput(
                        passthrough.input->tag->getName())}
                  : std::nullopt};
          auto adapted = adaptForwardTransportSignals(
              bodyBuilder, location, passthrough.inputType,
              passthrough.outputType, std::move(source));
          if (!adapted) {
            materializationError = llvm::toString(adapted.takeError());
            return;
          }
          accessor.setOutput(passthrough.output->valid.getName(),
                             adapted->valid);
          if (passthrough.output->data)
            accessor.setOutput(passthrough.output->data->getName(),
                               *adapted->payload);
          if (passthrough.output->tag)
            accessor.setOutput(passthrough.output->tag->getName(),
                               *adapted->tag);
          accessor.setOutput(
              passthrough.input->ready.getName(),
              accessor.getInput(passthrough.output->ready.getName()));
        }
      });
  if (materializationError)
    return skeletonError(*materializationError);

  ModuleRootCirctSkeleton result{std::move(module), {}};
  if (llvm::Error error = verifyCommonCirctSkeleton(
          *result.module, configurationAbi, result.operationLeaves))
    return std::move(error);
  return result;
}

llvm::Error verifyCommonCirctSkeleton(
    mlir::ModuleOp module, const ConfigurationABI &configurationAbi,
    llvm::ArrayRef<FabricOperationLeafAssociation> operationLeaves) {
  if (mlir::failed(mlir::verify(module)))
    return skeletonError("common CIRCT module does not verify");
  if (llvm::Error error = verifyNoUnresolvedStructuralLowering(module))
    return error;

  std::set<mlir::Operation *> declaredLeaves;
  bool hasInvalidSchema = false;
  module.walk([&](circt::hw::HWModuleGeneratedOp leaf) {
    if (!isFabricOperationLeaf(leaf))
      return;
    auto schema =
        mlir::cast<circt::hw::HWGeneratorSchemaOp>(leaf.getGeneratorKindOp());
    hasInvalidSchema |=
        schema.getDescriptor() != fabricOperationGeneratorDescriptor;
    declaredLeaves.insert(leaf.getOperation());
  });
  if (hasInvalidSchema)
    return skeletonError("Loom Fabric operation schema has an unexpected "
                         "descriptor");

  std::set<mlir::Operation *> associatedLeaves;
  std::set<std::vector<std::uint8_t>> associatedOccurrences;
  std::optional<fabric::SpatialCoreOccurrenceRef> associatedSpatialCore;
  for (const FabricOperationLeafAssociation &association : operationLeaves) {
    circt::hw::HWModuleGeneratedOp leaf = association.module;
    if (!leaf || leaf->getParentOfType<mlir::ModuleOp>() != module ||
        !isFabricOperationLeaf(leaf))
      return skeletonError(
          "operation association does not name a Loom leaf in this module");
    if (!associatedLeaves.insert(leaf.getOperation()).second)
      return skeletonError("Loom Fabric operation leaf is associated more than "
                           "once");

    std::vector<std::uint8_t> occurrenceBytes =
        fabric::canonicalFabricBytes(association.occurrence);
    if (!associatedOccurrences.insert(std::move(occurrenceBytes)).second)
      return skeletonError("Fabric operation occurrence is associated more "
                           "than once");
    auto operation = resolveFabricPhysicalOperation(
        configurationAbi.fabricSystem(), association.occurrence);
    if (!operation) {
      llvm::consumeError(operation.takeError());
      return skeletonError(
          "association does not resolve to a concrete Fabric operation "
          "capability");
    }
    const auto &internal = std::get<fabric::SpatialCoreInternalOccurrenceRef>(
        association.occurrence.payload());
    if (associatedSpatialCore && *associatedSpatialCore != internal.spatialCore)
      return skeletonError(
          "one Module skeleton associates multiple SpatialCore occurrences");
    associatedSpatialCore = internal.spatialCore;
    if (llvm::Error error = verifyFabricOperationLeafPorts(
            leaf, association.occurrence, *operation->capability,
            configurationAbi))
      return error;
  }

  if (declaredLeaves != associatedLeaves)
    return skeletonError(
        "Loom Fabric operation leaf has no exact Fabric occurrence "
        "association");
  auto expectedOccurrences = expectedOperationOccurrences(
      configurationAbi.fabricSystem(), associatedSpatialCore);
  if (!expectedOccurrences)
    return expectedOccurrences.takeError();
  if (*expectedOccurrences != associatedOccurrences)
    return skeletonError(
        llvm::Twine("operation association set does not exactly cover Fabric "
                    "operation occurrences: expected ") +
        llvm::Twine(expectedOccurrences->size()) + ", received " +
        llvm::Twine(associatedOccurrences.size()));
  return llvm::Error::success();
}

llvm::Expected<std::string>
lowerAndExportSpecializedSystemVerilog(mlir::ModuleOp module) {
  if (llvm::Error error = verifySpecializedCirctModule(module))
    return std::move(error);

  circt::LowerSeqToSVOptions loweringOptions;
  loweringOptions.disableRegRandomization = true;
  mlir::PassManager pipeline(module.getContext());
  pipeline.addPass(circt::createLowerSeqToSVPass(loweringOptions));
  if (mlir::failed(pipeline.run(module)))
    return skeletonError("Seq-to-SV lowering failed");
  if (llvm::Error error = verifySpecializedCirctModule(module))
    return std::move(error);

  llvm::SmallString<1024> storage;
  llvm::raw_svector_ostream output(storage);
  if (mlir::failed(circt::exportVerilog(module, output)))
    return skeletonError("ExportVerilog rejected the specialized module");
  return output.str().str();
}

llvm::Error verifySpecializedCirctModule(mlir::ModuleOp module) {
  if (mlir::failed(mlir::verify(module)))
    return skeletonError("specialized CIRCT module does not verify");
  if (llvm::Error error = verifyNoUnresolvedStructuralLowering(module))
    return error;
  return verifyNoUnresolvedFabricOperationLeaves(module);
}

} // namespace loom::hardware::rtl

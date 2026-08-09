#include "Hardware/Configuration/ConfigurationABI.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricArtifactCodec.h"
#include "Fabric/Artifact/FabricSystemContracts.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/Identity/FabricPeConfiguration.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FabricPhysicalConfigurationFieldRef;
using namespace loom::hardware;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *instance = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *instance;
}

mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef test,
                                        llvm::StringRef source) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  require(test, static_cast<bool>(module), "could not parse Fabric fixture");
  return module;
}

::fabric::ModuleOp moduleRoot(llvm::StringRef test, mlir::ModuleOp module) {
  ::fabric::ModuleOp selected;
  for (::fabric::ModuleOp candidate : module.getOps<::fabric::ModuleOp>()) {
    require(test, !selected, "fixture has more than one Module root");
    selected = candidate;
  }
  require(test, static_cast<bool>(selected), "fixture has no Module root");
  return selected;
}

::fabric::SystemOp systemRoot(llvm::StringRef test, mlir::ModuleOp module) {
  ::fabric::SystemOp selected;
  for (::fabric::SystemOp candidate : module.getOps<::fabric::SystemOp>()) {
    require(test, !selected, "fixture has more than one System root");
    selected = candidate;
  }
  require(test, static_cast<bool>(selected), "fixture has no System root");
  return selected;
}

std::string denseI8Assembly(llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  std::string text;
  llvm::raw_string_ostream stream(text);
  mlir::DenseI8ArrayAttr::get(&context(), signedBytes).print(stream);
  return text;
}

::fabric::ResourceContract instructionContextContract(llvm::StringRef test) {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.states = {::fabric::ResourceStateDeclaration{
      ::fabric::StateKey(0),
      {::fabric::CapacityDimensionDeclaration{::fabric::CapacityDimensionKey(0),
                                              ::fabric::CapacityUnits(1),
                                              ::fabric::CapacityUnits(0)}}}};
  declaration.timingContracts = {::fabric::TimingContractDeclaration{
      ::fabric::TimingContractKey(0), {0, 1}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.usePatterns = {::fabric::UsePatternDeclaration{
      ::fabric::UsePatternKey(0),
      ::fabric::RequesterKey(0),
      ::fabric::EligibilityKey(0),
      ::fabric::EventKey(0),
      ::fabric::EventKey(1),
      std::nullopt,
      ::fabric::TimingContractKey(0),
      {::fabric::ClaimDeclaration{::fabric::ClaimKey(0), ::fabric::StateKey(0),
                                  ::fabric::CapacityDimensionKey(0),
                                  ::fabric::CapacityUnits(1)}},
      {::fabric::InternalTransactionDeclaration{{::fabric::ClaimKey(0)}}}}};
  return take(test, ::fabric::ResourceContract::create(declaration));
}

std::vector<std::uint8_t> instructionArchitecture(llvm::StringRef test) {
  loom::fabric::RiscVArchitectureDeclaration declaration;
  declaration.xlen = loom::fabric::RiscVXLen::X64;
  declaration.base = loom::fabric::RiscVBase::I;
  declaration.extensions = {loom::fabric::RiscVExtension::M,
                            loom::fabric::RiscVExtension::Zicsr};
  declaration.endianness = loom::fabric::InstructionEndianness::Little;
  declaration.physicalAddressWidthBits = 48;
  declaration.privilegeModes = {loom::fabric::PrivilegeMode::Machine};
  declaration.abiCapabilities = {loom::fabric::RiscVAbi::Lp64};
  declaration.memoryOrdering = loom::fabric::RiscVMemoryOrdering::Rvwmo;
  declaration.syncScopes = {loom::fabric::InstructionSyncScope::Hart};
  declaration.codeModels = {loom::fabric::RiscVCodeModel::MediumAny};
  declaration.relocationModels = {loom::fabric::RelocationModel::Static};
  declaration.runtimeServices = {
      loom::fabric::InstructionRuntimeService::ThreadDispatch,
      loom::fabric::InstructionRuntimeService::SpatialLaunch};
  auto contract =
      take(test, loom::fabric::InstructionCoreArchitecturalContract::create(
                     std::move(declaration)));
  return take(
      test, loom::fabric::encodeInstructionCoreArchitecturalContract(contract));
}

std::vector<std::uint8_t> instructionMicroarchitecture(llvm::StringRef test) {
  loom::fabric::InstructionCoreCommonDeclaration common{
      1,
      {{loom::fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1}},
      instructionContextContract(test)};
  loom::fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1,
                                                             1, 1, 2, 1};
  auto realization = take(
      test,
      loom::fabric::InstructionCoreMicroarchitecturalRealization::createInOrder(
          std::move(common), pipeline));
  return take(test,
              loom::fabric::encodeInstructionCoreMicroarchitecturalRealization(
                  realization));
}

void appendInstructionCore(
    llvm::raw_ostream &stream, llvm::StringRef test, llvm::StringRef operation,
    std::optional<std::uint64_t> entityId,
    const std::optional<loom::fabric::FabricImportedModuleTargetRef>
        &spatialCore = std::nullopt) {
  stream << operation
         << " architecture = " << denseI8Assembly(instructionArchitecture(test))
         << " microarchitecture = "
         << denseI8Assembly(instructionMicroarchitecture(test));
  if (spatialCore)
    stream << " spatial_core = "
           << denseI8Assembly(loom::fabric::encodeFabricImportedModuleTargetRef(
                  *spatialCore));
  if (entityId)
    stream << " {entity_id = #fabric.entity_id<" << *entityId << ">}";
  stream << '\n';
}

void appendSpatialAttachments(
    llvm::raw_ostream &stream, llvm::StringRef test,
    const loom::fabric::FabricArtifactView &module,
    const loom::fabric::FabricImportedModuleTargetRef &target,
    std::uint64_t accCoreId) {
  const auto owner = loom::fabric::FabricTransportEndpointOwnerRef::of(
      loom::fabric::SpatialCoreOccurrenceRef{
          loom::fabric::AccCoreOccurrenceRef(accCoreId)});
  const std::array<loom::fabric::FabricPortDirection, 2> directions = {
      loom::fabric::FabricPortDirection::Input,
      loom::fabric::FabricPortDirection::Output};
  for (loom::fabric::FabricPortDirection direction : directions) {
    const std::uint64_t count =
        module.moduleBoundaryEndpointCount(target.target, direction);
    for (std::uint64_t ordinal = 0; ordinal < count; ++ordinal) {
      const loom::fabric::FabricModuleBoundaryEndpointRef local{
          target.target, direction, ordinal};
      require(test,
              module.moduleBoundaryEndpointPlane(local) ==
                  loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::
                      Transport,
              "fixture has a non-transport Module boundary");
      const std::optional<loom::fabric::FabricOrdinal> occurrence =
          module.moduleBoundaryEndpointOccurrenceOrdinal(local);
      require(test, occurrence.has_value(),
              "fixture boundary has no occurrence endpoint");
      const loom::fabric::FabricImportedModuleBoundaryEndpointRef imported{
          target.dependencyOrdinal, local};
      const auto spatial = take(
          test,
          loom::fabric::FabricSpatialAttachmentEndpointRef::create(
              loom::fabric::FabricTransportEndpointRef{owner, *occurrence}));
      stream << "fabric.system.spatial_attachment module_endpoint = "
             << denseI8Assembly(
                    loom::fabric::encodeFabricImportedModuleBoundaryEndpointRef(
                        imported))
             << " spatial_endpoint = "
             << denseI8Assembly(
                    loom::fabric::encodeFabricSpatialAttachmentEndpointRef(
                        spatial))
             << '\n';
    }
  }
}

std::string
systemSource(llvm::StringRef test,
             const loom::fabric::FabricArtifactView &module,
             const loom::fabric::FabricImportedModuleTargetRef &target) {
  constexpr std::array<std::uint64_t, 2> accCoreIds = {17, 29};
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module { fabric.system @configured_system {\n";
  appendInstructionCore(stream, test, "fabric.system.host_core", std::nullopt);
  for (std::uint64_t id : accCoreIds)
    appendInstructionCore(stream, test, "fabric.system.acc_core", id, target);
  for (std::uint64_t id : accCoreIds)
    appendSpatialAttachments(stream, test, module, target, id);
  stream << "} }\n";
  return text;
}

struct FabricFixture final {
  loom::fabric::FinalizedFabricRoot module;
  loom::fabric::FinalizedFabricRoot system;
};

FabricFixture makeFabric(llvm::StringRef test, const ArtifactStore &store) {
  auto source = parse(test, R"mlir(
    module {
      fabric.module @configured(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>,
               %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %shuffled = fabric.op [@vector.shuffle] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorShuffle>,
               hw_params = {
                 integer_element_widths = [32 : i32],
                 float_element_formats = [],
                 max_operand_payload_bits = 32 : i32,
                 max_result_payload_bits = 32 : i32,
                 max_block_payload_bits = 32 : i32,
                 max_source_blocks = 2 : i32,
                 max_result_blocks = 1 : i32}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            %first = fabric.op [@arith.addi, @arith.subi] (%shuffled, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            %second = fabric.op [@arith.addi, @arith.subi] (%first, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %second : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir");

  std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  std::vector<std::int8_t> signedContract;
  signedContract.reserve(contract.size());
  for (std::uint8_t byte : contract)
    signedContract.push_back(static_cast<std::int8_t>(byte));
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       mlir::DenseI8ArrayAttr::get(&context(), signedContract));
  });

  loom::fabric::FinalizedFabricRoot module = take(
      test, loom::fabric::finalizeFabricRoot(moduleRoot(test, *source), store));
  const std::optional<loom::fabric::FabricModuleTemplateRef> moduleTemplate =
      module.view().moduleRootTemplate();
  require(test, moduleTemplate.has_value(),
          "finalized Module has no unique root template");
  const loom::fabric::FabricImportedModuleTargetRef target{0, *moduleTemplate};
  auto systemModule = parse(test, systemSource(test, module.view(), target));
  loom::fabric::FinalizedFabricRoot system = take(
      test, loom::fabric::finalizeFabricRoot(systemRoot(test, *systemModule),
                                             {module.reference()}, store));
  require(test,
          system.reference().schemaIdentity ==
                  loom::fabric::fabricArtifactSchema.identity &&
              system.reference().schemaVersion == loom::SchemaVersion{4, 0},
          "fixture is not a loom.fabric 4.0 artifact");
  return FabricFixture{std::move(module), std::move(system)};
}

loom::fabric::FabricInventoryOwnerRef
inventoryOwner(const loom::fabric::FabricModulePhysicalOwnerRef &owner) {
  return std::visit(
      [](const auto &value) -> loom::fabric::FabricInventoryOwnerRef {
        using Type = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Type, loom::fabric::LocalMemoryServiceRef>)
          return loom::fabric::FabricInventoryOwnerRef::of(value.underlying());
        else
          return loom::fabric::FabricInventoryOwnerRef::of(value);
      },
      owner.payload());
}

loom::fabric::FabricPhysicalOccurrenceOwnerRef
qualifyOwner(llvm::StringRef test,
             const loom::fabric::SpatialCoreOccurrenceRef &spatialCore,
             const loom::fabric::FabricModulePhysicalOwnerRef &owner) {
  auto target =
      take(test, loom::fabric::FabricModulePhysicalTargetRef::create(owner));
  return take(test, loom::fabric::FabricPhysicalOccurrenceOwnerRef::create(
                        loom::fabric::SpatialCoreInternalOccurrenceRef{
                            spatialCore, std::move(target)}));
}

FabricPhysicalConfigurationFieldRef
qualifyField(llvm::StringRef test,
             const loom::fabric::SpatialCoreOccurrenceRef &spatialCore,
             const loom::fabric::FabricSemanticConfigFieldRef &field) {
  auto target =
      take(test, loom::fabric::FabricModulePhysicalTargetRef::create(field));
  return take(test, FabricPhysicalConfigurationFieldRef::create(
                        loom::fabric::SpatialCoreInternalOccurrenceRef{
                            spatialCore, std::move(target)}));
}

struct SemanticFieldDomain final {
  loom::fabric::FabricPhysicalConfigurationSlotRef slot;
  std::vector<std::vector<std::uint8_t>> values;
  std::vector<std::uint8_t> inactive;
  std::optional<std::uint64_t> directBitCount;
};

void setBit(std::vector<std::uint8_t> &bytes, std::uint64_t index, bool value) {
  std::uint8_t &byte = bytes[static_cast<std::size_t>(index / 8)];
  const std::uint8_t mask = static_cast<std::uint8_t>(1U << (index % 8));
  byte = value ? static_cast<std::uint8_t>(byte | mask)
               : static_cast<std::uint8_t>(byte & ~mask);
}

void writePackedField(std::vector<std::uint8_t> &bytes, std::uint64_t offset,
                      std::uint64_t bitCount, std::uint64_t value) {
  for (std::uint64_t index = 0; index < bitCount; ++index)
    setBit(bytes, offset + index, ((value >> index) & 1U) != 0);
}

SemanticFieldDomain
fieldDomain(llvm::StringRef test,
            const loom::fabric::FabricArtifactView &module,
            const loom::fabric::SpatialCoreOccurrenceRef &spatialCore,
            const loom::fabric::FabricSemanticConfigFieldRef &local,
            loom::fabric::FabricConfigurationResidency residency) {
  const auto field = qualifyField(test, spatialCore, local);
  SemanticFieldDomain result{
      take(test, loom::fabric::qualifyFabricConfigurationSlot(
                     field, std::move(residency))),
      {},
      {},
      std::nullopt};
  const loom::fabric::FabricInventoryOwnerRef &owner = local.owner.catalog();
  if (owner.kind() == loom::fabric::FabricInventoryOwnerKind::FuOccurrence) {
    auto relation = take(test, module.semanticFieldRelation(local, context()));
    require(test,
            relation.kind() ==
                loom::fabric::FabricSemanticFieldRelationKind::Finite,
            "FU topology field is not finite");
    for (const loom::CanonicalSemanticBytes &value : relation.finiteDomain())
      result.values.emplace_back(value.bytes().begin(), value.bytes().end());
    require(test, !result.values.empty(), "FU topology domain is empty");
    result.inactive = result.values.front();
  } else if (owner.kind() ==
             loom::fabric::FabricInventoryOwnerKind::PeOccurrence) {
    auto schema =
        take(test,
             module.spatialPeConfigurationSchema(
                 std::get<loom::fabric::FabricPeOccurrenceRef>(owner.payload)));
    auto domain = take(test, schema.finiteDomain(local));
    for (const loom::fabric::FabricPeConfigurationValue &value : domain) {
      const loom::CanonicalSemanticBytes encoded =
          take(test, schema.encode(local, value));
      result.values.emplace_back(encoded.bytes().begin(),
                                 encoded.bytes().end());
    }
    const auto descriptor = std::find_if(
        schema.fields().begin(), schema.fields().end(),
        [&](const auto &field) { return field.reference == local; });
    require(test, descriptor != schema.fields().end(),
            "PE field is absent from its sealed schema");
    if (descriptor->kind ==
        loom::fabric::FabricPeConfigurationFieldKind::Activation) {
      const loom::CanonicalSemanticBytes disabled =
          take(test, schema.encode(local, loom::fabric::FabricPeDisabled{}));
      result.inactive.assign(disabled.bytes().begin(), disabled.bytes().end());
    } else {
      require(test, !result.values.empty(), "PE field domain is empty");
      result.inactive = result.values.front();
    }
  } else {
    require(test,
            owner.kind() ==
                loom::fabric::FabricInventoryOwnerKind::FuOccurrenceNode,
            "fixture exposes an unsupported configuration owner");
    const auto occurrence =
        std::get<loom::fabric::FabricFuOccurrenceNodeRef>(owner.payload);
    const loom::fabric::ResolvedFabricOpCapabilityView *capability =
        module.resolvedFabricOpCapability(occurrence);
    require(test, capability != nullptr,
            "operation field has no sealed capability");
    require(test,
            capability->configurationFieldSchema.size() == 1 &&
                capability->configurationFieldSchema.front().ordinal ==
                    local.ordinal,
            "operation field is not its capability's exact field");
    auto relation =
        take(test, capability->resolveSemanticFieldRelation(context()));
    if (relation.kind() ==
        ::fabric::FabricOpSemanticFieldRelationKind::Direct) {
      const auto *layout = relation.fixedVectorShuffleLayout();
      require(test, layout != nullptr && relation.directEncodedBitCount() != 0,
              "fixture Direct relation has no shuffle layout");
      result.directBitCount = relation.directEncodedBitCount();
      result.inactive.assign((*result.directBitCount + 7) / 8, 0);
      writePackedField(result.inactive, layout->blockWidthBitOffset,
                       layout->blockWidthBitCount, 31);
      if (llvm::Error error = relation.validateSemanticValue(result.inactive))
        fail(test, llvm::toString(std::move(error)));
      return result;
    }
    require(test,
            relation.kind() ==
                ::fabric::FabricOpSemanticFieldRelationKind::Finite,
            "fixture operation field has no semantic domain");
    for (const auto &point : relation.finiteBehaviorDomain()) {
      require(test, point.semanticConfiguration.has_value(),
              "finite operation point has no semantic carrier");
      result.values.emplace_back(point.semanticConfiguration->bytes().begin(),
                                 point.semanticConfiguration->bytes().end());
    }
    require(test, !result.values.empty(), "operation field domain is empty");
    result.inactive = result.values.front();
  }

  std::set<std::vector<std::uint8_t>> unique(result.values.begin(),
                                             result.values.end());
  require(test, unique.size() == result.values.size(),
          "sealed field domain contains duplicate carriers");
  require(test, unique.find(result.inactive) != unique.end(),
          "inactive carrier is outside its sealed domain");
  return result;
}

std::uint64_t encodedBitCount(std::size_t domainSize) {
  std::uint64_t bits = 2;
  while ((std::uint64_t(1) << bits) <= domainSize)
    ++bits;
  return bits;
}

std::vector<std::uint8_t> bitVector(std::uint64_t value,
                                    std::uint64_t bitCount) {
  std::vector<std::uint8_t> result((bitCount + 7) / 8, 0);
  for (std::uint64_t bit = 0; bit < bitCount; ++bit)
    if ((value & (std::uint64_t(1) << bit)) != 0)
      result[static_cast<std::size_t>(bit / 8)] |=
          static_cast<std::uint8_t>(1U << (bit % 8));
  return result;
}

ConfigurationFieldEncoding fieldEncoding(llvm::StringRef test,
                                         SemanticFieldDomain domain) {
  if (domain.directBitCount)
    return ConfigurationFieldEncoding{
        std::move(domain.slot),
        DirectBitsEncoding{*domain.directBitCount},
        {},
        std::move(domain.inactive)};
  const auto inactive =
      std::find(domain.values.begin(), domain.values.end(), domain.inactive);
  require(test, inactive != domain.values.end(),
          "inactive value is absent from its field domain");
  std::rotate(domain.values.begin(), inactive, std::next(inactive));
  const std::uint64_t bits = encodedBitCount(domain.values.size());
  std::vector<FiniteCodebookEntry> entries;
  entries.reserve(domain.values.size());
  for (const auto &[ordinal, value] : llvm::enumerate(domain.values))
    entries.push_back(FiniteCodebookEntry{value, bitVector(ordinal, bits)});
  return ConfigurationFieldEncoding{
      std::move(domain.slot),
      FiniteCodebookEncoding{bits, std::move(entries)},
      {},
      std::move(domain.inactive)};
}

ProgrammingUnitDraft
makeProgrammingUnit(llvm::StringRef test,
                    const loom::fabric::FabricArtifactView &module,
                    const loom::fabric::SpatialCoreOccurrenceRef &spatialCore) {
  std::vector<loom::fabric::FabricPhysicalOccurrenceOwnerRef> closure;
  std::vector<ConfigurationFieldEncoding> fields;
  bool sawPeField = false;
  bool sawOperationField = false;
  for (const loom::fabric::FabricModuleDomainMemberRef &member :
       module.moduleDomainMembers()) {
    if (member.kind() != loom::fabric::FabricModuleDomainMemberKind::Internal)
      continue;
    const auto &physicalOwner =
        std::get<loom::fabric::FabricModulePhysicalOwnerRef>(member.payload);
    const loom::fabric::FabricInventoryOwnerRef owner =
        inventoryOwner(physicalOwner);
    const std::uint64_t fieldCount = module.inventorySize(
        owner, loom::fabric::FabricInventoryKind::SemanticConfigField);
    if (fieldCount == 0)
      continue;
    closure.push_back(qualifyOwner(test, spatialCore, physicalOwner));
    for (std::uint64_t ordinal = 0; ordinal < fieldCount; ++ordinal) {
      const loom::fabric::FabricSemanticConfigFieldRef local{
          loom::fabric::FabricConfigurationOwnerRef(owner), ordinal};
      sawPeField |=
          owner.kind() == loom::fabric::FabricInventoryOwnerKind::PeOccurrence;
      sawOperationField |=
          owner.kind() ==
          loom::fabric::FabricInventoryOwnerKind::FuOccurrenceNode;
      const auto residencies =
          take(test, module.configurationResidencies(local));
      for (const auto &residency : residencies)
        fields.push_back(fieldEncoding(
            test, fieldDomain(test, module, spatialCore, local, residency)));
    }
  }
  require(test, sawPeField && sawOperationField,
          "fixture does not cover both sealed PE and operation domains");

  std::uint64_t totalFieldBits = 0;
  for (const ConfigurationFieldEncoding &field : fields)
    totalFieldBits += field.encodedBitCount();
  std::uint64_t globalSourceBit = 0;
  for (ConfigurationFieldEncoding &field : fields) {
    for (std::uint64_t sourceBit = 0; sourceBit < field.encodedBitCount();
         ++sourceBit) {
      field.destinationSlices.push_back(
          {sourceBit, totalFieldBits - 1 - globalSourceBit, 1});
      ++globalSourceBit;
    }
  }
  require(test, !closure.empty() && !fields.empty() && totalFieldBits != 0,
          "fixture produced an empty programming unit");
  return ProgrammingUnitDraft{std::move(closure), totalFieldBits + 1,
                              std::move(fields)};
}

void requireNoDirectConfigurationFields(
    llvm::StringRef test, const loom::fabric::FabricSystemRootView &system) {
  const auto requireEmpty = [&](loom::fabric::FabricInventoryOwnerRef owner) {
    require(
        test,
        system.artifact().inventorySize(
            owner, loom::fabric::FabricInventoryKind::SemanticConfigField) == 0,
        "fixture unexpectedly exposes a direct System configuration field");
  };
  for (loom::fabric::HostCoreOccurrenceRef host :
       system.artifact().hostCoreOccurrences())
    requireEmpty(loom::fabric::FabricInventoryOwnerRef::of(host));
  for (loom::fabric::AccCoreOccurrenceRef core :
       system.artifact().accCoreOccurrences()) {
    requireEmpty(loom::fabric::FabricInventoryOwnerRef::of(core));
    requireEmpty(loom::fabric::FabricInventoryOwnerRef::of(
        loom::fabric::InstructionCoreContextRef{core}));
    requireEmpty(loom::fabric::FabricInventoryOwnerRef::of(
        loom::fabric::SpatialCoreOccurrenceRef{core}));
  }
}

ConfigurationABIDraft makeDraft(const FabricFixture &fixture) {
  const llvm::StringRef test = __func__;
  loom::fabric::FabricSystemRootView system =
      take(test, loom::fabric::requireSystemRoot(fixture.system.view()));
  require(test, system.artifact().importedModules().size() == 1,
          "System fixture does not have one imported Module artifact");
  require(test, system.artifact().accCoreOccurrences().size() == 2,
          "System fixture does not have two SpatialCore occurrences");
  requireNoDirectConfigurationFields(test, system);

  const loom::fabric::FabricArtifactView &module =
      system.artifact().importedModules().front();
  std::vector<ProgrammingUnitDraft> units;
  for (loom::fabric::AccCoreOccurrenceRef core :
       system.artifact().accCoreOccurrences()) {
    const loom::fabric::SpatialCoreOccurrenceRef spatialCore{core};
    const std::optional<loom::fabric::FabricImportedModuleTargetRef> target =
        system.spatialCoreTarget(core);
    require(test,
            target.has_value() && target->dependencyOrdinal == 0 &&
                target->target == module.moduleRootTemplate(),
            "SpatialCore occurrence does not select the imported Module");
    units.push_back(makeProgrammingUnit(test, module, spatialCore));
  }
  require(test,
          units.front().exactFabricResourceClosure.front() !=
                  units.back().exactFabricResourceClosure.front() &&
              units.front().fields.front().slot !=
                  units.back().fields.front().slot,
          "occurrence qualification aliased two imported Module instances");
  return ConfigurationABIDraft{fixture.system.reference(), std::move(units)};
}

bool bit(llvm::ArrayRef<std::uint8_t> bytes, std::uint64_t index) {
  return ((bytes[static_cast<std::size_t>(index / 8)] >> (index % 8)) & 1U) !=
         0;
}

std::vector<std::uint8_t>
unusedPhysicalCode(llvm::StringRef test,
                   const FiniteCodebookEncoding &codebook) {
  const std::uint64_t limit = std::uint64_t(1) << codebook.encodedBitCount;
  for (std::uint64_t ordinal = 0; ordinal < limit; ++ordinal) {
    std::vector<std::uint8_t> candidate =
        bitVector(ordinal, codebook.encodedBitCount);
    if (std::none_of(codebook.entries.begin(), codebook.entries.end(),
                     [&](const FiniteCodebookEntry &entry) {
                       return entry.physicalCode == candidate;
                     }))
      return candidate;
  }
  fail(test, "finite codebook has no unused physical code");
}

std::vector<std::uint8_t>
outsideSemanticCarrier(llvm::StringRef test,
                       const FiniteCodebookEncoding &codebook) {
  require(test,
          !codebook.entries.empty() &&
              !codebook.entries.front().semanticValue.empty(),
          "finite codebook has no nonempty semantic carrier");
  std::vector<std::uint8_t> candidate = codebook.entries.front().semanticValue;
  for (std::size_t index = 0; index < candidate.size(); ++index) {
    const std::uint8_t original = candidate[index];
    for (unsigned value = 0; value <= 0xff; ++value) {
      candidate[index] = static_cast<std::uint8_t>(value);
      if (std::none_of(codebook.entries.begin(), codebook.entries.end(),
                       [&](const FiniteCodebookEntry &entry) {
                         return entry.semanticValue == candidate;
                       }))
        return candidate;
    }
    candidate[index] = original;
  }
  fail(test, "could not derive an out-of-domain semantic carrier");
}

bool isOperationField(const ConfigurationFieldEncoding &field) {
  const auto physicalField = loom::fabric::configurationField(field.slot);
  const auto &internal =
      std::get<loom::fabric::SpatialCoreInternalOccurrenceRef>(
          physicalField.payload());
  const auto &local = std::get<loom::fabric::FabricSemanticConfigFieldRef>(
      internal.target.payload());
  return local.owner.catalog().kind() ==
         loom::fabric::FabricInventoryOwnerKind::FuOccurrenceNode;
}

const FiniteCodebookEntry &
activeEntry(llvm::StringRef test, const ConfigurationFieldEncoding &field) {
  const auto &codebook =
      std::get<FiniteCodebookEncoding>(field.semanticEncoding);
  const auto selected =
      std::find_if(codebook.entries.begin(), codebook.entries.end(),
                   [&](const FiniteCodebookEntry &entry) {
                     return entry.semanticValue != field.inactiveValue;
                   });
  require(test, selected != codebook.entries.end(),
          "field domain has no non-inactive value");
  return *selected;
}

const ConfigurationFieldEncoding &
finiteField(llvm::StringRef test,
            llvm::ArrayRef<ConfigurationFieldEncoding> fields) {
  const auto field = llvm::find_if(fields, [](const auto &candidate) {
    return std::holds_alternative<FiniteCodebookEncoding>(
        candidate.semanticEncoding);
  });
  require(test, field != fields.end(), "fixture has no finite field");
  return *field;
}

const ConfigurationFieldEncoding &
finiteOperationField(llvm::StringRef test,
                     llvm::ArrayRef<ConfigurationFieldEncoding> fields) {
  const auto field = llvm::find_if(fields, [](const auto &candidate) {
    return isOperationField(candidate) &&
           std::holds_alternative<FiniteCodebookEncoding>(
               candidate.semanticEncoding);
  });
  require(test, field != fields.end(), "fixture has no finite operation field");
  return *field;
}

const SemanticConfigurationValue *
findValue(llvm::ArrayRef<SemanticConfigurationValue> values,
          const loom::fabric::FabricPhysicalConfigurationSlotRef &slot) {
  const auto found = std::find_if(values.begin(), values.end(),
                                  [&](const SemanticConfigurationValue &value) {
                                    return value.slot == slot;
                                  });
  return found == values.end() ? nullptr : &*found;
}

void requireDecodedValues(llvm::StringRef test,
                          llvm::ArrayRef<ConfigurationFieldEncoding> fields,
                          llvm::ArrayRef<SemanticConfigurationValue> selected,
                          llvm::ArrayRef<SemanticConfigurationValue> decoded) {
  require(test, decoded.size() == fields.size(),
          "decode did not return every programming-unit field");
  for (const ConfigurationFieldEncoding &field : fields) {
    const SemanticConfigurationValue *actual = findValue(decoded, field.slot);
    const SemanticConfigurationValue *selection =
        findValue(selected, field.slot);
    require(test,
            actual != nullptr &&
                actual->value ==
                    (selection ? selection->value : field.inactiveValue),
            "decode changed a selected or inactive semantic value");
  }
}

void canonicalArtifactAndBitRoundTrip(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);

  FinalizedConfigurationABI first =
      take(test, finalizeConfigurationABI(makeDraft(fabric), store));
  require(test, first.abi().programmingUnits().size() == 2,
          "ABI did not preserve two occurrence-owned programming units");

  const auto &unit = first.abi().programmingUnits().front();
  std::vector<SemanticConfigurationValue> selected;
  for (const ConfigurationFieldEncoding &field : unit.fields) {
    const auto *codebook =
        std::get_if<FiniteCodebookEncoding>(&field.semanticEncoding);
    selected.push_back({field.slot, codebook
                                        ? activeEntry(test, field).semanticValue
                                        : field.inactiveValue});
  }
  std::vector<std::uint8_t> payload =
      take(test, first.abi().encode(unit.id, selected));
  require(test,
          payload.size() == (unit.payloadBitCount + 7) / 8 &&
              std::any_of(payload.begin(), payload.end(),
                          [](std::uint8_t byte) { return byte != 0; }),
          "complete configuration image has the wrong shape or content");

  auto decoded = take(test, first.abi().decode(unit.id, payload));
  requireDecodedValues(test, unit.fields, selected, decoded);
  require(test, take(test, first.abi().encode(unit.id, decoded)) == payload,
          "decoded complete image did not re-encode identically");

  std::vector<SemanticConfigurationValue> partial{selected.front()};
  std::vector<std::uint8_t> partialPayload =
      take(test, first.abi().encode(unit.id, partial));
  auto partialDecoded = take(test, first.abi().decode(unit.id, partialPayload));
  requireDecodedValues(test, unit.fields, partial, partialDecoded);

  ConfigurationABIDraft reordered = makeDraft(fabric);
  std::reverse(reordered.programmingUnits.begin(),
               reordered.programmingUnits.end());
  for (ProgrammingUnitDraft &reorderedUnit : reordered.programmingUnits) {
    std::reverse(reorderedUnit.exactFabricResourceClosure.begin(),
                 reorderedUnit.exactFabricResourceClosure.end());
    std::reverse(reorderedUnit.fields.begin(), reorderedUnit.fields.end());
    for (ConfigurationFieldEncoding &field : reorderedUnit.fields) {
      std::reverse(field.destinationSlices.begin(),
                   field.destinationSlices.end());
      if (auto *codebook =
              std::get_if<FiniteCodebookEncoding>(&field.semanticEncoding))
        std::reverse(codebook->entries.begin(), codebook->entries.end());
    }
  }
  FinalizedConfigurationABI second =
      take(test, finalizeConfigurationABI(std::move(reordered), store));
  require(test,
          first.reference() == second.reference() &&
              first.canonicalBytes().bytes() == second.canonicalBytes().bytes(),
          "authoring order changed ConfigurationABI identity");

  FinalizedConfigurationABI imported =
      take(test, importConfigurationABI(first.reference(), store));
  require(test,
          imported.reference() == first.reference() &&
              imported.canonicalBytes().bytes() ==
                  first.canonicalBytes().bytes(),
          "strict import changed the canonical ABI");

  ConfigurationABIDraft changed = makeDraft(fabric);
  auto &changedFields = changed.programmingUnits.front().fields;
  auto changedFieldIt = llvm::find_if(changedFields, [](const auto &field) {
    return std::holds_alternative<FiniteCodebookEncoding>(
        field.semanticEncoding);
  });
  require(test, changedFieldIt != changedFields.end(),
          "changed draft has no finite field");
  auto &changedField = *changedFieldIt;
  auto &changedCodebook =
      std::get<FiniteCodebookEncoding>(changedField.semanticEncoding);
  const auto changedEntry = std::find_if(
      changedCodebook.entries.begin(), changedCodebook.entries.end(),
      [&](const FiniteCodebookEntry &entry) {
        return entry.semanticValue != changedField.inactiveValue;
      });
  require(test, changedEntry != changedCodebook.entries.end(),
          "changed field has no active entry");
  changedEntry->physicalCode = unusedPhysicalCode(test, changedCodebook);
  FinalizedConfigurationABI changedAbi =
      take(test, finalizeConfigurationABI(std::move(changed), store));
  require(test, changedAbi.reference() != first.reference(),
          "semantic codebook change did not change ABI identity");
}

void invalidSemanticDomainsAreRejected(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi =
      take(test, finalizeConfigurationABI(makeDraft(fabric), store));

  const ProgrammingUnit &unit = abi.abi().programmingUnits().front();
  const ConfigurationFieldEncoding &field =
      finiteOperationField(test, unit.fields);
  const auto &codebook =
      std::get<FiniteCodebookEncoding>(field.semanticEncoding);
  std::vector<std::uint8_t> outside = outsideSemanticCarrier(test, codebook);
  require(test, outside.size() == codebook.entries.front().semanticValue.size(),
          "out-of-domain carrier does not preserve the valid carrier width");
  expectError(test,
              abi.abi().encode(unit.id, {SemanticConfigurationValue{
                                            field.slot, std::move(outside)}}),
              "finite behavior domain");

  ConfigurationABIDraft foreignDomain = makeDraft(fabric);
  bool replaced = false;
  for (ProgrammingUnitDraft &draftUnit : foreignDomain.programmingUnits) {
    for (ConfigurationFieldEncoding &draftField : draftUnit.fields) {
      if (!isOperationField(draftField))
        continue;
      auto *draftCodebook =
          std::get_if<FiniteCodebookEncoding>(&draftField.semanticEncoding);
      if (!draftCodebook)
        continue;
      std::vector<std::uint8_t> foreign =
          outsideSemanticCarrier(test, *draftCodebook);
      const auto replacedEntry = std::find_if(
          draftCodebook->entries.begin(), draftCodebook->entries.end(),
          [&](const FiniteCodebookEntry &entry) {
            return entry.semanticValue != draftField.inactiveValue &&
                   entry.semanticValue.size() == foreign.size();
          });
      if (replacedEntry == draftCodebook->entries.end())
        continue;
      replacedEntry->semanticValue = std::move(foreign);
      replaced = true;
      break;
    }
    if (replaced)
      break;
  }
  require(test, replaced,
          "fixture has no replaceable finite operation-domain carrier");
  expectError(test, finalizeConfigurationABI(std::move(foreignDomain), store),
              "outside the finite behavior domain");

  ConfigurationABIDraft incompleteDomain = makeDraft(fabric);
  bool removed = false;
  for (ProgrammingUnitDraft &draftUnit : incompleteDomain.programmingUnits) {
    for (ConfigurationFieldEncoding &draftField : draftUnit.fields) {
      auto *draftCodebook =
          std::get_if<FiniteCodebookEncoding>(&draftField.semanticEncoding);
      if (!draftCodebook)
        continue;
      const auto extra = std::find_if(
          draftCodebook->entries.begin(), draftCodebook->entries.end(),
          [&](const FiniteCodebookEntry &entry) {
            return entry.semanticValue != draftField.inactiveValue;
          });
      if (extra == draftCodebook->entries.end())
        continue;
      draftCodebook->entries.erase(extra);
      removed = true;
      break;
    }
    if (removed)
      break;
  }
  require(test, removed, "fixture has no removable finite-domain value");
  expectError(test,
              finalizeConfigurationABI(std::move(incompleteDomain), store),
              "does not equal");

  const auto direct = llvm::find_if(unit.fields, [](const auto &candidate) {
    return std::holds_alternative<DirectBitsEncoding>(
        candidate.semanticEncoding);
  });
  require(test, direct != unit.fields.end(), "fixture has no DirectBits field");
  std::vector<std::uint8_t> outsideDirect(direct->inactiveValue.size(), 0);
  require(test, outsideDirect != direct->inactiveValue,
          "DirectBits invalid carrier aliases the valid inactive value");
  expectError(test,
              abi.abi().encode(unit.id, {SemanticConfigurationValue{
                                            direct->slot, outsideDirect}}),
              "outside its domain");

  std::vector<std::uint8_t> invalidDirectPayload =
      take(test, abi.abi().encode(unit.id, {}));
  for (const DestinationSlice &slice : direct->destinationSlices)
    for (std::uint64_t index = 0; index < slice.bitCount; ++index)
      setBit(invalidDirectPayload, slice.destinationBitOffset + index, false);
  expectError(test, abi.abi().decode(unit.id, invalidDirectPayload),
              "outside its domain");
}

void invalidImagesAndLayoutsAreRejected(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi =
      take(test, finalizeConfigurationABI(makeDraft(fabric), store));
  const ProgrammingUnit &unit = abi.abi().programmingUnits().front();

  std::vector<std::uint8_t> reserved =
      take(test, abi.abi().encode(unit.id, {}));
  setBit(reserved, unit.payloadBitCount - 1, true);
  expectError(test, abi.abi().decode(unit.id, reserved), "reserved bit");

  const ConfigurationFieldEncoding &field = finiteField(test, unit.fields);
  const auto &codebook =
      std::get<FiniteCodebookEncoding>(field.semanticEncoding);
  const std::vector<std::uint8_t> unused = unusedPhysicalCode(test, codebook);
  std::vector<std::uint8_t> invalidCode =
      take(test, abi.abi().encode(unit.id, {}));
  for (const DestinationSlice &slice : field.destinationSlices)
    for (std::uint64_t index = 0; index < slice.bitCount; ++index)
      setBit(invalidCode, slice.destinationBitOffset + index,
             bit(unused, slice.sourceBitOffset + index));
  expectError(test, abi.abi().decode(unit.id, invalidCode), "codebook");
  expectError(test, abi.abi().decode(abi.abi().programmingUnits().size(), {}),
              "programming unit");

  ConfigurationABIDraft overlap = makeDraft(fabric);
  auto &overlapFields = overlap.programmingUnits.front().fields;
  overlapFields.back().destinationSlices.front().destinationBitOffset =
      overlapFields.front().destinationSlices.front().destinationBitOffset;
  expectError(test, finalizeConfigurationABI(std::move(overlap), store),
              "destination bit");

  ConfigurationABIDraft incomplete = makeDraft(fabric);
  incomplete.programmingUnits.front()
      .fields.front()
      .destinationSlices.pop_back();
  expectError(test, finalizeConfigurationABI(std::move(incomplete), store),
              "source bit");

  ConfigurationABIDraft missingOwner = makeDraft(fabric);
  missingOwner.programmingUnits.front().exactFabricResourceClosure.pop_back();
  expectError(test, finalizeConfigurationABI(std::move(missingOwner), store),
              "resource closure");

  ConfigurationABIDraft missingField = makeDraft(fabric);
  missingField.programmingUnits.front().fields.pop_back();
  expectError(test, finalizeConfigurationABI(std::move(missingField), store),
              "cover every Fabric configuration slot");
}

void schemaAndRootBoundaryAreVersionedAtomically(
    const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  require(test,
          configurationAbiSchema.identity == "loom.configuration_abi" &&
              configurationAbiSchema.version == loom::SchemaVersion{3, 0},
          "ConfigurationABI schema is not loom.configuration_abi 3.0");

  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  ConfigurationABIDraft moduleRootDraft = makeDraft(fabric);
  moduleRootDraft.fabric = fabric.module.reference();
  expectError(test, finalizeConfigurationABI(std::move(moduleRootDraft), store),
              "System root");
}

void programmingUnitReferenceCodecIsExact(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FabricFixture fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi =
      take(test, finalizeConfigurationABI(makeDraft(fabric), store));

  const ProgrammingUnitRef endianReference{abi.reference(),
                                           UINT64_C(0x0102030405060708)};
  const std::vector<std::uint8_t> endianBytes =
      encodeProgrammingUnitRef(endianReference);
  require(test,
          endianBytes.size() >= 8 &&
              std::equal(endianBytes.end() - 8, endianBytes.end(),
                         std::array<std::uint8_t, 8>{0x01, 0x02, 0x03, 0x04,
                                                     0x05, 0x06, 0x07, 0x08}
                             .begin()),
          "ProgrammingUnitRef does not end in the exact u64be unit ID");
  expectError(test, decodeProgrammingUnitRef(endianBytes, store),
              "unknown programming unit");

  require(test, !abi.abi().programmingUnits().empty(),
          "ConfigurationABI fixture has no programming unit");
  const ProgrammingUnitRef reference{abi.reference(),
                                     abi.abi().programmingUnits().front().id};
  const std::vector<std::uint8_t> encoded = encodeProgrammingUnitRef(reference);
  require(test,
          take(test, decodeProgrammingUnitRef(encoded, store)) == reference,
          "ProgrammingUnitRef did not round-trip");

  const loom::ArtifactRootReference missingAbi{
      configurationAbiSchema.identity.str(), configurationAbiSchema.version,
      loom::finalizeArtifactIdentity(
          configurationAbiSchema,
          loom::CanonicalSemanticBytes(std::vector<std::uint8_t>{0xa5}))};
  expectError(
      test,
      decodeProgrammingUnitRef(encodeProgrammingUnitRef(ProgrammingUnitRef{
                                   missingAbi, reference.unitId}),
                               store),
      "stored object is missing");

  std::vector<std::uint8_t> truncated(encoded.begin(), encoded.end() - 1);
  expectError(test, decodeProgrammingUnitRef(truncated, store), "u64be");
  std::vector<std::uint8_t> extended = encoded;
  extended.push_back(0);
  expectError(test, decodeProgrammingUnitRef(extended, store), "u64be");

  const ProgrammingUnitRef wrongSchema{fabric.system.reference(), 0};
  expectError(
      test,
      decodeProgrammingUnitRef(encodeProgrammingUnitRef(wrongSchema), store),
      "loom.configuration_abi 3.0");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test-directory argument");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  canonicalArtifactAndBitRoundTrip(root / "canonical");
  invalidSemanticDomainsAreRejected(root / "semantic-domains");
  invalidImagesAndLayoutsAreRejected(root / "invalid");
  schemaAndRootBoundaryAreVersionedAtomically(root / "schema-root");
  programmingUnitReferenceCodecIsExact(root / "programming-unit-ref");
  return 0;
}

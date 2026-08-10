#include "ConfigurationABITestSupport.h"

#include "ADG/Builder.h"
#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/Identity/FabricPeConfiguration.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace loom::hardware::test {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

mlir::MLIRContext &relationContext() {
  static mlir::MLIRContext *context = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *context;
}

llvm::Expected<fabric::InstructionCoreMicroarchitecturalRealization>
makeInstructionCoreMicroarchitecture() {
  fabric::InstructionCoreCommonDeclaration common{
      1,
      {{fabric::InstructionOperationClass::IntegerAlu, 1, 1, 1}},
      ::fabric::oneCycleElasticOperationResourceContract()};
  fabric::InOrderMicroarchitectureDeclaration pipeline{1, 1, 1, 1, 1, 1, 2, 1};
  return fabric::InstructionCoreMicroarchitecturalRealization::createInOrder(
      std::move(common), pipeline);
}

fabric::FabricInventoryOwnerRef
inventoryOwner(const fabric::FabricModulePhysicalOwnerRef &owner) {
  return std::visit(
      [](const auto &value) -> fabric::FabricInventoryOwnerRef {
        using Type = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Type, fabric::LocalMemoryServiceRef>)
          return fabric::FabricInventoryOwnerRef::of(value.underlying());
        else
          return fabric::FabricInventoryOwnerRef::of(value);
      },
      owner.payload());
}

llvm::Expected<fabric::FabricPhysicalConfigurationFieldRef>
qualifyField(fabric::SpatialCoreOccurrenceRef spatialCore,
             const fabric::FabricSemanticConfigFieldRef &field) {
  auto target = fabric::FabricModulePhysicalTargetRef::create(field);
  if (!target)
    return target.takeError();
  return fabric::FabricPhysicalConfigurationFieldRef::create(
      fabric::SpatialCoreInternalOccurrenceRef{spatialCore,
                                               std::move(*target)});
}

} // namespace

llvm::Expected<fabric::FinalizedFabricRoot>
makeSpatialCoreSystem(const fabric::FinalizedFabricRoot &module,
                      const ArtifactStore &store,
                      std::uint64_t spatialCoreCount) {
  if (spatialCoreCount == 0)
    return invalid("test System requires at least one SpatialCore");
  adg::DesignBuilder design(store);
  auto system = design.createSystem("hardware-test-system");
  if (!system)
    return system.takeError();
  auto imported = system->importSpatialCore(module);
  if (!imported)
    return imported.takeError();
  auto architecture = adg::getBuiltinInstructionCoreArchitecture();
  if (!architecture)
    return architecture.takeError();
  auto microarchitecture = makeInstructionCoreMicroarchitecture();
  if (!microarchitecture)
    return microarchitecture.takeError();
  auto host = system->addHostCore(*architecture, *microarchitecture);
  if (!host)
    return host.takeError();
  std::vector<adg::HardwareDomainMember> domainMembers{host->domainMember()};
  for (std::uint64_t ordinal = 0; ordinal < spatialCoreCount; ++ordinal) {
    auto core =
        system->addAccCore(*architecture, *microarchitecture, *imported);
    if (!core)
      return core.takeError();
    domainMembers.push_back(core->instructionCoreDomainMember());
    domainMembers.push_back(core->spatialCoreDomainMember());
  }
  auto clock = system->createHardwareDomain();
  if (!clock)
    return clock.takeError();
  auto clockContract = fabric::ClockDomainContractRecord::create(1'000, 0);
  if (!clockContract)
    return clockContract.takeError();
  if (llvm::Error error =
          clock->close(domainMembers, std::move(*clockContract)))
    return std::move(error);
  auto reset = system->createHardwareDomain();
  if (!reset)
    return reset.takeError();
  auto resetContract = fabric::ResetDomainContractRecord::create(
      fabric::ResetPolarity::ActiveHigh, fabric::ResetTiming::Asynchronous,
      fabric::ResetTiming::Asynchronous, fabric::ResetInitialState::Asserted,
      std::nullopt, 0);
  if (!resetContract)
    return resetContract.takeError();
  if (llvm::Error error =
          reset->close(domainMembers, std::move(*resetContract)))
    return std::move(error);
  if (llvm::Error error = system->close())
    return std::move(error);
  auto finalized = std::move(design).finalize();
  if (!finalized)
    return finalized.takeError();
  if (finalized->roots().size() != 1)
    return invalid("test System did not finalize exactly one root");
  return fabric::importEntireFabricRoot(finalized->roots().front().reference(),
                                        store);
}

llvm::Expected<fabric::FinalizedFabricRoot>
makeSingleSpatialCoreSystem(const fabric::FinalizedFabricRoot &module,
                            const ArtifactStore &store) {
  return makeSpatialCoreSystem(module, store, 1);
}

llvm::Expected<fabric::FabricPhysicalConfigurationFieldRef>
qualifyPhysicalConfigurationField(
    const fabric::FabricPhysicalOccurrenceOwnerRef &owner,
    fabric::FabricOrdinal fieldOrdinal) {
  if (owner.kind() !=
      fabric::FabricPhysicalOccurrenceOwnerKind::SpatialCoreInternal)
    return invalid("configuration owner is not Module-internal");
  const auto &internal =
      std::get<fabric::SpatialCoreInternalOccurrenceRef>(owner.payload());
  if (internal.target.kind() != fabric::FabricModulePhysicalTargetKind::Owner)
    return invalid("configuration owner is not a physical owner target");
  const auto &localOwner =
      std::get<fabric::FabricModulePhysicalOwnerRef>(internal.target.payload());
  const fabric::FabricSemanticConfigFieldRef localField{
      fabric::FabricConfigurationOwnerRef(inventoryOwner(localOwner)),
      fieldOrdinal};
  return qualifyField(internal.spatialCore, localField);
}

llvm::Expected<ConfigurationABIDraft> makeCompleteConfigurationABIDraft(
    const fabric::FinalizedFabricRoot &systemRoot,
    llvm::ArrayRef<ConfigurationFieldEncodingOverride> overrides) {
  return derivePackedConfigurationABIDraft(systemRoot, relationContext(),
                                           overrides);
}

} // namespace loom::hardware::test

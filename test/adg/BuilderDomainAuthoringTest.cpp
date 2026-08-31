#include "ADGBuilderTestSupport.h"

#include "ADG/Builder.h"
#include "Fabric/Artifact/FabricModuleRootView.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ModuleDomain.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

namespace loom::adg::test {
namespace {

using loom::adg::BoundaryResult;
using loom::adg::DesignBuilder;
using loom::adg::FifoResult;
using loom::adg::FifoSpec;
using loom::adg::FinalizedFabricDesign;
using loom::adg::FuNode;
using loom::adg::FuSpec;
using loom::adg::LocalMemoryServiceSpec;
using loom::adg::MemoryConnectivitySpec;
using loom::adg::MemoryEngineSpec;
using loom::adg::MemoryResult;
using loom::adg::MemorySpec;
using loom::adg::ModuleDomainMemberHandle;
using loom::adg::ModuleDomainSlotHandle;
using loom::adg::ModuleInstanceDomainSlotBinding;
using loom::adg::PeSpec;
using loom::adg::PortType;
using loom::adg::SpatialCoreBuilder;
using loom::adg::SpatialValue;
using loom::adg::SwitchResult;

using ::fabric::ModuleDomainAuthoringRelation;
using loom::fabric::FabricClockResetKind;
using loom::fabric::FabricOrdinal;
using loom::fabric::FabricPortDirection;

// The two-argument Module instantiate overload is not retained: every call
// supplies the domain-binding authoring input explicitly.
template <typename Builder, typename = void>
struct HasTwoArgumentInstantiate : std::false_type {};
template <typename Builder>
struct HasTwoArgumentInstantiate<
    Builder, std::void_t<decltype(std::declval<Builder &>().instantiate(
                 std::declval<const Builder &>(),
                 std::declval<llvm::ArrayRef<SpatialValue>>()))>>
    : std::true_type {};

static_assert(!HasTwoArgumentInstantiate<loom::adg::SpatialCoreBuilder>::value,
              "current loom.fabric retains no two-argument instantiate "
              "overload");

static_assert(std::is_same<decltype(std::declval<const FifoResult &>().value()),
                           SpatialValue>::value,
              "a FIFO result exposes its connectivity value");
static_assert(
    std::is_same<decltype(std::declval<const FifoResult &>().domainMember()),
                 ModuleDomainMemberHandle>::value,
    "a FIFO result exposes its occurrence member");
static_assert(
    std::is_same<decltype(std::declval<const BoundaryResult &>().values()),
                 llvm::ArrayRef<SpatialValue>>::value,
    "a boundary result exposes its connectivity values");
static_assert(
    std::is_same<decltype(std::declval<const BoundaryResult &>()
                              .domainMember()),
                 ModuleDomainMemberHandle>::value,
    "a boundary result exposes its occurrence member");
static_assert(
    std::is_same<decltype(std::declval<const SwitchResult &>().values()),
                 llvm::ArrayRef<SpatialValue>>::value,
    "a switch result exposes its connectivity values");
static_assert(
    std::is_same<decltype(std::declval<const SwitchResult &>().domainMember()),
                 ModuleDomainMemberHandle>::value,
    "a switch result exposes its occurrence member");
static_assert(
    std::is_same<decltype(std::declval<const MemoryResult &>().values()),
                 llvm::ArrayRef<SpatialValue>>::value,
    "a memory result exposes its connectivity values");
static_assert(
    std::is_same<decltype(std::declval<const MemoryResult &>().domainMember()),
                 ModuleDomainMemberHandle>::value,
    "a memory result exposes its occurrence member");
static_assert(
    std::is_same<decltype(std::declval<const MemoryResult &>()
                              .operationPortMember(std::size_t{0})),
                 llvm::Expected<ModuleDomainMemberHandle>>::value,
    "a memory result exposes every operation port member");
static_assert(
    std::is_same<decltype(std::declval<const MemoryResult &>()
                              .localServiceMember()),
                 std::optional<ModuleDomainMemberHandle>>::value,
    "a memory result exposes its local service member when present");
static_assert(
    std::is_same<decltype(std::declval<const loom::adg::PeBuilder &>()
                              .domainMember()),
                 ModuleDomainMemberHandle>::value,
    "a PE result exposes its occurrence member");
static_assert(
    std::is_same<decltype(std::declval<const loom::adg::PeBuilder &>()
                              .instructionContextMember(std::size_t{0})),
                 llvm::Expected<ModuleDomainMemberHandle>>::value,
    "a PE result exposes every instruction context member");
static_assert(
    std::is_same<decltype(std::declval<const loom::adg::FuBuilder &>()
                              .domainMember()),
                 ModuleDomainMemberHandle>::value,
    "an FU result exposes its occurrence member");
static_assert(
    std::is_same<decltype(std::declval<const FuNode &>().domainMember()),
                 ModuleDomainMemberHandle>::value,
    "each FU-node result exposes its own occurrence node member");

static_assert(
    std::is_same<decltype(std::declval<loom::adg::SpatialCoreBuilder &>()
                              .declareDomainSlot(FabricClockResetKind::Clock)),
                 llvm::Expected<ModuleDomainSlotHandle>>::value,
    "declareDomainSlot authors one opaque slot handle");
static_assert(
    std::is_same<decltype(std::declval<const loom::adg::SpatialCoreBuilder &>()
                              .domainSlots(FabricClockResetKind::Clock)),
                 llvm::Expected<std::vector<ModuleDomainSlotHandle>>>::value,
    "domainSlots exposes the closed Module's effective slot handles");
static_assert(
    std::is_same<decltype(std::declval<const loom::adg::SpatialCoreBuilder &>()
                              .inputDomainMember(std::size_t{0})),
                 llvm::Expected<ModuleDomainMemberHandle>>::value,
    "inputDomainMember selects a boundary face directly");
static_assert(
    std::is_same<decltype(std::declval<const loom::adg::SpatialCoreBuilder &>()
                              .outputDomainMember(std::size_t{0})),
                 llvm::Expected<ModuleDomainMemberHandle>>::value,
    "outputDomainMember selects a boundary face directly");
static_assert(
    std::is_same<decltype(std::declval<loom::adg::SpatialCoreBuilder &>()
                              .assignDomainSlot(
                                  std::declval<const ModuleDomainMemberHandle
                                                   &>(),
                                  std::declval<const ModuleDomainSlotHandle
                                                   &>())),
                 llvm::Error>::value,
    "assignDomainSlot consumes the one unified owner-checked handle");
static_assert(
    std::is_same<
        decltype(std::declval<loom::adg::SpatialCoreBuilder &>().instantiate(
            std::declval<const loom::adg::SpatialCoreBuilder &>(),
            std::declval<llvm::ArrayRef<SpatialValue>>(),
            std::declval<llvm::ArrayRef<ModuleInstanceDomainSlotBinding>>())),
        llvm::Expected<std::vector<SpatialValue>>>::value,
    "Module instantiation takes explicit total domain-slot bindings");

void domainAuthoringRelationRejectsOutOfCatalogValues() {
  const llvm::StringRef test = __func__;
  ModuleDomainAuthoringRelation relation;
  expectError(test,
              relation.declareSlot(static_cast<FabricClockResetKind>(7)),
              "outside the catalog");

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::OwningOpRef<mlir::ModuleOp> owner =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OwningOpRef<mlir::ModuleOp> otherOwner =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  constexpr ModuleDomainAuthoringRelation::InternalMemberRole occurrence =
      ModuleDomainAuthoringRelation::InternalMemberRole::Occurrence;

  expectError(test, relation.noteInternalMember(nullptr, occurrence, 0),
              "no draft operation");
  expectError(test,
              relation.noteInternalMember(
                  owner->getOperation(),
                  static_cast<ModuleDomainAuthoringRelation::InternalMemberRole>(
                      99),
                  0),
              "outside the catalog");
  for (ModuleDomainAuthoringRelation::InternalMemberRole role :
       {ModuleDomainAuthoringRelation::InternalMemberRole::Occurrence,
        ModuleDomainAuthoringRelation::InternalMemberRole::FuNode,
        ModuleDomainAuthoringRelation::InternalMemberRole::LocalMemoryService})
    expectError(test,
                relation.noteInternalMember(owner->getOperation(), role, 1),
                "does not take a sub-ordinal");
  expectError(test,
              relation.assignInternal(owner->getOperation(), occurrence, 0,
                                      FabricClockResetKind::Clock, 0),
              "unregistered");
  if (llvm::Error error =
          relation.noteInternalMember(owner->getOperation(), occurrence, 0))
    fail(test, llvm::toString(std::move(error)));
  expectError(test,
              relation.noteInternalMember(owner->getOperation(), occurrence, 0),
              "already registered");
  // A distinct owner with the same role and sub-ordinal is a different
  // member.
  if (llvm::Error error = relation.noteInternalMember(
          otherOwner->getOperation(), occurrence, 0))
    fail(test, llvm::toString(std::move(error)));

  const FabricOrdinal clock =
      take(test, relation.declareSlot(FabricClockResetKind::Clock));
  const FabricOrdinal reset =
      take(test, relation.declareSlot(FabricClockResetKind::Reset));
  expectError(test,
              relation.assignBoundary(static_cast<FabricPortDirection>(9), 0,
                                      FabricClockResetKind::Clock, clock),
              "outside the catalog");
  expectError(test,
              relation.assignInternal(owner->getOperation(), occurrence, 0,
                                      static_cast<FabricClockResetKind>(7), 0),
              "outside the catalog");
  expectError(test,
              relation.assignInternal(owner->getOperation(), occurrence, 0,
                                      FabricClockResetKind::Clock, clock + 1),
              "out-of-range");
  if (llvm::Error error =
          relation.assignInternal(owner->getOperation(), occurrence, 0,
                                  FabricClockResetKind::Clock, clock))
    fail(test, llvm::toString(std::move(error)));
  expectError(test,
              relation.assignInternal(owner->getOperation(), occurrence, 0,
                                      FabricClockResetKind::Clock, clock),
              "already has an assignment");
  if (llvm::Error error =
          relation.assignInternal(owner->getOperation(), occurrence, 0,
                                  FabricClockResetKind::Reset, reset))
    fail(test, llvm::toString(std::move(error)));
}

void domainAuthoringRelationEnforcesTotality() {
  const llvm::StringRef test = __func__;
  constexpr ModuleDomainAuthoringRelation::InternalMemberRole occurrence =
      ModuleDomainAuthoringRelation::InternalMemberRole::Occurrence;
  const FabricOrdinal max = std::numeric_limits<FabricOrdinal>::max();

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  mlir::OwningOpRef<mlir::ModuleOp> owner =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));

  ModuleDomainAuthoringRelation relation;
  const FabricOrdinal clock =
      take(test, relation.declareSlot(FabricClockResetKind::Clock));
  const FabricOrdinal reset =
      take(test, relation.declareSlot(FabricClockResetKind::Reset));
  if (llvm::Error error =
          relation.noteInternalMember(owner->getOperation(), occurrence, 0))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = relation.assignBoundary(
          FabricPortDirection::Input, 0, FabricClockResetKind::Clock, clock))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = relation.assignBoundary(
          FabricPortDirection::Input, 0, FabricClockResetKind::Reset, reset))
    fail(test, llvm::toString(std::move(error)));
  expectError(test, relation.validateTotality(1, 0), "not total");
  if (llvm::Error error =
          relation.assignInternal(owner->getOperation(), occurrence, 0,
                                  FabricClockResetKind::Clock, clock))
    fail(test, llvm::toString(std::move(error)));
  // A fully assigned boundary face and an internal owner carrying only its
  // Clock row still miss the owner's Reset row: totality must reject.
  expectError(test, relation.validateTotality(1, 0), "not total");
  if (llvm::Error error =
          relation.assignInternal(owner->getOperation(), occurrence, 0,
                                  FabricClockResetKind::Reset, reset))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = relation.validateTotality(1, 0))
    fail(test, llvm::toString(std::move(error)));
  expectError(test, relation.validateTotality(max, 1), "overflows");

  ModuleDomainAuthoringRelation outOfSignature;
  const FabricOrdinal outClock =
      take(test, outOfSignature.declareSlot(FabricClockResetKind::Clock));
  const FabricOrdinal outReset =
      take(test, outOfSignature.declareSlot(FabricClockResetKind::Reset));
  // Two boundary members carry four rows, so cardinality passes and the
  // per-row sweep must reject the out-of-signature Output endpoint.
  if (llvm::Error error = outOfSignature.assignBoundary(
          FabricPortDirection::Input, 0, FabricClockResetKind::Clock, outClock))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = outOfSignature.assignBoundary(
          FabricPortDirection::Input, 0, FabricClockResetKind::Reset, outReset))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = outOfSignature.assignBoundary(
          FabricPortDirection::Output, 3, FabricClockResetKind::Clock,
          outClock))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = outOfSignature.assignBoundary(
          FabricPortDirection::Output, 3, FabricClockResetKind::Reset,
          outReset))
    fail(test, llvm::toString(std::move(error)));
  expectError(test, outOfSignature.validateTotality(1, 1),
              "outside the signature");
}

void instanceBindingRelationMustBeCanonicalAndTotal() {
  const llvm::StringRef test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  context.getOrLoadDialect<::fabric::FabricDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> container =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToStart(container->getBody());
  auto instance = ::fabric::InstantiateOp::create(
      builder, mlir::UnknownLoc::get(&context), mlir::TypeRange{},
      mlir::FlatSymbolRefAttr::get(&context, "child"), mlir::ValueRange{},
      llvm::ArrayRef<mlir::Type>{},
      ::fabric::encodeModuleInstanceDomainSlotBindings(
          &context, {{FabricClockResetKind::Clock, 0, 0}}));

  ModuleDomainAuthoringRelation child;
  take(test, child.declareSlot(FabricClockResetKind::Clock));
  take(test, child.declareSlot(FabricClockResetKind::Reset));
  ModuleDomainAuthoringRelation parent;
  take(test, parent.declareSlot(FabricClockResetKind::Clock));
  take(test, parent.declareSlot(FabricClockResetKind::Reset));

  expectError(test, parent.noteInstanceBindings(instance.getOperation(), child),
              "binding count");
  const ::fabric::ModuleInstanceDomainSlotBinding canonical[] = {
      {FabricClockResetKind::Clock, 0, 0}, {FabricClockResetKind::Reset, 0, 0}};
  instance.setDomainSlotBindingsAttr(
      ::fabric::encodeModuleInstanceDomainSlotBindings(&context, canonical));
  if (llvm::Error error =
          parent.noteInstanceBindings(instance.getOperation(), child))
    fail(test, llvm::toString(std::move(error)));
  expectError(test, parent.noteInstanceBindings(instance.getOperation(), child),
              "already recorded");
  instance.setDomainSlotBindingsAttr(
      ::fabric::encodeModuleInstanceDomainSlotBindings(&context, {}));
  mlir::IRMapping mapping;
  expectError(test, parent.composeInstance(instance.getOperation(), mapping),
              "empty");
}

void builderDomainAuthoringRejectsForeignAndStaleHandles() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(test, PortType::bits(32));

  auto first =
      take(test, design.createSpatialCore("first", {bits32}, {bits32}));
  auto second =
      take(test, design.createSpatialCore("second", {bits32}, {bits32}));

  const ModuleDomainSlotHandle foreignClock =
      take(test, second.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainMemberHandle firstInput =
      take(test, first.inputDomainMember(0));
  expectError(test, first.assignDomainSlot(firstInput, foreignClock),
              "foreign SpatialCore");
  const ModuleDomainSlotHandle firstClock =
      take(test, first.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainMemberHandle secondInput =
      take(test, second.inputDomainMember(0));
  expectError(test, first.assignDomainSlot(secondInput, firstClock),
              "foreign SpatialCore");
  expectError(test, first.inputDomainMember(1),
              "outside the Module signature");
  expectError(test, first.outputDomainMember(1),
              "outside the Module signature");
  expectError(test, first.declareDomainSlot(static_cast<FabricClockResetKind>(7)),
              "outside the catalog");
}

void builderDomainAuthoringRequiresTotalAssignment() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(test, PortType::bits(32));

  auto core = take(test, design.createSpatialCore("domain", {bits32}, {bits32}));
  const FifoResult queued =
      take(test, core.addFifo(take(test, core.input(0)),
                              FifoSpec{bits32, 2, false, std::nullopt}));
  const ModuleDomainSlotHandle clock =
      take(test, core.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainSlotHandle reset =
      take(test, core.declareDomainSlot(FabricClockResetKind::Reset));
  const ModuleDomainMemberHandle inputFace =
      take(test, core.inputDomainMember(0));
  const ModuleDomainMemberHandle outputFace =
      take(test, core.outputDomainMember(0));
  const ModuleDomainMemberHandle occurrence = queued.domainMember();

  for (const ModuleDomainMemberHandle *member :
       {&inputFace, &outputFace, &occurrence})
    if (llvm::Error error =
            core.assignDomainSlot(*member, clock))
      fail(test, llvm::toString(std::move(error)));
  expectError(test, core.assignDomainSlot(inputFace, clock),
              "already has an assignment");
  expectError(test, core.close({queued.value()}), "not total");

  for (const ModuleDomainMemberHandle *member :
       {&inputFace, &outputFace, &occurrence})
    if (llvm::Error error = core.assignDomainSlot(*member, reset))
      fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = core.close({queued.value()}))
    fail(test, llvm::toString(std::move(error)));
  expectError(test, core.assignDomainSlot(inputFace, reset),
              "already closed");
  FinalizedFabricDesign finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "complete Module domain relation did not finalize");
}

void moduleInstanceDomainSlotBindingsAreExplicit() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits16 = take(test, PortType::bits(16));

  auto pipeline =
      take(test, design.createSpatialCore("pipeline", {bits16}, {bits16}));
  const FifoResult stage =
      take(test, pipeline.addFifo(take(test, pipeline.input(0)),
                                  FifoSpec{bits16, 2, true, std::nullopt}));
  const ModuleDomainSlotHandle childClock =
      take(test, pipeline.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainSlotHandle childReset =
      take(test, pipeline.declareDomainSlot(FabricClockResetKind::Reset));
  for (const ModuleDomainMemberHandle &member :
       {take(test, pipeline.inputDomainMember(0)),
        take(test, pipeline.outputDomainMember(0)), stage.domainMember()}) {
    if (llvm::Error error = pipeline.assignDomainSlot(member, childClock))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = pipeline.assignDomainSlot(member, childReset))
      fail(test, llvm::toString(std::move(error)));
  }
  if (llvm::Error error = pipeline.close({stage.value()}))
    fail(test, llvm::toString(std::move(error)));

  auto top = take(test, design.createSpatialCore("top", {bits16}, {bits16}));
  const SpatialValue topInput = take(test, top.input(0));
  expectError(test, top.instantiate(pipeline, {topInput}, {}),
              "domain slot binding");

  const ModuleDomainSlotHandle parentClock =
      take(test, top.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainSlotHandle parentReset =
      take(test, top.declareDomainSlot(FabricClockResetKind::Reset));
  expectError(test,
              top.instantiate(pipeline, {topInput},
                              {{childClock, parentReset},
                               {childReset, parentReset}}),
              "kind");
  expectError(test,
              top.instantiate(pipeline, {topInput},
                              {{parentClock, parentClock},
                               {childReset, parentReset}}),
              "foreign");
  expectError(test,
              top.instantiate(pipeline, {topInput},
                              {{childClock, parentClock}}),
              "domain slot binding");
  expectError(test,
              top.instantiate(pipeline, {topInput},
                              {{childClock, parentClock},
                               {childClock, parentReset},
                               {childReset, parentReset}}),
              "domain slot binding");

  auto outputs = take(test,
                      top.instantiate(pipeline, {topInput},
                                      {{childClock, parentClock},
                                       {childReset, parentReset}}));
  require(test, outputs.size() == 1,
          "typed SpatialCore instance returned the wrong result count");
  for (const ModuleDomainMemberHandle &member :
       {take(test, top.inputDomainMember(0)),
        take(test, top.outputDomainMember(0))}) {
    if (llvm::Error error = top.assignDomainSlot(member, parentClock))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = top.assignDomainSlot(member, parentReset))
      fail(test, llvm::toString(std::move(error)));
  }
  if (llvm::Error error = top.close(outputs))
    fail(test, llvm::toString(std::move(error)));
  FinalizedFabricDesign finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "instantiated design did not publish both Module roots");
  auto topView =
      take(test, loom::fabric::requireModuleRoot(finalized.roots()[1].view()));
  require(test, topView.domainSlots().size() == 2,
          "instantiated Module changed the parent slot inventory");
  require(test, topView.domainAssignments().size() == 6,
          "instantiated Module did not compose child assignments");
  require(test, topView.artifact().moduleDomainMembers().size() == 3,
          "instantiated Module did not flatten the child physical member");
}

void constructionResultsExposeRoleTypedMembers() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits8 = take(test, PortType::bits(8));

  auto core = take(
      test, design.createSpatialCore("pe-members", {bits8, bits8}, {bits8}));
  auto pe = take(
      test, core.addPe({take(test, core.input(0)), take(test, core.input(1))},
                       PeSpec::spatial({bits8, bits8}, {bits8})));
  auto fu =
      take(test, pe.addFu({take(test, pe.input(0)), take(test, pe.input(1))},
                          FuSpec{{bits8, bits8}, {bits8}}));
  auto sum = take(
      test,
      fu.addOperation({take(test, fu.input(0)), take(test, fu.input(1))},
                      integerCapability(
                          ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                          ::dataflow::OperationSchemaId::ArithAddI, bits8)));
  if (llvm::Error error = fu.addCapabilityTemplate({{sum}, {}}))
    fail(test, llvm::toString(std::move(error)));

  const ModuleDomainMemberHandle peOccurrence = pe.domainMember();
  const ModuleDomainMemberHandle context =
      take(test, pe.instructionContextMember(0));
  expectError(test, pe.instructionContextMember(1), "instruction context");
  const ModuleDomainMemberHandle fuOccurrence = fu.domainMember();
  const ModuleDomainMemberHandle node = sum.domainMember();

  const ModuleDomainSlotHandle clock =
      take(test, core.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainSlotHandle reset =
      take(test, core.declareDomainSlot(FabricClockResetKind::Reset));
  const std::vector<ModuleDomainMemberHandle> members = {
      take(test, core.inputDomainMember(0)),
      take(test, core.inputDomainMember(1)),
      take(test, core.outputDomainMember(0)),
      peOccurrence,
      context,
      fuOccurrence,
      node,
  };
  for (const ModuleDomainMemberHandle &member : members) {
    if (llvm::Error error = core.assignDomainSlot(member, clock))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = core.assignDomainSlot(member, reset))
      fail(test, llvm::toString(std::move(error)));
  }

  if (llvm::Error error = fu.close({take(test, sum.output(0))}))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = pe.close())
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = core.close({take(test, pe.output(0))}))
    fail(test, llvm::toString(std::move(error)));

  FinalizedFabricDesign finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "domain-authored design did not publish one Module root");
  loom::fabric::FinalizedFabricRoot imported =
      take(test, loom::fabric::importEntireFabricRoot(
                     finalized.roots()[0].reference(), store));
  auto view = take(test, loom::fabric::requireModuleRoot(imported.view()));
  require(test,
          view.artifact().rootKind() == loom::fabric::FabricRootKind::Module,
          "domain-authored root changed its kind");
  require(test, view.domainSlots().size() == 2,
          "finalized Module did not preserve its exact domain-slot inventory");
  require(test, view.domainAssignments().size() == 14,
          "finalized Module did not preserve its complete assignment relation");
  require(test, view.artifact().moduleDomainMembers().size() == 7,
          "finalized Module changed its domain-member inventory");

  bool sawFuNode = false;
  for (const loom::fabric::FabricModuleDomainMemberRef &member :
       view.artifact().moduleDomainMembers()) {
    unsigned clocks = 0;
    unsigned resets = 0;
    for (const loom::fabric::ModuleDomainAssignment &assignment :
         view.domainAssignments()) {
      if (assignment.member != member)
        continue;
      clocks += assignment.slot.kind == FabricClockResetKind::Clock;
      resets += assignment.slot.kind == FabricClockResetKind::Reset;
    }
    require(test, clocks == 1 && resets == 1,
            "finalized Module assignment relation is not total by kind");
    if (member.kind() == loom::fabric::FabricModuleDomainMemberKind::Internal) {
      const auto &owner =
          std::get<loom::fabric::FabricModulePhysicalOwnerRef>(member.payload);
      sawFuNode |=
          owner.kind() ==
          loom::fabric::FabricModulePhysicalOwnerKind::FuOccurrenceNode;
    }
  }
  require(test, sawFuNode,
          "finalized Module lost the FU occurrence-node domain member");
}

void omittedModuleDomainMatchesExplicitSingleDomain() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits16 = take(test, PortType::bits(16));

  auto omitted =
      take(test, design.createSpatialCore("omitted", {bits16}, {bits16}));
  if (llvm::Error error = omitted.close({take(test, omitted.input(0))}))
    fail(test, llvm::toString(std::move(error)));

  auto explicitDomain =
      take(test, design.createSpatialCore("explicit", {bits16}, {bits16}));
  const ModuleDomainSlotHandle clock =
      take(test, explicitDomain.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainSlotHandle reset =
      take(test, explicitDomain.declareDomainSlot(FabricClockResetKind::Reset));
  for (const ModuleDomainMemberHandle &member :
       {take(test, explicitDomain.inputDomainMember(0)),
        take(test, explicitDomain.outputDomainMember(0))}) {
    if (llvm::Error error = explicitDomain.assignDomainSlot(member, clock))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = explicitDomain.assignDomainSlot(member, reset))
      fail(test, llvm::toString(std::move(error)));
  }
  if (llvm::Error error =
          explicitDomain.close({take(test, explicitDomain.input(0))}))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "single-domain design did not publish both Module roots");
  require(test,
          finalized.roots()[0].reference() == finalized.roots()[1].reference(),
          "omitted and explicit single-domain Modules changed identity");
}

void defaultChildRequiresExplicitInstanceBindings() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits16 = take(test, PortType::bits(16));

  auto pipeline =
      take(test, design.createSpatialCore("pipeline", {bits16}, {bits16}));
  expectError(test, pipeline.domainSlots(FabricClockResetKind::Clock),
              "must be closed");
  const FifoResult stage =
      take(test, pipeline.addFifo(take(test, pipeline.input(0)),
                                  FifoSpec{bits16, 2, true, std::nullopt}));
  if (llvm::Error error = pipeline.close({stage.value()}))
    fail(test, llvm::toString(std::move(error)));
  const std::vector<ModuleDomainSlotHandle> childClocks =
      take(test, pipeline.domainSlots(FabricClockResetKind::Clock));
  const std::vector<ModuleDomainSlotHandle> childResets =
      take(test, pipeline.domainSlots(FabricClockResetKind::Reset));
  require(test, childClocks.size() == 1 && childResets.size() == 1,
          "default child did not expose one effective slot of each kind");
  expectError(test, pipeline.domainSlots(static_cast<FabricClockResetKind>(7)),
              "outside the catalog");

  auto top = take(test, design.createSpatialCore("top", {bits16}, {bits16}));
  const SpatialValue input = take(test, top.input(0));
  const ModuleDomainSlotHandle parentClock =
      take(test, top.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainSlotHandle parentReset =
      take(test, top.declareDomainSlot(FabricClockResetKind::Reset));
  expectError(test, top.instantiate(pipeline, {input}, {}),
              "domain slot binding");
  expectError(
      test,
      top.instantiate(pipeline, {input}, {{childClocks.front(), parentClock}}),
      "domain slot binding");
  auto outputs =
      take(test, top.instantiate(pipeline, {input},
                                 {{childClocks.front(), parentClock},
                                  {childResets.front(), parentReset}}));
  for (const ModuleDomainMemberHandle &member :
       {take(test, top.inputDomainMember(0)),
        take(test, top.outputDomainMember(0))}) {
    if (llvm::Error error = top.assignDomainSlot(member, parentClock))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = top.assignDomainSlot(member, parentReset))
      fail(test, llvm::toString(std::move(error)));
  }
  if (llvm::Error error = top.close(outputs))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "explicitly bound design did not publish both Module roots");
  auto pipelineView =
      take(test, loom::fabric::requireModuleRoot(finalized.roots()[0].view()));
  auto topView =
      take(test, loom::fabric::requireModuleRoot(finalized.roots()[1].view()));
  require(test,
          pipelineView.domainSlots().size() == 2 &&
              pipelineView.domainAssignments().size() == 6,
          "default child Module domain relation is absent or incomplete");
  require(test,
          topView.domainSlots().size() == 2 &&
              topView.domainAssignments().size() == 6,
          "explicit parent Module domain relation is absent or incomplete");
}

void failedInstantiationPreservesParentDomainAuthoring() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits8 = take(test, PortType::bits(8));

  auto child = take(test, design.createSpatialCore("child", {bits8}, {bits8}));
  if (llvm::Error error = child.close({take(test, child.input(0))}))
    fail(test, llvm::toString(std::move(error)));

  auto parent =
      take(test, design.createSpatialCore("parent", {bits8}, {bits8}));
  expectError(test, parent.instantiate(child, {}, {}), "input count");

  const ModuleDomainSlotHandle clock =
      take(test, parent.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainSlotHandle reset =
      take(test, parent.declareDomainSlot(FabricClockResetKind::Reset));
  const ModuleDomainMemberHandle members[] = {
      take(test, parent.inputDomainMember(0)),
      take(test, parent.outputDomainMember(0))};
  for (const ModuleDomainMemberHandle &member : members) {
    if (llvm::Error error = parent.assignDomainSlot(member, clock))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = parent.assignDomainSlot(member, reset))
      fail(test, llvm::toString(std::move(error)));
  }
  if (llvm::Error error = parent.close({take(test, parent.input(0))}))
    fail(test, llvm::toString(std::move(error)));
  take(test, std::move(design).finalize());
}

void moduleConnectionsRemainWithinOneSymbolicDomain() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits8 = take(test, PortType::bits(8));

  auto core =
      take(test, design.createSpatialCore("crossing", {bits8}, {bits8}));
  const ModuleDomainSlotHandle firstClock =
      take(test, core.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainSlotHandle secondClock =
      take(test, core.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainSlotHandle reset =
      take(test, core.declareDomainSlot(FabricClockResetKind::Reset));
  const ModuleDomainMemberHandle input = take(test, core.inputDomainMember(0));
  const ModuleDomainMemberHandle output =
      take(test, core.outputDomainMember(0));
  if (llvm::Error error = core.assignDomainSlot(input, firstClock))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = core.assignDomainSlot(output, secondClock))
    fail(test, llvm::toString(std::move(error)));
  for (const ModuleDomainMemberHandle *member : {&input, &output})
    if (llvm::Error error = core.assignDomainSlot(*member, reset))
      fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = core.close({take(test, core.input(0))}))
    fail(test, llvm::toString(std::move(error)));

  expectError(test, std::move(design).finalize(),
              "crosses symbolic Clock or Reset slots");
}

void nestedConnectionsRemainWithinOneSymbolicDomain() {
  const llvm::StringRef test = __func__;
  enum class Crossing { None, PeToFu, FuToNode };
  const auto build = [&](Crossing crossing) {
    TemporaryDirectory directory(test);
    loom::ArtifactStore store(directory.path());
    DesignBuilder design(store);
    const PortType bits8 = take(test, PortType::bits(8));

    auto core =
        take(test, design.createSpatialCore("nested-domain", {bits8}, {bits8}));
    auto pe = take(test, core.addPe({take(test, core.input(0))},
                                    PeSpec::spatial({bits8}, {bits8})));
    auto fu = take(
        test, pe.addFu({take(test, pe.input(0))}, FuSpec{{bits8}, {bits8}}));
    auto operation = take(
        test, fu.addOperation(
                  {take(test, fu.input(0)), take(test, fu.input(0))},
                  integerCapability(
                      ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                      ::dataflow::OperationSchemaId::ArithAddI, bits8)));
    if (llvm::Error error = fu.addCapabilityTemplate({{operation}, {}}))
      fail(test, llvm::toString(std::move(error)));

    const ModuleDomainSlotHandle firstClock =
        take(test, core.declareDomainSlot(FabricClockResetKind::Clock));
    const ModuleDomainSlotHandle secondClock =
        crossing == Crossing::None
            ? firstClock
            : take(test, core.declareDomainSlot(FabricClockResetKind::Clock));
    const ModuleDomainSlotHandle reset =
        take(test, core.declareDomainSlot(FabricClockResetKind::Reset));
    const std::vector<ModuleDomainMemberHandle> outerMembers = {
        take(test, core.inputDomainMember(0)),
        take(test, core.outputDomainMember(0)), pe.domainMember(),
        take(test, pe.instructionContextMember(0))};
    for (const ModuleDomainMemberHandle &member : outerMembers) {
      if (llvm::Error error = core.assignDomainSlot(member, firstClock))
        fail(test, llvm::toString(std::move(error)));
      if (llvm::Error error = core.assignDomainSlot(member, reset))
        fail(test, llvm::toString(std::move(error)));
    }
    const ModuleDomainSlotHandle &fuClock =
        crossing == Crossing::PeToFu ? secondClock : firstClock;
    const ModuleDomainSlotHandle &nodeClock =
        crossing == Crossing::FuToNode ? secondClock : fuClock;
    for (const auto &[member, clock] : std::initializer_list<
             std::pair<ModuleDomainMemberHandle, ModuleDomainSlotHandle>>{
             {fu.domainMember(), fuClock},
             {operation.domainMember(), nodeClock}}) {
      if (llvm::Error error = core.assignDomainSlot(member, clock))
        fail(test, llvm::toString(std::move(error)));
      if (llvm::Error error = core.assignDomainSlot(member, reset))
        fail(test, llvm::toString(std::move(error)));
    }

    if (llvm::Error error = fu.close({take(test, operation.output(0))}))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = pe.close())
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = core.close({take(test, pe.output(0))}))
      fail(test, llvm::toString(std::move(error)));
    return std::move(design).finalize();
  };

  auto sameDomain = build(Crossing::None);
  if (!sameDomain)
    fail(test, llvm::toString(sameDomain.takeError()));
  expectError(test, build(Crossing::PeToFu),
              "crosses symbolic Clock or Reset slots");
  expectError(test, build(Crossing::FuToNode),
              "crosses symbolic Clock or Reset slots");
}

MemoryConnectivitySpec memoryConnectivity(llvm::StringRef test,
                                          bool exposesSubordinate) {
  ::fabric::MemoryConnectivityDeclaration declaration;
  ::fabric::MemoryOperationPortDispatchDeclaration operationPort;
  operationPort.capabilityTargetDomains = {{managerMemoryTarget(0)}};
  declaration.operationPorts.push_back(std::move(operationPort));
  if (exposesSubordinate) {
    ::fabric::MemorySubordinateDispatchDeclaration subordinate;
    subordinate.maxExposedBindings = 1;
    subordinate.targetDomain = {managerMemoryTarget(0)};
    declaration.subordinateEndpoints.push_back(std::move(subordinate));
  }
  return take(test, MemoryConnectivitySpec::create(std::move(declaration)));
}

enum class MemoryInternalRelation {
  OperationDispatch,
  SubordinateDispatch,
  EngineTokenConnection,
  LocalOperationEndpoints,
};

llvm::Expected<FinalizedFabricDesign>
buildMemoryInternalRelation(llvm::StringRef test,
                            MemoryInternalRelation relation,
                            std::optional<FabricClockResetKind> crossingKind) {
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits0 = take(test, PortType::bits(0));
  const PortType bits32 = take(test, PortType::bits(32));
  const PortType memory32 =
      take(test, PortType::memory({PortType::kDynamicExtent}, bits32));

  std::vector<PortType> inputTypes;
  switch (relation) {
  case MemoryInternalRelation::OperationDispatch:
    inputTypes = {memory32, bits32, bits0};
    break;
  case MemoryInternalRelation::SubordinateDispatch:
    break;
  case MemoryInternalRelation::EngineTokenConnection:
    inputTypes = {memory32, bits32, bits0, bits32, bits0};
    break;
  case MemoryInternalRelation::LocalOperationEndpoints:
    inputTypes = {bits32, bits0};
    break;
  }
  SpatialCoreBuilder core = take(
      test, design.createSpatialCore("memory-internal-domain", inputTypes, {}));
  std::vector<SpatialValue> inputs;
  inputs.reserve(inputTypes.size());
  for (std::size_t ordinal = 0; ordinal < inputTypes.size(); ++ordinal)
    inputs.push_back(take(test, core.input(ordinal)));

  mlir::MLIRContext contractContext(mlir::MLIRContext::Threading::DISABLED);
  std::optional<LocalMemoryServiceSpec> localService;
  if (relation != MemoryInternalRelation::OperationDispatch)
    localService =
        take(test, LocalMemoryServiceSpec::create(
                       4096, localMemoryContract(test, contractContext)));

  llvm::Expected<MemorySpec> memorySpec = [&]() -> llvm::Expected<MemorySpec> {
    if (relation == MemoryInternalRelation::OperationDispatch) {
      ::fabric::MemoryConnectivityDeclaration connectivity;
      ::fabric::MemoryOperationPortDispatchDeclaration port;
      port.capabilityTargetDomains = {{managerMemoryTarget(0)}};
      connectivity.operationPorts.push_back(std::move(port));
      return MemorySpec::create(
          {memory32, bits32, bits0}, {bits32, bits0}, {0}, {},
          MemoryEngineSpec::spatial({loadPortDeclaration()}), std::nullopt,
          take(test, MemoryConnectivitySpec::create(std::move(connectivity))));
    }
    if (relation == MemoryInternalRelation::SubordinateDispatch) {
      ::fabric::MemoryConnectivityDeclaration connectivity;
      ::fabric::MemorySubordinateDispatchDeclaration subordinate;
      subordinate.maxExposedBindings = 1;
      subordinate.targetDomain = {localMemoryTarget()};
      connectivity.subordinateEndpoints.push_back(std::move(subordinate));
      return MemorySpec::create(
          {}, {memory32}, {}, {0}, std::nullopt, std::move(localService),
          take(test, MemoryConnectivitySpec::create(std::move(connectivity))));
    }
    if (relation == MemoryInternalRelation::LocalOperationEndpoints) {
      ::fabric::MemoryConnectivityDeclaration connectivity;
      ::fabric::MemoryOperationPortDispatchDeclaration port;
      port.capabilityTargetDomains = {{localMemoryTarget()}};
      connectivity.operationPorts.push_back(std::move(port));
      return MemorySpec::create(
          {bits32, bits0}, {bits32, bits0}, {}, {},
          MemoryEngineSpec::spatial({loadPortDeclaration()}),
          std::move(localService),
          take(test, MemoryConnectivitySpec::create(std::move(connectivity))));
    }

    ::fabric::MemoryOperationPortDeclaration first = loadPortDeclaration();
    first.endpointInventory = {0, 1, 4, 5};
    for (::fabric::MemoryRoleEndpointBindingRecord &binding :
         first.capabilityAlternatives.front().roleToEndpoint)
      if (binding.endpointOrdinal >= 2)
        binding.endpointOrdinal += 2;
    ::fabric::MemoryOperationPortDeclaration second = loadPortDeclaration();
    second.endpointInventory = {2, 3, 6, 7};
    for (::fabric::MemoryRoleEndpointBindingRecord &binding :
         second.capabilityAlternatives.front().roleToEndpoint)
      binding.endpointOrdinal += binding.endpointOrdinal < 2 ? 2 : 4;

    ::fabric::MemoryConnectivityDeclaration connectivity;
    ::fabric::MemoryOperationPortDispatchDeclaration firstDispatch;
    firstDispatch.capabilityTargetDomains = {{managerMemoryTarget(0)}};
    connectivity.operationPorts.push_back(std::move(firstDispatch));
    ::fabric::MemoryOperationPortDispatchDeclaration secondDispatch;
    secondDispatch.capabilityTargetDomains = {{localMemoryTarget()}};
    connectivity.operationPorts.push_back(std::move(secondDispatch));
    connectivity.internalConnections.push_back({4, 2});
    return MemorySpec::create(
        {memory32, bits32, bits0, bits32, bits0},
        {bits32, bits0, bits32, bits0}, {0}, {},
        MemoryEngineSpec::spatial({std::move(first), std::move(second)}),
        std::move(localService),
        take(test, MemoryConnectivitySpec::create(std::move(connectivity))));
  }();
  if (!memorySpec)
    return memorySpec.takeError();
  MemoryResult memory =
      take(test, core.addMemory(inputs, std::move(*memorySpec)));

  const ModuleDomainSlotHandle firstClock =
      take(test, core.declareDomainSlot(FabricClockResetKind::Clock));
  const ModuleDomainSlotHandle secondClock =
      crossingKind == FabricClockResetKind::Clock
          ? take(test, core.declareDomainSlot(FabricClockResetKind::Clock))
          : firstClock;
  const ModuleDomainSlotHandle firstReset =
      take(test, core.declareDomainSlot(FabricClockResetKind::Reset));
  const ModuleDomainSlotHandle secondReset =
      crossingKind == FabricClockResetKind::Reset
          ? take(test, core.declareDomainSlot(FabricClockResetKind::Reset))
          : firstReset;
  const auto assign = [&](ModuleDomainMemberHandle member,
                          const ModuleDomainSlotHandle &clock,
                          const ModuleDomainSlotHandle &reset) {
    if (llvm::Error error = core.assignDomainSlot(member, clock))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = core.assignDomainSlot(member, reset))
      fail(test, llvm::toString(std::move(error)));
  };
  for (std::size_t ordinal = 0; ordinal < inputTypes.size(); ++ordinal)
    assign(take(test, core.inputDomainMember(ordinal)), firstClock, firstReset);
  assign(memory.domainMember(), firstClock, firstReset);

  switch (relation) {
  case MemoryInternalRelation::OperationDispatch:
    assign(take(test, memory.operationPortMember(0)), secondClock, secondReset);
    break;
  case MemoryInternalRelation::SubordinateDispatch:
    assign(*memory.localServiceMember(), secondClock, secondReset);
    break;
  case MemoryInternalRelation::EngineTokenConnection:
    assign(take(test, memory.operationPortMember(0)), firstClock, firstReset);
    assign(take(test, memory.operationPortMember(1)), secondClock, secondReset);
    assign(*memory.localServiceMember(), secondClock, secondReset);
    break;
  case MemoryInternalRelation::LocalOperationEndpoints:
    assign(take(test, memory.operationPortMember(0)), secondClock, secondReset);
    assign(*memory.localServiceMember(), secondClock, secondReset);
    break;
  }

  if (llvm::Error error = core.close({}))
    return std::move(error);
  return std::move(design).finalize();
}

void operationDispatchRemainsWithinOneSymbolicDomain() {
  const llvm::StringRef test = __func__;
  auto sameDomain = buildMemoryInternalRelation(
      test, MemoryInternalRelation::OperationDispatch, std::nullopt);
  if (!sameDomain)
    fail(test, llvm::toString(sameDomain.takeError()));
  for (FabricClockResetKind kind :
       {FabricClockResetKind::Clock, FabricClockResetKind::Reset})
    expectError(test,
                buildMemoryInternalRelation(
                    test, MemoryInternalRelation::OperationDispatch, kind),
                "crosses symbolic Clock or Reset slots");
}

void subordinateDispatchRemainsWithinOneSymbolicDomain() {
  const llvm::StringRef test = __func__;
  auto sameDomain = buildMemoryInternalRelation(
      test, MemoryInternalRelation::SubordinateDispatch, std::nullopt);
  if (!sameDomain)
    fail(test, llvm::toString(sameDomain.takeError()));
  for (FabricClockResetKind kind :
       {FabricClockResetKind::Clock, FabricClockResetKind::Reset})
    expectError(test,
                buildMemoryInternalRelation(
                    test, MemoryInternalRelation::SubordinateDispatch, kind),
                "crosses symbolic Clock or Reset slots");
}

void memoryEngineConnectionsRemainWithinOneSymbolicDomain() {
  const llvm::StringRef test = __func__;
  auto sameDomain = buildMemoryInternalRelation(
      test, MemoryInternalRelation::EngineTokenConnection, std::nullopt);
  if (!sameDomain)
    fail(test, llvm::toString(sameDomain.takeError()));
  for (FabricClockResetKind kind :
       {FabricClockResetKind::Clock, FabricClockResetKind::Reset})
    expectError(test,
                buildMemoryInternalRelation(
                    test, MemoryInternalRelation::EngineTokenConnection, kind),
                "crosses symbolic Clock or Reset slots");
}

void memoryOperationEndpointsRemainWithinOneSymbolicDomain() {
  const llvm::StringRef test = __func__;
  auto sameDomain = buildMemoryInternalRelation(
      test, MemoryInternalRelation::LocalOperationEndpoints, std::nullopt);
  if (!sameDomain)
    fail(test, llvm::toString(sameDomain.takeError()));
  for (FabricClockResetKind kind :
       {FabricClockResetKind::Clock, FabricClockResetKind::Reset})
    expectError(
        test,
        buildMemoryInternalRelation(
            test, MemoryInternalRelation::LocalOperationEndpoints, kind),
        "crosses symbolic Clock or Reset slots");
}

void memoryConnectionsRemainWithinOneSymbolicDomain() {
  const llvm::StringRef test = __func__;
  const auto build = [&](bool crossing) {
    TemporaryDirectory directory(test);
    loom::ArtifactStore store(directory.path());
    DesignBuilder design(store);
    const PortType bits0 = take(test, PortType::bits(0));
    const PortType bits32 = take(test, PortType::bits(32));
    const PortType memory32 =
        take(test, PortType::memory({PortType::kDynamicExtent}, bits32));
    auto core = take(
        test, design.createSpatialCore(
                  "memory-connection",
                  {memory32, bits32, bits0, bits32, bits0, bits32, bits0}, {}));
    auto provider = take(
        test,
        core.addMemory(
            {take(test, core.input(0)), take(test, core.input(1)),
             take(test, core.input(2))},
            take(test,
                 MemorySpec::create(
                     {memory32, bits32, bits0}, {memory32, bits32, bits0}, {0},
                     {0}, MemoryEngineSpec::spatial({loadPortDeclaration()}),
                     std::nullopt, memoryConnectivity(test, true)))));
    auto requester = take(
        test,
        core.addMemory(
            {provider.values()[0], take(test, core.input(3)),
             take(test, core.input(4))},
            take(test, MemorySpec::create(
                           {memory32, bits32, bits0}, {bits32, bits0}, {0}, {},
                           MemoryEngineSpec::spatial({loadPortDeclaration()}),
                           std::nullopt, memoryConnectivity(test, false)))));
    auto secondRequester = take(
        test,
        core.addMemory(
            {provider.values()[0], take(test, core.input(5)),
             take(test, core.input(6))},
            take(test, MemorySpec::create(
                           {memory32, bits32, bits0}, {bits32, bits0}, {0}, {},
                           MemoryEngineSpec::spatial({loadPortDeclaration()}),
                           std::nullopt, memoryConnectivity(test, false)))));

    const ModuleDomainSlotHandle firstClock =
        take(test, core.declareDomainSlot(FabricClockResetKind::Clock));
    const ModuleDomainSlotHandle secondClock =
        crossing
            ? take(test, core.declareDomainSlot(FabricClockResetKind::Clock))
            : firstClock;
    const ModuleDomainSlotHandle reset =
        take(test, core.declareDomainSlot(FabricClockResetKind::Reset));
    for (std::size_t ordinal = 0; ordinal != 7; ++ordinal) {
      const ModuleDomainMemberHandle member =
          take(test, core.inputDomainMember(ordinal));
      if (llvm::Error error = core.assignDomainSlot(
              member, ordinal >= 3 && ordinal < 5 ? secondClock : firstClock))
        fail(test, llvm::toString(std::move(error)));
      if (llvm::Error error = core.assignDomainSlot(member, reset))
        fail(test, llvm::toString(std::move(error)));
    }
    for (const ModuleDomainMemberHandle &member :
         {provider.domainMember(),
          take(test, provider.operationPortMember(0))}) {
      for (const ModuleDomainSlotHandle *slot : {&firstClock, &reset})
        if (llvm::Error error = core.assignDomainSlot(member, *slot))
          fail(test, llvm::toString(std::move(error)));
    }
    for (const ModuleDomainMemberHandle &member :
         {requester.domainMember(),
          take(test, requester.operationPortMember(0))}) {
      for (const ModuleDomainSlotHandle *slot : {&secondClock, &reset})
        if (llvm::Error error = core.assignDomainSlot(member, *slot))
          fail(test, llvm::toString(std::move(error)));
    }
    for (const ModuleDomainMemberHandle &member :
         {secondRequester.domainMember(),
          take(test, secondRequester.operationPortMember(0))}) {
      for (const ModuleDomainSlotHandle *slot : {&firstClock, &reset})
        if (llvm::Error error = core.assignDomainSlot(member, *slot))
          fail(test, llvm::toString(std::move(error)));
    }
    if (llvm::Error error = core.close({}))
      fail(test, llvm::toString(std::move(error)));
    return std::move(design).finalize();
  };

  auto sameDomain = build(false);
  if (!sameDomain)
    fail(test, llvm::toString(sameDomain.takeError()));
  if (sameDomain->roots().front().view().memoryServiceConnections().size() != 2)
    fail(test, "Module artifact omitted an internal memory connection");
  expectError(test, build(true), "crosses symbolic Clock or Reset slots");
}

} // namespace

void runDomainAuthoringTests() {
  domainAuthoringRelationRejectsOutOfCatalogValues();
  domainAuthoringRelationEnforcesTotality();
  instanceBindingRelationMustBeCanonicalAndTotal();
  builderDomainAuthoringRejectsForeignAndStaleHandles();
  builderDomainAuthoringRequiresTotalAssignment();
  moduleInstanceDomainSlotBindingsAreExplicit();
  constructionResultsExposeRoleTypedMembers();
  omittedModuleDomainMatchesExplicitSingleDomain();
  defaultChildRequiresExplicitInstanceBindings();
  failedInstantiationPreservesParentDomainAuthoring();
  moduleConnectionsRemainWithinOneSymbolicDomain();
  nestedConnectionsRemainWithinOneSymbolicDomain();
  operationDispatchRemainsWithinOneSymbolicDomain();
  subordinateDispatchRemainsWithinOneSymbolicDomain();
  memoryEngineConnectionsRemainWithinOneSymbolicDomain();
  memoryOperationEndpointsRemainWithinOneSymbolicDomain();
  memoryConnectionsRemainWithinOneSymbolicDomain();
}

} // namespace loom::adg::test

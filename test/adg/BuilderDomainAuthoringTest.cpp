#include "ADGBuilderTestSupport.h"

#include "ADG/Builder.h"
#include "Fabric/IR/ModuleDomain.h"

#include "mlir/IR/BuiltinOps.h"
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
using loom::adg::FuNode;
using loom::adg::FuSpec;
using loom::adg::MemoryResult;
using loom::adg::ModuleDomainMemberHandle;
using loom::adg::ModuleDomainSlotHandle;
using loom::adg::ModuleInstanceDomainSlotBinding;
using loom::adg::PeSpec;
using loom::adg::PortType;
using loom::adg::SpatialValue;
using loom::adg::SwitchResult;

using ::fabric::ModuleDomainAuthoringRelation;
using loom::fabric::FabricClockResetKind;
using loom::fabric::FabricOrdinal;
using loom::fabric::FabricPortDirection;

// The two-argument Module instantiate overload is not retained: a slotless
// target is expressed by an explicit empty binding range.
template <typename Builder, typename = void>
struct HasTwoArgumentInstantiate : std::false_type {};
template <typename Builder>
struct HasTwoArgumentInstantiate<
    Builder, std::void_t<decltype(std::declval<Builder &>().instantiate(
                 std::declval<const Builder &>(),
                 std::declval<llvm::ArrayRef<SpatialValue>>()))>>
    : std::true_type {};

static_assert(!HasTwoArgumentInstantiate<loom::adg::SpatialCoreBuilder>::value,
              "loom.fabric 3.0 retains no two-argument instantiate overload");

static_assert(
    std::is_same<decltype(std::declval<const FifoResult &>().value()),
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
                              FifoSpec{bits32, 2, false}));
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
  expectError(test, std::move(design).finalize(), "domain slot");
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
                                  FifoSpec{bits16, 2, true}));
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
  expectError(test, std::move(design).finalize(), "domain slot");
}

void constructionResultsExposeRoleTypedMembers() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(test, PortType::bits(32));

  auto core =
      take(test, design.createSpatialCore("pe-members", {bits32}, {bits32}));
  auto pe = take(test, core.addPe({take(test, core.input(0))},
                                  PeSpec::spatial({bits32}, {bits32})));
  auto fu = take(test, pe.addFu({take(test, pe.input(0))},
                                FuSpec{{bits32}, {bits32}}));
  auto sum = take(test,
                  fu.addOperation({take(test, fu.input(0))},
                                  integerCapability(
                                      ::fabric::ImplementationFamilyId::
                                          ScalarIntegerAddSub,
                                      ::dataflow::OperationSchemaId::ArithAddI,
                                      bits32)));
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
  expectError(test, std::move(design).finalize(), "domain slot");
}

void slotlessModuleInstanceStaysOnLegacyPath() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits16 = take(test, PortType::bits(16));

  auto pipeline =
      take(test, design.createSpatialCore("pipeline", {bits16}, {bits16}));
  const FifoResult stage =
      take(test, pipeline.addFifo(take(test, pipeline.input(0)),
                                  FifoSpec{bits16, 2, true}));
  if (llvm::Error error = pipeline.close({stage.value()}))
    fail(test, llvm::toString(std::move(error)));

  auto top = take(test, design.createSpatialCore("top", {bits16}, {bits16}));
  auto outputs = take(
      test, top.instantiate(pipeline, {take(test, top.input(0))}, {}));
  if (llvm::Error error = top.close(outputs))
    fail(test, llvm::toString(std::move(error)));

  loom::adg::FinalizedFabricDesign finalized =
      take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 2,
          "slotless module instance left the legacy 1.1 path");
}

} // namespace

void runDomainAuthoringTests() {
  domainAuthoringRelationRejectsOutOfCatalogValues();
  domainAuthoringRelationEnforcesTotality();
  builderDomainAuthoringRejectsForeignAndStaleHandles();
  builderDomainAuthoringRequiresTotalAssignment();
  moduleInstanceDomainSlotBindingsAreExplicit();
  constructionResultsExposeRoleTypedMembers();
  slotlessModuleInstanceStaysOnLegacyPath();
}

} // namespace loom::adg::test

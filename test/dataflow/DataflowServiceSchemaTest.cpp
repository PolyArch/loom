//===- DataflowServiceSchemaTest.cpp - Canonical Service Schema 2.0 -------===//
//
// Proves the Canonical Service Schema 2.0 projection of the Dataflow semantic
// layer: exactly six kinds, the deterministic argument and result order each
// kind owns, its one typed effect, its transfer legs and completion leg, and
// the immutable addressed-access view that parameterizes the four memory
// access kinds.
//
// Every projected type is compared against the exact type the actor already
// owns, so the schema may add order, roles, effects, legs, and completion but
// can never become a second owner of an actor's type, shape, mask, or
// contract. Every projected width is stated exactly, per kind and per mask
// form, so a width rule cannot be changed without a failure.
//
//===----------------------------------------------------------------------===//

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"

#include "Common/IndexWidth.h"
#include "Common/VectorWidth.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>

using namespace dataflow;
using namespace dataflow::semantics;

namespace {

//===----------------------------------------------------------------------===//
// The view is a derived projection, not a record a caller can fabricate
//===----------------------------------------------------------------------===//

static_assert(!std::is_default_constructible_v<CanonicalMemoryAccessView>,
              "an access view exists only as a projection of one exact actor");
static_assert(!std::is_aggregate_v<CanonicalMemoryAccessView>,
              "an access view must not be assembled field by field");
static_assert(!std::is_copy_assignable_v<CanonicalMemoryAccessView>,
              "a derived view must not be overwritten in place");
static_assert(!std::is_move_assignable_v<CanonicalMemoryAccessView>,
              "a derived view must not be overwritten in place");
static_assert(std::is_copy_constructible_v<CanonicalMemoryAccessView>,
              "a derived view is still an ordinary value");

// One canonical actor per function. Together they cover the three access
// forms, both mask forms for every addressed kind, the three compare-exchange
// success arms, an element type that is itself a vector, and an index width a
// closer scope declares, without enumerating their cross product.
constexpr llvm::StringLiteral fixture = R"mlir(
module {
  func.func @read_element(%mem: memref<10xi32>, %addr: index, %ctrl: none)
      -> (i32, none) {
    %data, %done = dataflow.load %mem[%addr] %ctrl : memref<10xi32>
    return %data, %done : i32, none
  }

  // A closer scope declares this actor's index width, so its address geometry
  // follows that declaration rather than the configured fallback the other
  // top-level actors take.
  module @scoped_index attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
  } {
    func.func @read_indexed_masked(%mem: memref<10xi32>,
                                   %addr: vector<2x3xindex>,
                                   %mask: vector<2x3xi1>, %ctrl: none)
        -> (vector<2x3xi32>, none) {
      %data, %done = dataflow.load %mem[%addr] %ctrl mask %mask
          : memref<10xi32>, vector<2x3xindex>, vector<2x3xi32>
      return %data, %done : vector<2x3xi32>, none
    }
  }

  func.func @write_contiguous(%mem: memref<10xi32>, %addr: index,
                              %value: vector<4xi32>, %ctrl: none) -> none {
    %done = dataflow.store %mem[%addr] %value %ctrl
        : memref<10xi32>, vector<4xi32>
    return %done : none
  }

  func.func @write_contiguous_masked(%mem: memref<10xi32>, %addr: index,
                                     %value: vector<4xi32>,
                                     %mask: vector<4xi1>, %ctrl: none) -> none {
    %done = dataflow.store %mem[%addr] %value %ctrl mask %mask
        : memref<10xi32>, vector<4xi32>
    return %done : none
  }

  func.func @rmw_element(%mem: memref<10xi32>, %addr: index, %value: i32,
                         %ctrl: none) -> (i32, none) {
    %old, %done = dataflow.atomic_rmw %mem[%addr] %value %ctrl
        {contract = #dataflow.rmw_contract<
            kind = add,
            access = <ordering = monotonic, sync_scope = <system>>>}
        : memref<10xi32>
    return %old, %done : i32, none
  }

  func.func @rmw_indexed_masked(%mem: memref<10xi32>, %addr: vector<2x3xindex>,
                                %value: vector<2x3xi32>,
                                %mask: vector<2x3xi1>, %ctrl: none)
      -> (vector<2x3xi32>, none) {
    %old, %done = dataflow.atomic_rmw %mem[%addr] %value %ctrl mask %mask
        {contract = #dataflow.rmw_contract<
            kind = add,
            access = <ordering = monotonic, sync_scope = <system>,
                      vector_granularity = per_lane>>}
        : memref<10xi32>, vector<2x3xindex>, vector<2x3xi32>
    return %old, %done : vector<2x3xi32>, none
  }

  func.func @cmpxchg_element(%mem: memref<10xi32>, %addr: index,
                             %expected: i32, %desired: i32, %ctrl: none)
      -> (i32, i1, none) {
    %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %expected %desired %ctrl
        {contract = #dataflow.cmpxchg_contract<success_ordering = seq_cst,
                                               failure_ordering = monotonic,
                                               sync_scope = <system>>}
        : memref<10xi32> -> i1
    return %old, %ok, %done : i32, i1, none
  }

  func.func @cmpxchg_whole_payload(%mem: memref<8xvector<4xi32>>, %addr: index,
                                   %expected: vector<4xi32>,
                                   %desired: vector<4xi32>, %ctrl: none)
      -> (vector<4xi32>, i1, none) {
    %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %expected %desired %ctrl
        {contract = #dataflow.cmpxchg_contract<
            success_ordering = seq_cst, failure_ordering = monotonic,
            sync_scope = <system>, vector_granularity = whole_payload>}
        : memref<8xvector<4xi32>> -> i1
    return %old, %ok, %done : vector<4xi32>, i1, none
  }

  func.func @cmpxchg_per_lane_masked(%mem: memref<10xi32>, %addr: index,
                                     %expected: vector<4xi32>,
                                     %desired: vector<4xi32>,
                                     %mask: vector<4xi1>, %ctrl: none)
      -> (vector<4xi32>, vector<4xi1>, none) {
    %old, %ok, %done = dataflow.cmpxchg %mem[%addr] %expected %desired %ctrl
        mask %mask
        {contract = #dataflow.cmpxchg_contract<
            success_ordering = seq_cst, failure_ordering = monotonic,
            sync_scope = <system>, vector_granularity = per_lane>}
        : memref<10xi32>, vector<4xi32> -> vector<4xi1>
    return %old, %ok, %done : vector<4xi32>, vector<4xi1>, none
  }

  func.func @fence(%ctrl: none) -> none {
    %done = dataflow.fence %ctrl
        {contract = #dataflow.fence_contract<ordering = seq_cst,
                                             sync_scope = <system>>}
    return %done : none
  }

  // One memory element that is itself a vector, whose exact width is past the
  // unsigned range and well inside an exact 64-bit count.
  func.func @read_wide_element(%mem: memref<8xvector<257xi16777215>>,
                               %addr: index, %ctrl: none)
      -> (vector<257xi16777215>, none) {
    %data, %done = dataflow.load %mem[%addr] %ctrl
        : memref<8xvector<257xi16777215>>
    return %data, %done : vector<257xi16777215>, none
  }

  // The same element rule with a shape product no exact count can hold.
  func.func @read_unrepresentable_element(
      %mem: memref<4xvector<4294967296x4294967296xi8>>, %addr: index,
      %ctrl: none) -> (vector<4294967296x4294967296xi8>, none) {
    %data, %done = dataflow.load %mem[%addr] %ctrl
        : memref<4xvector<4294967296x4294967296xi8>>
    return %data, %done : vector<4294967296x4294967296xi8>, none
  }

  // The host for source-owner changes. Its spare operands carry the exact
  // types each change needs, so one actor can be moved between geometries.
  func.func @mutation_host(%mem: memref<10xi32>, %addr: index,
                           %lanes: vector<2x3xindex>, %mask: vector<2x3xi1>,
                           %narrow: memref<8xi16>, %ctrl: none)
      -> (i32, none) {
    %data, %done = dataflow.load %mem[%addr] %ctrl : memref<10xi32>
    return %data, %done : i32, none
  }

  // A closer scope declares the canonical index width, and this memory's
  // element type is itself an index, so both widths follow that declaration
  // rather than the configured one.
  module @declared_scope attributes {
    dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 128>>
  } {
    func.func @read_declared_index(%mem: memref<10xindex>, %addr: index,
                                   %ctrl: none) -> (index, none) {
      %data, %done = dataflow.load %mem[%addr] %ctrl : memref<10xindex>
      return %data, %done : index, none
    }
  }
}
)mlir";

using Role = ServiceValueRole;

// The schema's own ordered signatures, written once.
constexpr Role readArguments[] = {Role::Address, Role::Control};
constexpr Role maskedReadArguments[] = {Role::Address, Role::Mask,
                                        Role::Control};
constexpr Role readResults[] = {Role::Data, Role::Completion};
constexpr Role writeArguments[] = {Role::Address, Role::Data, Role::Control};
constexpr Role maskedWriteArguments[] = {Role::Address, Role::Data, Role::Mask,
                                         Role::Control};
constexpr Role writeResults[] = {Role::Completion};
constexpr Role rmwArguments[] = {Role::Address, Role::Update, Role::Control};
constexpr Role maskedRmwArguments[] = {Role::Address, Role::Update, Role::Mask,
                                       Role::Control};
constexpr Role rmwResults[] = {Role::Old, Role::Completion};
constexpr Role cmpxchgArguments[] = {Role::Address, Role::Expected,
                                     Role::Desired, Role::Control};
constexpr Role maskedCmpxchgArguments[] = {
    Role::Address, Role::Expected, Role::Desired, Role::Mask, Role::Control};
constexpr Role cmpxchgResults[] = {Role::Old, Role::Success, Role::Completion};
constexpr Role fenceArguments[] = {Role::Control};
constexpr Role fenceResults[] = {Role::Completion};
constexpr Role messageArguments[] = {Role::Payload};
constexpr Role messageResults[] = {Role::Completion};

constexpr std::int64_t indexedShape[] = {2, 3};
constexpr std::int64_t contiguousShape[] = {4};

/// The widths one addressed access derives from the types its fixture states.
/// The index and address widths are not stated here: they belong to the shared
/// closest-scope resolver, which `checkAccess` consults for each actor.
struct WidthExpectation {
  std::uint64_t element;
  std::uint64_t data;
  std::uint64_t mask;
};

/// The addressed-access facts the parameterized kinds derive from the actor.
struct AccessExpectation {
  MemoryAccessForm form;
  MemoryMaskForm mask;
  /// Empty for an element access, whose one lane is one complete element.
  llvm::ArrayRef<std::int64_t> laneShape;
  std::uint64_t laneCount;
  /// One logical address for an element or contiguous access, one per lane
  /// when the access is indexed.
  std::uint64_t addressCount;
  std::optional<VectorAtomicGranularity> granularity;
  WidthExpectation widths;
};

struct Expectation {
  llvm::StringRef function;
  ServiceKind kind;
  ServiceEffect effect;
  llvm::ArrayRef<Role> arguments;
  llvm::ArrayRef<Role> results;
  std::optional<AccessExpectation> access;
};

const Expectation expectations[] = {
    {"read_indexed_masked", ServiceKind::MemoryRead, ServiceEffect::Read,
     maskedReadArguments, readResults,
     AccessExpectation{MemoryAccessForm::Indexed,
                       MemoryMaskForm::Dynamic,
                       indexedShape,
                       6,
                       6,
                       std::nullopt,
                       {32, 192, 6}}},
    {"write_contiguous_masked", ServiceKind::MemoryWrite, ServiceEffect::Write,
     maskedWriteArguments, writeResults,
     AccessExpectation{MemoryAccessForm::Contiguous,
                       MemoryMaskForm::Dynamic,
                       contiguousShape,
                       4,
                       1,
                       std::nullopt,
                       {32, 128, 4}}},
    {"rmw_element", ServiceKind::MemoryAtomicRmw,
     ServiceEffect::ReadModifyWrite, rmwArguments, rmwResults,
     AccessExpectation{MemoryAccessForm::Element,
                       MemoryMaskForm::Absent,
                       {},
                       1,
                       1,
                       std::nullopt,
                       {32, 32, 0}}},
    {"rmw_indexed_masked", ServiceKind::MemoryAtomicRmw,
     ServiceEffect::ReadModifyWrite, maskedRmwArguments, rmwResults,
     AccessExpectation{MemoryAccessForm::Indexed,
                       MemoryMaskForm::Dynamic,
                       indexedShape,
                       6,
                       6,
                       VectorAtomicGranularity::PerLane,
                       {32, 192, 6}}},
    {"cmpxchg_element", ServiceKind::MemoryCompareExchange,
     ServiceEffect::CompareExchange, cmpxchgArguments, cmpxchgResults,
     AccessExpectation{MemoryAccessForm::Element,
                       MemoryMaskForm::Absent,
                       {},
                       1,
                       1,
                       std::nullopt,
                       {32, 32, 0}}},
    // One vector-valued memory element is one atomic object and one lane, so
    // it keeps element geometry while its element width is the whole vector.
    {"cmpxchg_whole_payload", ServiceKind::MemoryCompareExchange,
     ServiceEffect::CompareExchange, cmpxchgArguments, cmpxchgResults,
     AccessExpectation{MemoryAccessForm::Element,
                       MemoryMaskForm::Absent,
                       {},
                       1,
                       1,
                       VectorAtomicGranularity::WholePayload,
                       {128, 128, 0}}},
    {"cmpxchg_per_lane_masked", ServiceKind::MemoryCompareExchange,
     ServiceEffect::CompareExchange, maskedCmpxchgArguments, cmpxchgResults,
     AccessExpectation{MemoryAccessForm::Contiguous,
                       MemoryMaskForm::Dynamic,
                       contiguousShape,
                       4,
                       1,
                       VectorAtomicGranularity::PerLane,
                       {32, 128, 4}}},
    // The closer declaration owns this actor's index width, and its memory
    // element is an index, so the element width follows the same owner.
    {"read_declared_index", ServiceKind::MemoryRead, ServiceEffect::Read,
     readArguments, readResults,
     AccessExpectation{MemoryAccessForm::Element,
                       MemoryMaskForm::Absent,
                       {},
                       1,
                       1,
                       std::nullopt,
                       {128, 128, 0}}},
    {"read_wide_element", ServiceKind::MemoryRead, ServiceEffect::Read,
     readArguments, readResults,
     AccessExpectation{MemoryAccessForm::Element,
                       MemoryMaskForm::Absent,
                       {},
                       1,
                       1,
                       std::nullopt,
                       {4311744255ull, 4311744255ull, 0}}},
    {"fence", ServiceKind::MemoryFence, ServiceEffect::Order, fenceArguments,
     fenceResults, std::nullopt},
};

bool fail(llvm::StringRef subject, const llvm::Twine &message) {
  llvm::errs() << subject << ": " << message << '\n';
  return false;
}

/// One fixture function anywhere in the module, including a nested scope.
mlir::func::FuncOp findFunction(mlir::ModuleOp module, llvm::StringRef name) {
  mlir::func::FuncOp found;
  module.walk([&](mlir::func::FuncOp function) {
    if (function.getSymName() == name)
      found = function;
  });
  return found;
}

/// The single Dataflow actor in one fixture function.
mlir::Operation *findActor(mlir::func::FuncOp function) {
  mlir::Operation *actor = nullptr;
  unsigned found = 0;
  function.walk([&](mlir::Operation *op) {
    if (op->getName().getDialectNamespace() != "dataflow")
      return;
    actor = op;
    ++found;
  });
  return found == 1 ? actor : nullptr;
}

llvm::SmallVector<Role> rolesOf(llvm::ArrayRef<ServiceValue> values) {
  llvm::SmallVector<Role> roles;
  for (const ServiceValue &value : values)
    roles.push_back(value.role);
  return roles;
}

std::string spellRoles(llvm::ArrayRef<Role> roles) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  llvm::interleaveComma(
      roles, stream, [&](Role role) { stream << static_cast<unsigned>(role); });
  return text;
}

bool checkRoles(llvm::StringRef subject, llvm::StringRef what,
                llvm::ArrayRef<ServiceValue> values,
                llvm::ArrayRef<Role> expected) {
  llvm::SmallVector<Role> roles = rolesOf(values);
  if (llvm::ArrayRef<Role>(roles) == expected)
    return true;
  return fail(subject, what + " roles are " + spellRoles(roles) +
                           ", expected " + spellRoles(expected));
}

bool sameValues(llvm::ArrayRef<ServiceValue> lhs,
                llvm::ArrayRef<ServiceValue> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip_equal(lhs, rhs))
    if (left.role != right.role || left.type != right.type)
      return false;
  return true;
}

/// The actor's own value operands: everything but its memory operand and its
/// `none` control token, in operand order.
llvm::SmallVector<mlir::Type> actorValueOperands(mlir::Operation *actor) {
  llvm::SmallVector<mlir::Type> types;
  for (mlir::Value operand : actor->getOperands())
    if (!llvm::isa<mlir::MemRefType, mlir::NoneType>(operand.getType()))
      types.push_back(operand.getType());
  return types;
}

/// The actor's own value results: everything but its `none` retirement event.
llvm::SmallVector<mlir::Type> actorValueResults(mlir::Operation *actor) {
  llvm::SmallVector<mlir::Type> types;
  for (mlir::Value result : actor->getResults())
    if (!llvm::isa<mlir::NoneType>(result.getType()))
      types.push_back(result.getType());
  return types;
}

/// Proves the schema copies no type: every value it names other than the
/// `none` control and completion events is exactly one type the actor owns,
/// in the actor's own order.
bool checkDerivedTypes(llvm::StringRef subject, llvm::StringRef what,
                       llvm::ArrayRef<ServiceValue> values, Role event,
                       llvm::ArrayRef<mlir::Type> owned) {
  llvm::SmallVector<mlir::Type> projected;
  for (const ServiceValue &value : values) {
    if (!value.type)
      return fail(subject, what + " has an untyped value");
    if (value.role == event) {
      if (!llvm::isa<mlir::NoneType>(value.type))
        return fail(subject, what + " event value is not 'none'");
      continue;
    }
    projected.push_back(value.type);
  }
  if (llvm::ArrayRef<mlir::Type>(projected) == owned)
    return true;
  return fail(subject, what + " types are not the actor's own types");
}

bool checkWidth(llvm::StringRef subject, llvm::StringRef what,
                std::uint64_t actual, std::uint64_t expected) {
  if (actual == expected)
    return true;
  return fail(subject, what + " is " + llvm::Twine(actual) + ", expected " +
                           llvm::Twine(expected));
}

bool checkAccess(llvm::StringRef subject, const CanonicalMemoryAccessView &view,
                 const AccessExpectation &expected, mlir::Operation *actor) {
  bool ok = true;
  if (view.form() != expected.form)
    ok = fail(subject, "access form differs from the actor geometry");
  if (view.maskForm() != expected.mask)
    ok = fail(subject, "mask form differs from the actor");
  if (view.laneShape() != expected.laneShape)
    ok = fail(subject, "lane shape differs from the actor geometry");
  if (view.laneCount() != expected.laneCount)
    ok = fail(subject, "lane count differs from the actor geometry");
  ok &= checkWidth(subject, "address count", view.addressCount(),
                   expected.addressCount);
  if (view.atomicGranularity() != expected.granularity)
    ok = fail(subject, "vector atomic granularity differs from the contract");
  if (view.memoryElementType() !=
      llvm::cast<mlir::MemRefType>(actor->getOperand(0).getType())
          .getElementType())
    ok = fail(subject, "memory element type is not the memref element type");
  if (view.addressType() != actor->getOperand(1).getType())
    ok = fail(subject, "address type is not the actor's address type");
  // The view references the actor's one aggregate contract; it never rebuilds
  // an equal copy of it. A bare plain access owns no attribute, so only an
  // explicit contract can be compared by identity.
  if (mlir::Attribute contract = actor->getAttr("contract"))
    if (view.contract().aggregate != contract)
      ok = fail(subject, "the view does not reference the actor's contract");
  const bool masked = expected.mask == MemoryMaskForm::Dynamic;
  if (masked != static_cast<bool>(view.maskType()))
    ok = fail(subject, "mask type presence differs from the mask form");

  const WidthExpectation &widths = expected.widths;
  ok &=
      checkWidth(subject, "element width", view.elementBits(), widths.element);
  ok &= checkWidth(subject, "data width", view.dataBits(), widths.data);
  ok &= checkWidth(subject, "mask width", view.maskBits(), widths.mask);

  // The index width belongs to the shared closest-scope resolver, asked here
  // about this exact actor. The address width is that width over the address
  // type the fixture itself states, formed by the one owner of a fixed vector
  // product. Neither is read back from the projection being checked.
  llvm::Expected<unsigned> index = loom::getIndexBitWidth(actor);
  if (!index) {
    llvm::consumeError(index.takeError());
    return fail(subject, "the actor has no canonical index width");
  }
  ok &= checkWidth(subject, "index width", view.indexBits(), *index);

  std::uint64_t addressBits = *index;
  if (auto addresses =
          llvm::dyn_cast<mlir::VectorType>(actor->getOperand(1).getType())) {
    llvm::Expected<std::uint64_t> spread =
        loom::getFixedVectorBitWidth(addresses, *index);
    if (!spread) {
      llvm::consumeError(spread.takeError());
      return fail(subject, "the stated address shape has no exact width");
    }
    addressBits = *spread;
  }
  ok &= checkWidth(subject, "address width", view.addressBits(), addressBits);
  return ok;
}

bool checkService(llvm::StringRef subject, const CanonicalService &service,
                  const Expectation &expected, mlir::Operation *actor) {
  bool ok = true;
  if (service.kind() != expected.kind)
    ok = fail(subject, "service kind is not the actor's kind");
  if (service.effect() != expected.effect)
    ok = fail(subject, "service effect is not the kind's effect");

  ServiceValues arguments = service.arguments();
  ServiceValues results = service.results();
  ok &= checkRoles(subject, "argument", arguments, expected.arguments);
  ok &= checkRoles(subject, "result", results, expected.results);
  ok &= checkDerivedTypes(subject, "argument", arguments, Role::Control,
                          actorValueOperands(actor));
  ok &= checkDerivedTypes(subject, "result", results, Role::Completion,
                          actorValueResults(actor));

  // Every memory kind is a request leg answered by a response leg, and its
  // completion is the acceptance of that response.
  if (service.legCount() != 2)
    ok = fail(subject, "a memory service has two transfer legs");
  else if (service.legDirection(0) != ServiceLegDirection::InitiatorToServer ||
           service.legDirection(1) != ServiceLegDirection::ServerToInitiator)
    ok = fail(subject, "transfer leg directions are not request then response");
  else if (!sameValues(service.legPayload(0), arguments) ||
           !sameValues(service.legPayload(1), results))
    ok = fail(subject, "leg payloads are not the arguments then the results");
  if (service.completionLeg() != service.legCount() - 1)
    ok = fail(subject, "completion is not the acceptance of the final leg");

  if (expected.access) {
    ok &= checkAccess(subject, service.access(), *expected.access, actor);
  } else if (service.fenceContract() != actor->getAttr("contract")) {
    ok = fail(subject, "the fence service does not reference its contract");
  }
  return ok;
}

/// A message transfer is the one kind no memory actor projects: it carries one
/// exact supported payload type over a single leg whose acceptance is its
/// completion. The kind has no addressed geometry, so it derives no width.
bool checkMessageTransfer(mlir::MLIRContext &context) {
  mlir::Builder builder(&context);
  mlir::Type payload = mlir::VectorType::get({4}, builder.getF32Type());
  llvm::Expected<CanonicalService> service =
      CanonicalService::messageTransfer(payload);
  if (!service) {
    llvm::consumeError(service.takeError());
    return fail("message_transfer", "an exact payload type was rejected");
  }
  bool ok = true;
  if (service->kind() != ServiceKind::MessageTransfer)
    ok = fail("message_transfer", "kind is not a message transfer");
  if (service->effect() != ServiceEffect::None)
    ok = fail("message_transfer", "a message transfer has no effect");
  ServiceValues arguments = service->arguments();
  ServiceValues results = service->results();
  ok &= checkRoles("message_transfer", "argument", arguments, messageArguments);
  ok &= checkRoles("message_transfer", "result", results, messageResults);
  if (arguments.size() != 1 || arguments.front().type != payload)
    ok = fail("message_transfer", "the payload type is not the exact type");
  if (results.size() != 1 || !llvm::isa<mlir::NoneType>(results.front().type))
    ok = fail("message_transfer", "completion is not 'none'");
  if (service->legCount() != 1 ||
      service->legDirection(0) != ServiceLegDirection::InitiatorToServer)
    ok = fail("message_transfer", "a message transfer has one sending leg");
  else if (!sameValues(service->legPayload(0), arguments))
    ok = fail("message_transfer", "the leg payload is not the arguments");
  if (service->completionLeg() != 0)
    ok = fail("message_transfer", "completion is not the acceptance of leg 0");

  if (llvm::Expected<CanonicalService> untyped =
          CanonicalService::messageTransfer(mlir::Type())) {
    ok = fail("message_transfer", "an absent payload type was accepted");
  } else {
    llvm::consumeError(untyped.takeError());
  }

  // The payload domain is the one Dataflow transfer payload domain a channel
  // element type already obeys, recursive containment included. A nested
  // channel proves the schema defers to that owner instead of restating a
  // shallower rule of its own.
  mlir::Type nested = mlir::TupleType::get(
      &context,
      {builder.getI32Type(), ChannelType::get(&context, builder.getI32Type())});
  if (llvm::Expected<CanonicalService> unsupported =
          CanonicalService::messageTransfer(nested)) {
    ok =
        fail("message_transfer", "a payload containing a channel was accepted");
  } else {
    llvm::consumeError(unsupported.takeError());
  }
  return ok;
}

/// Rejections: an operation that is not a canonical actor, an actor asked for
/// a kind it does not project, a fence asked for an addressed access view, a
/// fence its own contract owner rejects, an actor whose access geometry is
/// malformed, and no operation at all.
bool checkRejections(mlir::ModuleOp module) {
  bool ok = true;
  mlir::MLIRContext *context = module.getContext();
  auto readElement = findFunction(module, "read_element");
  auto write = findFunction(module, "write_contiguous");
  auto fence = findFunction(module, "fence");
  if (!readElement || !write || !fence)
    return fail("rejection", "the fixture is incomplete");

  mlir::Operation *notAnActor = readElement.getBody().front().getTerminator();
  if (llvm::Expected<CanonicalService> service =
          CanonicalService::forActor(notAnActor)) {
    ok = fail("rejection", "an operation that is not an actor was projected");
  } else {
    llvm::consumeError(service.takeError());
  }

  mlir::Operation *store = findActor(write);
  if (llvm::Expected<CanonicalService> service =
          CanonicalService::forActor(store, ServiceKind::MemoryRead)) {
    ok = fail("rejection", "a store was projected as a memory read");
  } else {
    llvm::consumeError(service.takeError());
  }
  if (llvm::Expected<CanonicalService> service =
          CanonicalService::forActor(store, ServiceKind::MessageTransfer)) {
    ok = fail("rejection", "an actor was projected as a message transfer");
  } else {
    llvm::consumeError(service.takeError());
  }

  mlir::Operation *fenceActor = findActor(fence);
  if (llvm::Expected<CanonicalMemoryAccessView> view =
          getCanonicalMemoryAccessView(fenceActor)) {
    ok = fail("rejection", "a fence was given an addressed access view");
  } else {
    llvm::consumeError(view.takeError());
  }

  // A fence contract may be structurally well formed and still be rejected by
  // its own contract owner, here because no compiler-target contract resolves
  // its target synchronization scope. The service kind does not relax that.
  mlir::Block &fenceBody = fence.getBody().front();
  mlir::OpBuilder fenceBuilder(fenceBody.getTerminator());
  auto unresolvedScope = SyncScopeRefAttr::get(
      context, SyncScopeKind::Target, mlir::StringAttr::get(context, "vendor"),
      mlir::StringAttr::get(context, "key"));
  FenceOp unresolved = FenceOp::create(
      fenceBuilder, fenceBuilder.getUnknownLoc(), fenceBuilder.getNoneType(),
      fenceBody.getArgument(0),
      FenceContractAttr::get(context, AtomicOrdering::SeqCst, unresolvedScope));
  if (llvm::Expected<CanonicalService> service =
          CanonicalService::forActor(unresolved)) {
    ok = fail("rejection", "a fence its contract owner rejects was projected");
  } else {
    llvm::consumeError(service.takeError());
  }
  unresolved->erase();

  // A load whose data type is neither the memory element type nor a vector of
  // it has no canonical geometry, so it has no service projection either.
  mlir::Block &body = readElement.getBody().front();
  mlir::OpBuilder builder(body.getTerminator());
  LoadOp malformed =
      LoadOp::create(builder, builder.getUnknownLoc(), builder.getF32Type(),
                     builder.getNoneType(), body.getArgument(0),
                     body.getArgument(1), body.getArgument(2));
  if (llvm::Expected<CanonicalService> service =
          CanonicalService::forActor(malformed)) {
    ok = fail("rejection", "a malformed access geometry was projected");
  } else {
    llvm::consumeError(service.takeError());
  }
  malformed->erase();

  // The query API answers rather than relying on cast behavior when there is
  // no operation to project at all.
  if (llvm::Expected<CanonicalService> service =
          CanonicalService::forActor(nullptr)) {
    ok = fail("rejection", "an absent operation was projected");
  } else {
    llvm::consumeError(service.takeError());
  }
  return ok;
}

/// One structurally malformed operation carrying a canonical actor's name.
/// Building it through a raw `OperationState` bypasses the builders and the
/// verifier, which is exactly the state a projection must refuse before it
/// reaches a generated accessor or an unchecked cast.
mlir::Operation *createRaw(mlir::OpBuilder &builder, llvm::StringRef name,
                           mlir::ValueRange operands, mlir::TypeRange results) {
  mlir::OperationState state(builder.getUnknownLoc(), name);
  state.addOperands(operands);
  state.addTypes(results);
  return builder.create(state);
}

/// Malformed actor structure is refused deterministically. Each case would
/// otherwise reach a generated accessor or an unchecked cast: an absent
/// operand range, an operand whose type is not the memory it is read as,
/// missing comparands, and an absent aggregate contract.
bool checkMalformedActors(mlir::ModuleOp module) {
  auto host = findFunction(module, "read_element");
  if (!host)
    return fail("malformed", "the fixture is incomplete");
  mlir::Block &body = host.getBody().front();
  mlir::OpBuilder builder(body.getTerminator());
  mlir::Value memory = body.getArgument(0);
  mlir::Value address = body.getArgument(1);
  mlir::Value control = body.getArgument(2);
  mlir::Type i32 = builder.getI32Type();
  mlir::Type none = builder.getNoneType();

  const std::pair<llvm::StringRef, mlir::Operation *> malformed[] = {
      {"load without operands", createRaw(builder, "dataflow.load", {}, {})},
      {"load whose memory operand is not a memref",
       createRaw(builder, "dataflow.load", {address, address, control},
                 {i32, none})},
      {"cmpxchg without its comparands",
       createRaw(builder, "dataflow.cmpxchg", {memory, address, control},
                 {i32, builder.getI1Type(), none})},
      {"fence without its contract",
       createRaw(builder, "dataflow.fence", {control}, {none})}};

  bool ok = true;
  for (auto [name, op] : malformed) {
    if (llvm::Expected<CanonicalService> service =
            CanonicalService::forActor(op)) {
      ok = fail("malformed", name + " was projected");
    } else {
      llvm::consumeError(service.takeError());
    }
    op->erase();
  }
  return ok;
}

/// An access its own owner accepts can still have a lane count or width that
/// no exact projection can represent. The projection refuses it instead of
/// publishing a wrapped product.
bool checkUnrepresentableProjection(mlir::ModuleOp module) {
  auto host = findFunction(module, "read_element");
  if (!host)
    return fail("width", "the fixture is incomplete");
  mlir::Block &body = host.getBody().front();
  mlir::OpBuilder builder(body.getTerminator());
  // Structurally an ordinary contiguous load of this memory's element type;
  // only its lane count puts the data width past an exact 64-bit value.
  auto lanes = mlir::VectorType::get({std::numeric_limits<std::int64_t>::max()},
                                     builder.getI32Type());
  LoadOp huge = LoadOp::create(builder, builder.getUnknownLoc(), lanes,
                               builder.getNoneType(), body.getArgument(0),
                               body.getArgument(1), body.getArgument(2));

  bool ok = true;
  // The anchor means something only because the actor's own verification
  // admits this shape, so the projection is the one place left to refuse it.
  if (mlir::failed(mlir::verify(huge))) {
    ok = fail("width", "the actor owner already rejects the shape");
  } else if (llvm::Expected<CanonicalService> service =
                 CanonicalService::forActor(huge)) {
    ok = fail("width", "an unrepresentable data width was projected as " +
                           llvm::Twine(service->access().dataBits()));
  } else {
    llvm::consumeError(service.takeError());
  }
  huge->erase();

  // The same rule on the element side: a shape product past an exact count is
  // refused by arithmetic alone, never by forming a value that large.
  if (auto over = findFunction(module, "read_unrepresentable_element")) {
    mlir::Operation *actor = findActor(over);
    // The actor's own structure is accepted; only the projection arithmetic
    // refuses it, so the two rejections are not confused for one another.
    if (mlir::failed(mlir::verify(actor))) {
      ok = fail("width", "the actor owner already rejects the shape");
    } else if (llvm::Expected<CanonicalService> service =
                   CanonicalService::forActor(actor)) {
      ok = fail("width", "an unrepresentable element width was projected as " +
                             llvm::Twine(service->access().elementBits()));
    } else {
      llvm::consumeError(service.takeError());
    }
  } else {
    ok = fail("width", "the fixture is incomplete");
  }
  return ok;
}

/// The projection tracks the owners it is derived from. Each step changes one
/// owned fact of a valid actor and requires the exact changed projection or a
/// deterministic rejection, then restores or replaces that actor. A stale or
/// cached projection keeps answering with the previous fact and fails here.
bool checkSourceTracking(mlir::ModuleOp module) {
  mlir::MLIRContext *context = module.getContext();
  mlir::Builder builder(context);
  auto hostFn = findFunction(module, "mutation_host");
  auto rmwFn = findFunction(module, "rmw_element");
  auto cmpxchgFn = findFunction(module, "cmpxchg_element");
  auto fenceFn = findFunction(module, "fence");
  if (!hostFn || !rmwFn || !cmpxchgFn || !fenceFn)
    return fail("tracking", "the fixture is incomplete");

  bool ok = true;
  // Every step re-derives the service itself, not just the view helper, so a
  // service that answered from anything but the actor in front of it fails.
  auto holds = [&](llvm::StringRef what, mlir::Operation *actor,
                   ServiceKind kind,
                   llvm::function_ref<bool(const CanonicalMemoryAccessView &)>
                       predicate) {
    llvm::Expected<CanonicalService> service =
        CanonicalService::forActor(actor);
    if (!service) {
      llvm::consumeError(service.takeError());
      ok = fail("tracking", what + " lost its service");
      return;
    }
    if (service->kind() != kind) {
      ok = fail("tracking", what + " obliges another service kind");
      return;
    }
    if (!predicate(service->access()))
      ok = fail("tracking", what + " did not follow its owner");
  };

  auto host = llvm::cast<LoadOp>(findActor(hostFn));
  mlir::Block &body = hostFn.getBody().front();
  mlir::Value scalarAddress = body.getArgument(1);
  mlir::Value laneAddresses = body.getArgument(2);
  mlir::Value laneMask = body.getArgument(3);
  mlir::Value narrowMemory = body.getArgument(4);
  mlir::Value memory = host.getMem();
  mlir::Type element = builder.getI32Type();
  auto laneShape = mlir::VectorType::get({2, 3}, element);

  // The oracle is the one shared closest-scope resolver, not a second default.
  llvm::Expected<unsigned> resolved = loom::getIndexBitWidth(host);
  if (!resolved) {
    llvm::consumeError(resolved.takeError());
    return fail("tracking", "the host has no canonical index width");
  }
  const std::uint64_t index = *resolved;

  holds("the element baseline", host, ServiceKind::MemoryRead,
        [](const CanonicalMemoryAccessView &v) {
          return v.form() == MemoryAccessForm::Element && v.laneCount() == 1 &&
                 v.dataBits() == 32 && v.addressCount() == 1;
        });

  // Shape and element geometry: one element becomes a row-major lane shape.
  host.getData().setType(laneShape);
  holds("a wider data shape", host, ServiceKind::MemoryRead,
        [](const CanonicalMemoryAccessView &v) {
          return v.form() == MemoryAccessForm::Contiguous &&
                 v.laneCount() == 6 &&
                 v.laneShape() == llvm::ArrayRef<std::int64_t>({2, 3}) &&
                 v.dataBits() == 192 && v.addressCount() == 1;
        });

  host.getData().setType(element);

  // Element type and width: the same actor reads a narrower memory, so the
  // exact element it names and every width over it follow that memory.
  host.setOperand(0, narrowMemory);
  host.getData().setType(builder.getIntegerType(16));
  holds("a narrower memory element", host, ServiceKind::MemoryRead,
        [&](const CanonicalMemoryAccessView &v) {
          return v.memoryElementType() == builder.getIntegerType(16) &&
                 v.elementBits() == 16 && v.dataBits() == 16 &&
                 v.dataType() == builder.getIntegerType(16);
        });
  host.setOperand(0, memory);
  host.getData().setType(laneShape);

  // Indexed address shape and count: one address becomes one per lane.
  host.setOperand(1, laneAddresses);
  holds("a per-lane address", host, ServiceKind::MemoryRead,
        [&](const CanonicalMemoryAccessView &v) {
          return v.form() == MemoryAccessForm::Indexed &&
                 v.addressCount() == 6 && v.addressBits() == 6 * index &&
                 v.addressType() == laneAddresses.getType();
        });

  // Mask presence and shape: the actor gains a dynamic lane mask.
  host->insertOperands(3, laneMask);
  holds("a dynamic mask", host, ServiceKind::MemoryRead,
        [&](const CanonicalMemoryAccessView &v) {
          return v.maskForm() == MemoryMaskForm::Dynamic && v.maskBits() == 6 &&
                 v.maskType() == laneMask.getType();
        });
  host->eraseOperand(3);
  holds("the mask removed again", host, ServiceKind::MemoryRead,
        [](const CanonicalMemoryAccessView &v) {
          return v.maskForm() == MemoryMaskForm::Absent && v.maskBits() == 0;
        });

  host.setOperand(1, scalarAddress);
  host.getData().setType(element);

  // Aggregate contract: a plain access becomes an ordered one, and the view
  // names the exact attribute the actor now owns.
  auto atomic = AtomicAccessContractAttr::get(
      context, AtomicOrdering::Acquire,
      SyncScopeRefAttr::get(context, SyncScopeKind::System, {}, {}),
      std::nullopt, /*is_volatile=*/false);
  host->setAttr("contract", atomic);
  holds("an ordered contract", host, ServiceKind::MemoryRead,
        [&](const CanonicalMemoryAccessView &v) {
          return v.contract().aggregate == atomic && v.contract().atomic;
        });
  host->removeAttr("contract");

  // A change its own owners reject is a deterministic rejection, not a stale
  // answer carried over from the last valid shape.
  host.getData().setType(builder.getF32Type());
  if (llvm::Expected<CanonicalService> service =
          CanonicalService::forActor(host)) {
    ok = fail("tracking", "a rejected geometry was still projected");
  } else {
    llvm::consumeError(service.takeError());
  }
  host.getData().setType(element);

  // Actor kind, effect and signature: the same memory read by a different
  // actor obliges a different service.
  mlir::OpBuilder inserter(host);
  StoreOp store = StoreOp::create(
      inserter, host.getLoc(), builder.getNoneType(), host.getMem(),
      scalarAddress, host.getData(), host.getCtrl());
  if (llvm::Expected<CanonicalService> service =
          CanonicalService::forActor(store)) {
    if (service->kind() != ServiceKind::MemoryWrite ||
        service->effect() != ServiceEffect::Write)
      ok = fail("tracking", "a store did not oblige a memory write");
    ok &= checkRoles("tracking", "replaced argument", service->arguments(),
                     writeArguments);
    ok &= checkRoles("tracking", "replaced result", service->results(),
                     writeResults);
  } else {
    llvm::consumeError(service.takeError());
    ok = fail("tracking", "the replacing store lost its projection");
  }
  store->erase();

  // The remaining actor kinds each own one aggregate contract, and the
  // projection names whichever one the actor holds now.
  mlir::Operation *rmw = findActor(rmwFn);
  auto rmwContract =
      llvm::cast<AtomicRmwContractAttr>(rmw->getAttr("contract"));
  auto rmwXor = AtomicRmwContractAttr::get(context, AtomicRmwKind::Xor,
                                           rmwContract.getAccess());
  rmw->setAttr("contract", rmwXor);
  holds("a changed rmw action", rmw, ServiceKind::MemoryAtomicRmw,
        [&](const CanonicalMemoryAccessView &v) {
          return v.contract().aggregate == rmwXor;
        });
  rmw->setAttr("contract", rmwContract);

  mlir::Operation *cmpxchg = findActor(cmpxchgFn);
  auto cmpxchgContract =
      llvm::cast<CompareExchangeContractAttr>(cmpxchg->getAttr("contract"));
  auto weak = CompareExchangeContractAttr::get(
      context, cmpxchgContract.getSuccessOrdering(),
      cmpxchgContract.getFailureOrdering(), cmpxchgContract.getSyncScope(),
      cmpxchgContract.getVectorGranularity(), /*weak=*/true,
      cmpxchgContract.getIsVolatile());
  cmpxchg->setAttr("contract", weak);
  holds("a weak compare-exchange", cmpxchg, ServiceKind::MemoryCompareExchange,
        [&](const CanonicalMemoryAccessView &v) {
          return v.contract().aggregate == weak;
        });
  cmpxchg->setAttr("contract", cmpxchgContract);

  mlir::Operation *fence = findActor(fenceFn);
  auto fenceContract =
      llvm::cast<FenceContractAttr>(fence->getAttr("contract"));
  auto acquire = FenceContractAttr::get(context, AtomicOrdering::Acquire,
                                        fenceContract.getSyncScope());
  fence->setAttr("contract", acquire);
  if (llvm::Expected<CanonicalService> service =
          CanonicalService::forActor(fence)) {
    if (service->fenceContract() != acquire)
      ok = fail("tracking", "a changed fence contract was not projected");
  } else {
    llvm::consumeError(service.takeError());
    ok = fail("tracking", "the changed fence lost its projection");
  }
  fence->setAttr("contract", fenceContract);

  // The one kind no actor projects follows its parameter just as directly.
  llvm::Expected<CanonicalService> narrow =
      CanonicalService::messageTransfer(builder.getI32Type());
  llvm::Expected<CanonicalService> wide =
      CanonicalService::messageTransfer(mlir::VectorType::get({4}, element));
  if (!narrow || !wide) {
    llvm::consumeError(narrow.takeError());
    llvm::consumeError(wide.takeError());
    ok = fail("tracking", "an exact payload type was rejected");
  } else if (narrow->payload() == wide->payload()) {
    ok = fail("tracking", "two payload types projected the same transfer");
  }
  return ok;
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry
      .insert<DataflowDialect, mlir::func::FuncDialect, mlir::DLTIDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(fixture, &context);
  if (!module) {
    llvm::errs() << "failed to parse the canonical actor fixture\n";
    return EXIT_FAILURE;
  }

  bool ok = true;
  for (const Expectation &expectation : expectations) {
    mlir::func::FuncOp function = findFunction(*module, expectation.function);
    mlir::Operation *actor = function ? findActor(function) : nullptr;
    if (!actor) {
      llvm::errs() << "the fixture does not hold exactly one actor in "
                   << expectation.function << '\n';
      ok = false;
      continue;
    }
    llvm::Expected<CanonicalService> service =
        CanonicalService::forActor(actor, expectation.kind);
    if (!service) {
      llvm::errs() << expectation.function << ": "
                   << llvm::toString(service.takeError()) << '\n';
      ok = false;
      continue;
    }
    ok &= checkService(expectation.function, *service, expectation, actor);
  }

  ok &= checkMessageTransfer(context);
  ok &= checkRejections(*module);
  ok &= checkMalformedActors(*module);
  ok &= checkUnrepresentableProjection(*module);
  ok &= checkSourceTracking(*module);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

#include "Fabric/IR/MemoryOperationPort.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricTypes.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace dataflow;
using namespace dataflow::semantics;
using namespace fabric;

namespace {

constexpr llvm::StringLiteral fixture = R"mlir(
module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  func.func @element_f32(%mem: memref<8xf32>, %address: index, %ctrl: none)
      -> (f32, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl : memref<8xf32>
    return %data, %done : f32, none
  }
  func.func @vector4_f32(%mem: memref<8xf32>, %address: index, %ctrl: none)
      -> (vector<4xf32>, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl
        : memref<8xf32>, vector<4xf32>
    return %data, %done : vector<4xf32>, none
  }
  func.func @vector2_f64(%mem: memref<8xf64>, %address: index, %ctrl: none)
      -> (vector<2xf64>, none) {
    %data, %done = dataflow.load %mem[%address] %ctrl
        : memref<8xf64>, vector<2xf64>
    return %data, %done : vector<2xf64>, none
  }
}
)mlir";

[[noreturn]] void fail(llvm::StringRef test, const llvm::Twine &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "accepted an invalid memory-operation port");
  llvm::consumeError(value.takeError());
}

UnsignedDomain singleton(std::uint64_t value) {
  return take("singleton",
              UnsignedDomain::fromCanonical({UnsignedInterval{value, value}}));
}

AlignmentDomain allAlignments() {
  return take("alignment",
              AlignmentDomain::create(take(
                  "alignment range",
                  UnsignedDomain::fromCanonical({UnsignedInterval{0, 63}}))));
}

ClosedEnumDomain<ReadSubwordSemantics>
readDomain(ReadSubwordSemantics semantics) {
  return take(
      "read semantics",
      ClosedEnumDomain<ReadSubwordSemantics>::fromCanonical({semantics}));
}

ClosedEnumDomain<WriteSubwordSemantics> noWrite() {
  return take("write semantics",
              ClosedEnumDomain<WriteSubwordSemantics>::fromCanonical(
                  {WriteSubwordSemantics::NotApplicable}));
}

MemoryAccessClass elementAccess() {
  return take("element access",
              MemoryAccessClass::create(
                  MemoryAccessForm::Element, singleton(32), singleton(1),
                  {MaskInactivePair{MemoryMaskForm::Absent,
                                    InactiveLaneSemantics::NotApplicable}},
                  allAlignments(), readDomain(ReadSubwordSemantics::ZeroExtend),
                  noWrite()));
}

MemoryAccessClass vectorAccess() {
  return take(
      "vector access",
      MemoryAccessClass::create(
          MemoryAccessForm::Contiguous, singleton(32), singleton(4),
          {MaskInactivePair{MemoryMaskForm::Absent,
                            InactiveLaneSemantics::NotApplicable},
           MaskInactivePair{MemoryMaskForm::Dynamic,
                            InactiveLaneSemantics::SuppressAndZeroFill}},
          allAlignments(), readDomain(ReadSubwordSemantics::Exact), noWrite()));
}

ResourceContract directContract() {
  ResourceContractDeclaration declaration;
  declaration.states = {ResourceStateDeclaration{
      StateKey(0),
      {CapacityDimensionDeclaration{CapacityDimensionKey(0), CapacityUnits(1),
                                    CapacityUnits(0)}}}};
  declaration.requesters = {RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 1;
  declaration.timingContracts = {
      TimingContractDeclaration{TimingContractKey(0), {0}}};
  declaration.usePatterns = {
      UsePatternDeclaration{UsePatternKey(0),
                            RequesterKey(0),
                            EligibilityKey(0),
                            EventKey(0),
                            EventKey(0),
                            std::nullopt,
                            TimingContractKey(0),
                            {},
                            {InternalTransactionDeclaration{{}}}}};
  return take("resource contract", ResourceContract::create(declaration));
}

MemoryCapabilityAlternativeRecord hybridAlternative() {
  MemoryActorContractClause plain = LoadStorePlainContractClause{{false}};
  MemoryActorContractDomain actors =
      take("actor domain", MemoryActorContractDomain::create(
                               OperationSchemaId::DataflowLoad, {plain}));
  ParameterizedMemoryAccessDomain accesses =
      take("access domain", ParameterizedMemoryAccessDomain::create(
                                {elementAccess(), vectorAccess()}));
  return MemoryCapabilityAlternativeRecord{std::move(actors),
                                           {{ServiceValueRole::Address, 0},
                                            {ServiceValueRole::Mask, 1},
                                            {ServiceValueRole::Control, 2},
                                            {ServiceValueRole::Data, 3},
                                            {ServiceValueRole::Completion, 4}},
                                           std::move(accesses),
                                           {UsePatternKey(0)}};
}

MemoryCapabilityAlternativeRecord
alternativeForAccess(MemoryAccessClass access) {
  MemoryActorContractClause plain = LoadStorePlainContractClause{{false}};
  auto actors =
      take("split actor domain", MemoryActorContractDomain::create(
                                     OperationSchemaId::DataflowLoad, {plain}));
  auto accesses = take("split access domain",
                       ParameterizedMemoryAccessDomain::create({access}));
  return MemoryCapabilityAlternativeRecord{std::move(actors),
                                           {{ServiceValueRole::Address, 0},
                                            {ServiceValueRole::Mask, 1},
                                            {ServiceValueRole::Control, 2},
                                            {ServiceValueRole::Data, 3},
                                            {ServiceValueRole::Completion, 4}},
                                           std::move(accesses),
                                           {UsePatternKey(0)}};
}

mlir::Operation *findActor(mlir::ModuleOp module, llvm::StringRef name) {
  mlir::Operation *actor = nullptr;
  module.walk([&](mlir::func::FuncOp function) {
    if (function.getSymName() != name)
      return;
    function.walk([&](LoadOp load) { actor = load; });
  });
  if (!actor)
    fail(name, "actor was not found");
  return actor;
}

struct ActorProjection {
  CanonicalActorSchemaProjection actor;
  CanonicalService service;
  CanonicalMemoryAccessView access;
};

ActorProjection project(mlir::ModuleOp module, llvm::StringRef name) {
  mlir::Operation *operation = findActor(module, name);
  return ActorProjection{
      take(name, projectRegisteredActorSchemaProjection(operation)),
      take(name, CanonicalService::forActor(operation)),
      take(name, getCanonicalMemoryAccessView(operation))};
}

std::vector<MemoryTransportEndpointDescriptor>
hybridEndpoints(mlir::MLIRContext &context) {
  mlir::FunctionType type = mlir::FunctionType::get(
      &context,
      {BitsType::get(&context, 32), BitsType::get(&context, 4),
       BitsType::get(&context, 0)},
      {BitsType::get(&context, 128), BitsType::get(&context, 0)});
  return take("endpoint inventory",
              deriveMemoryTransportEndpointInventory(type));
}

MemoryOperationPortRecord hybridPort(mlir::MLIRContext &context) {
  MemoryOperationPortDeclaration declaration{
      {0, 1, 2, 3, 4},
      directContract(),
      {{MemoryPortTransactionProjection::Direct}},
      {hybridAlternative()}};
  return take("hybrid port",
              MemoryOperationPortRecord::create(&context, Schedule::Spatial,
                                                hybridEndpoints(context),
                                                std::move(declaration)));
}

void checkHybridGeometry(mlir::ModuleOp module, mlir::MLIRContext &context) {
  MemoryOperationPortRecord port = hybridPort(context);
  for (llvm::StringRef accepted : {"element_f32", "vector4_f32"}) {
    ActorProjection actor = project(module, accepted);
    auto matches =
        take(accepted, port.matchingCapabilities(actor.actor, actor.service,
                                                 actor.access));
    require(accepted,
            matches.size() == 1 && matches.front().alternativeOrdinal == 0 &&
                matches.front().admissibleUsePatterns.size() == 1 &&
                matches.front().admissibleUsePatterns.front().ordinal() == 0,
            "the hybrid port did not admit its declared geometry");
  }

  ActorProjection rejected = project(module, "vector2_f64");
  auto matches = take(
      "vector2_f64", port.matchingCapabilities(rejected.actor, rejected.service,
                                               rejected.access));
  require("vector2_f64", matches.empty(),
          "equal total width erased element and lane geometry");

  auto bytes = take("encode", encodeMemoryOperationPortRecord(port));
  MemoryOperationPortRecord decoded =
      take("decode",
           decodeMemoryOperationPortRecord(bytes, &context, Schedule::Spatial,
                                           hybridEndpoints(context)));
  auto roundTrip = take("reencode", encodeMemoryOperationPortRecord(decoded));
  require("roundtrip", bytes == roundTrip,
          "strict memory-operation port import changed canonical bytes");
  bytes.push_back(0);
  expectRejected<MemoryOperationPortRecord>(
      "trailing bytes",
      decodeMemoryOperationPortRecord(bytes, &context, Schedule::Spatial,
                                      hybridEndpoints(context)));
}

void checkRoleDirection(mlir::MLIRContext &context) {
  MemoryCapabilityAlternativeRecord alternative = hybridAlternative();
  alternative.roleToEndpoint[3].endpointOrdinal = 0;
  MemoryOperationPortDeclaration declaration{
      {0, 1, 2, 3, 4},
      directContract(),
      {{MemoryPortTransactionProjection::Direct}},
      {std::move(alternative)}};
  expectRejected<MemoryOperationPortRecord>(
      "role direction", MemoryOperationPortRecord::create(
                            &context, Schedule::Spatial,
                            hybridEndpoints(context), std::move(declaration)));
}

void checkCompleteRelationNormalization(mlir::MLIRContext &context) {
  MemoryOperationPortDeclaration split{
      {0, 1, 2, 3, 4},
      directContract(),
      {{MemoryPortTransactionProjection::Direct}},
      {alternativeForAccess(elementAccess()),
       alternativeForAccess(vectorAccess())}};
  MemoryOperationPortRecord normalized =
      take("normalize split relation",
           MemoryOperationPortRecord::create(&context, Schedule::Spatial,
                                             hybridEndpoints(context), split));
  auto normalizedBytes = take("normalized relation bytes",
                              encodeMemoryOperationPortRecord(normalized));
  auto hybridBytes = take("hybrid relation bytes",
                          encodeMemoryOperationPortRecord(hybridPort(context)));
  require("relation normalization", normalizedBytes == hybridBytes,
          "equivalent capability decompositions did not normalize equally");
  expectRejected<MemoryOperationPortRecord>(
      "strict split relation", MemoryOperationPortRecord::fromCanonical(
                                   &context, Schedule::Spatial,
                                   hybridEndpoints(context), std::move(split)));
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<DataflowDialect, FabricDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(fixture, &context);
  if (!module)
    fail("fixture", "failed to parse memory actor fixture");

  checkHybridGeometry(*module, context);
  checkRoleDirection(context);
  checkCompleteRelationNormalization(context);
  return EXIT_SUCCESS;
}

#include "Fabric/IR/MemoryOperationPort.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FabricTypes.h"
#include "Fabric/IR/MemoryConnectivityContract.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
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

MemoryAddressDomain rootRelativeAddress(std::uint64_t indexBits) {
  return take("root-relative address domain",
              MemoryAddressDomain::rootRelative(singleton(indexBits)));
}

MemoryAddressDomain pointerAddress(std::uint32_t representationBits) {
  return take(
      "pointer-addressed domain",
      MemoryAddressDomain::pointerAddressed(PointerFormatRelation::get(
          {{0, representationBits, representationBits,
            loom::PointerLayoutKind::StableIntegral}})));
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
                  noWrite(), rootRelativeAddress(32)));
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
          allAlignments(), readDomain(ReadSubwordSemantics::Exact), noWrite(),
          rootRelativeAddress(32)));
}

MemoryAccessClass vectorAccess(MemoryAccessForm form,
                               MemoryAddressDomain addressDomain,
                               std::uint64_t maximumLanes) {
  return take(
      "parameterized vector access",
      MemoryAccessClass::create(
          form, singleton(32),
          take("lane domain", UnsignedDomain::fromCanonical(
                                  {UnsignedInterval{2, maximumLanes}})),
          {MaskInactivePair{MemoryMaskForm::Absent,
                            InactiveLaneSemantics::NotApplicable},
           MaskInactivePair{MemoryMaskForm::Dynamic,
                            InactiveLaneSemantics::SuppressAndZeroFill}},
          allAlignments(), readDomain(ReadSubwordSemantics::ZeroExtend),
          noWrite(), std::move(addressDomain)));
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

ResourceContract committedDirectContract() {
  ResourceContractDeclaration declaration = directContract().declaration();
  declaration.resourceTransitions = {ResourceTransitionKey(0)};
  declaration.usePatterns.front().commit =
      CommitDeclaration{EventKey(0), ResourceTransitionKey(0)};
  return take("committed resource contract",
              ResourceContract::create(declaration));
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
hybridEndpoints(mlir::MLIRContext &context,
                std::uint32_t addressPayloadBits = 32) {
  mlir::FunctionType type = mlir::FunctionType::get(
      &context,
      {BitsType::get(&context, addressPayloadBits), BitsType::get(&context, 4),
       BitsType::get(&context, 0)},
      {BitsType::get(&context, 128), BitsType::get(&context, 0)});
  return take("endpoint inventory",
              deriveMemoryTransportEndpointInventory(type));
}

MemoryOperationPortDeclaration declarationForAccess(MemoryAccessClass access) {
  return {{0, 1, 2, 3, 4},
          directContract(),
          {{MemoryPortTransactionProjection::Direct}},
          {alternativeForAccess(std::move(access))}};
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

MemoryConnectivityContractAttr managerConnectivity(mlir::MLIRContext &context) {
  MemoryConnectivityDeclaration declaration;
  declaration.operationPorts = {
      {{{MemoryDispatchTarget(std::in_place_type<ManagerMemoryDispatchTarget>,
                              ManagerMemoryDispatchTarget{0})}}}};
  declaration.subordinateEndpoints = {
      {1,
       {},
       MemoryProviderAddressTransform::None,
       {MemoryDispatchTarget(std::in_place_type<ManagerMemoryDispatchTarget>,
                             ManagerMemoryDispatchTarget{0})}}};
  MemoryConnectivityContractRecord record =
      take("memory connectivity",
           MemoryConnectivityContractRecord::create(std::move(declaration)));
  auto bytes = take("memory connectivity encoding",
                    encodeMemoryConnectivityContractRecord(record));
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return MemoryConnectivityContractAttr::get(
      &context, mlir::DenseI8ArrayAttr::get(&context, signedBytes));
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

void checkIssueTiming(mlir::MLIRContext &context) {
  MemoryOperationPortDeclaration declaration{
      {0, 1, 2, 3, 4},
      committedDirectContract(),
      {{MemoryPortTransactionProjection::Direct}},
      {hybridAlternative()}};
  MemoryOperationPortRecord committed = take(
      "committed issue timing",
      MemoryOperationPortRecord::create(&context, Schedule::Spatial,
                                        hybridEndpoints(context),
                                        std::move(declaration)));
  auto issue = take("project committed issue timing",
                    projectMemoryOperationIssueLatency(committed,
                                                       UsePatternKey(0)));
  require("committed issue timing", issue && *issue == 0,
          "commit was not the memory operation issue event");

  MemoryOperationPortRecord uncommitted = hybridPort(context);
  auto absent = take("project absent issue timing",
                     projectMemoryOperationIssueLatency(uncommitted,
                                                        UsePatternKey(0)));
  require("absent issue timing", !absent,
          "resource release was substituted for absent issue timing");
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

void checkAddressPayloadCapacity(mlir::MLIRContext &context) {
  auto create = [&](MemoryAccessClass access, std::uint32_t addressBits) {
    return MemoryOperationPortRecord::create(
        &context, Schedule::Spatial, hybridEndpoints(context, addressBits),
        declarationForAccess(std::move(access)));
  };

  take("indexed RootRelative address capacity",
       create(vectorAccess(MemoryAccessForm::Indexed,
                           rootRelativeAddress(64), 2),
              128));
  expectRejected<MemoryOperationPortRecord>(
      "indexed RootRelative address overflow",
      create(vectorAccess(MemoryAccessForm::Indexed,
                          rootRelativeAddress(64), 3),
             128));
  take("contiguous RootRelative address capacity",
       create(vectorAccess(MemoryAccessForm::Contiguous,
                           rootRelativeAddress(64), 4),
              64));

  take("indexed pointer address capacity",
       create(vectorAccess(MemoryAccessForm::Indexed, pointerAddress(64), 2),
              128));
  expectRejected<MemoryOperationPortRecord>(
      "indexed pointer address overflow",
      create(vectorAccess(MemoryAccessForm::Indexed, pointerAddress(64), 3),
             128));
}

void checkConnectivityContract(mlir::MLIRContext &context) {
  MemoryConnectivityDeclaration declaration;
  declaration.operationPorts = {
      {{{MemoryDispatchTarget(std::in_place_type<ManagerMemoryDispatchTarget>,
                              ManagerMemoryDispatchTarget{0})}}}};
  MemoryConnectivityContractRecord record =
      take("connectivity contract",
           MemoryConnectivityContractRecord::create(declaration));
  auto bytes = take("connectivity encoding",
                    encodeMemoryConnectivityContractRecord(record));
  MemoryConnectivityContractRecord decoded = take(
      "connectivity decoding", decodeMemoryConnectivityContractRecord(bytes));
  require("connectivity roundtrip",
          take("connectivity reencoding",
               encodeMemoryConnectivityContractRecord(decoded)) == bytes,
          "strict import changed canonical connectivity bytes");

  llvm::Error wrongOwner = validateMemoryConnectivityContract(
      decoded, {hybridPort(context)}, hybridEndpoints(context), 0, 0, false);
  require("connectivity owner", static_cast<bool>(wrongOwner),
          "accepted an unknown manager target");
  llvm::consumeError(std::move(wrongOwner));

  MemoryConnectivityDeclaration reordered;
  reordered.operationPorts = {
      {{{MemoryDispatchTarget(std::in_place_type<ManagerMemoryDispatchTarget>,
                              ManagerMemoryDispatchTarget{1}),
         MemoryDispatchTarget(std::in_place_type<ManagerMemoryDispatchTarget>,
                              ManagerMemoryDispatchTarget{0})}}}};
  expectRejected<MemoryConnectivityContractRecord>(
      "connectivity order",
      MemoryConnectivityContractRecord::fromCanonical(std::move(reordered)));
}

void checkExactFabricCarrier(mlir::MLIRContext &context) {
  const llvm::StringRef test = "exact fabric.mem carrier";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
        module {
          fabric.module @memory_engine(
              %manager: memref<?x!fabric.bits<128>>,
              %address: !fabric.bits<32>,
              %mask: !fabric.bits<4>,
              %ctrl: !fabric.bits<0>) {
            %filtered = fabric.fifo %address
                [max_depth = 1, bypassable = false] : !fabric.bits<32>
            fabric.yield
          }
        }
      )mlir",
                                              &context);
  require(test, static_cast<bool>(module), "failed to parse carrier fixture");

  fabric::ModuleOp root = *module->getOps<fabric::ModuleOp>().begin();
  mlir::Block &body = root.getBody().front();
  auto fifo = *body.getOps<fabric::FifoOp>().begin();
  mlir::OpBuilder builder(body.getTerminator());

  auto bytes = take(test, encodeMemoryOperationPortRecord(hybridPort(context)));
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  auto records = builder.getArrayAttr(
      {mlir::DenseI8ArrayAttr::get(&context, signedBytes)});
  auto managers = mlir::DenseI32ArrayAttr::get(&context, {0});
  auto subordinates = mlir::DenseI32ArrayAttr::get(&context, {0});
  auto contract = fabric::MemoryContractAttr::get(
      &context,
      fabric::MemoryEngineAttr::get(&context, fabric::Schedule::Spatial,
                                    fabric::MemoryResidentContextsAttr()),
      fabric::LocalMemoryServiceAttr(), managerConnectivity(context), managers,
      subordinates);
  llvm::SmallVector<mlir::Value> inputs = {
      body.getArgument(0), fifo.getResult(), body.getArgument(2),
      body.getArgument(3)};
  auto memoryOp = fabric::MemOp::create(
      builder, root.getLoc(),
      mlir::TypeRange{body.getArgument(0).getType(),
                      fabric::BitsType::get(&context, 128),
                      fabric::BitsType::get(&context, 0)},
      inputs, mlir::StringAttr(), mlir::TypeAttr(), contract,
      llvm::ArrayRef<mlir::Type>{}, mlir::ArrayAttr(), records);
  root.setFunctionType(
      mlir::FunctionType::get(&context, root.getFunctionType().getInputs(),
                              mlir::TypeRange{body.getArgument(0).getType()}));
  body.getTerminator()->setOperands(memoryOp.getResult(0));

  require(test, succeeded(mlir::verify(*module)),
          "exact memory-operation inventory did not verify");
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  module->print(stream);
  stream.flush();
  require(test, llvm::StringRef(printed).contains("capabilities [array<i8:"),
          "custom printer omitted the exact capability inventory");
  mlir::OwningOpRef<mlir::ModuleOp> roundTrip =
      mlir::parseSourceString<mlir::ModuleOp>(printed, &context);
  require(test, static_cast<bool>(roundTrip),
          "printed exact memory operation did not parse");

  llvm::SmallString<128> storePath;
  if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
          "loom-memory-operation-port", storePath))
    fail(test, error.message());
  llvm::scope_exit cleanup(
      [&] { llvm::sys::fs::remove_directories(storePath); });
  loom::ArtifactStore store(storePath);
  auto finalized = take(test, loom::fabric::finalizeFabricRoot(root, store));

  std::optional<loom::fabric::FabricMemoryOccurrenceRef> memory;
  for (std::uint64_t id = 0;; ++id) {
    auto kind = finalized.view().entityKind(id);
    if (!kind)
      break;
    if (*kind == loom::fabric::FabricEntityKind::FabricMemoryOccurrence)
      memory = loom::fabric::FabricMemoryOccurrenceRef(id);
  }
  require(test, memory.has_value(),
          "finalized Fabric omitted its memory occurrence");
  auto portRefs = finalized.view().memoryOperationPorts(*memory);
  require(test, portRefs.size() == 1,
          "finalized Fabric changed the operation-port inventory");
  const loom::fabric::MemoryOperationPortView *port =
      finalized.view().memoryOperationPort(portRefs.front());
  require(
      test,
      port && port->capabilityAlternatives().size() == 1 &&
          finalized.view().memoryCapabilityAlternative({portRefs.front(), 0}),
      "finalized Fabric omitted the exact operation-port relation");
  const MemoryConnectivityContractRecord *connectivity =
      finalized.view().memoryConnectivity(*memory);
  require(test,
          connectivity && connectivity->operationPorts().size() == 1 &&
              connectivity->operationPorts()
                      .front()
                      .capabilityTargetDomains.size() == 1,
          "finalized Fabric omitted its memory connectivity owner");
  auto transportOwner =
      loom::fabric::FabricTransportEndpointOwnerRef::of(*memory);
  auto memoryOwner = loom::fabric::FabricMemoryEndpointOwnerRef::of(*memory);
  require(test,
          finalized.view().transportEndpointCount(transportOwner) == 5 &&
              finalized.view().memoryEndpointCount(memoryOwner) == 2,
          "token and memory endpoint inventories were not separated");
  const auto memoryAttachments =
      finalized.view().moduleBoundaryMemoryAttachments();
  require(test, memoryAttachments.size() == 2,
          "module memory boundary did not retain both capability paths");
  const auto hasAttachment = [&](loom::fabric::FabricPortDirection direction,
                                 loom::fabric::FabricOrdinal boundaryOrdinal,
                                 loom::fabric::FabricOrdinal endpointOrdinal) {
    return llvm::any_of(memoryAttachments, [&](const auto &attachment) {
      return attachment.boundary ==
                 loom::fabric::FabricModuleBoundaryEndpointRef{
                     *finalized.view().moduleRootTemplate(), direction,
                     boundaryOrdinal} &&
             attachment.endpoint == loom::fabric::FabricMemoryEndpointRef{
                                        memoryOwner, endpointOrdinal};
    });
  };
  require(test, hasAttachment(loom::fabric::FabricPortDirection::Input, 0, 0),
          "module memory input changed its exact manager endpoint");
  require(test, hasAttachment(loom::fabric::FabricPortDirection::Output, 0, 1),
          "module memory output changed its exact subordinate endpoint");
  bool fifoFeedsAddress = false;
  for (const loom::fabric::FabricPointConnectionPayload &connection :
       finalized.view().pointConnections())
    fifoFeedsAddress |= connection.destination.owner == transportOwner &&
                        connection.destination.ordinal == 0;
  require(test, fifoFeedsAddress,
          "memory token endpoint ordinal retained the manager memref offset");
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<DataflowDialect, FabricDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry,
                            mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(fixture, &context);
  if (!module)
    fail("fixture", "failed to parse memory actor fixture");

  checkHybridGeometry(*module, context);
  checkIssueTiming(context);
  checkRoleDirection(context);
  checkCompleteRelationNormalization(context);
  checkAddressPayloadCapacity(context);
  checkConnectivityContract(context);
  checkExactFabricCarrier(context);
  return EXIT_SUCCESS;
}

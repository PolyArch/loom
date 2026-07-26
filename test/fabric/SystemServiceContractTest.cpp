#include "Fabric/IR/SystemServiceContract.h"

#include "Fabric/IR/FabricDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace loom::fabric;

namespace {

[[noreturn]] void fail(llvm::StringRef test, const llvm::Twine &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectRejected(llvm::StringRef test, llvm::Expected<T> value) {
  if (value)
    fail(test, "unexpectedly accepted");
  llvm::consumeError(value.takeError());
}

std::string denseI8Assembly(mlir::MLIRContext &context,
                            llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  std::string text;
  llvm::raw_string_ostream stream(text);
  mlir::DenseI8ArrayAttr::get(&context, signedBytes).print(stream);
  stream.flush();
  return text;
}

::fabric::UnsignedDomain singleton(std::uint64_t lower, std::uint64_t upper) {
  return take("unsigned domain",
              ::fabric::UnsignedDomain::fromCanonical({{lower, upper}}));
}

::fabric::MemoryAccessClass elementAccess() {
  auto alignment =
      take("alignment", ::fabric::AlignmentDomain::create(singleton(0, 0)));
  auto reads = take(
      "read semantics",
      ::fabric::ClosedEnumDomain<::fabric::ReadSubwordSemantics>::fromCanonical(
          {::fabric::ReadSubwordSemantics::Exact}));
  auto writes =
      take("write semantics",
           ::fabric::ClosedEnumDomain<::fabric::WriteSubwordSemantics>::
               fromCanonical({::fabric::WriteSubwordSemantics::Exact}));
  return take("element access",
              ::fabric::MemoryAccessClass::create(
                  dataflow::semantics::MemoryAccessForm::Element,
                  singleton(32, 32), singleton(1, 1),
                  {{dataflow::semantics::MemoryMaskForm::Absent,
                    ::fabric::InactiveLaneSemantics::NotApplicable}},
                  std::move(alignment), std::move(reads), std::move(writes)));
}

::fabric::ParameterizedMemoryAccessDomain accessDomain() {
  return take(
      "access domain",
      ::fabric::ParameterizedMemoryAccessDomain::create({elementAccess()}));
}

dataflow::SyncScopeProjection systemScope() {
  return {dataflow::SyncScopeKind::System, {}, {}};
}

::fabric::MemoryActorContractDomain
actorDomain(dataflow::OperationSchemaId schema) {
  using Clause = ::fabric::MemoryActorContractClause;
  switch (schema) {
  case dataflow::OperationSchemaId::DataflowLoad:
  case dataflow::OperationSchemaId::DataflowStore:
    return take(
        "plain actor domain",
        ::fabric::MemoryActorContractDomain::create(
            schema, {Clause(::fabric::LoadStorePlainContractClause{{false}})}));
  case dataflow::OperationSchemaId::DataflowAtomicRmw:
    return take("RMW actor domain",
                ::fabric::MemoryActorContractDomain::create(
                    schema, {Clause(::fabric::AtomicRmwContractClause{
                                {dataflow::AtomicRmwKind::Add},
                                {dataflow::AtomicOrdering::Monotonic},
                                {systemScope()},
                                {std::nullopt},
                                {false}})}));
  case dataflow::OperationSchemaId::DataflowCmpXchg:
    return take("compare-exchange actor domain",
                ::fabric::MemoryActorContractDomain::create(
                    schema, {Clause(::fabric::CompareExchangeContractClause{
                                {{dataflow::AtomicOrdering::Monotonic,
                                  dataflow::AtomicOrdering::Monotonic}},
                                {systemScope()},
                                {std::nullopt},
                                {false},
                                {false}})}));
  case dataflow::OperationSchemaId::DataflowFence:
    return take("fence actor domain",
                ::fabric::MemoryActorContractDomain::create(
                    schema, {Clause(::fabric::FenceContractClause{
                                {dataflow::AtomicOrdering::SeqCst},
                                {systemScope()}})}));
  default:
    fail("actor domain", "unexpected operation schema");
  }
}

ServiceRateContractRecord rate() {
  ClockDomainRef clock(HardwareDomainRef(90));
  return take(
      "service rate",
      ServiceRateContractRecord::create(
          clock, 1, 1, 8,
          ServiceProgress(std::in_place_type<::fabric::BoundedCompletion>,
                          ::fabric::BoundedCompletion{clock, 16})));
}

AddressedMemoryCapabilityDomain addressed(dataflow::OperationSchemaId schema,
                                          bool consistent) {
  std::optional<MemoryConsistencyDomainRef> domain;
  if (consistent)
    domain = MemoryConsistencyDomainRef(HardwareDomainRef(91));
  return take("addressed capability", AddressedMemoryCapabilityDomain::create(
                                          actorDomain(schema), accessDomain(),
                                          singleton(0, 4095), 128, domain));
}

CanonicalServiceCapabilityRecord capability(
    dataflow::semantics::ServiceKind kind,
    CanonicalServiceEndpointRole role = CanonicalServiceEndpointRole::Serve) {
  using Kind = dataflow::semantics::ServiceKind;
  CanonicalServiceCapabilityDomain domain = [&]() {
    switch (kind) {
    case Kind::MessageTransfer:
      fail("service capability", "message capability requires its context");
    case Kind::MemoryRead:
      return CanonicalServiceCapabilityDomain(
          addressed(dataflow::OperationSchemaId::DataflowLoad, false));
    case Kind::MemoryWrite:
      return CanonicalServiceCapabilityDomain(
          addressed(dataflow::OperationSchemaId::DataflowStore, false));
    case Kind::MemoryAtomicRmw:
      return CanonicalServiceCapabilityDomain(
          addressed(dataflow::OperationSchemaId::DataflowAtomicRmw, true));
    case Kind::MemoryCompareExchange:
      return CanonicalServiceCapabilityDomain(
          addressed(dataflow::OperationSchemaId::DataflowCmpXchg, true));
    case Kind::MemoryFence:
      return CanonicalServiceCapabilityDomain(
          take("fence capability",
               FenceCapabilityDomain::create(
                   actorDomain(dataflow::OperationSchemaId::DataflowFence),
                   MemoryConsistencyDomainRef(HardwareDomainRef(91)))));
    }
    std::abort();
  }();
  return take("service capability", CanonicalServiceCapabilityRecord::create(
                                        kind, role, std::move(domain), rate()));
}

void checkCapabilityCatalog(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral test = "Canonical Service capability catalog";
  using Kind = dataflow::semantics::ServiceKind;
  CanonicalServiceCapabilityRecord message = take(
      test, CanonicalServiceCapabilityRecord::create(
                Kind::MessageTransfer, CanonicalServiceEndpointRole::Initiate,
                take(test, MessageTransferCapabilityDomain::create(
                               {mlir::VectorType::get(
                                   {4}, mlir::Float32Type::get(&context))})),
                rate()));
  CanonicalServiceCapabilitySet messageCapabilities =
      take(test, CanonicalServiceCapabilitySet::create({std::move(message)}));

  std::vector<CanonicalServiceCapabilityRecord> records;
  for (Kind kind : {Kind::MemoryRead, Kind::MemoryWrite, Kind::MemoryAtomicRmw,
                    Kind::MemoryCompareExchange, Kind::MemoryFence})
    records.push_back(capability(kind));

  CanonicalServiceCapabilitySet capabilities =
      take(test, CanonicalServiceCapabilitySet::create(std::move(records)));
  std::vector<std::uint8_t> encoded =
      take(test, encodeCanonicalServiceCapabilitySet(capabilities));
  CanonicalServiceCapabilitySet decoded =
      take(test, decodeCanonicalServiceCapabilitySet(encoded, &context));
  if (take(test, encodeCanonicalServiceCapabilitySet(decoded)) != encoded)
    fail(test, "strict roundtrip changed capability bytes");
  if (decoded.capabilities().size() != 5)
    fail(test, "closed memory-service catalog lost a capability");

  expectRejected(test,
                 CanonicalServiceCapabilityRecord::create(
                     Kind::MemoryRead, CanonicalServiceEndpointRole::Serve,
                     take(test, MessageTransferCapabilityDomain::create(
                                    {mlir::IntegerType::get(&context, 32)})),
                     rate()));

  CanonicalServiceCapabilityRecord messageI32 =
      take(test, CanonicalServiceCapabilityRecord::create(
                     Kind::MessageTransfer, CanonicalServiceEndpointRole::Serve,
                     take(test, MessageTransferCapabilityDomain::create(
                                    {mlir::IntegerType::get(&context, 32)})),
                     rate()));
  CanonicalServiceCapabilityRecord messageI64 =
      take(test, CanonicalServiceCapabilityRecord::create(
                     Kind::MessageTransfer, CanonicalServiceEndpointRole::Serve,
                     take(test, MessageTransferCapabilityDomain::create(
                                    {mlir::IntegerType::get(&context, 64)})),
                     rate()));
  expectRejected(test, CanonicalServiceCapabilitySet::create(
                           {std::move(messageI32), std::move(messageI64)}));

  CanonicalServiceCapabilityRecord mixedMessage =
      take(test, CanonicalServiceCapabilityRecord::create(
                     Kind::MessageTransfer, CanonicalServiceEndpointRole::Serve,
                     take(test, MessageTransferCapabilityDomain::create(
                                    {mlir::IntegerType::get(&context, 32)})),
                     rate()));
  expectRejected(test,
                 CanonicalServiceCapabilitySet::create(
                     {std::move(mixedMessage), capability(Kind::MemoryRead)}));
  expectRejected(test,
                 CanonicalServiceCapabilitySet::create(
                     {capability(Kind::MemoryRead),
                      capability(Kind::MemoryWrite,
                                 CanonicalServiceEndpointRole::Initiate)}));

  std::vector<std::uint8_t> messageBytes =
      take(test, encodeCanonicalServiceCapabilitySet(messageCapabilities));
  SystemServiceEndpointOwnerRef endpointOwner =
      take(test, SystemServiceEndpointOwnerRef::create(
                     FabricInventoryOwnerRef::of(AccCoreOccurrenceRef(12))));
  std::vector<std::uint8_t> ownerBytes =
      encodeSystemServiceEndpointOwnerRef(endpointOwner);
  const std::string validMessage =
      "module { fabric.system @soc { "
      "fabric.system.service_endpoint owner = " +
      denseI8Assembly(context, ownerBytes) +
      " capabilities = " + denseI8Assembly(context, messageBytes) +
      " carrier = !fabric.bits<128> "
      "{entity_id = #fabric.entity_id<30>} } }";
  mlir::OwningOpRef<mlir::ModuleOp> messageModule =
      mlir::parseSourceString<mlir::ModuleOp>(validMessage, &context);
  if (!messageModule || mlir::failed(mlir::verify(*messageModule)))
    fail(test, "valid message service endpoint did not verify");

  const std::string missingCarrier =
      "module { fabric.system @soc { "
      "fabric.system.service_endpoint owner = " +
      denseI8Assembly(context, ownerBytes) +
      " capabilities = " + denseI8Assembly(context, messageBytes) +
      " {entity_id = #fabric.entity_id<30>} } }";
  mlir::OwningOpRef<mlir::ModuleOp> missingCarrierModule =
      mlir::parseSourceString<mlir::ModuleOp>(missingCarrier, &context);
  if (missingCarrierModule &&
      mlir::succeeded(mlir::verify(*missingCarrierModule)))
    fail(test, "message service endpoint accepted no carrier");

  const std::string narrowCarrier =
      "module { fabric.system @soc { "
      "fabric.system.service_endpoint owner = " +
      denseI8Assembly(context, ownerBytes) +
      " capabilities = " + denseI8Assembly(context, messageBytes) +
      " carrier = !fabric.bits<64> "
      "{entity_id = #fabric.entity_id<30>} } }";
  mlir::OwningOpRef<mlir::ModuleOp> narrowCarrierModule =
      mlir::parseSourceString<mlir::ModuleOp>(narrowCarrier, &context);
  if (narrowCarrierModule &&
      mlir::succeeded(mlir::verify(*narrowCarrierModule)))
    fail(test, "message service endpoint accepted a narrow carrier");

  const std::string memoryWithCarrier =
      "module { fabric.system @soc { "
      "fabric.system.service_endpoint owner = " +
      denseI8Assembly(context, ownerBytes) +
      " capabilities = " + denseI8Assembly(context, encoded) +
      " carrier = !fabric.bits<128> "
      "{entity_id = #fabric.entity_id<30>} } }";
  mlir::OwningOpRef<mlir::ModuleOp> memoryWithCarrierModule =
      mlir::parseSourceString<mlir::ModuleOp>(memoryWithCarrier, &context);
  if (memoryWithCarrierModule &&
      mlir::succeeded(mlir::verify(*memoryWithCarrierModule)))
    fail(test, "memory service endpoint accepted a message carrier");
}

void checkOwnerAndTransform(mlir::MLIRContext &context) {
  constexpr llvm::StringLiteral test = "System service owner and transform";
  SystemServiceEndpointOwnerRef owner =
      take(test, SystemServiceEndpointOwnerRef::create(
                     FabricInventoryOwnerRef::of(AccCoreOccurrenceRef(12))));
  std::vector<std::uint8_t> encodedOwner =
      encodeSystemServiceEndpointOwnerRef(owner);
  take(test, decodeSystemServiceEndpointOwnerRef(encodedOwner));
  expectRejected(test,
                 SystemServiceEndpointOwnerRef::create(
                     FabricInventoryOwnerRef::of(FabricPeOccurrenceRef(12))));

  FabricMemoryEndpointRef input{
      FabricMemoryEndpointOwnerRef::of(SystemServiceEndpointRef(30)), 0};
  SystemServiceTransformRecord interleave = take(
      test,
      SystemServiceTransformRecord::create(
          {input},
          {{FabricMemoryEndpointOwnerRef::of(SystemServiceEndpointRef(32)), 0},
           {FabricMemoryEndpointOwnerRef::of(SystemServiceEndpointRef(33)), 0}},
          StaticInterleaveTransform{64, 2}));
  std::vector<std::uint8_t> encoded =
      take(test, encodeSystemServiceTransformRecord(interleave));
  SystemServiceTransformRecord decoded =
      take(test, decodeSystemServiceTransformRecord(encoded));
  if (take(test, encodeSystemServiceTransformRecord(decoded)) != encoded)
    fail(test, "strict roundtrip changed transform bytes");

  expectRejected(
      test,
      SystemServiceTransformRecord::create(
          {input},
          {{FabricMemoryEndpointOwnerRef::of(SystemServiceEndpointRef(32)), 0}},
          StaticInterleaveTransform{64, 2}));

  FabricMemoryServiceRef memory =
      FabricMemoryServiceRef::system(SystemMemoryServiceRef(20));
  SystemServiceTransformRecord coherent = take(
      test,
      SystemServiceTransformRecord::create(
          {input},
          {{FabricMemoryEndpointOwnerRef::of(SystemServiceEndpointRef(32)), 0}},
          CoherentMemoryTransform{
              MemoryConsistencyDomainRef(HardwareDomainRef(91)),
              {{{memory, 0}, {memory, 1}}}}));
  std::vector<std::uint8_t> coherentBytes =
      take(test, encodeSystemServiceTransformRecord(coherent));
  SystemServiceTransformRecord coherentDecoded =
      take(test, decodeSystemServiceTransformRecord(coherentBytes));
  if (take(test, encodeSystemServiceTransformRecord(coherentDecoded)) !=
      coherentBytes)
    fail(test, "CoherentMemory roundtrip changed region correspondence");

  CanonicalServiceCapabilitySet capabilities = take(
      test,
      CanonicalServiceCapabilitySet::create({take(
          test, CanonicalServiceCapabilityRecord::create(
                    dataflow::semantics::ServiceKind::MessageTransfer,
                    CanonicalServiceEndpointRole::Initiate,
                    take(test, MessageTransferCapabilityDomain::create(
                                   {mlir::IntegerType::get(&context, 32)})),
                    rate()))}));
  std::vector<std::uint8_t> capabilityBytes =
      take(test, encodeCanonicalServiceCapabilitySet(capabilities));
  SystemServiceEndpointOwnerRef externalOwner =
      take(test, SystemServiceEndpointOwnerRef::create(
                     FabricInventoryOwnerRef::of(ExternalBoundaryRef(40))));
  const std::string source =
      "module {\n"
      "  fabric.system @soc {\n"
      "    fabric.system.service_endpoint owner = " +
      denseI8Assembly(context, encodedOwner) +
      " capabilities = " + denseI8Assembly(context, capabilityBytes) +
      " carrier = !fabric.bits<32>" +
      " {entity_id = #fabric.entity_id<30>}\n"
      "    fabric.system.service_transform contract = " +
      denseI8Assembly(context, encoded) +
      " {entity_id = #fabric.entity_id<31>}\n"
      "    fabric.system.external_boundary "
      "{entity_id = #fabric.entity_id<40>}\n"
      "    fabric.system.service_endpoint owner = " +
      denseI8Assembly(context,
                      encodeSystemServiceEndpointOwnerRef(externalOwner)) +
      " capabilities = " + denseI8Assembly(context, capabilityBytes) +
      " carrier = !fabric.bits<32> "
      "{entity_id = #fabric.entity_id<41>}\n"
      "  }\n"
      "}\n";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  if (!module || mlir::failed(mlir::verify(*module)))
    fail(test, "typed service operations did not parse and verify");
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<::fabric::FabricDialect>();
  mlir::MLIRContext context(registry);
  checkCapabilityCatalog(context);
  checkOwnerAndTransform(context);
  return EXIT_SUCCESS;
}

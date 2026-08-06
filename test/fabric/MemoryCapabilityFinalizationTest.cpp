#include "Fabric/IR/MemoryCapabilityFinalization.h"
#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryServiceContract.h"

#include "Fabric/IR/FabricDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace fabric;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void expectInvalid(const char *test, llvm::Error error,
                   MemoryCapabilityFinalizationReason expectedReason,
                   const char *expectedText) {
  if (!error)
    fail(test, "accepted an unavailable persistent contract");

  std::optional<MemoryCapabilityFinalizationReason> observedReason;
  std::string text;
  llvm::raw_string_ostream stream(text);
  llvm::handleAllErrors(std::move(error),
                        [&](const MemoryCapabilityFinalizationError &failure) {
                          observedReason = failure.reason();
                          failure.log(stream);
                        });
  stream.flush();

  if (observedReason != expectedReason)
    fail(test, "received a different typed finalization failure");
  if (text != expectedText)
    fail(test, "diagnostic differs: " + text);
}

MemoryContractAttr contract(mlir::MLIRContext &context, MemoryEngineAttr engine,
                            LocalMemoryServiceAttr localService,
                            MemoryConnectivityContractAttr connectivity = {}) {
  mlir::DenseI32ArrayAttr noEndpoints =
      mlir::DenseI32ArrayAttr::get(&context, {});
  return MemoryContractAttr::get(&context, engine, localService, connectivity,
                                 noEndpoints, noEndpoints);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

UnsignedDomain singleton(std::uint64_t value) {
  return take("singleton",
              UnsignedDomain::fromCanonical({UnsignedInterval{value, value}}));
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

MemoryAccessClass elementAccess() {
  auto alignments = take("alignments", AlignmentDomain::create(singleton(0)));
  auto reads = take("read semantics",
                    ClosedEnumDomain<ReadSubwordSemantics>::fromCanonical(
                        {ReadSubwordSemantics::Exact}));
  auto writes = take("write semantics",
                     ClosedEnumDomain<WriteSubwordSemantics>::fromCanonical(
                         {WriteSubwordSemantics::NotApplicable}));
  auto address = take("address domain",
                      MemoryAddressDomain::rootRelative(singleton(64)));
  return take("element access",
              MemoryAccessClass::create(
                  dataflow::semantics::MemoryAccessForm::Element, singleton(32),
                  singleton(1),
                  {MaskInactivePair{dataflow::semantics::MemoryMaskForm::Absent,
                                    InactiveLaneSemantics::NotApplicable}},
                  std::move(alignments), std::move(reads), std::move(writes),
                  std::move(address)));
}

MemoryServiceContractRecord
serviceRecord(mlir::MLIRContext &context, MemoryServiceOwnerKind owner,
              MemoryServiceConsistencyBinding consistency,
              MemoryServiceRegionBehavior behavior) {
  MemoryActorContractDomain actors = take(
      "actor domain",
      MemoryActorContractDomain::create(
          dataflow::OperationSchemaId::DataflowLoad,
          {MemoryActorContractClause(LoadStorePlainContractClause{{false}})}));
  ParameterizedMemoryAccessDomain accesses =
      take("access domain",
           ParameterizedMemoryAccessDomain::create({elementAccess()}));
  std::optional<ParameterizedMemoryAccessDomain> mmioDomain;
  if (behavior == MemoryServiceRegionBehavior::Mmio)
    mmioDomain = accesses;
  MemoryServiceContractDeclaration declaration{
      {{0, 4096, behavior, std::move(mmioDomain)}},
      directContract(),
      {{std::move(actors),
        std::move(accesses),
        {0},
        128,
        {UsePatternKey(0)},
        std::move(consistency)}}};
  return take("memory service contract",
              MemoryServiceContractRecord::create(&context, owner,
                                                  std::move(declaration)));
}

MemoryServiceContractAttr serviceAttr(mlir::MLIRContext &context,
                                      llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return MemoryServiceContractAttr::get(
      &context, mlir::DenseI8ArrayAttr::get(&context, signedBytes));
}

MemoryConnectivityContractAttr
emptyConnectivityAttr(mlir::MLIRContext &context) {
  MemoryConnectivityContractRecord record =
      take("empty connectivity", MemoryConnectivityContractRecord::create({}));
  std::vector<std::uint8_t> bytes = take(
      "connectivity encoding", encodeMemoryConnectivityContractRecord(record));
  MemoryConnectivityContractRecord decoded = take(
      "connectivity roundtrip", decodeMemoryConnectivityContractRecord(bytes));
  if (take("connectivity reencoding",
           encodeMemoryConnectivityContractRecord(decoded)) != bytes)
    fail("connectivity roundtrip", "canonical bytes changed");
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return MemoryConnectivityContractAttr::get(
      &context, mlir::DenseI8ArrayAttr::get(&context, signedBytes));
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<FabricDialect>();
  mlir::MLIRContext context(registry,
                            mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();

  if (llvm::Error error = validateMemoryCapabilityFinalization({}, {}))
    fail("empty occurrence", llvm::toString(std::move(error)));

  MemoryEngineAttr engine = MemoryEngineAttr::get(&context, Schedule::Spatial,
                                                  MemoryResidentContextsAttr());
  expectInvalid(
      "incomplete engine",
      validateMemoryCapabilityFinalization(contract(context, engine, {}), {}),
      MemoryCapabilityFinalizationReason::MissingMemoryCapabilityContract,
      "Invalid(missing-memory-capability-contract)");

  MemoryServiceContractRecord service =
      serviceRecord(context, MemoryServiceOwnerKind::Local,
                    MemoryServiceConsistencyBinding(
                        std::in_place_type<NoMemoryServiceConsistency>),
                    MemoryServiceRegionBehavior::Mmio);
  std::vector<std::uint8_t> serviceBytes =
      take("service encoding", encodeMemoryServiceContractRecord(service));
  MemoryServiceContractRecord decoded =
      take("service roundtrip",
           decodeMemoryServiceContractRecord(serviceBytes, &context,
                                             MemoryServiceOwnerKind::Local));
  if (take("service reencoding", encodeMemoryServiceContractRecord(decoded)) !=
      serviceBytes)
    fail("service roundtrip", "canonical bytes changed");

  MemoryServiceContractRecord systemDecoded =
      take("shared owner record",
           decodeMemoryServiceContractRecord(serviceBytes, &context,
                                             MemoryServiceOwnerKind::System));
  if (take("shared owner reencoding",
           encodeMemoryServiceContractRecord(systemDecoded)) != serviceBytes)
    fail("shared owner record", "System import changed canonical bytes");

  LocalProviderConsistency localConsistency{
      ReleaseVisibilityPoint::ByRetirement,
      LocalProviderProgress(std::in_place_type<LocalBoundedCompletionCycles>,
                            LocalBoundedCompletionCycles{4})};
  MemoryServiceContractRecord localOnly =
      serviceRecord(context, MemoryServiceOwnerKind::Local,
                    MemoryServiceConsistencyBinding(
                        std::in_place_type<LocalProviderConsistency>,
                        std::move(localConsistency)),
                    MemoryServiceRegionBehavior::Storage);
  std::vector<std::uint8_t> localOnlyBytes = take(
      "local provider encoding", encodeMemoryServiceContractRecord(localOnly));

  auto wrongOwner = decodeMemoryServiceContractRecord(
      localOnlyBytes, &context, MemoryServiceOwnerKind::System);
  if (wrongOwner)
    fail("owner binding", "accepted LocalProvider for a System service");
  llvm::consumeError(wrongOwner.takeError());

  MemoryServiceContractAttr serviceContract =
      serviceAttr(context, serviceBytes);
  std::string printedServiceContract;
  llvm::raw_string_ostream printedStream(printedServiceContract);
  static_cast<mlir::Attribute>(serviceContract).print(printedStream);
  printedStream.flush();
  mlir::Attribute parsedAttribute =
      mlir::parseAttribute(printedServiceContract, &context);
  auto parsedServiceContract =
      mlir::dyn_cast_or_null<MemoryServiceContractAttr>(parsedAttribute);
  if (!parsedServiceContract ||
      parsedServiceContract.getRecord() != serviceContract.getRecord())
    fail("service attribute roundtrip", "canonical record bytes changed");
  LocalMemoryServiceAttr local =
      LocalMemoryServiceAttr::get(&context, 4096, serviceContract);
  if (llvm::Error error = validateMemoryCapabilityFinalization(
          contract(context, {}, local, emptyConnectivityAttr(context)), {}))
    fail("local service", llvm::toString(std::move(error)));
  llvm::Error zeroCapacity = validateLocalMemoryServiceCapacity(decoded, 0);
  if (!zeroCapacity)
    fail("local capacity", "accepted zero local-service capacity");
  llvm::consumeError(std::move(zeroCapacity));
  expectInvalid(
      "engine with local service",
      validateMemoryCapabilityFinalization(contract(context, engine, local),
                                           {}),
      MemoryCapabilityFinalizationReason::MissingMemoryCapabilityContract,
      "Invalid(missing-memory-capability-contract)");
  return EXIT_SUCCESS;
}

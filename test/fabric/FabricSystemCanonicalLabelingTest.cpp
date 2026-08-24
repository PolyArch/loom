#include "FabricSystemCanonicalLabeling.h"

#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/ResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"
#include "Fabric/IR/SystemServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::fabric::FabricTransportEndpointOwnerRef;
using loom::fabric::FabricTransportEndpointRef;
using loom::fabric::HardwareDomainRef;
using loom::fabric::SystemServiceEndpointRef;
using loom::fabric::SystemTransportResourceRef;
using loom::fabric::detail::FabricSystemCanonicalLabeling;

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

mlir::MLIRContext &context() {
  static mlir::MLIRContext *ctx = [] {
    mlir::DialectRegistry registry;
    registry.insert<::fabric::FabricDialect>();
    auto *result = new mlir::MLIRContext(
        registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *ctx;
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

::fabric::ResourceContract transportContract(llvm::StringRef test) {
  ::fabric::ResourceContractDeclaration declaration;
  declaration.timingContracts = {::fabric::TimingContractDeclaration{
      ::fabric::TimingContractKey(0), {0, 1}}};
  declaration.requesters = {::fabric::RequesterKey(0)};
  declaration.eligibilityCount = 1;
  declaration.eventCount = 2;
  declaration.usePatterns = {
      ::fabric::UsePatternDeclaration{::fabric::UsePatternKey(0),
                                      ::fabric::RequesterKey(0),
                                      ::fabric::EligibilityKey(0),
                                      ::fabric::EventKey(0),
                                      ::fabric::EventKey(1),
                                      std::nullopt,
                                      ::fabric::TimingContractKey(0),
                                      {},
                                      {}}};
  return take(test, ::fabric::ResourceContract::create(declaration));
}

std::string resource(llvm::StringRef test, std::uint64_t id,
                     std::uint32_t width) {
  const std::vector<std::uint8_t> contract = take(
      test, ::fabric::encodeResourceContractRecord(transportContract(test)));
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "fabric.system.transport_resource ports = "
            "(!fabric.bits<"
         << width << ">) -> (!fabric.bits<" << width
         << ">) contract = " << denseI8Assembly(contract)
         << " {entity_id = #fabric.entity_id<" << id << ">}\n";
  return text;
}

std::string connection(std::uint64_t sourceId, std::uint64_t destinationId) {
  const FabricTransportEndpointRef source{
      FabricTransportEndpointOwnerRef::of(SystemTransportResourceRef(sourceId)),
      0};
  const FabricTransportEndpointRef destination{
      FabricTransportEndpointOwnerRef::of(
          SystemTransportResourceRef(destinationId)),
      0};
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "fabric.system.connection source = "
         << denseI8Assembly(loom::fabric::canonicalFabricBytes(source))
         << " destination = "
         << denseI8Assembly(loom::fabric::canonicalFabricBytes(destination))
         << "\n";
  return text;
}

std::string systemSource(bool reverseOrder, bool reverseConnection) {
  const std::uint64_t firstId = reverseOrder ? 91 : 17;
  const std::uint64_t secondId = reverseOrder ? 4 : 63;
  const std::string first = resource(__func__, firstId, 32);
  const std::string second = resource(__func__, secondId, 64);
  const std::string edge = reverseConnection ? connection(secondId, firstId)
                                             : connection(firstId, secondId);

  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module { fabric.system @soc {\n";
  if (reverseOrder)
    stream << second << edge << first;
  else
    stream << first << second << edge;
  stream << "} }\n";
  return text;
}

std::string serviceEndpointSystemSource(std::uint32_t carrierWidth) {
  const llvm::StringRef test = __func__;
  constexpr std::uint64_t boundaryId = 40;
  constexpr std::uint64_t endpointId = 41;
  constexpr std::uint64_t clockId = 42;

  loom::fabric::SystemServiceEndpointOwnerRef owner =
      take(test, loom::fabric::SystemServiceEndpointOwnerRef::create(
                     loom::fabric::FabricInventoryOwnerRef::of(
                         loom::fabric::ExternalBoundaryRef(boundaryId))));
  loom::fabric::ClockDomainRef clock{HardwareDomainRef(clockId)};
  loom::fabric::ServiceRateContractRecord rate =
      take(test, loom::fabric::ServiceRateContractRecord::create(
                     clock, 1, 1, 4,
                     loom::fabric::ServiceProgress(
                         std::in_place_type<::fabric::FairEventual>)));
  loom::fabric::MessageTransferCapabilityDomain domain =
      take(test, loom::fabric::MessageTransferCapabilityDomain::create(
                     {mlir::IntegerType::get(&context(), 32)}));
  loom::fabric::CanonicalServiceCapabilityRecord capability =
      take(test, loom::fabric::CanonicalServiceCapabilityRecord::create(
                     dataflow::semantics::ServiceKind::MessageTransfer,
                     loom::fabric::CanonicalServiceEndpointRole::Initiate,
                     std::move(domain), std::move(rate)));
  loom::fabric::CanonicalServiceCapabilitySet capabilities =
      take(test, loom::fabric::CanonicalServiceCapabilitySet::create(
                     {std::move(capability)}));
  std::vector<std::uint8_t> capabilityBytes = take(
      test, loom::fabric::encodeCanonicalServiceCapabilitySet(capabilities));

  loom::fabric::ClockDomainContractRecord clockContract =
      take(test, loom::fabric::ClockDomainContractRecord::create(1'000, 0));
  loom::fabric::HardwareDomainContractRecord hardwareDomain =
      take(test, loom::fabric::HardwareDomainContractRecord::create(
                     {loom::fabric::FabricHardwareDomainMemberRef::of(
                          loom::fabric::FabricInventoryOwnerRef::of(
                              loom::fabric::ExternalBoundaryRef(boundaryId))),
                      loom::fabric::FabricHardwareDomainMemberRef::of(
                          loom::fabric::FabricInventoryOwnerRef::of(
                              SystemServiceEndpointRef(endpointId)))},
                     std::move(clockContract)));
  std::vector<std::uint8_t> domainBytes = take(
      test, loom::fabric::encodeHardwareDomainContractRecord(hardwareDomain));

  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << "module { fabric.system @soc {\n"
         << "fabric.system.external_boundary "
            "{entity_id = #fabric.entity_id<"
         << boundaryId << ">}\n"
         << "fabric.system.service_endpoint owner = "
         << denseI8Assembly(
                loom::fabric::encodeSystemServiceEndpointOwnerRef(owner))
         << " capabilities = " << denseI8Assembly(capabilityBytes)
         << " carrier = !fabric.bits<" << carrierWidth
         << "> {entity_id = #fabric.entity_id<" << endpointId << ">}\n"
         << "fabric.system.hardware_domain contract = "
         << denseI8Assembly(domainBytes) << " {entity_id = #fabric.entity_id<"
         << clockId << ">}\n"
         << "} }\n";
  return text;
}

FabricSystemCanonicalLabeling label(llvm::StringRef test,
                                    llvm::StringRef source) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail(test, "unable to parse Fabric System fixture");
  auto root = *module->getOps<::fabric::SystemOp>().begin();
  return take(test, loom::fabric::detail::computeFabricSystemCanonicalLabeling(
                        root, {}));
}

struct ParsedSystem {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ::fabric::SystemOp root;
};

ParsedSystem parseSystem(llvm::StringRef test, llvm::StringRef source) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail(test, "unable to parse Fabric System fixture");
  auto root = *module->getOps<::fabric::SystemOp>().begin();
  return {std::move(module), root};
}

void provisionalIdentityAndOrderDoNotAffectCanonicalRelation() {
  const FabricSystemCanonicalLabeling first =
      label(__func__, systemSource(false, false));
  const FabricSystemCanonicalLabeling reordered =
      label(__func__, systemSource(true, false));
  require(__func__,
          first.relationBytes.bytes().equals(reordered.relationBytes.bytes()),
          "temporary EntityIds or declaration order changed System identity");
}

void relationDirectionAffectsCanonicalRelation() {
  const FabricSystemCanonicalLabeling forward =
      label(__func__, systemSource(false, false));
  const FabricSystemCanonicalLabeling reversed =
      label(__func__, systemSource(false, true));
  require(__func__,
          !forward.relationBytes.bytes().equals(reversed.relationBytes.bytes()),
          "a directed System connection change preserved canonical identity");
}

void materializationPreservesConnectionReferenceSemantics() {
  ParsedSystem parsed = parseSystem(__func__, systemSource(false, false));
  FabricSystemCanonicalLabeling labeling =
      take(__func__, loom::fabric::detail::computeFabricSystemCanonicalLabeling(
                         parsed.root, {}));
  if (llvm::Error error =
          loom::fabric::detail::materializeFabricSystemCanonicalForm(
              parsed.root, labeling))
    fail(__func__, llvm::toString(std::move(error)));

  std::vector<std::uint64_t> ids;
  parsed.root.getBody().walk([&](mlir::Operation *operation) {
    if (auto id = operation->getAttrOfType<::fabric::EntityIdAttr>("entity_id"))
      ids.push_back(id.getId());
  });
  llvm::sort(ids);
  require(__func__, ids == std::vector<std::uint64_t>({0, 1}),
          "System entities did not receive one dense canonical ID range");

  auto relabeled =
      take(__func__, loom::fabric::detail::computeFabricSystemCanonicalLabeling(
                         parsed.root, {}));
  require(
      __func__,
      labeling.relationBytes.bytes().equals(relabeled.relationBytes.bytes()),
      "materializing canonical refs changed System semantics");
}

void serviceEndpointCarrierIsIdentityCritical() {
  const FabricSystemCanonicalLabeling narrow =
      label(__func__, serviceEndpointSystemSource(32));
  const FabricSystemCanonicalLabeling wide =
      label(__func__, serviceEndpointSystemSource(64));
  require(__func__,
          !narrow.relationBytes.bytes().equals(wide.relationBytes.bytes()),
          "message carrier width did not affect System identity");

  ParsedSystem parsed = parseSystem(__func__, serviceEndpointSystemSource(32));
  FabricSystemCanonicalLabeling labeling =
      take(__func__, loom::fabric::detail::computeFabricSystemCanonicalLabeling(
                         parsed.root, {}));
  if (llvm::Error error =
          loom::fabric::detail::materializeFabricSystemCanonicalForm(
              parsed.root, labeling))
    fail(__func__, llvm::toString(std::move(error)));
  FabricSystemCanonicalLabeling relabeled =
      take(__func__, loom::fabric::detail::computeFabricSystemCanonicalLabeling(
                         parsed.root, {}));
  require(
      __func__,
      labeling.relationBytes.bytes().equals(relabeled.relationBytes.bytes()),
      "service endpoint materialization changed System identity");
}

} // namespace

int main() {
  provisionalIdentityAndOrderDoNotAffectCanonicalRelation();
  relationDirectionAffectsCanonicalRelation();
  materializationPreservesConnectionReferenceSemantics();
  serviceEndpointCarrierIsIdentityCritical();
  llvm::outs() << "fabric system canonical labeling ok\n";
  return 0;
}

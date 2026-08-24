#include "ADG/Builtin.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricHardwareDomainContracts.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/SystemServiceContract.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FinalizedFabricRoot;

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
void expectRejected(llvm::StringRef test, llvm::Expected<T> value,
                    llvm::StringRef diagnostic) {
  if (value)
    fail(test, "accepted invalid input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(diagnostic), message);
}

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-system-attachment-test", path))
      fail(test, error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << test_ << ": unable to remove temporary directory: "
                   << error.message() << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string test_;
  std::string path_;
};

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

mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef test,
                                        llvm::StringRef source) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail(test, "unable to parse Fabric source");
  return module;
}

::fabric::SystemOp systemRoot(llvm::StringRef test, mlir::ModuleOp module) {
  ::fabric::SystemOp selected;
  for (::fabric::SystemOp candidate : module.getOps<::fabric::SystemOp>()) {
    if (selected)
      fail(test, "fixture has more than one System root");
    selected = candidate;
  }
  if (!selected)
    fail(test, "fixture has no System root");
  return selected;
}

mlir::DenseI8ArrayAttr denseI8Attr(llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(&context(), signedBytes);
}

llvm::ArrayRef<std::uint8_t> unsignedBytes(mlir::DenseI8ArrayAttr attribute) {
  llvm::ArrayRef<std::int8_t> bytes = attribute.asArrayRef();
  return {reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()};
}

::fabric::SystemSpatialAttachmentOp firstSpatialAttachment(
    llvm::StringRef test, ::fabric::SystemOp root,
    loom::fabric::FabricSpatialAttachmentEndpointRef::Plane plane) {
  for (::fabric::SystemSpatialAttachmentOp attachment :
       root.getBody().getOps<::fabric::SystemSpatialAttachmentOp>()) {
    auto endpoint = loom::fabric::decodeFabricSpatialAttachmentEndpointRef(
        unsignedBytes(attachment.getSpatialEndpointAttr()));
    if (!endpoint)
      fail(test, llvm::toString(endpoint.takeError()));
    if (endpoint->plane() == plane)
      return attachment;
  }
  fail(test, "fixture has no requested spatial attachment");
}

::fabric::SystemServiceEndpointOp
serviceEndpoint(llvm::StringRef test, ::fabric::SystemOp root,
                loom::fabric::SystemServiceEndpointRef reference) {
  for (::fabric::SystemServiceEndpointOp endpoint :
       root.getBody().getOps<::fabric::SystemServiceEndpointOp>()) {
    ::fabric::EntityIdAttr id = endpoint.getEntityIdAttr();
    if (id && id.getId() == reference.id())
      return endpoint;
  }
  fail(test, "fixture has no requested service endpoint");
}

loom::fabric::SystemServiceEndpointRef addDistinctEquivalentServiceEndpoint(
    llvm::StringRef test, ::fabric::SystemOp root,
    loom::fabric::SystemServiceEndpointRef originalReference) {
  ::fabric::SystemServiceEndpointOp original =
      serviceEndpoint(test, root, originalReference);
  std::optional<loom::fabric::AccCoreOccurrenceRef> owner;
  std::uint64_t maxId = 0;
  root.getBody().walk([&](mlir::Operation *operation) {
    if (auto id = operation->getAttrOfType<::fabric::EntityIdAttr>("entity_id"))
      maxId = std::max(maxId, id.getId());
    if (owner)
      return;
    if (auto core = mlir::dyn_cast<::fabric::SystemAccCoreOp>(operation))
      if (::fabric::EntityIdAttr id = core.getEntityIdAttr())
        owner = loom::fabric::AccCoreOccurrenceRef(id.getId());
  });
  require(test, owner.has_value(), "fixture has no AccCore endpoint owner");
  const loom::fabric::SystemServiceEndpointRef added(maxId + 1);
  auto ownerReference = take(
      test, loom::fabric::SystemServiceEndpointOwnerRef::create(
                loom::fabric::FabricInventoryOwnerRef::of(*owner)));

  mlir::OpBuilder builder(&context());
  builder.setInsertionPointToEnd(&root.getBody().front());
  ::fabric::SystemServiceEndpointOp::create(
      builder, root.getLoc(),
      ::fabric::EntityIdAttr::get(&context(), added.id()),
      denseI8Attr(loom::fabric::encodeSystemServiceEndpointOwnerRef(
          ownerReference)),
      original.getCapabilitiesAttr(), mlir::TypeAttr());

  const loom::fabric::FabricInventoryOwnerRef originalMember =
      loom::fabric::FabricInventoryOwnerRef::of(originalReference);
  const loom::fabric::FabricInventoryOwnerRef addedMember =
      loom::fabric::FabricInventoryOwnerRef::of(added);
  for (::fabric::SystemHardwareDomainOp domain :
       root.getBody().getOps<::fabric::SystemHardwareDomainOp>()) {
    auto contract = take(
        test, loom::fabric::decodeHardwareDomainContractRecord(
                  unsignedBytes(domain.getContractAttr())));
    std::vector<loom::fabric::FabricHardwareDomainMemberRef> members(
        contract.members().begin(), contract.members().end());
    bool containsOriginal = false;
    const auto originalDomainMember =
        loom::fabric::FabricHardwareDomainMemberRef::of(originalMember);
    for (const auto &member : members)
      containsOriginal |= member == originalDomainMember;
    if (!containsOriginal)
      continue;
    members.push_back(loom::fabric::FabricHardwareDomainMemberRef::of(
        addedMember));
    auto expanded = take(
        test, loom::fabric::HardwareDomainContractRecord::create(
                  std::move(members), contract.contract()));
    domain.setContractAttr(denseI8Attr(take(
        test, loom::fabric::encodeHardwareDomainContractRecord(expanded))));
  }

  std::vector<loom::fabric::ServiceLegCarrierAttachmentRecord> addedLegs;
  for (::fabric::SystemServiceLegCarrierAttachmentOp attachment :
       root.getBody().getOps<
           ::fabric::SystemServiceLegCarrierAttachmentOp>()) {
    auto record = take(
        test, loom::fabric::decodeServiceLegCarrierAttachmentRecord(
                  unsignedBytes(attachment.getRecordAttr())));
    if (record.endpoint().owner !=
        loom::fabric::FabricMemoryEndpointOwnerRef::of(originalReference))
      continue;
    const loom::fabric::FabricMemoryEndpointRef endpoint{
        loom::fabric::FabricMemoryEndpointOwnerRef::of(added),
        record.endpoint().ordinal};
    addedLegs.push_back(take(
        test, loom::fabric::ServiceLegCarrierAttachmentRecord::create(
                  endpoint, record.kind(), record.legOrdinal(),
                  std::vector<loom::fabric::FabricTransportEndpointRef>(
                      record.carriers().begin(), record.carriers().end()))));
  }
  for (const auto &record : addedLegs)
    ::fabric::SystemServiceLegCarrierAttachmentOp::create(
        builder, root.getLoc(),
        denseI8Attr(take(
            test,
            loom::fabric::encodeServiceLegCarrierAttachmentRecord(record))));
  return added;
}

void systemRejectsInvalidSpatialMemoryBindings() {
  const llvm::StringRef test = __func__;
  mlir::ScopedDiagnosticHandler diagnostics(
      &context(), [](mlir::Diagnostic &) { return mlir::success(); });
  TemporaryDirectory directory(test);
  ArtifactStore store(directory.path());
  loom::adg::FinalizedFabricDesign design =
      take(test, loom::adg::buildBuiltinTarget(
                     store, loom::adg::BuiltinTargetPreset::Small));
  require(test, design.roots().size() == 1,
          "builtin fixture did not publish one System root");
  const FinalizedFabricRoot &published = design.roots().front();

  std::string source;
  llvm::raw_string_ostream stream(source);
  if (llvm::Error error = loom::fabric::writeFabricMlir(published, stream))
    fail(test, llvm::toString(std::move(error)));
  stream.flush();

  std::vector<loom::ArtifactRootReference> dependencies;
  for (const loom::fabric::FabricDirectDependency &dependency :
       published.directDependencies())
    dependencies.push_back(dependency.root);

  auto finalize = [&](mlir::OwningOpRef<mlir::ModuleOp> module) {
    return loom::fabric::finalizeFabricRoot(systemRoot(test, *module),
                                            dependencies, store);
  };
  auto fresh = [&]() { return parse(test, source); };

  loom::fabric::SystemServiceEndpointRef boundEndpoint;
  {
    auto module = fresh();
    auto attachment = firstSpatialAttachment(
        test, systemRoot(test, *module),
        loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory);
    boundEndpoint = take(
        test, loom::fabric::decodeFabricRef<
                  loom::fabric::SystemServiceEndpointRef>(
                  unsignedBytes(attachment.getServiceEndpointAttr())));
  }

  {
    auto module = fresh();
    auto attachment = firstSpatialAttachment(
        test, systemRoot(test, *module),
        loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory);
    attachment->removeAttr("service_endpoint");
    expectRejected(test, finalize(std::move(module)),
                   "System authoring module does not verify");
  }

  {
    auto module = fresh();
    ::fabric::SystemOp root = systemRoot(test, *module);
    auto attachment = firstSpatialAttachment(
        test, root,
        loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Transport);
    attachment.setServiceEndpointAttr(
        denseI8Attr(loom::fabric::canonicalFabricBytes(boundEndpoint)));
    expectRejected(test, finalize(std::move(module)),
                   "System authoring module does not verify");
  }

  {
    auto module = fresh();
    ::fabric::SystemOp root = systemRoot(test, *module);
    auto attachment = firstSpatialAttachment(
        test, root,
        loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory);
    attachment.setServiceEndpointAttr(denseI8Attr(
        loom::fabric::canonicalFabricBytes(
            loom::fabric::SystemServiceEndpointRef(999999))));
    expectRejected(test, finalize(std::move(module)),
                   "System relation references an unknown entity");
  }

  {
    auto module = fresh();
    ::fabric::SystemOp root = systemRoot(test, *module);
    auto attachment = firstSpatialAttachment(
        test, root,
        loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory);
    root.getBody().front().push_back(attachment->clone());
    expectRejected(test, finalize(std::move(module)),
                   "SpatialCore boundary endpoint is attached twice");
  }

  {
    auto module = fresh();
    ::fabric::SystemOp root = systemRoot(test, *module);
    auto endpoint = serviceEndpoint(test, root, boundEndpoint);
    auto capabilities = take(
        test, loom::fabric::decodeCanonicalServiceCapabilitySet(
                  unsignedBytes(endpoint.getCapabilitiesAttr()), &context()));
    std::vector<loom::fabric::CanonicalServiceCapabilityRecord> changed;
    for (const auto &capability : capabilities.capabilities())
      changed.push_back(take(
          test, loom::fabric::CanonicalServiceCapabilityRecord::create(
                    capability.kind(),
                    loom::fabric::CanonicalServiceEndpointRole::Initiate,
                    capability.domain(), capability.rate())));
    auto changedSet = take(
        test, loom::fabric::CanonicalServiceCapabilitySet::create(
                  std::move(changed)));
    endpoint.setCapabilitiesAttr(denseI8Attr(take(
        test, loom::fabric::encodeCanonicalServiceCapabilitySet(changedSet))));
    expectRejected(test, finalize(std::move(module)),
                   "joins equal endpoint roles");
  }

  {
    auto module = fresh();
    ::fabric::SystemOp root = systemRoot(test, *module);
    auto endpoint = serviceEndpoint(test, root, boundEndpoint);
    auto capabilities = take(
        test, loom::fabric::decodeCanonicalServiceCapabilitySet(
                  unsignedBytes(endpoint.getCapabilitiesAttr()), &context()));
    auto domain = take(
        test, loom::fabric::MessageTransferCapabilityDomain::create(
                  {mlir::IntegerType::get(&context(), 32)}));
    auto capability = take(
        test, loom::fabric::CanonicalServiceCapabilityRecord::create(
                  dataflow::semantics::ServiceKind::MessageTransfer,
                  loom::fabric::CanonicalServiceEndpointRole::Serve,
                  std::move(domain),
                  capabilities.capabilities().front().rate()));
    auto changedSet = take(
        test, loom::fabric::CanonicalServiceCapabilitySet::create(
                  {std::move(capability)}));
    endpoint.setCapabilitiesAttr(denseI8Attr(take(
        test, loom::fabric::encodeCanonicalServiceCapabilitySet(changedSet))));
    endpoint.setCarrierTypeAttr(
        mlir::TypeAttr::get(::fabric::BitsType::get(&context(), 32)));
    expectRejected(test, finalize(std::move(module)),
                   "transport-plane service endpoint");
  }

  auto finalizeBindingVariant = [&](bool rebind) {
    auto module = fresh();
    ::fabric::SystemOp root = systemRoot(test, *module);
    const loom::fabric::SystemServiceEndpointRef added =
        addDistinctEquivalentServiceEndpoint(test, root, boundEndpoint);
    if (rebind) {
      auto attachment = firstSpatialAttachment(
          test, root,
          loom::fabric::FabricSpatialAttachmentEndpointRef::Plane::Memory);
      attachment.setServiceEndpointAttr(
          denseI8Attr(loom::fabric::canonicalFabricBytes(added)));
    }
    return take(test, finalize(std::move(module)));
  };
  FinalizedFabricRoot originalBinding = finalizeBindingVariant(false);
  FinalizedFabricRoot distinctBinding = finalizeBindingVariant(true);
  require(test,
          originalBinding.reference().artifact !=
              distinctBinding.reference().artifact,
          "changing the exact memory service binding preserved System "
          "identity");
}

} // namespace

int main() {
  systemRejectsInvalidSpatialMemoryBindings();
  return EXIT_SUCCESS;
}

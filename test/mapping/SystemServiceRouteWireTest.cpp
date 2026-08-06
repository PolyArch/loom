#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "Mapping/Artifact/SystemMappingIdentity.h"
#include "Mapping/IR/MappingDialect.h"

#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "System service route wire anchor failed: " << message
               << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, const llvm::Twine &message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

loom::ArtifactIdentity identity(std::uint8_t seed) {
  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index)
    bytes[index] = static_cast<std::uint8_t>(seed + index);
  return take(loom::ArtifactIdentity::fromBytes(bytes));
}

mlir::DenseI8ArrayAttr denseBytes(mlir::MLIRContext *context,
                                  llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::int8_t> signedBytes;
  signedBytes.reserve(bytes.size());
  for (std::uint8_t byte : bytes)
    signedBytes.push_back(static_cast<std::int8_t>(byte));
  return mlir::DenseI8ArrayAttr::get(context, signedBytes);
}

template <typename Attr, typename Ref>
Attr dataflowAttr(mlir::MLIRContext *context,
                  const loom::ArtifactIdentity &owner, const Ref &reference) {
  return Attr::get(context,
                   denseBytes(context, take(dataflow::encodeDataflowReference(
                                           owner, reference))));
}

template <typename Attr, typename Ref>
Attr fabricAttr(mlir::MLIRContext *context, const Ref &reference) {
  return Attr::get(
      context,
      denseBytes(context, loom::fabric::canonicalFabricBytes(reference)));
}

::mapping::ArtifactIdentityAttr
identityAttr(mlir::MLIRContext *context, const loom::ArtifactIdentity &value) {
  return ::mapping::ArtifactIdentityAttr::get(
      context, denseBytes(context, value.bytes()));
}

struct Fixture final {
  loom::ArtifactIdentity dataflowIdentity;
  loom::ArtifactIdentity fabricIdentity;
  dataflow::RootThreadLaunchRef root;
  loom::mapping::SystemServiceObligationKey obligation;
  loom::mapping::CanonicalServiceLegKey leg;
  loom::mapping::SystemTransferSinkTerminalKey sink;
  loom::fabric::AccCoreOccurrenceRef core;
  loom::fabric::FabricTransportEndpointRef source;
  loom::fabric::FabricTransportEndpointRef destination;
  loom::fabric::FabricPhysicalTraversalRef traversal;
};

Fixture makeFixture() {
  auto dataflowIdentity = identity(1);
  auto fabricIdentity = identity(65);
  dataflow::RootThreadLaunchRef root{dataflowIdentity,
                                     dataflow::RootThreadLaunchId(7)};
  dataflow::CanonicalProducerTerminalRef producer{
      dataflow::RootThreadBoundarySourceRef{
          dataflow::RootThreadBoundaryTransferRef{
              dataflow::RootThreadStartTransferRef{root}}}};
  loom::mapping::SystemServiceObligationKey obligation(producer);
  loom::mapping::CanonicalServiceLegKey leg{
      obligation,
      dataflow::ServiceMemberRef(dataflow::MessageTransferMemberRef{}), 0};
  loom::fabric::SystemTransportResourceRef resource(21);
  loom::fabric::FabricTransportEndpointOwnerRef owner =
      loom::fabric::FabricTransportEndpointOwnerRef::of(resource);
  loom::fabric::FabricTransportEndpointRef source{owner, 0};
  loom::fabric::FabricTransportEndpointRef destination{owner, 1};
  return Fixture{std::move(dataflowIdentity),
                 std::move(fabricIdentity),
                 root,
                 obligation,
                 leg,
                 {leg, 0},
                 loom::fabric::AccCoreOccurrenceRef(8),
                 source,
                 destination,
                 loom::fabric::FabricPhysicalTraversalRef::pointConnection(
                     source, destination)};
}

mlir::OwningOpRef<mlir::Operation *>
buildSystem(mlir::MLIRContext &context, const Fixture &fixture,
            std::uint64_t nodeOrdinal = 9, std::uint64_t parentOrdinal = 0,
            bool sinkFirst = false) {
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  auto root = ::mapping::SystemOp::create(
      builder, location, identityAttr(&context, fixture.dataflowIdentity),
      identityAttr(&context, fixture.fabricIdentity), builder.getArrayAttr({}),
      builder.getArrayAttr({dataflowAttr<::mapping::RootThreadLaunchRefAttr>(
          &context, fixture.dataflowIdentity, fixture.root)}));
  root.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&root.getBody().front());

  mlir::OperationState threadState(
      location, ::mapping::ThreadExecutionBindingOp::getOperationName());
  threadState.addAttribute(
      "key", dataflowAttr<::mapping::RootThreadLaunchRefAttr>(
                 &context, fixture.dataflowIdentity, fixture.root));
  threadState.addAttribute(
      "relation_kind",
      ::mapping::SystemBindingRelationKindAttr::get(
          &context, ::mapping::SystemBindingRelationKind::PresburgerPartition));
  threadState.addAttribute(
      "default_target", fabricAttr<::mapping::FabricAccCoreOccurrenceRefAttr>(
                            &context, fixture.core));
  threadState.addRegion();
  auto thread = mlir::cast<::mapping::ThreadExecutionBindingOp>(
      builder.create(threadState));
  thread.getBody().emplaceBlock();

  builder.setInsertionPointToEnd(&root.getBody().front());
  auto service = ::mapping::ServiceRealizationOp::create(
      builder, location,
      ::mapping::SystemServiceObligationKeyAttr::get(
          &context,
          denseBytes(&context,
                     take(loom::mapping::encodeSystemServiceObligationKey(
                         fixture.dataflowIdentity, fixture.obligation)))));
  service.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&service.getBody().front());
  auto plan = ::mapping::ServicePlanOp::create(builder, location, 4);
  plan.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&plan.getBody().front());
  auto route = ::mapping::TransferLegRealizationOp::create(
      builder, location,
      ::mapping::CanonicalServiceLegKeyAttr::get(
          &context,
          denseBytes(&context, take(loom::mapping::encodeCanonicalServiceLegKey(
                                   fixture.dataflowIdentity, fixture.leg)))),
      fabricAttr<::mapping::FabricTransportEndpointRefAttr>(&context,
                                                            fixture.source));
  route.getBody().emplaceBlock();
  builder.setInsertionPointToEnd(&route.getBody().front());
  const auto emitSink = [&] {
    ::mapping::SystemRouteSinkOp::create(
        builder, location,
        ::mapping::SystemTransferTerminalKeyAttr::get(
            &context,
            denseBytes(
                &context,
                take(loom::mapping::encodeSystemTransferTerminalKey(
                    fixture.dataflowIdentity,
                    loom::mapping::SystemTransferTerminalKey(fixture.sink))))),
        nodeOrdinal);
  };
  if (sinkFirst)
    emitSink();
  ::mapping::SystemRouteNodeOp::create(
      builder, location, nodeOrdinal, parentOrdinal,
      fabricAttr<::mapping::FabricPhysicalTraversalRefAttr>(&context,
                                                            fixture.traversal));
  if (!sinkFirst)
    emitSink();
  return mlir::OwningOpRef<mlir::Operation *>(root.getOperation());
}

mlir::OwningOpRef<mlir::ModuleOp>
parseCanonical(mlir::MLIRContext &context,
               const loom::CanonicalSemanticBytes &bytes) {
  std::string wrapped = "module {\n";
  wrapped.append(reinterpret_cast<const char *>(bytes.bytes().data()),
                 bytes.bytes().size());
  wrapped += "}\n";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(wrapped, &context);
  if (!module)
    fail("canonical route payload did not parse");
  return module;
}

void requireVerificationFailure(mlir::Operation *operation,
                                llvm::StringRef expected) {
  std::vector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      operation->getContext(), [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
        return mlir::success();
      });
  require(mlir::failed(mlir::verify(operation)),
          "adverse route unexpectedly verified");
  require(llvm::any_of(diagnostics,
                       [&](const std::string &diagnostic) {
                         return llvm::StringRef(diagnostic).contains(expected);
                       }),
          "adverse route diagnostic changed");
}

} // namespace

int main() {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.getOrLoadDialect<::mapping::MappingDialect>();
  const Fixture fixture = makeFixture();

  auto authored = buildSystem(context, fixture);
  auto first = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(authored.get())));
  auto second = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(authored.get())));
  require(first.bytes() == second.bytes(),
          "System service route canonicalization is nondeterministic");
  auto alternate = buildSystem(context, fixture, 41, 0, true);
  auto alternateBytes = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(alternate.get())));
  require(first.bytes() == alternateBytes.bytes(),
          "System route canonicalization depends on authoring order");

  const auto sinkKeyBytes = take(loom::mapping::encodeSystemTransferTerminalKey(
      fixture.dataflowIdentity,
      loom::mapping::SystemTransferTerminalKey(fixture.sink)));
  const auto decodedSinkKey =
      take(loom::mapping::decodeSystemTransferTerminalKey(
          sinkKeyBytes, fixture.dataflowIdentity));
  require(decodedSinkKey ==
              loom::mapping::SystemTransferTerminalKey(fixture.sink),
          "Mapping transfer-terminal codec changed the canonical key");

  auto parsed = parseCanonical(context, first);
  auto root = mlir::cast<::mapping::SystemOp>(parsed->getBody()->front());
  auto service =
      *root.getBody().front().getOps<::mapping::ServiceRealizationOp>().begin();
  auto plan =
      *service.getBody().front().getOps<::mapping::ServicePlanOp>().begin();
  auto route = *plan.getBody()
                    .front()
                    .getOps<::mapping::TransferLegRealizationOp>()
                    .begin();
  auto node =
      *route.getBody().front().getOps<::mapping::SystemRouteNodeOp>().begin();
  auto sink =
      *route.getBody().front().getOps<::mapping::SystemRouteSinkOp>().begin();
  require(plan.getPlanOrdinal() == 0 && node.getNodeOrdinal() == 1 &&
              node.getParentNodeOrdinal() == 0 && sink.getNodeOrdinal() == 1,
          "System route ordinals were not canonically renumbered");
  require(route.getRootEndpoint() ==
              fabricAttr<::mapping::FabricTransportEndpointRefAttr>(
                  &context, fixture.source),
          "System route lost its root transport terminal");
  require(node->getAttrs().size() == 3 && !node->hasAttr("refinements") &&
              !node->hasAttr("endpoint"),
          "System route node copied a Spatial-only field");

  mlir::OwningOpRef<mlir::Operation *> duplicate(authored->clone());
  auto duplicateRoot = mlir::cast<::mapping::SystemOp>(duplicate.get());
  auto duplicateService = *duplicateRoot.getBody()
                               .front()
                               .getOps<::mapping::ServiceRealizationOp>()
                               .begin();
  auto duplicatePlan = *duplicateService.getBody()
                            .front()
                            .getOps<::mapping::ServicePlanOp>()
                            .begin();
  auto duplicateRoute = *duplicatePlan.getBody()
                             .front()
                             .getOps<::mapping::TransferLegRealizationOp>()
                             .begin();
  duplicateRoute.getBody().front().push_back(
      duplicateRoute.getBody().front().back().clone());
  requireVerificationFailure(duplicate.get(),
                             "duplicates a System route sink key");

  mlir::OwningOpRef<mlir::Operation *> sourceAsSink(authored->clone());
  auto sourceRoot = mlir::cast<::mapping::SystemOp>(sourceAsSink.get());
  auto sourceService = *sourceRoot.getBody()
                            .front()
                            .getOps<::mapping::ServiceRealizationOp>()
                            .begin();
  auto sourcePlan = *sourceService.getBody()
                         .front()
                         .getOps<::mapping::ServicePlanOp>()
                         .begin();
  auto sourceRoute = *sourcePlan.getBody()
                          .front()
                          .getOps<::mapping::TransferLegRealizationOp>()
                          .begin();
  auto sourceSink = *sourceRoute.getBody()
                         .front()
                         .getOps<::mapping::SystemRouteSinkOp>()
                         .begin();
  sourceSink.setTerminalAttr(::mapping::SystemTransferTerminalKeyAttr::get(
      &context,
      denseBytes(&context,
                 take(loom::mapping::encodeSystemTransferTerminalKey(
                     fixture.dataflowIdentity,
                     loom::mapping::SystemTransferTerminalKey(
                         loom::mapping::SystemTransferSourceTerminalKey{
                             fixture.leg}))))));
  requireVerificationFailure(sourceAsSink.get(),
                             "must name a sink terminal key");

  auto disconnected = buildSystem(context, fixture, 9, 11);
  requireVerificationFailure(disconnected.get(),
                             "references an absent parent node");
  return EXIT_SUCCESS;
}

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

template <typename T>
void requireError(llvm::Expected<T> value, llvm::StringRef message) {
  require(!value, message);
  llvm::consumeError(value.takeError());
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

  builder.setInsertionPointToEnd(&service.getBody().front());
  loom::mapping::ServicePlanSelectionKey selectionKey{
      loom::mapping::ServiceMemberPlanSelectionAnchor{
          dataflow::ServiceMemberRef(dataflow::MessageTransferMemberRef{})},
      loom::mapping::InstructionExecutionContextKey{fixture.core}};
  mlir::OperationState selectionState(
      location, ::mapping::ServicePlanSelectionOp::getOperationName());
  selectionState.addAttribute(
      "key", ::mapping::ServicePlanSelectionKeyAttr::get(
                 &context,
                 denseBytes(&context,
                            take(loom::mapping::encodeServicePlanSelectionKey(
                                fixture.dataflowIdentity, selectionKey)))));
  selectionState.addAttribute(
      "relation_kind",
      ::mapping::SystemBindingRelationKindAttr::get(
          &context, ::mapping::SystemBindingRelationKind::PresburgerPartition));
  selectionState.addAttribute("default_plan_ordinal",
                              builder.getI64IntegerAttr(4));
  selectionState.addRegion();
  auto selection = mlir::cast<::mapping::ServicePlanSelectionOp>(
      builder.create(selectionState));
  selection.getBody().emplaceBlock();
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
  const dataflow::RootedGraphLaunchRef rootedGraph{
      fixture.root,
      dataflow::StaticGraphLaunchRef{fixture.dataflowIdentity,
                                     dataflow::StaticGraphLaunchId(12)}};

  const loom::mapping::ExecutionContextKey instructionContext =
      loom::mapping::InstructionExecutionContextKey{fixture.core};
  const auto instructionContextBytes =
      take(loom::mapping::encodeExecutionContextKey(instructionContext));
  require(take(loom::mapping::decodeExecutionContextKey(
              instructionContextBytes)) == instructionContext,
          "Instruction execution-context key did not round-trip");
  const loom::mapping::ExecutionContextKey spatialContext =
      loom::mapping::SpatialExecutionContextKey{fixture.core, identity(97)};
  const auto spatialContextBytes =
      take(loom::mapping::encodeExecutionContextKey(spatialContext));
  require(take(loom::mapping::decodeExecutionContextKey(spatialContextBytes)) ==
                  spatialContext &&
              spatialContextBytes != instructionContextBytes,
          "Spatial execution-context key did not preserve its identity");
  auto trailingContextBytes = instructionContextBytes;
  trailingContextBytes.push_back(0);
  requireError(loom::mapping::decodeExecutionContextKey(trailingContextBytes),
               "execution-context codec accepted trailing bytes");
  auto unknownContextBytes = instructionContextBytes;
  unknownContextBytes[3] = 2;
  requireError(loom::mapping::decodeExecutionContextKey(unknownContextBytes),
               "execution-context codec accepted an unknown variant");

  const loom::mapping::ServicePlanSelectionKey encodedSelectionKey{
      loom::mapping::ServiceMemberPlanSelectionAnchor{
          dataflow::ServiceMemberRef(dataflow::MessageTransferMemberRef{})},
      instructionContext};
  const auto selectionKeyBytes =
      take(loom::mapping::encodeServicePlanSelectionKey(
          fixture.dataflowIdentity, encodedSelectionKey));
  require(take(loom::mapping::decodeServicePlanSelectionKey(
              selectionKeyBytes, fixture.dataflowIdentity)) ==
              encodedSelectionKey,
          "service-plan selection key did not round-trip");
  const loom::mapping::ServicePlanSelectionKey exposureSelectionKey{
      loom::mapping::MemoryExposurePlanSelectionAnchor{
          dataflow::MemoryExposureRef{rootedGraph, 3}},
      spatialContext};
  const auto exposureSelectionKeyBytes =
      take(loom::mapping::encodeServicePlanSelectionKey(
          fixture.dataflowIdentity, exposureSelectionKey));
  require(take(loom::mapping::decodeServicePlanSelectionKey(
              exposureSelectionKeyBytes, fixture.dataflowIdentity)) ==
                  exposureSelectionKey &&
              exposureSelectionKeyBytes != selectionKeyBytes,
          "exposure selection key did not preserve its closed variants");
  auto trailingSelectionKeyBytes = selectionKeyBytes;
  trailingSelectionKeyBytes.push_back(0);
  requireError(loom::mapping::decodeServicePlanSelectionKey(
                   trailingSelectionKeyBytes, fixture.dataflowIdentity),
               "selection-key codec accepted trailing bytes");
  auto unknownSelectionKeyBytes = selectionKeyBytes;
  unknownSelectionKeyBytes[3] = 2;
  requireError(loom::mapping::decodeServicePlanSelectionKey(
                   unknownSelectionKeyBytes, fixture.dataflowIdentity),
               "selection-key codec accepted an unknown anchor variant");

  auto missingSelection = buildSystem(context, fixture);
  auto missingSelectionRoot =
      mlir::cast<::mapping::SystemOp>(missingSelection.get());
  auto missingSelectionService = *missingSelectionRoot.getBody()
                                      .front()
                                      .getOps<::mapping::ServiceRealizationOp>()
                                      .begin();
  auto selectionToErase = *missingSelectionService.getBody()
                               .front()
                               .getOps<::mapping::ServicePlanSelectionOp>()
                               .begin();
  selectionToErase.erase();
  requireVerificationFailure(missingSelection.get(),
                             "requires at least one ServicePlanSelection");

  auto duplicateSelection = buildSystem(context, fixture);
  auto duplicateSelectionRoot =
      mlir::cast<::mapping::SystemOp>(duplicateSelection.get());
  auto duplicateSelectionService =
      *duplicateSelectionRoot.getBody()
           .front()
           .getOps<::mapping::ServiceRealizationOp>()
           .begin();
  auto selectionToClone = *duplicateSelectionService.getBody()
                               .front()
                               .getOps<::mapping::ServicePlanSelectionOp>()
                               .begin();
  duplicateSelectionService.getBody().front().push_back(
      selectionToClone->clone());
  requireVerificationFailure(duplicateSelection.get(),
                             "duplicates a ServicePlanSelection key");

  auto absentPlan = buildSystem(context, fixture);
  auto absentPlanRoot = mlir::cast<::mapping::SystemOp>(absentPlan.get());
  auto absentPlanService = *absentPlanRoot.getBody()
                                .front()
                                .getOps<::mapping::ServiceRealizationOp>()
                                .begin();
  auto absentPlanSelection = *absentPlanService.getBody()
                                  .front()
                                  .getOps<::mapping::ServicePlanSelectionOp>()
                                  .begin();
  absentPlanSelection->setAttr("default_plan_ordinal",
                               mlir::Builder(&context).getI64IntegerAttr(99));
  requireVerificationFailure(absentPlan.get(),
                             "names an absent ServicePlan ordinal");
  {
    mlir::ScopedDiagnosticHandler capture(
        &context, [](mlir::Diagnostic &) { return mlir::success(); });
    requireError(
        loom::mapping::writeCanonicalSystemMappingAssembly(absentPlanRoot),
        "canonical writer repaired an absent plan ordinal");
  }

  auto unusedPlan = buildSystem(context, fixture);
  auto unusedPlanRoot = mlir::cast<::mapping::SystemOp>(unusedPlan.get());
  auto unusedPlanService = *unusedPlanRoot.getBody()
                                .front()
                                .getOps<::mapping::ServiceRealizationOp>()
                                .begin();
  auto authoredPlan = *unusedPlanService.getBody()
                           .front()
                           .getOps<::mapping::ServicePlanOp>()
                           .begin();
  auto clonedPlan = mlir::cast<::mapping::ServicePlanOp>(authoredPlan->clone());
  clonedPlan.setPlanOrdinalAttr(mlir::Builder(&context).getI64IntegerAttr(8));
  unusedPlanService.getBody().front().push_back(clonedPlan);
  requireVerificationFailure(unusedPlan.get(),
                             "contains an unselected ServicePlan ordinal");

  auto wrongAnchor = buildSystem(context, fixture);
  auto wrongAnchorRoot = mlir::cast<::mapping::SystemOp>(wrongAnchor.get());
  auto wrongAnchorService = *wrongAnchorRoot.getBody()
                                 .front()
                                 .getOps<::mapping::ServiceRealizationOp>()
                                 .begin();
  auto wrongAnchorSelection = *wrongAnchorService.getBody()
                                   .front()
                                   .getOps<::mapping::ServicePlanSelectionOp>()
                                   .begin();
  dataflow::ContextualActorRef actor{
      rootedGraph,
      dataflow::ActorRef{fixture.dataflowIdentity, dataflow::ActorId(13)}};
  loom::mapping::ServicePlanSelectionKey wrongKey{
      loom::mapping::ServiceMemberPlanSelectionAnchor{
          dataflow::ServiceMemberRef(
              dataflow::AddressedMemoryActorMemberRef{actor})},
      instructionContext};
  wrongAnchorSelection.setKeyAttr(::mapping::ServicePlanSelectionKeyAttr::get(
      &context,
      denseBytes(&context, take(loom::mapping::encodeServicePlanSelectionKey(
                               fixture.dataflowIdentity, wrongKey)))));
  requireVerificationFailure(
      wrongAnchor.get(),
      "transfer obligation requires its singleton MessageTransfer anchor");

  auto authored = buildSystem(context, fixture);
  auto first = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(authored.get())));
  auto second = take(loom::mapping::writeCanonicalSystemMappingAssembly(
      mlir::cast<::mapping::SystemOp>(authored.get())));
  require(first.bytes() == second.bytes(),
          "System service route canonicalization is nondeterministic");
  std::string currentText(first.bytes().begin(), first.bytes().end());
  const std::string currentVersion = "version<3, 0>";
  const std::size_t versionPosition = currentText.find(currentVersion);
  require(versionPosition != std::string::npos,
          "canonical SystemMapping does not use version 3.0");
  currentText.replace(versionPosition, currentVersion.size(), "version<2, 0>");
  {
    mlir::ScopedDiagnosticHandler capture(
        &context, [](mlir::Diagnostic &) { return mlir::success(); });
    auto legacy = mlir::parseSourceString<mlir::ModuleOp>(
        "module {\n" + currentText + "}\n", &context);
    require(!legacy, "mapping.system 2.0 was accepted by the 3.0 parser");
  }
  auto duplicatePlanSystem = buildSystem(context, fixture);
  auto duplicatePlanRoot =
      mlir::cast<::mapping::SystemOp>(duplicatePlanSystem.get());
  auto duplicatePlanService = *duplicatePlanRoot.getBody()
                                   .front()
                                   .getOps<::mapping::ServiceRealizationOp>()
                                   .begin();
  auto originalPlan = *duplicatePlanService.getBody()
                           .front()
                           .getOps<::mapping::ServicePlanOp>()
                           .begin();
  auto equivalentPlan =
      mlir::cast<::mapping::ServicePlanOp>(originalPlan->clone());
  equivalentPlan.setPlanOrdinalAttr(
      mlir::Builder(&context).getI64IntegerAttr(8));
  duplicatePlanService.getBody().front().push_back(equivalentPlan);
  auto duplicatePlanSelection =
      *duplicatePlanService.getBody()
           .front()
           .getOps<::mapping::ServicePlanSelectionOp>()
           .begin();
  mlir::OpBuilder duplicateBuilder(&context);
  duplicateBuilder.setInsertionPointToEnd(
      &duplicatePlanSelection.getBody().front());
  ::mapping::ServicePlanPresburgerClauseOp::create(
      duplicateBuilder, duplicateBuilder.getUnknownLoc(),
      duplicateBuilder.getArrayAttr({::mapping::SystemPresburgerCellAttr::get(
          &context, 0, 0, 0, duplicateBuilder.getArrayAttr({}),
          duplicateBuilder.getArrayAttr({}))}),
      8);
  auto deduplicated = take(
      loom::mapping::writeCanonicalSystemMappingAssembly(duplicatePlanRoot));
  require(deduplicated.bytes() == first.bytes(),
          "equivalent service plans were not canonically deduplicated");
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
  auto selection = *service.getBody()
                        .front()
                        .getOps<::mapping::ServicePlanSelectionOp>()
                        .begin();
  auto route = *plan.getBody()
                    .front()
                    .getOps<::mapping::TransferLegRealizationOp>()
                    .begin();
  auto node =
      *route.getBody().front().getOps<::mapping::SystemRouteNodeOp>().begin();
  auto sink =
      *route.getBody().front().getOps<::mapping::SystemRouteSinkOp>().begin();
  require(
      plan.getPlanOrdinal() == 0 &&
          selection->getAttrOfType<mlir::IntegerAttr>("default_plan_ordinal")
                  .getInt() == 0 &&
          node.getNodeOrdinal() == 1 && node.getParentNodeOrdinal() == 0 &&
          sink.getNodeOrdinal() == 1,
      "System service ordinals were not canonically renumbered");
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

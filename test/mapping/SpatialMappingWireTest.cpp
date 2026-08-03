#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/IR/MappingDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "spatial mapping wire test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

std::string byteList(llvm::ArrayRef<std::uint8_t> bytes) {
  std::string text = "[";
  for (auto [ordinal, byte] : llvm::enumerate(bytes)) {
    if (ordinal)
      text += ", ";
    text += std::to_string(static_cast<std::int8_t>(byte));
  }
  text += "]";
  return text;
}

loom::ArtifactIdentity identity(std::uint8_t value) {
  std::array<std::uint8_t, loom::ArtifactIdentity::byteSize> bytes{};
  bytes.fill(value);
  return take(loom::ArtifactIdentity::fromBytes(bytes));
}

std::string identityAttr(const loom::ArtifactIdentity &value) {
  return "#mapping.artifact_identity<" + byteList(value.bytes()) + ">";
}

template <typename Ref>
std::string dataflowAttr(llvm::StringRef spelling,
                         const loom::ArtifactIdentity &owner, const Ref &ref) {
  return "#mapping." + spelling.str() + "<" +
         byteList(take(dataflow::encodeDataflowReference(owner, ref))) + ">";
}

template <typename Ref>
std::string fabricAttr(llvm::StringRef spelling, const Ref &ref) {
  return "#mapping." + spelling.str() + "<" +
         byteList(loom::fabric::canonicalFabricBytes(ref)) + ">";
}

std::string spatialModule(bool duplicateUse, bool missingOwner) {
  const loom::ArtifactIdentity dataflowOwner = identity(17);
  const dataflow::ActorRef actor{dataflowOwner, dataflow::ActorId(7)};
  const dataflow::CanonicalGraphProducerEndpointRef producer =
      dataflow::ActorTokenResultRef{actor, 0};
  const loom::fabric::FabricFuOccurrenceRef occurrence(5);
  const loom::fabric::InstructionContextRef context{
      loom::fabric::FabricPeOccurrenceRef(3), 0};
  const loom::fabric::FabricFuOccurrenceNodeRef operation{
      loom::fabric::FabricFuNodeKind::Op, occurrence, 0};
  const loom::fabric::FabricUsePatternRef pattern{
      loom::fabric::FabricUsePatternOwnerRef(
          loom::fabric::FabricInventoryOwnerRef::of(operation)),
      0};

  const std::uint64_t binding = missingOwner ? 9 : 7;
  const std::string actorEvent =
      "#mapping.actor_transition_event<actor = " +
      dataflowAttr("actor_ref", dataflowOwner, actor) + ", transition = 0>";
  const std::string producerEvent =
      dataflowAttr("graph_producer_endpoint_ref", dataflowOwner, producer);
  const std::string activation =
      "#mapping.spatial_relative_activation<trigger = "
      "#mapping.spatial_event_point<event = " +
      actorEvent +
      ">, release = #mapping.spatial_event_point<event = " + producerEvent +
      ">>";
  const std::string use =
      "    mapping.resource_use owner(#mapping.compute_realization_ref<" +
      std::to_string(binding) + ">) use_site(" +
      fabricAttr("fabric_use_pattern_ref", pattern) + ") activation(" +
      activation + ") parameters([]) sharing([])\n";

  return "module {\n"
         "  mapping.spatial version<2, 0> tech_mapping(" +
         identityAttr(identity(25)) + ") dataflow(" +
         identityAttr(dataflowOwner) + ") fabric(" +
         identityAttr(identity(34)) +
         ") {\n"
         "    mapping.compute_binding realization("
         "#mapping.compute_realization_ref<7>) occurrence(" +
         fabricAttr("fabric_fu_occurrence_ref", occurrence) + ") context(" +
         fabricAttr("instruction_context_ref", context) +
         ") refinements([])\n" + use + (duplicateUse ? use : "") + "  }\n}\n";
}

std::string routeModule(bool canonicalOrdinals, bool duplicateSink,
                        bool missingParent) {
  const loom::ArtifactIdentity dataflowOwner = identity(17);
  const dataflow::ActorRef producerActor{dataflowOwner, dataflow::ActorId(7)};
  const dataflow::ActorRef consumerActor{dataflowOwner, dataflow::ActorId(8)};
  const dataflow::CanonicalGraphProducerEndpointRef producer =
      dataflow::ActorTokenResultRef{producerActor, 0};
  const dataflow::CanonicalGraphConsumerEndpointRef consumer =
      dataflow::ActorTokenOperandRef{consumerActor, 0};

  const loom::fabric::FabricFuOccurrenceRef sourceOwner(5);
  const loom::fabric::FabricFuOccurrenceRef middleOwner(6);
  const loom::fabric::FabricFuOccurrenceRef sinkOwner(7);
  const loom::fabric::FabricTransportEndpointRef source{
      loom::fabric::FabricTransportEndpointOwnerRef::of(sourceOwner), 0};
  const loom::fabric::FabricTransportEndpointRef middle{
      loom::fabric::FabricTransportEndpointOwnerRef::of(middleOwner), 0};
  const loom::fabric::FabricTransportEndpointRef sink{
      loom::fabric::FabricTransportEndpointOwnerRef::of(sinkOwner), 0};
  const auto first =
      loom::fabric::FabricPhysicalTraversalRef::pointConnection(source, middle);
  const auto second =
      loom::fabric::FabricPhysicalTraversalRef::pointConnection(middle, sink);

  const std::uint64_t root = canonicalOrdinals ? 0 : 9;
  const std::uint64_t middleNode = canonicalOrdinals ? 1 : 4;
  const std::uint64_t sinkNode = canonicalOrdinals ? 2 : 7;
  const std::uint64_t secondParent = missingParent ? 23 : middleNode;
  const std::string rootNode = "      mapping.route_node node " +
                               std::to_string(root) + " refinements([])\n";
  const std::string middleRecord =
      "      mapping.route_node node " + std::to_string(middleNode) +
      " parent " + std::to_string(root) + " traversal(" +
      fabricAttr("fabric_physical_traversal_ref", first) +
      ") refinements([])\n";
  const std::string sinkRecord =
      "      mapping.route_node node " + std::to_string(sinkNode) + " parent " +
      std::to_string(secondParent) + " traversal(" +
      fabricAttr("fabric_physical_traversal_ref", second) +
      ") refinements([])\n";
  const std::string attachment =
      "      mapping.route_sink sink(" +
      dataflowAttr("graph_consumer_endpoint_ref", dataflowOwner, consumer) +
      ") node " + std::to_string(sinkNode) + "\n";

  return "module {\n"
         "  mapping.spatial version<2, 0> tech_mapping(" +
         identityAttr(identity(25)) + ") dataflow(" +
         identityAttr(dataflowOwner) + ") fabric(" +
         identityAttr(identity(34)) +
         ") {\n"
         "    mapping.route_tree logical_net(" +
         dataflowAttr("graph_producer_endpoint_ref", dataflowOwner, producer) +
         ") root_endpoint(" +
         fabricAttr("fabric_transport_endpoint_ref", source) + ") {\n" +
         (canonicalOrdinals ? sinkRecord + rootNode + middleRecord
                            : middleRecord + sinkRecord + rootNode) +
         attachment + (duplicateSink ? attachment : "") +
         "    }\n"
         "  }\n"
         "}\n";
}

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext &context,
                                        llvm::StringRef text) {
  return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
}

bool rejected(mlir::MLIRContext &context, llvm::StringRef text) {
  auto module = parse(context, text);
  return !module || mlir::failed(mlir::verify(*module));
}

void testTypedSpatialResourceUse() {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);

  auto module = parse(context, spatialModule(false, false));
  if (!module || mlir::failed(mlir::verify(*module)))
    fail("typed SpatialMapping ResourceUse did not verify");

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  module->print(stream);
  stream.flush();
  auto reparsed = parse(context, printed);
  if (!reparsed || mlir::failed(mlir::verify(*reparsed)))
    fail("typed SpatialMapping ResourceUse did not round trip");
}

void testResourceUseOwnerClosure() {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);

  if (!rejected(context, spatialModule(false, true)))
    fail("ResourceUse accepted an absent ComputeBinding owner");
  if (!rejected(context, spatialModule(true, false)))
    fail("SpatialMapping accepted a duplicate ResourceUse key");
}

void testRouteTreeCanonicalizationAndShape() {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);

  auto authored = parse(context, routeModule(false, false, false));
  auto canonical = parse(context, routeModule(true, false, false));
  if (!authored || !canonical || mlir::failed(mlir::verify(*authored)) ||
      mlir::failed(mlir::verify(*canonical)))
    fail("valid RouteTree fixture did not verify");

  auto authoredRoot = authored->getOps<::mapping::SpatialOp>();
  auto canonicalRoot = canonical->getOps<::mapping::SpatialOp>();
  auto authoredBytes = take(loom::mapping::writeCanonicalSpatialMappingAssembly(
      *authoredRoot.begin()));
  auto canonicalBytes =
      take(loom::mapping::writeCanonicalSpatialMappingAssembly(
          *canonicalRoot.begin()));
  if (!authoredBytes.bytes().equals(canonicalBytes.bytes()))
    fail("RouteTree authoring order changed canonical Spatial bytes");

  if (!rejected(context, routeModule(false, false, true)))
    fail("RouteTree accepted a node whose parent is absent");
  if (!rejected(context, routeModule(false, true, false)))
    fail("RouteTree accepted a duplicate sink obligation");
}

} // namespace

int main() {
  testTypedSpatialResourceUse();
  testResourceUseOwnerClosure();
  testRouteTreeCanonicalizationAndShape();
  llvm::outs() << "spatial mapping wire tests passed\n";
  return 0;
}

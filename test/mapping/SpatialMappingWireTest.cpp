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

enum class ReleaseShape {
  Single,
  CanonicalPair,
  ReversedPair,
  Duplicate,
  CrossLengthPair,
  ReversedCrossLengthPair,
};

std::string spatialModule(bool duplicateUse, bool missingOwner,
                          ReleaseShape releaseShape = ReleaseShape::Single) {
  const loom::ArtifactIdentity dataflowOwner = identity(17);
  const dataflow::ActorRef actor{dataflowOwner, dataflow::ActorId(7)};
  const dataflow::CanonicalGraphProducerEndpointRef producer =
      dataflow::ActorTokenResultRef{actor, 0};
  const dataflow::GraphRef graph{dataflowOwner, dataflow::GraphId(9)};
  const dataflow::CanonicalGraphProducerEndpointRef graphInput =
      dataflow::GraphIngressTokenRef{
          dataflow::GraphValueInputTokenRef{graph, 0}};
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
  const std::string graphInputEvent =
      dataflowAttr("graph_producer_endpoint_ref", dataflowOwner, graphInput);
  const std::string actorPoint =
      "#mapping.spatial_event_point<event = " + actorEvent + ">";
  const std::string producerPoint =
      "#mapping.spatial_event_point<event = " + producerEvent + ">";
  const std::string graphInputPoint =
      "#mapping.spatial_event_point<event = " + graphInputEvent + ">";
  std::string release;
  switch (releaseShape) {
  case ReleaseShape::Single:
    release = producerPoint;
    break;
  case ReleaseShape::CanonicalPair:
    release = actorPoint + ", " + producerPoint;
    break;
  case ReleaseShape::ReversedPair:
    release = producerPoint + ", " + actorPoint;
    break;
  case ReleaseShape::Duplicate:
    release = producerPoint + ", " + producerPoint;
    break;
  case ReleaseShape::CrossLengthPair:
    release = graphInputPoint + ", " + producerPoint;
    break;
  case ReleaseShape::ReversedCrossLengthPair:
    release = producerPoint + ", " + graphInputPoint;
    break;
  }
  const std::string activation =
      "#mapping.spatial_relative_activation<trigger = "
      "#mapping.spatial_event_point<event = " +
      actorEvent + ">, release = [" + release + "]>";
  const std::string use =
      "    mapping.resource_use owner(#mapping.compute_realization_ref<" +
      std::to_string(binding) + ">) use_site(" +
      fabricAttr("fabric_use_pattern_ref", pattern) + ") activation(" +
      activation + ") parameters([]) sharing([])\n";

  return "module {\n"
         "  mapping.spatial version<6, 0> tech_mapping(" +
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
         "  mapping.spatial version<6, 0> tech_mapping(" +
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

std::string memoryBindingModule(bool localRegion, bool zeroSizedRange,
                                bool duplicateBinding) {
  const loom::ArtifactIdentity dataflowOwner = identity(17);
  const dataflow::LogicalMemoryRootRef firstRoot{
      dataflowOwner, dataflow::LogicalMemoryRootId(3)};
  const dataflow::LogicalMemoryRootRef secondRoot{
      dataflowOwner, dataflow::LogicalMemoryRootId(4)};
  const dataflow::LogicalMemoryRootOrViewRef firstMemory(firstRoot);
  const dataflow::LogicalMemoryRootOrViewRef secondMemory(secondRoot);
  const loom::fabric::FabricMemoryOccurrenceRef occurrence(11);
  const loom::fabric::FabricMemoryServiceRegionRef serviceRegion{
      loom::fabric::FabricMemoryServiceRef::local(occurrence), 2};

  const std::string firstInterval =
      localRegion
          ? "#mapping.memory_byte_range<offset_bytes = 8, size_bytes = " +
                std::to_string(zeroSizedRange ? 0 : 16) + ">"
          : "#mapping.memory_whole_interval";
  const std::string firstTarget =
      localRegion
          ? "#mapping.memory_local_region<service_region = " +
                fabricAttr("fabric_memory_service_region_ref", serviceRegion) +
                ", physical_offset_bytes = 32>"
          : "#mapping.memory_boundary_proxy";
  const std::string first = "    mapping.memory_binding 7 logical_memory(" +
                            dataflowAttr("logical_memory_root_or_view_ref",
                                         dataflowOwner, firstMemory) +
                            ") interval(" + firstInterval + ") target(" +
                            firstTarget + ") {}\n";
  const std::string second = "    mapping.memory_binding " +
                             std::to_string(duplicateBinding ? 7 : 8) +
                             " logical_memory(" +
                             dataflowAttr("logical_memory_root_or_view_ref",
                                          dataflowOwner, secondMemory) +
                             ") interval(#mapping.memory_whole_interval) "
                             "target(#mapping.memory_boundary_proxy) {}\n";

  return "module {\n"
         "  mapping.spatial version<6, 0> tech_mapping(" +
         identityAttr(identity(25)) + ") dataflow(" +
         identityAttr(dataflowOwner) + ") fabric(" +
         identityAttr(identity(34)) + ") {\n" + first + second +
         "  }\n"
         "}\n";
}

std::string memoryBindingCanonicalModule(bool reverseAuthoringOrder) {
  const loom::ArtifactIdentity dataflowOwner = identity(17);
  const dataflow::LogicalMemoryRootOrViewRef firstMemory(
      dataflow::LogicalMemoryRootRef{dataflowOwner,
                                     dataflow::LogicalMemoryRootId(3)});
  const dataflow::LogicalMemoryRootOrViewRef secondMemory(
      dataflow::LogicalMemoryRootRef{dataflowOwner,
                                     dataflow::LogicalMemoryRootId(4)});
  const loom::fabric::FabricMemoryServiceRegionRef serviceRegion{
      loom::fabric::FabricMemoryServiceRef::local(
          loom::fabric::FabricMemoryOccurrenceRef(11)),
      2};
  const std::string first =
      "    mapping.memory_binding " +
      std::to_string(reverseAuthoringOrder ? 91 : 0) + " logical_memory(" +
      dataflowAttr("logical_memory_root_or_view_ref", dataflowOwner,
                   firstMemory) +
      ") interval(#mapping.memory_byte_range<offset_bytes = 8, "
      "size_bytes = 16>) target(#mapping.memory_local_region<service_region "
      "= " +
      fabricAttr("fabric_memory_service_region_ref", serviceRegion) +
      ", physical_offset_bytes = 32>) {}\n";
  const std::string second = "    mapping.memory_binding " +
                             std::to_string(reverseAuthoringOrder ? 27 : 1) +
                             " logical_memory(" +
                             dataflowAttr("logical_memory_root_or_view_ref",
                                          dataflowOwner, secondMemory) +
                             ") interval(#mapping.memory_whole_interval) "
                             "target(#mapping.memory_boundary_proxy) {}\n";
  const std::string body =
      reverseAuthoringOrder ? second + first : first + second;
  return "module {\n"
         "  mapping.spatial version<6, 0> tech_mapping(" +
         identityAttr(identity(25)) + ") dataflow(" +
         identityAttr(dataflowOwner) + ") fabric(" +
         identityAttr(identity(34)) + ") {\n" + body +
         "  }\n"
         "}\n";
}

std::string memoryOperationModule(bool localBinding, bool localDispatch,
                                  bool reverseUses = false,
                                  bool duplicateUse = false,
                                  bool emptyUses = false) {
  const loom::ArtifactIdentity dataflowOwner = identity(17);
  const dataflow::ActorRef actor{dataflowOwner, dataflow::ActorId(5)};
  const dataflow::LogicalMemoryRootOrViewRef firstMemory(
      dataflow::LogicalMemoryRootRef{dataflowOwner,
                                     dataflow::LogicalMemoryRootId(3)});
  const dataflow::LogicalMemoryRootOrViewRef secondMemory(
      dataflow::LogicalMemoryRootRef{dataflowOwner,
                                     dataflow::LogicalMemoryRootId(4)});
  const dataflow::RootedGraphLaunchRef firstLaunch{
      dataflow::RootThreadLaunchRef{dataflowOwner,
                                    dataflow::RootThreadLaunchId(1)},
      dataflow::StaticGraphLaunchRef{dataflowOwner,
                                     dataflow::StaticGraphLaunchId(2)}};
  const dataflow::RootedGraphLaunchRef secondLaunch{
      dataflow::RootThreadLaunchRef{dataflowOwner,
                                    dataflow::RootThreadLaunchId(1)},
      dataflow::StaticGraphLaunchRef{dataflowOwner,
                                     dataflow::StaticGraphLaunchId(3)}};
  const loom::fabric::FabricMemoryOccurrenceRef occurrence(11);
  const loom::fabric::FabricMemoryOperationPortRef port{occurrence, 0};
  const loom::fabric::FabricMemoryServiceRef service =
      loom::fabric::FabricMemoryServiceRef::local(occurrence);
  const loom::fabric::FabricMemoryServiceRegionRef serviceRegion{service, 2};
  const loom::fabric::LocalMemoryServiceRef localService(service);
  const loom::fabric::ManagerEndpointRef manager(
      loom::fabric::FabricMemoryEndpointRef{
          loom::fabric::FabricMemoryEndpointOwnerRef::of(occurrence), 0});
  const std::string target =
      localBinding
          ? "#mapping.memory_local_region<service_region = " +
                fabricAttr("fabric_memory_service_region_ref", serviceRegion) +
                ", physical_offset_bytes = 0>"
          : "#mapping.memory_boundary_proxy";
  const std::string dispatch =
      localDispatch ? fabricAttr("local_memory_service_ref", localService)
                    : fabricAttr("manager_endpoint_ref", manager);
  auto use = [&](const dataflow::RootedGraphLaunchRef &launch,
                 std::uint64_t binding) {
    return "        mapping.addressed_memory_use launch(" +
           dataflowAttr("rooted_graph_launch_ref", dataflowOwner, launch) +
           ") binding(#mapping.memory_binding_ref<" + std::to_string(binding) +
           ">) dispatch(" + dispatch + ")\n";
  };
  const std::string firstUse = use(firstLaunch, 7);
  const std::string secondUse = use(secondLaunch, 8);
  const std::string uses =
      emptyUses ? ""
                : (reverseUses ? secondUse + firstUse : firstUse + secondUse) +
                      (duplicateUse ? firstUse : "");

  return "module {\n"
         "  mapping.spatial version<6, 0> tech_mapping(" +
         identityAttr(identity(25)) + ") dataflow(" +
         identityAttr(dataflowOwner) + ") fabric(" +
         identityAttr(identity(34)) +
         ") {\n"
         "    mapping.memory_engine_binding realization("
         "#mapping.memory_realization_ref<2>) occurrence(" +
         fabricAttr("fabric_memory_occurrence_ref", occurrence) +
         ") {\n"
         "      mapping.addressed_memory_operation actor(" +
         dataflowAttr("actor_ref", dataflowOwner, actor) + ") placement(" +
         fabricAttr("fabric_memory_operation_port_ref", port) + ") {\n" + uses +
         "      }\n"
         "    }\n"
         "    mapping.memory_binding 7 logical_memory(" +
         dataflowAttr("logical_memory_root_or_view_ref", dataflowOwner,
                      firstMemory) +
         ") interval(#mapping.memory_byte_range<offset_bytes = 0, "
         "size_bytes = 16>) target(" +
         target +
         ") {}\n"
         "    mapping.memory_binding 8 logical_memory(" +
         dataflowAttr("logical_memory_root_or_view_ref", dataflowOwner,
                      secondMemory) +
         ") interval(#mapping.memory_byte_range<offset_bytes = 0, "
         "size_bytes = 16>) target(" +
         target +
         ") {}\n"
         "  }\n"
         "}\n";
}

std::string fenceOperationModule(bool reverseUses, bool duplicateUse,
                                 bool emptyUses) {
  const loom::ArtifactIdentity dataflowOwner = identity(17);
  const dataflow::ActorRef actor{dataflowOwner, dataflow::ActorId(6)};
  const dataflow::RootedGraphLaunchRef firstLaunch{
      dataflow::RootThreadLaunchRef{dataflowOwner,
                                    dataflow::RootThreadLaunchId(1)},
      dataflow::StaticGraphLaunchRef{dataflowOwner,
                                     dataflow::StaticGraphLaunchId(2)}};
  const dataflow::RootedGraphLaunchRef secondLaunch{
      dataflow::RootThreadLaunchRef{dataflowOwner,
                                    dataflow::RootThreadLaunchId(1)},
      dataflow::StaticGraphLaunchRef{dataflowOwner,
                                     dataflow::StaticGraphLaunchId(3)}};
  const loom::fabric::FabricMemoryOccurrenceRef occurrence(11);
  const loom::fabric::FabricMemoryOperationPortRef port{occurrence, 0};
  const loom::fabric::MemoryConsistencyDomainRef consistency(
      loom::fabric::HardwareDomainRef(19));
  auto use = [&](const dataflow::RootedGraphLaunchRef &launch) {
    return "        mapping.fence_memory_use launch(" +
           dataflowAttr("rooted_graph_launch_ref", dataflowOwner, launch) +
           ") consistency(" +
           fabricAttr("memory_consistency_domain_ref", consistency) + ")\n";
  };
  const std::string firstUse = use(firstLaunch);
  const std::string secondUse = use(secondLaunch);
  const std::string uses =
      emptyUses ? ""
                : (reverseUses ? secondUse + firstUse : firstUse + secondUse) +
                      (duplicateUse ? firstUse : "");

  return "module {\n"
         "  mapping.spatial version<6, 0> tech_mapping(" +
         identityAttr(identity(25)) + ") dataflow(" +
         identityAttr(dataflowOwner) + ") fabric(" +
         identityAttr(identity(34)) +
         ") {\n"
         "    mapping.memory_engine_binding realization("
         "#mapping.memory_realization_ref<2>) occurrence(" +
         fabricAttr("fabric_memory_occurrence_ref", occurrence) +
         ") {\n"
         "      mapping.fence_memory_operation actor(" +
         dataflowAttr("actor_ref", dataflowOwner, actor) + ") placement(" +
         fabricAttr("fabric_memory_operation_port_ref", port) + ") {\n" + uses +
         "      }\n"
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

  std::string legacy = spatialModule(false, false);
  const std::string current = "version<6, 0>";
  const std::size_t position = legacy.find(current);
  if (position == std::string::npos)
    fail("SpatialMapping version fixture has no current version");
  legacy.replace(position, current.size(), "version<5, 0>");
  if (parse(context, legacy))
    fail("mapping.spatial 5.0 was accepted by the 6.0 parser");

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  module->print(stream);
  stream.flush();
  auto reparsed = parse(context, printed);
  if (!reparsed || mlir::failed(mlir::verify(*reparsed)))
    fail("typed SpatialMapping ResourceUse did not round trip");

  auto conjunctive =
      parse(context, spatialModule(false, false, ReleaseShape::CanonicalPair));
  if (!conjunctive || mlir::failed(mlir::verify(*conjunctive)))
    fail("canonical conjunctive release did not verify");
  if (!rejected(context,
                spatialModule(false, false, ReleaseShape::ReversedPair)))
    fail("conjunctive release accepted noncanonical member order");
  if (!rejected(context, spatialModule(false, false, ReleaseShape::Duplicate)))
    fail("conjunctive release accepted a duplicate event point");

  auto crossLength = parse(
      context, spatialModule(false, false, ReleaseShape::CrossLengthPair));
  if (!crossLength || mlir::failed(mlir::verify(*crossLength)))
    fail("canonical release ordered by exact reference bytes did not verify");
  if (!rejected(context, spatialModule(false, false,
                                       ReleaseShape::ReversedCrossLengthPair)))
    fail("release ordering used framed reference lengths as an authority");

  auto singleRoot = module->getOps<::mapping::SpatialOp>();
  auto conjunctiveRoot = conjunctive->getOps<::mapping::SpatialOp>();
  auto singleBytes = take(
      loom::mapping::writeCanonicalSpatialMappingAssembly(*singleRoot.begin()));
  auto conjunctiveBytes =
      take(loom::mapping::writeCanonicalSpatialMappingAssembly(
          *conjunctiveRoot.begin()));
  if (singleBytes.bytes().equals(conjunctiveBytes.bytes()))
    fail("release membership did not affect canonical Spatial identity bytes");
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

void testMemoryBindingTargetWire() {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);

  for (bool localRegion : {false, true}) {
    auto module =
        parse(context, memoryBindingModule(localRegion, false, false));
    if (!module || mlir::failed(mlir::verify(*module)))
      fail(localRegion ? "valid LocalRegion MemoryBinding did not verify"
                       : "valid BoundaryProxy MemoryBinding did not verify");

    std::string printed;
    llvm::raw_string_ostream stream(printed);
    module->print(stream);
    stream.flush();
    auto reparsed = parse(context, printed);
    if (!reparsed || mlir::failed(mlir::verify(*reparsed)))
      fail("MemoryBinding target did not round trip");
  }

  if (!rejected(context, memoryBindingModule(true, true, false)))
    fail("MemoryBinding accepted a zero-sized ByteRange");
  if (!rejected(context, memoryBindingModule(false, false, true)))
    fail("SpatialMapping accepted a duplicate MemoryBinding EntityId");

  for (bool local : {false, true}) {
    auto module = parse(context, memoryOperationModule(local, local));
    if (!module || mlir::failed(mlir::verify(*module)))
      fail("valid memory operation dispatch did not verify");
    if (!rejected(context, memoryOperationModule(local, !local)))
      fail("memory operation accepted a dispatch/binding target mismatch");
  }
  if (!rejected(context,
                memoryOperationModule(false, false, false, true, false)))
    fail("memory operation accepted a duplicate rooted use");
  if (!rejected(context,
                memoryOperationModule(false, false, false, false, true)))
    fail("memory operation accepted an empty rooted-use inventory");

  auto authoredUses = parse(context, memoryOperationModule(false, false, true));
  auto canonicalUses =
      parse(context, memoryOperationModule(false, false, false));
  if (!authoredUses || !canonicalUses ||
      mlir::failed(mlir::verify(*authoredUses)) ||
      mlir::failed(mlir::verify(*canonicalUses)))
    fail("rooted memory-use canonicalization fixture did not verify");
  auto authoredUseRoot = authoredUses->getOps<::mapping::SpatialOp>();
  auto canonicalUseRoot = canonicalUses->getOps<::mapping::SpatialOp>();
  auto authoredUseBytes =
      take(loom::mapping::writeCanonicalSpatialMappingAssembly(
          *authoredUseRoot.begin()));
  auto canonicalUseBytes =
      take(loom::mapping::writeCanonicalSpatialMappingAssembly(
          *canonicalUseRoot.begin()));
  if (!authoredUseBytes.bytes().equals(canonicalUseBytes.bytes()))
    fail("rooted memory-use authoring order changed canonical bytes");

  auto authoredFence = parse(context, fenceOperationModule(true, false, false));
  auto canonicalFence =
      parse(context, fenceOperationModule(false, false, false));
  if (!authoredFence || !canonicalFence ||
      mlir::failed(mlir::verify(*authoredFence)) ||
      mlir::failed(mlir::verify(*canonicalFence)))
    fail("rooted fence-use canonicalization fixture did not verify");
  auto authoredFenceRoot = authoredFence->getOps<::mapping::SpatialOp>();
  auto canonicalFenceRoot = canonicalFence->getOps<::mapping::SpatialOp>();
  auto authoredFenceBytes =
      take(loom::mapping::writeCanonicalSpatialMappingAssembly(
          *authoredFenceRoot.begin()));
  auto canonicalFenceBytes =
      take(loom::mapping::writeCanonicalSpatialMappingAssembly(
          *canonicalFenceRoot.begin()));
  if (!authoredFenceBytes.bytes().equals(canonicalFenceBytes.bytes()))
    fail("rooted fence-use authoring order changed canonical bytes");
  if (!rejected(context, fenceOperationModule(false, true, false)))
    fail("fence operation accepted a duplicate rooted use");
  if (!rejected(context, fenceOperationModule(false, false, true)))
    fail("fence operation accepted an empty rooted-use inventory");

  auto authored = parse(context, memoryBindingCanonicalModule(true));
  auto canonical = parse(context, memoryBindingCanonicalModule(false));
  if (!authored || !canonical || mlir::failed(mlir::verify(*authored)) ||
      mlir::failed(mlir::verify(*canonical)))
    fail("MemoryBinding canonicalization fixture did not verify");
  auto authoredRoot = authored->getOps<::mapping::SpatialOp>();
  auto canonicalRoot = canonical->getOps<::mapping::SpatialOp>();
  auto authoredBytes = take(loom::mapping::writeCanonicalSpatialMappingAssembly(
      *authoredRoot.begin()));
  auto canonicalBytes =
      take(loom::mapping::writeCanonicalSpatialMappingAssembly(
          *canonicalRoot.begin()));
  if (!authoredBytes.bytes().equals(canonicalBytes.bytes()))
    fail("MemoryBinding authoring order or EntityId changed canonical bytes");
}

} // namespace

int main() {
  testTypedSpatialResourceUse();
  testResourceUseOwnerClosure();
  testRouteTreeCanonicalizationAndShape();
  testMemoryBindingTargetWire();
  llvm::outs() << "spatial mapping wire tests passed\n";
  return 0;
}

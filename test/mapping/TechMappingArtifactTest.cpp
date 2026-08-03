#include "ADG/Builtin.h"

#include "Common/ArtifactStore.h"
#include "Common/ResolvedConfig.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/MappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/IR/MappingOps.h"
#include "PnR/EndpointRouter.h"
#include "PnR/HandshakeCandidateState.h"
#include "PnR/MappingObjective.h"
#include "PnR/PnrConfig.h"
#include "PnR/RouteTreeState.h"
#include "PnR/SpatialCandidateState.h"
#include "PnR/SpatialNetRouter.h"
#include "PnR/SpatialPathFinderRouter.h"
#include "PnR/SpatialPnrProblem.h"
#include "PnR/SpatialRouteCostState.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "tech mapping artifact test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

template <typename T> bool rejected(llvm::Expected<T> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
}

bool rejectedAs(llvm::Expected<loom::pnr::FrozenSpatialPnrProblemHandle> value,
                loom::pnr::SpatialPnrFreezeFailureKind expected) {
  if (value)
    return false;
  bool matched = false;
  llvm::handleAllErrors(
      value.takeError(),
      [&](const loom::pnr::SpatialPnrFreezeFailure &failure) {
        matched = failure.kind() == expected;
      },
      [&](const llvm::ErrorInfoBase &) {});
  return matched;
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-tech-mapping-artifact", path))
      fail("cannot create ArtifactStore directory: " + error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << "cannot remove test directory: " << error.message()
                   << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string path_;
};

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

std::string identityAttr(const loom::ArtifactIdentity &identity) {
  return "#mapping.artifact_identity<" + byteList(identity.bytes()) + ">";
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

mlir::MLIRContext makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::mapping::MappingDialect,
                  mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::func::FuncDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect>();
  return mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
}

dataflow::CanonicalDataflowArtifact buildDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @load_only(
      %start: none, %index: index, %memory: memref<4xi32>) -> (i32, i32)
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 2, 0, 0>} {
    %value, %done = dataflow.load %memory[%index] %start : memref<4xi32>
    dataflow.graph.return values(%value, %value : i32, i32) streams() memories()
        complete(%done : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the canonical Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalDataflowArtifact
buildComputeDataflow(mlir::MLIRContext &context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  dataflow.graph private @synchronize(
      %start: none, %input: i32) -> i32
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result:2 = dataflow.sync %start, %input
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%result#1 : i32) streams() memories()
        complete(%result#0 : none)
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the compute Dataflow fixture");
  return take(dataflow::finalizeCanonicalDataflow(*module));
}

dataflow::CanonicalActorView
memoryActor(const dataflow::CanonicalDataflowProgramView &view) {
  for (const dataflow::CanonicalActorView &actor : view.actors())
    if (actor.kind == dataflow::CanonicalDataflowActorKind::Memory)
      return actor;
  fail("Dataflow fixture has no memory actor");
}

dataflow::CanonicalActorView
computeActor(const dataflow::CanonicalDataflowProgramView &view) {
  for (const dataflow::CanonicalActorView &actor : view.actors())
    if (actor.kind != dataflow::CanonicalDataflowActorKind::Memory)
      return actor;
  fail("Dataflow fixture has no compute actor");
}

struct SelectedComputeCapability final {
  loom::fabric::FabricFuCapabilityTemplateRef capability;
  loom::fabric::FabricFuTemplateNodeRef operation;
  std::vector<std::uint64_t> operandPorts;
  std::vector<std::uint64_t> resultPorts;
  std::vector<loom::fabric::FabricFuTemplatePortRef> operandBoundaries;
  std::vector<loom::fabric::FabricFuTemplatePortRef> resultBoundaries;
};

std::vector<loom::fabric::FabricFuCapabilityTemplateEndpointRef> successors(
    const loom::fabric::FabricFuCapabilityTemplateRecord &record,
    const loom::fabric::FabricFuCapabilityTemplateEndpointRef &endpoint) {
  std::vector<loom::fabric::FabricFuCapabilityTemplateEndpointRef> result;
  for (const auto &edge : record.activeEdges) {
    if (edge.source == endpoint)
      result.push_back(edge.destination);
  }
  const auto *node =
      std::get_if<loom::fabric::FabricFuNodePortRef>(&endpoint.payload);
  if (!node || node->direction != loom::fabric::FabricPortDirection::Input ||
      node->node.node == loom::fabric::FabricFuNodeKind::Op)
    return result;
  for (const auto &edge : record.activeEdges) {
    const auto *source =
        std::get_if<loom::fabric::FabricFuNodePortRef>(&edge.source.payload);
    if (source && source->node == node->node &&
        source->direction == loom::fabric::FabricPortDirection::Output)
      result.push_back(edge.source);
  }
  return result;
}

bool reaches(
    const loom::fabric::FabricFuCapabilityTemplateRecord &record,
    loom::fabric::FabricFuCapabilityTemplateEndpointRef source,
    const loom::fabric::FabricFuCapabilityTemplateEndpointRef &destination) {
  std::vector<loom::fabric::FabricFuCapabilityTemplateEndpointRef> visited = {
      source};
  for (std::size_t index = 0; index < visited.size(); ++index) {
    if (visited[index] == destination)
      return true;
    for (auto next : successors(record, visited[index]))
      if (!llvm::is_contained(visited, next))
        visited.push_back(std::move(next));
  }
  return false;
}

std::optional<loom::fabric::FabricFuTemplatePortRef>
routedBoundary(const loom::fabric::FabricFuCapabilityTemplateRecord &record,
               const loom::fabric::FabricFuTemplateNodeRef &operation,
               loom::fabric::FabricPortDirection direction,
               std::uint64_t portOrdinal) {
  const auto operationEndpoint =
      loom::fabric::FabricFuCapabilityTemplateEndpointRef::nodePort(
          {operation, direction, portOrdinal});
  for (const auto &edge : record.activeEdges) {
    const auto *boundary =
        direction == loom::fabric::FabricPortDirection::Input
            ? std::get_if<loom::fabric::FabricFuTemplatePortRef>(
                  &edge.source.payload)
            : std::get_if<loom::fabric::FabricFuTemplatePortRef>(
                  &edge.destination.payload);
    if (!boundary || boundary->direction != direction)
      continue;
    const auto boundaryEndpoint =
        loom::fabric::FabricFuCapabilityTemplateEndpointRef::boundaryPort(
            *boundary);
    const bool connected =
        direction == loom::fabric::FabricPortDirection::Input
            ? reaches(record, boundaryEndpoint, operationEndpoint)
            : reaches(record, operationEndpoint, boundaryEndpoint);
    if (connected)
      return *boundary;
  }
  return std::nullopt;
}

SelectedComputeCapability
selectComputeCapability(const dataflow::CanonicalActorView &actor,
                        const loom::fabric::FabricArtifactView &fabric) {
  auto projection =
      take(dataflow::projectRegisteredActorSchemaProjection(actor.op));
  for (std::uint64_t id = 0;; ++id) {
    const auto kind = fabric.entityKind(id);
    if (!kind)
      break;
    if (*kind != loom::fabric::FabricEntityKind::FabricFuTemplate)
      continue;
    const loom::fabric::FabricFuTemplateRef fu(id);
    for (auto [templateOrdinal, record] :
         llvm::enumerate(fabric.fuCapabilityTemplates(fu))) {
      for (const auto &node : record.activeNodes) {
        if (node.node != loom::fabric::FabricFuNodeKind::Op)
          continue;
        const auto *capability = fabric.resolvedFabricOpCapability(node);
        if (!capability)
          continue;
        if (llvm::Error error = capability->admit(projection, 64)) {
          llvm::consumeError(std::move(error));
          continue;
        }

        SelectedComputeCapability selected{
            {fu, templateOrdinal}, node, {}, {}, {}, {}};
        for (const auto &port : capability->physicalPorts) {
          auto *ports = port.reference.direction ==
                                loom::fabric::FabricPortDirection::Input
                            ? &selected.operandPorts
                            : &selected.resultPorts;
          auto *boundaries = port.reference.direction ==
                                     loom::fabric::FabricPortDirection::Input
                                 ? &selected.operandBoundaries
                                 : &selected.resultBoundaries;
          const std::size_t required =
              port.reference.direction ==
                      loom::fabric::FabricPortDirection::Input
                  ? projection.type.getNumInputs()
                  : projection.type.getNumResults();
          if (ports->size() >= required)
            continue;
          auto boundary = routedBoundary(record, node, port.reference.direction,
                                         port.reference.ordinal);
          if (!boundary)
            continue;
          ports->push_back(port.reference.ordinal);
          boundaries->push_back(*boundary);
        }
        if (selected.operandPorts.size() == projection.type.getNumInputs() &&
            selected.resultPorts.size() == projection.type.getNumResults())
          return selected;
      }
    }
  }
  fail("builtin SpatialCore has no routed compute capability");
}

struct SelectedMemoryCapability final {
  loom::fabric::FabricMemoryEngineTemplateRef engine;
  loom::fabric::FabricMemoryEngineTemplateOperationPortRef port;
  loom::fabric::FabricMemoryEngineTemplateCapabilityAlternativeRef alternative;
  std::vector<loom::fabric::FabricMemoryEngineTemplateEndpointRef> arguments;
  std::vector<loom::fabric::FabricMemoryEngineTemplateEndpointRef> results;
};

SelectedMemoryCapability
selectMemoryCapability(const dataflow::CanonicalActorView &actor,
                       const loom::fabric::FabricArtifactView &fabric) {
  auto projection =
      take(dataflow::projectRegisteredActorSchemaProjection(actor.op));
  auto service =
      take(dataflow::semantics::CanonicalService::forActor(actor.op));
  auto access =
      take(dataflow::semantics::getCanonicalMemoryAccessView(actor.op));

  for (std::uint64_t id = 0;; ++id) {
    const auto kind = fabric.entityKind(id);
    if (!kind)
      break;
    if (*kind != loom::fabric::FabricEntityKind::FabricMemoryEngineTemplate)
      continue;
    const loom::fabric::FabricMemoryEngineTemplateRef engine(id);
    const auto *record = fabric.memoryEngineTemplate(engine);
    if (!record)
      continue;
    for (std::uint64_t portOrdinal = 0;
         portOrdinal < record->operationPorts.size(); ++portOrdinal) {
      const loom::fabric::FabricMemoryEngineTemplateOperationPortRef port{
          engine, portOrdinal};
      const auto *portView = fabric.memoryEngineTemplateOperationPort(port);
      auto matches = take(portView->matchingCapabilities(
          projection, service, std::optional{access}));
      if (matches.empty())
        continue;
      const auto alternative =
          loom::fabric::FabricMemoryEngineTemplateCapabilityAlternativeRef{
              port, matches.front().alternativeOrdinal};
      const auto *capability =
          fabric.memoryEngineTemplateCapabilityAlternative(alternative);
      if (!capability)
        fail("selected memory capability does not resolve");

      const auto endpointFor = [&](dataflow::semantics::ServiceValueRole role) {
        for (const auto &binding : capability->roleToEndpoint)
          if (binding.role == role)
            return loom::fabric::FabricMemoryEngineTemplateEndpointRef{
                engine, binding.endpointOrdinal};
        fail("selected memory capability omits a service role");
      };
      SelectedMemoryCapability selected{engine, port, alternative, {}, {}};
      for (const auto &value : service.arguments())
        selected.arguments.push_back(endpointFor(value.role));
      for (const auto &value : service.results())
        selected.results.push_back(endpointFor(value.role));
      return selected;
    }
  }
  fail("builtin SpatialCore has no matching memory capability");
}

std::string endpointArray(
    llvm::ArrayRef<loom::fabric::FabricMemoryEngineTemplateEndpointRef>
        endpoints,
    bool staleFirst = false) {
  std::string text = "[";
  for (auto [ordinal, source] : llvm::enumerate(endpoints)) {
    if (ordinal)
      text += ", ";
    auto endpoint = source;
    if (ordinal == 0 && staleFirst)
      endpoint.ordinal += 4096;
    text += fabricAttr("fabric_memory_engine_template_endpoint_ref", endpoint);
  }
  text += "]";
  return text;
}

std::string mappingText(const dataflow::CanonicalDataflowProgramView &dataflow,
                        const loom::fabric::FabricArtifactView &fabric,
                        const SelectedMemoryCapability &selected,
                        bool staleOperand) {
  const dataflow::GraphRef graph = dataflow.graphs().front().ref;
  const dataflow::ActorRef actor = memoryActor(dataflow).ref;

  const auto endpointFor = [&](dataflow::semantics::ServiceValueRole role) {
    const auto *capability =
        fabric.memoryEngineTemplateCapabilityAlternative(selected.alternative);
    for (const auto &binding : capability->roleToEndpoint)
      if (binding.role == role)
        return loom::fabric::FabricMemoryEngineTemplateEndpointRef{
            selected.engine, binding.endpointOrdinal};
    fail("selected memory capability omits a boundary role");
  };

  const auto producer =
      [&](const dataflow::CanonicalGraphProducerEndpointRef &reference) {
        return dataflowAttr("graph_producer_endpoint_ref", dataflow.identity(),
                            reference);
      };
  const auto consumer =
      [&](const dataflow::CanonicalGraphConsumerEndpointRef &reference) {
        return dataflowAttr("graph_consumer_endpoint_ref", dataflow.identity(),
                            reference);
      };
  const auto boundary = [&](const std::string &terminal,
                            dataflow::semantics::ServiceValueRole role) {
    return "        mapping.memory_graph_boundary terminal(" + terminal +
           ") endpoint(" +
           fabricAttr("fabric_memory_engine_template_endpoint_ref",
                      endpointFor(role)) +
           ")\n";
  };

  std::string children;
  children +=
      "        mapping.memory_actor actor(" +
      dataflowAttr("actor_ref", dataflow.identity(), actor) +
      ") operation_port(" +
      fabricAttr("fabric_memory_engine_template_operation_port_ref",
                 selected.port) +
      ") capability(" +
      fabricAttr("fabric_memory_engine_template_capability_alternative_ref",
                 selected.alternative) +
      ") operand_ports(" + endpointArray(selected.arguments, staleOperand) +
      ") result_ports(" + endpointArray(selected.results) + ")\n";
  children += boundary(
      producer(dataflow::CanonicalGraphProducerEndpointRef{
          dataflow::GraphIngressTokenRef{dataflow::GraphStartTokenRef{graph}}}),
      dataflow::semantics::ServiceValueRole::Control);
  children += boundary(producer(dataflow::CanonicalGraphProducerEndpointRef{
                           dataflow::GraphIngressTokenRef{
                               dataflow::GraphValueInputTokenRef{graph, 0}}}),
                       dataflow::semantics::ServiceValueRole::Address);
  children += boundary(consumer(dataflow::CanonicalGraphConsumerEndpointRef{
                           dataflow::GraphEgressTokenRef{
                               dataflow::GraphValueOutputTokenRef{graph, 0}}}),
                       dataflow::semantics::ServiceValueRole::Data);
  children += boundary(consumer(dataflow::CanonicalGraphConsumerEndpointRef{
                           dataflow::GraphEgressTokenRef{
                               dataflow::GraphValueOutputTokenRef{graph, 1}}}),
                       dataflow::semantics::ServiceValueRole::Data);
  children +=
      boundary(consumer(dataflow::CanonicalGraphConsumerEndpointRef{
                   dataflow::GraphEgressTokenRef{
                       dataflow::GraphCompletionFrontierTokenRef{graph, 0}}}),
               dataflow::semantics::ServiceValueRole::Completion);

  return "module {\n  mapping.tech version<2, 0> dataflow(" +
         identityAttr(dataflow.identity()) + ") fabric(" +
         identityAttr(fabric.identity()) + ") covers([" +
         dataflowAttr("graph_ref", dataflow.identity(), graph) +
         "]) {\n      mapping.memory_realization 17 engine(" +
         fabricAttr("fabric_memory_engine_template_ref", selected.engine) +
         ") {\n" + children + "      }\n  }\n}\n";
}

std::string ordinalArray(llvm::ArrayRef<std::uint64_t> values) {
  std::string text = "[";
  for (auto [index, value] : llvm::enumerate(values)) {
    if (index)
      text += ", ";
    text += std::to_string(value);
  }
  text += "]";
  return text;
}

std::string
computeMappingText(const dataflow::CanonicalDataflowProgramView &dataflow,
                   const loom::fabric::FabricArtifactView &fabric,
                   const SelectedComputeCapability &selected,
                   bool includeBoundaries) {
  const dataflow::GraphRef graph = dataflow.graphs().front().ref;
  const dataflow::ActorRef actor = computeActor(dataflow).ref;
  std::string children =
      "        mapping.compute_actor actor(" +
      dataflowAttr("actor_ref", dataflow.identity(), actor) + ") op(" +
      fabricAttr("fabric_fu_template_node_ref", selected.operation) +
      ") operand_ports(" + ordinalArray(selected.operandPorts) +
      ") result_ports(" + ordinalArray(selected.resultPorts) + ")\n";
  if (includeBoundaries) {
    for (auto [ordinal, port] : llvm::enumerate(selected.operandBoundaries))
      children += "        mapping.compute_boundary actor(" +
                  dataflowAttr("actor_ref", dataflow.identity(), actor) +
                  ") input " + std::to_string(ordinal) + " fu_port(" +
                  fabricAttr("fabric_fu_template_port_ref", port) + ")\n";
    for (auto [ordinal, port] : llvm::enumerate(selected.resultBoundaries))
      children += "        mapping.compute_boundary actor(" +
                  dataflowAttr("actor_ref", dataflow.identity(), actor) +
                  ") output " + std::to_string(ordinal) + " fu_port(" +
                  fabricAttr("fabric_fu_template_port_ref", port) + ")\n";
  }
  return "module {\n  mapping.tech version<2, 0> dataflow(" +
         identityAttr(dataflow.identity()) + ") fabric(" +
         identityAttr(fabric.identity()) + ") covers([" +
         dataflowAttr("graph_ref", dataflow.identity(), graph) +
         "]) {\n      mapping.compute_realization 23 capability(" +
         fabricAttr("fabric_fu_capability_template_ref", selected.capability) +
         ") {\n" + children + "      }\n  }\n}\n";
}

mlir::OwningOpRef<mlir::ModuleOp> parseMapping(mlir::MLIRContext &context,
                                               llvm::StringRef text) {
  return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
}

std::string
spatialConstraintText(const dataflow::CanonicalDataflowProgramView &dataflow,
                      const loom::mapping::TechMappingView &techMapping,
                      const loom::fabric::FabricArtifactView &fabric,
                      llvm::StringRef clauses) {
  return "module {\n  mapping.constraints.spatial dataflow(" +
         identityAttr(dataflow.identity()) + ") tech_mapping(" +
         identityAttr(techMapping.identity()) + ") fabric(" +
         identityAttr(fabric.identity()) + ") {\n" + clauses.str() + "  }\n}\n";
}

void artifactRoundTripAndReferenceValidation() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildDataflow(context);
  auto dataflowReference =
      take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflowView = take(dataflowArtifact.view());

  loom::adg::DesignBuilder builder(store);
  auto expansion = take(loom::adg::expandBuiltinSpatialCore(
      builder, loom::adg::BuiltinTargetPreset::Small));
  if (llvm::Error error = expansion.spatialCore.close(expansion.outputs))
    fail(llvm::toString(std::move(error)));
  auto design = take(std::move(builder).finalize());
  if (design.roots().size() != 1)
    fail("builtin SpatialCore did not publish exactly one Fabric root");
  const auto &fabricRoot = design.roots().front();
  const auto selected =
      selectMemoryCapability(memoryActor(dataflowView), fabricRoot.view());

  auto module = parseMapping(
      context, mappingText(dataflowView, fabricRoot.view(), selected, false));
  if (!module)
    fail("valid TechMapping fixture did not parse");
  auto roots = module->getOps<::mapping::TechOp>();
  auto finalized = take(loom::mapping::finalizeTechMapping(
      *roots.begin(), dataflowView, fabricRoot.view(), store));
  if (finalized.view().dataflowIdentity() != dataflowReference.artifact ||
      finalized.view().fabricIdentity() != fabricRoot.reference().artifact ||
      finalized.view().memoryRealizations().size() != 1)
    fail("sealed TechMapping view lost its exact upstream binding");
  if (finalized.view().residualLogicalNets().size() != 4)
    fail("sealed TechMapping view omitted a residual logical net");
  for (const auto &net : finalized.view().residualLogicalNets()) {
    auto consumers = take(dataflowView.graphConsumers(net.producer));
    if (net.sinks.empty() || !llvm::equal(net.sinks, consumers))
      fail("sealed TechMapping view changed a residual sink relation");
  }

  TemporaryDirectory emptyDirectory;
  loom::ArtifactStore emptyStore(emptyDirectory.path());
  if (!rejected(loom::mapping::finalizeTechMapping(
          *roots.begin(), dataflowView, fabricRoot.view(), emptyStore)))
    fail("sealed upstream views bypassed durable publication");

  auto imported =
      take(loom::mapping::importTechMapping(finalized.reference(), store));
  if (imported.reference() != finalized.reference() ||
      !imported.canonicalBytes().bytes().equals(
          finalized.canonicalBytes().bytes()) ||
      imported.view().memoryRealizations().front().actors.size() != 1)
    fail("strict TechMapping import changed the canonical artifact");

  auto emptyConstraints = parseMapping(
      context,
      spatialConstraintText(dataflowView, finalized.view(), fabricRoot.view(),
                            /*clauses=*/""));
  if (!emptyConstraints)
    fail("empty Spatial MappingConstraintSet fixture did not parse");
  auto constraintRoots =
      emptyConstraints->getOps<::mapping::ConstraintsSpatialOp>();
  auto finalizedConstraints =
      take(loom::mapping::finalizeSpatialMappingConstraintSet(
          *constraintRoots.begin(), dataflowView, finalized.view(),
          fabricRoot.view(), store));
  if (finalizedConstraints.view().dataflowIdentity() !=
          dataflowReference.artifact ||
      finalizedConstraints.view().techMappingIdentity() !=
          finalized.reference().artifact ||
      finalizedConstraints.view().fabricIdentity() !=
          fabricRoot.reference().artifact ||
      !finalizedConstraints.view().clauses().empty())
    fail("sealed Spatial MappingConstraintSet lost its exact empty binding");

  auto importedConstraints =
      take(loom::mapping::importSpatialMappingConstraintSet(
          finalizedConstraints.reference(), store));
  if (importedConstraints.reference() != finalizedConstraints.reference() ||
      !importedConstraints.canonicalBytes().bytes().equals(
          finalizedConstraints.canonicalBytes().bytes()))
    fail("strict Spatial MappingConstraintSet import changed the artifact");

  const loom::pnr::ResolvedPnrConfigView spatialConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(
          loom::defaultResolvedConfig()));
  loom::pnr::FrozenSpatialPnrProblemHandle frozen =
      take(loom::pnr::freezeSpatialPnrProblem(dataflowView, finalized.view(),
                                              fabricRoot.view(), spatialConfig,
                                              importedConstraints.view()));
  if (frozen->dataflowIdentity() != dataflowView.identity() ||
      frozen->techMappingIdentity() != finalized.view().identity() ||
      frozen->fabricIdentity() != fabricRoot.view().identity() ||
      frozen->constraintSetIdentity() != importedConstraints.view().identity())
    fail("aggregate Spatial freeze lost an exact input identity");
  if (frozen->realizations().memoryRealizations().size() != 1 ||
      frozen->routing().routingEndpoints().empty() ||
      frozen->routing().routingArcs().empty())
    fail("aggregate Spatial freeze omitted realizations or routing topology");
  if (frozen->routing().routeClaims().empty() ||
      frozen->routing().traversalClaimKeys().empty())
    fail("Spatial freeze omitted traversal-implied resource claims");
  const auto &routing = frozen->routing();
  if (routing.capacityRouteClaimOffsets().size() !=
          frozen->resources().capacityDimensions().size() + 1 ||
      routing.routeClaimTraversalOffsets().size() !=
          routing.routeClaims().size() + 1 ||
      routing.traversalArcOffsets().size() != routing.traversals().size() + 1)
    fail("Spatial freeze omitted route-cost reverse CSR offsets");
  for (loom::pnr::PnrIndex claim = 0; claim < routing.routeClaims().size();
       ++claim) {
    const auto &record = routing.routeClaims()[claim];
    const auto capacityClaims = routing.capacityRouteClaims().slice(
        routing.capacityRouteClaimOffsets()[record.capacityDimension],
        routing.capacityRouteClaimOffsets()[record.capacityDimension + 1] -
            routing.capacityRouteClaimOffsets()[record.capacityDimension]);
    if (!llvm::is_contained(capacityClaims, claim))
      fail("capacity-to-route-claim reverse incidence is incomplete");
  }
  for (loom::pnr::PnrIndex traversal = 0;
       traversal < routing.traversals().size(); ++traversal) {
    const auto &record = routing.traversals()[traversal];
    for (loom::pnr::PnrIndex claim : routing.traversalClaimKeys().slice(
             record.routeClaimOffset, record.routeClaimCount)) {
      const auto claimTraversals = routing.routeClaimTraversals().slice(
          routing.routeClaimTraversalOffsets()[claim],
          routing.routeClaimTraversalOffsets()[claim + 1] -
              routing.routeClaimTraversalOffsets()[claim]);
      if (!llvm::is_contained(claimTraversals, traversal))
        fail("route-claim-to-traversal reverse incidence is incomplete");
    }
    for (loom::pnr::PnrIndex arc : routing.traversalArcs().slice(
             routing.traversalArcOffsets()[traversal],
             routing.traversalArcOffsets()[traversal + 1] -
                 routing.traversalArcOffsets()[traversal]))
      if (arc >= routing.routingArcs().size() ||
          routing.routingArcs()[arc].traversal != traversal)
        fail("traversal-to-arc reverse incidence is inconsistent");
  }
  bool observedSharedSwitchIngress = false;
  for (std::size_t first = 0; first < frozen->routing().traversals().size() &&
                              !observedSharedSwitchIngress;
       ++first) {
    const auto &lhs = frozen->routing().traversals()[first];
    const auto *lhsSwitch =
        std::get_if<loom::fabric::FabricSwitchTraversalPayload>(
            &lhs.reference.payload);
    if (!lhsSwitch || lhs.routeClaimCount == 0)
      continue;
    for (std::size_t second = first + 1;
         second < frozen->routing().traversals().size(); ++second) {
      const auto &rhs = frozen->routing().traversals()[second];
      const auto *rhsSwitch =
          std::get_if<loom::fabric::FabricSwitchTraversalPayload>(
              &rhs.reference.payload);
      if (!rhsSwitch || rhs.routeClaimCount == 0 ||
          lhsSwitch->owner != rhsSwitch->owner ||
          lhsSwitch->input != rhsSwitch->input ||
          lhsSwitch->output == rhsSwitch->output)
        continue;
      const auto lhsClaims = frozen->routing().traversalClaimKeys().slice(
          lhs.routeClaimOffset, lhs.routeClaimCount);
      const auto rhsClaims = frozen->routing().traversalClaimKeys().slice(
          rhs.routeClaimOffset, rhs.routeClaimCount);
      std::size_t common = 0;
      for (loom::pnr::PnrIndex lhsClaim : lhsClaims)
        common += llvm::is_contained(rhsClaims, lhsClaim);
      if (common != 1)
        fail("switch broadcast branches did not share exactly one ingress "
             "claim key");
      observedSharedSwitchIngress = true;
      break;
    }
  }
  if (!observedSharedSwitchIngress)
    fail("Spatial freeze has no temporal-switch broadcast claim anchor");
  if (frozen->realizations().computeActorRealizations().size() !=
          frozen->realizations().computeActors().size() ||
      frozen->realizations().memoryActorRealizations().size() !=
          frozen->realizations().memoryActors().size())
    fail("aggregate Spatial freeze omitted actor-owner reverse projection");
  for (auto [actorOrdinal, owner] :
       llvm::enumerate(frozen->realizations().memoryActorRealizations())) {
    if (owner >= frozen->realizations().memoryRealizations().size())
      fail("memory actor-owner reverse projection is out of range");
    const auto &realization =
        frozen->realizations().memoryRealizations()[owner];
    if (actorOrdinal < realization.actorOffset ||
        actorOrdinal - realization.actorOffset >= realization.actorCount)
      fail("memory actor-owner reverse projection changed its owner slice");
  }
  const auto &handshake = frozen->handshake();
  if (handshake.nodeSignals().size() != handshake.nodeCount() ||
      handshake.adjacencyOffsets().size() != handshake.nodeCount() + 1 ||
      handshake.reverseAdjacencyOffsets().size() != handshake.nodeCount() + 1)
    fail("aggregate Spatial freeze omitted handshake node incidence");
  if (handshake.traversalFragmentOffsets().size() !=
      frozen->routing().traversals().size() + 1)
    fail("aggregate Spatial freeze omitted traversal handshake incidence");
  if (handshake.memoryOperationDomains().empty() ||
      handshake.memoryOperationPlans().empty())
    fail("aggregate Spatial freeze omitted memory handshake plan domains");
  if (handshake.memoryPlacementDomainOffsets().size() !=
      frozen->realizations().memoryPlacements().size() + 1)
    fail("aggregate Spatial freeze omitted memory-placement plan incidence");
  for (auto [placementOrdinal, placement] :
       llvm::enumerate(frozen->realizations().memoryPlacements())) {
    const auto &realization =
        frozen->realizations().memoryRealizations()[placement.realization];
    const auto offsets = handshake.memoryPlacementDomainOffsets();
    if (offsets[placementOrdinal + 1] - offsets[placementOrdinal] !=
        realization.actorCount)
      fail("memory-placement plan incidence changed its actor domain");
  }

  std::vector<loom::pnr::SpatialComputeBindingSelection> computeBindings;
  computeBindings.reserve(frozen->realizations().computeRealizations().size());
  for (const auto &realization : frozen->realizations().computeRealizations()) {
    const loom::pnr::PnrIndex placement = realization.placementOffset;
    computeBindings.push_back(
        {placement,
         frozen->realizations().computePlacements()[placement].contextOffset});
  }
  std::vector<loom::pnr::SpatialMemoryBindingSelection> memoryBindings;
  memoryBindings.reserve(frozen->realizations().memoryRealizations().size());
  for (const auto &realization : frozen->realizations().memoryRealizations())
    memoryBindings.push_back({realization.placementOffset});

  std::vector<loom::pnr::PnrIndex> portAttachments;
  portAttachments.reserve(frozen->ports().portDemands().size());
  for (const auto &demand : frozen->ports().portDemands()) {
    const loom::pnr::PnrIndex placement =
        demand.kind == loom::pnr::FrozenSpatialPortDemandKind::Compute
            ? computeBindings[demand.realization].placement
            : memoryBindings[demand.realization].placement;
    const loom::pnr::PnrIndex realizationPlacementOffset =
        demand.kind == loom::pnr::FrozenSpatialPortDemandKind::Compute
            ? frozen->realizations()
                  .computeRealizations()[demand.realization]
                  .placementOffset
            : frozen->realizations()
                  .memoryRealizations()[demand.realization]
                  .placementOffset;
    const auto &domain =
        frozen->ports()
            .placementDomains()[demand.placementDomainOffset + placement -
                                realizationPlacementOffset];
    portAttachments.push_back(domain.attachmentOptionOffset);
  }
  std::vector<loom::pnr::PnrIndex> boundaryAttachments;
  boundaryAttachments.reserve(frozen->ports().graphBoundaries().size());
  std::optional<loom::pnr::PnrIndex> multicastNet;
  for (auto [netOrdinal, net] :
       llvm::enumerate(frozen->transfers().logicalNets()))
    if (net.sinkCount > 1)
      multicastNet = static_cast<loom::pnr::PnrIndex>(netOrdinal);
  if (!multicastNet)
    fail("Spatial candidate fixture has no multicast net");
  for (const auto &boundary : frozen->ports().graphBoundaries())
    boundaryAttachments.push_back(boundary.attachmentOptionOffset);

  std::vector<loom::pnr::PnrIndex> memoryPlans(
      frozen->realizations().memoryActors().size(),
      loom::pnr::getInvalidPnrIndex());
  for (auto [realizationOrdinal, realization] :
       llvm::enumerate(frozen->realizations().memoryRealizations())) {
    const loom::pnr::PnrIndex placement =
        memoryBindings[realizationOrdinal].placement;
    const loom::pnr::PnrIndex domainOffset =
        handshake.memoryPlacementDomainOffsets()[placement];
    for (loom::pnr::PnrIndex localActor = 0;
         localActor < realization.actorCount; ++localActor) {
      const auto &domain =
          handshake.memoryOperationDomains()[domainOffset + localActor];
      memoryPlans[realization.actorOffset + localActor] = domain.planOffset;
    }
  }

  if (!computeBindings.empty()) {
    auto malformedBindings = computeBindings;
    malformedBindings.front().placement = loom::pnr::getInvalidPnrIndex();
    if (!rejected(loom::pnr::SpatialCandidateState::create(
            frozen, {malformedBindings, memoryBindings, portAttachments,
                     boundaryAttachments, memoryPlans})))
      fail("Spatial candidate accepted a foreign compute placement");
  }

  auto spatialCandidate = take(loom::pnr::SpatialCandidateState::create(
      frozen, {computeBindings, memoryBindings, portAttachments,
               boundaryAttachments, memoryPlans}));
  requireSuccess(spatialCandidate->verify());

  std::optional<loom::pnr::PnrIndex> routedNet;
  for (auto [netOrdinal, net] :
       llvm::enumerate(frozen->transfers().logicalNets())) {
    if (!routedNet && net.sinkCount == 1 &&
        spatialCandidate->logicalNetSourceEndpoint(netOrdinal) !=
            spatialCandidate->logicalNetSinkEndpoint(netOrdinal, 0))
      routedNet = static_cast<loom::pnr::PnrIndex>(netOrdinal);
  }
  if (!routedNet)
    fail("Spatial candidate fixture has no nontrivial single-sink net");
  const loom::pnr::PnrIndex routeSource =
      spatialCandidate->logicalNetSourceEndpoint(*routedNet);
  const loom::pnr::PnrIndex routeTarget =
      spatialCandidate->logicalNetSinkEndpoint(*routedNet, 0);
  std::vector<loom::pnr::RouteCost> routeCosts(
      frozen->routing().routingArcs().size(), 1);
  loom::pnr::EndpointRouteSearchScratch routeSearch;
  requireSuccess(routeSearch.prepare(
      loom::pnr::endpointRoutingGraphView(frozen->routing())));
  const loom::pnr::PnrIndex unrestrictedReplicationGroup =
      loom::pnr::getInvalidPnrIndex();
  const loom::pnr::PnrIndex firstTargetRank = 0;
  std::optional<std::vector<loom::pnr::PnrIndex>> routedArcs;
  for (loom::pnr::PnrIndex claimArc = 0;
       claimArc < frozen->routing().routingArcs().size() && !routedArcs;
       ++claimArc) {
    const auto &claimArcRecord = frozen->routing().routingArcs()[claimArc];
    if (frozen->routing()
                .traversals()[claimArcRecord.traversal]
                .routeClaimCount == 0 ||
        claimArcRecord.payloadCapacityBits <
            spatialCandidate->logicalNetPayloadWidth(*routedNet))
      continue;
    const loom::pnr::PnrIndex claimSource =
        frozen->routing().arcSources()[claimArc];
    const loom::pnr::PnrIndex claimTarget = claimArcRecord.target;
    auto prefix = routeSearch.search(
        {{&routeSource, 1},
         {&unrestrictedReplicationGroup, 1},
         {&claimSource, 1},
         {&firstTargetRank, 1},
         routeCosts,
         routeCosts,
         spatialCandidate->logicalNetPayloadWidth(*routedNet),
         0,
         262144});
    if (!prefix) {
      llvm::consumeError(prefix.takeError());
      continue;
    }
    std::vector<loom::pnr::PnrIndex> candidate(prefix->forwardArcs.begin(),
                                               prefix->forwardArcs.end());
    auto suffix = routeSearch.search(
        {{&claimTarget, 1},
         {&unrestrictedReplicationGroup, 1},
         {&routeTarget, 1},
         {&firstTargetRank, 1},
         routeCosts,
         routeCosts,
         spatialCandidate->logicalNetPayloadWidth(*routedNet),
         0,
         262144});
    if (!suffix) {
      llvm::consumeError(suffix.takeError());
      continue;
    }
    candidate.push_back(claimArc);
    candidate.insert(candidate.end(), suffix->forwardArcs.begin(),
                     suffix->forwardArcs.end());

    std::vector<std::uint8_t> seen(frozen->routing().routingEndpoints().size(),
                                   0);
    loom::pnr::PnrIndex endpoint = routeSource;
    seen[endpoint] = 1;
    bool simple = true;
    for (loom::pnr::PnrIndex arc : candidate) {
      if (frozen->routing().arcSources()[arc] != endpoint) {
        simple = false;
        break;
      }
      endpoint = frozen->routing().routingArcs()[arc].target;
      if (seen[endpoint]) {
        simple = false;
        break;
      }
      seen[endpoint] = 1;
    }
    if (simple && endpoint == routeTarget)
      routedArcs = std::move(candidate);
  }
  if (!routedArcs || routedArcs->empty())
    fail("Spatial candidate fixture has no simple claim-bearing route");

  loom::pnr::SpatialCandidateScratch candidateScratch;
  requireSuccess(candidateScratch.prepare(*frozen));
  auto routeMove = take(spatialCandidate->beginMove(candidateScratch));
  requireSuccess(routeMove.bindRouteSource(*routedNet, routeSource));
  requireSuccess(routeMove.bindRouteSink(*routedNet, 0, routeTarget));
  requireSuccess(
      routeMove.attachRoutePath(*routedNet, routeSource, *routedArcs, 0));
  if (!take(routeMove.close()))
    fail("valid Spatial route closed a combinational handshake cycle");
  requireSuccess(routeMove.commit());
  if (!spatialCandidate->routeTree(*routedNet).isRouted())
    fail("Spatial move did not commit its RouteTree");
  if (loom::pnr::spatialMappingMeasureValue(
          *spatialCandidate,
          loom::pnr::MappingMeasureKind::TotalSelectedTraversalClaim) == 0)
    fail("routed Spatial candidate has no exact traversal-claim objective");
  bool observedActiveClaimBit = false;
  const auto activeClaimBits =
      spatialCandidate->logicalNetRouteClaimBits(*routedNet);
  for (loom::pnr::PnrIndex claim = 0;
       claim < frozen->routing().routeClaims().size(); ++claim) {
    const bool active = (activeClaimBits[claim / 64] >> (claim % 64)) & 1;
    if (active != (spatialCandidate->logicalNetRouteClaimRefcount(*routedNet,
                                                                  claim) != 0))
      fail("route-claim active bit diverges from exact net refcount");
    observedActiveClaimBit |= active;
  }
  if (!observedActiveClaimBit)
    fail("claim-bearing route has no active-claim bit");
  const std::uint64_t committedTraversalClaim =
      spatialCandidate->totalSelectedTraversalClaim();
  requireSuccess(spatialCandidate->verify());

  const auto *pathFinder = std::get_if<loom::ResolvedPathFinderPolicy>(
      &spatialConfig.policy().search.routing.negotiation);
  if (!pathFinder)
    fail("default Spatial routing policy is not PathFinder");
  auto routeCostState =
      take(loom::pnr::SpatialRouteCostState::create(*spatialCandidate));
  if (routeCostState.lowerBoundArcCosts().size() !=
          frozen->routing().routingArcs().size() ||
      routeCostState.currentArcCosts().size() !=
          frozen->routing().routingArcs().size())
    fail("PathFinder route-cost overlay omitted a routing arc");
  bool observedDynamicBaseline = false;
  for (auto [lower, current] :
       llvm::zip_equal(routeCostState.lowerBoundArcCosts(),
                       routeCostState.currentArcCosts())) {
    if (current < lower)
      fail("PathFinder current arc cost fell below its lower bound");
    observedDynamicBaseline |= current > lower;
  }
  if (!observedDynamicBaseline)
    fail("PathFinder route-cost overlay ignored committed route occupancy");

  std::vector<std::uint64_t> baselineUsage;
  baselineUsage.reserve(frozen->resources().capacityDimensions().size());
  for (loom::pnr::PnrIndex capacity = 0;
       capacity < frozen->resources().capacityDimensions().size(); ++capacity)
    baselineUsage.push_back(routeCostState.workingCapacityUsageRaw(capacity));
  const std::vector<loom::pnr::RouteCost> baselineCosts(
      routeCostState.currentArcCosts().begin(),
      routeCostState.currentArcCosts().end());
  std::vector<std::uint64_t> excludedUsage(baselineUsage.size(), 0);
  for (loom::pnr::PnrIndex claim = 0;
       claim < frozen->routing().routeClaims().size(); ++claim) {
    if (((activeClaimBits[claim / 64] >> (claim % 64)) & 1) == 0)
      continue;
    const auto &record = frozen->routing().routeClaims()[claim];
    excludedUsage[record.capacityDimension] += record.amount;
  }
  const std::size_t warmedRouteCostBytes =
      routeCostState.retainedStorageBytes();
  requireSuccess(routeCostState.selectLogicalNet(*routedNet));
  if (routeCostState.selectedLogicalNet() != routedNet)
    fail("PathFinder route-cost overlay lost its selected logical net");
  for (loom::pnr::PnrIndex capacity = 0; capacity < baselineUsage.size();
       ++capacity) {
    if (baselineUsage[capacity] < excludedUsage[capacity] ||
        routeCostState.workingCapacityUsageRaw(capacity) !=
            baselineUsage[capacity] - excludedUsage[capacity])
      fail("PathFinder route-cost overlay did not remove exactly one old net");
  }
  bool observedRipUpCostChange = false;
  for (auto [baseline, current] :
       llvm::zip_equal(baselineCosts, routeCostState.currentArcCosts()))
    observedRipUpCostChange |= baseline != current;
  if (!observedRipUpCostChange)
    fail("PathFinder route-cost overlay did not reprice the rip-up closure");
  requireSuccess(
      routeCostState.updateSelectedLogicalNetClaims(activeClaimBits));
  if (!llvm::equal(routeCostState.currentArcCosts(), baselineCosts))
    fail("PathFinder route-cost overlay did not price a prospective install");
  requireSuccess(routeCostState.updateSelectedLogicalNetClaims({}));
  for (loom::pnr::PnrIndex capacity = 0; capacity < baselineUsage.size();
       ++capacity)
    if (routeCostState.workingCapacityUsageRaw(capacity) !=
        baselineUsage[capacity] - excludedUsage[capacity])
      fail("PathFinder prospective rip-up did not restore excluded occupancy");
  requireSuccess(routeCostState.selectLogicalNet(std::nullopt));
  if (routeCostState.selectedLogicalNet() ||
      !llvm::equal(routeCostState.currentArcCosts(), baselineCosts) ||
      routeCostState.retainedStorageBytes() != warmedRouteCostBytes)
    fail("PathFinder route-cost overlay did not restore its warmed baseline");
  for (loom::pnr::PnrIndex capacity = 0; capacity < baselineUsage.size();
       ++capacity)
    if (routeCostState.workingCapacityUsageRaw(capacity) !=
        baselineUsage[capacity])
      fail("PathFinder route-cost overlay did not restore raw occupancy");

  auto routedCandidate = take(loom::pnr::SpatialCandidateState::create(
      frozen, {computeBindings, memoryBindings, portAttachments,
               boundaryAttachments, memoryPlans}));
  loom::pnr::SpatialCandidateScratch routedCandidateScratch;
  requireSuccess(routedCandidateScratch.prepare(*frozen));
  auto routedCostState =
      take(loom::pnr::SpatialRouteCostState::create(*routedCandidate));
  requireSuccess(routedCostState.selectLogicalNet(*multicastNet));
  loom::pnr::SpatialNetRouterScratch netRouter;
  requireSuccess(netRouter.prepare(*frozen));
  auto wholeNetMove = take(routedCandidate->beginMove(routedCandidateScratch));
  const loom::pnr::RouteCost wholeNetCost = take(netRouter.routeWholeNet(
      wholeNetMove, *routedCandidate, routedCostState, *multicastNet,
      spatialConfig.policy().search.routing.endpointExpansionLimit));
  if (wholeNetCost == loom::pnr::routeCostInfinity)
    fail("whole-net routing returned the infinity sentinel");
  if (!take(wholeNetMove.close()))
    fail("whole-net routing closed a combinational handshake cycle");
  requireSuccess(wholeNetMove.commit());
  requireSuccess(routedCostState.selectLogicalNet(std::nullopt));
  if (!routedCandidate->routeTree(*multicastNet).isRouted())
    fail("whole-net routing did not commit a complete RouteTree");
  requireSuccess(routedCandidate->verify());

  const std::uint64_t routedObjective =
      routedCandidate->totalSelectedTraversalClaim();
  const std::vector<loom::pnr::RouteCost> routedBaselineCosts(
      routedCostState.currentArcCosts().begin(),
      routedCostState.currentArcCosts().end());
  requireSuccess(routedCostState.selectLogicalNet(*multicastNet));
  auto limitedWholeNet =
      take(routedCandidate->beginMove(routedCandidateScratch));
  auto limitedResult = netRouter.routeWholeNet(
      limitedWholeNet, *routedCandidate, routedCostState, *multicastNet, 1);
  bool observedWorkLimit = false;
  if (limitedResult) {
    fail("whole-net routing ignored its endpoint work limit");
  } else {
    llvm::handleAllErrors(
        limitedResult.takeError(),
        [&](const loom::pnr::EndpointRouteSearchFailure &error) {
          observedWorkLimit =
              error.kind() ==
              loom::pnr::EndpointRouteSearchFailureKind::WorkLimit;
        });
  }
  if (!observedWorkLimit)
    fail("whole-net routing returned the wrong bounded failure");
  limitedWholeNet.rollback();
  requireSuccess(routedCostState.selectLogicalNet(std::nullopt));
  if (routedCandidate->totalSelectedTraversalClaim() != routedObjective ||
      !llvm::equal(routedCostState.currentArcCosts(), routedBaselineCosts))
    fail("failed whole-net routing did not restore candidate and costs");
  requireSuccess(routedCandidate->verify());

  loom::pnr::SpatialPathFinderRouterScratch negotiatedRouter;
  requireSuccess(negotiatedRouter.prepare(*frozen));
  auto failedIteration = negotiatedRouter.routeToClosure(
      *routedCandidate, routedCandidateScratch, routedCostState, {1, 1}, {});
  if (failedIteration)
    fail("bounded PathFinder iteration unexpectedly reached closure");
  llvm::consumeError(failedIteration.takeError());
  if (routedCandidate->totalSelectedTraversalClaim() != routedObjective ||
      !llvm::equal(routedCostState.currentArcCosts(), routedBaselineCosts))
    fail("failed PathFinder iteration did not roll back its complete overlay");
  requireSuccess(routedCandidate->verify());
  auto selectedCycle = negotiatedRouter.routeToClosure(
      *routedCandidate, routedCandidateScratch, routedCostState,
      {spatialConfig.policy().search.routing.endpointExpansionLimit,
       spatialConfig.policy().search.routing.negotiationIterationLimit},
      {});
  bool rejectedSelectedCycle = false;
  if (selectedCycle) {
    fail("cyclic global route fixture unexpectedly reached closure");
  } else {
    llvm::handleAllErrors(
        selectedCycle.takeError(),
        [&](const loom::pnr::SpatialPathFinderClosureFailure &failure) {
          rejectedSelectedCycle = failure.kind() ==
                                  loom::pnr::SpatialPathFinderClosureFailure::
                                      Kind::SelectedCombinationalHandshakeCycle;
        },
        [&](const llvm::ErrorInfoBase &) {});
  }
  if (!rejectedSelectedCycle ||
      routedCandidate->totalSelectedTraversalClaim() != routedObjective ||
      !llvm::equal(routedCostState.currentArcCosts(), routedBaselineCosts))
    fail("cyclic PathFinder overlay was not rejected and rolled back");
  requireSuccess(routedCandidate->verify());

  auto rollbackMove = take(spatialCandidate->beginMove(candidateScratch));
  requireSuccess(rollbackMove.ripUpWholeRoute(*routedNet));
  if (!take(rollbackMove.close()))
    fail("route deletion reported a combinational handshake cycle");
  rollbackMove.rollback();
  if (!spatialCandidate->routeTree(*routedNet).isRouted())
    fail("Spatial move rollback discarded the committed RouteTree");
  if (spatialCandidate->totalSelectedTraversalClaim() !=
      committedTraversalClaim)
    fail("Spatial move rollback changed traversal resource accounting");
  requireSuccess(spatialCandidate->verify());
  const std::size_t warmedCandidateScratchBytes =
      candidateScratch.retainedStorageBytes();
  auto repeatedRollback = take(spatialCandidate->beginMove(candidateScratch));
  requireSuccess(repeatedRollback.ripUpWholeRoute(*routedNet));
  if (!take(repeatedRollback.close()))
    fail("repeated route deletion reported a handshake cycle");
  repeatedRollback.rollback();
  if (candidateScratch.retainedStorageBytes() != warmedCandidateScratchBytes)
    fail("warmed Spatial move grew worker-local scratch storage");
  requireSuccess(spatialCandidate->verify());

  if (handshake.allTraversalGroups().empty())
    fail("aggregate Spatial freeze omitted atomic traversal activation");
  {
    auto handshakeOwner =
        std::shared_ptr<const loom::pnr::FrozenSpatialHandshakeIndex>(
            frozen, &frozen->handshake());
    auto candidate =
        take(loom::pnr::HandshakeCandidateState::create(handshakeOwner));
    loom::pnr::HandshakeCandidateScratch scratch;
    requireSuccess(scratch.prepare(*handshakeOwner));
    const auto group = handshake.allTraversalGroups().front();
    const auto witnesses = handshake.allTraversalGroupWitnesses().slice(
        group.witnessOffset, group.witnessCount);
    auto transaction = take(candidate->beginTransaction(scratch));
    for (auto [ordinal, traversal] : llvm::enumerate(witnesses)) {
      requireSuccess(transaction.addTraversalUses(traversal, 2));
      if (candidate->traversalRefcount(traversal) != 2)
        fail("handshake candidate lost a traversal use count");
      const bool shouldBeActive = ordinal + 1 == witnesses.size();
      if ((candidate->fragmentRefcount(group.fragment) != 0) != shouldBeActive)
        fail("all-traversal fragment ignored its complete witness set");
    }
    for (loom::pnr::PnrIndex traversal : witnesses) {
      requireSuccess(transaction.removeTraversalUses(traversal, 1));
      if (candidate->traversalRefcount(traversal) != 1 ||
          !candidate->isTraversalSelected(traversal))
        fail("partial traversal-use removal changed handshake selection");
    }
    transaction.rollback();
    for (loom::pnr::PnrIndex traversal : witnesses)
      if (candidate->traversalRefcount(traversal) != 0 ||
          candidate->isTraversalSelected(traversal))
        fail("all-traversal rollback retained a selected witness");
    const loom::pnr::PnrIndex repeatedTraversal = witnesses.front();
    {
      auto add = take(candidate->beginTransaction(scratch));
      requireSuccess(add.addTraversalUses(repeatedTraversal, 2));
      if (!take(add.close()))
        fail("repeated traversal use closed a handshake cycle");
      requireSuccess(add.commit());
    }
    {
      auto removeOne = take(candidate->beginTransaction(scratch));
      requireSuccess(removeOne.removeTraversalUses(repeatedTraversal, 1));
      if (!take(removeOne.close()))
        fail("partial traversal use removal reported a handshake cycle");
      requireSuccess(removeOne.commit());
    }
    if (candidate->traversalRefcount(repeatedTraversal) != 1 ||
        !candidate->isTraversalSelected(repeatedTraversal))
      fail("partial committed removal deselected a live traversal");
    {
      auto removeLast = take(candidate->beginTransaction(scratch));
      requireSuccess(removeLast.removeTraversalUses(repeatedTraversal, 1));
      if (!take(removeLast.close()))
        fail("final traversal use removal reported a handshake cycle");
      requireSuccess(removeLast.commit());
    }
    if (candidate->traversalRefcount(repeatedTraversal) != 0 ||
        candidate->isTraversalSelected(repeatedTraversal))
      fail("final committed removal retained a traversal selection");
    requireSuccess(candidate->verify());
  }

  if (frozen->transfers().logicalNets().size() !=
          finalized.view().residualLogicalNets().size() ||
      frozen->transfers().logicalNetSinks().size() != 5)
    fail("aggregate Spatial freeze omitted residual transfer obligations");
  if (frozen->ports().portDemands().size() != 4 ||
      frozen->ports().graphBoundaries().size() != 5)
    fail("aggregate Spatial freeze omitted memory or graph-boundary demands");
  if (frozen->transfers().logicalNetSourceBindings().size() !=
          frozen->transfers().logicalNets().size() ||
      frozen->transfers().logicalNetSinkBindings().size() !=
          frozen->transfers().logicalNetSinks().size())
    fail("aggregate Spatial freeze omitted transfer attachment bindings");
  for (const auto &demand : frozen->ports().portDemands()) {
    if (demand.kind != loom::pnr::FrozenSpatialPortDemandKind::Memory ||
        demand.placementDomainCount == 0)
      fail("memory PortDemand lost its factorized placement domain");
    for (const auto &domain : frozen->ports().placementDomains().slice(
             demand.placementDomainOffset, demand.placementDomainCount)) {
      if (domain.attachmentOptionCount == 0)
        fail("memory PortDemand retained an empty attachment domain");
      const auto &placement =
          frozen->realizations().memoryPlacements()[domain.placement];
      for (const auto &option : frozen->ports().attachmentOptions().slice(
               domain.attachmentOptionOffset, domain.attachmentOptionCount)) {
        const auto &endpoint =
            frozen->routing().routingEndpoints()[option.endpoint].reference;
        if (endpoint.owner.kind() !=
                loom::fabric::FabricTransportEndpointOwnerKind::
                    FabricMemoryOccurrence ||
            std::get<loom::fabric::FabricMemoryOccurrenceRef>(
                endpoint.owner.payload) != placement.memory ||
            option.localTraversal)
          fail("memory PortDemand did not project its exact occurrence "
               "endpoint");
      }
    }
  }
  if (frozen->ports().endpointAttachmentOffsets().size() !=
          frozen->routing().routingEndpoints().size() + 1 ||
      frozen->ports().endpointAttachmentOptions().size() !=
          frozen->ports().attachmentOptions().size())
    fail("aggregate Spatial freeze omitted attachment reverse incidence");
  std::size_t expectedResourceStates = 0;
  std::size_t expectedUsePatterns = 0;
  std::size_t expectedCommits = 0;
  for (const auto &owner : fabricRoot.view().moduleResourceOwners()) {
    const ::fabric::ResourceContract *contract =
        fabricRoot.view().resourceContract(owner);
    if (!contract)
      fail("physical resource-owner inventory contains no contract");
    expectedResourceStates += contract->stateCount();
    expectedUsePatterns += contract->usePatternCount();
    for (std::uint32_t ordinal = 0; ordinal < contract->usePatternCount();
         ++ordinal)
      expectedCommits += contract->usePattern(::fabric::UsePatternKey(ordinal))
                             .commit.has_value();
  }
  if (frozen->resources().resourceOwners().size() !=
          fabricRoot.view().moduleResourceOwners().size() ||
      frozen->resources().resourceStates().size() != expectedResourceStates ||
      frozen->resources().usePatterns().size() != expectedUsePatterns)
    fail("aggregate Spatial freeze omitted Fabric resource contracts");
  const std::size_t frozenCommits = llvm::count_if(
      frozen->resources().usePatterns(),
      [](const auto &pattern) { return pattern.commit.has_value(); });
  if (frozenCommits != expectedCommits)
    fail("aggregate Spatial freeze changed owner-defined commit transitions");
  const auto traversalViews = fabricRoot.view().physicalTraversals();
  if (frozen->routing().traversals().size() != traversalViews.size())
    fail("aggregate Spatial freeze changed the traversal inventory");
  for (auto [ordinal, traversal] :
       llvm::enumerate(frozen->routing().traversals())) {
    if (traversal.resourceStateCount !=
        traversalViews[ordinal].resourceStates.size())
      fail("aggregate Spatial freeze changed traversal resource states");
    const auto frozenStates = frozen->routing().traversalResourceStates().slice(
        traversal.resourceStateOffset, traversal.resourceStateCount);
    for (auto [stateOrdinal, state] : llvm::enumerate(frozenStates))
      if (frozen->resources().resourceStates()[state].reference !=
          traversalViews[ordinal].resourceStates[stateOrdinal])
        fail("aggregate Spatial freeze rebound a traversal resource state");
  }
  if (!frozen->constraints().empty())
    fail("empty MappingConstraintSet produced a nonempty constraint index");
  if (frozen->workBudget().empty())
    fail("aggregate Spatial freeze omitted the derived work budget");

  loom::pnr::FrozenSpatialPnrProblemHandle repeated =
      take(loom::pnr::freezeSpatialPnrProblem(dataflowView, finalized.view(),
                                              fabricRoot.view(), spatialConfig,
                                              importedConstraints.view()));
  if (frozen->cacheKey() != repeated->cacheKey())
    fail("identical Spatial freeze inputs changed the cache key");
  requireSuccess(loom::pnr::revalidateFrozenSpatialPnrCacheHit(
      *frozen, dataflowView, finalized.view(), fabricRoot.view(), spatialConfig,
      importedConstraints.view()));

  loom::ResolvedConfig changedResolved = loom::defaultResolvedConfig();
  ++changedResolved.dse.spatialPnr.search.routing.endpointExpansionLimit;
  const loom::pnr::ResolvedPnrConfigView changedConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(changedResolved));
  loom::pnr::FrozenSpatialPnrProblemHandle changed =
      take(loom::pnr::freezeSpatialPnrProblem(dataflowView, finalized.view(),
                                              fabricRoot.view(), changedConfig,
                                              importedConstraints.view()));
  if (frozen->cacheKey() == changed->cacheKey())
    fail("changed selected PnR view reused the freeze cache key");
  if (llvm::Error error = loom::pnr::revalidateFrozenSpatialPnrCacheHit(
          *frozen, dataflowView, finalized.view(), fabricRoot.view(),
          changedConfig, importedConstraints.view()))
    llvm::consumeError(std::move(error));
  else
    fail("cache-hit validation accepted a changed selected PnR view");

  const auto offsets = frozen->routing().adjacencyOffsets();
  const auto reverseOffsets = frozen->routing().reverseAdjacencyOffsets();
  const auto reverseArcs = frozen->routing().reverseArcOrdinals();
  if (reverseOffsets.size() !=
          frozen->routing().routingEndpoints().size() + 1 ||
      reverseArcs.size() != frozen->routing().routingArcs().size())
    fail("aggregate Spatial routing graph omitted reverse CSR");
  std::vector<bool> reverseSeen(reverseArcs.size(), false);
  for (std::size_t endpoint = 0; endpoint + 1 < reverseOffsets.size();
       ++endpoint) {
    for (loom::pnr::PnrIndex cursor = reverseOffsets[endpoint];
         cursor < reverseOffsets[endpoint + 1]; ++cursor) {
      const loom::pnr::PnrIndex reverseArc = reverseArcs[cursor];
      if (reverseArc >= frozen->routing().routingArcs().size() ||
          frozen->routing().routingArcs()[reverseArc].target != endpoint ||
          reverseSeen[reverseArc])
        fail("aggregate Spatial routing reverse CSR is not exact");
      reverseSeen[reverseArc] = true;
    }
  }
  if (llvm::find(reverseSeen, false) != reverseSeen.end())
    fail("aggregate Spatial routing reverse CSR omitted an arc");
  std::optional<loom::pnr::PnrIndex> source;
  for (std::size_t endpoint = 0; endpoint + 1 < offsets.size(); ++endpoint) {
    if (offsets[endpoint] != offsets[endpoint + 1]) {
      source = static_cast<loom::pnr::PnrIndex>(endpoint);
      break;
    }
  }
  if (!source)
    fail("aggregate Spatial routing graph has no routable source");
  const loom::pnr::PnrIndex arc = offsets[*source];
  if (frozen->routing().arcSources().size() !=
          frozen->routing().routingArcs().size() ||
      frozen->routing().arcSources()[arc] != *source)
    fail("aggregate Spatial routing graph lost its arc-source projection");
  const loom::pnr::PnrIndex target =
      frozen->routing().routingArcs()[arc].target;
  auto routingOwner =
      std::shared_ptr<const loom::pnr::FrozenSpatialRoutingGraph>(
          frozen, &frozen->routing());
  loom::pnr::RouteTreeStateHandle routeTree =
      take(loom::pnr::RouteTreeState::create(std::move(routingOwner), 2));
  loom::pnr::RouteTreeTransactionScratch scratch;
  auto transaction = take(routeTree->beginTransaction(scratch));
  requireSuccess(transaction.bindSource(*source));
  requireSuccess(transaction.bindSink(0, target));
  requireSuccess(transaction.bindSink(1, target));
  requireSuccess(transaction.attachPath(*source, {arc}, 0));
  requireSuccess(transaction.attachPath(target, {}, 1));
  const auto addedTraversals = take(transaction.prepare());
  if (addedTraversals.size() != 1 ||
      addedTraversals.front().traversal !=
          frozen->routing().routingArcs()[arc].traversal ||
      addedTraversals.front().added != 1 ||
      addedTraversals.front().removed != 0)
    fail("shared RouteTree path did not produce one canonical traversal delta");
  requireSuccess(transaction.commit());
  requireSuccess(routeTree->verify());
  {
    loom::pnr::RouteTreeTransactionScratch rollbackScratch;
    auto rollback = take(routeTree->beginTransaction(rollbackScratch));
    requireSuccess(rollback.ripUpWholeNet());
    const auto removedTraversals = take(rollback.prepare());
    if (removedTraversals.size() != 1 ||
        removedTraversals.front().traversal !=
            frozen->routing().routingArcs()[arc].traversal ||
        removedTraversals.front().removed != 1 ||
        removedTraversals.front().added != 0)
      fail("RouteTree rip-up did not produce one canonical traversal delta");
    rollback.rollback();
  }
  if (!routeTree->isRouted())
    fail("route-tree rollback discarded committed state");
  requireSuccess(routeTree->verify());
  {
    loom::pnr::RouteTreeTransactionScratch ripUpScratch;
    auto ripUp = take(routeTree->beginTransaction(ripUpScratch));
    requireSuccess(ripUp.ripUpWholeNet());
    const auto removedTraversals = take(ripUp.prepare());
    if (removedTraversals.size() != 1 ||
        removedTraversals.front().removed != 1 ||
        removedTraversals.front().added != 0)
      fail("committed RouteTree rip-up lost its traversal delta");
    requireSuccess(ripUp.commit());
  }
  if (!routeTree->isUnrouted())
    fail("route-tree whole-net rip-up retained committed state");
  requireSuccess(routeTree->verify());

  const auto replicationGroups = frozen->routing().traversalReplicationGroups();
  if (replicationGroups.size() != frozen->routing().traversals().size())
    fail("aggregate Spatial routing graph omitted replication groups");
  struct BranchPair final {
    loom::pnr::PnrIndex source;
    loom::pnr::PnrIndex firstArc;
    loom::pnr::PnrIndex secondArc;
  };
  const auto findBranchPair =
      [&](bool explicitReplication) -> std::optional<BranchPair> {
    for (loom::pnr::PnrIndex branchSource = 0;
         branchSource + 1 < offsets.size(); ++branchSource) {
      for (loom::pnr::PnrIndex firstArc = offsets[branchSource];
           firstArc < offsets[branchSource + 1]; ++firstArc) {
        const auto &first = frozen->routing().routingArcs()[firstArc];
        const loom::pnr::PnrIndex firstGroup =
            replicationGroups[first.traversal];
        for (loom::pnr::PnrIndex secondArc = firstArc + 1;
             secondArc < offsets[branchSource + 1]; ++secondArc) {
          const auto &second = frozen->routing().routingArcs()[secondArc];
          if (first.target == second.target)
            continue;
          const loom::pnr::PnrIndex secondGroup =
              replicationGroups[second.traversal];
          const bool sameExplicitGroup =
              firstGroup != loom::pnr::getInvalidPnrIndex() &&
              firstGroup == secondGroup;
          if (sameExplicitGroup == explicitReplication)
            return BranchPair{branchSource, firstArc, secondArc};
        }
      }
    }
    return std::nullopt;
  };
  const auto exerciseBranch = [&](const BranchPair &branch,
                                  bool shouldSucceed) {
    auto branchOwner =
        std::shared_ptr<const loom::pnr::FrozenSpatialRoutingGraph>(
            frozen, &frozen->routing());
    auto tree =
        take(loom::pnr::RouteTreeState::create(std::move(branchOwner), 2));
    loom::pnr::RouteTreeTransactionScratch branchScratch;
    auto branchTransaction = take(tree->beginTransaction(branchScratch));
    const loom::pnr::PnrIndex firstTarget =
        frozen->routing().routingArcs()[branch.firstArc].target;
    const loom::pnr::PnrIndex secondTarget =
        frozen->routing().routingArcs()[branch.secondArc].target;
    requireSuccess(branchTransaction.bindSource(branch.source));
    requireSuccess(branchTransaction.bindSink(0, firstTarget));
    requireSuccess(branchTransaction.bindSink(1, secondTarget));
    requireSuccess(
        branchTransaction.attachPath(branch.source, {branch.firstArc}, 0));
    requireSuccess(
        branchTransaction.attachPath(branch.source, {branch.secondArc}, 1));
    auto prepared = branchTransaction.prepare();
    if (shouldSucceed) {
      if (!prepared)
        fail("explicit switch replication was rejected: " +
             llvm::toString(prepared.takeError()));
      requireSuccess(branchTransaction.commit());
      requireSuccess(tree->verify());
    } else if (prepared) {
      fail("selector alternatives were accepted as a broadcast branch");
    } else {
      llvm::consumeError(prepared.takeError());
    }
  };

  const std::optional<BranchPair> broadcastBranch = findBranchPair(true);
  const std::optional<BranchPair> selectorBranch = findBranchPair(false);
  if (!broadcastBranch || !selectorBranch)
    fail("builtin SpatialCore lacks replication and selector branch anchors");
  exerciseBranch(*broadcastBranch, true);
  exerciseBranch(*selectorBranch, false);

  const loom::pnr::ResolvedPnrConfigView systemConfig =
      take(loom::pnr::projectResolvedSystemPnrConfigView(
          loom::defaultResolvedConfig()));
  if (!rejectedAs(loom::pnr::freezeSpatialPnrProblem(
                      dataflowView, finalized.view(), fabricRoot.view(),
                      systemConfig, importedConstraints.view()),
                  loom::pnr::SpatialPnrFreezeFailureKind::Invalid))
    fail("aggregate Spatial freeze accepted a System PnR config view");

  const std::string emptyMemoryDomainClause =
      "    mapping.constraint.domain_restriction "
      "projection(memory_placement) subject("
      "#mapping.memory_realization_ref<" +
      std::to_string(finalized.view().memoryRealizations().front().entityId) +
      ">) admissible_domain([])\n";
  auto constrained = parseMapping(
      context,
      spatialConstraintText(dataflowView, finalized.view(), fabricRoot.view(),
                            emptyMemoryDomainClause));
  if (!constrained)
    fail("typed Spatial MappingConstraintSet fixture did not parse");
  auto constrainedRoots =
      constrained->getOps<::mapping::ConstraintsSpatialOp>();
  auto finalizedConstrained =
      take(loom::mapping::finalizeSpatialMappingConstraintSet(
          *constrainedRoots.begin(), dataflowView, finalized.view(),
          fabricRoot.view(), store));
  if (finalizedConstrained.view().clauses().size() != 1 ||
      !std::holds_alternative<loom::mapping::SpatialDomainRestrictionView>(
          finalizedConstrained.view().clauses().front()))
    fail("sealed Spatial MappingConstraintSet lost its typed clause");
  if (!rejectedAs(loom::pnr::freezeSpatialPnrProblem(
                      dataflowView, finalized.view(), fabricRoot.view(),
                      spatialConfig, finalizedConstrained.view()),
                  loom::pnr::SpatialPnrFreezeFailureKind::ProvenInfeasible))
    fail("empty singleton placement domain was not proven infeasible");

  const auto selectedMemory =
      frozen->realizations().memoryPlacements().front().memory;
  const std::string selectedMemoryDomainClause =
      "    mapping.constraint.domain_restriction "
      "projection(memory_placement) subject("
      "#mapping.memory_realization_ref<" +
      std::to_string(finalized.view().memoryRealizations().front().entityId) +
      ">) admissible_domain([" +
      fabricAttr("fabric_memory_occurrence_ref", selectedMemory) + "])\n";
  auto selectedMemoryConstraints = parseMapping(
      context,
      spatialConstraintText(dataflowView, finalized.view(), fabricRoot.view(),
                            selectedMemoryDomainClause));
  if (!selectedMemoryConstraints)
    fail("nonempty Spatial MappingConstraintSet fixture did not parse");
  auto selectedMemoryConstraintRoots =
      selectedMemoryConstraints->getOps<::mapping::ConstraintsSpatialOp>();
  auto finalizedSelectedMemoryConstraints =
      take(loom::mapping::finalizeSpatialMappingConstraintSet(
          *selectedMemoryConstraintRoots.begin(), dataflowView,
          finalized.view(), fabricRoot.view(), store));
  loom::pnr::FrozenSpatialPnrProblemHandle constrainedFreeze =
      take(loom::pnr::freezeSpatialPnrProblem(
          dataflowView, finalized.view(), fabricRoot.view(), spatialConfig,
          finalizedSelectedMemoryConstraints.view()));
  if (constrainedFreeze->cacheKey() == frozen->cacheKey())
    fail("changed Spatial MappingConstraintSet reused the freeze cache key");
  const auto constrainedDomain =
      constrainedFreeze->constraints()
          .shard(::mapping::SpatialConstraintProjection::MemoryPlacement)
          .restrictedDomain(loom::mapping::SpatialConstraintSubject{
              loom::mapping::TechMemoryRealizationRef{
                  finalized.view().memoryRealizations().front().entityId}});
  if (!constrainedDomain || constrainedDomain->size() != 1 ||
      !std::holds_alternative<loom::fabric::FabricMemoryOccurrenceRef>(
          constrainedDomain->front()) ||
      std::get<loom::fabric::FabricMemoryOccurrenceRef>(
          constrainedDomain->front()) != selectedMemory ||
      constrainedFreeze->realizations().memoryPlacements().size() != 1 ||
      constrainedFreeze->realizations().memoryPlacements().front().memory !=
          selectedMemory)
    fail("nonempty memory-placement constraint did not restrict the freeze");

  const std::string staleClause =
      "    mapping.constraint.domain_restriction "
      "projection(memory_placement) subject("
      "#mapping.memory_realization_ref<999>) admissible_domain([])\n";
  auto staleConstraints = parseMapping(
      context, spatialConstraintText(dataflowView, finalized.view(),
                                     fabricRoot.view(), staleClause));
  if (!staleConstraints)
    fail("stale Spatial MappingConstraintSet fixture did not parse");
  auto staleConstraintRoots =
      staleConstraints->getOps<::mapping::ConstraintsSpatialOp>();
  if (!rejected(loom::mapping::finalizeSpatialMappingConstraintSet(
          *staleConstraintRoots.begin(), dataflowView, finalized.view(),
          fabricRoot.view(), store)))
    fail("stale TechMapping realization constraint was published");

  const std::string staleFabricDomainClause =
      "    mapping.constraint.domain_restriction "
      "projection(memory_placement) subject("
      "#mapping.memory_realization_ref<" +
      std::to_string(finalized.view().memoryRealizations().front().entityId) +
      ">) admissible_domain([" +
      fabricAttr("fabric_memory_occurrence_ref",
                 loom::fabric::FabricMemoryOccurrenceRef(
                     std::numeric_limits<std::uint64_t>::max())) +
      "])\n";
  auto staleFabricConstraints = parseMapping(
      context,
      spatialConstraintText(dataflowView, finalized.view(), fabricRoot.view(),
                            staleFabricDomainClause));
  if (!staleFabricConstraints)
    fail("stale Fabric constraint fixture did not parse");
  auto staleFabricRoots =
      staleFabricConstraints->getOps<::mapping::ConstraintsSpatialOp>();
  if (!rejected(loom::mapping::finalizeSpatialMappingConstraintSet(
          *staleFabricRoots.begin(), dataflowView, finalized.view(),
          fabricRoot.view(), store)))
    fail("stale Fabric domain reference was published");

  auto stale = parseMapping(
      context, mappingText(dataflowView, fabricRoot.view(), selected, true));
  if (!stale)
    fail("stale-reference fixture did not parse");
  auto staleRoots = stale->getOps<::mapping::TechOp>();
  if (!rejected(loom::mapping::finalizeTechMapping(*staleRoots.begin(), store)))
    fail("stale Fabric memory endpoint was published");
}

void exerciseHandshakeCandidateRefcounts(
    const loom::pnr::FrozenSpatialPnrProblemHandle &frozen) {
  const auto &handshake = frozen->handshake();
  auto handshakeOwner =
      std::shared_ptr<const loom::pnr::FrozenSpatialHandshakeIndex>(
          frozen, &frozen->handshake());
  auto candidate =
      take(loom::pnr::HandshakeCandidateState::create(handshakeOwner));
  requireSuccess(candidate->verify());
  loom::pnr::HandshakeCandidateScratch scratch;
  requireSuccess(scratch.prepare(*handshakeOwner));
  const std::size_t retainedScratchBytes = scratch.retainedStorageBytes();
  const auto offsets = handshake.computePlacementFragmentOffsets();
  const auto fragments = handshake.computePlacementFragments().slice(
      offsets.front(), offsets[1] - offsets.front());
  std::optional<loom::pnr::PnrIndex> observedFragment;
  std::optional<loom::pnr::PnrIndex> observedArc;
  for (loom::pnr::PnrIndex fragment : fragments) {
    const auto record = handshake.fragments()[fragment];
    if (record.contributionCount == 0)
      continue;
    observedFragment = fragment;
    observedArc = handshake.fragmentArcOrdinals()[record.contributionOffset];
    break;
  }
  if (!observedFragment || !observedArc)
    fail("compute placement has no observable handshake contribution");
  const loom::pnr::PnrIndex baseArcRefcount =
      candidate->arcRefcount(*observedArc);
  for (unsigned selection = 0; selection < 2; ++selection) {
    auto transaction = take(candidate->beginTransaction(scratch));
    requireSuccess(transaction.addFragments(fragments));
    if (!take(transaction.close()))
      fail("exact compute placement closed a handshake cycle");
    requireSuccess(transaction.commit());
  }
  if (candidate->fragmentRefcount(*observedFragment) != 2)
    fail("shared handshake fragment lost its decision refcount");
  const loom::pnr::PnrIndex selectedArcRefcount =
      candidate->arcRefcount(*observedArc);
  if (selectedArcRefcount <= baseArcRefcount)
    fail("selected handshake fragment did not activate its arc");
  {
    auto transaction = take(candidate->beginTransaction(scratch));
    requireSuccess(transaction.removeFragments(fragments));
    if (!take(transaction.close()))
      fail("handshake deletion reported a cycle");
    transaction.rollback();
  }
  if (candidate->fragmentRefcount(*observedFragment) != 2 ||
      candidate->arcRefcount(*observedArc) != selectedArcRefcount)
    fail("handshake rollback changed the committed refcounts");
  for (unsigned selection = 0; selection < 2; ++selection) {
    auto transaction = take(candidate->beginTransaction(scratch));
    requireSuccess(transaction.removeFragments(fragments));
    if (!take(transaction.close()))
      fail("handshake deletion reported a cycle");
    requireSuccess(transaction.commit());
  }
  if (candidate->fragmentRefcount(*observedFragment) != 0 ||
      candidate->arcRefcount(*observedArc) != baseArcRefcount ||
      scratch.retainedStorageBytes() != retainedScratchBytes)
    fail("handshake selection removal retained state or expanded scratch");
  requireSuccess(candidate->verify());
}

void computeBoundaryClosure() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  mlir::MLIRContext context = makeContext();

  auto dataflowArtifact = buildComputeDataflow(context);
  take(dataflow::publishCanonicalDataflow(dataflowArtifact, store));
  auto dataflowView = take(dataflowArtifact.view());

  loom::adg::DesignBuilder builder(store);
  auto expansion = take(loom::adg::expandBuiltinSpatialCore(
      builder, loom::adg::BuiltinTargetPreset::Small));
  if (llvm::Error error = expansion.spatialCore.close(expansion.outputs))
    fail(llvm::toString(std::move(error)));
  auto design = take(std::move(builder).finalize());
  const auto &fabricRoot = design.roots().front();
  const auto selected =
      selectComputeCapability(computeActor(dataflowView), fabricRoot.view());

  auto complete =
      parseMapping(context, computeMappingText(dataflowView, fabricRoot.view(),
                                               selected, true));
  if (!complete)
    fail("complete compute TechMapping fixture did not parse");
  auto completeRoots = complete->getOps<::mapping::TechOp>();
  auto finalized = take(loom::mapping::finalizeTechMapping(
      *completeRoots.begin(), dataflowView, fabricRoot.view(), store));

  auto emptyConstraints = parseMapping(
      context,
      spatialConstraintText(dataflowView, finalized.view(), fabricRoot.view(),
                            /*clauses=*/""));
  if (!emptyConstraints)
    fail("compute Spatial MappingConstraintSet fixture did not parse");
  auto constraintRoots =
      emptyConstraints->getOps<::mapping::ConstraintsSpatialOp>();
  auto constraints = take(loom::mapping::finalizeSpatialMappingConstraintSet(
      *constraintRoots.begin(), dataflowView, finalized.view(),
      fabricRoot.view(), store));
  const loom::pnr::ResolvedPnrConfigView spatialConfig =
      take(loom::pnr::projectResolvedSpatialPnrConfigView(
          loom::defaultResolvedConfig()));
  auto frozen = take(loom::pnr::freezeSpatialPnrProblem(
      dataflowView, finalized.view(), fabricRoot.view(), spatialConfig,
      constraints.view()));
  const auto &handshake = frozen->handshake();
  if (handshake.computePlacementFragmentOffsets().size() !=
          frozen->realizations().computePlacements().size() + 1 ||
      handshake.computePlacementFragments().empty())
    fail("compute freeze omitted exact placement handshake fragments");
  exerciseHandshakeCandidateRefcounts(frozen);
  if (frozen->ports().portDemands().size() != 4 ||
      frozen->ports().graphBoundaries().size() != 4)
    fail("compute freeze omitted actor or graph-boundary demands");
  for (const auto &demand : frozen->ports().portDemands()) {
    if (demand.kind != loom::pnr::FrozenSpatialPortDemandKind::Compute ||
        demand.placementDomainCount == 0)
      fail("compute PortDemand lost its factorized placement domain");
    for (const auto &domain : frozen->ports().placementDomains().slice(
             demand.placementDomainOffset, demand.placementDomainCount)) {
      const auto &placement =
          frozen->realizations().computePlacements()[domain.placement];
      for (const auto &option : frozen->ports().attachmentOptions().slice(
               domain.attachmentOptionOffset, domain.attachmentOptionCount)) {
        if (!option.localTraversal)
          fail("compute PortDemand omitted its exact PE selector traversal");
        const auto &traversal =
            frozen->routing().traversals()[*option.localTraversal].reference;
        const auto *selector =
            std::get_if<loom::fabric::FabricPeSelectorPayload>(
                &traversal.payload);
        if (!selector || selector->owner != placement.parentPe)
          fail("compute PortDemand selected a foreign local traversal");
      }
    }
  }

  auto missing =
      parseMapping(context, computeMappingText(dataflowView, fabricRoot.view(),
                                               selected, false));
  if (!missing)
    fail("missing-boundary TechMapping fixture did not parse");
  auto missingRoots = missing->getOps<::mapping::TechOp>();
  if (!rejected(
          loom::mapping::finalizeTechMapping(*missingRoots.begin(), store)))
    fail("compute realization without its FU boundaries was published");
}

} // namespace

int main() {
  artifactRoundTripAndReferenceValidation();
  computeBoundaryClosure();
  llvm::outs() << "tech mapping artifact tests passed\n";
  return 0;
}

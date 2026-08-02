#include "ADG/Builtin.h"

#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/IR/MappingOps.h"

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

template <typename T> bool rejected(llvm::Expected<T> value) {
  if (value)
    return false;
  llvm::consumeError(value.takeError());
  return true;
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
      %start: none, %index: index, %memory: memref<4xi32>) -> i32
      attributes {input_segments = array<i32: 1, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value, %done = dataflow.load %memory[%index] %start : memref<4xi32>
    dataflow.graph.return values(%value : i32) streams() memories()
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
  auto finalized =
      take(loom::mapping::finalizeTechMapping(*roots.begin(), store));
  if (finalized.view().dataflowIdentity() != dataflowReference.artifact ||
      finalized.view().fabricIdentity() != fabricRoot.reference().artifact ||
      finalized.view().memoryRealizations().size() != 1)
    fail("sealed TechMapping view lost its exact upstream binding");

  auto imported =
      take(loom::mapping::importTechMapping(finalized.reference(), store));
  if (imported.reference() != finalized.reference() ||
      !imported.canonicalBytes().bytes().equals(
          finalized.canonicalBytes().bytes()) ||
      imported.view().memoryRealizations().front().actors.size() != 1)
    fail("strict TechMapping import changed the canonical artifact");

  auto stale = parseMapping(
      context, mappingText(dataflowView, fabricRoot.view(), selected, true));
  if (!stale)
    fail("stale-reference fixture did not parse");
  auto staleRoots = stale->getOps<::mapping::TechOp>();
  if (!rejected(loom::mapping::finalizeTechMapping(*staleRoots.begin(), store)))
    fail("stale Fabric memory endpoint was published");
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
  take(loom::mapping::finalizeTechMapping(*completeRoots.begin(), store));

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

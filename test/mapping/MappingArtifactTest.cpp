#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <string>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "mapping artifact test failure: " << message << '\n';
  std::exit(1);
}

std::string rawByteArray(std::size_t count, std::uint8_t value) {
  std::string result = "[";
  for (std::size_t index = 0; index < count; ++index) {
    if (index != 0)
      result += ", ";
    result += std::to_string(value);
  }
  result += "]";
  return result;
}

std::string identityAttr(std::uint8_t value) {
  return "#mapping.artifact_identity<" + rawByteArray(32, value) + ">";
}

std::string rawU64Ref(std::uint64_t value) {
  std::string result = "[";
  for (unsigned byte = 0; byte < 8; ++byte) {
    if (byte != 0)
      result += ", ";
    const unsigned shift = 8 * (7 - byte);
    result += std::to_string(static_cast<std::uint8_t>(value >> shift));
  }
  result += "]";
  return result;
}

std::string graphRef(std::uint64_t value) {
  return "#mapping.graph_ref<" + rawU64Ref(value) + ">";
}

std::string actorRef(std::uint64_t value) {
  return "#mapping.actor_ref<" + rawU64Ref(value) + ">";
}

std::string fuCapabilityRef(std::uint64_t fu, std::uint64_t ordinal) {
  std::string result = "[0, 0, 0, 2";
  for (unsigned byte = 0; byte < 16; ++byte) {
    result += ", ";
    const std::uint64_t value = byte < 8 ? fu : ordinal;
    const unsigned shift = 8 * (7 - (byte % 8));
    result += std::to_string(static_cast<std::uint8_t>(value >> shift));
  }
  result += "]";
  return "#mapping.fabric_fu_capability_template_ref<" + result + ">";
}

std::string fuNodeRef(std::uint64_t fu, std::uint64_t ordinal) {
  std::string result = "[0, 0, 0, 0, 0, 0, 0, 2";
  for (unsigned byte = 0; byte < 16; ++byte) {
    const std::uint64_t value = byte < 8 ? fu : ordinal;
    const unsigned shift = 8 * (7 - (byte % 8));
    result += ", " + std::to_string(static_cast<std::uint8_t>(value >> shift));
  }
  result += "]";
  return "#mapping.fabric_fu_template_node_ref<" + result + ">";
}

std::string fuPortRef(std::uint64_t fu, bool output, std::uint64_t ordinal) {
  std::string result = "[0, 0, 0, 2";
  for (unsigned byte = 0; byte < 8; ++byte) {
    result += ", " +
              std::to_string(static_cast<std::uint8_t>(fu >> (8 * (7 - byte))));
  }
  result += output ? ", 0, 0, 0, 1" : ", 0, 0, 0, 0";
  for (unsigned byte = 0; byte < 8; ++byte)
    result +=
        ", " +
        std::to_string(static_cast<std::uint8_t>(ordinal >> (8 * (7 - byte))));
  result += "]";
  return "#mapping.fabric_fu_template_port_ref<" + result + ">";
}

std::string fabricEntityRef(std::uint32_t kind, std::uint64_t entity) {
  std::string result = "[";
  for (unsigned byte = 0; byte < 4; ++byte) {
    if (byte != 0)
      result += ", ";
    result +=
        std::to_string(static_cast<std::uint8_t>(kind >> (8 * (3 - byte))));
  }
  for (unsigned byte = 0; byte < 8; ++byte)
    result += ", " + std::to_string(
                         static_cast<std::uint8_t>(entity >> (8 * (7 - byte))));
  result += "]";
  return result;
}

std::string appendRawU64(std::string prefix, std::uint64_t value) {
  prefix.pop_back();
  for (unsigned byte = 0; byte < 8; ++byte)
    prefix += ", " + std::to_string(
                         static_cast<std::uint8_t>(value >> (8 * (7 - byte))));
  prefix += "]";
  return prefix;
}

std::string memoryEngineTemplateRef(std::uint64_t engine) {
  return "#mapping.fabric_memory_engine_template_ref<" +
         fabricEntityRef(16, engine) + ">";
}

std::string memoryEngineOperationPortRef(std::uint64_t engine,
                                         std::uint64_t ordinal) {
  return "#mapping.fabric_memory_engine_template_operation_port_ref<" +
         appendRawU64(fabricEntityRef(16, engine), ordinal) + ">";
}

std::string memoryEngineCapabilityAlternativeRef(std::uint64_t engine,
                                                 std::uint64_t port,
                                                 std::uint64_t ordinal) {
  return "#mapping.fabric_memory_engine_template_capability_alternative_ref<" +
         appendRawU64(appendRawU64(fabricEntityRef(16, engine), port),
                      ordinal) +
         ">";
}

std::string memoryEngineEndpointRef(std::uint64_t engine,
                                    std::uint64_t ordinal) {
  return "#mapping.fabric_memory_engine_template_endpoint_ref<" +
         appendRawU64(fabricEntityRef(16, engine), ordinal) + ">";
}

std::string memoryEngineConnectionRef(std::uint64_t engine,
                                      std::uint64_t source,
                                      std::uint64_t sink) {
  std::string bytes = fabricEntityRef(16, engine);
  bytes.pop_back();
  const auto append = [&](std::uint64_t endpoint, std::string &result) {
    const std::string encoded =
        appendRawU64(fabricEntityRef(16, engine), endpoint);
    result += ", " + encoded.substr(1, encoded.size() - 2);
  };
  append(source, bytes);
  append(sink, bytes);
  bytes += "]";
  return "#mapping.fabric_memory_engine_template_internal_connection_ref<" +
         bytes + ">";
}

std::string graphIngressProducerRef(std::uint64_t graph) {
  return "#mapping.graph_producer_endpoint_ref<[0, 0, 0, 0, 0, 0, 0, 0, " +
         rawU64Ref(graph).substr(1) + ">";
}

std::string actorResultProducerRef(std::uint64_t actor, std::uint64_t ordinal) {
  return "#mapping.graph_producer_endpoint_ref<[0, 0, 0, 1, " +
         rawU64Ref(actor).substr(1, rawU64Ref(actor).size() - 2) + ", " +
         rawU64Ref(ordinal).substr(1) + ">";
}

std::string actorOperandConsumerRef(std::uint64_t actor,
                                    std::uint64_t ordinal) {
  return "#mapping.graph_consumer_endpoint_ref<[0, 0, 0, 0, " +
         rawU64Ref(actor).substr(1, rawU64Ref(actor).size() - 2) + ", " +
         rawU64Ref(ordinal).substr(1) + ">";
}

std::string techModule(bool reverseRealizations, bool semanticDelta) {
  const std::string dataflow = identityAttr(17);
  const std::string fabric = identityAttr(34);
  const std::string graph = graphRef(0);

  auto realization = [&](std::uint64_t actor, std::uint64_t fu,
                         std::uint64_t authoredId) {
    const unsigned operandPort = semanticDelta && actor == 2 ? 1 : 0;
    const std::string actorReference = actorRef(actor);
    return "\n      mapping.compute_realization " + std::to_string(authoredId) +
           " capability(" + fuCapabilityRef(fu, 0) + ") {\n" +
           "        mapping.compute_boundary actor(" + actorReference +
           ") input 0 fu_port(" + fuPortRef(fu, false, 0) + ")\n" +
           "        mapping.compute_actor actor(" + actorReference + ") op(" +
           fuNodeRef(fu, 0) + ") operand_ports([" +
           std::to_string(operandPort) + "]) result_ports([0])\n" + "      }\n";
  };

  const std::string first = realization(1, 7, reverseRealizations ? 1 : 0);
  const std::string second = realization(2, 9, reverseRealizations ? 0 : 1);
  const std::string body =
      reverseRealizations ? second + first : first + second;
  return "module {\n  mapping.tech version<2, 0> dataflow(" + dataflow +
         ") fabric(" + fabric + ") covers([" + graph + "]) {" + body + "  }\n}";
}

std::string memoryTechModule(bool reverseChildren, bool semanticDelta,
                             bool wrongOwner) {
  const std::string dataflow = identityAttr(17);
  const std::string fabric = identityAttr(34);
  const std::uint64_t engine = 41;
  const std::uint64_t childEngine = wrongOwner ? 42 : engine;
  const std::uint64_t resultEndpoint = semanticDelta ? 4 : 3;

  const std::string actor =
      "        mapping.memory_actor actor(" + actorRef(7) +
      ") operation_port(" + memoryEngineOperationPortRef(childEngine, 0) +
      ") capability(" +
      memoryEngineCapabilityAlternativeRef(childEngine, 0, 1) +
      ") operand_ports([" + memoryEngineEndpointRef(childEngine, 0) + ", " +
      memoryEngineEndpointRef(childEngine, 1) + "]) result_ports([" +
      memoryEngineEndpointRef(childEngine, resultEndpoint) + "])\n";
  const std::string boundary =
      "        mapping.memory_graph_boundary terminal(" +
      graphIngressProducerRef(0) + ") endpoint(" +
      memoryEngineEndpointRef(childEngine, 0) + ")\n";
  const std::string edge = "        mapping.memory_internal_edge producer(" +
                           actorResultProducerRef(7, 0) + ") consumer(" +
                           actorOperandConsumerRef(8, 1) + ") connection(" +
                           memoryEngineConnectionRef(childEngine, 3, 2) + ")\n";
  const std::string children =
      reverseChildren ? edge + boundary + actor : actor + boundary + edge;

  return "module {\n  mapping.tech version<2, 0> dataflow(" + dataflow +
         ") fabric(" + fabric + ") covers([" + graphRef(0) + "]) {\n" +
         "      mapping.memory_realization 9 engine(" +
         memoryEngineTemplateRef(engine) + ") {\n" + children +
         "      }\n  }\n}";
}

std::string memoryRealizationCardinalityModule() {
  const std::string dataflow = identityAttr(17);
  const std::string fabric = identityAttr(34);
  const std::uint64_t engine = 41;
  const auto actor = [&](std::uint64_t ordinal) {
    return "        mapping.memory_actor actor(" + actorRef(ordinal) +
           ") operation_port(" + memoryEngineOperationPortRef(engine, 0) +
           ") capability(" +
           memoryEngineCapabilityAlternativeRef(engine, 0, 1) +
           ") operand_ports([" + memoryEngineEndpointRef(engine, 0) +
           "]) result_ports([" + memoryEngineEndpointRef(engine, 1) + "])\n";
  };
  return "module {\n  mapping.tech version<2, 0> dataflow(" + dataflow +
         ") fabric(" + fabric + ") covers([" + graphRef(0) + "]) {\n" +
         "      mapping.memory_realization 9 engine(" +
         memoryEngineTemplateRef(engine) + ") {\n" + actor(1) + actor(3) +
         "      }\n" + "      mapping.memory_realization 4 engine(" +
         memoryEngineTemplateRef(engine) + ") {\n" + actor(2) +
         "      }\n  }\n}";
}

mlir::OwningOpRef<mlir::ModuleOp> parse(mlir::MLIRContext &context,
                                        llvm::StringRef text) {
  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(text, "mapping-artifact-test"),
      llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
}

loom::CanonicalSemanticBytes canonicalBytes(mlir::ModuleOp module) {
  auto roots = module.getOps<::mapping::TechOp>();
  if (std::distance(roots.begin(), roots.end()) != 1)
    fail("expected exactly one mapping.tech root");
  auto bytes = loom::mapping::writeCanonicalMappingAssembly(*roots.begin());
  if (!bytes)
    fail(llvm::toString(bytes.takeError()));
  return std::move(*bytes);
}

void testCanonicalAuthoringOrder() {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry);

  auto ordered = parse(context, techModule(false, false));
  auto reversed = parse(context, techModule(true, false));
  if (!ordered || !reversed) {
    llvm::errs() << techModule(false, false) << '\n';
    fail("valid mapping.tech syntax did not parse");
  }
  if (mlir::failed(mlir::verify(*ordered)) ||
      mlir::failed(mlir::verify(*reversed)))
    fail("valid mapping.tech syntax did not verify");

  const auto orderedBytes = canonicalBytes(*ordered);
  const auto reversedBytes = canonicalBytes(*reversed);
  if (!orderedBytes.bytes().equals(reversedBytes.bytes()))
    fail("authoring order or draft entity IDs changed canonical bytes");

  auto delta = parse(context, techModule(false, true));
  if (!delta || mlir::failed(mlir::verify(*delta)))
    fail("semantic-delta mapping.tech did not verify");
  const auto deltaBytes = canonicalBytes(*delta);
  if (orderedBytes.bytes().equals(deltaBytes.bytes()))
    fail("semantic port correspondence did not change canonical bytes");
}

void testMalformedScopedReference() {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry);

  std::string text = techModule(false, false);
  const std::string valid = "covers([" + graphRef(0) + "])";
  const std::string invalid = "covers([#mapping.graph_ref<[0, 0, 0, 0]>])";
  const std::size_t position = text.find(valid);
  if (position == std::string::npos)
    fail("test fixture did not contain its graph reference");
  text.replace(position, valid.size(), invalid);

  auto module = parse(context, text);
  if (!module)
    return;
  if (mlir::succeeded(mlir::verify(*module)))
    fail("malformed GraphRef payload passed verification");
}

void testComputePortOrdinalRange() {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry);

  std::string text = techModule(false, false);
  const std::string valid = "operand_ports([0])";
  const std::size_t position = text.find(valid);
  if (position == std::string::npos)
    fail("compute port fixture has no operand map");
  text.replace(position, valid.size(), "operand_ports([-4294967296])");
  auto module = parse(context, text);
  if (!module)
    return;
  if (mlir::succeeded(mlir::verify(*module)))
    fail("negative 64-bit compute port ordinal passed verification");
}

void testCanonicalMemoryRealization() {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry);

  auto ordered = parse(context, memoryTechModule(false, false, false));
  auto reversed = parse(context, memoryTechModule(true, false, false));
  if (!ordered || !reversed) {
    llvm::errs() << memoryTechModule(false, false, false) << '\n';
    fail("valid memory realization syntax did not parse");
  }
  if (mlir::failed(mlir::verify(*ordered)) ||
      mlir::failed(mlir::verify(*reversed)))
    fail("valid memory realization did not verify");

  const auto orderedBytes = canonicalBytes(*ordered);
  const auto reversedBytes = canonicalBytes(*reversed);
  if (!orderedBytes.bytes().equals(reversedBytes.bytes()))
    fail("memory realization authoring order changed canonical bytes");

  auto delta = parse(context, memoryTechModule(false, true, false));
  if (!delta || mlir::failed(mlir::verify(*delta)))
    fail("memory realization semantic delta did not verify");
  if (orderedBytes.bytes().equals(canonicalBytes(*delta).bytes()))
    fail("memory endpoint correspondence did not change canonical bytes");
}

void testMemoryTemplateOwnerMismatch() {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry);

  auto module = parse(context, memoryTechModule(false, false, true));
  if (!module)
    return;
  if (mlir::succeeded(mlir::verify(*module)))
    fail("wrong-owner memory template reference passed verification");
}

void testMemoryPayloadCardinalityOrder() {
  mlir::DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  mlir::MLIRContext context(registry);

  auto authored = parse(context, memoryRealizationCardinalityModule());
  if (!authored || mlir::failed(mlir::verify(*authored)))
    fail("memory cardinality fixture did not verify");
  const auto bytes = canonicalBytes(*authored);
  const std::string text(bytes.bytes().begin(), bytes.bytes().end());
  auto canonical = parse(context, text);
  if (!canonical || mlir::failed(mlir::verify(*canonical)))
    fail("canonical memory cardinality fixture did not parse");
  auto roots = canonical->getOps<::mapping::TechOp>();
  auto root = *roots.begin();
  auto realizations =
      root.getBody().front().getOps<::mapping::MemoryRealizationOp>();
  if (std::distance(realizations.begin(), realizations.end()) != 2)
    fail("canonical memory cardinality fixture changed realization count");
  auto first = *realizations.begin();
  auto actors = first.getBody().front().getOps<::mapping::MemoryActorOp>();
  if (first.getEntityId() != 0 ||
      std::distance(actors.begin(), actors.end()) != 1)
    fail("memory payload cardinality did not determine canonical row order");
}

} // namespace

int main() {
  testCanonicalAuthoringOrder();
  testMalformedScopedReference();
  testComputePortOrdinalRange();
  testCanonicalMemoryRealization();
  testMemoryTemplateOwnerMismatch();
  testMemoryPayloadCardinalityOrder();
  llvm::outs() << "mapping artifact tests passed\n";
  return 0;
}

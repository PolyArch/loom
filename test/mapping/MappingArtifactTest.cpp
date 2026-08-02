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

} // namespace

int main() {
  testCanonicalAuthoringOrder();
  testMalformedScopedReference();
  llvm::outs() << "mapping artifact tests passed\n";
  return 0;
}

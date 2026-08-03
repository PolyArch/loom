#include "Common/Artifact.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Identity/FabricRefBytes.h"
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

} // namespace

int main() {
  testTypedSpatialResourceUse();
  testResourceUseOwnerClosure();
  llvm::outs() << "spatial mapping wire tests passed\n";
  return 0;
}

#include "Hardware/RTL/RtlBlockSource.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Hardware/Implementation/RepresentationIndex.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/AsmParser/AsmParser.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <set>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_block_source_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> bytesOf(llvm::StringRef text) {
  return {reinterpret_cast<const std::uint8_t *>(text.data()), text.size()};
}

llvm::StringRef textOf(llvm::ArrayRef<std::uint8_t> bytes) {
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

struct SourceFacts final {
  RtlBlockSourceProjection projection;
  RtlDomainPortNames ports;
  std::optional<fabric::ClockDomainContractRecord> clock;
};

llvm::Expected<std::string> encode(const SourceFacts &facts) {
  std::optional<std::string> clockHex;
  if (facts.clock) {
    auto bytes = fabric::encodeClockDomainContractRecord(*facts.clock);
    if (!bytes)
      return bytes.takeError();
    clockHex = formatArtifactLocalPayloadHex(*bytes);
  }
  const RtlModuleGraphProjection &graph = facts.projection.graph;
  std::string result;
  llvm::raw_string_ostream output(result);
  llvm::json::OStream json(output);
  json.object([&] {
    json.attribute("source", formatBlobDigestHex(*graph.sourceDigest));
    json.attribute("preamble_bytes",
                   graph.preamble ? graph.preamble->byteCount : 0);
    json.attribute("clock_port", facts.ports.clock
                                     ? llvm::json::Value(*facts.ports.clock)
                                     : llvm::json::Value(nullptr));
    json.attribute("reset_port", facts.ports.reset
                                     ? llvm::json::Value(*facts.ports.reset)
                                     : llvm::json::Value(nullptr));
    json.attribute("clock_contract", clockHex ? llvm::json::Value(*clockHex)
                                              : llvm::json::Value(nullptr));
    json.attributeArray("definitions", [&] {
      for (const RtlModuleProjection &module : graph.modules)
        json.object([&] {
          json.attribute("name", module.emittedName);
          json.attribute("kind", static_cast<std::uint32_t>(module.kind));
          json.attribute("parameters", module.parameters);
          json.attribute("source_bytes",
                         module.emission ? module.emission->byteCount : 0);
          json.attributeArray("ports", [&] {
            for (const auto &port : module.ports)
              json.object([&] {
                json.attribute("name", port.name);
                json.attribute("type", port.type);
                json.attribute("attributes", port.attributes);
                json.attribute("direction",
                               static_cast<std::uint32_t>(port.direction));
              });
          });
          json.attributeArray("dependencies", [&] {
            for (const auto &dependency : module.dependencies)
              json.object([&] {
                json.attribute("definition", static_cast<std::uint64_t>(
                                                 dependency.targetModule));
                json.attribute("multiplicity", dependency.multiplicity);
              });
          });
        });
    });
  });
  output.flush();
  return result;
}

llvm::Expected<SourceFacts> decode(llvm::StringRef text,
                                   const BlobStore &blobs) {
  auto document = llvm::json::parse(text);
  if (!document)
    return document.takeError();
  llvm::json::Path::Root path;
  llvm::json::ObjectMapper object(*document, path);
  std::string sourceDigest;
  std::uint64_t preambleBytes = 0;
  SourceFacts facts;
  std::optional<std::string> clockContract;
  if (!object || !object.map("source", sourceDigest) ||
      !object.map("preamble_bytes", preambleBytes) ||
      !object.map("clock_port", facts.ports.clock) ||
      !object.map("reset_port", facts.ports.reset) ||
      !object.map("clock_contract", clockContract))
    return path.getError();
  auto digest = parseBlobDigestHex(sourceDigest);
  if (!digest)
    return digest.takeError();
  auto source = blobs.get(*digest);
  if (!source)
    return source.takeError();
  facts.projection.source = textOf(*source).str();
  auto &graph = facts.projection.graph;
  graph.sourceDigest = *digest;
  graph.sourceByteCount = source->size();
  if (clockContract) {
    auto bytes = parseArtifactLocalPayloadHex(*clockContract);
    if (!bytes)
      return bytes.takeError();
    auto clock = fabric::decodeClockDomainContractRecord(*bytes);
    if (!clock)
      return clock.takeError();
    facts.clock = std::move(*clock);
  }
  if (preambleBytes > source->size())
    return invalid("preamble exceeds the stored source");
  if (preambleBytes != 0)
    graph.preamble = RtlModuleEmissionRange{
        0, preambleBytes,
        computeBlobDigest(bytesOf(llvm::StringRef(facts.projection.source)
                                      .take_front(preambleBytes)))};
  std::uint64_t offset = preambleBytes;
  const auto *definitions = document->getAsObject()->getArray("definitions");
  if (!definitions || definitions->empty())
    return invalid("definition catalog is absent or empty");
  for (std::size_t ordinal = 0; ordinal < definitions->size(); ++ordinal) {
    const auto &value = (*definitions)[ordinal];
    llvm::json::ObjectMapper definition(
        value, llvm::json::Path(path).field("definitions").index(ordinal));
    RtlModuleProjection module;
    std::uint32_t kind = 0;
    std::uint64_t sourceBytes = 0;
    if (!definition || !definition.map("name", module.emittedName) ||
        !definition.map("kind", kind) ||
        !definition.map("parameters", module.parameters) ||
        !definition.map("source_bytes", sourceBytes))
      return path.getError();
    if (kind > static_cast<std::uint32_t>(RtlModuleDefinitionKind::External))
      return invalid("unknown definition kind");
    module.kind = static_cast<RtlModuleDefinitionKind>(kind);
    module.irSymbol = module.emittedName;
    module.reachable = true;
    if (sourceBytes > source->size() - offset)
      return invalid("definition exceeds the stored source");
    if (module.kind == RtlModuleDefinitionKind::Concrete) {
      if (sourceBytes == 0)
        return invalid("concrete definition has no source");
      module.emission = RtlModuleEmissionRange{
          offset, sourceBytes,
          computeBlobDigest(bytesOf(llvm::StringRef(facts.projection.source)
                                        .substr(offset, sourceBytes)))};
    } else if (sourceBytes != 0) {
      return invalid("external definition owns source bytes");
    }
    offset += sourceBytes;
    const auto *ports = value.getAsObject()->getArray("ports");
    const auto *dependencies = value.getAsObject()->getArray("dependencies");
    if (!ports || !dependencies)
      return invalid("definition interface or dependency catalog is absent");
    for (std::size_t index = 0; index < ports->size(); ++index) {
      llvm::json::ObjectMapper port((*ports)[index], llvm::json::Path(path)
                                                         .field("definitions")
                                                         .index(ordinal)
                                                         .field("ports")
                                                         .index(index));
      RtlModulePortProjection projected;
      std::uint32_t direction = 0;
      if (!port || !port.map("name", projected.name) ||
          !port.map("type", projected.type) ||
          !port.map("attributes", projected.attributes) ||
          !port.map("direction", direction))
        return path.getError();
      if (direction > static_cast<std::uint32_t>(RtlModulePortDirection::Inout))
        return invalid("unknown port direction");
      projected.direction = static_cast<RtlModulePortDirection>(direction);
      module.ports.push_back(std::move(projected));
    }
    for (std::size_t index = 0; index < dependencies->size(); ++index) {
      llvm::json::ObjectMapper dependency((*dependencies)[index],
                                          llvm::json::Path(path)
                                              .field("definitions")
                                              .index(ordinal)
                                              .field("dependencies")
                                              .index(index));
      std::uint64_t target = 0, multiplicity = 0;
      if (!dependency || !dependency.map("definition", target) ||
          !dependency.map("multiplicity", multiplicity))
        return path.getError();
      if (target >= ordinal || multiplicity == 0)
        return invalid("dependency is not an earlier definition with positive "
                       "multiplicity");
      module.dependencies.push_back(
          {static_cast<std::size_t>(target), multiplicity});
    }
    graph.modules.push_back(std::move(module));
  }
  if (offset != source->size())
    return invalid("definition ranges do not consume the complete source");
  graph.topModule = graph.modules.size() - 1;
  return facts;
}

RepresentationSignalDirection
representationDirection(RtlModulePortDirection direction) {
  switch (direction) {
  case RtlModulePortDirection::Input:
    return RepresentationSignalDirection::Input;
  case RtlModulePortDirection::Output:
    return RepresentationSignalDirection::Output;
  case RtlModulePortDirection::Inout:
    return RepresentationSignalDirection::Inout;
  }
  llvm_unreachable("unknown RTL module port direction");
}

llvm::Error validateRootInterface(const SourceFacts &facts) {
  auto format = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::SystemVerilogRtl);
  if (!format)
    return format.takeError();
  const auto &root = facts.projection.graph.modules.back();
  auto index = indexProspectiveRepresentation(
      *format, {RepresentationObjectKind::Module, root.emittedName},
      {{PayloadRole::RtlSource, "rtl/block.sv",
        bytesOf(facts.projection.source)}});
  if (!index)
    return index.takeError();
  if (!index->unresolvedExternalDefinitions().empty())
    return invalid("portable block source has unresolved external definitions");
  const auto boundary = index->rootBoundaryPorts();
  if (boundary.size() != root.ports.size())
    return invalid("projected root interface differs from the emitted source");
  mlir::MLIRContext context;
  context.loadDialect<circt::hw::HWDialect>();
  std::set<std::string> names;
  for (const auto &port : root.ports) {
    if (!names.insert(port.name).second)
      return invalid("projected root port name is duplicated");
    auto object = index->lookup(
        {RepresentationObjectKind::Port, root.emittedName + "." + port.name});
    if (!object)
      return object.takeError();
    mlir::Type type = mlir::parseType(port.type, &context);
    const auto width = type ? circt::hw::getBitWidth(type) : -1;
    if (!*object || !(**object).signalGeometry || width <= 0 ||
        (**object).signalGeometry->bitWidth !=
            static_cast<std::uint64_t>(width) ||
        (**object).signalGeometry->direction !=
            representationDirection(port.direction))
      return invalid(
          "projected root port geometry differs from the emitted source");
  }
  return llvm::Error::success();
}

llvm::Expected<BlobDigest> validate(SourceFacts &facts) {
  if (facts.clock.has_value() != facts.ports.clock.has_value())
    return invalid("clock port and domain contract must be present together");
  const auto &graph = facts.projection.graph;
  auto source = bindRtlModuleGraphSource(graph, facts.projection.source);
  if (!source)
    return source.takeError();
  auto closure =
      deriveRtlBlockClosure(graph, *source, graph.topModule, facts.ports);
  if (!closure)
    return closure.takeError();
  if (closure->clockPort != facts.ports.clock ||
      closure->resetPort != facts.ports.reset)
    return invalid("domain port does not belong to the block root");
  if (closure->members.size() != graph.modules.size())
    return invalid("definition catalog has aliases or unreachable members");
  auto normalized = projectRtlBlockClosureSource(*closure, graph, *source);
  if (!normalized)
    return normalized.takeError();
  SourceFacts canonical{std::move(*normalized), facts.ports, facts.clock};
  auto expected = encode(canonical);
  if (!expected)
    return expected.takeError();
  auto actual = encode(facts);
  if (!actual)
    return actual.takeError();
  if (canonical.projection.source != facts.projection.source ||
      *expected != *actual)
    return invalid(
        "source and graph are not the canonical normalized block closure");
  if (llvm::Error error = validateRootInterface(facts))
    return std::move(error);
  return closure->identity();
}

llvm::Expected<SourceFacts>
derive(const FinalizedConfigurationABI &configurationAbi,
       const FinalizedHardwareImplementation &implementation,
       std::size_t definition, const BlobStore &blobs) {
  auto graph = projectPortableSpatialCoreRtlModuleGraph(configurationAbi,
                                                        implementation);
  if (!graph)
    return graph.takeError();
  if (!*graph)
    return invalid("source is not the exact canonical portable implementation");
  auto source = blobs.get(*(**graph).sourceDigest);
  if (!source)
    return source.takeError();
  auto bound = bindRtlModuleGraphSource(**graph, textOf(*source));
  if (!bound)
    return bound.takeError();
  auto domain = deriveSpatialCoreClockBinding(
      configurationAbi, implementation.implementation().interfaces());
  if (!domain)
    return domain.takeError();
  auto closure = deriveRtlBlockClosure(**graph, *bound, definition,
                                       {domain->clockPort, domain->resetPort});
  if (!closure)
    return closure.takeError();
  auto projected = projectRtlBlockClosureSource(*closure, **graph, *bound);
  if (!projected)
    return projected.takeError();
  SourceFacts facts{std::move(*projected),
                    {closure->clockPort, closure->resetPort},
                    closure->clockPort ? std::optional(domain->clock)
                                       : std::nullopt};
  auto identity = validate(facts);
  if (!identity)
    return identity.takeError();
  return facts;
}

} // namespace

std::string FinalizedRtlBlockSource::generationConstraint() const {
  return clock_ ? renderCreateClockConstraint(*clock_, *domainPorts_.clock)
                : std::string();
}

llvm::Expected<FinalizedRtlBlockSource>
importRtlBlockSource(const ArtifactRootReference &reference,
                     const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (reference.schemaIdentity != rtlBlockSourceSchema.identity ||
      reference.schemaVersion != rtlBlockSourceSchema.version)
    return invalid("root reference has another schema");
  auto canonical = artifacts.get(rtlBlockSourceSchema, reference.artifact);
  if (!canonical)
    return canonical.takeError();
  const llvm::StringRef text = textOf(canonical->bytes());
  auto facts = decode(text, blobs);
  if (!facts)
    return facts.takeError();
  auto identity = validate(*facts);
  if (!identity)
    return identity.takeError();
  auto encoded = encode(*facts);
  if (!encoded)
    return encoded.takeError();
  if (*encoded != text)
    return invalid("stored source Artifact does not have canonical bytes");
  return FinalizedRtlBlockSource(reference, std::move(facts->projection),
                                 std::move(facts->ports),
                                 std::move(facts->clock), *identity);
}

llvm::Expected<FinalizedRtlBlockSource> finalizePortableRtlBlockSource(
    const FinalizedConfigurationABI &configurationAbi,
    const FinalizedHardwareImplementation &implementation,
    std::size_t definition, const ArtifactStore &artifacts,
    const BlobStore &blobs) {
  auto derived = derive(configurationAbi, implementation, definition, blobs);
  if (!derived)
    return derived.takeError();
  const SourceFacts &facts = *derived;
  auto published = blobs.put(bytesOf(facts.projection.source));
  if (!published)
    return published.takeError();
  auto text = encode(facts);
  if (!text)
    return text.takeError();
  auto artifact =
      artifacts.put(rtlBlockSourceSchema,
                    CanonicalSemanticBytes(
                        std::vector<std::uint8_t>(text->begin(), text->end())));
  if (!artifact)
    return artifact.takeError();
  return importRtlBlockSource({rtlBlockSourceSchema.identity.str(),
                               rtlBlockSourceSchema.version, *artifact},
                              artifacts, blobs);
}

llvm::Error verifyPortableRtlBlockSourceDerivation(
    const FinalizedRtlBlockSource &source,
    const FinalizedConfigurationABI &configurationAbi,
    const FinalizedHardwareImplementation &implementation,
    std::size_t definition, const BlobStore &blobs) {
  auto derived = derive(configurationAbi, implementation, definition, blobs);
  if (!derived)
    return derived.takeError();
  auto text = encode(*derived);
  if (!text)
    return text.takeError();
  const auto identity = finalizeArtifactIdentity(
      rtlBlockSourceSchema, CanonicalSemanticBytes(std::vector<std::uint8_t>(
                                text->begin(), text->end())));
  if (identity != source.reference().artifact)
    return invalid("source Artifact is not the selected block of this parent "
                   "implementation");
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl

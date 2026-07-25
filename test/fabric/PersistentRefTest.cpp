#include "Common/Artifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace loom;
using namespace loom::fabric;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << "\n";
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(const char *test, llvm::Error error,
                    const std::string &message) {
  if (error)
    fail(test, message + ": " + llvm::toString(std::move(error)));
}

void requireKind(const char *test, llvm::Error error, FabricRefErrorKind kind,
                 const std::string &message) {
  if (!error)
    fail(test, message + ": expected " + fabricRefKeyword(kind).str());
  const FabricRefErrorKind actual = takeFabricRefErrorKind(std::move(error));
  if (actual != kind)
    fail(test, message + ": expected " + fabricRefKeyword(kind).str() +
                   ", got " + fabricRefKeyword(actual).str());
}

template <typename T>
void requireRejected(const char *test, llvm::Expected<T> value,
                     const std::string &message) {
  if (value)
    fail(test, message + " must be rejected");
  llvm::consumeError(value.takeError());
}

/// Parsing then canonically printing must reproduce the accepted spelling.
template <typename Ref>
void requireCanonical(const char *test, llvm::StringRef spelling) {
  const std::string printed =
      printFabricRef(takeExpected(test, parseFabricRef<Ref>(spelling)));
  if (printed != spelling)
    fail(test, "canonical print of '" + spelling.str() + "' was '" + printed +
                   "'");
}

template <typename Ref>
void requireParseKind(const char *test, llvm::StringRef spelling,
                      FabricRefErrorKind kind) {
  llvm::Expected<Ref> parsed = parseFabricRef<Ref>(spelling);
  if (parsed)
    fail(test, "accepted '" + spelling.str() + "'");
  requireKind(test, parsed.takeError(), kind, "parse '" + spelling.str() + "'");
}

ArtifactIdentity identity(const char *test, std::uint8_t seed) {
  std::vector<std::uint8_t> bytes(ArtifactIdentity::byteSize, seed);
  return takeExpected(test, ArtifactIdentity::fromBytes(bytes));
}

/// Owner-declared inventory answers for one small elaborated Fabric. The view
/// stores exactly the facts the importer asks for: nothing here is a topology
/// catalog, object graph, property map, or dense persistent index.
class TestView : public FabricArtifactView {
public:
  TestView(ArtifactIdentity artifact, FabricRootKind root)
      : artifact_(std::move(artifact)), root_(root) {}

  const ArtifactIdentity &identity() const override { return artifact_; }
  FabricRootKind rootKind() const override { return root_; }

  std::optional<FabricEntityKind> entityKind(FabricEntityId id) const override {
    auto entry = entities_.find(id);
    if (entry == entities_.end())
      return std::nullopt;
    return entry->second;
  }

  std::uint64_t transportEndpointCount(
      const FabricTransportEndpointOwnerRef &owner) const override {
    return lookup(transportEndpoints_, canonicalFabricBytes(owner));
  }

  std::uint64_t memoryEndpointCount(
      const FabricMemoryEndpointOwnerRef &owner) const override {
    return lookup(memoryEndpoints_, canonicalFabricBytes(owner));
  }

  std::uint64_t inventorySize(const FabricInventoryOwnerRef &owner,
                              FabricInventoryKind inventory) const override {
    std::vector<std::uint8_t> key = canonicalFabricBytes(owner);
    key.push_back(static_cast<std::uint8_t>(inventory));
    return lookup(inventories_, key);
  }

  std::optional<FabricFuTemplateRef>
  fuTemplateOf(FabricFuOccurrenceRef occurrence) const override {
    auto entry = fuTemplates_.find(occurrence.id());
    if (entry == fuTemplates_.end())
      return std::nullopt;
    return FabricFuTemplateRef(entry->second);
  }

  bool
  hasPointConnection(const FabricTransportEndpointRef &source,
                     const FabricTransportEndpointRef &target) const override {
    std::vector<std::uint8_t> key = canonicalFabricBytes(source);
    std::vector<std::uint8_t> tail = canonicalFabricBytes(target);
    key.insert(key.end(), tail.begin(), tail.end());
    return lookup(connections_, key) != 0;
  }

  bool admitsTraversal(const FabricPhysicalTraversalRef &t) const override {
    return lookup(traversals_, canonicalFabricBytes(t)) != 0;
  }

  void addEntity(FabricEntityId id, FabricEntityKind kind) {
    entities_[id] = kind;
  }
  void elaborate(FabricFuOccurrenceRef occurrence, FabricFuTemplateRef from) {
    fuTemplates_[occurrence.id()] = from.id();
  }
  void setTransportEndpoints(const FabricTransportEndpointOwnerRef &owner,
                             std::uint64_t count) {
    transportEndpoints_[canonicalFabricBytes(owner)] = count;
  }
  void setMemoryEndpoints(const FabricMemoryEndpointOwnerRef &owner,
                          std::uint64_t count) {
    memoryEndpoints_[canonicalFabricBytes(owner)] = count;
  }
  void setInventory(const FabricInventoryOwnerRef &owner,
                    FabricInventoryKind inventory, std::uint64_t count) {
    std::vector<std::uint8_t> key = canonicalFabricBytes(owner);
    key.push_back(static_cast<std::uint8_t>(inventory));
    inventories_[key] = count;
  }
  void connect(const FabricTransportEndpointRef &source,
               const FabricTransportEndpointRef &target) {
    std::vector<std::uint8_t> key = canonicalFabricBytes(source);
    std::vector<std::uint8_t> tail = canonicalFabricBytes(target);
    key.insert(key.end(), tail.begin(), tail.end());
    connections_[key] = 1;
  }
  void admit(const FabricPhysicalTraversalRef &traversal) {
    traversals_[canonicalFabricBytes(traversal)] = 1;
  }

private:
  using ByteMap = std::map<std::vector<std::uint8_t>, std::uint64_t>;

  static std::uint64_t lookup(const ByteMap &map,
                              const std::vector<std::uint8_t> &key) {
    auto entry = map.find(key);
    return entry == map.end() ? 0 : entry->second;
  }

  ArtifactIdentity artifact_;
  FabricRootKind root_;
  llvm::DenseMap<FabricEntityId, FabricEntityKind> entities_;
  llvm::DenseMap<FabricEntityId, FabricEntityId> fuTemplates_;
  ByteMap transportEndpoints_;
  ByteMap memoryEndpoints_;
  ByteMap inventories_;
  ByteMap connections_;
  ByteMap traversals_;
};

// Entity identifiers of the fixture Fabric.
constexpr FabricEntityId kFuTemplate = 7;
constexpr FabricEntityId kFuOccurrenceA = 11;
constexpr FabricEntityId kFuOccurrenceB = 12;
constexpr FabricEntityId kSwitch = 21;
constexpr FabricEntityId kMemory = 31;
constexpr FabricEntityId kAccCore = 41;
constexpr FabricEntityId kHardwareDomain = 51;

TestView buildView(const char *test) {
  TestView view(identity(test, 0x11), FabricRootKind::Module);
  view.addEntity(kFuTemplate, FabricEntityKind::FabricFuTemplate);
  view.addEntity(kFuOccurrenceA, FabricEntityKind::FabricFuOccurrence);
  view.addEntity(kFuOccurrenceB, FabricEntityKind::FabricFuOccurrence);
  view.addEntity(kSwitch, FabricEntityKind::FabricSwitchOccurrence);
  view.addEntity(kMemory, FabricEntityKind::FabricMemoryOccurrence);
  view.addEntity(kAccCore, FabricEntityKind::AccCoreOccurrence);
  view.addEntity(kHardwareDomain, FabricEntityKind::HardwareDomain);

  const FabricFuTemplateRef fuTemplate(kFuTemplate);
  const FabricFuOccurrenceRef occurrenceA(kFuOccurrenceA);
  const FabricFuOccurrenceRef occurrenceB(kFuOccurrenceB);
  view.elaborate(occurrenceA, fuTemplate);
  view.elaborate(occurrenceB, fuTemplate);

  view.setInventory(FabricInventoryOwnerRef::of(fuTemplate),
                    FabricInventoryKind::FuNode, 4);
  for (FabricFuOccurrenceRef occurrence : {occurrenceA, occurrenceB}) {
    view.setInventory(FabricInventoryOwnerRef::of(occurrence),
                      FabricInventoryKind::FuNode, 4);
    view.setTransportEndpoints(FabricTransportEndpointOwnerRef::of(occurrence),
                               3);
  }

  const FabricSwitchOccurrenceRef switchOccurrence(kSwitch);
  view.setInventory(FabricInventoryOwnerRef::of(switchOccurrence),
                    FabricInventoryKind::SwitchInput, 2);
  view.setInventory(FabricInventoryOwnerRef::of(switchOccurrence),
                    FabricInventoryKind::SwitchOutput, 2);
  view.setInventory(FabricInventoryOwnerRef::of(switchOccurrence),
                    FabricInventoryKind::ResourceState, 3);
  view.setTransportEndpoints(
      FabricTransportEndpointOwnerRef::of(switchOccurrence), 4);

  // The memory occurrence exposes four token endpoints but only one memory
  // capability endpoint. Equal ordinals therefore do not select equal objects.
  const FabricMemoryOccurrenceRef memory(kMemory);
  view.setTransportEndpoints(FabricTransportEndpointOwnerRef::of(memory), 4);
  view.setMemoryEndpoints(FabricMemoryEndpointOwnerRef::of(memory), 1);
  view.setInventory(FabricInventoryOwnerRef::of(memory),
                    FabricInventoryKind::MemoryOperationPort, 2);
  view.setInventory(FabricInventoryOwnerRef::of(
                        FabricMemoryServiceRef::local(memory)),
                    FabricInventoryKind::MemoryServiceRegion, 2);

  view.setTransportEndpoints(
      FabricTransportEndpointOwnerRef::of(
          SpatialCoreOccurrenceRef{AccCoreOccurrenceRef(kAccCore)}),
      2);
  return view;
}

FabricTransportEndpointRef switchEndpoint(FabricOrdinal ordinal) {
  return FabricTransportEndpointRef{
      FabricTransportEndpointOwnerRef::of(FabricSwitchOccurrenceRef(kSwitch)),
      ordinal};
}

FabricPhysicalTraversalRef switchTraversal(FabricOrdinal in, FabricOrdinal out) {
  return FabricPhysicalTraversalRef::switchTraversal(
      FabricSwitchOccurrenceRef(kSwitch), in, out);
}

const FabricImportBinding &binding(const char *test) {
  static const FabricImportBinding value{identity(test, 0x11),
                                         FabricRootKind::Module};
  return value;
}

/// Distinct physical occurrences elaborated from one FU template stay distinct
/// while their exact template correspondence remains recoverable.
void testFuOccurrenceIdentity() {
  const char *test = "fu-occurrence-identity";
  TestView view = buildView(test);

  const FabricFuTemplateRef fuTemplate(kFuTemplate);
  const FabricFuOccurrenceRef occurrenceA(kFuOccurrenceA);
  const FabricFuOccurrenceRef occurrenceB(kFuOccurrenceB);

  static_assert(!std::is_same_v<FabricFuTemplateRef, FabricFuOccurrenceRef>,
                "template and occurrence references are distinct types");
  static_assert(
      !std::is_convertible_v<FabricFuTemplateRef, FabricFuOccurrenceRef>,
      "a template reference never converts to an occurrence reference");

  require(test, occurrenceA != occurrenceB, "occurrences must stay distinct");
  require(test,
          canonicalFabricBytes(occurrenceA) != canonicalFabricBytes(occurrenceB),
          "distinct occurrences must have distinct canonical bytes");
  // One template reused by two occurrences never merges their identity, and a
  // template reference is not an occurrence reference with the same number.
  require(test,
          canonicalFabricBytes(FabricFuTemplateRef(kFuOccurrenceA)) !=
              canonicalFabricBytes(occurrenceA),
          "entity kind must separate equal identifiers");

  require(test, view.fuTemplateOf(occurrenceA) == fuTemplate,
          "occurrence A must retain its template");
  require(test, view.fuTemplateOf(occurrenceB) == fuTemplate,
          "occurrence B must retain its template");

  const FabricFuTemplateNodeRef templateNode{FabricFuNodeKind::Mux, fuTemplate,
                                             2};
  const FabricFuOccurrenceNodeRef nodeA =
      takeExpected(test, deriveFabricFuOccurrenceNode(view, templateNode,
                                                      occurrenceA));
  const FabricFuOccurrenceNodeRef nodeB =
      takeExpected(test, deriveFabricFuOccurrenceNode(view, templateNode,
                                                      occurrenceB));
  require(test, nodeA != nodeB, "derived occurrence nodes must stay distinct");
  require(test, nodeA.node == templateNode.node && nodeA.ordinal == 2,
          "derivation must preserve the selected template node");

  // A node of an unrelated template cannot be paired with this occurrence.
  const FabricFuTemplateNodeRef foreignNode{FabricFuNodeKind::Mux,
                                            FabricFuTemplateRef(kSwitch), 2};
  llvm::Expected<FabricFuOccurrenceNodeRef> unrelated =
      deriveFabricFuOccurrenceNode(view, foreignNode, occurrenceA);
  require(test, !unrelated, "unrelated node ordinals must not pair");
  requireKind(test, unrelated.takeError(), FabricRefErrorKind::WrongEntityKind,
              "unrelated template node");

  requireSuccess(test, validateFabricRef(view, templateNode), "template node");
  requireSuccess(test, validateFabricRef(view, nodeA), "occurrence node");
  requireKind(test,
              validateFabricRef(view, FabricFuTemplateNodeRef{
                                          FabricFuNodeKind::Op, fuTemplate, 4}),
              FabricRefErrorKind::OrdinalOutOfRange, "node ordinal 4 of 4");
}

/// A switch traversal, a point connection, and an induced resource state stay
/// three distinct typed objects.
void testTraversalDistinctions() {
  const char *test = "traversal-distinctions";
  TestView view = buildView(test);

  static_assert(!std::is_same_v<FabricPhysicalTraversalRef,
                                FabricResourceStateRef>,
                "a traversal is not a resource state");
  static_assert(!std::is_convertible_v<FabricPhysicalTraversalRef,
                                       FabricResourceStateRef>,
                "a traversal never converts to a resource state");

  const FabricPhysicalTraversalRef traversal = switchTraversal(0, 1);
  const FabricPhysicalTraversalRef connection =
      FabricPhysicalTraversalRef::pointConnection(switchEndpoint(2),
                                                  switchEndpoint(3));
  const FabricResourceStateRef state{
      FabricInventoryOwnerRef::of(FabricSwitchOccurrenceRef(kSwitch)), 1};

  require(test, traversal != connection,
          "switch traversal and point connection must differ");
  require(test,
          canonicalFabricBytes(traversal) != canonicalFabricBytes(connection),
          "traversal variants must have distinct canonical bytes");
  require(test,
          canonicalFabricBytes(traversal) != canonicalFabricBytes(state),
          "a traversal is not its induced resource state");
  require(test, printFabricRef(traversal) != printFabricRef(state),
          "a traversal never prints as a resource state");

  view.admit(traversal);
  view.connect(switchEndpoint(2), switchEndpoint(3));

  requireSuccess(test, validateFabricRef(view, traversal), "switch traversal");
  requireSuccess(test, validateFabricRef(view, connection), "point connection");
  requireSuccess(test, validateFabricRef(view, state), "resource state");

  // An in-range traversal the switch contract does not admit is invalid.
  requireKind(test, validateFabricRef(view, switchTraversal(1, 0)),
              FabricRefErrorKind::TraversalNotAdmitted, "unadmitted traversal");
  // An out-of-range switch ordinal is an ordinal failure, not a contract one.
  requireKind(test, validateFabricRef(view, switchTraversal(0, 2)),
              FabricRefErrorKind::OrdinalOutOfRange, "switch output 2 of 2");
  // A well-formed connection absent from the elaborated Fabric is invalid.
  requireKind(test,
              validateFabricRef(view, FabricPhysicalTraversalRef::pointConnection(
                                          switchEndpoint(3), switchEndpoint(2))),
              FabricRefErrorKind::AbsentPointConnection, "absent connection");
  requireKind(test,
              validateFabricRef(
                  view, FabricResourceStateRef{FabricInventoryOwnerRef::of(
                                                   FabricSwitchOccurrenceRef(
                                                       kSwitch)),
                                               3}),
              FabricRefErrorKind::OrdinalOutOfRange, "state ordinal 3 of 3");
}

/// Token transport and memory-service capability stay separate planes.
void testEndpointPlanes() {
  const char *test = "endpoint-planes";
  TestView view = buildView(test);

  static_assert(
      !std::is_same_v<FabricTransportEndpointRef, FabricMemoryEndpointRef>,
      "token and memory endpoints are distinct types");
  static_assert(!std::is_convertible_v<FabricTransportEndpointRef,
                                       FabricMemoryEndpointRef>,
                "a token endpoint never converts to a memory endpoint");

  const FabricMemoryOccurrenceRef memory(kMemory);
  const FabricTransportEndpointRef token{
      FabricTransportEndpointOwnerRef::of(memory), 3};
  const FabricMemoryEndpointRef capability{
      FabricMemoryEndpointOwnerRef::of(memory), 3};

  // Equal owner and equal ordinal select different inventories.
  requireSuccess(test, validateFabricRef(view, token), "token endpoint 3");
  requireKind(test, validateFabricRef(view, capability),
              FabricRefErrorKind::OrdinalOutOfRange, "memory endpoint 3 of 1");

  // A token-plane owner used on the memory plane is plane misuse, while an
  // owner that exposes neither plane is an invalid owner family.
  requireParseKind<FabricMemoryEndpointRef>(
      test, "fabric.memory_endpoint<fabric.switch_occurrence<21>, 0>",
      FabricRefErrorKind::PlaneMisuse);
  requireParseKind<FabricTransportEndpointRef>(
      test, "fabric.transport_endpoint<fabric.hardware_domain<51>, 0>",
      FabricRefErrorKind::InvalidOwnerFamily);

  // A SpatialCore attachment is not the AccCore that owns it.
  const AccCoreOccurrenceRef accCore(kAccCore);
  const FabricTransportEndpointRef spatial{
      FabricTransportEndpointOwnerRef::of(SpatialCoreOccurrenceRef{accCore}), 1};
  const FabricTransportEndpointRef core{
      FabricTransportEndpointOwnerRef::of(accCore), 1};
  require(test, canonicalFabricBytes(spatial) != canonicalFabricBytes(core),
          "spatial core and acc core owners must differ");
  requireSuccess(test, validateFabricRef(view, spatial), "spatial endpoint");
  requireKind(test, validateFabricRef(view, core),
              FabricRefErrorKind::OrdinalOutOfRange, "acc core endpoint 1 of 0");
}

/// Import rejects every identity failure the specification classifies.
void testImportRejection() {
  const char *test = "import-rejection";
  TestView view = buildView(test);

  const FabricFuOccurrenceRef occurrence(kFuOccurrenceA);
  requireSuccess(test,
                 importFabricRef(view, binding(test),
                                 ArtifactReference<FabricFuOccurrenceRef>{
                                     identity(test, 0x11), occurrence}),
                 "exact artifact");
  requireKind(test,
              importFabricRef(view, binding(test),
                              ArtifactReference<FabricFuOccurrenceRef>{
                                  identity(test, 0x22), occurrence}),
              FabricRefErrorKind::ForeignArtifact, "foreign artifact");

  const FabricImportBinding wrongRoot{identity(test, 0x11),
                                      FabricRootKind::System};
  requireKind(test,
              importFabricRef(view, wrongRoot,
                              ArtifactReference<FabricFuOccurrenceRef>{
                                  identity(test, 0x11), occurrence}),
              FabricRefErrorKind::WrongRootKind, "wrong root kind");

  requireKind(test, validateFabricRef(view, FabricFuOccurrenceRef(999)),
              FabricRefErrorKind::UnknownEntity, "unknown entity");
  requireKind(test, validateFabricRef(view, FabricFuOccurrenceRef(kFuTemplate)),
              FabricRefErrorKind::WrongEntityKind, "wrong entity kind");

  // Owner-union membership never implies a nonempty inventory.
  requireKind(test,
              validateFabricRef(
                  view, FabricUsePatternRef{FabricInventoryOwnerRef::of(
                                                HardwareDomainRef(
                                                    kHardwareDomain)),
                                            0}),
              FabricRefErrorKind::OrdinalOutOfRange, "empty use patterns");

  // Deprecated and generic escapes are rejected as such, not as syntax noise.
  requireParseKind<FabricFuOccurrenceRef>(test, "fabric.fu<11>",
                                          FabricRefErrorKind::DeprecatedAlias);
  requireParseKind<FabricFuOccurrenceRef>(test, "fabric.fu_occurrence<11>.port[3]",
                                          FabricRefErrorKind::DeprecatedAlias);
  requireParseKind<FabricFuOccurrenceRef>(test, "@fu_occurrence",
                                          FabricRefErrorKind::DeprecatedAlias);
  requireParseKind<FabricFuOccurrenceRef>(test, "#fabric.fu_occurrence<11>",
                                          FabricRefErrorKind::DeprecatedAlias);
  requireParseKind<FabricFuOccurrenceRef>(
      test, "fabric.fu_occurrence<11> loc(\"x\")",
      FabricRefErrorKind::DeprecatedAlias);
}

/// The strict text codec accepts exactly the canonical typed language.
void testStrictText() {
  const char *test = "strict-text";

  // Each family is its own accepted language; parsing then canonically
  // printing reproduces exactly the accepted spelling.
  requireCanonical<FabricFuOccurrenceRef>(test, "fabric.fu_occurrence<11>");
  requireCanonical<FabricFuTemplateNodeRef>(
      test, "fabric.fu_template_node<mux, fabric.fu_template<7>, 2>");
  requireCanonical<FabricFuNodePortRef>(
      test, "fabric.fu_node_port<fabric.fu_template_node<op, "
            "fabric.fu_template<7>, 0>, input, 1>");
  requireCanonical<FabricTransportEndpointRef>(
      test, "fabric.transport_endpoint<fabric.switch_occurrence<21>, 0>");
  requireCanonical<FabricMemoryEndpointRef>(
      test, "fabric.memory_endpoint<fabric.memory_occurrence<31>, 0>");
  requireCanonical<FabricMemoryServiceRegionRef>(
      test, "fabric.memory_service_region<fabric.memory_service<local, "
            "fabric.memory_occurrence<31>>, 1>");
  requireCanonical<InstructionCoreContextRef>(
      test, "fabric.instruction_core_context<fabric.acc_core_occurrence<41>>");
  requireCanonical<FabricResourceStateRef>(
      test, "fabric.resource_state<fabric.switch_occurrence<21>, 1>");
  requireCanonical<FabricPhysicalTraversalRef>(
      test, "fabric.traversal<switch, fabric.switch_occurrence<21>, 0, 1>");
  requireCanonical<FabricPhysicalTraversalRef>(
      test, "fabric.traversal<fifo, fabric.fifo_occurrence<61>, bypass>");

  // Unknown variants, wrong arity, and noncanonical numbers are rejected.
  requireParseKind<FabricPhysicalTraversalRef>(
      test, "fabric.traversal<crossbar, fabric.switch_occurrence<21>, 0, 1>",
      FabricRefErrorKind::MalformedSyntax);
  requireParseKind<FabricPhysicalTraversalRef>(
      test, "fabric.traversal<switch, fabric.switch_occurrence<21>, 0>",
      FabricRefErrorKind::MalformedSyntax);
  requireParseKind<FabricPhysicalTraversalRef>(
      test, "fabric.traversal<switch, fabric.switch_occurrence<21>, 0, 1, 2>",
      FabricRefErrorKind::MalformedSyntax);
  requireParseKind<FabricFuOccurrenceRef>(test, "fabric.fu_occurrence<-1>",
                                          FabricRefErrorKind::MalformedSyntax);
  requireParseKind<FabricFuOccurrenceRef>(test, "fabric.fu_occurrence<011>",
                                          FabricRefErrorKind::MalformedSyntax);
  requireParseKind<FabricFuOccurrenceRef>(test, "fabric.fu_occurrence<0x11>",
                                          FabricRefErrorKind::MalformedSyntax);
  requireParseKind<FabricFuOccurrenceRef>(
      test, "fabric.fu_occurrence<18446744073709551616>",
      FabricRefErrorKind::MalformedSyntax);
  // A well-formed reference of another family is not this family.
  requireParseKind<FabricFuOccurrenceRef>(test, "fabric.fu_template<7>",
                                          FabricRefErrorKind::WrongEntityKind);
}

/// Text, canonical bytes, and import agree on one exact typed record.
void testRoundTrip() {
  const char *test = "round-trip";
  TestView view = buildView(test);

  const FabricPhysicalTraversalRef traversal = switchTraversal(0, 1);
  view.admit(traversal);

  const std::string text = printFabricRef(traversal);
  const FabricPhysicalTraversalRef parsed =
      takeExpected(test, parseFabricRef<FabricPhysicalTraversalRef>(text));
  require(test, parsed == traversal, "parse must recover the exact record");

  const std::vector<std::uint8_t> bytes = canonicalFabricBytes(parsed);
  const FabricPhysicalTraversalRef decoded = takeExpected(
      test, decodeFabricRef<FabricPhysicalTraversalRef>(bytes));
  require(test, decoded == traversal, "decode must recover the exact record");
  require(test, printFabricRef(decoded) == text,
          "canonical print must be unique");
  requireSuccess(test, validateFabricRef(view, decoded), "imported traversal");

  // Canonical bytes are unsigned 32-bit big-endian tags followed by unsigned
  // 64-bit big-endian semantic fields, with no padding or native layout.
  const std::vector<std::uint8_t> entity =
      canonicalFabricBytes(FabricFuOccurrenceRef(kFuOccurrenceB));
  const std::vector<std::uint8_t> expected = {
      0x00, 0x00, 0x00,
      static_cast<std::uint8_t>(FabricEntityKind::FabricFuOccurrence),
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c};
  require(test, entity == expected, "canonical entity bytes");
  static_assert(static_cast<std::uint32_t>(
                    FabricEntityKind::FabricModuleTemplate) == 0,
                "closed sums are zero based in declaration order");
  static_assert(static_cast<std::uint32_t>(
                    FabricPhysicalTraversalKind::PointConnection) == 0,
                "traversal variants are zero based in declaration order");

  // Trailing or truncated canonical bytes are not a second encoding.
  std::vector<std::uint8_t> extended = bytes;
  extended.push_back(0);
  requireRejected(test, decodeFabricRef<FabricPhysicalTraversalRef>(extended),
                  "trailing canonical bytes");
  std::vector<std::uint8_t> truncated = bytes;
  truncated.pop_back();
  requireRejected(test, decodeFabricRef<FabricPhysicalTraversalRef>(truncated),
                  "truncated canonical bytes");
}

} // namespace

int main() {
  testFuOccurrenceIdentity();
  testTraversalDistinctions();
  testEndpointPlanes();
  testImportRejection();
  testStrictText();
  testRoundTrip();
  llvm::outs() << "fabric persistent references ok\n";
  return 0;
}

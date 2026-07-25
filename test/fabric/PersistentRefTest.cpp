#include "Common/Artifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricRefText.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
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
                    const std::string &what) {
  if (error)
    fail(test, what + ": " + llvm::toString(std::move(error)));
}

void requireKind(const char *test, llvm::Error error, FabricRefErrorKind kind,
                 const std::string &what) {
  if (!error)
    fail(test, what + ": expected " + fabricRefKeyword(kind).str());
  const FabricRefErrorKind actual = takeFabricRefErrorKind(std::move(error));
  if (actual != kind)
    fail(test, what + ": expected " + fabricRefKeyword(kind).str() + ", got " +
                   fabricRefKeyword(actual).str());
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

template <typename T>
void requireRejected(const char *test, llvm::Expected<T> value,
                     const std::string &what) {
  if (value)
    fail(test, what + " must be rejected");
  llvm::consumeError(value.takeError());
}

ArtifactIdentity identity(const char *test, std::uint8_t seed) {
  std::vector<std::uint8_t> bytes(ArtifactIdentity::byteSize, seed);
  return takeExpected(test, ArtifactIdentity::fromBytes(bytes));
}

// The fixed anchor entities. Identifiers are chosen so that one numeric value
// never names two kinds, which lets the wrong-kind anchors stay unambiguous.
constexpr FabricEntityId kFuTemplate = 7;
constexpr FabricEntityId kOtherFuTemplate = 8;
constexpr FabricEntityId kOccurrenceA = 11;
constexpr FabricEntityId kOccurrenceB = 12;
constexpr FabricEntityId kSwitch = 21;
constexpr FabricEntityId kMemory = 31;
constexpr FabricEntityId kBareMemory = 32;
constexpr FabricEntityId kAccCore = 41;
constexpr FabricEntityId kConsistencyDomain = 51;
constexpr FabricEntityId kClockDomain = 52;
constexpr FabricEntityId kSystemService = 61;
constexpr FabricEntityId kAbsentEntity = 999;

/// One small elaborated Fabric answering only from its own typed facts. Every
/// answer is a switch over typed reference fields, so the validation oracle
/// stays independent of the text and byte codecs under test and cannot be
/// confused by two references that happen to encode alike.
class AnchorFabric : public FabricArtifactView {
public:
  AnchorFabric(ArtifactIdentity artifact, FabricRootKind root)
      : artifact_(std::move(artifact)), root_(root) {}

  const ArtifactIdentity &identity() const override { return artifact_; }
  FabricRootKind rootKind() const override { return root_; }

  std::optional<FabricEntityKind> entityKind(FabricEntityId id) const override {
    switch (id) {
    case kFuTemplate:
    case kOtherFuTemplate:
      return FabricEntityKind::FabricFuTemplate;
    case kOccurrenceA:
    case kOccurrenceB:
      return FabricEntityKind::FabricFuOccurrence;
    case kSwitch:
      return FabricEntityKind::FabricSwitchOccurrence;
    case kMemory:
    case kBareMemory:
      return FabricEntityKind::FabricMemoryOccurrence;
    case kAccCore:
      return FabricEntityKind::AccCoreOccurrence;
    case kConsistencyDomain:
    case kClockDomain:
      return FabricEntityKind::HardwareDomain;
    case kSystemService:
      return FabricEntityKind::SystemMemoryService;
    default:
      return std::nullopt;
    }
  }

  // The switch and both FU occurrences expose token terminals; the memory
  // occurrence exposes four of them but only one memory capability endpoint.
  std::uint64_t transportEndpointCount(
      const FabricTransportEndpointOwnerRef &owner) const override {
    switch (owner.kind()) {
    case FabricTransportEndpointOwnerKind::FabricSwitchOccurrence:
    case FabricTransportEndpointOwnerKind::FabricMemoryOccurrence:
      return 4;
    case FabricTransportEndpointOwnerKind::FabricFuOccurrence:
      return 3;
    case FabricTransportEndpointOwnerKind::SpatialCoreOccurrence:
      return 2;
    default:
      return 0;
    }
  }

  std::uint64_t memoryEndpointCount(
      const FabricMemoryEndpointOwnerRef &owner) const override {
    switch (owner.kind()) {
    case FabricMemoryEndpointOwnerKind::FabricMemoryOccurrence:
    case FabricMemoryEndpointOwnerKind::SystemMemoryService:
      return 1;
    default:
      return 0;
    }
  }

  std::uint64_t inventorySize(const FabricInventoryOwnerRef &owner,
                              FabricInventoryKind inventory) const override {
    switch (inventory) {
    case FabricInventoryKind::FuNode:
      return isFuOwner(owner) ? 4 : 0;
    case FabricInventoryKind::ResourceState:
      return isSwitchOwner(owner) ? 3 : 0;
    case FabricInventoryKind::SwitchInput:
    case FabricInventoryKind::SwitchOutput:
      return isSwitchOwner(owner) ? 2 : 0;
    case FabricInventoryKind::MemoryOperationPort:
      return owner.kind() == FabricInventoryOwnerKind::MemoryOccurrence ? 2 : 0;
    case FabricInventoryKind::MemoryServiceRegion:
      return owner.kind() == FabricInventoryOwnerKind::MemoryService ? 2 : 0;
    default:
      // No owner of this Fabric declares a use pattern, configuration field,
      // or refinement domain, so an owner union member with an empty
      // inventory stays observable.
      return 0;
    }
  }

  // Node ordinals carry exactly one kind each in the configured graph both FU
  // templates declare, and an occurrence inherits its template's graph.
  std::optional<FabricFuNodeKind>
  fuNodeKind(const FabricInventoryOwnerRef &owner,
             FabricOrdinal ordinal) const override {
    if (!isFuOwner(owner) || ordinal >= 4)
      return std::nullopt;
    return ordinal == 2 ? FabricFuNodeKind::Mux : FabricFuNodeKind::Op;
  }

  bool declaresLocalMemoryService(
      FabricMemoryOccurrenceRef memory) const override {
    return memory.id() == kMemory;
  }

  std::optional<FabricMemoryEndpointRole>
  memoryEndpointRole(const FabricMemoryEndpointRef &endpoint) const override {
    switch (endpoint.owner.kind()) {
    case FabricMemoryEndpointOwnerKind::FabricMemoryOccurrence:
      return FabricMemoryEndpointRole::Subordinate;
    case FabricMemoryEndpointOwnerKind::SystemMemoryService:
      return FabricMemoryEndpointRole::Manager;
    default:
      return std::nullopt;
    }
  }

  std::optional<FabricHardwareDomainKind>
  hardwareDomainKind(HardwareDomainRef domain) const override {
    switch (domain.id()) {
    case kConsistencyDomain:
      return FabricHardwareDomainKind::MemoryConsistency;
    case kClockDomain:
      return FabricHardwareDomainKind::Clock;
    default:
      return std::nullopt;
    }
  }

  std::optional<FabricFuTemplateRef>
  fuTemplateOf(FabricFuOccurrenceRef occurrence) const override {
    if (occurrence.id() != kOccurrenceA && occurrence.id() != kOccurrenceB)
      return std::nullopt;
    return FabricFuTemplateRef(kFuTemplate);
  }

  // Exactly one directed fixed connection exists between switch terminals two
  // and three; the reverse direction is deliberately absent.
  bool
  hasPointConnection(const FabricTransportEndpointRef &source,
                     const FabricTransportEndpointRef &target) const override {
    return source == switchEndpoint(2) && target == switchEndpoint(3);
  }

  // The switch contract admits exactly one input-to-output turn.
  bool admitsTraversal(const FabricPhysicalTraversalRef &t) const override {
    return t == FabricPhysicalTraversalRef::switchTraversal(
                    FabricSwitchOccurrenceRef(kSwitch), 0, 1);
  }

  static FabricTransportEndpointRef switchEndpoint(FabricOrdinal ordinal) {
    return FabricTransportEndpointRef{
        FabricTransportEndpointOwnerRef::of(FabricSwitchOccurrenceRef(kSwitch)),
        ordinal};
  }

private:
  static bool isFuOwner(const FabricInventoryOwnerRef &owner) {
    return owner.kind() == FabricInventoryOwnerKind::FuTemplate ||
           owner.kind() == FabricInventoryOwnerKind::FuOccurrence;
  }
  static bool isSwitchOwner(const FabricInventoryOwnerRef &owner) {
    return owner.kind() == FabricInventoryOwnerKind::SwitchOccurrence;
  }

  ArtifactIdentity artifact_;
  FabricRootKind root_;
};

AnchorFabric fabric(const char *test) {
  return AnchorFabric(identity(test, 0x11), FabricRootKind::Module);
}

FabricImportBinding binding(const char *test) {
  return FabricImportBinding{identity(test, 0x11), FabricRootKind::Module};
}

/// Distinct occurrences elaborated from one template stay distinct while their
/// exact template correspondence remains recoverable.
void testFuOccurrenceIdentity() {
  const char *test = "fu-occurrence-identity";
  const AnchorFabric view = fabric(test);

  const FabricFuTemplateRef fuTemplate(kFuTemplate);
  const FabricFuOccurrenceRef occurrenceA(kOccurrenceA);
  const FabricFuOccurrenceRef occurrenceB(kOccurrenceB);

  static_assert(!std::is_same_v<FabricFuTemplateRef, FabricFuOccurrenceRef>,
                "template and occurrence references are distinct types");
  static_assert(
      !std::is_convertible_v<FabricFuTemplateRef, FabricFuOccurrenceRef>,
      "a template reference never converts to an occurrence reference");

  require(test, occurrenceA != occurrenceB, "occurrences must stay distinct");
  require(test,
          canonicalFabricBytes(occurrenceA) != canonicalFabricBytes(occurrenceB),
          "distinct occurrences need distinct canonical bytes");
  // Template reuse never merges identity, and equal numbers under different
  // entity kinds are different targets.
  require(test,
          canonicalFabricBytes(FabricFuTemplateRef(kOccurrenceA)) !=
              canonicalFabricBytes(occurrenceA),
          "entity kind must separate equal identifiers");
  require(test,
          view.fuTemplateOf(occurrenceA) == fuTemplate &&
              view.fuTemplateOf(occurrenceB) == fuTemplate,
          "both occurrences must retain their template");

  const FabricFuTemplateNodeRef node{FabricFuNodeKind::Mux, fuTemplate, 2};
  const FabricFuOccurrenceNodeRef nodeA =
      takeExpected(test, deriveFabricFuOccurrenceNode(view, node, occurrenceA));
  const FabricFuOccurrenceNodeRef nodeB =
      takeExpected(test, deriveFabricFuOccurrenceNode(view, node, occurrenceB));
  require(test, nodeA != nodeB, "derived occurrence nodes stay distinct");
  require(test, nodeA.node == node.node && nodeA.ordinal == node.ordinal,
          "derivation preserves the selected template node");
  requireSuccess(test, validateFabricRef(view, nodeA), "derived node");
}

/// A traversal, a point connection, and an induced resource state remain three
/// distinct typed objects with three distinct validation rules.
void testTraversalDistinctions() {
  const char *test = "traversal-distinctions";
  const AnchorFabric view = fabric(test);

  static_assert(
      !std::is_same_v<FabricPhysicalTraversalRef, FabricResourceStateRef>,
      "a traversal is not a resource state");
  static_assert(
      !std::is_convertible_v<FabricPhysicalTraversalRef, FabricResourceStateRef>,
      "a traversal never converts to a resource state");

  const FabricSwitchOccurrenceRef switchOccurrence(kSwitch);
  const FabricPhysicalTraversalRef traversal =
      FabricPhysicalTraversalRef::switchTraversal(switchOccurrence, 0, 1);
  const FabricPhysicalTraversalRef connection =
      FabricPhysicalTraversalRef::pointConnection(
          AnchorFabric::switchEndpoint(2), AnchorFabric::switchEndpoint(3));
  const FabricResourceStateRef state{
      FabricResourceStateOwnerRef(
          FabricInventoryOwnerRef::of(switchOccurrence)),
      1};

  require(test, traversal != connection,
          "a switch traversal is not a point connection");
  require(test,
          canonicalFabricBytes(traversal) != canonicalFabricBytes(connection) &&
              canonicalFabricBytes(traversal) != canonicalFabricBytes(state),
          "the three objects need three canonical encodings");
  require(test, printFabricRef(traversal) != printFabricRef(state),
          "a traversal never prints as a resource state");

  requireSuccess(test, validateFabricRef(view, traversal), "switch traversal");
  requireSuccess(test, validateFabricRef(view, connection), "point connection");
  requireSuccess(test, validateFabricRef(view, state), "resource state");

  // An in-range turn the switch contract does not admit is a contract failure,
  // while an out-of-range ordinal remains an ordinal failure.
  requireKind(test,
              validateFabricRef(view, FabricPhysicalTraversalRef::switchTraversal(
                                          switchOccurrence, 1, 0)),
              FabricRefErrorKind::TraversalNotAdmitted, "unadmitted turn");
  requireKind(test,
              validateFabricRef(view, FabricPhysicalTraversalRef::switchTraversal(
                                          switchOccurrence, 0, 2)),
              FabricRefErrorKind::OrdinalOutOfRange, "switch output 2 of 2");
  requireKind(test,
              validateFabricRef(view,
                                FabricPhysicalTraversalRef::pointConnection(
                                    AnchorFabric::switchEndpoint(3),
                                    AnchorFabric::switchEndpoint(2))),
              FabricRefErrorKind::AbsentPointConnection, "reverse direction");
  requireKind(test,
              validateFabricRef(
                  view, FabricResourceStateRef{
                            FabricResourceStateOwnerRef(
                                FabricInventoryOwnerRef::of(switchOccurrence)),
                            3}),
              FabricRefErrorKind::OrdinalOutOfRange, "state ordinal 3 of 3");
}

/// Token transport and memory-service capability stay separate planes.
void testEndpointPlanes() {
  const char *test = "endpoint-planes";
  const AnchorFabric view = fabric(test);

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

  // One owner, one ordinal, two inventories.
  requireSuccess(test, validateFabricRef(view, token), "token endpoint 3");
  requireKind(test, validateFabricRef(view, capability),
              FabricRefErrorKind::OrdinalOutOfRange, "memory endpoint 3 of 1");

  // An owner of the other plane is plane misuse; an owner of neither plane is
  // an invalid owner family.
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

/// The four owner projections share one constructor catalog while remaining
/// four distinct static types with one canonical owner encoding.
void testOwnerProjections() {
  const char *test = "owner-projections";
  const AnchorFabric view = fabric(test);

  static_assert(
      !std::is_same_v<FabricResourceStateOwnerRef, FabricUsePatternOwnerRef>,
      "resource-state and use-pattern owners are distinct types");
  static_assert(!std::is_convertible_v<FabricResourceStateOwnerRef,
                                       FabricConfigurationOwnerRef>,
                "one owner projection never converts to another");
  static_assert(!std::is_convertible_v<FabricRefinementOwnerRef,
                                       FabricUsePatternOwnerRef>,
                "one owner projection never converts to another");

  const FabricInventoryOwnerRef catalog =
      FabricInventoryOwnerRef::of(FabricSwitchOccurrenceRef(kSwitch));
  const FabricResourceStateRef state{FabricResourceStateOwnerRef(catalog), 1};
  const FabricUsePatternRef pattern{FabricUsePatternOwnerRef(catalog), 1};

  // The projection is static type information: it never becomes a second
  // serialized owner identity.
  require(test,
          canonicalFabricBytes(state.owner) == canonicalFabricBytes(catalog) &&
              printFabricRef(state.owner) == printFabricRef(catalog),
          "a projection must not change the canonical owner form");

  // The same catalog instance is valid in one projection and empty in another.
  requireSuccess(test, validateFabricRef(view, state), "resource state");
  requireKind(test, validateFabricRef(view, pattern),
              FabricRefErrorKind::OrdinalOutOfRange, "empty use patterns");
}

/// The role-specific refinements select an underlying reference by static type
/// and add no wrapper, tag, copied role field, or second identity.
void testTypedRefinements() {
  const char *test = "typed-refinements";
  const AnchorFabric view = fabric(test);

  static_assert(!std::is_same_v<ManagerEndpointRef, SubordinateEndpointRef>,
                "manager and subordinate endpoints are distinct types");
  static_assert(
      !std::is_convertible_v<FabricMemoryEndpointRef, ManagerEndpointRef>,
      "an unrefined endpoint never converts to a refined one");

  const FabricMemoryOccurrenceRef memory(kMemory);
  const FabricMemoryServiceRef service = FabricMemoryServiceRef::local(memory);
  const LocalMemoryServiceRef local(service);
  require(test,
          canonicalFabricBytes(local) == canonicalFabricBytes(service) &&
              printFabricRef(local) == printFabricRef(service),
          "a refinement keeps the underlying canonical form");
  requireSuccess(test, validateFabricRef(view, local), "local memory service");
  requireSuccess(test, validateFabricRef(view, service), "generic local");

  // The Local variant is valid only when the memory occurrence declares its
  // optional Local Memory Service. The rule belongs to the generic service
  // reference, so the refined name, the generic name, and every nested use
  // reject an undeclared service alike.
  const FabricMemoryOccurrenceRef bare(kBareMemory);
  const FabricMemoryServiceRef absent = FabricMemoryServiceRef::local(bare);
  requireSuccess(test, validateFabricRef(view, bare), "bare memory occurrence");
  requireKind(test, validateFabricRef(view, absent),
              FabricRefErrorKind::WrongEntityKind, "generic absent service");
  requireKind(test, validateFabricRef(view, LocalMemoryServiceRef(absent)),
              FabricRefErrorKind::WrongEntityKind, "refined absent service");
  requireKind(test,
              validateFabricRef(view,
                                FabricMemoryServiceRegionRef{absent, 0}),
              FabricRefErrorKind::WrongEntityKind, "region of absent service");
  requireKind(test,
              validateFabricRef(view, FabricInventoryOwnerRef::of(absent)),
              FabricRefErrorKind::WrongEntityKind, "owner of absent service");

  // The owner inventory, not the reference, decides which endpoint name holds.
  const FabricMemoryEndpointRef subordinate{
      FabricMemoryEndpointOwnerRef::of(memory), 0};
  const FabricMemoryEndpointRef manager{
      FabricMemoryEndpointOwnerRef::of(SystemMemoryServiceRef(kSystemService)),
      0};
  requireSuccess(test,
                 validateFabricRef(view, SubordinateEndpointRef(subordinate)),
                 "subordinate endpoint");
  requireSuccess(test, validateFabricRef(view, ManagerEndpointRef(manager)),
                 "manager endpoint");
  requireKind(test, validateFabricRef(view, ManagerEndpointRef(subordinate)),
              FabricRefErrorKind::WrongEntityKind, "subordinate as manager");
  requireKind(test, validateFabricRef(view, SubordinateEndpointRef(manager)),
              FabricRefErrorKind::WrongEntityKind, "manager as subordinate");

  requireSuccess(test,
                 validateFabricRef(view, MemoryConsistencyDomainRef{
                                             HardwareDomainRef(
                                                 kConsistencyDomain)}),
                 "memory consistency domain");
  requireKind(test,
              validateFabricRef(view, MemoryConsistencyDomainRef{
                                          HardwareDomainRef(kClockDomain)}),
              FabricRefErrorKind::WrongEntityKind, "clock domain");
}

/// An owner mismatch between two individually valid objects is its own
/// identity failure, and one node ordinal admits one node kind.
void testWrongOwner() {
  const char *test = "wrong-owner";
  const AnchorFabric view = fabric(test);

  const FabricFuOccurrenceRef occurrence(kOccurrenceA);
  const FabricFuTemplateNodeRef otherNode{
      FabricFuNodeKind::Mux, FabricFuTemplateRef(kOtherFuTemplate), 2};
  requireSuccess(test, validateFabricRef(view, otherNode), "other node");
  requireSuccess(test, validateFabricRef(view, occurrence), "occurrence");

  llvm::Expected<FabricFuOccurrenceNodeRef> paired =
      deriveFabricFuOccurrenceNode(view, otherNode, occurrence);
  require(test, !paired, "nodes of another template must not pair");
  requireKind(test, paired.takeError(), FabricRefErrorKind::WrongOwner,
              "unrelated template owner");

  requireKind(test,
              validateFabricRef(view, FabricFuTemplateNodeRef{
                                          FabricFuNodeKind::Op,
                                          FabricFuTemplateRef(kFuTemplate), 2}),
              FabricRefErrorKind::WrongEntityKind, "node kind at ordinal 2");
  requireKind(test,
              validateFabricRef(view, FabricFuTemplateNodeRef{
                                          FabricFuNodeKind::Op,
                                          FabricFuTemplateRef(kFuTemplate), 4}),
              FabricRefErrorKind::OrdinalOutOfRange, "node ordinal 4 of 4");
}

/// Import rejects every artifact-scope and entity failure it classifies.
void testImportRejection() {
  const char *test = "import-rejection";
  const AnchorFabric view = fabric(test);
  const FabricFuOccurrenceRef occurrence(kOccurrenceA);

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
  requireKind(test,
              importFabricRef(
                  view,
                  FabricImportBinding{identity(test, 0x11),
                                      FabricRootKind::System},
                  ArtifactReference<FabricFuOccurrenceRef>{
                      identity(test, 0x11), occurrence}),
              FabricRefErrorKind::WrongRootKind, "wrong root kind");

  requireKind(test, validateFabricRef(view, FabricFuOccurrenceRef(kAbsentEntity)),
              FabricRefErrorKind::UnknownEntity, "stale entity");
  requireKind(test, validateFabricRef(view, FabricFuOccurrenceRef(kFuTemplate)),
              FabricRefErrorKind::WrongEntityKind, "wrong entity kind");

  // Deprecated and generic escapes are reported as such, not as syntax noise.
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

  // One entity family, one variant family with closed keyword and ordinals,
  // and one nested union under an owner-relative family.
  requireCanonical<FabricFuOccurrenceRef>(test, "fabric.fu_occurrence<11>");
  requireCanonical<FabricPhysicalTraversalRef>(
      test, "fabric.traversal<switch, fabric.switch_occurrence<21>, 0, 1>");
  requireCanonical<FabricMemoryServiceRegionRef>(
      test, "fabric.memory_service_region<fabric.memory_service<local, "
            "fabric.memory_occurrence<31>>, 1>");

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
  requireParseKind<FabricFuOccurrenceRef>(test, "fabric.fu_template<7>",
                                          FabricRefErrorKind::WrongEntityKind);
}

/// Text, canonical bytes, and import agree on one exact typed record.
void testRoundTrip() {
  const char *test = "round-trip";
  const AnchorFabric view = fabric(test);

  const FabricPhysicalTraversalRef traversal =
      FabricPhysicalTraversalRef::switchTraversal(
          FabricSwitchOccurrenceRef(kSwitch), 0, 1);
  const std::string text = printFabricRef(traversal);
  const FabricPhysicalTraversalRef parsed =
      takeExpected(test, parseFabricRef<FabricPhysicalTraversalRef>(text));
  require(test, parsed == traversal, "parse recovers the exact record");

  const std::vector<std::uint8_t> bytes = canonicalFabricBytes(parsed);
  const FabricPhysicalTraversalRef decoded =
      takeExpected(test, decodeFabricRef<FabricPhysicalTraversalRef>(bytes));
  require(test, decoded == traversal && printFabricRef(decoded) == text,
          "decode recovers the exact record and prints canonically");
  requireSuccess(test, validateFabricRef(view, decoded), "imported traversal");

  // Canonical bytes are unsigned 32-bit big-endian variant tags followed by
  // unsigned 64-bit big-endian fields, with no padding or native layout.
  const auto tagBytes = [](std::uint32_t value) {
    return std::vector<std::uint8_t>{
        static_cast<std::uint8_t>(value >> 24),
        static_cast<std::uint8_t>(value >> 16),
        static_cast<std::uint8_t>(value >> 8),
        static_cast<std::uint8_t>(value)};
  };
  const auto fieldBytes = [](std::uint64_t value) {
    std::vector<std::uint8_t> out;
    for (int shift = 56; shift >= 0; shift -= 8)
      out.push_back(static_cast<std::uint8_t>(value >> shift));
    return out;
  };
  std::vector<std::uint8_t> expected =
      tagBytes(static_cast<std::uint32_t>(
          FabricTransportEndpointOwnerKind::FabricSwitchOccurrence));
  for (std::vector<std::uint8_t> part :
       {tagBytes(static_cast<std::uint32_t>(
            FabricEntityKind::FabricSwitchOccurrence)),
        fieldBytes(kSwitch), fieldBytes(2)})
    expected.insert(expected.end(), part.begin(), part.end());
  require(test,
          canonicalFabricBytes(AnchorFabric::switchEndpoint(2)) == expected,
          "canonical endpoint bytes");

  // Canonical bytes are not a container format and carry no unknown variants.
  std::vector<std::uint8_t> extended = bytes;
  extended.push_back(0);
  requireRejected(test, decodeFabricRef<FabricPhysicalTraversalRef>(extended),
                  "trailing canonical bytes");
  std::vector<std::uint8_t> truncated = bytes;
  truncated.pop_back();
  requireRejected(test, decodeFabricRef<FabricPhysicalTraversalRef>(truncated),
                  "truncated canonical bytes");
  std::vector<std::uint8_t> unknown =
      canonicalFabricBytes(FabricFuOccurrenceRef(kOccurrenceB));
  unknown[3] = 0xff;
  llvm::Expected<FabricFuOccurrenceRef> unknownDecode =
      decodeFabricRef<FabricFuOccurrenceRef>(unknown);
  require(test, !unknownDecode, "unknown entity discriminant is rejected");
  requireKind(test, unknownDecode.takeError(),
              FabricRefErrorKind::MalformedSyntax, "unknown entity kind tag");
}

} // namespace

int main() {
  testFuOccurrenceIdentity();
  testTraversalDistinctions();
  testEndpointPlanes();
  testOwnerProjections();
  testTypedRefinements();
  testWrongOwner();
  testImportRejection();
  testStrictText();
  testRoundTrip();
  llvm::outs() << "fabric persistent references ok\n";
  return 0;
}

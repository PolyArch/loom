#include "FabricCanonicalLabeling.h"
#include "FabricModuleDomainMaterialization.h"
#include "FabricModuleDomainNormalization.h"

#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/FuCapabilityDomain.h"
#include "Fabric/IR/ModuleDomain.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace {

using loom::fabric::FabricEntityKind;
using loom::fabric::detail::FabricCanonicalLabeling;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(1);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *ctx = [] {
    mlir::DialectRegistry registry;
    registry.insert<::fabric::FabricDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *ctx;
}

struct LabeledModule {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  FabricCanonicalLabeling labeling;
};

LabeledModule label(llvm::StringRef test, llvm::StringRef source) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail(test, "unable to parse Fabric fixture");

  ::fabric::ModuleOp root;
  module->walk([&](::fabric::ModuleOp candidate) {
    if (!root)
      root = candidate;
  });
  if (!root)
    fail(test, "fixture has no fabric.module root");

  llvm::Expected<FabricCanonicalLabeling> result =
      loom::fabric::detail::computeFabricModuleCanonicalLabeling(root);
  if (!result)
    fail(test, llvm::toString(result.takeError()));
  return {std::move(module), std::move(*result)};
}

mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef test,
                                        llvm::StringRef source) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(source, &context());
  if (!module)
    fail(test, "unable to parse Fabric fixture");
  return module;
}

std::uint64_t readU64(llvm::StringRef test, llvm::StringRef bytes,
                      std::size_t &offset) {
  if (bytes.size() - std::min(bytes.size(), offset) < 8)
    fail(test, "truncated fabric.op intrinsic");
  std::uint64_t value = 0;
  for (unsigned ordinal = 0; ordinal != 8; ++ordinal)
    value = (value << 8) | static_cast<std::uint8_t>(bytes[offset + ordinal]);
  offset += 8;
  return value;
}

void fabricOpIntrinsicUsesPersistentSchemaIdentity() {
  const llvm::StringRef test = __func__;
  auto module = parse(test, R"mlir(
    module {
      fabric.module @root(%left: !fabric.bits<32>,
                          %right: !fabric.bits<32>) {
        %pe = fabric.pe [spatial]
            (%pe_left = %left : !fabric.bits<32>,
             %pe_right = %right : !fabric.bits<32>) -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fu_left = %pe_left : !fabric.bits<32>,
               %fu_right = %pe_right : !fabric.bits<32>) -> !fabric.bits<32> {
            %result = fabric.op [@arith.subi, @arith.addi]
                (%fu_left, %fu_right)
                {implementation_family =
                   #fabric.implementation_family<ScalarIntegerAddSub>,
                 hw_params = {integer_widths = [32 : i32]}}
                : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %result : !fabric.bits<32>
          }
        }
        fabric.yield
      }
    }
  )mlir");
  ::fabric::OpOp operation;
  module->walk([&](::fabric::OpOp found) { operation = found; });
  require(test, static_cast<bool>(operation), "fixture has no fabric.op");
  auto intrinsic =
      loom::fabric::detail::encodeFabricOpCanonicalIntrinsic(operation);
  if (!intrinsic)
    fail(test, llvm::toString(intrinsic.takeError()));

  constexpr llvm::StringLiteral prefix = "FABRIC_OP\x1f";
  require(test, llvm::StringRef(*intrinsic).starts_with(prefix),
          "fabric.op intrinsic has the wrong domain");
  std::size_t offset = prefix.size() + sizeof(std::uint32_t);
  require(test, readU64(test, *intrinsic, offset) == 2,
          "fabric.op intrinsic has the wrong schema count");

  std::vector<std::vector<std::uint8_t>> identities;
  for (dataflow::OperationSchemaId schema :
       {dataflow::OperationSchemaId::ArithAddI,
        dataflow::OperationSchemaId::ArithSubI}) {
    auto identity = dataflow::encodeOperationSchemaId(schema);
    if (!identity)
      fail(test, llvm::toString(identity.takeError()));
    identities.push_back(identity->bytes().vec());
  }
  llvm::sort(identities);
  for (const std::vector<std::uint8_t> &identity : identities) {
    const std::uint64_t size = readU64(test, *intrinsic, offset);
    require(test, size == identity.size(),
            "fabric.op intrinsic used a noncanonical schema identity length");
    require(test, offset + size <= intrinsic->size(),
            "fabric.op intrinsic schema identity is truncated");
    llvm::StringRef actual(intrinsic->data() + offset, size);
    llvm::StringRef expected(reinterpret_cast<const char *>(identity.data()),
                             identity.size());
    require(test, actual == expected,
            "fabric.op intrinsic did not use the stable schema identity");
    offset += size;
  }
}

void yieldDefaultPropertyDoesNotChangeCanonicalIdentity() {
  static constexpr llvm::StringLiteral custom = R"mlir(
    module {
      fabric.module @root(%value: !fabric.bits<32>) -> !fabric.bits<32> {
        fabric.yield %value : !fabric.bits<32>
      }
    }
  )mlir";
  static constexpr llvm::StringLiteral builderEquivalent = R"mlir(
    module {
      fabric.module @root(%value: !fabric.bits<32>) -> !fabric.bits<32> {
        "fabric.yield"(%value) : (!fabric.bits<32>) -> ()
      }
    }
  )mlir";
  const LabeledModule parsed = label(__func__, custom);
  const LabeledModule built = label(__func__, builderEquivalent);
  require(__func__,
          parsed.labeling.relationBytes.bytes() ==
              built.labeling.relationBytes.bytes(),
          "default declared_types presence changed canonical identity");
}

std::string twoPeModule(llvm::StringRef rootName, bool reverse,
                        bool secondMultiplies) {
  const std::string first = R"mlir(
    fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
      %r = fabric.fu(%x = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
        %v = fabric.op [@arith.addi] (%x, %x)
             {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [32 : i32]}}
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %v : !fabric.bits<32>
      }
    }
  )mlir";
  const std::string second = secondMultiplies ? R"mlir(
    fabric.pe [spatial] (%pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
      %r = fabric.fu(%y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %v = fabric.op [@arith.muli] (%y, %y)
             {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [32 : i32]}}
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %v : !fabric.bits<32>
      }
    }
  )mlir"
                                              : R"mlir(
    fabric.pe [spatial] (%pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
      %r = fabric.fu(%y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %v = fabric.op [@arith.addi] (%y, %y)
             {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [32 : i32]}}
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %v : !fabric.bits<32>
      }
    }
  )mlir";

  std::string result;
  llvm::raw_string_ostream os(result);
  os << "module { fabric.module @" << rootName
     << "(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {\n";
  os << (reverse ? second : first) << (reverse ? first : second);
  os << "fabric.yield\n} }\n";
  return os.str();
}

void equivalentHardwareHasOneCanonicalRelation() {
  const LabeledModule forward =
      label(__func__, twoPeModule("left", false, false));
  const LabeledModule reverse =
      label(__func__, twoPeModule("right", true, false));
  require(__func__,
          forward.labeling.relationBytes.bytes().equals(
              reverse.labeling.relationBytes.bytes()),
          "source names or sibling construction order changed canonical bytes");
}

void semanticDifferenceChangesCanonicalRelation() {
  const LabeledModule adds = label(__func__, twoPeModule("same", false, false));
  const LabeledModule multiply =
      label(__func__, twoPeModule("same", false, true));
  require(__func__,
          !adds.labeling.relationBytes.bytes().equals(
              multiply.labeling.relationBytes.bytes()),
          "a concrete operation capability change preserved canonical bytes");
}

void identicalFuDefinitionsShareOneTemplate() {
  LabeledModule result = label(__func__, twoPeModule("root", false, false));
  std::size_t templates = 0;
  std::size_t occurrences = 0;
  std::optional<std::uint64_t> sharedTemplate;
  for (const auto &carrier : result.labeling.carriers) {
    if (carrier.kind == FabricEntityKind::FabricFuTemplate)
      ++templates;
    if (carrier.kind != FabricEntityKind::FabricFuOccurrence)
      continue;
    ++occurrences;
    auto found = result.labeling.fuTemplateIdByOccurrence.find(carrier.op);
    require(__func__, found != result.labeling.fuTemplateIdByOccurrence.end(),
            "FU occurrence has no canonical definition relation");
    if (!sharedTemplate)
      sharedTemplate = found->second;
    else
      require(__func__, *sharedTemplate == found->second,
              "isomorphic FU definitions received different template IDs");
  }
  require(__func__, templates == 1 && occurrences == 2,
          "FU template deduplication changed the entity inventory");

  for (const auto &carrier : result.labeling.carriers)
    if (carrier.op)
      carrier.op->setAttr(
          ::fabric::kEntityIdAttrName,
          ::fabric::EntityIdAttr::get(carrier.op->getContext(), 99));
  if (llvm::Error error =
          loom::fabric::detail::materializeFabricCanonicalIds(result.labeling))
    fail(__func__, llvm::toString(std::move(error)));

  for (const auto &carrier : result.labeling.carriers) {
    if (!carrier.op)
      continue;
    auto stored = carrier.op->getAttrOfType<::fabric::EntityIdAttr>(
        ::fabric::kEntityIdAttrName);
    require(__func__, stored && stored.getId() == carrier.id,
            "materialization did not replace an authored entity ID");
    if (carrier.kind != FabricEntityKind::FabricFuOccurrence)
      continue;
    auto templateId = carrier.op->getAttrOfType<::fabric::EntityIdAttr>(
        ::fabric::kFuTemplateIdAttrName);
    require(__func__, templateId && templateId.getId() == *sharedTemplate,
            "FU occurrence did not materialize its definition relation");
  }
}

void materializedIdsSurviveTextRoundTrip() {
  static constexpr llvm::StringLiteral source = R"mlir(
module {
  fabric.module @root(%data: !fabric.bits<32>, %tag: !fabric.bits<4>)
      -> !fabric.bits_tag<32, 4> {
    %fifo = fabric.fifo %data [max_depth = 2, bypassable = false]
        : !fabric.bits<32>
    %routed = fabric.switch [spatial] %fifo
        [{connectivity_table = ["1"]}]
        : (!fabric.bits<32>) -> !fabric.bits<32>
    %tagged = fabric.boundary [s2t] %routed, %tag
        : (!fabric.bits<32>, !fabric.bits<4>)
       -> !fabric.bits_tag<32, 4>
    fabric.yield %tagged : !fabric.bits_tag<32, 4>
  }
}
)mlir";

  LabeledModule original = label(__func__, source);
  if (llvm::Error error = loom::fabric::detail::materializeFabricCanonicalIds(
          original.labeling))
    fail(__func__, llvm::toString(std::move(error)));

  std::string text;
  llvm::raw_string_ostream stream(text);
  original.module->print(stream);
  stream.flush();
  mlir::OwningOpRef<mlir::ModuleOp> reparsed = parse(__func__, text);

  std::size_t expectedCarriers = 0;
  std::size_t retainedCarriers = 0;
  original.module->walk([&](mlir::Operation *op) {
    if (op->getAttr(::fabric::kEntityIdAttrName))
      ++expectedCarriers;
  });
  reparsed->walk([&](mlir::Operation *op) {
    if (op->getAttrOfType<::fabric::EntityIdAttr>(::fabric::kEntityIdAttrName))
      ++retainedCarriers;
  });
  require(__func__, retainedCarriers == expectedCarriers,
          "Fabric textual round-trip dropped materialized entity IDs");
}

std::string symmetricDomainModule(bool reverse) {
  const llvm::StringLiteral first = R"mlir(
    fabric.pe [spatial] (%pa = %a : !fabric.bits<32>) -> !fabric.bits<32> {
      %r = fabric.fu(%x = %pa : !fabric.bits<32>) -> !fabric.bits<32> {
        %v = fabric.op [@arith.addi] (%x, %x)
             {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [32 : i32]}}
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %v : !fabric.bits<32>
      }
    }
  )mlir";
  const llvm::StringLiteral second = R"mlir(
    fabric.pe [spatial] (%pb = %b : !fabric.bits<32>) -> !fabric.bits<32> {
      %s = fabric.fu(%y = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
        %w = fabric.op [@arith.addi] (%y, %y)
             {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [32 : i32]}}
             : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
        fabric.yield %w : !fabric.bits<32>
      }
    }
  )mlir";

  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << "module { fabric.module @root(%a: !fabric.bits<32>, "
            "%b: !fabric.bits<32>) {\n";
  stream << (reverse ? second : first) << (reverse ? first : second);
  stream << "fabric.yield\n} }\n";
  return stream.str();
}

std::vector<std::vector<std::uint8_t>> domainAssignmentBytes(bool reverse) {
  const llvm::StringRef test = "domainAssignmentBytes";
  mlir::OwningOpRef<mlir::ModuleOp> module =
      parse(test, symmetricDomainModule(reverse));
  ::fabric::ModuleOp root;
  llvm::SmallVector<::fabric::PeOp, 2> pes;
  module->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  root.walk([&](::fabric::PeOp pe) { pes.push_back(pe); });
  require(test, pes.size() == 2, "fixture has the wrong PE inventory");
  for (::fabric::PeOp pe : pes)
    pe->setOperand(0, root.getBody().front().getArgument(0));

  ::fabric::ModuleDomainAuthoringRelation relation;
  auto firstClock =
      relation.declareSlot(loom::fabric::FabricClockResetKind::Clock);
  auto secondClock =
      relation.declareSlot(loom::fabric::FabricClockResetKind::Clock);
  auto reset = relation.declareSlot(loom::fabric::FabricClockResetKind::Reset);
  if (!firstClock || !secondClock || !reset)
    fail(test, "unable to declare fixture domain slots");
  using Role = ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;
  mlir::Operation *logicalFirst = pes[reverse ? 1 : 0].getOperation();
  mlir::Operation *logicalSecond = pes[reverse ? 0 : 1].getOperation();
  for (mlir::Operation *pe : {logicalFirst, logicalSecond})
    if (llvm::Error error =
            relation.noteInternalMember(pe, Role::Occurrence, 0))
      fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = relation.assignBoundary(
          loom::fabric::FabricPortDirection::Input, 0,
          loom::fabric::FabricClockResetKind::Clock, *firstClock))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = relation.assignBoundary(
          loom::fabric::FabricPortDirection::Input, 0,
          loom::fabric::FabricClockResetKind::Reset, *reset))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = relation.assignBoundary(
          loom::fabric::FabricPortDirection::Input, 1,
          loom::fabric::FabricClockResetKind::Clock, *firstClock))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error = relation.assignBoundary(
          loom::fabric::FabricPortDirection::Input, 1,
          loom::fabric::FabricClockResetKind::Reset, *reset))
    fail(test, llvm::toString(std::move(error)));
  for (auto [pe, clock] : {std::pair(logicalFirst, *firstClock),
                           std::pair(logicalSecond, *secondClock)}) {
    if (llvm::Error error = relation.assignInternal(
            pe, Role::Occurrence, 0, loom::fabric::FabricClockResetKind::Clock,
            clock))
      fail(test, llvm::toString(std::move(error)));
    if (llvm::Error error = relation.assignInternal(
            pe, Role::Occurrence, 0, loom::fabric::FabricClockResetKind::Reset,
            *reset))
      fail(test, llvm::toString(std::move(error)));
  }

  auto preliminary =
      loom::fabric::detail::computeFabricModuleCanonicalLabeling(root);
  if (!preliminary)
    fail(test, llvm::toString(preliminary.takeError()));
  auto normalized =
      loom::fabric::detail::normalizeFabricModuleDomain(root, relation);
  if (!normalized)
    fail(test, llvm::toString(normalized.takeError()));

  auto canonical = loom::fabric::detail::computeFabricModuleCanonicalLabeling(
      root, *normalized);
  if (!canonical)
    fail(test, llvm::toString(canonical.takeError()));
  require(test,
          !canonical->relationBytes.bytes().equals(
              preliminary->relationBytes.bytes()),
          "Module domain relation is absent from canonical labeling");
  if (llvm::Error error =
          loom::fabric::detail::materializeFabricCanonicalIds(*canonical))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          loom::fabric::detail::materializeFabricModuleDomainRelation(
              root, *normalized, *canonical))
    fail(test, llvm::toString(std::move(error)));

  auto assignments =
      ::fabric::decodeModuleDomainAssignments(root.getDomainAssignmentsAttr());
  if (!assignments)
    fail(test, llvm::toString(assignments.takeError()));
  std::vector<std::vector<std::uint8_t>> bytes;
  bytes.reserve(assignments->size());
  for (const auto &assignment : *assignments)
    bytes.push_back(loom::fabric::canonicalFabricBytes(assignment));
  return bytes;
}

void moduleDomainsParticipateInCanonicalLabeling() {
  const auto forward = domainAssignmentBytes(false);
  const auto reverse = domainAssignmentBytes(true);
  require(__func__, forward == reverse,
          "domain-distinguished symmetric owners depend on authoring order");
}

std::vector<std::vector<std::uint8_t>>
fuNodeDomainAssignmentBytes(bool reverseNodes, bool reverseAssignments) {
  const llvm::StringRef test = "fuNodeDomainAssignmentBytes";
  std::string source;
  llvm::raw_string_ostream stream(source);
  stream << R"mlir(
    module {
      fabric.module @root(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
        %pe = fabric.pe [spatial] (%p = %a : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu(%x = %p : !fabric.bits<32>)
              -> !fabric.bits<32> {
)mlir";
  if (reverseNodes)
    stream << "            %right = fabric.mux %x, %x : "
              "!fabric.bits<32>\n"
              "            %left = fabric.mux %x, %x : "
              "!fabric.bits<32>\n";
  else
    stream << "            %left = fabric.mux %x, %x : "
              "!fabric.bits<32>\n"
              "            %right = fabric.mux %x, %x : "
              "!fabric.bits<32>\n";
  stream << R"mlir(            %out = fabric.op [@arith.muli] (%x, %x)
                {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [32 : i32]}}
                : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %out : !fabric.bits<32>
          }
        }
        fabric.yield
      }
    }
  )mlir";
  auto module = parse(test, source);
  ::fabric::ModuleOp root;
  llvm::SmallVector<::fabric::MuxOp, 2> nodes;
  module->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  root.walk([&](::fabric::MuxOp node) { nodes.push_back(node); });
  require(test, nodes.size() == 2, "fixture has the wrong FU-node inventory");

  using Kind = loom::fabric::FabricClockResetKind;
  using Role = ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;
  ::fabric::ModuleDomainAuthoringRelation relation;
  auto firstClock = relation.declareSlot(Kind::Clock);
  auto secondClock = relation.declareSlot(Kind::Clock);
  auto reset = relation.declareSlot(Kind::Reset);
  if (!firstClock || !secondClock || !reset)
    fail(test, "unable to declare fixture domain slots");
  for (auto [ordinal, clock] :
       {std::pair(0U, *firstClock), std::pair(1U, *secondClock)}) {
    for (Kind kind : {Kind::Clock, Kind::Reset})
      if (llvm::Error error = relation.assignBoundary(
              loom::fabric::FabricPortDirection::Input, ordinal, kind,
              kind == Kind::Clock ? clock : *reset))
        fail(test, llvm::toString(std::move(error)));
  }
  llvm::SmallVector<unsigned, 2> authoringOrder = {0, 1};
  if (reverseAssignments)
    std::reverse(authoringOrder.begin(), authoringOrder.end());
  for (unsigned ordinal : authoringOrder) {
    mlir::Operation *node = nodes[ordinal].getOperation();
    const unsigned semanticOrdinal = reverseNodes ? 1 - ordinal : ordinal;
    if (llvm::Error error = relation.noteInternalMember(node, Role::FuNode, 0))
      fail(test, llvm::toString(std::move(error)));
    for (Kind kind : {Kind::Clock, Kind::Reset})
      if (llvm::Error error = relation.assignInternal(
              node, Role::FuNode, 0, kind,
              kind == Kind::Clock
                  ? (semanticOrdinal == 0 ? *firstClock : *secondClock)
                  : *reset))
        fail(test, llvm::toString(std::move(error)));
  }

  auto normalized =
      loom::fabric::detail::normalizeFabricModuleDomain(root, relation);
  if (!normalized)
    fail(test, llvm::toString(normalized.takeError()));
  auto canonical = loom::fabric::detail::computeFabricModuleCanonicalLabeling(
      root, *normalized);
  if (!canonical)
    fail(test, llvm::toString(canonical.takeError()));
  if (llvm::Error error =
          loom::fabric::detail::materializeFabricCanonicalIds(*canonical))
    fail(test, llvm::toString(std::move(error)));
  if (llvm::Error error =
          loom::fabric::detail::materializeFabricModuleDomainRelation(
              root, *normalized, *canonical))
    fail(test, llvm::toString(std::move(error)));
  auto assignments =
      ::fabric::decodeModuleDomainAssignments(root.getDomainAssignmentsAttr());
  if (!assignments)
    fail(test, llvm::toString(assignments.takeError()));
  std::vector<std::vector<std::uint8_t>> bytes;
  for (const auto &assignment : *assignments)
    bytes.push_back(loom::fabric::canonicalFabricBytes(assignment));
  return bytes;
}

void fuNodeDomainsUseModuleCanonicalContext() {
  const auto baseline = fuNodeDomainAssignmentBytes(false, false);
  require(__func__, baseline == fuNodeDomainAssignmentBytes(false, true),
          "domain-distinguished FU nodes depend on assignment order");
  require(__func__, baseline == fuNodeDomainAssignmentBytes(true, false),
          "domain-distinguished FU nodes depend on physical order");
  require(__func__, baseline == fuNodeDomainAssignmentBytes(true, true),
          "domain-distinguished FU nodes depend on physical and assignment "
          "order");
}

void moduleDomainsRefineOnlyFuAutomorphisms() {
  const llvm::StringRef test = __func__;
  auto module = parse(test, R"mlir(
    module {
      fabric.module @root(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
        %first_pe:2 = fabric.pe [spatial] (%p = %a : !fabric.bits<32>)
            -> (!fabric.bits<32>, !fabric.bits<32>) {
          %first_fu:2 = fabric.fu(%x = %p : !fabric.bits<32>)
              -> (!fabric.bits<32>, !fabric.bits<32>) {
            %first_mux = fabric.mux %x, %x : !fabric.bits<32>
            %second_mux = fabric.mux %x, %x : !fabric.bits<32>
            %first_out = fabric.op [@arith.muli] (%first_mux, %first_mux)
                {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [32 : i32]}}
                : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            %second_out = fabric.op [@arith.muli] (%second_mux, %second_mux)
                {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [32 : i32]}}
                : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %first_out, %second_out
                : !fabric.bits<32>, !fabric.bits<32>
          }
        }
        %second_pe:2 = fabric.pe [spatial] (%p = %b : !fabric.bits<32>)
            -> (!fabric.bits<32>, !fabric.bits<32>) {
          %second_fu:2 = fabric.fu(%x = %p : !fabric.bits<32>)
              -> (!fabric.bits<32>, !fabric.bits<32>) {
            %second_mux = fabric.mux %x, %x : !fabric.bits<32>
            %first_mux = fabric.mux %x, %x : !fabric.bits<32>
            %second_out = fabric.op [@arith.muli] (%second_mux, %second_mux)
                {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [32 : i32]}}
                : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            %first_out = fabric.op [@arith.muli] (%first_mux, %first_mux)
                {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [32 : i32]}}
                : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %first_out, %second_out
                : !fabric.bits<32>, !fabric.bits<32>
          }
        }
        fabric.yield
      }
    }
  )mlir");
  ::fabric::ModuleOp root;
  llvm::SmallVector<::fabric::FuOp, 2> fus;
  module->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  root.walk([&](::fabric::FuOp fu) { fus.push_back(fu); });
  require(test, fus.size() == 2, "fixture has the wrong FU inventory");

  auto capability =
      ::fabric::FuCapabilityDomainRecord::create({{{2, 3}, {{0, 0}, {1, 0}}}});
  if (!capability)
    fail(test, llvm::toString(capability.takeError()));
  auto capabilityBytes = ::fabric::encodeFuCapabilityDomainRecord(*capability);
  if (!capabilityBytes)
    fail(test, llvm::toString(capabilityBytes.takeError()));
  std::vector<std::int8_t> signedCapabilityBytes;
  signedCapabilityBytes.reserve(capabilityBytes->size());
  for (std::uint8_t byte : *capabilityBytes)
    signedCapabilityBytes.push_back(static_cast<std::int8_t>(byte));
  for (::fabric::FuOp fu : fus)
    fu.setCapabilityTemplatesAttr(::fabric::FuCapabilityDomainAttr::get(
        fu.getContext(),
        mlir::DenseI8ArrayAttr::get(fu.getContext(), signedCapabilityBytes)));

  using Kind = loom::fabric::FabricClockResetKind;
  using Role = ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;
  ::fabric::ModuleDomainAuthoringRelation relation;
  auto firstClock = relation.declareSlot(Kind::Clock);
  auto secondClock = relation.declareSlot(Kind::Clock);
  auto firstReset = relation.declareSlot(Kind::Reset);
  auto secondReset = relation.declareSlot(Kind::Reset);
  if (!firstClock || !secondClock || !firstReset || !secondReset)
    fail(test, "unable to declare fixture domain slots");
  for (loom::fabric::FabricOrdinal ordinal = 0; ordinal != 2; ++ordinal)
    for (Kind kind : {Kind::Clock, Kind::Reset}) {
      const bool secondDomain = ordinal != 0;
      const loom::fabric::FabricOrdinal slot =
          kind == Kind::Clock ? (secondDomain ? *secondClock : *firstClock)
                              : (secondDomain ? *secondReset : *firstReset);
      if (llvm::Error error = relation.assignBoundary(
              loom::fabric::FabricPortDirection::Input, ordinal, kind, slot))
        fail(test, llvm::toString(std::move(error)));
    }

  for (::fabric::FuOp fu : fus) {
    llvm::SmallVector<mlir::Operation *, 4> nodes;
    fu->walk([&](mlir::Operation *operation) {
      if (mlir::isa<::fabric::OpOp, ::fabric::MuxOp>(operation))
        nodes.push_back(operation);
    });
    require(test, nodes.size() == 4,
            "fixture FU has the wrong physical-node inventory");
    for (auto [nodeOrdinal, node] : llvm::enumerate(nodes)) {
      if (llvm::Error error =
              relation.noteInternalMember(node, Role::FuNode, 0))
        fail(test, llvm::toString(std::move(error)));
      for (Kind kind : {Kind::Clock, Kind::Reset}) {
        const bool secondDomain = nodeOrdinal % 2 != 0;
        const loom::fabric::FabricOrdinal slot =
            kind == Kind::Clock ? (secondDomain ? *secondClock : *firstClock)
                                : (secondDomain ? *secondReset : *firstReset);
        if (llvm::Error error =
                relation.assignInternal(node, Role::FuNode, 0, kind, slot))
          fail(test, llvm::toString(std::move(error)));
      }
    }
  }

  auto normalized =
      loom::fabric::detail::normalizeFabricModuleDomain(root, relation);
  if (!normalized)
    fail(test, llvm::toString(normalized.takeError()));
  auto canonical = loom::fabric::detail::computeFabricModuleCanonicalLabeling(
      root, *normalized);
  if (!canonical)
    fail(test, llvm::toString(canonical.takeError()));
  require(test, canonical->fuTemplates.size() == 1,
          "occurrence-local Module domains changed FU template identity");

  for (unsigned output = 0; output != 2; ++output) {
    std::optional<loom::fabric::FabricOrdinal> operationOrdinal;
    std::optional<loom::fabric::FabricOrdinal> muxOrdinal;
    for (::fabric::FuOp fu : fus) {
      auto yield =
          mlir::cast<::fabric::YieldOp>(fu.getBody().front().getTerminator());
      mlir::Operation *operation =
          yield.getOperand(output).getDefiningOp<::fabric::OpOp>();
      mlir::Operation *mux =
          operation->getOperand(0).getDefiningOp<::fabric::MuxOp>();
      const auto currentOperation =
          canonical->definitionFuNodeOrdinalByOperation.lookup(operation);
      const auto currentMux =
          canonical->definitionFuNodeOrdinalByOperation.lookup(mux);
      if (!operationOrdinal) {
        operationOrdinal = currentOperation;
        muxOrdinal = currentMux;
      } else {
        require(test, *operationOrdinal == currentOperation,
                "Module domains changed output operation correspondence");
        require(test, *muxOrdinal == currentMux,
                "Module domains changed output mux correspondence");
      }
    }
  }
}

void fuTemplateIdentityExcludesOccurrenceDomains() {
  const llvm::StringRef test = __func__;
  auto module = parse(test, R"mlir(
    module {
      fabric.module @root(%a: !fabric.bits<32>, %b: !fabric.bits<32>) {
        %first_pe = fabric.pe [spatial] (%p = %a : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %first_fu = fabric.fu(%x = %p : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %left = fabric.mux %x, %x : !fabric.bits<32>
            %right = fabric.mux %x, %x : !fabric.bits<32>
            %out = fabric.op [@arith.muli] (%x, %x)
                {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [32 : i32]}}
                : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %out : !fabric.bits<32>
          }
        }
        %second_pe = fabric.pe [spatial] (%p = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %second_fu = fabric.fu(%x = %p : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %right = fabric.mux %x, %x : !fabric.bits<32>
            %left = fabric.mux %x, %x : !fabric.bits<32>
            %out = fabric.op [@arith.muli] (%x, %x)
                {implementation_family = #fabric.implementation_family<ScalarIntegerMultiply>, hw_params = {integer_widths = [32 : i32]}}
                : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %out : !fabric.bits<32>
          }
        }
        fabric.yield
      }
    }
  )mlir");
  ::fabric::ModuleOp root;
  llvm::SmallVector<::fabric::FuOp, 2> fus;
  module->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  root.walk([&](::fabric::FuOp fu) { fus.push_back(fu); });
  require(test, fus.size() == 2, "fixture has the wrong FU inventory");

  for (auto [ordinal, fu] : llvm::enumerate(fus)) {
    const std::vector<::fabric::FuCapabilityRouteSelection> routes =
        ordinal == 0
            ? std::vector<::fabric::FuCapabilityRouteSelection>{{0, 0}, {1, 1}}
            : std::vector<::fabric::FuCapabilityRouteSelection>{{0, 1}, {1, 0}};
    auto capability =
        ::fabric::FuCapabilityDomainRecord::create({{{2}, routes}});
    if (!capability)
      fail(test, llvm::toString(capability.takeError()));
    auto capabilityBytes =
        ::fabric::encodeFuCapabilityDomainRecord(*capability);
    if (!capabilityBytes)
      fail(test, llvm::toString(capabilityBytes.takeError()));
    std::vector<std::int8_t> signedCapabilityBytes;
    signedCapabilityBytes.reserve(capabilityBytes->size());
    for (std::uint8_t byte : *capabilityBytes)
      signedCapabilityBytes.push_back(static_cast<std::int8_t>(byte));
    fu.setCapabilityTemplatesAttr(::fabric::FuCapabilityDomainAttr::get(
        fu.getContext(),
        mlir::DenseI8ArrayAttr::get(fu.getContext(), signedCapabilityBytes)));
  }

  using Kind = loom::fabric::FabricClockResetKind;
  using Role = ::fabric::ModuleDomainAuthoringRelation::InternalMemberRole;
  ::fabric::ModuleDomainAuthoringRelation relation;
  auto firstClock = relation.declareSlot(Kind::Clock);
  auto secondClock = relation.declareSlot(Kind::Clock);
  auto firstReset = relation.declareSlot(Kind::Reset);
  auto secondReset = relation.declareSlot(Kind::Reset);
  if (!firstClock || !secondClock || !firstReset || !secondReset)
    fail(test, "unable to declare fixture domain slots");
  for (loom::fabric::FabricOrdinal ordinal = 0; ordinal != 2; ++ordinal)
    for (Kind kind : {Kind::Clock, Kind::Reset})
      if (llvm::Error error = relation.assignBoundary(
              loom::fabric::FabricPortDirection::Input, ordinal, kind,
              kind == Kind::Clock ? *firstClock : *firstReset))
        fail(test, llvm::toString(std::move(error)));

  llvm::SmallVector<llvm::SmallVector<mlir::Operation *, 4>, 2>
      nodesByOccurrence;
  for (auto [fuOrdinal, fu] : llvm::enumerate(fus)) {
    llvm::SmallVector<mlir::Operation *, 4> nodes;
    fu->walk([&](mlir::Operation *operation) {
      if (mlir::isa<::fabric::OpOp, ::fabric::MuxOp, ::fabric::DemuxOp>(
              operation))
        nodes.push_back(operation);
    });
    require(test, nodes.size() == 3,
            "fixture FU has the wrong physical-node inventory");
    nodesByOccurrence.push_back(nodes);
    for (auto [nodeOrdinal, node] : llvm::enumerate(nodes)) {
      if (llvm::Error error =
              relation.noteInternalMember(node, Role::FuNode, 0))
        fail(test, llvm::toString(std::move(error)));
      for (Kind kind : {Kind::Clock, Kind::Reset}) {
        const bool secondDomain = nodeOrdinal % 2 != 0;
        const loom::fabric::FabricOrdinal slot =
            kind == Kind::Clock ? (secondDomain ? *secondClock : *firstClock)
                                : (secondDomain ? *secondReset : *firstReset);
        if (llvm::Error error =
                relation.assignInternal(node, Role::FuNode, 0, kind, slot))
          fail(test, llvm::toString(std::move(error)));
      }
    }
  }

  auto normalized =
      loom::fabric::detail::normalizeFabricModuleDomain(root, relation);
  if (!normalized)
    fail(test, llvm::toString(normalized.takeError()));
  auto canonical = loom::fabric::detail::computeFabricModuleCanonicalLabeling(
      root, *normalized);
  if (!canonical)
    fail(test, llvm::toString(canonical.takeError()));
  require(test, canonical->fuTemplates.size() == 1,
          "occurrence-local Module domains changed FU template identity");

  std::optional<std::vector<std::uint8_t>> sharedDomain;
  for (auto [fuOrdinal, fu] : llvm::enumerate(fus)) {
    const auto &nodes = nodesByOccurrence[fuOrdinal];
    std::vector<loom::fabric::FabricOrdinal> ordinals;
    ordinals.reserve(nodes.size());
    for (mlir::Operation *node : nodes) {
      auto found = canonical->definitionFuNodeOrdinalByOperation.find(node);
      require(test,
              found != canonical->definitionFuNodeOrdinalByOperation.end(),
              "FU node has no canonical definition ordinal");
      ordinals.push_back(found->second);
    }
    if (fuOrdinal != 0) {
      require(test,
              ordinals[0] ==
                      canonical->definitionFuNodeOrdinalByOperation.lookup(
                          nodesByOccurrence.front()[1]) &&
                  ordinals[1] ==
                      canonical->definitionFuNodeOrdinalByOperation.lookup(
                          nodesByOccurrence.front()[0]) &&
                  ordinals[2] ==
                      canonical->definitionFuNodeOrdinalByOperation.lookup(
                          nodesByOccurrence.front()[2]),
              "equivalent capability routes changed definition-node "
              "correspondence");
    }

    auto domain = canonical->canonicalFuCapabilityDomainByOccurrence.find(fu);
    require(test,
            domain != canonical->canonicalFuCapabilityDomainByOccurrence.end(),
            "FU occurrence has no canonical capability domain");
    if (!sharedDomain)
      sharedDomain = domain->second;
    else
      require(test, *sharedDomain == domain->second,
              "shared FU template has occurrence-local capability bytes");
    auto decoded = ::fabric::decodeFuCapabilityDomainRecord(domain->second);
    if (!decoded)
      fail(test, llvm::toString(decoded.takeError()));
    require(test, decoded->templates().size() == 1,
            "fixture capability domain changed cardinality");
    const auto &selection = decoded->templates().front();
    require(test,
            selection.activeOperationNodeOrdinals.size() == 1 &&
                selection.activeOperationNodeOrdinals.front() == ordinals[2],
            "operation capability changed definition-node correspondence");
    const std::size_t left = fuOrdinal == 0 ? 0 : 1;
    const std::size_t right = fuOrdinal == 0 ? 1 : 0;
    std::vector<::fabric::FuCapabilityRouteSelection> expectedRoutes = {
        {ordinals[left], 0}, {ordinals[right], 1}};
    llvm::sort(expectedRoutes, [](const auto &left, const auto &right) {
      return left.selectorNodeOrdinal < right.selectorNodeOrdinal;
    });
    require(test, selection.routes == expectedRoutes,
            "selector capability changed definition-node correspondence");
  }

  const auto &definition = canonical->fuTemplates.front();
  for (auto [ordinal, node] : llvm::enumerate(definition.canonicalNodeOrder))
    require(test,
            canonical->definitionFuNodeOrdinalByOperation.lookup(node) ==
                ordinal,
            "template representative uses a non-definition node order");
}

} // namespace

int main() {
  fabricOpIntrinsicUsesPersistentSchemaIdentity();
  yieldDefaultPropertyDoesNotChangeCanonicalIdentity();
  equivalentHardwareHasOneCanonicalRelation();
  semanticDifferenceChangesCanonicalRelation();
  identicalFuDefinitionsShareOneTemplate();
  materializedIdsSurviveTextRoundTrip();
  moduleDomainsParticipateInCanonicalLabeling();
  fuNodeDomainsUseModuleCanonicalContext();
  moduleDomainsRefineOnlyFuAutomorphisms();
  fuTemplateIdentityExcludesOccurrenceDomains();
  llvm::outs() << "fabric canonical labeling ok\n";
  return 0;
}

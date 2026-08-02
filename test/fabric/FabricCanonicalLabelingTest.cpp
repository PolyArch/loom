#include "FabricCanonicalLabeling.h"

#include "Fabric/IR/FabricCanonicalEntity.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"

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

} // namespace

int main() {
  equivalentHardwareHasOneCanonicalRelation();
  semanticDifferenceChangesCanonicalRelation();
  identicalFuDefinitionsShareOneTemplate();
  materializedIdsSurviveTextRoundTrip();
  llvm::outs() << "fabric canonical labeling ok\n";
  return 0;
}

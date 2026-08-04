#include "Hardware/Configuration/ConfigurationABI.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "Dataflow/IR/DataflowDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FabricSemanticConfigFieldRef;
using namespace loom::hardware;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted invalid input");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

mlir::MLIRContext &context() {
  static mlir::MLIRContext *instance = [] {
    mlir::DialectRegistry registry;
    registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                    mlir::arith::ArithDialect, mlir::func::FuncDialect>();
    auto *result =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    result->loadAllAvailableDialects();
    return result;
  }();
  return *instance;
}

loom::fabric::FinalizedFabricRoot makeFabric(llvm::StringRef test,
                                             const ArtifactStore &store) {
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @configured(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
          -> !fabric.bits<32> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<32>, %pb = %b : !fabric.bits<32>)
            -> !fabric.bits<32> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<32>,
               %fb = %pb : !fabric.bits<32>)
              -> !fabric.bits<32> {
            %first = fabric.op [@arith.addi, @arith.subi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            %second = fabric.op [@arith.addi, @arith.subi] (%first, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
            fabric.yield %second : !fabric.bits<32>
          }
        }
        fabric.yield %pe : !fabric.bits<32>
      }
    }
  )mlir",
                                                        &context());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");

  std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  std::vector<std::int8_t> signedContract;
  signedContract.reserve(contract.size());
  for (std::uint8_t byte : contract)
    signedContract.push_back(static_cast<std::int8_t>(byte));
  source->walk([&](::fabric::OpOp operation) {
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       mlir::DenseI8ArrayAttr::get(&context(), signedContract));
  });

  ::fabric::ModuleOp root;
  source->walk([&](::fabric::ModuleOp candidate) { root = candidate; });
  require(test, static_cast<bool>(root), "Fabric fixture has no root");
  return take(test, loom::fabric::finalizeFabricRoot(root, store));
}

ConfigurationFieldEncoding codebookField(FabricSemanticConfigFieldRef field,
                                         std::uint64_t firstBit,
                                         std::uint64_t secondBit) {
  return ConfigurationFieldEncoding{
      std::move(field),
      FiniteCodebookEncoding{2, {{{0x00}, {0x00}}, {{0x01}, {0x03}}}},
      {{0, firstBit, 1}, {1, secondBit, 1}},
      {0x00}};
}

ConfigurationABIDraft
makeDraft(const loom::fabric::FinalizedFabricRoot &fabric) {
  const auto definitions = fabric.view().fuTemplates();
  if (definitions.size() != 1)
    fail(__func__, "fixture does not have one FU definition");
  const auto capabilities =
      fabric.view().resolvedFabricOpCapabilities(definitions.front());
  if (capabilities.size() != 2 ||
      capabilities[0].configurationFieldSchema.size() != 1 ||
      capabilities[1].configurationFieldSchema.size() != 1)
    fail(__func__, "fixture does not have two configuration fields");

  FabricSemanticConfigFieldRef first =
      capabilities[0].configurationFieldSchema.front();
  FabricSemanticConfigFieldRef second =
      capabilities[1].configurationFieldSchema.front();
  ProgrammingUnitDraft unit{
      {first.owner.catalog(), second.owner.catalog()},
      8,
      {codebookField(first, 1, 4), codebookField(second, 3, 6)}};
  return ConfigurationABIDraft{fabric.reference(), {std::move(unit)}};
}

void canonicalArtifactAndBitRoundTrip(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  loom::fabric::FinalizedFabricRoot fabric = makeFabric(test, store);

  ConfigurationABIDraft draft = makeDraft(fabric);
  FinalizedConfigurationABI first =
      take(test, finalizeConfigurationABI(draft, store));
  require(test, first.abi().programmingUnits().size() == 1,
          "ABI did not preserve one programming unit");

  const auto &fields = first.abi().programmingUnits().front().fields;
  std::vector<SemanticConfigurationValue> selected{
      {fields.front().field, {0x01}}};
  std::vector<std::uint8_t> payload =
      take(test, first.abi().encode(0, selected));
  require(test, payload == std::vector<std::uint8_t>{0x12},
          "configuration slices produced the wrong payload");

  auto decoded = take(test, first.abi().decode(0, payload));
  require(test,
          decoded.size() == 2 && decoded[0].value == selected[0].value &&
              decoded[1].value == std::vector<std::uint8_t>{0x00},
          "decode did not recover selected and inactive values");
  require(test, take(test, first.abi().encode(0, decoded)) == payload,
          "decoded image did not re-encode identically");

  ConfigurationABIDraft reordered = makeDraft(fabric);
  auto &unit = reordered.programmingUnits.front();
  std::reverse(unit.exactFabricResourceClosure.begin(),
               unit.exactFabricResourceClosure.end());
  std::reverse(unit.fields.begin(), unit.fields.end());
  for (ConfigurationFieldEncoding &field : unit.fields) {
    std::reverse(field.destinationSlices.begin(),
                 field.destinationSlices.end());
    auto &codebook = std::get<FiniteCodebookEncoding>(field.semanticEncoding);
    std::reverse(codebook.entries.begin(), codebook.entries.end());
  }
  FinalizedConfigurationABI second =
      take(test, finalizeConfigurationABI(std::move(reordered), store));
  require(test,
          first.reference() == second.reference() &&
              first.canonicalBytes().bytes() == second.canonicalBytes().bytes(),
          "authoring order changed ConfigurationABI identity");

  FinalizedConfigurationABI imported =
      take(test, importConfigurationABI(first.reference(), store));
  require(test,
          imported.reference() == first.reference() &&
              imported.canonicalBytes().bytes() ==
                  first.canonicalBytes().bytes(),
          "strict import changed the canonical ABI");

  ConfigurationABIDraft changed = makeDraft(fabric);
  std::get<FiniteCodebookEncoding>(
      changed.programmingUnits.front().fields.front().semanticEncoding)
      .entries.back()
      .physicalCode = {0x02};
  FinalizedConfigurationABI changedAbi =
      take(test, finalizeConfigurationABI(std::move(changed), store));
  require(test, changedAbi.reference() != first.reference(),
          "semantic codebook change did not change ABI identity");
}

void invalidImagesAndLayoutsAreRejected(const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  loom::fabric::FinalizedFabricRoot fabric = makeFabric(test, store);
  FinalizedConfigurationABI abi =
      take(test, finalizeConfigurationABI(makeDraft(fabric), store));

  expectError(test, abi.abi().decode(0, {0x01}), "reserved bit");
  expectError(test, abi.abi().decode(0, {0x02}), "codebook");
  expectError(test, abi.abi().decode(1, {0x00}), "programming unit");

  ConfigurationABIDraft overlap = makeDraft(fabric);
  overlap.programmingUnits.front()
      .fields.back()
      .destinationSlices.front()
      .destinationBitOffset = 1;
  expectError(test, finalizeConfigurationABI(std::move(overlap), store),
              "destination bit");

  ConfigurationABIDraft incomplete = makeDraft(fabric);
  incomplete.programmingUnits.front()
      .fields.front()
      .destinationSlices.pop_back();
  expectError(test, finalizeConfigurationABI(std::move(incomplete), store),
              "source bit");

  ConfigurationABIDraft missingOwner = makeDraft(fabric);
  missingOwner.programmingUnits.front().exactFabricResourceClosure.pop_back();
  expectError(test, finalizeConfigurationABI(std::move(missingOwner), store),
              "resource closure");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one test-directory argument");
  const std::filesystem::path root =
      std::filesystem::absolute(argv[1]).lexically_normal();
  std::filesystem::create_directories(root);
  canonicalArtifactAndBitRoundTrip(root / "canonical");
  invalidImagesAndLayoutsAreRejected(root / "invalid");
  return 0;
}

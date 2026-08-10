#include "Fabric/Identity/FabricMemoryConfiguration.h"
#include "ADG/Builder.h"
#include "ADG/MemoryLibrary.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

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
void expectRejected(llvm::StringRef test, llvm::Expected<T> value,
                    llvm::StringRef expected) {
  if (value)
    fail(test, "accepted noncanonical memory configuration");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

::fabric::UnsignedDomain singletonUnsigned(std::uint64_t value) {
  return take(__func__,
              ::fabric::UnsignedDomain::fromCanonical({{value, value}}));
}

loom::adg::MemorySpec
makeMemory(llvm::StringRef test,
           std::optional<loom::adg::TemporalMemoryParameters> temporal) {
  loom::adg::LocalMemoryParameters parameters;
  parameters.capacityBytes = 4096;
  parameters.interface = {loom::adg::MemoryAccessDomainParameters{
                              128, 128, 16, singletonUnsigned(64)},
                          128, 128};
  parameters.temporal = temporal;
  return take(test, loom::adg::makeGeneral64LocalMemory(parameters));
}

void addMemoryRoot(llvm::StringRef test, loom::adg::DesignBuilder &design,
                   llvm::StringRef name, loom::adg::MemorySpec memory) {
  auto spatial = take(test, design.createSpatialCore(name, memory.inputTypes(),
                                                     memory.outputTypes()));
  std::vector<loom::adg::SpatialValue> inputs;
  inputs.reserve(memory.inputTypes().size());
  for (std::size_t ordinal = 0; ordinal < memory.inputTypes().size(); ++ordinal)
    inputs.push_back(take(test, spatial.input(ordinal)));
  auto outputs = take(test, spatial.addMemory(inputs, memory));
  if (llvm::Error error = spatial.close(outputs.values()))
    fail(test, llvm::toString(std::move(error)));
}

void checkSchema(llvm::StringRef test,
                 const loom::fabric::FinalizedFabricRoot &root,
                 ::fabric::Schedule expectedSchedule,
                 std::uint32_t expectedRows, mlir::MLIRContext &context) {
  const auto &view = root.view();
  require(test, view.memoryOccurrences().size() == 1,
          "fixture did not finalize exactly one memory occurrence");
  const auto memory = view.memoryOccurrences().front();
  auto schema = take(test, view.memoryConfigurationSchema(memory));
  const auto &layout = schema.layout();
  require(test, layout.schedule == expectedSchedule,
          "memory configuration lost its schedule");
  require(test,
          layout.operationRowCount == expectedRows &&
              layout.operationRows.size() == expectedRows,
          "memory configuration has the wrong operation-row shape");
  require(test, layout.carrierBitCount > 1,
          "memory configuration carrier is unexpectedly empty");
  require(test,
          view.inventorySize(
              loom::fabric::FabricInventoryOwnerRef::of(memory),
              loom::fabric::FabricInventoryKind::SemanticConfigField) == 1,
          "memory occurrence does not own exactly one configuration field");

  auto relation =
      take(test, view.semanticFieldRelation(schema.field(), context));
  require(test,
          relation.kind() ==
                  loom::fabric::FabricSemanticFieldRelationKind::Direct &&
              relation.directEncodedBitCount() == layout.carrierBitCount,
          "semantic field relation disagrees with the memory carrier");

  const auto encoded =
      take(test, schema.encode(loom::fabric::FabricMemoryConfigurationValue{
                     loom::fabric::FabricMemoryDisabled{}}));
  require(test, encoded.bytes().size() == (layout.carrierBitCount + 7) / 8,
          "memory carrier byte width is not the exact bit-width ceiling");
  require(test,
          std::holds_alternative<loom::fabric::FabricMemoryDisabled>(
              take(test, schema.decode(encoded.bytes()))),
          "Disabled memory configuration did not round-trip");
  if (llvm::Error error = relation.validateSemanticValue(encoded.bytes()))
    fail(test, llvm::toString(std::move(error)));

  std::vector<std::uint8_t> dormant(encoded.bytes().begin(),
                                    encoded.bytes().end());
  dormant.front() |= 0x2;
  expectRejected(test, schema.decode(dormant),
                 "Disabled memory carrier has nonzero payload");

  std::vector<std::uint8_t> emptyActive(encoded.bytes().begin(),
                                        encoded.bytes().end());
  emptyActive.front() |= 0x1;
  expectRejected(test, schema.decode(emptyActive),
                 "empty Active memory configuration");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one temporary directory argument");
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  loom::ArtifactStore artifacts(root.string());
  loom::adg::DesignBuilder design(artifacts);
  addMemoryRoot("main", design, "spatial-memory",
                makeMemory("main", std::nullopt));
  addMemoryRoot("main", design, "temporal-memory",
                makeMemory("main", loom::adg::TemporalMemoryParameters{4, 4}));
  auto finalized = take("main", std::move(design).finalize());
  require("main", finalized.roots().size() == 2,
          "fixture did not publish both memory schedules");

  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  bool sawSpatial = false;
  bool sawTemporal = false;
  for (const auto &module : finalized.roots()) {
    const auto memory = module.view().memoryOccurrences().front();
    const auto schedule = module.view().memorySchedule(memory);
    if (schedule == ::fabric::Schedule::Spatial) {
      checkSchema("spatial", module, ::fabric::Schedule::Spatial, 2, context);
      sawSpatial = true;
    } else if (schedule == ::fabric::Schedule::Temporal) {
      checkSchema("temporal", module, ::fabric::Schedule::Temporal, 4, context);
      sawTemporal = true;
    } else {
      fail("main", "fixture memory has no operation schedule");
    }
  }
  require("main", sawSpatial && sawTemporal,
          "fixture lost one memory schedule");
  return EXIT_SUCCESS;
}

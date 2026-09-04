// Anchors the occurrence-free block closure identity of
// docs/spec-rtl-lowering.md on two authored Fabrics that share one
// structural temporal switch:
//  - the paired Fabric instantiates that switch twice next to a switch with
//    wider ports, the solo Fabric instantiates it once, and the generators
//    name the shared definition differently in the two implementations;
//  - the shared switch derives one content identity and byte-equal block
//    sources across both implementations, the wide switch another identity;
//  - the paired top closure counts both instances of the shared switch,
//    emits one normalized shared definition and references it twice;
//  - the source binding fails closed on a payload that is not the exact
//    framed source.
#include "Hardware/RTL/RtlBlockClosure.h"
#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "ConfigurationABITestSupport.h"
#include "DSE/RtlBlockSourceCandidateGenerator.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricSemanticFieldRelation.h"
#include "Hardware/RTL/CommonSkeleton.h"
#include "Hardware/RTL/RtlBlockSource.h"
#include "Hardware/RTL/RtlModuleGraph.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using loom::ArtifactStore;
using loom::fabric::FinalizedFabricRoot;
using loom::hardware::FinalizedConfigurationABI;
using namespace loom::hardware::rtl;

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

enum class Shape { PairedWithWide, Solo };

constexpr std::uint32_t narrowWidth = 8;
constexpr std::uint32_t wideWidth = 16;

/// Parallel 2 x 2 temporal switches with two-entry route tables and a
/// round-robin grant: two narrow switches and one wide switch, or one narrow
/// switch alone.
FinalizedFabricRoot makeFixtureModule(llvm::StringRef test,
                                      const ArtifactStore &store, Shape shape) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType narrow = take(test, PortType::taggedBits(narrowWidth, 2));
  const PortType wide = take(test, PortType::taggedBits(wideWidth, 2));
  std::vector<PortType> ports;
  for (const PortType &type :
       shape == Shape::PairedWithWide
           ? std::vector<PortType>{narrow, narrow, narrow, narrow, wide, wide}
           : std::vector<PortType>{narrow, narrow})
    ports.push_back(type);
  auto spatial =
      take(test, design.createSpatialCore(shape == Shape::PairedWithWide
                                              ? "rtl-block-closure-paired"
                                              : "rtl-block-closure-solo",
                                          ports, ports));
  std::vector<SpatialValue> outputs;
  for (std::size_t pair = 0; pair != ports.size() / 2; ++pair) {
    const PortType &type = ports[2 * pair];
    auto routed = take(
        test,
        spatial.addSwitch({take(test, spatial.input(2 * pair)),
                           take(test, spatial.input(2 * pair + 1))},
                          SwitchSpec::temporal(
                              {type, type}, {type, type}, {{0, 1}, {0, 1}}, 2,
                              ::fabric::TemporalSwitchRoundRobin{{0, 1}, 0})));
    for (const SpatialValue &value : routed.values())
      outputs.push_back(value);
  }
  if (llvm::Error error = spatial.close(outputs))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "closure fixture did not finalize one Module");
  return std::move(finalized.roots().front());
}

struct Fixture final {
  std::string systemVerilog;
  RtlModuleGraphProjection graph;
  std::size_t narrowSwitch = 0;
  std::optional<std::size_t> wideSwitch;
  std::optional<FinalizedRtlBlockSource> sourceArtifact;
};

/// The switch definitions of one graph, recognized by their data port width
/// because the System qualifies its own occurrence identities.
std::vector<std::size_t>
switchDefinitions(const RtlModuleGraphProjection &graph, std::uint32_t width) {
  std::vector<std::size_t> found;
  for (std::size_t ordinal = 0; ordinal != graph.modules.size(); ++ordinal) {
    const RtlModuleProjection &definition = graph.modules[ordinal];
    if (!llvm::StringRef(definition.emittedName)
             .starts_with("loom_fabric_switch_"))
      continue;
    for (const RtlModulePortProjection &port : definition.ports)
      if (port.name == "input_0_data" &&
          port.type == "i" + std::to_string(width))
        found.push_back(ordinal);
  }
  return found;
}

Fixture buildFixture(const std::filesystem::path &root, Shape shape) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(root);
  ArtifactStore store(root.string());
  FinalizedFabricRoot module = makeFixtureModule(test, store, shape);
  FinalizedFabricRoot system = take(
      test, loom::hardware::test::makeSingleSpatialCoreSystem(module, store));
  const loom::fabric::SpatialCoreOccurrenceRef spatialCore = take(
      test, loom::hardware::test::requireSingleSpatialCoreOccurrence(system));
  const auto &view = module.view();

  // Switch carriers are Direct fields; the ABI draft needs their bit widths.
  std::vector<loom::hardware::test::ConfigurationFieldEncodingOverride>
      overrides;
  for (const auto sw : view.switchOccurrences()) {
    const loom::fabric::FabricInventoryOwnerRef owner =
        loom::fabric::FabricInventoryOwnerRef::of(sw);
    const std::uint64_t fieldCount = view.inventorySize(
        owner, loom::fabric::FabricInventoryKind::SemanticConfigField);
    for (std::uint64_t ordinal = 0; ordinal < fieldCount; ++ordinal) {
      const loom::fabric::FabricSemanticConfigFieldRef field{
          loom::fabric::FabricConfigurationOwnerRef(owner), ordinal};
      auto relation =
          take(test,
               view.semanticFieldRelation(field, *const_cast<mlir::Operation *>(
                                                      view.canonicalOperation())
                                                      ->getContext()));
      if (relation.kind() !=
          loom::fabric::FabricSemanticFieldRelationKind::Direct)
        continue;
      const std::uint64_t bitCount = *relation.directEncodedBitCount();
      auto target = take(
          test, loom::fabric::FabricModulePhysicalTargetRef::create(field));
      auto qualified =
          take(test, loom::fabric::FabricPhysicalConfigurationFieldRef::create(
                         loom::fabric::SpatialCoreInternalOccurrenceRef{
                             spatialCore, std::move(target)}));
      overrides.push_back({std::move(qualified),
                           loom::hardware::DirectBitsEncoding{bitCount},
                           std::vector<std::uint8_t>((bitCount + 7) / 8, 0)});
    }
  }
  FinalizedConfigurationABI abi = take(
      test,
      loom::hardware::finalizeConfigurationABI(
          take(test, loom::hardware::test::makeCompleteConfigurationABIDraft(
                         system, overrides)),
          store));

  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  auto skeleton =
      take(test, buildModuleRootCirctSkeleton(context, spatialCore, abi));
  require(test, skeleton.operationLeaves.empty(),
          "closure fixture unexpectedly owns operation leaves");
  Fixture fixture;
  llvm::raw_string_ostream output(fixture.systemVerilog);
  if (llvm::Error error = lowerAndExportSpecializedSystemVerilog(
          *skeleton.module, output, {},
          RtlModuleGraphCapture{"loom_module", &fixture.graph}))
    fail(test, llvm::toString(std::move(error)));
  output.flush();
  const std::vector<std::size_t> narrow =
      switchDefinitions(fixture.graph, narrowWidth);
  const std::vector<std::size_t> wide =
      switchDefinitions(fixture.graph, wideWidth);
  // The generator already shares one definition between the two narrow
  // switch occurrences of the paired Fabric.
  require(test,
          narrow.size() == 1 &&
              wide.size() == (shape == Shape::PairedWithWide ? 1 : 0),
          "module graph does not carry the expected switch definitions");
  fixture.narrowSwitch = narrow.front();
  if (!wide.empty())
    fixture.wideSwitch = wide.front();
  std::filesystem::create_directories(root / "blobs");
  loom::BlobStore blobs((root / "blobs").string());
  auto implementation =
      take(test, finalizePortableSpatialCoreHardwareImplementation(
                     abi, spatialCore, std::nullopt, store, blobs));
  const auto inputs = take(
      test, loom::dse::bindRtlBlockSourceInputs(implementation.reference()));
  const auto binding =
      take(test, loom::dse::resolveRtlBlockSourceBinding(fixture.narrowSwitch));
  const auto generated = take(
      test, loom::dse::invokeCandidateGenerator(inputs, binding, store, blobs));
  const auto *completed =
      std::get_if<loom::dse::CompletedCandidateGeneratorResult>(
          &generated.outcome);
  require(
      test,
      completed && completed->outputBindings.size() == 1 &&
          completed->outputBindings.front().artifacts.size() == 1,
      "registered block source derivation did not complete exactly one output");
  fixture.sourceArtifact = take(
      test,
      importRtlBlockSource(completed->outputBindings.front().artifacts.front(),
                           store, blobs));
  auto imported =
      take(test, importRtlBlockSource(fixture.sourceArtifact->reference(),
                                      store, blobs));
  require(test, imported.reference() == fixture.sourceArtifact->reference(),
          "cold block source import changed its identity");
  if (llvm::Error error = loom::dse::verifyRtlBlockSourceDerivation(
          inputs, binding, imported.reference(), store, blobs))
    fail(test, llvm::toString(std::move(error)));
  if (fixture.wideSwitch) {
    const auto wrongBinding = take(
        test, loom::dse::resolveRtlBlockSourceBinding(*fixture.wideSwitch));
    llvm::Error wrongParent = loom::dse::verifyRtlBlockSourceDerivation(
        inputs, wrongBinding, imported.reference(), store, blobs);
    require(test, static_cast<bool>(wrongParent),
            "source Artifact admitted a different parent block association");
    llvm::consumeError(std::move(wrongParent));
  }

  if (llvm::Error error = loom::writeArtifactRootReferenceJsonFile(
          (root / "block-source-ref.json").string(), imported.reference()))
    fail(test, llvm::toString(std::move(error)));
  return fixture;
}

const RtlDomainPortNames domainPorts{std::string("clock"),
                                     std::string("reset")};

struct LeafClosure final {
  RtlBlockClosure closure;
  std::string source;
};

LeafClosure leafClosure(llvm::StringRef test, const Fixture &fixture,
                        const RtlModuleGraphSourceBinding &source,
                        std::size_t definition) {
  RtlBlockClosure closure =
      take(test, deriveRtlBlockClosure(fixture.graph, source, definition,
                                       domainPorts));
  RtlBlockSourceProjection projection =
      take(test, projectRtlBlockClosureSource(closure, fixture.graph, source));
  const auto rebound =
      take(test, bindRtlModuleGraphSource(projection.graph, projection.source));
  const auto rederived = take(
      test, deriveRtlBlockClosure(projection.graph, rebound,
                                  projection.graph.topModule, domainPorts));
  require(test, rederived.identity() == closure.identity(),
          "normalized graph does not cold-derive the same block identity");
  std::string rendered = std::move(projection.source);
  require(test, closure.members.size() == 1,
          "leaf switch closure has more than one member");
  require(test, closure.members.back().instanceCount == 1,
          "leaf switch closure root does not count one instance");
  require(test, closure.clockPort == "clock" && closure.resetPort == "reset",
          "leaf switch closure did not bind the threaded domain ports");
  require(test,
          rendered.find("module " + rtlBlockName(closure.identity()) + "(") !=
              std::string::npos,
          "block source does not define the block name");
  require(test, rendered.find("loom_fabric_switch_") == std::string::npos,
          "block source still carries an occurrence name");
  return LeafClosure{std::move(closure), std::move(rendered)};
}

void checkLeafIdentity(const Fixture &paired,
                       const RtlModuleGraphSourceBinding &pairedSource,
                       const Fixture &solo,
                       const RtlModuleGraphSourceBinding &soloSource,
                       const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  const LeafClosure pairedNarrow =
      leafClosure(test, paired, pairedSource, paired.narrowSwitch);
  const LeafClosure soloNarrow =
      leafClosure(test, solo, soloSource, solo.narrowSwitch);
  const LeafClosure pairedWide =
      leafClosure(test, paired, pairedSource, *paired.wideSwitch);
  std::ofstream(root / "paired_narrow.sv") << pairedNarrow.source;
  std::ofstream(root / "solo_narrow.sv") << soloNarrow.source;
  require(test,
          paired.graph.modules[paired.narrowSwitch].emittedName !=
              solo.graph.modules[solo.narrowSwitch].emittedName,
          "the two implementations unexpectedly name the shared switch "
          "identically");
  require(test,
          pairedNarrow.closure.identity() == soloNarrow.closure.identity(),
          "two occurrences of one structural switch derived two identities");
  require(test, pairedNarrow.source == soloNarrow.source,
          "two occurrences of one structural switch rendered different bytes");
  require(test,
          paired.sourceArtifact->reference() ==
              solo.sourceArtifact->reference(),
          "distinct source occurrences did not publish one reusable Artifact");
  require(test,
          paired.sourceArtifact->projection().source == pairedNarrow.source &&
              paired.sourceArtifact->closureIdentity() ==
                  pairedNarrow.closure.identity(),
          "source Artifact did not preserve its mechanically derived closure");
  require(test,
          pairedNarrow.closure.identity() != pairedWide.closure.identity(),
          "a switch with different ports derived the same identity");
}

void checkPreambleIdentity(const Fixture &fixture,
                           const RtlModuleGraphSourceBinding &source) {
  const llvm::StringRef test = __func__;
  const auto original =
      leafClosure(test, fixture, source, fixture.narrowSwitch);
  Fixture changed = fixture;
  const std::string prefix = "`default_nettype none\n";
  changed.systemVerilog.insert(0, prefix);
  const auto digestOf = [](llvm::StringRef bytes) {
    return loom::computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
  };
  for (auto &module : changed.graph.modules)
    if (module.emission)
      module.emission->offset += prefix.size();
  const std::size_t preambleSize = prefix.size() + source.preamble().size();
  changed.graph.preamble = RtlModuleEmissionRange{
      0, preambleSize,
      digestOf(
          llvm::StringRef(changed.systemVerilog).take_front(preambleSize))};
  changed.graph.sourceByteCount = changed.systemVerilog.size();
  changed.graph.sourceDigest = digestOf(changed.systemVerilog);
  const auto changedSource = take(
      test, bindRtlModuleGraphSource(changed.graph, changed.systemVerilog));
  const auto derived =
      leafClosure(test, changed, changedSource, changed.narrowSwitch);
  require(test, derived.closure.identity() != original.closure.identity(),
          "changed source preamble reused the previous block identity");
  require(test, llvm::StringRef(derived.source).starts_with(prefix),
          "normalized block dropped its source preamble");
}

void checkTopClosure(const Fixture &fixture,
                     const RtlModuleGraphSourceBinding &source,
                     const std::filesystem::path &root) {
  const llvm::StringRef test = __func__;
  RtlBlockClosure complete =
      take(test, deriveRtlBlockClosure(fixture.graph, source,
                                       fixture.graph.topModule, domainPorts));
  const RtlBlockClosureMember *shared = nullptr;
  for (const RtlBlockClosureMember &member : complete.members)
    if (member.definitions == std::vector<std::size_t>{fixture.narrowSwitch})
      shared = &member;
  require(test, shared != nullptr,
          "top closure does not carry the shared switch member");
  require(test, shared->instanceCount == 2,
          "shared switch member does not count both instances");
  require(test,
          complete.members.back().instanceCount == 1 &&
              complete.clockPort == "clock",
          "top closure root is not the implementation top");

  const std::string sharedName = rtlBlockName(shared->identity);
  const auto projection =
      take(test, projectRtlBlockClosureSource(complete, fixture.graph, source));
  const auto rebound =
      take(test, bindRtlModuleGraphSource(projection.graph, projection.source));
  const auto rederived = take(
      test, deriveRtlBlockClosure(projection.graph, rebound,
                                  projection.graph.topModule, domainPorts));
  const auto reprojected = take(
      test, projectRtlBlockClosureSource(rederived, projection.graph, rebound));
  require(test,
          rederived.identity() == complete.identity() &&
              reprojected.source == projection.source,
          "normalized hierarchy does not cold-derive identical source");
  const std::string &rendered = projection.source;
  std::ofstream(root / "top.sv") << rendered;
  const std::string sharedHeader = "module " + sharedName + "(";
  require(test, rendered.find(sharedHeader) != std::string::npos,
          "rendered top closure has no definition for the shared member");
  require(test, rendered.find(sharedHeader) == rendered.rfind(sharedHeader),
          "rendered top closure defines the shared member more than once");
  std::size_t references = 0;
  for (std::size_t at = rendered.find(sharedName); at != std::string::npos;
       at = rendered.find(sharedName, at + sharedName.size()))
    ++references;
  require(test, references == 3,
          "rendered top closure does not reference the shared member twice");
  require(test,
          rendered.find(rtlBlockName(complete.identity())) != std::string::npos,
          "rendered top closure does not define the root block");
}

} // namespace

int main(int argc, char **argv) {
  require("main", argc == 2, "expected exactly one output directory");
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  const Fixture paired =
      buildFixture(root / "store" / "paired", Shape::PairedWithWide);
  const Fixture solo = buildFixture(root / "store" / "solo", Shape::Solo);

  auto tampered =
      bindRtlModuleGraphSource(paired.graph, paired.systemVerilog + "\n");
  require("main", !tampered, "a tampered payload bound to the module graph");
  llvm::consumeError(tampered.takeError());
  const RtlModuleGraphSourceBinding pairedSource = take(
      "main", bindRtlModuleGraphSource(paired.graph, paired.systemVerilog));
  const RtlModuleGraphSourceBinding soloSource =
      take("main", bindRtlModuleGraphSource(solo.graph, solo.systemVerilog));
  checkLeafIdentity(paired, pairedSource, solo, soloSource, root);
  checkTopClosure(paired, pairedSource, root);
  checkPreambleIdentity(paired, pairedSource);
  return 0;
}

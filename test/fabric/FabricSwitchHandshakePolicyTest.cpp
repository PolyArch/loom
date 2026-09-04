#include "ADG/Builder.h"
#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricHandshake.h"

#include "../TestAllocationProbe.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace {

using loom::fabric::FabricHandshakeSelection;
using loom::fabric::FabricPhysicalTraversalRef;
using loom::fabric::FabricSwitchOccurrenceRef;
using loom::fabric::HandshakeOwnerModel;
using loom::fabric::HandshakeSignalKind;
using loom::fabric::HandshakeSignalRef;
using loom::fabric::ResolvedHandshakeActivation;

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

void requireRejected(llvm::StringRef test, llvm::Error error,
                     llvm::StringRef expected) {
  if (!error)
    fail(test, "invalid handshake selection was accepted");
  const std::string message = llvm::toString(std::move(error));
  if (!llvm::StringRef(message).contains(expected))
    fail(test, "unexpected rejection: " + message);
}

class TemporaryDirectory final {
public:
  explicit TemporaryDirectory(llvm::StringRef test) : test_(test.str()) {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-fabric-switch-handshake-policy-test", path))
      fail(test, error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() {
    if (std::error_code error = llvm::sys::fs::remove_directories(path_))
      llvm::errs() << test_ << ": unable to remove temporary directory: "
                   << error.message() << '\n';
  }

  llvm::StringRef path() const { return path_; }

private:
  std::string test_;
  std::string path_;
};

struct SwitchFixture final {
  FabricSwitchOccurrenceRef occurrence;
  std::map<std::pair<loom::fabric::FabricOrdinal, loom::fabric::FabricOrdinal>,
           FabricPhysicalTraversalRef>
      crosspoints;
  std::map<loom::fabric::FabricOrdinal,
           loom::fabric::FabricTransportEndpointRef>
      inputs;
  std::map<loom::fabric::FabricOrdinal,
           loom::fabric::FabricTransportEndpointRef>
      outputs;

  FabricPhysicalTraversalRef
  crosspoint(llvm::StringRef test, loom::fabric::FabricOrdinal input,
             loom::fabric::FabricOrdinal output) const {
    const auto found = crosspoints.find({input, output});
    if (found == crosspoints.end())
      fail(test, "fixture switch omitted an expected crosspoint");
    return found->second;
  }
};

loom::fabric::FinalizedFabricRoot buildFixture(llvm::StringRef test,
                                               loom::ArtifactStore &store) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType type = take(test, PortType::taggedBits(32, 4));
  auto spatial =
      take(test, design.createSpatialCore("fixed-priority-handshake",
                                          {type, type, type}, {type, type}));
  auto outputs = take(
      test, spatial.addSwitch(
                {take(test, spatial.input(0)), take(test, spatial.input(1)),
                 take(test, spatial.input(2))},
                SwitchSpec::temporal(
                    {type, type, type}, {type, type}, {{0, 1, 2}, {0, 1, 2}}, 3,
                    ::fabric::TemporalSwitchFixedPriority{{1, 0, 2}})));
  if (llvm::Error error = spatial.close(outputs.values()))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "fixture did not finalize one Fabric root");
  return std::move(finalized.roots().front());
}

loom::fabric::FinalizedFabricRoot
buildContentionFixture(llvm::StringRef test, loom::ArtifactStore &store) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType type = take(test, PortType::taggedBits(32, 4));
  auto spatial =
      take(test, design.createSpatialCore("configured-switch-contention",
                                          {type, type, type, type},
                                          {type, type, type, type, type}));
  auto fanout = take(test, spatial.addSwitch({take(test, spatial.input(0))},
                                             SwitchSpec::temporal(
                                                 {type}, {type, type},
                                                 {{0}, {0}}, 2, std::nullopt)));
  auto roundRobin = take(
      test,
      spatial.addSwitch({fanout[0], take(test, spatial.input(1)),
                         take(test, spatial.input(3))},
                        SwitchSpec::temporal(
                            {type, type, type}, {type, type, type},
                            {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}, 3,
                            ::fabric::TemporalSwitchRoundRobin{{0, 1, 2}, 0})));
  auto fixedPriority = take(
      test,
      spatial.addSwitch(
          {fanout[1], take(test, spatial.input(2))},
          SwitchSpec::temporal({type, type}, {type, type}, {{0, 1}, {0, 1}}, 2,
                               ::fabric::TemporalSwitchFixedPriority{{0, 1}})));
  std::vector<SpatialValue> outputs(roundRobin.values().begin(),
                                    roundRobin.values().end());
  outputs.insert(outputs.end(), fixedPriority.values().begin(),
                 fixedPriority.values().end());
  if (llvm::Error error = spatial.close(outputs))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "contention fixture did not finalize one Fabric root");
  return std::move(finalized.roots().front());
}

loom::fabric::FinalizedFabricRoot
buildSplitContentionFixture(llvm::StringRef test, loom::ArtifactStore &store) {
  using namespace loom::adg;
  DesignBuilder design(store);
  const PortType type = take(test, PortType::taggedBits(32, 4));
  auto spatial = take(
      test, design.createSpatialCore("split-configured-switch-contention",
                                     {type, type, type, type}, {type, type}));
  const std::vector<std::uint32_t> inputOrder{0, 1, 2, 3};
  auto outputs = take(
      test,
      spatial.addSwitch(
          {take(test, spatial.input(0)), take(test, spatial.input(1)),
           take(test, spatial.input(2)), take(test, spatial.input(3))},
          SwitchSpec::temporal(
              {type, type, type, type}, {type, type}, {inputOrder, inputOrder},
              4, ::fabric::TemporalSwitchFixedPriority{inputOrder})));
  if (llvm::Error error = spatial.close(outputs.values()))
    fail(test, llvm::toString(std::move(error)));
  auto finalized = take(test, std::move(design).finalize());
  require(test, finalized.roots().size() == 1,
          "split-contention fixture did not finalize one Fabric root");
  return std::move(finalized.roots().front());
}

std::vector<SwitchFixture>
describeSwitches(llvm::StringRef test,
                 const loom::fabric::FabricArtifactView &view) {
  std::vector<SwitchFixture> result;
  result.reserve(view.switchOccurrences().size());
  for (FabricSwitchOccurrenceRef occurrence : view.switchOccurrences())
    result.push_back({occurrence, {}, {}, {}});
  for (const auto &traversal : view.physicalTraversals()) {
    const auto *crosspoint =
        std::get_if<loom::fabric::FabricSwitchTraversalPayload>(
            &traversal.reference.payload);
    if (!crosspoint)
      continue;
    require(test,
            traversal.sources.size() == 1 && traversal.destinations.size() == 1,
            "switch traversal changed endpoint cardinality");
    const auto owner = llvm::find_if(result, [&](const auto &candidate) {
      return candidate.occurrence == crosspoint->owner;
    });
    require(test, owner != result.end(),
            "switch traversal names an unknown occurrence");
    owner->crosspoints.emplace(
        std::make_pair(crosspoint->input, crosspoint->output),
        traversal.reference);
    owner->inputs.emplace(crosspoint->input, traversal.sources.front());
    owner->outputs.emplace(crosspoint->output, traversal.destinations.front());
  }
  for (const SwitchFixture &sw : result)
    require(test, sw.crosspoints.size() == sw.inputs.size() * sw.outputs.size(),
            "fixture switch is not fully connected");
  return result;
}

SwitchFixture describeSwitch(llvm::StringRef test,
                             const loom::fabric::FabricArtifactView &view) {
  std::vector<SwitchFixture> switches = describeSwitches(test, view);
  require(test, switches.size() == 1,
          "fixture did not expose one switch occurrence");
  SwitchFixture result = std::move(switches.front());
  return result;
}

void fixedPriorityContentionPreservesGrantDirection() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  loom::fabric::FinalizedFabricRoot finalized = buildFixture(test, store);
  const auto context =
      take(test, loom::fabric::buildFabricHandshakeContext(finalized.view()));
  if (llvm::Error error = loom::fabric::revalidateFabricHandshakeContext(
          context, finalized.view()))
    fail(test, llvm::toString(std::move(error)));
  const SwitchFixture sw = describeSwitch(test, finalized.view());

  const std::array<loom::fabric::FabricSwitchSelectedCrosspoint, 4>
      selectedCrosspoints = {{{0, 0}, {1, 0}, {1, 1}, {2, 1}}};
  loom::fabric::FabricSwitchSelectedContentionScratch contentionScratch;
  loom::fabric::FabricSwitchSelectedContention contention;
  contentionScratch.prepare(selectedCrosspoints.size());
  contention.prepare(selectedCrosspoints.size());
  const std::size_t retainedContentionBytes =
      contentionScratch.retainedStorageBytes() +
      contention.retainedStorageBytes();
  require(test, loom::test::allocationProbeIsCalibrated(),
          "allocation probe is not calibrated");
  loom::test::startAllocationProbe();
  contention.rebuild(sw.occurrence, selectedCrosspoints, contentionScratch);
  const std::size_t contentionAllocations = loom::test::stopAllocationProbe();
  require(test,
          contentionAllocations == 0 &&
              contentionScratch.retainedStorageBytes() +
                      contention.retainedStorageBytes() ==
                  retainedContentionBytes,
          "first prepared contention derivation allocated or changed retained "
          "storage");

  FabricHandshakeSelection selection;
  selection.switchActivations = {
      {{sw.occurrence, 0, 0}, {sw.crosspoint(test, 0, 0)}},
      {{sw.occurrence, 1, 1},
       {sw.crosspoint(test, 1, 0), sw.crosspoint(test, 1, 1)}},
      {{sw.occurrence, 2, 2}, {sw.crosspoint(test, 2, 1)}}};
  const std::array<HandshakeSignalRef, 8> terminals = {
      HandshakeSignalRef{sw.inputs.at(0), HandshakeSignalKind::Valid},
      HandshakeSignalRef{sw.inputs.at(0), HandshakeSignalKind::Ready},
      HandshakeSignalRef{sw.inputs.at(1), HandshakeSignalKind::Valid},
      HandshakeSignalRef{sw.inputs.at(1), HandshakeSignalKind::Ready},
      HandshakeSignalRef{sw.inputs.at(2), HandshakeSignalKind::Valid},
      HandshakeSignalRef{sw.inputs.at(2), HandshakeSignalKind::Ready},
      HandshakeSignalRef{sw.outputs.at(0), HandshakeSignalKind::Valid},
      HandshakeSignalRef{sw.outputs.at(1), HandshakeSignalKind::Valid}};
  const auto reachability =
      take(test, loom::fabric::deriveSelectedHandshakeReachability(
                     finalized.view(), selection, terminals, context));
  const auto reaches = [&](const HandshakeSignalRef &from,
                           const HandshakeSignalRef &to) {
    return llvm::is_contained(reachability,
                              loom::fabric::HandshakeDependencyArc{from, to});
  };

  require(test,
          reaches(terminals[0], terminals[6]) &&
              reaches(terminals[2], terminals[6]) &&
              reaches(terminals[2], terminals[7]) &&
              reaches(terminals[4], terminals[7]),
          "fixed-priority grant dependencies lost their forward direction");
  require(test,
          !reaches(terminals[0], terminals[7]) &&
              !reaches(terminals[4], terminals[6]),
          "a lower-priority requester reached an unrelated output");
  require(test,
          reaches(terminals[4], terminals[1]) &&
              reaches(terminals[0], terminals[5]),
          "idle presentation lost all-input readiness dependence");
}

void unusedCrosspointsDoNotMergeConfiguredContention() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  loom::fabric::FinalizedFabricRoot finalized =
      buildSplitContentionFixture(test, store);
  const SwitchFixture sw = describeSwitch(test, finalized.view());
  const loom::fabric::FabricHandshakeContext context =
      take(test, loom::fabric::buildFabricHandshakeContext(finalized.view()));
  FabricHandshakeSelection selection;
  selection.switchActivations = {
      {{sw.occurrence, 0, 0}, {sw.crosspoint(test, 0, 0)}},
      {{sw.occurrence, 1, 1}, {sw.crosspoint(test, 1, 0)}},
      {{sw.occurrence, 2, 2}, {sw.crosspoint(test, 2, 1)}},
      {{sw.occurrence, 3, 3}, {sw.crosspoint(test, 3, 1)}}};
  const std::array<HandshakeSignalRef, 8> terminals = {
      HandshakeSignalRef{sw.inputs.at(0), HandshakeSignalKind::Valid},
      HandshakeSignalRef{sw.inputs.at(0), HandshakeSignalKind::Ready},
      HandshakeSignalRef{sw.inputs.at(1), HandshakeSignalKind::Ready},
      HandshakeSignalRef{sw.inputs.at(2), HandshakeSignalKind::Valid},
      HandshakeSignalRef{sw.inputs.at(2), HandshakeSignalKind::Ready},
      HandshakeSignalRef{sw.inputs.at(3), HandshakeSignalKind::Ready},
      HandshakeSignalRef{sw.outputs.at(0), HandshakeSignalKind::Valid},
      HandshakeSignalRef{sw.outputs.at(1), HandshakeSignalKind::Valid}};
  const auto reachability =
      take(test, loom::fabric::deriveSelectedHandshakeReachability(
                     finalized.view(), selection, terminals, context));
  const auto reaches = [&](std::size_t from, std::size_t to) {
    return llvm::is_contained(
        reachability,
        loom::fabric::HandshakeDependencyArc{terminals[from], terminals[to]});
  };
  require(test,
          reaches(0, 2) && reaches(3, 5) && reaches(0, 6) && reaches(3, 7),
          "configured contention groups lost their local dependencies");
  require(test,
          !reaches(0, 4) && !reaches(3, 1) && !reaches(0, 7) && !reaches(3, 6),
          "unused crosspoints merged configured contention groups");
}

void temporalContentionOwnsConfiguredDependencies() {
  const llvm::StringRef test = __func__;
  TemporaryDirectory directory(test);
  loom::ArtifactStore store(directory.path());
  loom::fabric::FinalizedFabricRoot finalized =
      buildContentionFixture(test, store);
  const loom::fabric::FabricHandshakeContext context =
      take(test, loom::fabric::buildFabricHandshakeContext(finalized.view()));
  if (llvm::Error error = loom::fabric::revalidateFabricHandshakeContext(
          context, finalized.view()))
    fail(test, llvm::toString(std::move(error)));
  const FabricHandshakeSelection emptySelection;
  for (const HandshakeOwnerModel &model : context.ownerModels()) {
    const ResolvedHandshakeActivation inactive = take(
        test, loom::fabric::resolveSelectedHandshake(model, emptySelection));
    for (std::uint32_t fragment : inactive.fragmentOrdinals())
      require(test,
              model.fragment(fragment).activationKind !=
                  loom::fabric::HandshakeActivationKind::SwitchContention,
              "a switch-contention fragment became unconditional");
  }

  const auto switches = describeSwitches(test, finalized.view());
  const SwitchFixture *fanout = nullptr;
  std::vector<const SwitchFixture *> sinks;
  for (const auto &candidate : switches) {
    if (candidate.inputs.size() == 1)
      fanout = &candidate;
    else
      sinks.push_back(&candidate);
  }
  require(test, fanout && sinks.size() == 2,
          "contention fixture did not expose one fanout and two sinks");

  FabricHandshakeSelection contended;
  contended.switchActivations.push_back(
      {{fanout->occurrence, 0, 0},
       {fanout->crosspoint(test, 0, 0), fanout->crosspoint(test, 0, 1)}});
  for (const SwitchFixture *sink : sinks) {
    contended.switchActivations.push_back(
        {{sink->occurrence, 0, 0}, {sink->crosspoint(test, 0, 0)}});
    contended.switchActivations.push_back(
        {{sink->occurrence, 1, 1}, {sink->crosspoint(test, 1, 0)}});
  }
  requireRejected(test,
                  loom::fabric::verifySelectedCombinationalHandshakeAcyclic(
                      finalized.view(), contended, context),
                  "SelectedCombinationalHandshakeCycle");

  FabricHandshakeSelection oneContender = contended;
  oneContender.switchActivations.pop_back();
  if (llvm::Error error =
          loom::fabric::verifySelectedCombinationalHandshakeAcyclic(
              finalized.view(), oneContender, context))
    fail(test, "one contending sink closed a cycle: " +
                   llvm::toString(std::move(error)));

  const auto transitiveSink = llvm::find_if(sinks, [](const auto *candidate) {
    return candidate->inputs.size() == 3;
  });
  require(test, transitiveSink != sinks.end(),
          "contention fixture has no transitive sink");
  const SwitchFixture &sink = **transitiveSink;
  const HandshakeSignalRef input0Valid{sink.inputs.at(0),
                                       HandshakeSignalKind::Valid};
  const HandshakeSignalRef input0Ready{sink.inputs.at(0),
                                       HandshakeSignalKind::Ready};
  const HandshakeSignalRef input1Valid{sink.inputs.at(1),
                                       HandshakeSignalKind::Valid};
  const HandshakeSignalRef input1Ready{sink.inputs.at(1),
                                       HandshakeSignalKind::Ready};
  const HandshakeSignalRef input2Valid{sink.inputs.at(2),
                                       HandshakeSignalKind::Valid};
  const HandshakeSignalRef input2Ready{sink.inputs.at(2),
                                       HandshakeSignalKind::Ready};
  const HandshakeSignalRef output0Valid{sink.outputs.at(0),
                                        HandshakeSignalKind::Valid};
  const HandshakeSignalRef output0Ready{sink.outputs.at(0),
                                        HandshakeSignalKind::Ready};
  const HandshakeSignalRef output1Valid{sink.outputs.at(1),
                                        HandshakeSignalKind::Valid};
  const HandshakeSignalRef output2Valid{sink.outputs.at(2),
                                        HandshakeSignalKind::Valid};
  const std::array<HandshakeSignalRef, 10> terminals = {
      input0Valid, input0Ready,  input1Valid,  input1Ready,  input2Valid,
      input2Ready, output0Valid, output0Ready, output1Valid, output2Valid};
  const auto project = [&](const FabricHandshakeSelection &selection) {
    return take(test, loom::fabric::deriveSelectedHandshakeReachability(
                          finalized.view(), selection, terminals, context));
  };
  const auto reaches =
      [](llvm::ArrayRef<loom::fabric::HandshakeDependencyArc> reachability,
         const HandshakeSignalRef &from, const HandshakeSignalRef &to) {
        return llvm::is_contained(
            reachability, loom::fabric::HandshakeDependencyArc{from, to});
      };

  FabricHandshakeSelection single;
  single.switchActivations = {
      {{sink.occurrence, 0, 0}, {sink.crosspoint(test, 0, 0)}}};
  const auto singleReachability = project(single);
  require(test,
          !reaches(singleReachability, input0Valid, input0Ready) &&
              reaches(singleReachability, output0Ready, input0Ready) &&
              reaches(singleReachability, input0Valid, output0Valid),
          "a one-input component gained a contention dependency");

  FabricHandshakeSelection disjoint;
  disjoint.switchActivations = {
      {{sink.occurrence, 0, 0}, {sink.crosspoint(test, 0, 0)}},
      {{sink.occurrence, 1, 1}, {sink.crosspoint(test, 1, 1)}}};
  const auto disjointReachability = project(disjoint);
  require(test,
          !reaches(disjointReachability, input0Valid, input0Ready) &&
              !reaches(disjointReachability, input1Valid, input0Ready) &&
              !reaches(disjointReachability, input0Valid, output1Valid),
          "unselected crosspoints added a configured disjoint-row "
          "dependency");

  FabricHandshakeSelection shared;
  shared.switchActivations = {
      {{sink.occurrence, 0, 0}, {sink.crosspoint(test, 0, 0)}},
      {{sink.occurrence, 1, 1}, {sink.crosspoint(test, 1, 0)}}};
  const auto sharedReachability = project(shared);
  require(test,
          reaches(sharedReachability, input0Valid, input0Ready) &&
              reaches(sharedReachability, input1Valid, input0Ready) &&
              reaches(sharedReachability, input0Valid, input1Ready) &&
              reaches(sharedReachability, input1Valid, input1Ready) &&
              reaches(sharedReachability, input1Valid, output0Valid) &&
              reaches(sharedReachability, input0Valid, output0Valid),
          "configured contention lost grant or presentation reachability");
  require(test,
          !reaches(sharedReachability, input0Valid, output1Valid) &&
              !reaches(sharedReachability, output0Ready, output0Valid),
          "configured contention reached outside its selected component");

  FabricHandshakeSelection transitive;
  transitive.switchActivations = {
      {{sink.occurrence, 0, 0}, {sink.crosspoint(test, 0, 0)}},
      {{sink.occurrence, 1, 1},
       {sink.crosspoint(test, 1, 0), sink.crosspoint(test, 1, 1)}},
      {{sink.occurrence, 2, 2},
       {sink.crosspoint(test, 2, 1), sink.crosspoint(test, 2, 2)}}};
  const auto transitiveReachability = project(transitive);
  require(test,
          reaches(transitiveReachability, input0Valid, input2Ready) &&
              reaches(transitiveReachability, input0Valid, output2Valid) &&
              reaches(transitiveReachability, input2Valid, input0Ready) &&
              reaches(transitiveReachability, input2Valid, output0Valid),
          "sparse transitive contention lost component reachability");
}

} // namespace

int main() {
  fixedPriorityContentionPreservesGrantDirection();
  unusedCrosspointsDoNotMergeConfiguredContention();
  temporalContentionOwnsConfiguredDependencies();
  llvm::outs() << "Fabric switch handshake policy tests passed\n";
  return EXIT_SUCCESS;
}

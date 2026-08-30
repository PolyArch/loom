#include "ADG/Builder.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Identity/FabricRefImport.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "ADG mesh switch network test: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(bool condition, llvm::StringRef message) {
  if (!condition)
    fail(message);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(llvm::Expected<T> value, llvm::StringRef diagnostic) {
  if (value)
    fail("accepted invalid mesh switch network authoring");
  const std::string message = llvm::toString(value.takeError());
  require(llvm::StringRef(message).contains(diagnostic), message);
}

class TemporaryDirectory final {
public:
  TemporaryDirectory() {
    llvm::SmallString<128> path;
    if (std::error_code error = llvm::sys::fs::createUniqueDirectory(
            "loom-adg-mesh-switch-network", path))
      fail(error.message());
    path_ = path.str().str();
  }

  ~TemporaryDirectory() { llvm::sys::fs::remove_directories(path_); }

  llvm::StringRef path() const { return path_; }

private:
  std::string path_;
};

std::pair<std::uint64_t, std::uint64_t>
switchShape(const loom::fabric::FabricArtifactView &view,
            loom::fabric::FabricSwitchOccurrenceRef occurrence) {
  const auto owner =
      loom::fabric::FabricTransportEndpointOwnerRef::of(occurrence);
  std::uint64_t inputs = 0;
  std::uint64_t outputs = 0;
  const std::uint64_t count = view.transportEndpointCount(owner);
  for (std::uint64_t ordinal = 0; ordinal != count; ++ordinal) {
    const loom::fabric::FabricTransportEndpointRef endpoint{
        owner, loom::fabric::FabricOrdinal(ordinal)};
    const auto direction = view.transportEndpointDirection(endpoint);
    require(direction.has_value(), "switch endpoint lost its direction");
    if (*direction == loom::fabric::FabricPortDirection::Input)
      ++inputs;
    else
      ++outputs;
  }
  return {inputs, outputs};
}

void checkSpatialNetwork() {
  using namespace loom::adg;

  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(PortType::bits(32));

  auto core = take(design.createSpatialCore(
      "regular-network", {bits32, bits32, bits32}, {bits32, bits32, bits32}));
  std::vector<MeshCellAttachmentSpec> attachments{
      {1, 1, {bits32, bits32}, {bits32, bits32}}, {0, 0, {bits32}, {bits32}}};
  auto network =
      take(core.addMeshSwitchNetwork(take(MeshSwitchNetworkSpec::spatial(
          3, 3, maximumMeshLanesPerDirection, bits32, 1,
          ::fabric::FifoQueueDiscipline::StrictFifo, attachments))));

  auto center = take(network.attachment(0));
  auto corner = take(network.attachment(1));
  require(center.inputs().size() == 2 && corner.inputs().size() == 1,
          "mesh attachment input banks changed shape");
  std::vector<SpatialValue> centerOutputs{take(core.input(0)),
                                          take(core.input(1))};
  if (llvm::Error error = center.connectOutputs(centerOutputs))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = corner.connectOutputs({take(core.input(2))}))
    fail(llvm::toString(std::move(error)));

  std::vector<SpatialValue> outputs(center.inputs().begin(),
                                    center.inputs().end());
  outputs.insert(outputs.end(), corner.inputs().begin(), corner.inputs().end());
  if (llvm::Error error = core.close(outputs))
    fail(llvm::toString(std::move(error)));

  auto finalized = take(std::move(design).finalize());
  require(finalized.roots().size() == 1,
          "spatial mesh did not finalize one Module");
  const auto &view = finalized.roots().front().view();
  require(view.fifoOccurrences().size() == 96,
          "3x3 four-lane mesh did not emit one FIFO per directed link");

  bool foundInteriorTransit = false;
  for (const auto occurrence : view.switchOccurrences()) {
    const auto [inputs, outputs] = switchShape(view, occurrence);
    require(inputs <= 16 && outputs <= 16 && inputs * outputs <= 256,
            "mesh helper emitted a switch beyond its crosspoint limit");
    foundInteriorTransit |= inputs == 16 && outputs == 16;
  }
  require(foundInteriorTransit,
          "3x3 four-lane mesh lost its interior 16x16 transit switch");

  std::string text;
  llvm::raw_string_ostream stream(text);
  if (llvm::Error error =
          loom::fabric::writeFabricMlir(finalized.roots().front(), stream))
    fail(llvm::toString(std::move(error)));
  stream.flush();
  const llvm::StringRef printed(text);
  require(!printed.contains("mesh_") && !printed.contains("coordinate") &&
              !printed.contains("distance") && !printed.contains("tile_"),
          "mesh authoring metadata escaped into finalized Fabric");
}

loom::fabric::FinalizedFabricRoot
buildTemporalNetwork(loom::ArtifactStore &store,
                     loom::adg::MeshSwitchGrantPolicyKind policy,
                     llvm::StringRef label) {
  using namespace loom::adg;

  DesignBuilder design(store);
  const PortType tagged32 = take(PortType::taggedBits(32, 4));
  auto core = take(design.createSpatialCore(label, {tagged32}, {tagged32}));
  auto network =
      take(core.addMeshSwitchNetwork(take(MeshSwitchNetworkSpec::temporal(
          2, 2, 1, tagged32, 1, ::fabric::FifoQueueDiscipline::StrictFifo, 4,
          policy, {{0, 0, {tagged32}, {tagged32}}}))));
  auto attachment = take(network.attachment(0));
  if (llvm::Error error = attachment.connectOutputs({take(core.input(0))}))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = core.close(attachment.inputs()))
    fail(llvm::toString(std::move(error)));
  auto finalized = take(std::move(design).finalize());
  require(finalized.roots().size() == 1,
          "temporal mesh did not finalize one Module");
  return finalized.roots().front();
}

void checkTemporalPolicies() {
  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  const auto fixed = buildTemporalNetwork(
      store, loom::adg::MeshSwitchGrantPolicyKind::FixedPriority,
      "fixed-network");
  const auto roundRobin = buildTemporalNetwork(
      store, loom::adg::MeshSwitchGrantPolicyKind::RoundRobin,
      "round-robin-network");

  const auto check = [](const loom::fabric::FinalizedFabricRoot &root,
                        bool expectRoundRobin) {
    const auto &view = root.view();
    std::size_t competing = 0;
    for (const auto occurrence : view.switchOccurrences()) {
      const auto owner = loom::fabric::FabricInventoryOwnerRef::of(occurrence);
      const auto *contract = view.resourceContract(owner);
      require(contract != nullptr,
              "temporal switch lost its resource contract");
      if (contract->requesterCount() <= 1) {
        require(!contract->grantPolicy(),
                "noncompeting temporal switch gained arbitration policy");
        continue;
      }
      ++competing;
      const auto policy = contract->grantPolicy();
      require(policy.has_value(),
              "competing temporal switch lost arbitration policy");
      if (expectRoundRobin) {
        const auto *selected = std::get_if<::fabric::RoundRobinView>(&*policy);
        require(selected &&
                    selected->requesterCycle().size() ==
                        contract->requesterCount() &&
                    selected->resetCursor().ordinal() == 0,
                "temporal mesh lost its complete round-robin policy");
      } else {
        const auto *selected =
            std::get_if<::fabric::FixedPriorityView>(&*policy);
        require(selected && selected->requesterOrder().size() ==
                                contract->requesterCount(),
                "temporal mesh lost its complete fixed-priority policy");
      }
    }
    require(competing != 0,
            "temporal mesh did not contain a physical fan-in switch");
  };

  check(fixed, false);
  check(roundRobin, true);
}

void checkExplicitDomains() {
  using namespace loom::adg;

  TemporaryDirectory directory;
  loom::ArtifactStore store(directory.path());
  DesignBuilder design(store);
  const PortType bits32 = take(PortType::bits(32));
  auto core =
      take(design.createSpatialCore("domain-network", {bits32}, {bits32}));
  const auto clock =
      take(core.declareDomainSlot(loom::fabric::FabricClockResetKind::Clock));
  const auto reset =
      take(core.declareDomainSlot(loom::fabric::FabricClockResetKind::Reset));
  auto network =
      take(core.addMeshSwitchNetwork(take(MeshSwitchNetworkSpec::spatial(
          2, 2, 1, bits32, 1, ::fabric::FifoQueueDiscipline::StrictFifo,
          {{0, 0, {bits32}, {bits32}}}))));
  require(!network.domainMembers().empty(),
          "mesh network did not expose its domain members");
  for (const ModuleDomainMemberHandle &member : network.domainMembers()) {
    if (llvm::Error error = core.assignDomainSlot(member, clock))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error = core.assignDomainSlot(member, reset))
      fail(llvm::toString(std::move(error)));
  }
  for (std::size_t ordinal = 0; ordinal != 1; ++ordinal) {
    const auto input = take(core.inputDomainMember(ordinal));
    const auto output = take(core.outputDomainMember(ordinal));
    if (llvm::Error error = core.assignDomainSlot(input, clock))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error = core.assignDomainSlot(input, reset))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error = core.assignDomainSlot(output, clock))
      fail(llvm::toString(std::move(error)));
    if (llvm::Error error = core.assignDomainSlot(output, reset))
      fail(llvm::toString(std::move(error)));
  }
  auto attachment = take(network.attachment(0));
  if (llvm::Error error = attachment.connectOutputs({take(core.input(0))}))
    fail(llvm::toString(std::move(error)));
  if (llvm::Error error = core.close(attachment.inputs()))
    fail(llvm::toString(std::move(error)));
  auto finalized = take(std::move(design).finalize());
  require(finalized.roots().size() == 1,
          "explicit-domain mesh did not finalize one Module");
}

void checkInvalidSpecifications() {
  using namespace loom::adg;

  const PortType bits32 = take(PortType::bits(32));
  const PortType tagged32 = take(PortType::taggedBits(32, 4));
  const MeshCellAttachmentSpec bank{0, 0, {bits32}, {bits32}};
  constexpr ::fabric::FifoQueueDiscipline strict =
      ::fabric::FifoQueueDiscipline::StrictFifo;
  expectError(MeshSwitchNetworkSpec::spatial(0, 2, 1, bits32, 1, strict, {bank}),
              "positive");
  expectError(
      MeshSwitchNetworkSpec::spatial(1, 1, 1, bits32, 1, strict, {bank}),
      "at least two");
  expectError(MeshSwitchNetworkSpec::spatial(
                  2, 2, maximumMeshLanesPerDirection + 1, bits32, 1, strict,
                  {bank}),
              "between one and 4");
  expectError(MeshSwitchNetworkSpec::spatial(2, 2, 1, bits32, 0, strict, {bank}),
              "interconnect FIFO depth");
  expectError(MeshSwitchNetworkSpec::spatial(2, 2, 1, tagged32, 1, strict,
                                             {bank}),
              "untagged bits");
  expectError(
      MeshSwitchNetworkSpec::temporal(
          2, 2, 1, bits32, 1, strict, 4, MeshSwitchGrantPolicyKind::RoundRobin,
          {bank}),
      "tagged bits");
  expectError(MeshSwitchNetworkSpec::temporal(
                  2, 2, 1, tagged32, 1, strict, 0,
                  MeshSwitchGrantPolicyKind::RoundRobin,
                  {{0, 0, {tagged32}, {tagged32}}}),
              "positive route-table");
  expectError(MeshSwitchNetworkSpec::spatial(2, 2, 1, bits32, 1, strict,
                                             {{2, 0, {bits32}, {bits32}}}),
              "outside");
  expectError(MeshSwitchNetworkSpec::spatial(2, 2, 1, bits32, 1, strict,
                                             {{0, 0, {}, {}}}),
              "at least one port");
  expectError(
      MeshSwitchNetworkSpec::spatial(
          2, 2, 1, bits32, 1, strict,
          {{0, 0, std::vector<PortType>(9, bits32), {}}}),
      "at most eight");
  expectError(
      MeshSwitchNetworkSpec::spatial(2, 2, 1, bits32, 1, strict,
                                     std::vector<MeshCellAttachmentSpec>(8,
                                                                         bank)),
      "at most seven");
}

} // namespace

int main() {
  checkInvalidSpecifications();
  checkSpatialNetwork();
  checkTemporalPolicies();
  checkExplicitDomains();
  return EXIT_SUCCESS;
}

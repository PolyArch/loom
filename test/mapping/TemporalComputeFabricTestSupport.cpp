#include "TechMappingCandidateTestSupport.h"

#include "ADG/FuLibrary.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "temporal compute Fabric test: " << message << '\n';
  std::exit(1);
}

template <typename T> T take(llvm::Expected<T> value) {
  if (!value)
    fail(llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(llvm::Error error) {
  if (error)
    fail(llvm::toString(std::move(error)));
}

enum class TemporalComputeRoutingKind : std::uint8_t {
  Direct,
  PackedSwitch,
  FeedbackCycle,
};

loom::adg::FinalizedFabricDesign
buildTemporalComputeFabric(const loom::ArtifactStore &store,
                           TemporalComputeRoutingKind routingKind,
                           std::uint64_t residentRows = 2) {
  using namespace loom::adg;

  DesignBuilder design(store);
  const PortType bits128 = take(PortType::bits(128));
  const PortType tagged128 = take(PortType::taggedBits(128, 4));
  const std::vector<PortType> moduleInputs(10, tagged128);
  const std::vector<PortType> moduleOutputs(8, tagged128);
  llvm::StringRef name;
  switch (routingKind) {
  case TemporalComputeRoutingKind::Direct:
    name = "capacity-envelope";
    break;
  case TemporalComputeRoutingKind::PackedSwitch:
    name = "switch-row-packing";
    break;
  case TemporalComputeRoutingKind::FeedbackCycle:
    name = "handshake-feedback-cycle";
    break;
  }
  auto spatial =
      take(design.createSpatialCore(name, moduleInputs, moduleOutputs));

  std::vector<SpatialValue> outputs;
  outputs.reserve(moduleOutputs.size());
  for (unsigned peOrdinal = 0; peOrdinal != 2; ++peOrdinal) {
    std::vector<SpatialValue> peInputs;
    peInputs.reserve(5);
    for (unsigned input = 0; input != 5; ++input)
      peInputs.push_back(take(spatial.input(peOrdinal * 5 + input)));
    if (routingKind == TemporalComputeRoutingKind::PackedSwitch) {
      const std::vector<PortType> switchTypes(5, tagged128);
      const std::vector<std::uint32_t> switchInputsByPriority{0, 1, 2, 3, 4};
      const std::vector<std::vector<std::uint32_t>> sourcesByOutput(
          5, switchInputsByPriority);
      auto switched = take(spatial.addSwitch(
          peInputs,
          SwitchSpec::temporal(
              switchTypes, switchTypes, sourcesByOutput, residentRows,
              ::fabric::TemporalSwitchFixedPriority{switchInputsByPriority})));
      peInputs.assign(switched.values().begin(), switched.values().end());
    } else if (routingKind == TemporalComputeRoutingKind::FeedbackCycle &&
               peOrdinal == 0) {
      auto backedge = take(spatial.createBackedge(tagged128));
      auto switched = take(spatial.addSwitch(
          {peInputs.front(), backedge.value()},
          SwitchSpec::temporal({tagged128, tagged128}, {tagged128, tagged128},
                               {{0, 1}, {0, 1}}, 2,
                               ::fabric::TemporalSwitchFixedPriority{{0, 1}})));
      auto feedback =
          take(spatial.addFifo(switched[0], FifoSpec{tagged128, 2, true}));
      requireSuccess(
          spatial.resolveBackedge(std::move(backedge), feedback.value()));
      peInputs.front() = switched[1];
    }
    const ::fabric::OperandBufferMode mode =
        peOrdinal == 0 ? ::fabric::OperandBufferMode::AllFuShare
                       : ::fabric::OperandBufferMode::PerInstruction;
    auto pe = take(spatial.addPe(
        peInputs, PeSpec::temporal(std::vector<PortType>(5, bits128),
                                   std::vector<PortType>(4, tagged128),
                                   TemporalPeParameters{
                                       2, FuConfigurationMode::PerInstruction,
                                       mode, 2, std::nullopt})));
    std::vector<PeValue> fuInputs;
    fuInputs.reserve(5);
    for (unsigned input = 0; input != 5; ++input)
      fuInputs.push_back(take(pe.input(input)));
    requireSuccess(
        addTokenControlFu(pe, fuInputs, TokenControlFuParameters{128, 64}));
    requireSuccess(pe.close());
    for (unsigned output = 0; output != 4; ++output)
      outputs.push_back(take(pe.output(output)));
  }
  requireSuccess(spatial.close(outputs));
  return take(std::move(design).finalize());
}

} // namespace

loom::adg::FinalizedFabricDesign
loom::test::buildTemporalCapacityFabric(const ArtifactStore &store) {
  return buildTemporalComputeFabric(store, TemporalComputeRoutingKind::Direct);
}

loom::adg::FinalizedFabricDesign
loom::test::buildTemporalSwitchPackingFabric(const ArtifactStore &store,
                                             std::uint64_t residentRows) {
  return buildTemporalComputeFabric(
      store, TemporalComputeRoutingKind::PackedSwitch, residentRows);
}

loom::adg::FinalizedFabricDesign
loom::test::buildTemporalHandshakeCycleFabric(const ArtifactStore &store) {
  return buildTemporalComputeFabric(store,
                                    TemporalComputeRoutingKind::FeedbackCycle);
}

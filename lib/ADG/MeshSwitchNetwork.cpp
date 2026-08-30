#include "ADG/Builder.h"

#include "BuilderInternal.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace loom::adg {
namespace detail {

struct MeshCellAttachmentState final {
  std::vector<SpatialValue> inputs;
  std::vector<SpatialBackedge> outputBackedges;
  bool outputsConnected = false;
};

struct MeshSwitchNetworkState final {
  std::weak_ptr<DesignState> design;
  std::size_t rootOrdinal = 0;
  std::vector<MeshCellAttachmentState> attachments;
  std::vector<ModuleDomainMemberHandle> domainMembers;
};

} // namespace detail
namespace {

llvm::Error
validateMeshSpec(::fabric::Schedule schedule, std::uint32_t width,
                 std::uint32_t height, std::uint32_t lanesPerDirection,
                 const PortType &linkType, std::uint32_t interconnectFifoDepth,
                 std::optional<std::uint32_t> routeTableSize,
                 std::optional<MeshSwitchGrantPolicyKind> grantPolicyKind,
                 llvm::ArrayRef<MeshCellAttachmentSpec> attachments) {
  if (width == 0 || height == 0)
    return detail::invalid("mesh width and height must be positive");
  const std::uint64_t cellCount =
      static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height);
  if (cellCount < 2)
    return detail::invalid("mesh must contain at least two cells");
  if (cellCount > std::numeric_limits<std::size_t>::max())
    return detail::invalid("mesh cell count exceeds the host address space");
  if (lanesPerDirection == 0 ||
      lanesPerDirection > maximumMeshLanesPerDirection)
    return detail::invalid("mesh lanes per direction must be between one and " +
                           llvm::Twine(maximumMeshLanesPerDirection));
  if (interconnectFifoDepth == 0)
    return detail::invalid("mesh interconnect FIFO depth must be positive");

  if (schedule == ::fabric::Schedule::Spatial) {
    if (linkType.kind() != PortType::Kind::Bits)
      return detail::invalid("spatial mesh link type must be untagged bits");
    if (routeTableSize || grantPolicyKind)
      return detail::invalid(
          "spatial mesh cannot declare temporal switch parameters");
  } else {
    if (linkType.kind() != PortType::Kind::TaggedBits)
      return detail::invalid("temporal mesh link type must be tagged bits");
    if (!routeTableSize || *routeTableSize == 0)
      return detail::invalid(
          "temporal mesh requires a positive route-table size");
    if (!grantPolicyKind)
      return detail::invalid("temporal mesh requires a grant policy kind");
  }

  llvm::DenseMap<std::uint64_t, std::uint32_t> banksPerCell;
  for (const MeshCellAttachmentSpec &attachment : attachments) {
    if (attachment.x >= width || attachment.y >= height)
      return detail::invalid("mesh attachment cell is outside the network");
    if (attachment.inputTypes.empty() && attachment.outputTypes.empty())
      return detail::invalid("mesh attachment bank requires at least one port");
    if (attachment.inputTypes.size() > 8 || attachment.outputTypes.size() > 8)
      return detail::invalid(
          "mesh attachment bank admits at most eight inputs and outputs");
    for (const PortType &type : attachment.inputTypes)
      if (type.kind() != linkType.kind())
        return detail::invalid(
            "mesh attachment input has a different transport kind");
    for (const PortType &type : attachment.outputTypes)
      if (type.kind() != linkType.kind())
        return detail::invalid(
            "mesh attachment output has a different transport kind");
    const std::uint64_t cell =
        static_cast<std::uint64_t>(attachment.y) * width + attachment.x;
    if (++banksPerCell[cell] > 7)
      return detail::invalid("mesh cell admits at most seven attachment banks");
  }

  return llvm::Error::success();
}

std::vector<std::vector<std::uint32_t>>
completeConnectivity(std::size_t inputCount, std::size_t outputCount) {
  std::vector<std::uint32_t> sources(inputCount);
  std::iota(sources.begin(), sources.end(), 0);
  return std::vector<std::vector<std::uint32_t>>(outputCount, sources);
}

std::vector<std::vector<std::uint32_t>>
fanoutConnectivity(std::size_t outputCount) {
  return std::vector<std::vector<std::uint32_t>>(outputCount, {0});
}

struct MeshCellBuildState final {
  struct OutgoingLink final {
    std::size_t destinationCell;
    std::size_t destinationIncomingOrdinal;
  };

  std::vector<SpatialBackedge> incoming;
  std::vector<OutgoingLink> outgoing;
  std::vector<SpatialValue> outgoingSources;
  std::vector<std::size_t> attachmentOrdinals;
};

} // namespace

llvm::Expected<MeshSwitchNetworkSpec> MeshSwitchNetworkSpec::spatial(
    std::uint32_t width, std::uint32_t height, std::uint32_t lanesPerDirection,
    const PortType &linkType, std::uint32_t interconnectFifoDepth,
    ::fabric::FifoQueueDiscipline interconnectFifoQueueDiscipline,
    std::vector<MeshCellAttachmentSpec> attachments) {
  if (llvm::Error error = validateMeshSpec(
          ::fabric::Schedule::Spatial, width, height, lanesPerDirection,
          linkType, interconnectFifoDepth, std::nullopt, std::nullopt,
          attachments))
    return std::move(error);
  return MeshSwitchNetworkSpec(::fabric::Schedule::Spatial, width, height,
                               lanesPerDirection, linkType,
                               interconnectFifoDepth,
                               interconnectFifoQueueDiscipline, std::nullopt,
                               std::nullopt, std::move(attachments));
}

llvm::Expected<MeshSwitchNetworkSpec> MeshSwitchNetworkSpec::temporal(
    std::uint32_t width, std::uint32_t height, std::uint32_t lanesPerDirection,
    const PortType &linkType, std::uint32_t interconnectFifoDepth,
    ::fabric::FifoQueueDiscipline interconnectFifoQueueDiscipline,
    std::uint32_t routeTableSize, MeshSwitchGrantPolicyKind grantPolicyKind,
    std::vector<MeshCellAttachmentSpec> attachments) {
  if (llvm::Error error = validateMeshSpec(
          ::fabric::Schedule::Temporal, width, height, lanesPerDirection,
          linkType, interconnectFifoDepth, routeTableSize, grantPolicyKind,
          attachments))
    return std::move(error);
  return MeshSwitchNetworkSpec(::fabric::Schedule::Temporal, width, height,
                               lanesPerDirection, linkType,
                               interconnectFifoDepth,
                               interconnectFifoQueueDiscipline, routeTableSize,
                               grantPolicyKind, std::move(attachments));
}

llvm::ArrayRef<SpatialValue> MeshCellAttachment::inputs() const {
  if (!state_ || ordinal_ >= state_->attachments.size())
    return {};
  return state_->attachments[ordinal_].inputs;
}

llvm::Error
MeshCellAttachment::connectOutputs(llvm::ArrayRef<SpatialValue> outputs) {
  if (!state_ || ordinal_ >= state_->attachments.size())
    return detail::invalid("mesh attachment handle is invalid");
  auto design = detail::activeState(state_->design);
  if (!design)
    return design.takeError();
  if (state_->rootOrdinal >= (*design)->spatialRoots.size())
    return detail::invalid("mesh attachment has an invalid owner ordinal");
  detail::MeshCellAttachmentState &attachment = state_->attachments[ordinal_];
  if (attachment.outputsConnected)
    return detail::invalid("mesh attachment outputs are already connected");
  if (outputs.size() != attachment.outputBackedges.size())
    return detail::invalid(
        "mesh attachment output count does not match its typed bank");

  SpatialCoreBuilder builder(*design, state_->rootOrdinal);
  llvm::SmallVector<mlir::Value, 8> resolvedSources;
  resolvedSources.reserve(outputs.size());
  for (auto [output, backedge] :
       llvm::zip(outputs, attachment.outputBackedges)) {
    auto source = builder.resolveValue(*design, output);
    if (!source)
      return source.takeError();
    auto placeholder = builder.resolveValue(*design, backedge.value());
    if (!placeholder)
      return placeholder.takeError();
    if (!source->use_empty())
      return detail::invalid(
          "mesh attachment output source already has a consumer");
    if (source->getType() != placeholder->getType())
      return detail::invalid(
          "mesh attachment output type does not match its typed bank");
    if (llvm::is_contained(resolvedSources, *source))
      return detail::invalid(
          "mesh attachment output source cannot drive multiple ports");
    resolvedSources.push_back(*source);
  }

  for (std::size_t ordinal = 0; ordinal != outputs.size(); ++ordinal)
    if (llvm::Error error = builder.resolveBackedge(
            std::move(attachment.outputBackedges[ordinal]), outputs[ordinal]))
      return error;
  attachment.outputsConnected = true;
  return llvm::Error::success();
}

llvm::Expected<MeshCellAttachment>
MeshSwitchNetwork::attachment(std::size_t ordinal) const {
  if (!state_ || ordinal >= state_->attachments.size())
    return detail::invalid("mesh attachment ordinal is out of range");
  return MeshCellAttachment(state_, ordinal);
}

llvm::ArrayRef<ModuleDomainMemberHandle>
MeshSwitchNetwork::domainMembers() const {
  if (!state_)
    return {};
  return state_->domainMembers;
}

std::size_t MeshSwitchNetwork::size() const {
  return state_ ? state_->attachments.size() : 0;
}

llvm::Expected<MeshSwitchNetwork>
SpatialCoreBuilder::addMeshSwitchNetwork(const MeshSwitchNetworkSpec &spec) {
  auto design = detail::activeState(state_);
  if (!design)
    return design.takeError();
  if (rootOrdinal_ >= (*design)->spatialRoots.size())
    return detail::invalid("SpatialCore handle has an invalid owner ordinal");
  if ((*design)->spatialRoots[rootOrdinal_].closed)
    return detail::invalid("SpatialCore is already closed");

  const std::size_t cellCount =
      static_cast<std::size_t>(spec.width_) * spec.height_;
  std::vector<MeshCellBuildState> cells(cellCount);
  auto networkState = std::make_shared<detail::MeshSwitchNetworkState>();
  networkState->design = state_;
  networkState->rootOrdinal = rootOrdinal_;
  networkState->attachments.resize(spec.attachments_.size());

  for (std::size_t ordinal = 0; ordinal != spec.attachments_.size();
       ++ordinal) {
    const MeshCellAttachmentSpec &attachment = spec.attachments_[ordinal];
    const std::size_t cell =
        static_cast<std::size_t>(attachment.y) * spec.width_ + attachment.x;
    cells[cell].attachmentOrdinals.push_back(ordinal);
    auto &state = networkState->attachments[ordinal];
    state.outputBackedges.reserve(attachment.outputTypes.size());
    for (const PortType &type : attachment.outputTypes) {
      auto backedge = createBackedge(type);
      if (!backedge)
        return backedge.takeError();
      state.outputBackedges.push_back(std::move(*backedge));
    }
  }

  constexpr std::array<std::int32_t, 4> dx{0, 1, 0, -1};
  constexpr std::array<std::int32_t, 4> dy{-1, 0, 1, 0};
  for (std::uint32_t y = 0; y != spec.height_; ++y) {
    for (std::uint32_t x = 0; x != spec.width_; ++x) {
      const std::size_t sourceCell =
          static_cast<std::size_t>(y) * spec.width_ + x;
      for (std::size_t direction = 0; direction != dx.size(); ++direction) {
        const std::int64_t destinationX =
            static_cast<std::int64_t>(x) + dx[direction];
        const std::int64_t destinationY =
            static_cast<std::int64_t>(y) + dy[direction];
        if (destinationX < 0 || destinationY < 0 ||
            destinationX >= spec.width_ || destinationY >= spec.height_)
          continue;
        const std::size_t destinationCell =
            static_cast<std::size_t>(destinationY) * spec.width_ +
            static_cast<std::size_t>(destinationX);
        for (std::uint32_t lane = 0; lane != spec.lanesPerDirection_; ++lane) {
          auto incoming = createBackedge(spec.linkType_);
          if (!incoming)
            return incoming.takeError();
          const std::size_t incomingOrdinal =
              cells[destinationCell].incoming.size();
          cells[destinationCell].incoming.push_back(std::move(*incoming));
          cells[sourceCell].outgoing.push_back(
              {destinationCell, incomingOrdinal});
        }
      }
    }
  }

  auto addNetworkSwitch =
      [&](llvm::ArrayRef<SpatialValue> inputs,
          llvm::ArrayRef<PortType> inputTypes,
          llvm::ArrayRef<PortType> outputTypes,
          std::vector<std::vector<std::uint32_t>> connectivity)
      -> llvm::Expected<SwitchResult> {
    llvm::Expected<SwitchResult> result =
        [&]() -> llvm::Expected<SwitchResult> {
      if (spec.schedule_ == ::fabric::Schedule::Spatial)
        return addSwitch(inputs, SwitchSpec::spatial(inputTypes.vec(),
                                                     outputTypes.vec(),
                                                     std::move(connectivity)));
      const bool hasFanIn =
          llvm::any_of(connectivity, [](llvm::ArrayRef<std::uint32_t> sources) {
            return sources.size() > 1;
          });
      std::optional<::fabric::TemporalSwitchGrantPolicy> policy;
      if (hasFanIn) {
        std::vector<std::uint32_t> requesters(inputs.size());
        std::iota(requesters.begin(), requesters.end(), 0);
        if (*spec.grantPolicyKind_ == MeshSwitchGrantPolicyKind::FixedPriority)
          policy = ::fabric::TemporalSwitchFixedPriority{std::move(requesters)};
        else
          policy = ::fabric::TemporalSwitchRoundRobin{std::move(requesters), 0};
      }
      return addSwitch(inputs, SwitchSpec::temporal(
                                   inputTypes.vec(), outputTypes.vec(),
                                   std::move(connectivity),
                                   *spec.routeTableSize_, std::move(policy)));
    }();
    if (!result)
      return result.takeError();
    networkState->domainMembers.push_back(result->domainMember());
    return std::move(*result);
  };

  for (std::size_t cellOrdinal = 0; cellOrdinal != cells.size();
       ++cellOrdinal) {
    MeshCellBuildState &cell = cells[cellOrdinal];
    const std::size_t degree = cell.incoming.size();
    llvm::SmallVector<std::size_t, 7> inputBanks;
    llvm::SmallVector<std::size_t, 7> outputBanks;
    for (std::size_t attachmentOrdinal : cell.attachmentOrdinals) {
      const MeshCellAttachmentSpec &attachment =
          spec.attachments_[attachmentOrdinal];
      if (!attachment.inputTypes.empty())
        inputBanks.push_back(attachmentOrdinal);
      if (!attachment.outputTypes.empty())
        outputBanks.push_back(attachmentOrdinal);
    }

    std::vector<SpatialValue> transitInputs;
    transitInputs.reserve(degree);
    std::vector<std::vector<SpatialValue>> neighborEjectionInputs(
        inputBanks.size());
    for (SpatialBackedge &incoming : cell.incoming) {
      if (inputBanks.empty()) {
        transitInputs.push_back(incoming.value());
        continue;
      }
      std::vector<PortType> outputTypes(1 + inputBanks.size(), spec.linkType_);
      auto fanout =
          addNetworkSwitch({incoming.value()}, {spec.linkType_}, outputTypes,
                           fanoutConnectivity(outputTypes.size()));
      if (!fanout)
        return fanout.takeError();
      transitInputs.push_back(fanout->front());
      for (std::size_t ordinal = 0; ordinal != inputBanks.size(); ++ordinal)
        neighborEjectionInputs[ordinal].push_back((*fanout)[ordinal + 1]);
    }

    std::vector<PortType> transitTypes(degree, spec.linkType_);
    auto transit = addNetworkSwitch(transitInputs, transitTypes, transitTypes,
                                    completeConnectivity(degree, degree));
    if (!transit)
      return transit.takeError();

    std::vector<std::vector<std::size_t>> localInputGroups;
    std::size_t groupedInputPortCount = 0;
    for (std::size_t inputBank = 0; inputBank != inputBanks.size();
         ++inputBank) {
      const std::size_t inputCount =
          spec.attachments_[inputBanks[inputBank]].inputTypes.size();
      if (localInputGroups.empty() || inputCount > 8 - groupedInputPortCount) {
        localInputGroups.emplace_back();
        groupedInputPortCount = 0;
      }
      localInputGroups.back().push_back(inputBank);
      groupedInputPortCount += inputCount;
    }

    std::vector<std::vector<std::vector<SpatialValue>>> localEjectionInputs(
        inputBanks.size());
    for (std::size_t inputBank = 0; inputBank != inputBanks.size(); ++inputBank)
      localEjectionInputs[inputBank].resize(
          spec.attachments_[inputBanks[inputBank]].inputTypes.size());

    std::vector<std::vector<SpatialValue>> injectionOutputs(degree);
    for (std::size_t attachmentOrdinal : outputBanks) {
      const MeshCellAttachmentSpec &attachment =
          spec.attachments_[attachmentOrdinal];
      std::vector<SpatialValue> injectionInputs;
      std::vector<std::vector<SpatialValue>> localGroupInputs(
          localInputGroups.size());
      const auto &backedges =
          networkState->attachments[attachmentOrdinal].outputBackedges;
      for (std::size_t input = 0; input != backedges.size(); ++input) {
        if (localInputGroups.empty()) {
          injectionInputs.push_back(backedges[input].value());
          continue;
        }
        std::vector<PortType> fanoutTypes(1 + localInputGroups.size(),
                                          attachment.outputTypes[input]);
        auto fanout = addNetworkSwitch(
            {backedges[input].value()}, {attachment.outputTypes[input]},
            fanoutTypes, fanoutConnectivity(fanoutTypes.size()));
        if (!fanout)
          return fanout.takeError();
        injectionInputs.push_back(fanout->front());
        for (std::size_t group = 0; group != localInputGroups.size(); ++group)
          localGroupInputs[group].push_back((*fanout)[group + 1]);
      }
      auto injection = addNetworkSwitch(
          injectionInputs, attachment.outputTypes, transitTypes,
          completeConnectivity(attachment.outputTypes.size(), degree));
      if (!injection)
        return injection.takeError();
      for (std::size_t ordinal = 0; ordinal != degree; ++ordinal)
        injectionOutputs[ordinal].push_back((*injection)[ordinal]);

      for (std::size_t group = 0; group != localInputGroups.size(); ++group) {
        std::vector<PortType> localOutputTypes;
        for (std::size_t inputBank : localInputGroups[group]) {
          const auto &types =
              spec.attachments_[inputBanks[inputBank]].inputTypes;
          localOutputTypes.insert(localOutputTypes.end(), types.begin(),
                                  types.end());
        }
        auto local = addNetworkSwitch(
            localGroupInputs[group], attachment.outputTypes, localOutputTypes,
            completeConnectivity(attachment.outputTypes.size(),
                                 localOutputTypes.size()));
        if (!local)
          return local.takeError();
        std::size_t localOutput = 0;
        for (std::size_t inputBank : localInputGroups[group]) {
          for (std::size_t port = 0;
               port !=
               spec.attachments_[inputBanks[inputBank]].inputTypes.size();
               ++port)
            localEjectionInputs[inputBank][port].push_back(
                (*local)[localOutput++]);
        }
      }
    }

    for (std::size_t inputBank = 0; inputBank != inputBanks.size();
         ++inputBank) {
      const std::size_t attachmentOrdinal = inputBanks[inputBank];
      const MeshCellAttachmentSpec &attachment =
          spec.attachments_[attachmentOrdinal];
      auto ejection = addNetworkSwitch(
          neighborEjectionInputs[inputBank], transitTypes,
          attachment.inputTypes,
          completeConnectivity(degree, attachment.inputTypes.size()));
      if (!ejection)
        return ejection.takeError();
      auto &attachmentInputs =
          networkState->attachments[attachmentOrdinal].inputs;
      attachmentInputs.reserve(attachment.inputTypes.size());
      for (std::size_t port = 0; port != attachment.inputTypes.size(); ++port) {
        std::vector<SpatialValue> mergeInputs{(*ejection)[port]};
        mergeInputs.insert(mergeInputs.end(),
                           localEjectionInputs[inputBank][port].begin(),
                           localEjectionInputs[inputBank][port].end());
        if (mergeInputs.size() == 1) {
          attachmentInputs.push_back(mergeInputs.front());
          continue;
        }
        std::vector<PortType> mergeTypes(mergeInputs.size(),
                                         attachment.inputTypes[port]);
        auto merge = addNetworkSwitch(
            mergeInputs, mergeTypes, {attachment.inputTypes[port]},
            completeConnectivity(mergeInputs.size(), 1));
        if (!merge)
          return merge.takeError();
        attachmentInputs.push_back(merge->front());
      }
    }

    cell.outgoingSources.reserve(degree);
    for (std::size_t ordinal = 0; ordinal != degree; ++ordinal) {
      if (injectionOutputs[ordinal].empty()) {
        cell.outgoingSources.push_back((*transit)[ordinal]);
        continue;
      }
      std::vector<SpatialValue> mergeInputs{(*transit)[ordinal]};
      mergeInputs.insert(mergeInputs.end(), injectionOutputs[ordinal].begin(),
                         injectionOutputs[ordinal].end());
      std::vector<PortType> mergeTypes(mergeInputs.size(), spec.linkType_);
      auto merge =
          addNetworkSwitch(mergeInputs, mergeTypes, {spec.linkType_},
                           completeConnectivity(mergeInputs.size(), 1));
      if (!merge)
        return merge.takeError();
      cell.outgoingSources.push_back(merge->front());
    }
  }

  // The queue discipline applies only to tag-carrying link FIFOs; untagged
  // link FIFOs always remain strict.
  std::optional<::fabric::FifoQueueDiscipline> linkFifoDiscipline;
  if (spec.linkType_.kind() == PortType::Kind::TaggedBits &&
      spec.interconnectFifoQueueDiscipline_ ==
          ::fabric::FifoQueueDiscipline::PerTagVirtualChannel)
    linkFifoDiscipline = spec.interconnectFifoQueueDiscipline_;
  for (MeshCellBuildState &cell : cells) {
    if (cell.outgoing.size() != cell.outgoingSources.size())
      return detail::invalid("mesh link construction lost an outgoing lane");
    for (std::size_t ordinal = 0; ordinal != cell.outgoing.size(); ++ordinal) {
      auto fifo =
          addFifo(cell.outgoingSources[ordinal],
                  FifoSpec{spec.linkType_, spec.interconnectFifoDepth_, false,
                           linkFifoDiscipline});
      if (!fifo)
        return fifo.takeError();
      networkState->domainMembers.push_back(fifo->domainMember());
      const MeshCellBuildState::OutgoingLink target = cell.outgoing[ordinal];
      if (llvm::Error error = resolveBackedge(
              std::move(cells[target.destinationCell]
                            .incoming[target.destinationIncomingOrdinal]),
              fifo->value()))
        return error;
    }
  }

  return MeshSwitchNetwork(std::move(networkState));
}

} // namespace loom::adg

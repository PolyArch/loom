#include "SpatialFixedTerminalCutConstraint.h"

#include "llvm/Support/Error.h"

#include <array>
#include <optional>
#include <system_error>
#include <vector>

using namespace loom::pnr;
using namespace loom::pnr::detail;
using namespace operations_research::sat;

namespace {

llvm::Error cutConstraintError(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invalid Spatial fixed-terminal cut constraint: %s",
      detail.str().c_str());
}

} // namespace

llvm::Expected<SpatialFixedTerminalCutConstraintResult>
loom::pnr::detail::addSpatialFixedTerminalCutEscapeConstraint(
    CpModelBuilder &model, const SpatialCandidateState &candidate,
    const SpatialBindingRelationModel &bindings,
    llvm::ArrayRef<IntVar> variables, llvm::ArrayRef<int> decisionVariables,
    llvm::ArrayRef<PnrIndex> legalValueOffsets,
    llvm::ArrayRef<std::int64_t> legalValues,
    const SpatialFixedTerminalCutCertificate &certificate,
    std::vector<std::uint8_t> &blockedTraversals_,
    std::vector<std::uint8_t> &reachableEndpoints_,
    std::vector<PnrIndex> &worklist_) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const FrozenSpatialRoutingGraph &routing = problem.routing();
  const FrozenSpatialPortIndex &ports = problem.ports();
  const FrozenSpatialTransferIndex &transfers = problem.transfers();
  const PnrIndex capacity = certificate.capacity;
  const llvm::ArrayRef<SpatialFixedTerminalCutNet> cuts =
      certificate.forcedNetCuts;
  if (capacity >= problem.resources().capacityDimensions().size())
    return cutConstraintError("capacity is out of range");
  if (cuts.empty())
    return cutConstraintError("certificate has no forced net");

  blockedTraversals_.assign(routing.traversals().size(), 0);
  const auto capacityClaimOffsets = routing.capacityRouteClaimOffsets();
  const auto capacityClaims = routing.capacityRouteClaims();
  const auto claimTraversalOffsets = routing.routeClaimTraversalOffsets();
  const auto claimTraversals = routing.routeClaimTraversals();
  if (capacity + 1 >= capacityClaimOffsets.size())
    return cutConstraintError("capacity-to-claim incidence is incomplete");
  for (PnrIndex entry = capacityClaimOffsets[capacity];
       entry < capacityClaimOffsets[capacity + 1]; ++entry) {
    if (entry >= capacityClaims.size())
      return cutConstraintError("capacity claim is out of range");
    const PnrIndex claim = capacityClaims[entry];
    if (claim >= routing.routeClaims().size() ||
        claim + 1 >= claimTraversalOffsets.size())
      return cutConstraintError("claim-to-traversal incidence is malformed");
    if (routing.routeClaims()[claim].amount == 0)
      continue;
    for (PnrIndex traversalEntry = claimTraversalOffsets[claim];
         traversalEntry < claimTraversalOffsets[claim + 1]; ++traversalEntry) {
      if (traversalEntry >= claimTraversals.size() ||
          claimTraversals[traversalEntry] >= blockedTraversals_.size())
        return cutConstraintError("claim traversal is out of range");
      blockedTraversals_[claimTraversals[traversalEntry]] = 1;
    }
  }

  const auto terminalLocal = [&](FrozenSpatialTerminalBinding binding)
      -> llvm::Expected<std::optional<PnrIndex>> {
    if (binding.kind == FrozenSpatialTerminalBindingKind::PortDemand &&
        binding.index >= ports.portDemands().size())
      return cutConstraintError("PortDemand is out of range");
    if (binding.kind == FrozenSpatialTerminalBindingKind::GraphBoundary &&
        binding.index >= ports.graphBoundaries().size())
      return cutConstraintError("graph boundary is out of range");
    const PnrIndex decision =
        binding.kind == FrozenSpatialTerminalBindingKind::PortDemand
            ? bindings.portDecisionOffset() + binding.index
            : bindings.graphBoundaryDecisionOffset() + binding.index;
    if (decision >= decisionVariables.size())
      return cutConstraintError("terminal decision is out of range");
    const int local = decisionVariables[decision];
    if (local < 0)
      return std::optional<PnrIndex>();
    if (static_cast<std::size_t>(local) >= variables.size())
      return cutConstraintError("terminal variable is out of range");
    return std::optional<PnrIndex>(static_cast<PnrIndex>(local));
  };
  const auto terminalChoices =
      [&](FrozenSpatialTerminalBinding binding) -> llvm::ArrayRef<PnrIndex> {
    return binding.kind == FrozenSpatialTerminalBindingKind::PortDemand
               ? bindings.portAttachmentChoices(binding.index)
               : bindings.graphBoundaryAttachmentChoices(binding.index);
  };
  const auto endpointForChoice =
      [&](llvm::ArrayRef<PnrIndex> choices,
          std::int64_t choice) -> llvm::Expected<PnrIndex> {
    if (choice < 0 || static_cast<std::size_t>(choice) >= choices.size())
      return cutConstraintError("attachment choice is out of range");
    const PnrIndex option = choices[choice];
    if (option >= ports.attachmentOptions().size())
      return cutConstraintError("attachment option is out of range");
    return ports.attachmentOptions()[option].endpoint;
  };
  const auto legalDomain =
      [&](PnrIndex local) -> llvm::Expected<llvm::ArrayRef<std::int64_t>> {
    if (local + 1 >= legalValueOffsets.size() ||
        legalValueOffsets[local] > legalValueOffsets[local + 1] ||
        legalValueOffsets[local + 1] > legalValues.size())
      return cutConstraintError("legal-value domain is malformed");
    return legalValues.slice(legalValueOffsets[local],
                             legalValueOffsets[local + 1] -
                                 legalValueOffsets[local]);
  };

  const auto arcs = routing.routingArcs();
  const auto adjacencyOffsets = routing.adjacencyOffsets();
  const auto markReachable = [&](PnrIndex source,
                                 std::uint32_t payloadWidth) -> llvm::Error {
    if (source >= routing.routingEndpoints().size())
      return cutConstraintError("source endpoint is out of range");
    reachableEndpoints_.assign(routing.routingEndpoints().size(), 0);
    worklist_.clear();
    reachableEndpoints_[source] = 1;
    worklist_.push_back(source);
    for (std::size_t cursor = 0; cursor < worklist_.size(); ++cursor) {
      const PnrIndex endpoint = worklist_[cursor];
      if (endpoint + 1 >= adjacencyOffsets.size())
        return cutConstraintError("routing adjacency is malformed");
      for (PnrIndex arc = adjacencyOffsets[endpoint];
           arc < adjacencyOffsets[endpoint + 1]; ++arc) {
        if (arc >= arcs.size())
          return cutConstraintError("routing arc is out of range");
        const EndpointRoutingArc &record = arcs[arc];
        if (record.traversal >= blockedTraversals_.size() ||
            record.target >= reachableEndpoints_.size())
          return cutConstraintError("routing topology is malformed");
        if (blockedTraversals_[record.traversal] ||
            record.payloadCapacityBits < payloadWidth ||
            reachableEndpoints_[record.target])
          continue;
        reachableEndpoints_[record.target] = 1;
        worklist_.push_back(record.target);
      }
    }
    return llvm::Error::success();
  };

  std::vector<BoolVar> escapedCuts;
  escapedCuts.reserve(cuts.size());
  bool currentAssignmentEscapes = false;
  for (const SpatialFixedTerminalCutNet &cut : cuts) {
    if (cut.logicalNet >= transfers.logicalNets().size())
      return cutConstraintError("logical net is out of range");
    const FrozenSpatialLogicalNet &net =
        transfers.logicalNets()[cut.logicalNet];
    if (cut.unreachableSink >= net.sinkCount)
      return cutConstraintError("sink is out of range");
    const FrozenSpatialTerminalBinding sourceBinding =
        transfers.logicalNetSourceBindings()[cut.logicalNet];
    const FrozenSpatialTerminalBinding sinkBinding =
        transfers
            .logicalNetSinkBindings()[net.sinkOffset + cut.unreachableSink];
    auto sourceLocal = terminalLocal(sourceBinding);
    if (!sourceLocal)
      return sourceLocal.takeError();
    auto sinkLocal = terminalLocal(sinkBinding);
    if (!sinkLocal)
      return sinkLocal.takeError();
    if (!*sourceLocal || !*sinkLocal)
      return SpatialFixedTerminalCutConstraintResult{};
    auto sourceValues = legalDomain(**sourceLocal);
    if (!sourceValues)
      return sourceValues.takeError();
    auto sinkValues = legalDomain(**sinkLocal);
    if (!sinkValues)
      return sinkValues.takeError();
    const auto sourceChoices = terminalChoices(sourceBinding);
    const auto sinkChoices = terminalChoices(sinkBinding);
    const std::uint32_t payloadWidth =
        candidate.logicalNetPayloadWidth(cut.logicalNet);
    if (llvm::Error error = markReachable(
            candidate.logicalNetSourceEndpoint(cut.logicalNet), payloadWidth))
      return std::move(error);
    const PnrIndex currentSink =
        candidate.logicalNetSinkEndpoint(cut.logicalNet, cut.unreachableSink);
    if (currentSink >= reachableEndpoints_.size())
      return cutConstraintError("current sink endpoint is out of range");
    currentAssignmentEscapes |= reachableEndpoints_[currentSink] != 0;
    const BoolVar escaped = model.NewBoolVar();
    escapedCuts.push_back(escaped);

    if (**sourceLocal == **sinkLocal) {
      TableConstraint table = model.AddAllowedAssignments(
          {variables[**sourceLocal], IntVar(escaped)});
      for (std::int64_t sourceChoice : *sourceValues) {
        auto source = endpointForChoice(sourceChoices, sourceChoice);
        if (!source)
          return source.takeError();
        auto sink = endpointForChoice(sinkChoices, sourceChoice);
        if (!sink)
          return sink.takeError();
        if (llvm::Error error = markReachable(*source, payloadWidth))
          return std::move(error);
        if (*sink >= reachableEndpoints_.size())
          return cutConstraintError("sink endpoint is out of range");
        const std::array<std::int64_t, 2> tuple{
            sourceChoice, reachableEndpoints_[*sink] ? 1 : 0};
        table.AddTuple(tuple);
      }
      continue;
    }

    TableConstraint table = model.AddAllowedAssignments(
        {variables[**sourceLocal], variables[**sinkLocal], IntVar(escaped)});
    for (std::int64_t sourceChoice : *sourceValues) {
      auto source = endpointForChoice(sourceChoices, sourceChoice);
      if (!source)
        return source.takeError();
      if (llvm::Error error = markReachable(*source, payloadWidth))
        return std::move(error);
      for (std::int64_t sinkChoice : *sinkValues) {
        auto sink = endpointForChoice(sinkChoices, sinkChoice);
        if (!sink)
          return sink.takeError();
        if (*sink >= reachableEndpoints_.size())
          return cutConstraintError("sink endpoint is out of range");
        const std::array<std::int64_t, 3> tuple{
            sourceChoice, sinkChoice, reachableEndpoints_[*sink] ? 1 : 0};
        table.AddTuple(tuple);
      }
    }
  }
  model.AddAtLeastOne(escapedCuts);
  return SpatialFixedTerminalCutConstraintResult{true,
                                                 currentAssignmentEscapes};
}

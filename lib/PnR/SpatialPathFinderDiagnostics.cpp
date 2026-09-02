#include "SpatialPathFinderRouterInternal.h"

#include "Fabric/Identity/FabricRefText.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <type_traits>
#include <utility>

using namespace loom::pnr;

char SpatialPathFinderClosureFailure::ID;

SpatialPathFinderClosureFailure::SpatialPathFinderClosureFailure(
    Kind kind, std::string message,
    SpatialFixedTerminalCutCertificate certificate,
    std::uint64_t mandatoryUsage, std::uint64_t physicalCapacity,
    std::uint64_t regionalLogicalNetCount,
    std::uint64_t regionalLogicalNetLimit)
    : kind_(kind), message_(std::move(message)),
      certificate_(std::move(certificate)), mandatoryUsage_(mandatoryUsage),
      physicalCapacity_(physicalCapacity),
      regionalLogicalNetCount_(regionalLogicalNetCount),
      regionalLogicalNetLimit_(regionalLogicalNetLimit) {}

void SpatialPathFinderClosureFailure::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code SpatialPathFinderClosureFailure::convertToErrorCode() const {
  switch (kind_) {
  case Kind::NonClosure:
  case Kind::NoProgress:
    return std::make_error_code(std::errc::resource_unavailable_try_again);
  case Kind::RegionalLimit:
    return std::make_error_code(std::errc::value_too_large);
  case Kind::FixedTerminalCapacityCut:
    return std::make_error_code(std::errc::address_not_available);
  case Kind::SelectedCombinationalHandshakeCycle:
    return std::make_error_code(std::errc::state_not_recoverable);
  case Kind::Interrupted:
    return std::make_error_code(std::errc::operation_canceled);
  }
  llvm_unreachable("invalid Spatial PathFinder closure failure kind");
}

namespace loom::pnr::detail {
namespace {

llvm::json::Object encodeProducerReference(
    const dataflow::CanonicalGraphProducerEndpointRef &producer) {
  llvm::json::Object result;
  std::visit(
      [&](const auto &endpoint) {
        using Endpoint = std::decay_t<decltype(endpoint)>;
        if constexpr (std::is_same_v<Endpoint, dataflow::ActorTokenResultRef>) {
          result["kind"] = "actor_result";
          result["owner"] = endpoint.actor.entity.value();
          result["ordinal"] = endpoint.ordinal;
        } else {
          std::visit(
              [&](const auto &ingress) {
                using Ingress = std::decay_t<decltype(ingress)>;
                result["owner"] = ingress.graph.entity.value();
                if constexpr (std::is_same_v<Ingress,
                                             dataflow::GraphStartTokenRef>) {
                  result["kind"] = "graph_start";
                } else if constexpr (std::is_same_v<
                                         Ingress,
                                         dataflow::GraphValueInputTokenRef>) {
                  result["kind"] = "graph_value_input";
                  result["ordinal"] = ingress.ordinal;
                } else {
                  result["kind"] = "graph_stream_input";
                  result["ordinal"] = ingress.ordinal;
                }
              },
              endpoint);
        }
      },
      producer);
  return result;
}

} // namespace

llvm::Error pathFinderError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      ("invalid Spatial PathFinder route: " + message).str(),
      std::make_error_code(std::errc::invalid_argument));
}

std::string errorMessage(const llvm::ErrorInfoBase &error) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  error.log(stream);
  return message;
}

llvm::Error classifyIterationFailure(llvm::Error failure, bool &completed) {
  completed = false;
  return llvm::handleErrors(
      std::move(failure),
      [&](std::unique_ptr<EndpointRouteSearchFailure> typed) -> llvm::Error {
        completed = typed->kind() != EndpointRouteSearchFailureKind::Invalid;
        return llvm::Error(std::move(typed));
      },
      [&](std::unique_ptr<RoutingNegotiationError> typed) -> llvm::Error {
        completed =
            typed->kind() == RoutingNegotiationError::Kind::ArithmeticOverflow;
        return llvm::Error(std::move(typed));
      },
      [&](std::unique_ptr<SpatialPathFinderClosureFailure> typed)
          -> llvm::Error {
        completed = typed->kind() !=
                    SpatialPathFinderClosureFailure::Kind::Interrupted;
        return llvm::Error(std::move(typed));
      });
}

std::optional<PnrIndex>
resourceStateForCapacity(const FrozenSpatialResourceIndex &resources,
                         PnrIndex capacity) {
  for (auto [state, record] : llvm::enumerate(resources.resourceStates()))
    if (capacity >= record.capacityOffset &&
        capacity - record.capacityOffset < record.capacityCount)
      return static_cast<PnrIndex>(state);
  return std::nullopt;
}

std::optional<PnrIndex>
resourceOwnerForState(const FrozenSpatialResourceIndex &resources,
                      PnrIndex state) {
  for (auto [owner, record] : llvm::enumerate(resources.resourceOwners()))
    if (state >= record.stateOffset &&
        state - record.stateOffset < record.stateCount)
      return static_cast<PnrIndex>(owner);
  return std::nullopt;
}

llvm::json::Object
encodeLogicalNetDetail(const SpatialCandidateState &candidate,
                       PnrIndex logicalNet) {
  const FrozenSpatialPnrProblem &problem = candidate.problem();
  const auto &routing = problem.routing();
  const FrozenSpatialLogicalNet &net =
      problem.transfers().logicalNets()[logicalNet];
  const PnrIndex source = candidate.logicalNetSourceEndpoint(logicalNet);
  llvm::json::Object result;
  result["logical_net"] = logicalNet;
  result["producer"] = encodeProducerReference(net.producer);
  result["source_endpoint"] = source;
  result["source_endpoint_ref"] = loom::fabric::printFabricRef(
      routing.routingEndpoints()[source].reference);
  llvm::json::Array sinks;
  const auto progressOffsets =
      problem.transfers().sinkProgressDependencyOffsets();
  const auto progressDependencies =
      problem.transfers().sinkProgressDependencies();
  for (PnrIndex sink = 0; sink < net.sinkCount; ++sink) {
    const PnrIndex endpoint =
        candidate.logicalNetSinkEndpoint(logicalNet, sink);
    llvm::json::Object row;
    row["endpoint"] = endpoint;
    row["endpoint_ref"] = loom::fabric::printFabricRef(
        routing.routingEndpoints()[endpoint].reference);
    row["sink"] = sink;
    const PnrIndex globalSink = net.sinkOffset + sink;
    llvm::json::Array prerequisites;
    for (const FrozenSpatialProgressPrerequisite &prerequisite :
         progressDependencies.slice(progressOffsets[globalSink],
                                    progressOffsets[globalSink + 1] -
                                        progressOffsets[globalSink])) {
      llvm::json::Object encoded;
      std::visit(
          [&](const auto &typed) {
            using T = std::decay_t<decltype(typed)>;
            if constexpr (std::is_same_v<
                              T, FrozenSpatialExternalSinkPrerequisite>) {
              encoded["kind"] = "external_sink";
              encoded["sink"] = typed.sink;
            } else if constexpr (
                std::is_same_v<
                    T, FrozenSpatialInternalMemoryConnectionPrerequisite>) {
              encoded["kind"] = "internal_memory_connection";
              encoded["memory_realization"] = typed.memoryRealization;
              encoded["internal_edge"] = typed.internalEdge;
            } else {
              encoded["kind"] = "initialized_feedback";
            }
          },
          prerequisite);
      prerequisites.push_back(std::move(encoded));
    }
    row["progress_prerequisites"] = std::move(prerequisites);
    sinks.push_back(std::move(row));
  }
  result["sinks"] = std::move(sinks);
  return result;
}

llvm::json::Array
encodeSelectedOrdinalRanges(llvm::ArrayRef<std::uint8_t> selected) {
  llvm::json::Array ranges;
  for (std::size_t begin = 0; begin < selected.size();) {
    while (begin < selected.size() && !selected[begin])
      ++begin;
    if (begin == selected.size())
      break;
    std::size_t end = begin + 1;
    while (end < selected.size() && selected[end])
      ++end;
    llvm::json::Object range;
    range["begin"] = static_cast<std::uint64_t>(begin);
    range["end"] = static_cast<std::uint64_t>(end);
    ranges.push_back(std::move(range));
    begin = end;
  }
  return ranges;
}

} // namespace loom::pnr::detail

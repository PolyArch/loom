#include "SpatialEngineSession.h"

#include "Common/ArtifactText.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Runtime/OrderedChannelABI.h"
#include "Simulator/SpatialChannelWire.h"
#include "Simulator/SpatialInvocation.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <time.h>
#include <variant>

namespace loom::gem5engine {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "gem5_spatial_engine_invalid: " + message);
}

namespace {

struct ChannelSequenceState final {
  loom::runtime::OrderedChannelABI abi;
  std::map<std::string, std::uint32_t> consumerOrdinals;
};

struct PendingChannelPublication final {
  std::uint64_t channelOrdinal = 0;
  std::vector<std::vector<std::uint8_t>> payloads;
  std::size_t nextPayload = 0;
};

llvm::Expected<std::uint64_t>
spatialCoordinateTicks(const loom::sim::SpatialEventCoordinate &coordinate,
                       std::uint64_t ticksPerCycle) {
  using u128 = unsigned __int128;
  const u128 scaled = static_cast<u128>(coordinate.referenceCycle.numerator()) *
                      static_cast<std::uint64_t>(ticksPerCycle);
  const std::uint64_t denominator = coordinate.referenceCycle.denominator();
  if (scaled % denominator != 0 ||
      scaled / denominator > std::numeric_limits<std::uint64_t>::max())
    return invalid("Spatial service coordinate is not an integral gem5 tick");
  return static_cast<std::uint64_t>(scaled / denominator);
}

llvm::Expected<std::uint64_t> spatialServiceDelay(
    const loom::sim::SpatialEventCoordinate &coordinate,
    std::uint64_t ticksPerCycle,
    const std::optional<loom::sim::SpatialEventCoordinate> &servicedThrough) {
  if (servicedThrough && loom::sim::compareSpatialEventCoordinates(
                             *servicedThrough, coordinate) > 0)
    return invalid("Spatial external service coordinates moved backward");
  auto ready = spatialCoordinateTicks(coordinate, ticksPerCycle);
  if (!ready)
    return ready.takeError();
  std::uint64_t prior = 0;
  if (servicedThrough) {
    auto projected = spatialCoordinateTicks(*servicedThrough, ticksPerCycle);
    if (!projected)
      return projected.takeError();
    prior = *projected;
  }
  return *ready - prior;
}

class Gem5CgraExternalMemoryProvider final
    : public loom::sim::CgraExternalMemoryProvider {
public:
  Gem5CgraExternalMemoryProvider(
      const loom::runtime::SpatialInvocationWire &invocation,
      std::uint64_t ticksPerCycle,
      std::optional<loom::sim::SpatialEventCoordinate> &servicedThrough)
      : invocation_(&invocation), ticksPerCycle_(ticksPerCycle),
        servicedThrough_(&servicedThrough) {}

  llvm::Expected<loom::sim::CgraExternalMemorySubmission>
  submit(const loom::sim::CgraExternalMemoryRequest &request) override {
    if (pending_)
      return invalid(
          "CGRA submitted memory while its prior request is pending");
    if (request.elements.empty() ||
        request.objectOrdinal >= invocation_->memoryObjects.size())
      return invalid("CGRA external memory request names no guest elements");
    auto delay = spatialServiceDelay(request.readyCoordinate, ticksPerCycle_,
                                      *servicedThrough_);
    if (!delay)
      return delay.takeError();
    const auto &object = invocation_->memoryObjects[request.objectOrdinal];
    const bool write =
        request.operation == loom::sim::CgraExternalMemoryOperation::Write;
    for (const auto &element : request.elements) {
      if (element.byteCount == 0 ||
          element.byteOffset > object.initialBytes.size() ||
          element.byteCount > object.initialBytes.size() - element.byteOffset ||
          object.address >
              std::numeric_limits<std::uint64_t>::max() - element.byteOffset)
        return invalid("CGRA external memory element exceeds its guest object");
      if ((write && element.writeData.size() != element.byteCount) ||
          (!write && !element.writeData.empty()))
        return invalid("CGRA external memory element has the wrong payload");
    }
    pending_.emplace(Pending{request, *delay, 0, {}});
    return loom::sim::CgraExternalMemoryPending{};
  }

  llvm::Expected<loom::runtime::Gem5BridgeMemoryRequest>
  nextElement(std::uint64_t requestId) const {
    if (!pending_ ||
        pending_->elementOrdinal >= pending_->request.elements.size())
      return invalid("CGRA external memory continuation has no next element");
    const auto &element = pending_->request.elements[pending_->elementOrdinal];
    const auto &object =
        invocation_->memoryObjects[pending_->request.objectOrdinal];
    return loom::runtime::Gem5BridgeMemoryRequest{
        pending_->request.operation ==
                loom::sim::CgraExternalMemoryOperation::Write
            ? loom::runtime::Gem5BridgeMemoryOperation::Write
            : loom::runtime::Gem5BridgeMemoryOperation::Read,
        pending_->elementOrdinal == 0 ? pending_->initialDelay : 0,
        requestId,
        object.address + element.byteOffset,
        element.byteCount,
        element.writeData};
  }

  llvm::Expected<bool>
  completeElement(loom::runtime::Gem5BridgeMemoryResponse response,
                  loom::sim::CgraExecutionSession &session) {
    if (!pending_ ||
        pending_->elementOrdinal >= pending_->request.elements.size())
      return invalid("CGRA external memory response has no pending element");
    const auto &element = pending_->request.elements[pending_->elementOrdinal];
    const auto &object =
        invocation_->memoryObjects[pending_->request.objectOrdinal];
    if (pending_->request.operation ==
        loom::sim::CgraExternalMemoryOperation::Write) {
      for (std::size_t byte = 0; byte != element.writeData.size(); ++byte)
        externallyCommittedBytes_[object.address + element.byteOffset + byte] =
            element.writeData[byte];
    } else {
      pending_->response.readData.push_back(std::move(response.data));
    }
    ++pending_->elementOrdinal;
    if (pending_->elementOrdinal != pending_->request.elements.size())
      return false;
    if (llvm::Error error = session.completeExternalMemory(
            pending_->request.id, std::move(pending_->response)))
      return std::move(error);
    *servicedThrough_ = pending_->request.readyCoordinate;
    pending_.reset();
    return true;
  }

  std::vector<loom::sim::SpatialInvocationMemoryWrite> retainUncommittedWrites(
      llvm::ArrayRef<loom::sim::SpatialInvocationMemoryWrite> writes) const {
    struct Interval final {
      std::uint64_t begin = 0;
      std::uint64_t end = 0;
    };
    std::vector<Interval> resultDestinations;
    resultDestinations.reserve(invocation_->results.size());
    for (const auto &destination : invocation_->results) {
      const std::uint64_t byteCount =
          (static_cast<std::uint64_t>(destination.bitCount) + 7) / 8;
      resultDestinations.push_back(
          {destination.address, destination.address + byteCount});
    }
    const auto isResultDestination = [&](std::uint64_t address) {
      return llvm::any_of(resultDestinations, [&](const Interval &interval) {
        return interval.begin <= address && address < interval.end;
      });
    };

    std::vector<loom::sim::SpatialInvocationMemoryWrite> retained;
    for (const loom::sim::SpatialInvocationMemoryWrite &write : writes) {
      std::optional<loom::sim::SpatialInvocationMemoryWrite> run;
      for (std::size_t ordinal = 0; ordinal != write.bytes.size(); ++ordinal) {
        const std::uint64_t address = write.address + ordinal;
        auto committed = externallyCommittedBytes_.find(address);
        const bool keep = isResultDestination(address) ||
                          committed == externallyCommittedBytes_.end() ||
                          committed->second != write.bytes[ordinal];
        if (!keep) {
          if (run) {
            retained.push_back(std::move(*run));
            run.reset();
          }
          continue;
        }
        if (!run)
          run = loom::sim::SpatialInvocationMemoryWrite{address, {}};
        run->bytes.push_back(write.bytes[ordinal]);
      }
      if (run)
        retained.push_back(std::move(*run));
    }
    return retained;
  }

private:
  struct Pending final {
    loom::sim::CgraExternalMemoryRequest request;
    std::uint64_t initialDelay;
    std::size_t elementOrdinal;
    loom::sim::CgraExternalMemoryResponse response;
  };
  const loom::runtime::SpatialInvocationWire *invocation_;
  const std::uint64_t ticksPerCycle_;
  std::optional<Pending> pending_;
  std::optional<loom::sim::SpatialEventCoordinate> *servicedThrough_;
  std::map<std::uint64_t, std::uint8_t> externallyCommittedBytes_;
};
void appendChannelKeyU64(std::string &key, std::uint64_t value) {
  for (unsigned byte = 0; byte != 8; ++byte)
    key.push_back(static_cast<char>(value >> (byte * 8)));
}

llvm::Expected<std::string>
channelConsumerKey(const loom::sim::ImportedSpatialSimulationWorkload &workload,
                   const loom::runtime::Gem5SpatialChannelInput &input) {
  const auto *spatial = workload.workload.spatial();
  if (!spatial)
    return invalid("channel consumer key lost its Spatial workload");
  std::string key;
  appendChannelKeyU64(key, input.channelOrdinal);
  appendChannelKeyU64(key, spatial->launchRef.rootThreadLaunch.entity.value());
  appendChannelKeyU64(key, spatial->launchRef.staticGraphLaunch.entity.value());
  appendChannelKeyU64(key, input.consumerStreamInputOrdinal);
  key.push_back(static_cast<char>(spatial->denseCoordinates.size()));
  for (std::uint64_t coordinate : spatial->denseCoordinates)
    appendChannelKeyU64(key, coordinate);
  return key;
}

llvm::Expected<loom::sim::CanonicalSimulationRuntimeInput>
cloneRuntimeInput(
    const loom::sim::ImportedSpatialSimulationWorkload &workload,
    const loom::sim::CanonicalSimulationRuntimeInput &runtimeInput) {
  const auto &view = workload.dataflow->view();
  return loom::sim::importSimulationRuntimeInput(
      runtimeInput.canonicalBytes().bytes(), workload.workload, view,
      runtimeInput.identity());
}

llvm::Expected<std::vector<PendingChannelPublication>>
prepareChannelPublications(
    const loom::runtime::Gem5SpatialChannelProjection &projection,
    const loom::sim::SpatialEngineBoundaryResult &result,
    const loom::sim::ImportedSpatialSimulationWorkload &workload,
    const loom::sim::CanonicalSimulationRuntimeInput &runtimeInput,
    std::map<std::uint64_t, ChannelSequenceState> &channelSequences) {
  if (projection.outputs.empty())
    return std::vector<PendingChannelPublication>{};
  if (!std::holds_alternative<loom::sim::RetiredExecution>(result.terminal))
    return invalid("channel producer did not retire");
  const auto *spatialWorkload = workload.workload.spatial();
  const auto *spatialRuntime = runtimeInput.spatial();
  if (!spatialWorkload || !spatialRuntime)
    return invalid("Spatial channel owner lost its typed payload");
  const auto &view = workload.dataflow->view();
  std::map<std::uint64_t, std::vector<std::vector<std::uint8_t>>> payloads;
  for (const loom::runtime::Gem5SpatialChannelOutput &channel :
       projection.outputs) {
    const auto found =
        llvm::find(spatialWorkload->observableContract.streamOutputs,
                   channel.producerStreamOutputOrdinal);
    if (found == spatialWorkload->observableContract.streamOutputs.end())
      return invalid("channel producer output is not observable");
    const std::size_t observation = static_cast<std::size_t>(std::distance(
        spatialWorkload->observableContract.streamOutputs.begin(), found));
    if (observation >= result.functionalObservations.streamOutputs.size())
      return invalid("channel producer omitted its selected stream output");
    if (channelSequences.find(channel.channelOrdinal) == channelSequences.end())
      return invalid("channel output has no ordered sequence state");
    const loom::sim::CanonicalStreamSequence &stream =
        result.functionalObservations.streamOutputs[observation];
    // ClosedAfterLast belongs to this graph observation horizon; the Dataflow
    // channel contract has no implicit EOS. Repeated producer launches append
    // to the same SendSeq stream until an explicit channel protocol closes it.
    if (stream.values.tokenCount != 0) {
      if (stream.values.lanes.size() % stream.values.tokenCount != 0)
        return invalid("channel stream lane count is not token aligned");
      const std::size_t lanesPerToken =
          stream.values.lanes.size() / stream.values.tokenCount;
      for (std::uint64_t token = 0; token != stream.values.tokenCount;
           ++token) {
        loom::sim::CanonicalStreamSequence one;
        one.values.tokenCount = 1;
        const auto begin = stream.values.lanes.begin() +
                           static_cast<std::size_t>(token) * lanesPerToken;
        one.values.lanes.assign(begin, begin + lanesPerToken);
        one.termination = token + 1 == stream.values.tokenCount
                              ? stream.termination
                              : loom::sim::StreamTermination::OpenAfterLast;
        auto encodedToken = loom::sim::encodeSpatialChannelStream(
            one, view, spatialWorkload->launchRef,
            channel.producerStreamOutputOrdinal,
            spatialRuntime->memoryObjects.size());
        if (!encodedToken)
          return encodedToken.takeError();
        payloads[channel.channelOrdinal].push_back(std::move(*encodedToken));
      }
    }
  }
  std::vector<PendingChannelPublication> publications;
  publications.reserve(payloads.size());
  for (auto &[channelOrdinal, channelPayloads] : payloads) {
    if (channelPayloads.empty())
      return invalid("channel output produced no message payload");
    publications.push_back({channelOrdinal, std::move(channelPayloads), 0});
  }
  return publications;
}

struct ChannelPublicationProgress final {
  bool complete = false;
  bool advanced = false;
};

llvm::Expected<ChannelPublicationProgress> publishAvailableChannelOutputs(
    std::vector<PendingChannelPublication> &publications,
    std::map<std::uint64_t, ChannelSequenceState> &channelSequences) {
  ChannelPublicationProgress progress{true, false};
  for (PendingChannelPublication &publication : publications) {
    auto state = channelSequences.find(publication.channelOrdinal);
    if (state == channelSequences.end())
      return invalid("pending channel output lost its sequence state");
    while (publication.nextPayload < publication.payloads.size()) {
      const auto sent =
          state->second.abi.send(publication.payloads[publication.nextPayload]);
      if (sent.kind == loom::runtime::OrderedChannelSendKind::WouldBlock)
        break;
      if (sent.kind == loom::runtime::OrderedChannelSendKind::SequenceExhausted)
        return invalid("ordered channel SendSeq is exhausted");
      ++publication.nextPayload;
      progress.advanced = true;
    }
    if (publication.nextPayload != publication.payloads.size())
      progress.complete = false;
  }
  return progress;
}

#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
llvm::Expected<loom::sim::SpatialEngineBoundaryResult>
finishDfg(loom::sim::DfgExecutionSession &session) {
  auto retired = session.takeRetiredSimulation();
  if (!retired)
    return retired.takeError();
  auto zero = loom::evaluation::ExactRatio::get(0, 1);
  auto retirement =
      loom::evaluation::ExactRatio::get(retired->report.wavefrontSteps, 1);
  if (!zero)
    return zero.takeError();
  if (!retirement)
    return retirement.takeError();
  const loom::sim::SpatialEventCoordinate launch{std::move(*zero), 0};
  const loom::sim::SpatialEventCoordinate terminal{std::move(*retirement), 0};
  return loom::sim::SpatialEngineBoundaryResult{
      loom::sim::RetiredExecution{},
      std::move(retired->observations),
      {launch, terminal, terminal},
      {}};
}

#else
struct CgraPerformanceProfile final {
  std::uint64_t invocationCount = 0;
  std::uint64_t activeWallNanoseconds = 0;
  std::uint64_t activeCpuNanoseconds = 0;
  std::uint64_t eventFrameCount = 0;
};

llvm::Expected<std::uint64_t> engineProcessCpuNanoseconds() {
  timespec current{};
  if (::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &current) != 0) {
    const int errorNumber = errno;
    return invalid(llvm::Twine("cannot read engine process CPU clock: ") +
                   std::strerror(errorNumber));
  }
  constexpr std::uint64_t nanosecondsPerSecond = 1'000'000'000;
  if (current.tv_sec < 0 || current.tv_nsec < 0 ||
      static_cast<std::uint64_t>(current.tv_nsec) >= nanosecondsPerSecond ||
      static_cast<std::uint64_t>(current.tv_sec) >
          (std::numeric_limits<std::uint64_t>::max() -
           static_cast<std::uint64_t>(current.tv_nsec)) /
              nanosecondsPerSecond)
    return invalid("engine process CPU clock is outside the profile domain");
  return static_cast<std::uint64_t>(current.tv_sec) * nanosecondsPerSecond +
         static_cast<std::uint64_t>(current.tv_nsec);
}

llvm::Error addProfileValue(std::uint64_t &total, std::uint64_t value,
                            llvm::StringRef field) {
  if (value > std::numeric_limits<std::uint64_t>::max() - total)
    return invalid(llvm::Twine("CGRA performance profile overflows '") + field +
                   "'");
  total += value;
  return llvm::Error::success();
}

llvm::Error writeCgraPerformanceProfile(llvm::StringRef path,
                                        const CgraPerformanceProfile &profile) {
  std::error_code openError;
  llvm::raw_fd_ostream output(path, openError, llvm::sys::fs::OF_Text);
  if (openError)
    return invalid(llvm::Twine("cannot open CGRA performance profile '") +
                   path + "': " + openError.message());
  {
    llvm::json::OStream json(output);
    json.object([&] {
      json.attribute("schema", "loom.gem5_spatial_engine_performance.1");
      json.attribute("engine", "cgra");
      json.attribute("invocation_count", profile.invocationCount);
      json.attribute("active_wall_nanoseconds", profile.activeWallNanoseconds);
      json.attribute("active_cpu_nanoseconds", profile.activeCpuNanoseconds);
      json.attribute("event_frame_count", profile.eventFrameCount);
    });
  }
  output << '\n';
  output.close();
  if (std::error_code writeError = output.error()) {
    output.clear_error();
    return invalid(llvm::Twine("cannot write CGRA performance profile '") +
                   path + "': " + writeError.message());
  }
  return llvm::Error::success();
}

llvm::Error
recordCgraPerformanceProfile(CgraPerformanceProfile &profile,
                             const loom::sim::RetiredCgraSimulation &retired,
                             std::uint64_t activeWallNanoseconds,
                             std::uint64_t activeCpuNanoseconds,
                             llvm::StringRef performanceProfilePath) {
  CgraPerformanceProfile updated = profile;
  if (llvm::Error error =
          addProfileValue(updated.invocationCount, 1, "invocation_count"))
    return error;
  if (llvm::Error error =
          addProfileValue(updated.activeWallNanoseconds, activeWallNanoseconds,
                          "active_wall_nanoseconds"))
    return error;
  if (llvm::Error error =
          addProfileValue(updated.activeCpuNanoseconds, activeCpuNanoseconds,
                          "active_cpu_nanoseconds"))
    return error;
  if (llvm::Error error = addProfileValue(updated.eventFrameCount,
                                          retired.counters.eventFrameCount,
                                          "event_frame_count"))
    return error;
  if (llvm::Error error =
          writeCgraPerformanceProfile(performanceProfilePath, updated))
    return error;
  profile = updated;
  return llvm::Error::success();
}

llvm::Expected<loom::sim::SpatialEngineBoundaryResult>
finishCgra(loom::sim::CgraExecutionSession &session,
           CgraPerformanceProfile *performanceProfile,
           llvm::StringRef performanceProfilePath,
           std::uint64_t activeWallNanoseconds,
           std::uint64_t activeCpuNanoseconds) {
  loom::sim::CgraSimulationOutcome observation;
  observation.state = session.state();
  if (observation.state == loom::sim::SpatialExecutionSessionState::Runnable)
    observation.state = loom::sim::SpatialExecutionSessionState::StoppedByLimit;
  observation.counters = session.counters();
  observation.closedWaitSet = session.closedWaitSet();
  if (observation.state == loom::sim::SpatialExecutionSessionState::Retired) {
    auto retired = session.takeRetiredSimulation();
    if (!retired)
      return retired.takeError();
    observation.retired = std::move(*retired);
  }
  const auto *outcome = &observation;
  if (outcome->state == loom::sim::SpatialExecutionSessionState::StoppedByLimit)
    return llvm::createStringError(
        std::make_error_code(std::errc::timed_out),
        "CGRA engine reached its work limit: frames=" +
            std::to_string(outcome->counters.eventFrameCount) +
            ", actor_commits=" +
            std::to_string(outcome->counters.actorCommitCount) +
            ", actor_retirements=" +
            std::to_string(outcome->counters.actorRetirementCount) +
            ", token_publications=" +
            std::to_string(outcome->counters.tokenPublicationCount) +
            ", physical_requests=" +
            std::to_string(outcome->counters.physicalRequestCount) +
            ", physical_grants=" +
            std::to_string(outcome->counters.physicalGrantCount) +
            ", physical_retirements=" +
            std::to_string(outcome->counters.physicalRetirementCount) +
            ", empty_frames=" +
            std::to_string(outcome->counters.emptyEventFrameCount) +
            ", source_frames=" +
            std::to_string(outcome->counters.computeSourceFrameCount) + "/" +
            std::to_string(outcome->counters.memorySourceFrameCount) + "/" +
            std::to_string(outcome->counters.transportSourceFrameCount) + "/" +
            std::to_string(outcome->counters.physicalSourceFrameCount));
  if (outcome->state != loom::sim::SpatialExecutionSessionState::Retired ||
      !outcome->retired) {
    const llvm::StringRef state = [&] {
      switch (outcome->state) {
      case loom::sim::SpatialExecutionSessionState::Runnable:
        return llvm::StringRef("runnable");
      case loom::sim::SpatialExecutionSessionState::Retired:
        return llvm::StringRef("retired_without_result");
      case loom::sim::SpatialExecutionSessionState::Halted:
        return llvm::StringRef("halted");
      case loom::sim::SpatialExecutionSessionState::StoppedByLimit:
        return llvm::StringRef("stopped_by_limit");
      case loom::sim::SpatialExecutionSessionState::WaitingForExternalMemory:
        return llvm::StringRef("waiting_for_external_memory");
      case loom::sim::SpatialExecutionSessionState::WaitingForExternalStreamInput:
        return llvm::StringRef("waiting_for_external_stream_input");
      case loom::sim::SpatialExecutionSessionState::Failed:
        return llvm::StringRef("failed");
      }
      llvm_unreachable("unknown Spatial execution session state");
    }();
    std::string diagnostic;
    llvm::raw_string_ostream stream(diagnostic);
    stream << "CGRA engine did not retire the graph: state=" << state
           << ", frames=" << outcome->counters.eventFrameCount
           << ", actor_commits=" << outcome->counters.actorCommitCount
           << ", actor_retirements=" << outcome->counters.actorRetirementCount
           << ", token_publications=" << outcome->counters.tokenPublicationCount
           << ", memory_linearizations="
           << outcome->counters.memoryLinearizationCount
           << ", physical_requests=" << outcome->counters.physicalRequestCount
           << ", physical_grants=" << outcome->counters.physicalGrantCount
           << ", physical_retirements="
           << outcome->counters.physicalRetirementCount;
    if (outcome->closedWaitSet)
      stream << ", pending_actor_firings="
             << outcome->closedWaitSet->pendingActorFirings
             << ", pending_transfers="
             << outcome->closedWaitSet->pendingTransfers
             << ", pending_physical_actions="
             << outcome->closedWaitSet->pendingPhysicalActions
             << ", graph_retirement_visible="
             << outcome->closedWaitSet->graphRetirementVisible;
    if (outcome->closedWaitSet)
      for (const auto &firing : outcome->closedWaitSet->actorFirings)
        stream << ", actor_firing={actor=" << firing.semanticActorOrdinal
               << ", occurrence=" << firing.occurrenceOrdinal
               << ", transition=" << firing.transitionCaseOrdinal
               << ", transfers=" << firing.completedTransfers << "/"
               << firing.expectedTransfers
               << ", physical_complete=" << firing.physicalComplete
               << ", causal_release=" << firing.causalReleaseSatisfied << "}";
    if (outcome->closedWaitSet)
      for (const auto &transfer : outcome->closedWaitSet->transfers) {
        stream << ", transfer={binding=" << transfer.bindingOrdinal
               << ", occurrence=" << transfer.occurrenceOrdinal
               << ", producer=" << transfer.producerActorOrdinal << ":"
               << transfer.producerResultOrdinal
               << ", blocked=" << transfer.blocked
               << ", arrival_scheduled=" << transfer.arrivalScheduled
               << ", publication_ready=" << transfer.publicationReady
               << ", published=" << transfer.published
               << ", consumed_requested=" << transfer.consumedRequested
               << ", operand_reserved=" << transfer.operandCapacityReserved
               << ", operand_blocked=" << transfer.operandCapacityBlocked
               << ", produced=" << transfer.producedRetired << "/"
               << transfer.producedPermitted
               << ", traversal=" << transfer.traversalRetired << "/"
               << transfer.traversalPermitted << ", traversal_terminals="
               << transfer.traversalTerminalsPermitted
               << ", consumed=" << transfer.consumedRetired << "/"
               << transfer.consumedPermitted
               << ", ready_sinks=" << transfer.readySinkCount
               << ", published_sinks=" << transfer.publishedSinkCount << "/"
               << transfer.sinkCount
               << ", publications=" << transfer.publishedPublicationCount << "/"
               << transfer.requestedPublicationCount << "/"
               << transfer.publicationCount << ", blocking_traversal="
               << transfer.blockingTraversalNodeOrdinal << ":"
               << transfer.blockingTraversalWaitingForStorage
               << ", blocking_storage=" << transfer.blockingStorageOrdinal
               << ":" << transfer.blockingStorageOccupancy << "+"
               << transfer.blockingStorageReservations << "/"
               << transfer.blockingStorageCapacity << ":";
        if (transfer.blockingStorageHead)
          stream << transfer.blockingStorageHead->bindingOrdinal << ":"
                 << transfer.blockingStorageHead->occurrenceOrdinal << ":"
                 << transfer.blockingStorageHead->traversalNodeOrdinal;
        else
          stream << "none";
        stream << ", downstream=" << transfer.blockingDownstreamStorageCount
               << ":" << transfer.blockingUnbufferedSinkCount << ":"
               << transfer.blockingDownstreamStorageOrdinal << ":"
               << transfer.blockingDownstreamStorageOccupancy << "+"
               << transfer.blockingDownstreamStorageReservations << "/"
               << transfer.blockingDownstreamStorageCapacity << ":"
               << transfer.blockingDownstreamStorageReserved << ":";
        if (transfer.blockingDownstreamStorageHead)
          stream
              << transfer.blockingDownstreamStorageHead->bindingOrdinal << ":"
              << transfer.blockingDownstreamStorageHead->occurrenceOrdinal
              << ":"
              << transfer.blockingDownstreamStorageHead->traversalNodeOrdinal;
        else
          stream << "none";
        stream << ", blocking_route_targets=[";
        for (auto indexed : llvm::enumerate(transfer.blockingTraversals)) {
          if (indexed.index())
            stream << ",";
          stream << llvm::toHex(
              loom::fabric::canonicalFabricBytes(indexed.value()), true);
        }
        stream << "], downstream_route_targets=[";
        for (auto indexed :
             llvm::enumerate(transfer.blockingDownstreamTraversals)) {
          if (indexed.index())
            stream << ",";
          stream << llvm::toHex(
              loom::fabric::canonicalFabricBytes(indexed.value()), true);
        }
        stream << "]";
        stream << ", blocking_actor=" << transfer.blockingActorOrdinal
               << ", blocking_ready=" << transfer.blockingReadyTokenCount
               << ", blocking_queue=" << transfer.blockingQueueOccupancy << "+"
               << transfer.blockingQueueReservations << "/"
               << transfer.blockingQueueCapacity << ", unpublished=[";
        for (std::size_t index = 0;
             index != transfer.unpublishedActorOrdinals.size(); ++index) {
          if (index != 0)
            stream << ",";
          stream << transfer.unpublishedActorOrdinals[index] << ":"
                 << transfer.unpublishedInputOrdinals[index] << ":"
                 << transfer.unpublishedReadyTokenCounts[index];
        }
        stream << "]}";
      }
    if (outcome->closedWaitSet) {
      stream << ", blocked_actor_inputs=[";
      for (std::size_t index = 0;
           index != outcome->closedWaitSet->blockedActorInputs.size();
           ++index) {
        if (index != 0)
          stream << ",";
        const auto &input = outcome->closedWaitSet->blockedActorInputs[index];
        stream << input.semanticActorOrdinal << ":" << input.actorEntityId
               << ":" << input.inputOrdinal << ":" << input.channelOrdinal
               << ":" << static_cast<unsigned>(input.sourceKind) << ":"
               << input.definingActorOrdinal << ":"
               << input.definingActorEntityId << ":"
               << input.definingActorTerminal;
      }
      stream << "]";
      stream << ", transfer_wait_cycle=[";
      for (std::size_t index = 0;
           index != outcome->closedWaitSet->transferWaitCycle.size(); ++index) {
        if (index != 0)
          stream << ",";
        const auto &edge = outcome->closedWaitSet->transferWaitCycle[index];
        stream << edge.waitingBindingOrdinal << ":"
               << edge.waitingOccurrenceOrdinal << "->"
               << edge.blockingActorOrdinal << ":"
               << edge.blockingBindingOrdinal << ":"
               << edge.blockingOccurrenceOrdinal << ":"
               << static_cast<unsigned>(edge.kind);
      }
      stream << "]";
      stream << ", actor_wait_cycle=[";
      for (std::size_t index = 0;
           index != outcome->closedWaitSet->actorWaitCycle.size(); ++index) {
        if (index != 0)
          stream << ",";
        const auto &edge = outcome->closedWaitSet->actorWaitCycle[index];
        stream << edge.waitingActorOrdinal << "->" << edge.blockingActorOrdinal
               << ":" << static_cast<unsigned>(edge.kind);
      }
      stream << "]";
      stream
          << ", operand_queue_summary={groups="
          << outcome->closedWaitSet->operandQueueGroupCount
          << ",blocking_groups="
          << outcome->closedWaitSet->operandQueuePotentiallyBlockingGroupCount
          << ",shared_ingress="
          << outcome->closedWaitSet->operandQueueSharedIngressPressure
          << ",distinct_ingress="
          << outcome->closedWaitSet->operandQueueDistinctIngressCount
          << ",pairing_keys="
          << outcome->closedWaitSet->operandQueuePairingKeyCount << ",status="
          << static_cast<unsigned>(
                 outcome->closedWaitSet->operandQueueProgressStatus)
          << ",support="
          << static_cast<unsigned>(
                 outcome->closedWaitSet->operandQueueProgressSupport)
          << ",digest=";
      if (outcome->closedWaitSet->operandQueueProjectionDigest)
        stream << loom::formatComponentViewDigestHex(
            *outcome->closedWaitSet->operandQueueProjectionDigest);
      else
        stream << "none";
      stream << "}";
      stream << ", operand_queue_heads=[";
      for (std::size_t index = 0;
           index != outcome->closedWaitSet->operandQueueHeads.size() &&
           index != 16;
           ++index) {
        if (index != 0)
          stream << ",";
        const auto &head = outcome->closedWaitSet->operandQueueHeads[index];
        llvm::SmallString<32> tag;
        head.headTag.toStringUnsigned(tag, 16);
        stream << "{context="
               << llvm::toHex(
                      ::loom::fabric::canonicalFabricBytes(head.queue.context),
                      true)
               << ",fu_occurrence=" << head.queue.fuOccurrence
               << ",fu_input=" << head.queue.fuInput
               << ",unit=" << head.allocationUnit
               << ",occupancy=" << head.occupancy
               << ",reservations=" << head.reservations
               << ",capacity=" << head.capacity
               << ",head=" << head.headBindingOrdinal << ":"
               << head.headOccurrenceOrdinal << ":"
               << head.headProducerSequenceOrdinal << ":" << tag.str()
               << ",exact=" << head.exactHead << "}";
      }
      if (outcome->closedWaitSet->operandQueueHeads.size() > 16)
        stream << ",...";
      stream << "]";
    }
    if (outcome->closedWaitSet)
      for (const auto &action : outcome->closedWaitSet->physicalActions)
        stream << ", physical_action={action=" << action.actionOrdinal
               << ", occurrence=" << action.occurrenceOrdinal
               << ", client=" << static_cast<unsigned>(action.clientKind)
               << ", granted=" << action.granted
               << ", has_commit=" << action.hasCommit
               << ", requires_causal_release=" << action.requiresCausalRelease
               << ", intrinsic_release=" << action.intrinsicReleaseReached
               << ", causal_release=" << action.causalReleaseReached << "}";
    return invalid(diagnostic);
  }
  if (performanceProfile)
    if (llvm::Error error = recordCgraPerformanceProfile(
            *performanceProfile, *outcome->retired, activeWallNanoseconds,
            activeCpuNanoseconds, performanceProfilePath))
      return std::move(error);
  return loom::sim::SpatialEngineBoundaryResult{
      loom::sim::RetiredExecution{},
      std::move(observation.retired->observations),
      std::move(observation.retired->progress),
      {}};
}
#endif

llvm::Expected<std::uint64_t> completionDelay(
    const loom::sim::SpatialEngineBoundaryResult &result,
    std::uint64_t ticksPerCycle,
    std::optional<loom::sim::SpatialEventCoordinate> externallyServicedThrough =
        std::nullopt) {
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  return 0;
#else
  if (!result.progressObservations.graphRetirementVisible)
    return 0;
  return spatialServiceDelay(*result.progressObservations.graphRetirementVisible,
                              ticksPerCycle, externallyServicedThrough);
#endif
}

llvm::Expected<std::size_t> selectSessionEntry(
    llvm::ArrayRef<SpatialSessionEntry> entries,
    const loom::runtime::Gem5SpatialLaunchEnvelope &launch,
    std::uint64_t bridgeOrdinal,
    const std::optional<loom::runtime::SpatialInvocationWire> &invocation) {
  std::optional<std::size_t> selected;
  for (const auto indexed : llvm::enumerate(entries)) {
    const SpatialSessionEntry &entry = indexed.value();
    if (entry.bridgeOrdinal != bridgeOrdinal)
      continue;
    if (entry.expectedLaunch != launch.staticLaunch)
      continue;
    const auto *workload = entry.workload.workload.spatial();
    if (!workload)
      return invalid("Spatial session entry lost its workload payload");
    bool matches = false;
    if (invocation) {
      matches = invocation->canonicalDataflowIdentity ==
                    entry.workload.dataflow->identity().bytes() &&
                invocation->rootThreadLaunchEntity ==
                    workload->launchRef.rootThreadLaunch.entity.value() &&
                invocation->graphLaunchEntity ==
                    workload->launchRef.staticGraphLaunch.entity.value() &&
                invocation->denseCoordinates == workload->denseCoordinates;
    } else {
      matches = entry.staticRuntime.has_value();
    }
    if (!matches)
      continue;
    if (selected)
      return invalid("Spatial launch matches multiple session entries");
    selected = indexed.index();
  }
  if (!selected)
    return invalid("Spatial launch matches no session entry");
  return *selected;
}

enum class InvocationPhase {
  RunningModel,
  NeedsStreamInputReadiness,
  AwaitingStreamInputReadiness,
  WaitingForStreamInput,
  ModelMemory,
  ResultWrites,
  NeedsChannelCommit,
  AwaitingChannelCommit,
  PublishingChannels,
};

struct PendingCompletion final {
  std::vector<PendingChannelPublication> publications;
  std::vector<loom::sim::SpatialInvocationMemoryWrite> writes;
  std::vector<std::uint8_t> result;
  std::size_t nextWrite = 0;
  std::uint64_t remainingDelay = 0;
  bool retired = false;
};

struct SpatialInvocation final {
  SpatialInvocation(std::size_t entryOrdinal,
                    loom::runtime::Gem5SpatialLaunchEnvelope launch,
                    std::optional<loom::runtime::SpatialInvocationWire> wire,
                    loom::sim::CanonicalSimulationRuntimeInput runtime)
      : entryOrdinal(entryOrdinal), launch(std::move(launch)),
        wire(std::move(wire)), runtime(std::move(runtime)) {}

  const std::size_t entryOrdinal;
  const loom::runtime::Gem5SpatialLaunchEnvelope launch;
  const std::optional<loom::runtime::SpatialInvocationWire> wire;
  const loom::sim::CanonicalSimulationRuntimeInput runtime;
  InvocationPhase phase = InvocationPhase::RunningModel;
  std::optional<loom::sim::SpatialEventCoordinate> servicedThrough;
  const loom::sim::CanonicalSimulationRuntimeInput *retiredRuntimeInput = nullptr;
  std::uint64_t nextRequestId = 0;
  std::optional<loom::runtime::Gem5BridgeMemoryRequest> outstandingMemory;
  std::optional<PendingCompletion> completion;
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  std::optional<loom::sim::DfgExecutionSession> dfg;
#else
  std::optional<Gem5CgraExternalMemoryProvider> externalMemory;
  std::optional<loom::sim::CgraExecutionSession> cgra;
  std::uint64_t activeWallNanoseconds = 0;
  std::uint64_t activeCpuNanoseconds = 0;
#endif

  const std::optional<loom::sim::SpatialStreamInputRequest> &
  pendingStreamInput() const {
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
    return dfg->pendingStreamInput();
#else
    return cgra->pendingStreamInput();
#endif
  }

  llvm::Error completeStreamInput(
      const loom::sim::SpatialStreamInputRequest &request,
      const loom::sim::CanonicalValueSequence &value) {
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
    return dfg->completeStreamInput(request, value);
#else
    return cgra->completeStreamInput(request, value);
#endif
  }
};

llvm::Expected<bool> receiveStreamInput(
    SpatialInvocation &invocation, const SpatialSessionEntry &entry,
    std::map<std::uint64_t, ChannelSequenceState> &channelSequences) {
  const auto &request = invocation.pendingStreamInput();
  if (!request)
    return invalid("Spatial stream wait lost its demanded input event");
  auto channel = llvm::find_if(entry.channels.inputs, [&](const auto &input) {
    return input.consumerStreamInputOrdinal == request->streamInputOrdinal;
  });
  if (channel == entry.channels.inputs.end())
    return invalid("demanded stream input has no ordered channel binding");
  auto state = channelSequences.find(channel->channelOrdinal);
  if (state == channelSequences.end())
    return invalid("stream input lost its ordered channel sequence owner");
  auto key = channelConsumerKey(entry.workload, *channel);
  if (!key)
    return key.takeError();
  auto branch = state->second.consumerOrdinals.find(*key);
  if (branch == state->second.consumerOrdinals.end())
    return invalid("stream input has no deterministic consumer branch");
  auto ticket = state->second.abi.receive(branch->second);
  if (!ticket)
    return ticket.takeError();
  if (ticket->kind == loom::runtime::OrderedChannelReceiveKind::WouldBlock)
    return false;
  // Only Message creates a reservation in OrderedChannelABI. A generation
  // terminal carries no live ticket to cancel.
  if (ticket->kind != loom::runtime::OrderedChannelReceiveKind::Message)
    return invalid("ordered channel ended before the demanded receive event");
  const auto &view = entry.workload.dataflow->view();
  auto stream = loom::sim::decodeSpatialChannelStream(
      ticket->payload, view, entry.workload.workload.spatial()->launchRef,
      request->streamInputOrdinal,
      invocation.runtime.spatial()->memoryObjects.size());
  if (!stream)
    return llvm::joinErrors(stream.takeError(), state->second.abi.cancel(*ticket));
  if (llvm::Error error = invocation.completeStreamInput(*request, stream->values))
    return llvm::joinErrors(std::move(error), state->second.abi.cancel(*ticket));
  if (llvm::Error error = state->second.abi.acknowledge(*ticket))
    return std::move(error);
  return true;
}

struct BridgeInvocation final {
  std::uint64_t nextSequence = 0;
  std::unique_ptr<SpatialInvocation> active;
};

} // namespace

struct SpatialEngineSession::Impl final {
  std::vector<SpatialSessionEntry> entries;
  std::vector<PreparedSpatialExecution> preparedExecutions;
  SpatialEngineLimits limits;
  std::string performanceProfilePath;
  std::map<std::uint64_t, ChannelSequenceState> channelSequences;
  std::map<std::uint64_t, BridgeInvocation> bridges;
  std::uint64_t lastGeneration = 0;
  std::uint64_t lastCausalTick = 0;
#if !defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  CgraPerformanceProfile performanceProfile;
#endif

  llvm::Error initializeChannels();
  llvm::Expected<std::unique_ptr<SpatialInvocation>>
  startInvocation(const loom::runtime::Gem5BridgeMessage &message);
  llvm::Error acceptInput(const loom::runtime::Gem5BridgeMessage &message);
  llvm::Expected<std::optional<loom::sim::SpatialEngineBoundaryResult>>
  advanceModel(SpatialInvocation &invocation);
  llvm::Error finishModel(SpatialInvocation &invocation,
                          loom::sim::SpatialEngineBoundaryResult result);
  llvm::Expected<std::optional<loom::runtime::Gem5BridgeMessage>>
  nextBoundary(std::uint64_t bridgeOrdinal, BridgeInvocation &bridge,
               bool &channelsAdvanced);
};

llvm::Error SpatialEngineSession::Impl::initializeChannels() {
  std::map<std::uint64_t, std::uint64_t> channelCapacities;
  std::map<std::uint64_t, std::set<std::string>> channelConsumerKeys;
  for (const SpatialSessionEntry &entry : entries) {
    for (const auto &output : entry.channels.outputs) {
      auto [position, inserted] = channelCapacities.emplace(
          output.channelOrdinal, output.capacityMessages);
      if (!inserted && position->second != output.capacityMessages)
        return invalid("ordered channel outputs disagree on capacity");
    }
    for (const auto &input : entry.channels.inputs) {
      auto [position, inserted] = channelCapacities.emplace(
          input.channelOrdinal, input.capacityMessages);
      if (!inserted && position->second != input.capacityMessages)
        return invalid("ordered channel inputs disagree on capacity");
      auto key = channelConsumerKey(entry.workload, input);
      if (!key)
        return key.takeError();
      channelConsumerKeys[input.channelOrdinal].insert(std::move(*key));
    }
  }
  for (const auto &[channelOrdinal, capacityMessages] : channelCapacities) {
    const auto keys = channelConsumerKeys.find(channelOrdinal);
    if (keys == channelConsumerKeys.end() || keys->second.empty())
      return invalid("ordered channel has no consumer branch");
    if (keys->second.size() > std::numeric_limits<std::uint32_t>::max())
      return invalid("ordered channel consumer count exceeds u32");
    const std::uint32_t consumers =
        static_cast<std::uint32_t>(keys->second.size());
    if (consumers == 0)
      return invalid("ordered channel has no consumer branch");
    auto abi =
        loom::runtime::OrderedChannelABI::create(capacityMessages, consumers);
    if (!abi)
      return abi.takeError();
    channelSequences.emplace(channelOrdinal,
                             ChannelSequenceState{std::move(*abi), {}});
    auto &ordinals =
        channelSequences.find(channelOrdinal)->second.consumerOrdinals;
    std::uint32_t ordinal = 0;
    for (const std::string &key : keys->second)
      ordinals.emplace(key, ordinal++);
  }
  for (const SpatialSessionEntry &entry : entries) {
    for (const auto &input : entry.channels.inputs) {
      auto state = channelSequences.find(input.channelOrdinal);
      if (state == channelSequences.end())
        return invalid("ordered channel input has no sequence state");
      auto key = channelConsumerKey(entry.workload, input);
      if (!key)
        return key.takeError();
      if (state->second.consumerOrdinals.find(*key) ==
          state->second.consumerOrdinals.end())
        return invalid("ordered channel branch identity was not "
                       "materialized canonically");
    }
  }

  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<SpatialInvocation>>
SpatialEngineSession::Impl::startInvocation(
    const loom::runtime::Gem5BridgeMessage &message) {
  loom::runtime::Gem5SpatialLaunchEnvelope launch;
  std::string diagnostic;
  if (!loom::runtime::decodeGem5SpatialLaunchEnvelope(message.payload, launch,
                                                      diagnostic))
    return invalid(diagnostic);
  std::optional<loom::runtime::SpatialInvocationWire> invocation;
  if (!launch.invocation.empty()) {
    loom::runtime::SpatialInvocationWire wire;
    if (!loom::runtime::decodeSpatialInvocationWire(launch.invocation, wire,
                                                    diagnostic))
      return invalid(diagnostic);
    invocation = std::move(wire);
  }
  auto selected = selectSessionEntry(entries, launch,
                                     message.bridgeSessionOrdinal, invocation);
  if (!selected)
    return selected.takeError();
  const auto &entry = entries[*selected];
  if (invocation) {
    if (entry.staticRuntime)
      return invalid("dynamic invocation has a competing static runtime input");
    auto runtime = loom::sim::materializeSpatialInvocationRuntimeInput(
        entry.workload, *invocation);
    if (!runtime)
      return runtime.takeError();
    return std::make_unique<SpatialInvocation>(*selected, std::move(launch),
                                               std::move(invocation),
                                               std::move(*runtime));
  }
  if (!entry.staticRuntime)
    return invalid("static launch has no Spatial runtime input");
  auto runtime = cloneRuntimeInput(entry.workload, *entry.staticRuntime);
  if (!runtime)
    return runtime.takeError();
  return std::make_unique<SpatialInvocation>(*selected, std::move(launch),
                                             std::nullopt, std::move(*runtime));
}

llvm::Error SpatialEngineSession::Impl::acceptInput(
    const loom::runtime::Gem5BridgeMessage &message) {
  auto found = bridges.find(message.bridgeSessionOrdinal);
  if (found == bridges.end())
    return invalid("causal input names a foreign physical bridge");
  auto &bridge = found->second;
  if (message.sequence != bridge.nextSequence)
    return invalid("causal input has a stale or foreign invocation sequence");
  if (message.kind == loom::runtime::Gem5BridgeMessageKind::SpatialLaunch) {
    if (bridge.active || bridge.nextSequence >= limits.maximumInvocations)
      return invalid(
          "bridge launch overlaps an invocation or exceeds its limit");
    auto invocation = startInvocation(message);
    if (!invocation)
      return invocation.takeError();
    bridge.active = std::move(*invocation);
    return llvm::Error::success();
  }
  if (!bridge.active)
    return invalid("causal continuation has no active invocation");
  auto &invocation = *bridge.active;
  if (message.kind == loom::runtime::Gem5BridgeMessageKind::ChannelCommitted) {
    if (!message.payload.empty())
      return invalid("channel commit acknowledgement has a payload");
    if (invocation.phase == InvocationPhase::AwaitingStreamInputReadiness) {
      const auto &request = invocation.pendingStreamInput();
      if (!request)
        return invalid("stream readiness acknowledgement lost its request");
      invocation.servicedThrough = request->readyCoordinate;
      invocation.phase = InvocationPhase::WaitingForStreamInput;
    } else if (invocation.phase == InvocationPhase::AwaitingChannelCommit) {
      invocation.phase = InvocationPhase::PublishingChannels;
    } else {
      return invalid("channel commit acknowledgement has no matching boundary");
    }
    return llvm::Error::success();
  }
  if (message.kind != loom::runtime::Gem5BridgeMessageKind::MemoryResponse ||
      !invocation.outstandingMemory)
    return invalid("causal continuation has no matching memory boundary");
  loom::runtime::Gem5BridgeMemoryResponse response;
  std::string diagnostic;
  if (!loom::runtime::decodeGem5BridgeMemoryResponse(message.payload, response,
                                                     diagnostic))
    return invalid(diagnostic);
  const auto &request = *invocation.outstandingMemory;
  if (response.requestId != request.requestId || !response.success ||
      (request.operation == loom::runtime::Gem5BridgeMemoryOperation::Read
           ? response.data.size() != request.size
           : !response.data.empty()))
    return invalid("bridge memory response does not match its request");
  if (invocation.phase == InvocationPhase::ResultWrites) {
    ++invocation.completion->nextWrite;
  }
#if !defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  else if (invocation.phase == InvocationPhase::ModelMemory) {
    auto completed = invocation.externalMemory->completeElement(
        std::move(response), *invocation.cgra);
    if (!completed)
      return completed.takeError();
    if (*completed)
      invocation.phase = InvocationPhase::RunningModel;
  }
#endif
  else {
    return invalid("memory response arrived in a non-memory invocation phase");
  }
  invocation.outstandingMemory.reset();
  return llvm::Error::success();
}

llvm::Expected<std::optional<loom::sim::SpatialEngineBoundaryResult>>
SpatialEngineSession::Impl::advanceModel(SpatialInvocation &invocation) {
  const auto &entry = entries[invocation.entryOrdinal];
  std::vector<std::uint64_t> liveInputs;
  for (const auto &input : entry.channels.inputs)
    liveInputs.push_back(input.consumerStreamInputOrdinal);
#if defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  if (!invocation.dfg) {
    auto session = loom::sim::startDfgExecutionSession(
        preparedExecutions[entry.preparedOrdinal], entry.workload.workload,
        invocation.runtime, liveInputs);
    if (!session)
      return session.takeError();
    invocation.dfg.emplace(std::move(*session));
  }
  const auto waves = invocation.dfg->wavefrontSteps();
  if (waves < limits.maximumWork) {
    auto advanced = invocation.dfg->advance(limits.maximumWork - waves);
    if (!advanced)
      return advanced.takeError();
  }
  if (invocation.dfg->state() ==
      loom::sim::SpatialExecutionSessionState::WaitingForExternalStreamInput) {
    invocation.phase = InvocationPhase::NeedsStreamInputReadiness;
    return std::optional<loom::sim::SpatialEngineBoundaryResult>{};
  }
  if (invocation.dfg->state() == loom::sim::SpatialExecutionSessionState::Runnable)
    return llvm::createStringError(std::errc::timed_out,
                                   "DFG maximum event steps reached");
  auto result = finishDfg(*invocation.dfg);
  if (!result)
    return result.takeError();
  invocation.retiredRuntimeInput = invocation.dfg->retiredRuntimeInput();
  return std::optional<loom::sim::SpatialEngineBoundaryResult>(
      std::move(*result));
#else
  const bool profile = !performanceProfilePath.empty();
  std::uint64_t startedCpu = 0;
  std::chrono::steady_clock::time_point startedWall;
  if (profile) {
    auto cpu = engineProcessCpuNanoseconds();
    if (!cpu)
      return cpu.takeError();
    startedCpu = *cpu;
    startedWall = std::chrono::steady_clock::now();
  }
  if (!invocation.cgra) {
    if (invocation.wire)
      invocation.externalMemory.emplace(*invocation.wire, limits.ticksPerCycle,
                                         invocation.servicedThrough);
    auto session = loom::sim::startCgraExecutionSession(
        preparedExecutions[entry.preparedOrdinal], entry.workload.workload,
        invocation.runtime, std::nullopt,
        invocation.externalMemory ? &*invocation.externalMemory : nullptr,
        liveInputs);
    if (!session)
      return session.takeError();
    invocation.cgra.emplace(std::move(*session));
  }
  const auto frames = invocation.cgra->counters().eventFrameCount;
  if (frames < limits.maximumWork) {
    auto advanced = invocation.cgra->advance(limits.maximumWork - frames);
    if (!advanced)
      return advanced.takeError();
  }
  if (profile) {
    const auto finishedWall = std::chrono::steady_clock::now();
    auto cpu = engineProcessCpuNanoseconds();
    if (!cpu)
      return cpu.takeError();
    const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          finishedWall - startedWall)
                          .count();
    if (*cpu < startedCpu || wall < 0)
      return invalid("CGRA performance clock moved backwards");
    if (auto error = addProfileValue(invocation.activeWallNanoseconds,
                                     static_cast<std::uint64_t>(wall),
                                     "active_wall_nanoseconds"))
      return std::move(error);
    if (auto error =
            addProfileValue(invocation.activeCpuNanoseconds, *cpu - startedCpu,
                            "active_cpu_nanoseconds"))
      return std::move(error);
  }
  if (invocation.cgra->state() ==
      loom::sim::SpatialExecutionSessionState::WaitingForExternalMemory) {
    invocation.phase = InvocationPhase::ModelMemory;
    return std::optional<loom::sim::SpatialEngineBoundaryResult>{};
  }
  if (invocation.cgra->state() ==
      loom::sim::SpatialExecutionSessionState::WaitingForExternalStreamInput) {
    invocation.phase = InvocationPhase::NeedsStreamInputReadiness;
    return std::optional<loom::sim::SpatialEngineBoundaryResult>{};
  }
  auto result =
      finishCgra(*invocation.cgra, profile ? &performanceProfile : nullptr,
                 performanceProfilePath, invocation.activeWallNanoseconds,
                 invocation.activeCpuNanoseconds);
  if (!result)
    return result.takeError();
  invocation.retiredRuntimeInput = invocation.cgra->retiredRuntimeInput();
  return std::optional<loom::sim::SpatialEngineBoundaryResult>(
      std::move(*result));
#endif
}

llvm::Error SpatialEngineSession::Impl::finishModel(
    SpatialInvocation &invocation,
    loom::sim::SpatialEngineBoundaryResult result) {
  const auto &entry = entries[invocation.entryOrdinal];
  if (!invocation.retiredRuntimeInput)
    return invalid("retired Spatial model omitted its runtime input snapshot");
  const auto &runtimeInput = *invocation.retiredRuntimeInput;
  auto encoded = loom::sim::encodeSpatialEngineBoundaryResult(
      result, entry.workload, runtimeInput);
  if (!encoded)
    return encoded.takeError();
  auto delay =
      completionDelay(result, limits.ticksPerCycle, invocation.servicedThrough);
  if (!delay)
    return delay.takeError();
  std::vector<loom::sim::SpatialInvocationMemoryWrite> invocationWrites;
  if (invocation.wire) {
    auto writes = loom::sim::projectSpatialInvocationResultWrites(
        *invocation.wire, entry.workload, result.functionalObservations);
    if (!writes)
      return writes.takeError();
#if !defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
    if (invocation.externalMemory)
      *writes = invocation.externalMemory->retainUncommittedWrites(*writes);
#endif
    invocationWrites = std::move(*writes);
  }
  std::optional<loom::runtime::SpatialInvocationRuntimeInputSnapshot> snapshot;
  if (invocation.wire)
    snapshot.emplace(loom::runtime::SpatialInvocationRuntimeInputSnapshot{
        runtimeInput.identity().bytes(),
        {runtimeInput.canonicalBytes().bytes().begin(),
         runtimeInput.canonicalBytes().bytes().end()}});
  auto completionResult = loom::runtime::encodeSpatialInvocationResultWire(
      {entry.sessionEntryOrdinal, invocation.launch.invocation,
       std::move(snapshot), std::move(*encoded)});
  if (completionResult.empty())
    return invalid("cannot encode Spatial invocation result");
  auto publications =
      prepareChannelPublications(entry.channels, result, entry.workload,
                                 runtimeInput, channelSequences);
  if (!publications)
    return publications.takeError();
  const bool retired =
      std::holds_alternative<loom::sim::RetiredExecution>(result.terminal);
  invocation.completion.emplace(
      PendingCompletion{std::move(*publications), std::move(invocationWrites),
                        std::move(completionResult), 0, *delay, retired});
  invocation.phase = InvocationPhase::ResultWrites;
  return llvm::Error::success();
}

llvm::Expected<std::optional<loom::runtime::Gem5BridgeMessage>>
SpatialEngineSession::Impl::nextBoundary(std::uint64_t bridgeOrdinal,
                                         BridgeInvocation &bridge,
                                         bool &channelsAdvanced) {
  using Message = loom::runtime::Gem5BridgeMessage;
  using Kind = loom::runtime::Gem5BridgeMessageKind;
  if (!bridge.active)
    return std::optional<Message>{};
  auto &invocation = *bridge.active;
  const auto &entry = entries[invocation.entryOrdinal];
  const auto emit = [&](Kind kind, std::vector<std::uint8_t> payload) {
    return std::optional<Message>(
        Message{kind, bridgeOrdinal, bridge.nextSequence, std::move(payload)});
  };
  if (invocation.phase == InvocationPhase::WaitingForStreamInput) {
    auto received = receiveStreamInput(invocation, entry, channelSequences);
    if (!received)
      return received.takeError();
    if (!*received)
      return std::optional<Message>{};
    channelsAdvanced = true;
    invocation.phase = InvocationPhase::RunningModel;
  }
  if (invocation.phase == InvocationPhase::RunningModel) {
    auto result = advanceModel(invocation);
    if (!result) {
      std::string diagnostic = llvm::toString(result.takeError());
      const auto *spatial = entry.workload.workload.spatial();
      return llvm::createStringError(
          std::make_error_code(std::errc::state_not_recoverable),
          "Spatial invocation sequence " + std::to_string(bridge.nextSequence) +
              " (bridge=" + std::to_string(bridgeOrdinal) +
              ", session_entry=" + std::to_string(invocation.entryOrdinal) +
              ", prepared_entry=" + std::to_string(entry.preparedOrdinal) +
              ", runtime_objects=" +
              std::to_string(
                  invocation.runtime.spatial()->memoryObjects.size()) +
              ", runtime_values=" +
              std::to_string(
                  invocation.runtime.spatial()->runtimeValues.size()) +
              ", runtime_streams=" +
              std::to_string(
                  invocation.runtime.spatial()->runtimeStreams.size()) +
              ", runtime_identity=" +
              formatArtifactIdentityHex(invocation.runtime.identity()) +
              ", dense_coordinates=" +
              (spatial ? std::to_string(spatial->denseCoordinates.size())
                       : "0") +
              "): " + diagnostic);
    }
    if (*result)
      if (auto error = finishModel(invocation, std::move(**result)))
        return std::move(error);
  }
  if (invocation.phase == InvocationPhase::NeedsStreamInputReadiness) {
    const auto &request = invocation.pendingStreamInput();
    if (!request)
      return invalid("Spatial model lost its stream input request");
    std::uint64_t delay = 0;
#if !defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
    auto projected = spatialServiceDelay(request->readyCoordinate,
                                         limits.ticksPerCycle,
                                         invocation.servicedThrough);
    if (!projected)
      return projected.takeError();
    delay = *projected;
#endif
    invocation.phase = InvocationPhase::AwaitingStreamInputReadiness;
    return emit(Kind::ChannelCommit,
                loom::runtime::encodeGem5BridgeChannelCommit({delay}));
  }
  if (invocation.outstandingMemory)
    return std::optional<Message>{};
  if (invocation.phase == InvocationPhase::ResultWrites &&
      invocation.completion->nextWrite == invocation.completion->writes.size())
    invocation.phase = InvocationPhase::NeedsChannelCommit;
  if (invocation.phase == InvocationPhase::ResultWrites ||
      invocation.phase == InvocationPhase::ModelMemory) {
    if (invocation.nextRequestId == std::numeric_limits<std::uint64_t>::max())
      return invalid("bridge memory request identity domain exhausted");
    const auto requestId = invocation.nextRequestId++;
#if !defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
    if (invocation.phase == InvocationPhase::ModelMemory) {
      auto request = invocation.externalMemory->nextElement(requestId);
      if (!request)
        return request.takeError();
      invocation.outstandingMemory = std::move(*request);
    } else
#endif
    {
      auto &completion = *invocation.completion;
      const auto &write = completion.writes[completion.nextWrite];
      invocation.outstandingMemory.emplace(
          loom::runtime::Gem5BridgeMemoryRequest{
              loom::runtime::Gem5BridgeMemoryOperation::Write,
              std::exchange(completion.remainingDelay, 0), requestId,
              write.address, write.bytes.size(), write.bytes});
    }
    return emit(Kind::MemoryRequest,
                loom::runtime::encodeGem5BridgeMemoryRequest(
                    *invocation.outstandingMemory));
  }
  if (invocation.phase == InvocationPhase::NeedsChannelCommit) {
    invocation.phase = InvocationPhase::AwaitingChannelCommit;
    return emit(Kind::ChannelCommit,
                loom::runtime::encodeGem5BridgeChannelCommit(
                    {std::exchange(invocation.completion->remainingDelay, 0)}));
  }
  if (invocation.phase == InvocationPhase::PublishingChannels) {
    auto publication = publishAvailableChannelOutputs(
        invocation.completion->publications, channelSequences);
    if (!publication)
      return publication.takeError();
    channelsAdvanced |= publication->advanced;
    if (!publication->complete)
      return std::optional<Message>{};
    auto message =
        emit(Kind::Completion, loom::runtime::encodeGem5BridgeCompletion(
                                   {0, invocation.completion->retired ? 0U : 1U,
                                    std::move(invocation.completion->result)}));
    bridge.active.reset();
    ++bridge.nextSequence;
    return message;
  }
  return std::optional<Message>{};
}

SpatialEngineSession::SpatialEngineSession(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
SpatialEngineSession::~SpatialEngineSession() = default;

llvm::Expected<std::unique_ptr<SpatialEngineSession>>
SpatialEngineSession::create(
    std::vector<SpatialSessionEntry> entries,
    std::vector<PreparedSpatialExecution> preparedExecutions,
    SpatialEngineLimits limits, std::string performanceProfilePath) {
  auto impl = std::make_unique<Impl>();
  impl->entries = std::move(entries);
  impl->preparedExecutions = std::move(preparedExecutions);
  impl->limits = limits;
  impl->performanceProfilePath = std::move(performanceProfilePath);
  for (const auto &entry : impl->entries)
    impl->bridges.try_emplace(entry.bridgeOrdinal);
  if (auto error = impl->initializeChannels())
    return std::move(error);
#if !defined(LOOM_GEM5_SPATIAL_ENGINE_DFG)
  if (!impl->performanceProfilePath.empty())
    if (auto error = writeCgraPerformanceProfile(impl->performanceProfilePath,
                                                 impl->performanceProfile))
      return std::move(error);
#endif
  return std::unique_ptr<SpatialEngineSession>(
      new SpatialEngineSession(std::move(impl)));
}

llvm::Expected<loom::runtime::Gem5BridgeAdvance>
SpatialEngineSession::advance(const loom::runtime::Gem5BridgeAdvance &input) {
  if (impl_->lastGeneration == std::numeric_limits<std::uint64_t>::max() ||
      input.generation != impl_->lastGeneration + 1 ||
      input.causalTick < impl_->lastCausalTick || input.messages.size() != 1)
    return invalid(
        "causal input generation, tick, or message count is invalid");
  if (auto error = impl_->acceptInput(input.messages.front()))
    return std::move(error);
  impl_->lastGeneration = input.generation;
  impl_->lastCausalTick = input.causalTick;
  loom::runtime::Gem5BridgeAdvance response{
      input.generation, input.causalTick, {}};
  std::set<std::uint64_t> emitted;
  bool channelsAdvanced;
  do {
    channelsAdvanced = false;
    for (auto &[ordinal, bridge] : impl_->bridges) {
      if (emitted.count(ordinal))
        continue;
      auto boundary = impl_->nextBoundary(ordinal, bridge, channelsAdvanced);
      if (!boundary)
        return boundary.takeError();
      if (*boundary) {
        emitted.insert(ordinal);
        response.messages.push_back(std::move(**boundary));
      }
    }
  } while (channelsAdvanced);
  if (response.messages.empty() &&
      llvm::all_of(impl_->bridges, [](const auto &entry) {
        const auto &active = entry.second.active;
        return active && (active->phase == InvocationPhase::WaitingForStreamInput ||
                          active->phase == InvocationPhase::PublishingChannels);
      }))
    return llvm::createStringError(
        std::make_error_code(std::errc::timed_out),
        "all Spatial bridge invocations form a closed ordered-channel wait");
  return response;
}

} // namespace loom::gem5engine

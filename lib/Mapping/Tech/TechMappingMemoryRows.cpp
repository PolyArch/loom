#include "TechMappingCandidateDomain.h"

#include "TechMappingCanonicalKeyInternal.h"

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/MemoryOperationPort.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

struct MemoryActorOption final {
  TechMemoryActorView actor;
  std::vector<TechMemoryGraphBoundaryView> boundaries;
  std::vector<unsigned> operandOperationOrdinals;
  std::vector<unsigned> resultOperationOrdinals;
  std::vector<std::uint8_t> key;
};

struct MemoryActorDomain final {
  const ::dataflow::CanonicalActorView *actor;
  std::vector<MemoryActorOption> options;
};

llvm::Expected<std::optional<std::vector<TechMemoryGraphBoundaryView>>>
mergeBoundaries(llvm::ArrayRef<const MemoryActorOption *> selection,
                const ArtifactIdentity &owner);

llvm::Expected<std::vector<TechMemoryGraphBoundaryView>>
collectBoundaries(llvm::ArrayRef<const MemoryActorOption *> selection,
                  const ArtifactIdentity &owner);

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "tech_mapping_generation_invalid: " + message);
}

llvm::Expected<::loom::fabric::FabricMemoryEngineTemplateEndpointRef>
endpointForRole(
    ::loom::fabric::FabricMemoryEngineTemplateRef engine,
    const ::loom::fabric::MemoryCapabilityAlternativeView &capability,
    ::dataflow::semantics::ServiceValueRole role) {
  const auto binding = llvm::find_if(
      capability.roleToEndpoint,
      [&](const ::fabric::MemoryRoleEndpointBindingRecord &candidate) {
        return candidate.role == role;
      });
  if (binding == capability.roleToEndpoint.end())
    return invalid("memory capability omits a canonical service role");
  return ::loom::fabric::FabricMemoryEngineTemplateEndpointRef{
      engine, binding->endpointOrdinal};
}

llvm::Expected<std::vector<std::uint8_t>>
boundaryTerminalKey(const TechMemoryGraphEndpointRef &terminal,
                    const ArtifactIdentity &owner) {
  std::vector<std::uint8_t> key;
  if (const auto *producer =
          std::get_if<::dataflow::CanonicalGraphProducerEndpointRef>(
              &terminal)) {
    appendU32(key, 0);
    if (llvm::Error error = appendDataflowRef(key, owner, *producer))
      return std::move(error);
  } else {
    appendU32(key, 1);
    if (llvm::Error error = appendDataflowRef(
            key, owner,
            std::get<::dataflow::CanonicalGraphConsumerEndpointRef>(terminal)))
      return std::move(error);
  }
  return key;
}

llvm::Expected<std::vector<std::uint8_t>>
internalEdgeKey(const TechMemoryInternalEdgeView &edge,
                const ArtifactIdentity &owner) {
  std::vector<std::uint8_t> key;
  if (llvm::Error error = appendDataflowRef(key, owner, edge.producer))
    return std::move(error);
  if (llvm::Error error = appendDataflowRef(key, owner, edge.consumer))
    return std::move(error);
  appendFabricRef(key, edge.connection);
  return key;
}

llvm::Expected<MemoryActorOption> buildActorOption(
    const TechMappingGenerationInputs &inputs,
    const ::dataflow::CanonicalActorView &actor,
    ::loom::fabric::FabricMemoryEngineTemplateRef engine,
    ::loom::fabric::FabricMemoryEngineTemplateOperationPortRef port,
    ::loom::fabric::FabricMemoryEngineTemplateCapabilityAlternativeRef
        alternative,
    const ::loom::fabric::MemoryCapabilityAlternativeView &capability,
    const ::dataflow::semantics::CanonicalService &service) {
  MemoryActorOption option{
      TechMemoryActorView{actor.ref, port, alternative, {}, {}},
      {},
      {},
      {},
      {}};
  for (const auto &value : service.arguments()) {
    auto endpoint = endpointForRole(engine, capability, value.role);
    if (!endpoint)
      return endpoint.takeError();
    option.actor.operandPorts.push_back(*endpoint);
    auto operand =
        service.argumentValue(actor.op, option.actor.operandPorts.size() - 1);
    if (!operand)
      return operand.takeError();
    option.operandOperationOrdinals.push_back((*operand)->getOperandNumber());
    const ::dataflow::CanonicalGraphConsumerEndpointRef consumer =
        ::dataflow::ActorTokenOperandRef{actor.ref,
                                         (*operand)->getOperandNumber()};
    auto producer = inputs.dataflow.graphProducer(consumer);
    if (!producer)
      return producer.takeError();
    if (std::holds_alternative<::dataflow::GraphIngressTokenRef>(*producer))
      option.boundaries.push_back(
          TechMemoryGraphBoundaryView{*producer, *endpoint});
  }
  for (const auto &value : service.results()) {
    auto endpoint = endpointForRole(engine, capability, value.role);
    if (!endpoint)
      return endpoint.takeError();
    option.actor.resultPorts.push_back(*endpoint);
    auto result =
        service.resultValue(actor.op, option.actor.resultPorts.size() - 1);
    if (!result)
      return result.takeError();
    option.resultOperationOrdinals.push_back(result->getResultNumber());
    const ::dataflow::CanonicalGraphProducerEndpointRef producer =
        ::dataflow::ActorTokenResultRef{actor.ref, result->getResultNumber()};
    auto consumers = inputs.dataflow.graphConsumers(producer);
    if (!consumers)
      return consumers.takeError();
    for (const auto &consumer : *consumers)
      if (std::holds_alternative<::dataflow::GraphEgressTokenRef>(consumer))
        option.boundaries.push_back(
            TechMemoryGraphBoundaryView{consumer, *endpoint});
  }

  const std::array<const MemoryActorOption *, 1> singleton{&option};
  auto boundaries = collectBoundaries(singleton, inputs.dataflow.identity());
  if (!boundaries)
    return boundaries.takeError();
  option.boundaries = std::move(*boundaries);

  auto key =
      canonicalTechMatchActorKey(option.actor, inputs.dataflow.identity());
  if (!key)
    return key.takeError();
  option.key = std::move(*key);
  return option;
}

llvm::Expected<std::vector<MemoryActorOption>> actorOptions(
    const TechMappingGenerationInputs &inputs,
    const ::dataflow::CanonicalActorView &actor,
    ::loom::fabric::FabricMemoryEngineTemplateRef engine,
    const ::loom::fabric::FabricMemoryEngineTemplateRecord &engineRecord) {
  auto projection =
      ::dataflow::projectRegisteredActorSchemaProjection(actor.op);
  if (!projection)
    return projection.takeError();
  auto service = ::dataflow::semantics::CanonicalService::forActor(actor.op);
  if (!service)
    return service.takeError();
  std::optional<::dataflow::semantics::CanonicalMemoryAccessView> access;
  if (service->kind() != ::dataflow::semantics::ServiceKind::MemoryFence) {
    auto projected =
        ::dataflow::semantics::getCanonicalMemoryAccessView(actor.op);
    if (!projected)
      return projected.takeError();
    access.emplace(std::move(*projected));
  }

  std::vector<MemoryActorOption> options;
  for (std::uint64_t portOrdinal = 0;
       portOrdinal < engineRecord.operationPorts.size(); ++portOrdinal) {
    const ::loom::fabric::FabricMemoryEngineTemplateOperationPortRef port{
        engine, portOrdinal};
    const auto *portView =
        inputs.fabric.memoryEngineTemplateOperationPort(port);
    if (!portView)
      return invalid("sealed Fabric memory operation port does not resolve");
    auto matches =
        portView->matchingCapabilities(*projection, *service, access);
    if (!matches)
      return matches.takeError();
    llvm::sort(*matches, [](const auto &lhs, const auto &rhs) {
      return lhs.alternativeOrdinal < rhs.alternativeOrdinal;
    });
    for (const ::fabric::MemoryCapabilityMatch &match : *matches) {
      const std::uint64_t alternativeOrdinal = match.alternativeOrdinal;
      const ::loom::fabric::FabricMemoryEngineTemplateCapabilityAlternativeRef
          alternative{port, alternativeOrdinal};
      const auto *capability =
          inputs.fabric.memoryEngineTemplateCapabilityAlternative(alternative);
      if (!capability)
        return invalid("memory capability alternative does not resolve");
      auto option = buildActorOption(inputs, actor, engine, port, alternative,
                                     *capability, *service);
      if (!option)
        return option.takeError();
      options.push_back(std::move(*option));
    }
  }
  llvm::sort(options, [](const auto &lhs, const auto &rhs) {
    return lhs.key < rhs.key;
  });
  options.erase(std::unique(options.begin(), options.end(),
                            [](const auto &lhs, const auto &rhs) {
                              return lhs.key == rhs.key;
                            }),
                options.end());
  return options;
}

llvm::Expected<std::optional<std::vector<TechMemoryGraphBoundaryView>>>
mergeBoundaries(llvm::ArrayRef<const MemoryActorOption *> selection,
                const ArtifactIdentity &owner) {
  auto merged = collectBoundaries(selection, owner);
  if (!merged)
    return merged.takeError();
  for (std::size_t index = 1; index < merged->size(); ++index) {
    auto previous = boundaryTerminalKey((*merged)[index - 1].terminal, owner);
    if (!previous)
      return previous.takeError();
    auto current = boundaryTerminalKey((*merged)[index].terminal, owner);
    if (!current)
      return current.takeError();
    if (*previous == *current &&
        (*merged)[index - 1].endpoint != (*merged)[index].endpoint)
      return std::optional<std::vector<TechMemoryGraphBoundaryView>>{};
  }
  return std::optional<std::vector<TechMemoryGraphBoundaryView>>(
      std::move(*merged));
}

llvm::Expected<std::vector<TechMemoryGraphBoundaryView>>
collectBoundaries(llvm::ArrayRef<const MemoryActorOption *> selection,
                  const ArtifactIdentity &owner) {
  using KeyedBoundary =
      std::pair<std::vector<std::uint8_t>, TechMemoryGraphBoundaryView>;
  std::vector<KeyedBoundary> keyed;
  for (const MemoryActorOption *option : selection)
    for (const TechMemoryGraphBoundaryView &boundary : option->boundaries) {
      auto key = boundaryTerminalKey(boundary.terminal, owner);
      if (!key)
        return key.takeError();
      keyed.emplace_back(std::move(*key), boundary);
    }
  llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
    if (lhs.first != rhs.first)
      return lhs.first < rhs.first;
    return ::loom::fabric::canonicalFabricBytes(lhs.second.endpoint) <
           ::loom::fabric::canonicalFabricBytes(rhs.second.endpoint);
  });

  std::vector<TechMemoryGraphBoundaryView> result;
  for (std::size_t index = 0; index < keyed.size(); ++index) {
    if (index != 0 && keyed[index - 1].first == keyed[index].first &&
        keyed[index - 1].second.endpoint == keyed[index].second.endpoint)
      continue;
    result.push_back(std::move(keyed[index].second));
  }
  return result;
}

const MemoryActorOption *
findActor(llvm::ArrayRef<const MemoryActorOption *> selection,
          ::dataflow::ActorRef actor) {
  auto found = llvm::find_if(selection, [&](const MemoryActorOption *option) {
    return option->actor.actor == actor;
  });
  return found == selection.end() ? nullptr : *found;
}

llvm::Expected<std::vector<TechMemoryInternalEdgeView>>
deriveInternalEdges(const TechMappingGenerationInputs &inputs,
                    ::loom::fabric::FabricMemoryEngineTemplateRef engine,
                    llvm::ArrayRef<const MemoryActorOption *> selection) {
  using KeyedEdge =
      std::pair<std::vector<std::uint8_t>, TechMemoryInternalEdgeView>;
  std::vector<KeyedEdge> keyed;
  for (const MemoryActorOption *source : selection) {
    for (auto [resultOrdinal, sourceEndpoint] :
         llvm::enumerate(source->actor.resultPorts)) {
      const ::dataflow::ActorTokenResultRef producer{
          source->actor.actor, source->resultOperationOrdinals[resultOrdinal]};
      auto consumers = inputs.dataflow.graphConsumers(
          ::dataflow::CanonicalGraphProducerEndpointRef{producer});
      if (!consumers)
        return consumers.takeError();
      for (const auto &consumerRef : *consumers) {
        const auto *consumer =
            std::get_if<::dataflow::ActorTokenOperandRef>(&consumerRef);
        if (!consumer)
          continue;
        const MemoryActorOption *sink = findActor(selection, consumer->actor);
        if (!sink)
          continue;
        for (auto [argumentOrdinal, sinkEndpoint] :
             llvm::enumerate(sink->actor.operandPorts)) {
          if (sink->operandOperationOrdinals[argumentOrdinal] !=
              consumer->ordinal)
            continue;
          const ::loom::fabric::FabricMemoryEngineTemplateInternalConnectionRef
              connection{engine, sourceEndpoint, sinkEndpoint};
          if (!inputs.fabric.hasMemoryEngineTemplateInternalConnection(
                  connection))
            break;
          TechMemoryInternalEdgeView edge{
              ::dataflow::CanonicalGraphProducerEndpointRef{producer},
              ::dataflow::CanonicalGraphConsumerEndpointRef{*consumer},
              connection};
          auto key = internalEdgeKey(edge, inputs.dataflow.identity());
          if (!key)
            return key.takeError();
          keyed.emplace_back(std::move(*key), std::move(edge));
          break;
        }
      }
    }
  }
  llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  std::vector<TechMemoryInternalEdgeView> edges;
  for (std::size_t index = 0; index < keyed.size(); ++index) {
    if (index != 0 && keyed[index - 1].first == keyed[index].first)
      continue;
    edges.push_back(std::move(keyed[index].second));
  }
  return edges;
}

class MemorySelectionCursor final {
public:
  MemorySelectionCursor(std::vector<MemoryActorDomain> domains,
                        std::size_t maxActorCount)
      : domains_(std::move(domains)), maxActorCount_(maxActorCount) {
    resetTarget();
  }

  std::optional<std::vector<const MemoryActorOption *>> next() {
    if (exhausted_)
      return std::nullopt;
    if (yielded_) {
      yielded_ = false;
      choices_.pop_back();
      frames_.resize(choices_.size() + 1);
    }

    while (targetActorCount_ <= maxActorCount_) {
      const std::size_t depth = choices_.size();
      if (depth == targetActorCount_) {
        std::vector<const MemoryActorOption *> selection;
        selection.reserve(choices_.size());
        for (const Choice &choice : choices_)
          selection.push_back(&domains_[choice.actor].options[choice.option]);
        yielded_ = true;
        return selection;
      }

      Frame &frame = frames_[depth];
      const std::size_t remaining = targetActorCount_ - depth;
      bool descended = false;
      while (frame.actor + remaining <= domains_.size()) {
        if (frame.graph && domains_[frame.actor].actor->graph != *frame.graph) {
          ++frame.actor;
          frame.option = 0;
          continue;
        }
        if (frame.option >= domains_[frame.actor].options.size()) {
          ++frame.actor;
          frame.option = 0;
          continue;
        }
        const std::size_t actor = frame.actor;
        const std::size_t option = frame.option++;
        const ::dataflow::GraphRef graph =
            frame.graph ? *frame.graph : domains_[actor].actor->graph;
        choices_.push_back({actor, option});
        frames_.push_back({actor + 1, 0, graph});
        descended = true;
        break;
      }
      if (descended)
        continue;

      if (depth == 0) {
        ++targetActorCount_;
        resetTarget();
        continue;
      }
      choices_.pop_back();
      frames_.resize(choices_.size() + 1);
    }
    exhausted_ = true;
    return std::nullopt;
  }

private:
  struct Choice final {
    std::size_t actor;
    std::size_t option;
  };
  struct Frame final {
    std::size_t actor;
    std::size_t option;
    std::optional<::dataflow::GraphRef> graph;
  };

  void resetTarget() {
    choices_.clear();
    frames_.clear();
    if (targetActorCount_ <= maxActorCount_)
      frames_.push_back({0, 0, std::nullopt});
  }

  std::vector<MemoryActorDomain> domains_;
  std::vector<Choice> choices_;
  std::vector<Frame> frames_;
  std::size_t maxActorCount_ = 0;
  std::size_t targetActorCount_ = 1;
  bool yielded_ = false;
  bool exhausted_ = false;
};

class ConstrainedInternalEdgeSubsetCursor final {
public:
  explicit ConstrainedInternalEdgeSubsetCursor(
      std::vector<TechMemoryInternalEdgeView> edges)
      : edges_(std::move(edges)), nextIndexByDepth_(1, 0) {}

  std::optional<std::vector<std::size_t>> next() {
    if (exhausted_)
      return std::nullopt;
    if (yielded_) {
      yielded_ = false;
      if (targetSize_ == 0) {
        ++targetSize_;
        resetTarget();
      } else {
        selected_.pop_back();
        nextIndexByDepth_.resize(selected_.size() + 1);
      }
    }

    while (targetSize_ <= edges_.size()) {
      const std::size_t depth = selected_.size();
      if (depth == targetSize_) {
        yielded_ = true;
        return selected_;
      }

      const std::size_t remaining = targetSize_ - depth;
      bool descended = false;
      std::size_t &nextIndex = nextIndexByDepth_[depth];
      while (nextIndex + remaining <= edges_.size()) {
        const std::size_t candidate = nextIndex++;
        if (!compatible(candidate))
          continue;
        selected_.push_back(candidate);
        nextIndexByDepth_.push_back(candidate + 1);
        descended = true;
        break;
      }
      if (descended)
        continue;

      if (depth == 0) {
        ++targetSize_;
        resetTarget();
        continue;
      }
      selected_.pop_back();
      nextIndexByDepth_.resize(selected_.size() + 1);
    }
    exhausted_ = true;
    return std::nullopt;
  }

  llvm::ArrayRef<TechMemoryInternalEdgeView> edges() const { return edges_; }

private:
  bool compatible(std::size_t candidate) const {
    const TechMemoryInternalEdgeView &edge = edges_[candidate];
    for (std::size_t selected : selected_) {
      const TechMemoryInternalEdgeView &other = edges_[selected];
      if (other.consumer == edge.consumer)
        return false;
      if (other.connection == edge.connection &&
          other.producer != edge.producer)
        return false;
    }
    return true;
  }

  void resetTarget() {
    selected_.clear();
    nextIndexByDepth_.assign(1, 0);
  }

  std::vector<TechMemoryInternalEdgeView> edges_;
  std::vector<std::size_t> selected_;
  std::vector<std::size_t> nextIndexByDepth_;
  std::size_t targetSize_ = 0;
  bool yielded_ = false;
  bool exhausted_ = false;
};

struct MemorySeedState final {
  MemorySeedState(TechMemoryRealizationView base,
                  std::vector<TechMemoryGraphBoundaryView> boundaries,
                  std::vector<::dataflow::ActorRef> coveredActors,
                  std::vector<TechMemoryInternalEdgeView> eligibleEdges)
      : base(std::move(base)), boundaries(std::move(boundaries)),
        coveredActors(std::move(coveredActors)),
        subsets(std::move(eligibleEdges)) {}

  TechMemoryRealizationView base;
  std::vector<TechMemoryGraphBoundaryView> boundaries;
  std::vector<::dataflow::ActorRef> coveredActors;
  ConstrainedInternalEdgeSubsetCursor subsets;
  std::optional<TechMemoryRealizationView> pending;
};

class MemoryRowFamilyCursor final : public TechMatchRowFamilyCursor {
public:
  MemoryRowFamilyCursor(
      const TechMappingGenerationInputs &inputs,
      ::loom::fabric::FabricMemoryEngineTemplateRef engine,
      const ::loom::fabric::FabricMemoryEngineTemplateRecord &engineRecord,
      MemorySelectionCursor selections)
      : inputs_(inputs), engine_(engine), engineRecord_(engineRecord),
        selections_(std::move(selections)) {}

  llvm::Error advance(TechMatchRowCollector &collector) override {
    while (!collector.truncated() && !collector.interrupted()) {
      if (!seed_) {
        auto prepared = prepareNextSelection();
        if (!prepared)
          return prepared.takeError();
        if (!*prepared) {
          exhausted_ = true;
          return llvm::Error::success();
        }
      }
      if (!seed_->pending) {
        auto prepared = prepareNextSubset();
        if (!prepared)
          return prepared.takeError();
        if (!*prepared) {
          seed_.reset();
          continue;
        }
      }

      auto key = canonicalTechMatchRowKey(*seed_->pending,
                                          inputs_.dataflow.identity());
      if (!key)
        return key.takeError();
      auto entered = collector.beginSeed(std::move(*key));
      if (!entered)
        return entered.takeError();
      if (!*entered)
        return llvm::Error::success();
      if (llvm::Error error =
              collector.admit(std::move(*seed_->pending), seed_->coveredActors))
        return error;
      seed_->pending.reset();
    }
    return llvm::Error::success();
  }

  bool exhausted() const override { return exhausted_; }

private:
  llvm::Expected<bool> prepareNextSelection() {
    while (auto selection = selections_.next()) {
      auto boundaries =
          collectBoundaries(*selection, inputs_.dataflow.identity());
      if (!boundaries)
        return boundaries.takeError();
      auto mergedBoundaries =
          mergeBoundaries(*selection, inputs_.dataflow.identity());
      if (!mergedBoundaries)
        return mergedBoundaries.takeError();
      if (!*mergedBoundaries)
        continue;

      bool capacityAdmitted = true;
      if (engineRecord_.schedule == ::fabric::Schedule::Temporal) {
        capacityAdmitted =
            engineRecord_.residentContextCount &&
            selection->size() <= *engineRecord_.residentContextCount;
      } else {
        std::vector<std::uint64_t> selectedPorts;
        selectedPorts.reserve(selection->size());
        for (const MemoryActorOption *option : *selection)
          selectedPorts.push_back(option->actor.operationPort.ordinal);
        llvm::sort(selectedPorts);
        capacityAdmitted =
            std::adjacent_find(selectedPorts.begin(), selectedPorts.end()) ==
            selectedPorts.end();
      }
      if (!capacityAdmitted)
        continue;

      auto internalEdges = deriveInternalEdges(inputs_, engine_, *selection);
      if (!internalEdges)
        return internalEdges.takeError();
      std::vector<TechMemoryActorView> actors;
      std::vector<::dataflow::ActorRef> covered;
      actors.reserve(selection->size());
      covered.reserve(selection->size());
      for (const MemoryActorOption *option : *selection) {
        actors.push_back(option->actor);
        covered.push_back(option->actor.actor);
      }
      seed_.emplace(
          TechMemoryRealizationView{
              0, engine_, std::move(actors), std::move(*boundaries), {}},
          std::move(**mergedBoundaries), std::move(covered),
          std::move(*internalEdges));
      return true;
    }
    return false;
  }

  llvm::Expected<bool> prepareNextSubset() {
    while (auto subset = seed_->subsets.next()) {
      TechMemoryRealizationView row = seed_->base;
      row.internalEdges.reserve(subset->size());
      for (std::size_t edge : *subset)
        row.internalEdges.push_back(seed_->subsets.edges()[edge]);
      auto legality = deriveTechMemoryInternalConnectionLegality(
          row.internalEdges, inputs_.dataflow.identity());
      if (!legality)
        return legality.takeError();
      if (*legality != TechMemoryInternalConnectionLegality::Admissible)
        return invalid("constrained internal-edge cursor emitted an illegal "
                       "partial assignment");
      if (engineRecord_.schedule == ::fabric::Schedule::Temporal) {
        auto distinct =
            techMemoryExternalIngressesAreDistinct(row, inputs_.dataflow);
        if (!distinct)
          return distinct.takeError();
        if (!*distinct)
          continue;
      }
      row.graphBoundaries = seed_->boundaries;
      seed_->pending.emplace(std::move(row));
      return true;
    }
    return false;
  }

  const TechMappingGenerationInputs &inputs_;
  ::loom::fabric::FabricMemoryEngineTemplateRef engine_;
  const ::loom::fabric::FabricMemoryEngineTemplateRecord &engineRecord_;
  MemorySelectionCursor selections_;
  std::optional<MemorySeedState> seed_;
  bool exhausted_ = false;
};

} // namespace

llvm::Expected<std::unique_ptr<TechMatchRowFamilyCursor>>
createMemoryRowFamilyCursor(
    const TechMappingGenerationInputs &inputs,
    llvm::ArrayRef<::dataflow::CanonicalActorView> selectedActors,
    ::loom::fabric::FabricMemoryEngineTemplateRef family) {
  const auto *engineRecord = inputs.fabric.memoryEngineTemplate(family);
  if (!engineRecord)
    return invalid("sealed Fabric memory template does not resolve");

  std::vector<MemoryActorDomain> domains;
  std::map<std::uint64_t, std::size_t> graphActorCounts;
  for (const ::dataflow::CanonicalActorView &actor : selectedActors) {
    if (actor.kind != ::dataflow::CanonicalDataflowActorKind::Memory)
      continue;
    auto options = actorOptions(inputs, actor, family, *engineRecord);
    if (!options)
      return options.takeError();
    domains.push_back(MemoryActorDomain{&actor, std::move(*options)});
    ++graphActorCounts[actor.graph.entity.value()];
  }
  llvm::sort(domains, [](const auto &lhs, const auto &rhs) {
    return lhs.actor->ref.entity.value() < rhs.actor->ref.entity.value();
  });
  const std::size_t maxActorCount =
      graphActorCounts.empty()
          ? 0
          : llvm::max_element(graphActorCounts, [](const auto &lhs,
                                                   const auto &rhs) {
              return lhs.second < rhs.second;
            })->second;
  std::unique_ptr<TechMatchRowFamilyCursor> cursor =
      std::make_unique<MemoryRowFamilyCursor>(
          inputs, family, *engineRecord,
          MemorySelectionCursor(std::move(domains), maxActorCount));
  return cursor;
}

std::vector<::loom::fabric::FabricMemoryEngineTemplateRef>
deriveMemoryRowFamilies(
    const TechMappingGenerationInputs &inputs,
    llvm::ArrayRef<::dataflow::CanonicalActorView> selectedActors) {
  std::set<::dataflow::OperationSchemaId> selectedSchemas;
  for (const ::dataflow::CanonicalActorView &actor : selectedActors) {
    if (actor.kind != ::dataflow::CanonicalDataflowActorKind::Memory)
      continue;
    const auto schema = ::dataflow::operationSchemaOf(actor.op);
    if (schema)
      selectedSchemas.insert(*schema);
  }
  if (selectedSchemas.empty())
    return {};
  std::vector<::loom::fabric::FabricMemoryEngineTemplateRef> families;
  for (const auto engine : inputs.fabric.memoryEngineTemplates()) {
    const auto *record = inputs.fabric.memoryEngineTemplate(engine);
    if (!record)
      continue;
    const bool supported =
        llvm::any_of(record->operationPorts, [&](const auto &port) {
          return llvm::any_of(
              port.capabilityAlternatives(), [&](const auto &alternative) {
                return selectedSchemas.count(
                           alternative.actorContractDomain.actorSchema()) != 0;
              });
        });
    if (supported)
      families.push_back(engine);
  }
  llvm::sort(families, [](const auto &lhs, const auto &rhs) {
    return ::loom::fabric::canonicalFabricBytes(lhs) <
           ::loom::fabric::canonicalFabricBytes(rhs);
  });
  return families;
}

llvm::Error
deriveMemoryRows(const TechMappingGenerationInputs &inputs,
                 llvm::ArrayRef<::dataflow::CanonicalActorView> selectedActors,
                 ::loom::fabric::FabricMemoryEngineTemplateRef family,
                 TechMatchRowCollector &collector) {
  auto cursor = createMemoryRowFamilyCursor(inputs, selectedActors, family);
  if (!cursor)
    return cursor.takeError();
  return (*cursor)->advance(collector);
}

} // namespace loom::mapping::detail

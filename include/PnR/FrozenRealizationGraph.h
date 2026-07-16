#ifndef LOOM_PNR_FROZENREALIZATIONGRAPH_H
#define LOOM_PNR_FROZENREALIZATIONGRAPH_H

#include "Mapping/Verifier.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {

enum class FrozenRealizationKind { Compute, Memory };

struct FrozenActorOwnership {
  mapping::ActorId actor;
  FrozenRealizationKind kind;
  PnrIndex realization;

  friend bool operator==(const FrozenActorOwnership &lhs,
                         const FrozenActorOwnership &rhs) {
    return lhs.actor == rhs.actor && lhs.kind == rhs.kind &&
           lhs.realization == rhs.realization;
  }
};

struct FrozenComputeRealization {
  mapping::ComputeRealizationId id;
  mapping::FuId fu;
  mapping::EncodingId encoding;

  friend bool operator==(const FrozenComputeRealization &lhs,
                         const FrozenComputeRealization &rhs) {
    return lhs.id == rhs.id && lhs.fu == rhs.fu && lhs.encoding == rhs.encoding;
  }
};

struct FrozenMemoryRealization {
  mapping::MemoryRealizationId id;
  mapping::MemorySemanticEncodingId encoding;
  mapping::MemoryImplementationId implementation;
  mapping::MemoryServiceDomainId service;

  friend bool operator==(const FrozenMemoryRealization &lhs,
                         const FrozenMemoryRealization &rhs) {
    return lhs.id == rhs.id && lhs.encoding == rhs.encoding &&
           lhs.implementation == rhs.implementation &&
           lhs.service == rhs.service;
  }
};

struct FrozenComputeTemplateTerminal {
  PnrIndex realization;
  mapping::FuId fu;
  mapping::PortDirection direction;
  PnrIndex port;

  friend bool operator==(const FrozenComputeTemplateTerminal &lhs,
                         const FrozenComputeTemplateTerminal &rhs) {
    return lhs.realization == rhs.realization && lhs.fu == rhs.fu &&
           lhs.direction == rhs.direction && lhs.port == rhs.port;
  }
};

struct FrozenMemoryTemplateTerminal {
  PnrIndex realization;
  mapping::MemoryOperationPortTemplateId operation;
  mapping::PortDirection direction;
  PnrIndex port;

  friend bool operator==(const FrozenMemoryTemplateTerminal &lhs,
                         const FrozenMemoryTemplateTerminal &rhs) {
    return lhs.realization == rhs.realization &&
           lhs.operation == rhs.operation && lhs.direction == rhs.direction &&
           lhs.port == rhs.port;
  }
};

using FrozenTemplateTerminal =
    std::variant<FrozenComputeTemplateTerminal, FrozenMemoryTemplateTerminal>;

struct FrozenGraphBoundaryTerminal {
  mapping::GraphId graph;
  mapping::PortDirection direction;
  PnrIndex port;

  friend bool operator==(const FrozenGraphBoundaryTerminal &lhs,
                         const FrozenGraphBoundaryTerminal &rhs) {
    return lhs.graph == rhs.graph && lhs.direction == rhs.direction &&
           lhs.port == rhs.port;
  }
};

struct FrozenTemplateTerminalRef {
  PnrIndex terminal;

  friend bool operator==(FrozenTemplateTerminalRef lhs,
                         FrozenTemplateTerminalRef rhs) {
    return lhs.terminal == rhs.terminal;
  }
};

using FrozenTerminal =
    std::variant<FrozenGraphBoundaryTerminal, FrozenTemplateTerminalRef>;

struct FrozenLogicalNet {
  FrozenTerminal source;
  PnrIndex sinkOffset;
  PnrIndex sinkCount;

  friend bool operator==(const FrozenLogicalNet &lhs,
                         const FrozenLogicalNet &rhs) {
    return lhs.source == rhs.source && lhs.sinkOffset == rhs.sinkOffset &&
           lhs.sinkCount == rhs.sinkCount;
  }
};

struct FrozenLogicalNetSink {
  mapping::EdgeId edge;
  FrozenTerminal terminal;

  friend bool operator==(const FrozenLogicalNetSink &lhs,
                         const FrozenLogicalNetSink &rhs) {
    return lhs.edge == rhs.edge && lhs.terminal == rhs.terminal;
  }
};

struct FrozenMemoryServiceObligation {
  mapping::LogicalMemoryRootId root;
  mapping::MemoryServiceDomainId service;

  friend bool operator==(const FrozenMemoryServiceObligation &lhs,
                         const FrozenMemoryServiceObligation &rhs) {
    return lhs.root == rhs.root && lhs.service == rhs.service;
  }
};

class FrozenRealizationGraph {
public:
  llvm::ArrayRef<FrozenActorOwnership> actorOwnerships() const {
    return actorOwnerships_;
  }
  llvm::ArrayRef<FrozenComputeRealization> computeRealizations() const {
    return computeRealizations_;
  }
  llvm::ArrayRef<FrozenMemoryRealization> memoryRealizations() const {
    return memoryRealizations_;
  }
  llvm::ArrayRef<FrozenTemplateTerminal> templateTerminals() const {
    return templateTerminals_;
  }
  llvm::ArrayRef<FrozenLogicalNet> logicalNets() const { return logicalNets_; }
  llvm::ArrayRef<FrozenLogicalNetSink> logicalNetSinks() const {
    return logicalNetSinks_;
  }
  llvm::ArrayRef<FrozenMemoryServiceObligation>
  memoryServiceObligations() const {
    return memoryServiceObligations_;
  }

  friend bool operator==(const FrozenRealizationGraph &lhs,
                         const FrozenRealizationGraph &rhs) {
    return lhs.actorOwnerships_ == rhs.actorOwnerships_ &&
           lhs.computeRealizations_ == rhs.computeRealizations_ &&
           lhs.memoryRealizations_ == rhs.memoryRealizations_ &&
           lhs.templateTerminals_ == rhs.templateTerminals_ &&
           lhs.logicalNets_ == rhs.logicalNets_ &&
           lhs.logicalNetSinks_ == rhs.logicalNetSinks_ &&
           lhs.memoryServiceObligations_ == rhs.memoryServiceObligations_;
  }
  friend bool operator!=(const FrozenRealizationGraph &lhs,
                         const FrozenRealizationGraph &rhs) {
    return !(lhs == rhs);
  }

private:
  FrozenRealizationGraph(
      std::vector<FrozenActorOwnership> actorOwnerships,
      std::vector<FrozenComputeRealization> computeRealizations,
      std::vector<FrozenMemoryRealization> memoryRealizations,
      std::vector<FrozenTemplateTerminal> templateTerminals,
      std::vector<FrozenLogicalNet> logicalNets,
      std::vector<FrozenLogicalNetSink> logicalNetSinks,
      std::vector<FrozenMemoryServiceObligation> memoryServiceObligations)
      : actorOwnerships_(std::move(actorOwnerships)),
        computeRealizations_(std::move(computeRealizations)),
        memoryRealizations_(std::move(memoryRealizations)),
        templateTerminals_(std::move(templateTerminals)),
        logicalNets_(std::move(logicalNets)),
        logicalNetSinks_(std::move(logicalNetSinks)),
        memoryServiceObligations_(std::move(memoryServiceObligations)) {}

  std::vector<FrozenActorOwnership> actorOwnerships_;
  std::vector<FrozenComputeRealization> computeRealizations_;
  std::vector<FrozenMemoryRealization> memoryRealizations_;
  std::vector<FrozenTemplateTerminal> templateTerminals_;
  std::vector<FrozenLogicalNet> logicalNets_;
  std::vector<FrozenLogicalNetSink> logicalNetSinks_;
  std::vector<FrozenMemoryServiceObligation> memoryServiceObligations_;

  friend llvm::Expected<FrozenRealizationGraph>
  freezeRealizationGraph(const mapping::DataflowProgramView &dataflow,
                         const mapping::FabricHardwareView &fabric,
                         const mapping::ValidatedTechMapping &mapping);
};

llvm::Expected<FrozenRealizationGraph>
freezeRealizationGraph(const mapping::DataflowProgramView &dataflow,
                       const mapping::FabricHardwareView &fabric,
                       const mapping::ValidatedTechMapping &mapping);

namespace detail {

llvm::Error preflightFrozenRealizationGraphCapacity(
    llvm::ArrayRef<mapping::ComputeRealizationDraft> computeRealizations,
    llvm::ArrayRef<mapping::MemoryRealizationDraft> memoryRealizations,
    std::uint64_t canonicalEdgeCount);

} // namespace detail

} // namespace loom::pnr

#endif // LOOM_PNR_FROZENREALIZATIONGRAPH_H

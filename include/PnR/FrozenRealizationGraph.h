#ifndef LOOM_PNR_FROZENREALIZATIONGRAPH_H
#define LOOM_PNR_FROZENREALIZATIONGRAPH_H

#include "Mapping/Artifact.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace loom::pnr {

struct PnrProblemInputs;

enum class FrozenMappingInfeasibilityCode {
  EmptyConcreteFuDomain,
  EmptyUnaryEligibleDomain,
  EmptyConcreteMemoryDomain,
  EmptyMemoryUnaryEligibleDomain,
};

using FrozenRealizationId =
    std::variant<mapping::ComputeRealizationId, mapping::MemoryRealizationId>;

class FrozenMappingInfeasibility final
    : public llvm::ErrorInfo<FrozenMappingInfeasibility> {
public:
  static char ID;

  FrozenMappingInfeasibility(FrozenMappingInfeasibilityCode code,
                             FrozenRealizationId realization,
                             std::string message)
      : code_(code), realization_(realization), message_(std::move(message)) {}

  FrozenMappingInfeasibilityCode code() const { return code_; }
  const mapping::ComputeRealizationId *computeRealization() const {
    return std::get_if<mapping::ComputeRealizationId>(&realization_);
  }
  const mapping::MemoryRealizationId *memoryRealization() const {
    return std::get_if<mapping::MemoryRealizationId>(&realization_);
  }
  const FrozenRealizationId &realizationId() const { return realization_; }
  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  FrozenMappingInfeasibilityCode code_;
  FrozenRealizationId realization_;
  std::string message_;
};

enum class FrozenRealizationKind { Compute, Memory };

struct FabricPeOccurrenceRef {
  mapping::ComputeOccurrenceId occurrence;

  friend bool operator==(FabricPeOccurrenceRef lhs, FabricPeOccurrenceRef rhs) {
    return lhs.occurrence == rhs.occurrence;
  }
  friend bool operator!=(FabricPeOccurrenceRef lhs, FabricPeOccurrenceRef rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(FabricPeOccurrenceRef lhs, FabricPeOccurrenceRef rhs) {
    return lhs.occurrence.value() < rhs.occurrence.value();
  }
};

struct FabricFuOccurrenceRef {
  FabricPeOccurrenceRef parentPe;
  mapping::FuId implementation;

  friend bool operator==(FabricFuOccurrenceRef lhs, FabricFuOccurrenceRef rhs) {
    return lhs.parentPe == rhs.parentPe &&
           lhs.implementation == rhs.implementation;
  }
  friend bool operator!=(FabricFuOccurrenceRef lhs, FabricFuOccurrenceRef rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(FabricFuOccurrenceRef lhs, FabricFuOccurrenceRef rhs) {
    if (lhs.parentPe != rhs.parentPe)
      return lhs.parentPe < rhs.parentPe;
    return lhs.implementation.value() < rhs.implementation.value();
  }
};

struct FabricMemoryOccurrenceRef {
  mapping::MemoryOccurrenceId occurrence;

  friend bool operator==(FabricMemoryOccurrenceRef lhs,
                         FabricMemoryOccurrenceRef rhs) {
    return lhs.occurrence == rhs.occurrence;
  }
  friend bool operator!=(FabricMemoryOccurrenceRef lhs,
                         FabricMemoryOccurrenceRef rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(FabricMemoryOccurrenceRef lhs,
                        FabricMemoryOccurrenceRef rhs) {
    return lhs.occurrence.value() < rhs.occurrence.value();
  }
};

class ContextOrdinal {
public:
  explicit constexpr ContextOrdinal(std::uint64_t value) : value_(value) {}

  constexpr std::uint64_t value() const { return value_; }

  friend constexpr bool operator==(ContextOrdinal lhs, ContextOrdinal rhs) {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr bool operator!=(ContextOrdinal lhs, ContextOrdinal rhs) {
    return !(lhs == rhs);
  }
  friend constexpr bool operator<(ContextOrdinal lhs, ContextOrdinal rhs) {
    return lhs.value_ < rhs.value_;
  }

private:
  std::uint64_t value_;
};

struct InstructionContextRef {
  FabricPeOccurrenceRef pe;
  ContextOrdinal ordinal;

  friend bool operator==(InstructionContextRef lhs, InstructionContextRef rhs) {
    return lhs.pe == rhs.pe && lhs.ordinal == rhs.ordinal;
  }
  friend bool operator!=(InstructionContextRef lhs, InstructionContextRef rhs) {
    return !(lhs == rhs);
  }
  friend bool operator<(InstructionContextRef lhs, InstructionContextRef rhs) {
    if (lhs.pe != rhs.pe)
      return lhs.pe < rhs.pe;
    return lhs.ordinal < rhs.ordinal;
  }
};

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
  PnrIndex implDomainOffset;
  PnrIndex implDomainCount;

  friend bool operator==(const FrozenComputeRealization &lhs,
                         const FrozenComputeRealization &rhs) {
    return lhs.id == rhs.id && lhs.fu == rhs.fu &&
           lhs.encoding == rhs.encoding &&
           lhs.implDomainOffset == rhs.implDomainOffset &&
           lhs.implDomainCount == rhs.implDomainCount;
  }
};

struct FrozenFabricPeOccurrence {
  FabricPeOccurrenceRef ref;
  mapping::ComputeScheduleKind schedule;
  PnrIndex contextCount;
  PnrIndex fuOccurrenceOffset;
  PnrIndex fuOccurrenceCount;
  PnrIndex endpointOffset;
  PnrIndex endpointCount;
  PnrIndex localArcOffset;
  PnrIndex localArcCount;

  friend bool operator==(const FrozenFabricPeOccurrence &lhs,
                         const FrozenFabricPeOccurrence &rhs) {
    return lhs.ref == rhs.ref && lhs.schedule == rhs.schedule &&
           lhs.contextCount == rhs.contextCount &&
           lhs.fuOccurrenceOffset == rhs.fuOccurrenceOffset &&
           lhs.fuOccurrenceCount == rhs.fuOccurrenceCount &&
           lhs.endpointOffset == rhs.endpointOffset &&
           lhs.endpointCount == rhs.endpointCount &&
           lhs.localArcOffset == rhs.localArcOffset &&
           lhs.localArcCount == rhs.localArcCount;
  }
};

struct FrozenFabricFuOccurrence {
  FabricFuOccurrenceRef ref;

  friend bool operator==(const FrozenFabricFuOccurrence &lhs,
                         const FrozenFabricFuOccurrence &rhs) {
    return lhs.ref == rhs.ref;
  }
};

struct FrozenPhysicalEndpoint {
  FabricPeOccurrenceRef parentPe;
  mapping::ComputeEndpointId id;
  mapping::PortDirection direction;
  mapping::PortKind kind;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
  PnrIndex compatibleTypeOffset;
  PnrIndex compatibleTypeCount;
  mapping::PortRoleKey role;

  friend bool operator==(const FrozenPhysicalEndpoint &lhs,
                         const FrozenPhysicalEndpoint &rhs) {
    return lhs.parentPe == rhs.parentPe && lhs.id == rhs.id &&
           lhs.direction == rhs.direction && lhs.kind == rhs.kind &&
           lhs.payloadCapacityBits == rhs.payloadCapacityBits &&
           lhs.tagCapacityBits == rhs.tagCapacityBits &&
           lhs.compatibleTypeOffset == rhs.compatibleTypeOffset &&
           lhs.compatibleTypeCount == rhs.compatibleTypeCount &&
           lhs.role == rhs.role;
  }
};

struct FrozenComputeLocalArc {
  FabricFuOccurrenceRef fuOccurrence;
  mapping::PortDirection direction;
  PnrIndex port;
  PnrIndex endpoint;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;

  friend bool operator==(const FrozenComputeLocalArc &lhs,
                         const FrozenComputeLocalArc &rhs) {
    return lhs.fuOccurrence == rhs.fuOccurrence &&
           lhs.direction == rhs.direction && lhs.port == rhs.port &&
           lhs.endpoint == rhs.endpoint &&
           lhs.payloadCapacityBits == rhs.payloadCapacityBits &&
           lhs.tagCapacityBits == rhs.tagCapacityBits;
  }
};

struct FrozenFabricMemoryOccurrence {
  FabricMemoryOccurrenceRef ref;
  mapping::MemoryImplementationId implementation;
  PnrIndex endpointOffset;
  PnrIndex endpointCount;
  PnrIndex localArcOffset;
  PnrIndex localArcCount;

  friend bool operator==(const FrozenFabricMemoryOccurrence &lhs,
                         const FrozenFabricMemoryOccurrence &rhs) {
    return lhs.ref == rhs.ref && lhs.implementation == rhs.implementation &&
           lhs.endpointOffset == rhs.endpointOffset &&
           lhs.endpointCount == rhs.endpointCount &&
           lhs.localArcOffset == rhs.localArcOffset &&
           lhs.localArcCount == rhs.localArcCount;
  }
};

struct FrozenMemoryPhysicalEndpoint {
  FabricMemoryOccurrenceRef parentMemory;
  mapping::MemoryEndpointId id;
  mapping::PortDirection direction;
  mapping::PortKind kind;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;
  PnrIndex compatibleTypeOffset;
  PnrIndex compatibleTypeCount;
  mapping::PortRoleKey role;

  friend bool operator==(const FrozenMemoryPhysicalEndpoint &lhs,
                         const FrozenMemoryPhysicalEndpoint &rhs) {
    return lhs.parentMemory == rhs.parentMemory && lhs.id == rhs.id &&
           lhs.direction == rhs.direction && lhs.kind == rhs.kind &&
           lhs.payloadCapacityBits == rhs.payloadCapacityBits &&
           lhs.tagCapacityBits == rhs.tagCapacityBits &&
           lhs.compatibleTypeOffset == rhs.compatibleTypeOffset &&
           lhs.compatibleTypeCount == rhs.compatibleTypeCount &&
           lhs.role == rhs.role;
  }
};

struct FrozenMemoryLocalArc {
  FabricMemoryOccurrenceRef memoryOccurrence;
  mapping::MemoryOperationPortTemplateId operation;
  mapping::PortDirection direction;
  PnrIndex port;
  PnrIndex endpoint;
  std::uint32_t payloadCapacityBits;
  std::uint32_t tagCapacityBits;

  friend bool operator==(const FrozenMemoryLocalArc &lhs,
                         const FrozenMemoryLocalArc &rhs) {
    return lhs.memoryOccurrence == rhs.memoryOccurrence &&
           lhs.operation == rhs.operation && lhs.direction == rhs.direction &&
           lhs.port == rhs.port && lhs.endpoint == rhs.endpoint &&
           lhs.payloadCapacityBits == rhs.payloadCapacityBits &&
           lhs.tagCapacityBits == rhs.tagCapacityBits;
  }
};

struct FrozenImplementationOccurrence {
  PnrIndex realization;
  FabricFuOccurrenceRef fuOccurrence;
  PnrIndex portDemandOffset;
  PnrIndex portDemandCount;
  bool unaryEligible;

  friend bool operator==(const FrozenImplementationOccurrence &lhs,
                         const FrozenImplementationOccurrence &rhs) {
    return lhs.realization == rhs.realization &&
           lhs.fuOccurrence == rhs.fuOccurrence &&
           lhs.portDemandOffset == rhs.portDemandOffset &&
           lhs.portDemandCount == rhs.portDemandCount &&
           lhs.unaryEligible == rhs.unaryEligible;
  }
};

struct FrozenMemoryImplementationOccurrence {
  PnrIndex realization;
  FabricMemoryOccurrenceRef memoryOccurrence;
  PnrIndex portDemandOffset;
  PnrIndex portDemandCount;
  bool unaryEligible;

  friend bool operator==(const FrozenMemoryImplementationOccurrence &lhs,
                         const FrozenMemoryImplementationOccurrence &rhs) {
    return lhs.realization == rhs.realization &&
           lhs.memoryOccurrence == rhs.memoryOccurrence &&
           lhs.portDemandOffset == rhs.portDemandOffset &&
           lhs.portDemandCount == rhs.portDemandCount &&
           lhs.unaryEligible == rhs.unaryEligible;
  }
};

struct FrozenPortDemand {
  PnrIndex implementation;
  mapping::FuId fu;
  mapping::PortDirection direction;
  PnrIndex port;
  mapping::PortKind kind;
  mapping::TypeKey type;
  mapping::PortRoleKey role;
  std::uint32_t payloadWidthBits;
  std::uint32_t tagWidthBits;
  PnrIndex endpointOffset;
  PnrIndex endpointCount;

  friend bool operator==(const FrozenPortDemand &lhs,
                         const FrozenPortDemand &rhs) {
    return lhs.implementation == rhs.implementation && lhs.fu == rhs.fu &&
           lhs.direction == rhs.direction && lhs.port == rhs.port &&
           lhs.kind == rhs.kind && lhs.type == rhs.type &&
           lhs.role == rhs.role &&
           lhs.payloadWidthBits == rhs.payloadWidthBits &&
           lhs.tagWidthBits == rhs.tagWidthBits &&
           lhs.endpointOffset == rhs.endpointOffset &&
           lhs.endpointCount == rhs.endpointCount;
  }
};

struct FrozenMemoryPortDemand {
  PnrIndex implementation;
  mapping::MemoryOperationPortTemplateId operation;
  mapping::PortDirection direction;
  PnrIndex port;
  mapping::PortKind kind;
  mapping::TypeKey type;
  mapping::PortRoleKey role;
  std::uint32_t payloadWidthBits;
  std::uint32_t tagWidthBits;
  PnrIndex endpointOffset;
  PnrIndex endpointCount;

  friend bool operator==(const FrozenMemoryPortDemand &lhs,
                         const FrozenMemoryPortDemand &rhs) {
    return lhs.implementation == rhs.implementation &&
           lhs.operation == rhs.operation && lhs.direction == rhs.direction &&
           lhs.port == rhs.port && lhs.kind == rhs.kind &&
           lhs.type == rhs.type && lhs.role == rhs.role &&
           lhs.payloadWidthBits == rhs.payloadWidthBits &&
           lhs.tagWidthBits == rhs.tagWidthBits &&
           lhs.endpointOffset == rhs.endpointOffset &&
           lhs.endpointCount == rhs.endpointCount;
  }
};

struct FrozenMemoryRealization {
  mapping::MemoryRealizationId id;
  mapping::MemorySemanticEncodingId encoding;
  mapping::MemoryImplementationId implementation;
  mapping::MemoryServiceDomainId service;
  PnrIndex implDomainOffset;
  PnrIndex implDomainCount;

  friend bool operator==(const FrozenMemoryRealization &lhs,
                         const FrozenMemoryRealization &rhs) {
    return lhs.id == rhs.id && lhs.encoding == rhs.encoding &&
           lhs.implementation == rhs.implementation &&
           lhs.service == rhs.service &&
           lhs.implDomainOffset == rhs.implDomainOffset &&
           lhs.implDomainCount == rhs.implDomainCount;
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
  mapping::DataflowEndpoint producer;
  FrozenTerminal source;
  PnrIndex sinkOffset;
  PnrIndex sinkCount;

  friend bool operator==(const FrozenLogicalNet &lhs,
                         const FrozenLogicalNet &rhs) {
    return lhs.producer == rhs.producer && lhs.source == rhs.source &&
           lhs.sinkOffset == rhs.sinkOffset && lhs.sinkCount == rhs.sinkCount;
  }
};

struct FrozenLogicalNetSink {
  mapping::DataflowEndpoint consumer;
  FrozenTerminal terminal;

  friend bool operator==(const FrozenLogicalNetSink &lhs,
                         const FrozenLogicalNetSink &rhs) {
    return lhs.consumer == rhs.consumer && lhs.terminal == rhs.terminal;
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
  llvm::ArrayRef<FrozenFabricPeOccurrence> fabricPeOccurrences() const {
    return fabricPeOccurrences_;
  }
  llvm::ArrayRef<FrozenFabricFuOccurrence> fabricFuOccurrences() const {
    return fabricFuOccurrences_;
  }
  const FrozenFabricPeOccurrence *
  findFabricPeOccurrence(FabricPeOccurrenceRef ref) const;
  const FrozenFabricFuOccurrence *
  findFabricFuOccurrence(FabricFuOccurrenceRef ref) const;
  std::optional<InstructionContextRef>
  instructionContext(FabricFuOccurrenceRef fuOccurrence,
                     ContextOrdinal ordinal) const;
  llvm::ArrayRef<FrozenPhysicalEndpoint> physicalEndpoints() const {
    return physicalEndpoints_;
  }
  llvm::ArrayRef<mapping::TypeKey> physicalEndpointCompatibleTypes() const {
    return physicalEndpointCompatibleTypes_;
  }
  llvm::ArrayRef<FrozenComputeLocalArc> computeLocalArcs() const {
    return computeLocalArcs_;
  }
  llvm::ArrayRef<FrozenImplementationOccurrence>
  implementationOccurrences() const {
    return implementationOccurrences_;
  }
  llvm::ArrayRef<FrozenPortDemand> portDemands() const { return portDemands_; }
  llvm::ArrayRef<PnrIndex> compatibleEndpoints() const {
    return compatibleEndpoints_;
  }
  llvm::ArrayRef<FrozenFabricMemoryOccurrence> fabricMemoryOccurrences() const {
    return fabricMemoryOccurrences_;
  }
  const FrozenFabricMemoryOccurrence *
  findFabricMemoryOccurrence(FabricMemoryOccurrenceRef ref) const;
  llvm::ArrayRef<FrozenMemoryPhysicalEndpoint> memoryPhysicalEndpoints() const {
    return memoryPhysicalEndpoints_;
  }
  llvm::ArrayRef<mapping::TypeKey>
  memoryPhysicalEndpointCompatibleTypes() const {
    return memoryPhysicalEndpointCompatibleTypes_;
  }
  llvm::ArrayRef<FrozenMemoryLocalArc> memoryLocalArcs() const {
    return memoryLocalArcs_;
  }
  llvm::ArrayRef<FrozenMemoryImplementationOccurrence>
  memoryImplementationOccurrences() const {
    return memoryImplementationOccurrences_;
  }
  llvm::ArrayRef<FrozenMemoryPortDemand> memoryPortDemands() const {
    return memoryPortDemands_;
  }
  llvm::ArrayRef<PnrIndex> compatibleMemoryEndpoints() const {
    return compatibleMemoryEndpoints_;
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
           lhs.fabricPeOccurrences_ == rhs.fabricPeOccurrences_ &&
           lhs.fabricFuOccurrences_ == rhs.fabricFuOccurrences_ &&
           lhs.physicalEndpoints_ == rhs.physicalEndpoints_ &&
           lhs.physicalEndpointCompatibleTypes_ ==
               rhs.physicalEndpointCompatibleTypes_ &&
           lhs.computeLocalArcs_ == rhs.computeLocalArcs_ &&
           lhs.implementationOccurrences_ == rhs.implementationOccurrences_ &&
           lhs.portDemands_ == rhs.portDemands_ &&
           lhs.compatibleEndpoints_ == rhs.compatibleEndpoints_ &&
           lhs.fabricMemoryOccurrences_ == rhs.fabricMemoryOccurrences_ &&
           lhs.memoryPhysicalEndpoints_ == rhs.memoryPhysicalEndpoints_ &&
           lhs.memoryPhysicalEndpointCompatibleTypes_ ==
               rhs.memoryPhysicalEndpointCompatibleTypes_ &&
           lhs.memoryLocalArcs_ == rhs.memoryLocalArcs_ &&
           lhs.memoryImplementationOccurrences_ ==
               rhs.memoryImplementationOccurrences_ &&
           lhs.memoryPortDemands_ == rhs.memoryPortDemands_ &&
           lhs.compatibleMemoryEndpoints_ == rhs.compatibleMemoryEndpoints_ &&
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
      std::vector<FrozenFabricPeOccurrence> fabricPeOccurrences,
      std::vector<FrozenFabricFuOccurrence> fabricFuOccurrences,
      std::vector<FrozenPhysicalEndpoint> physicalEndpoints,
      std::vector<mapping::TypeKey> physicalEndpointCompatibleTypes,
      std::vector<FrozenComputeLocalArc> computeLocalArcs,
      std::vector<FrozenImplementationOccurrence> implementationOccurrences,
      std::vector<FrozenPortDemand> portDemands,
      std::vector<PnrIndex> compatibleEndpoints,
      std::vector<FrozenFabricMemoryOccurrence> fabricMemoryOccurrences,
      std::vector<FrozenMemoryPhysicalEndpoint> memoryPhysicalEndpoints,
      std::vector<mapping::TypeKey> memoryPhysicalEndpointCompatibleTypes,
      std::vector<FrozenMemoryLocalArc> memoryLocalArcs,
      std::vector<FrozenMemoryImplementationOccurrence>
          memoryImplementationOccurrences,
      std::vector<FrozenMemoryPortDemand> memoryPortDemands,
      std::vector<PnrIndex> compatibleMemoryEndpoints,
      std::vector<FrozenMemoryRealization> memoryRealizations,
      std::vector<FrozenTemplateTerminal> templateTerminals,
      std::vector<FrozenLogicalNet> logicalNets,
      std::vector<FrozenLogicalNetSink> logicalNetSinks,
      std::vector<FrozenMemoryServiceObligation> memoryServiceObligations)
      : actorOwnerships_(std::move(actorOwnerships)),
        computeRealizations_(std::move(computeRealizations)),
        fabricPeOccurrences_(std::move(fabricPeOccurrences)),
        fabricFuOccurrences_(std::move(fabricFuOccurrences)),
        physicalEndpoints_(std::move(physicalEndpoints)),
        physicalEndpointCompatibleTypes_(
            std::move(physicalEndpointCompatibleTypes)),
        computeLocalArcs_(std::move(computeLocalArcs)),
        implementationOccurrences_(std::move(implementationOccurrences)),
        portDemands_(std::move(portDemands)),
        compatibleEndpoints_(std::move(compatibleEndpoints)),
        fabricMemoryOccurrences_(std::move(fabricMemoryOccurrences)),
        memoryPhysicalEndpoints_(std::move(memoryPhysicalEndpoints)),
        memoryPhysicalEndpointCompatibleTypes_(
            std::move(memoryPhysicalEndpointCompatibleTypes)),
        memoryLocalArcs_(std::move(memoryLocalArcs)),
        memoryImplementationOccurrences_(
            std::move(memoryImplementationOccurrences)),
        memoryPortDemands_(std::move(memoryPortDemands)),
        compatibleMemoryEndpoints_(std::move(compatibleMemoryEndpoints)),
        memoryRealizations_(std::move(memoryRealizations)),
        templateTerminals_(std::move(templateTerminals)),
        logicalNets_(std::move(logicalNets)),
        logicalNetSinks_(std::move(logicalNetSinks)),
        memoryServiceObligations_(std::move(memoryServiceObligations)) {}

  std::vector<FrozenActorOwnership> actorOwnerships_;
  std::vector<FrozenComputeRealization> computeRealizations_;
  std::vector<FrozenFabricPeOccurrence> fabricPeOccurrences_;
  std::vector<FrozenFabricFuOccurrence> fabricFuOccurrences_;
  std::vector<FrozenPhysicalEndpoint> physicalEndpoints_;
  std::vector<mapping::TypeKey> physicalEndpointCompatibleTypes_;
  std::vector<FrozenComputeLocalArc> computeLocalArcs_;
  std::vector<FrozenImplementationOccurrence> implementationOccurrences_;
  std::vector<FrozenPortDemand> portDemands_;
  std::vector<PnrIndex> compatibleEndpoints_;
  std::vector<FrozenFabricMemoryOccurrence> fabricMemoryOccurrences_;
  std::vector<FrozenMemoryPhysicalEndpoint> memoryPhysicalEndpoints_;
  std::vector<mapping::TypeKey> memoryPhysicalEndpointCompatibleTypes_;
  std::vector<FrozenMemoryLocalArc> memoryLocalArcs_;
  std::vector<FrozenMemoryImplementationOccurrence>
      memoryImplementationOccurrences_;
  std::vector<FrozenMemoryPortDemand> memoryPortDemands_;
  std::vector<PnrIndex> compatibleMemoryEndpoints_;
  std::vector<FrozenMemoryRealization> memoryRealizations_;
  std::vector<FrozenTemplateTerminal> templateTerminals_;
  std::vector<FrozenLogicalNet> logicalNets_;
  std::vector<FrozenLogicalNetSink> logicalNetSinks_;
  std::vector<FrozenMemoryServiceObligation> memoryServiceObligations_;

  friend llvm::Expected<FrozenRealizationGraph>
  freezeRealizationGraph(const PnrProblemInputs &inputs);
};

llvm::Expected<FrozenRealizationGraph>
freezeRealizationGraph(const PnrProblemInputs &inputs);

namespace detail {

llvm::Error preflightFrozenRangeCapacity(PnrCapacityContext context,
                                         std::uint64_t offset,
                                         std::uint64_t count);

llvm::Error preflightFrozenRealizationGraphCapacity(
    llvm::ArrayRef<mapping::ComputeRealizationDraft> computeRealizations,
    llvm::ArrayRef<mapping::MemoryRealizationDraft> memoryRealizations,
    std::uint64_t canonicalEdgeCount);

llvm::Error preflightFrozenMemoryDomainsCapacity(
    std::uint64_t memoryRealizationCount, std::uint64_t memoryOccurrenceCount,
    std::uint64_t memoryEndpointCount, std::uint64_t compatibleTypeCount,
    std::uint64_t localArcCount);

} // namespace detail

} // namespace loom::pnr

#endif // LOOM_PNR_FROZENREALIZATIONGRAPH_H

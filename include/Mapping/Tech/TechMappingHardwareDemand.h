#ifndef LOOM_MAPPING_TECH_TECHMAPPINGHARDWAREDEMAND_H
#define LOOM_MAPPING_TECH_TECHMAPPINGHARDWAREDEMAND_H

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace loom::fabric {
class FabricArtifactView;
}

namespace loom::mapping {

struct TechMappingComputeContextHallDemandGroup final {
  loom::fabric::FabricFuCapabilityTemplateRef capability;
  std::uint64_t demandCount = 0;
  std::vector<loom::fabric::InstructionContextRef> compatibleContexts;
};

/// One exact Hall deficit observed while exploring a Tech cover. Demand
/// multiplicity is retained by capability; compatible contexts are rebuilt
/// from the exact Fabric rather than accepted as an independent payload fact.
class TechMappingComputeContextHallDeficit final {
public:
  static llvm::Expected<TechMappingComputeContextHallDeficit>
  get(std::uint64_t coverDemandCount, std::uint64_t coverMaximumMatching,
      llvm::ArrayRef<TechMappingComputeContextHallDemandGroup> groups);

  std::uint64_t coverDemandCount() const { return coverDemandCount_; }
  std::uint64_t coverMaximumMatching() const { return coverMaximumMatching_; }
  std::uint64_t hallDemandCount() const { return hallDemandCount_; }
  std::uint64_t hallContextValueCount() const { return hallContextValueCount_; }
  std::uint64_t deficit() const {
    return hallDemandCount_ - hallContextValueCount_;
  }
  llvm::ArrayRef<TechMappingComputeContextHallDemandGroup> groups() const {
    return groups_;
  }

private:
  TechMappingComputeContextHallDeficit(
      std::uint64_t coverDemandCount, std::uint64_t coverMaximumMatching,
      std::uint64_t hallDemandCount, std::uint64_t hallContextValueCount,
      std::vector<TechMappingComputeContextHallDemandGroup> groups)
      : coverDemandCount_(coverDemandCount),
        coverMaximumMatching_(coverMaximumMatching),
        hallDemandCount_(hallDemandCount),
        hallContextValueCount_(hallContextValueCount),
        groups_(std::move(groups)) {}

  std::uint64_t coverDemandCount_;
  std::uint64_t coverMaximumMatching_;
  std::uint64_t hallDemandCount_;
  std::uint64_t hallContextValueCount_;
  std::vector<TechMappingComputeContextHallDemandGroup> groups_;
};

llvm::ArrayRef<std::uint8_t> techMappingComputeContextHallFeedbackSchemaBytes();

std::vector<std::uint8_t> encodeTechMappingComputeContextHallFeedback(
    const TechMappingComputeContextHallDeficit &feedback);

llvm::Expected<TechMappingComputeContextHallDeficit>
adoptTechMappingComputeContextHallFeedback(
    llvm::ArrayRef<std::uint8_t> bytes,
    const loom::fabric::FabricArtifactView &fabric);

/// Retains one deterministic, maximally actionable observation. A larger
/// exact gap wins, followed by the larger Hall demand set and canonical bytes.
void retainTechMappingComputeContextHallFeedback(
    std::optional<TechMappingComputeContextHallDeficit> &retained,
    TechMappingComputeContextHallDeficit candidate);

} // namespace loom::mapping

#endif // LOOM_MAPPING_TECH_TECHMAPPINGHARDWAREDEMAND_H

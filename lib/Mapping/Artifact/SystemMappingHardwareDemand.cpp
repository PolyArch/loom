#include "Mapping/Artifact/SystemMappingHardwareDemand.h"

#include "Common/ArtifactStore.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace loom::mapping {
namespace {

constexpr llvm::StringLiteral feedbackSchema =
    "loom.mapping.system_acc_core_capacity_pressure.1.0";

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "system_mapping_hardware_demand_invalid: " +
                                     message);
}

void canonicalize(std::vector<ArtifactRootReference> &references) {
  llvm::sort(references, artifactRootReferenceLess);
  references.erase(std::unique(references.begin(), references.end()),
                   references.end());
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Expected<std::uint64_t> readU64(llvm::ArrayRef<std::uint8_t> bytes,
                                      std::size_t &offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8)
    return invalid("payload is truncated");
  std::uint64_t value = 0;
  for (unsigned index = 0; index != 8; ++index)
    value = (value << 8) | bytes[offset++];
  return value;
}

llvm::Expected<ArtifactRootReference>
readRootReference(llvm::ArrayRef<std::uint8_t> bytes, std::size_t &offset) {
  if (offset > bytes.size())
    return invalid("root-reference offset is outside the payload");
  auto decoded = decodeArtifactRootReferencePrefix(bytes.drop_front(offset));
  if (!decoded)
    return decoded.takeError();
  if (decoded->byteCount > bytes.size() - offset)
    return invalid("root reference is truncated");
  offset += decoded->byteCount;
  return std::move(decoded->reference);
}

} // namespace

llvm::Expected<SystemAccCoreCapacityPressure>
SystemAccCoreCapacityPressure::get(
    ArtifactRootReference system, ArtifactRootReference targetModule,
    std::vector<ArtifactRootReference> spatialMappings,
    std::uint64_t compatibleAccCoreCount, std::uint64_t assignmentAttempts,
    std::uint64_t witnessUsage, std::uint64_t witnessCapacity) {
  if (system.schemaIdentity != fabric::fabricArtifactSchema.identity ||
      system.schemaVersion != fabric::fabricArtifactSchema.version ||
      targetModule.schemaIdentity != fabric::fabricArtifactSchema.identity ||
      targetModule.schemaVersion != fabric::fabricArtifactSchema.version)
    return invalid("System or target Module is not an exact Fabric root");
  if (spatialMappings.empty())
    return invalid("SpatialMapping input frontier is empty");
  const std::size_t originalMappingCount = spatialMappings.size();
  canonicalize(spatialMappings);
  if (spatialMappings.size() != originalMappingCount)
    return invalid("SpatialMapping input frontier contains a duplicate");
  if (llvm::any_of(spatialMappings, [](const auto &mapping) {
        return mapping.schemaIdentity != mappingArtifactSchema.identity ||
               mapping.schemaVersion != mappingArtifactSchema.version;
      }))
    return invalid("input frontier contains a non-Mapping root");
  if (compatibleAccCoreCount == 0 || assignmentAttempts == 0 ||
      witnessCapacity == 0 || witnessUsage <= witnessCapacity)
    return invalid("capacity-pressure cardinalities are inconsistent");
  return SystemAccCoreCapacityPressure(
      std::move(system), std::move(targetModule), std::move(spatialMappings),
      compatibleAccCoreCount, assignmentAttempts, witnessUsage,
      witnessCapacity);
}

llvm::ArrayRef<std::uint8_t> systemAccCoreCapacityPressureSchemaBytes() {
  return {reinterpret_cast<const std::uint8_t *>(feedbackSchema.data()),
          feedbackSchema.size()};
}

std::vector<std::uint8_t> encodeSystemAccCoreCapacityPressure(
    const SystemAccCoreCapacityPressure &feedback) {
  std::vector<std::uint8_t> bytes =
      encodeArtifactRootReference(feedback.system());
  const auto module = encodeArtifactRootReference(feedback.targetModule());
  bytes.insert(bytes.end(), module.begin(), module.end());
  appendU64(bytes, feedback.spatialMappings().size());
  for (const ArtifactRootReference &mapping : feedback.spatialMappings()) {
    const auto encoded = encodeArtifactRootReference(mapping);
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
  }
  appendU64(bytes, feedback.compatibleAccCoreCount());
  appendU64(bytes, feedback.assignmentAttempts());
  appendU64(bytes, feedback.witnessUsage());
  appendU64(bytes, feedback.witnessCapacity());
  return bytes;
}

llvm::Expected<SystemAccCoreCapacityPressure>
adoptSystemAccCoreCapacityPressure(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ArtifactRootReference &systemReference,
    llvm::ArrayRef<ArtifactRootReference> spatialMappingInputs,
    const ArtifactStore &store) {
  std::size_t offset = 0;
  auto system = readRootReference(bytes, offset);
  if (!system)
    return system.takeError();
  auto targetModule = readRootReference(bytes, offset);
  if (!targetModule)
    return targetModule.takeError();
  auto mappingCount = readU64(bytes, offset);
  if (!mappingCount)
    return mappingCount.takeError();
  if (*mappingCount > spatialMappingInputs.size())
    return invalid("payload Mapping count exceeds its input frontier");
  std::vector<ArtifactRootReference> mappings;
  mappings.reserve(static_cast<std::size_t>(*mappingCount));
  for (std::uint64_t ordinal = 0; ordinal != *mappingCount; ++ordinal) {
    auto mapping = readRootReference(bytes, offset);
    if (!mapping)
      return mapping.takeError();
    mappings.push_back(std::move(*mapping));
  }
  auto compatibleAccCoreCount = readU64(bytes, offset);
  if (!compatibleAccCoreCount)
    return compatibleAccCoreCount.takeError();
  auto assignmentAttempts = readU64(bytes, offset);
  if (!assignmentAttempts)
    return assignmentAttempts.takeError();
  auto witnessUsage = readU64(bytes, offset);
  if (!witnessUsage)
    return witnessUsage.takeError();
  auto witnessCapacity = readU64(bytes, offset);
  if (!witnessCapacity)
    return witnessCapacity.takeError();
  if (offset != bytes.size())
    return invalid("payload has trailing bytes");

  auto feedback = SystemAccCoreCapacityPressure::get(
      std::move(*system), std::move(*targetModule), std::move(mappings),
      *compatibleAccCoreCount, *assignmentAttempts, *witnessUsage,
      *witnessCapacity);
  if (!feedback)
    return feedback.takeError();
  if (feedback->system() != systemReference)
    return invalid("payload names a different System input");
  std::vector<ArtifactRootReference> canonicalInputs(
      spatialMappingInputs.begin(), spatialMappingInputs.end());
  canonicalize(canonicalInputs);
  if (!llvm::equal(feedback->spatialMappings(), canonicalInputs))
    return invalid("payload names a different SpatialMapping frontier");

  auto fabricArtifact = fabric::importEntireFabricRoot(systemReference, store);
  if (!fabricArtifact)
    return fabricArtifact.takeError();
  auto systemView = fabric::requireSystemRoot(fabricArtifact->view());
  if (!systemView)
    return systemView.takeError();
  std::uint64_t compatibleCount = 0;
  for (const auto core : systemView->artifact().accCoreOccurrences()) {
    const auto target = systemView->spatialCoreTarget(core);
    if (!target || target->dependencyOrdinal >=
                       systemView->artifact().importedModules().size())
      return invalid("System AccCore has no exact Module target");
    if (systemView->artifact()
            .importedModules()[target->dependencyOrdinal]
            .identity() == feedback->targetModule().artifact)
      ++compatibleCount;
  }
  if (compatibleCount != feedback->compatibleAccCoreCount())
    return invalid("payload compatible AccCore count disagrees with Fabric");
  bool targetUsed = false;
  for (const ArtifactRootReference &mappingReference :
       feedback->spatialMappings()) {
    auto mapping = importSpatialMapping(mappingReference, store);
    if (!mapping)
      return mapping.takeError();
    targetUsed |=
        mapping->view().fabricIdentity() == feedback->targetModule().artifact;
  }
  if (!targetUsed)
    return invalid("payload target Module is unused by its Mapping frontier");
  const auto canonical = encodeSystemAccCoreCapacityPressure(*feedback);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != bytes)
    return invalid("payload is not canonical");
  return feedback;
}

} // namespace loom::mapping

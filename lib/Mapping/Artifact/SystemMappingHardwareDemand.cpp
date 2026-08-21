#include "Mapping/Artifact/SystemMappingHardwareDemand.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/DataflowReferenceCodec.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Artifact/FabricSystemRootView.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace loom::mapping {
namespace {

constexpr llvm::StringLiteral feedbackSchema =
    "loom.mapping.system_acc_core_capacity_pressure.3.0";

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

void appendI64(std::vector<std::uint8_t> &bytes, std::int64_t value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  appendU64(bytes, bits);
}

void appendBlob(std::vector<std::uint8_t> &bytes,
                llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
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

llvm::Expected<std::int64_t> readI64(llvm::ArrayRef<std::uint8_t> bytes,
                                     std::size_t &offset) {
  auto bits = readU64(bytes, offset);
  if (!bits)
    return bits.takeError();
  std::int64_t value = 0;
  static_assert(sizeof(*bits) == sizeof(value));
  std::memcpy(&value, &*bits, sizeof(value));
  return value;
}

llvm::Expected<llvm::ArrayRef<std::uint8_t>>
readBlob(llvm::ArrayRef<std::uint8_t> bytes, std::size_t &offset) {
  auto size = readU64(bytes, offset);
  if (!size)
    return size.takeError();
  if (*size > bytes.size() - std::min(offset, bytes.size()))
    return invalid("blob is truncated");
  auto result = bytes.slice(offset, static_cast<std::size_t>(*size));
  offset += static_cast<std::size_t>(*size);
  return result;
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

void appendCell(std::vector<std::uint8_t> &bytes,
                const SystemPresburgerCell &cell) {
  appendU64(bytes, cell.dimensionCount);
  appendU64(bytes, cell.symbolCount);
  appendU64(bytes, cell.localCount);
  const auto appendRows = [&](llvm::ArrayRef<std::vector<std::int64_t>> rows) {
    appendU64(bytes, rows.size());
    for (const auto &row : rows) {
      appendU64(bytes, row.size());
      for (std::int64_t value : row)
        appendI64(bytes, value);
    }
  };
  appendRows(cell.equalities);
  appendRows(cell.inequalities);
}

llvm::Expected<SystemPresburgerCell>
readCell(llvm::ArrayRef<std::uint8_t> bytes, std::size_t &offset) {
  SystemPresburgerCell cell;
  auto dimensionCount = readU64(bytes, offset);
  auto symbolCount = readU64(bytes, offset);
  auto localCount = readU64(bytes, offset);
  if (!dimensionCount)
    return dimensionCount.takeError();
  if (!symbolCount)
    return symbolCount.takeError();
  if (!localCount)
    return localCount.takeError();
  if (*dimensionCount > std::numeric_limits<std::uint32_t>::max() ||
      *symbolCount > std::numeric_limits<std::uint32_t>::max() ||
      *localCount > std::numeric_limits<std::uint32_t>::max())
    return invalid("Presburger signature exceeds u32");
  cell.dimensionCount = static_cast<std::uint32_t>(*dimensionCount);
  cell.symbolCount = static_cast<std::uint32_t>(*symbolCount);
  cell.localCount = static_cast<std::uint32_t>(*localCount);
  const auto readRows =
      [&](std::vector<std::vector<std::int64_t>> &rows) -> llvm::Error {
    auto count = readU64(bytes, offset);
    if (!count)
      return count.takeError();
    if (*count > bytes.size())
      return invalid("Presburger row count exceeds payload size");
    rows.reserve(static_cast<std::size_t>(*count));
    for (std::uint64_t rowOrdinal = 0; rowOrdinal != *count; ++rowOrdinal) {
      auto width = readU64(bytes, offset);
      if (!width)
        return width.takeError();
      if (*width > bytes.size())
        return invalid("Presburger row width exceeds payload size");
      std::vector<std::int64_t> row;
      row.reserve(static_cast<std::size_t>(*width));
      for (std::uint64_t column = 0; column != *width; ++column) {
        auto value = readI64(bytes, offset);
        if (!value)
          return value.takeError();
        row.push_back(*value);
      }
      rows.push_back(std::move(row));
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = readRows(cell.equalities))
    return std::move(error);
  if (llvm::Error error = readRows(cell.inequalities))
    return std::move(error);
  return canonicalizeSystemPresburgerCell(cell);
}

llvm::Expected<std::vector<std::uint8_t>>
threadKey(const ArtifactIdentity &dataflow,
          const SystemThreadExecutionCheckpoint &binding) {
  auto root = ::dataflow::encodeDataflowReference(dataflow, binding.root);
  if (!root)
    return root.takeError();
  std::vector<std::uint8_t> key;
  appendBlob(key, *root);
  appendCell(key, binding.cell);
  return key;
}

llvm::Expected<std::vector<std::uint8_t>>
graphKey(const ArtifactIdentity &dataflow,
         const SystemGraphExecutionCheckpoint &binding) {
  auto launch = ::dataflow::encodeDataflowReference(dataflow, binding.launch);
  if (!launch)
    return launch.takeError();
  std::vector<std::uint8_t> key;
  appendBlob(key, *launch);
  appendCell(key, binding.cell);
  return key;
}

template <typename Binding, typename KeyFn>
llvm::Error canonicalizeBindings(std::vector<Binding> &bindings, KeyFn keyFn) {
  std::vector<std::pair<std::vector<std::uint8_t>, Binding>> keyed;
  keyed.reserve(bindings.size());
  for (Binding &binding : bindings) {
    auto canonicalCell = canonicalizeSystemPresburgerCell(binding.cell);
    if (!canonicalCell)
      return canonicalCell.takeError();
    if (*canonicalCell != binding.cell)
      return invalid("checkpoint contains a noncanonical Presburger cell");
    auto key = keyFn(binding);
    if (!key)
      return key.takeError();
    keyed.emplace_back(std::move(*key), std::move(binding));
  }
  llvm::sort(keyed, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  for (std::size_t index = 1; index != keyed.size(); ++index)
    if (keyed[index - 1].first == keyed[index].first)
      return invalid("checkpoint contains a duplicate execution decision");
  bindings.clear();
  bindings.reserve(keyed.size());
  for (auto &entry : keyed)
    bindings.push_back(std::move(entry.second));
  return llvm::Error::success();
}

llvm::Expected<CanonicalSemanticBytes> canonicalCheckpointBytes(
    ArtifactRootReference dataflow, ArtifactRootReference system,
    std::vector<SystemThreadExecutionCheckpoint> &threadBindings,
    std::vector<SystemGraphExecutionCheckpoint> &graphBindings) {
  if (dataflow.schemaIdentity != ::dataflow::canonicalDataflowSchema.identity ||
      dataflow.schemaVersion != ::dataflow::canonicalDataflowSchema.version ||
      system.schemaIdentity != fabric::fabricArtifactSchema.identity ||
      system.schemaVersion != fabric::fabricArtifactSchema.version)
    return invalid("checkpoint has a foreign Dataflow or System schema");
  if (threadBindings.empty() && graphBindings.empty())
    return invalid("checkpoint has no execution-binding decision");
  if (llvm::any_of(graphBindings, [](const auto &binding) {
        return binding.target.schemaIdentity !=
                   mappingArtifactSchema.identity ||
               binding.target.schemaVersion != mappingArtifactSchema.version;
      }))
    return invalid("checkpoint graph target is not a Mapping root");
  if (llvm::Error error =
          canonicalizeBindings(threadBindings, [&](const auto &binding) {
            return threadKey(dataflow.artifact, binding);
          }))
    return std::move(error);
  if (llvm::Error error =
          canonicalizeBindings(graphBindings, [&](const auto &binding) {
            return graphKey(dataflow.artifact, binding);
          }))
    return std::move(error);

  std::vector<std::uint8_t> bytes = encodeArtifactRootReference(dataflow);
  const auto encodedSystem = encodeArtifactRootReference(system);
  bytes.insert(bytes.end(), encodedSystem.begin(), encodedSystem.end());
  appendU64(bytes, threadBindings.size());
  for (const auto &binding : threadBindings) {
    auto root =
        ::dataflow::encodeDataflowReference(dataflow.artifact, binding.root);
    if (!root)
      return root.takeError();
    appendBlob(bytes, *root);
    appendCell(bytes, binding.cell);
    appendBlob(bytes, fabric::canonicalFabricBytes(binding.target));
  }
  appendU64(bytes, graphBindings.size());
  for (const auto &binding : graphBindings) {
    auto launch =
        ::dataflow::encodeDataflowReference(dataflow.artifact, binding.launch);
    if (!launch)
      return launch.takeError();
    appendBlob(bytes, *launch);
    appendCell(bytes, binding.cell);
    const auto target = encodeArtifactRootReference(binding.target);
    bytes.insert(bytes.end(), target.begin(), target.end());
  }
  return CanonicalSemanticBytes(std::move(bytes));
}

} // namespace

llvm::Expected<FinalizedSystemExecutionBindingCheckpoint>
finalizeSystemExecutionBindingCheckpoint(
    ArtifactRootReference dataflow, ArtifactRootReference system,
    std::vector<SystemThreadExecutionCheckpoint> threadBindings,
    std::vector<SystemGraphExecutionCheckpoint> graphBindings,
    const ArtifactStore &store) {
  auto canonical =
      canonicalCheckpointBytes(dataflow, system, threadBindings, graphBindings);
  if (!canonical)
    return canonical.takeError();
  auto identity =
      store.put(systemExecutionBindingCheckpointArtifactSchema, *canonical);
  if (!identity)
    return identity.takeError();
  ArtifactRootReference reference{
      systemExecutionBindingCheckpointArtifactSchema.identity.str(),
      systemExecutionBindingCheckpointArtifactSchema.version, *identity};
  return importSystemExecutionBindingCheckpoint(reference, store);
}

llvm::Expected<FinalizedSystemExecutionBindingCheckpoint>
importSystemExecutionBindingCheckpoint(const ArtifactRootReference &reference,
                                       const ArtifactStore &store) {
  if (reference.schemaIdentity !=
          systemExecutionBindingCheckpointArtifactSchema.identity ||
      reference.schemaVersion !=
          systemExecutionBindingCheckpointArtifactSchema.version)
    return invalid("checkpoint root has the wrong schema");
  auto stored = store.get(systemExecutionBindingCheckpointArtifactSchema,
                          reference.artifact);
  if (!stored)
    return stored.takeError();
  llvm::ArrayRef<std::uint8_t> bytes = stored->bytes();
  std::size_t offset = 0;
  auto dataflowReference = readRootReference(bytes, offset);
  if (!dataflowReference)
    return dataflowReference.takeError();
  auto systemReference = readRootReference(bytes, offset);
  if (!systemReference)
    return systemReference.takeError();
  auto threadCount = readU64(bytes, offset);
  if (!threadCount)
    return threadCount.takeError();
  if (*threadCount > bytes.size())
    return invalid("checkpoint thread count exceeds payload size");
  std::vector<SystemThreadExecutionCheckpoint> threadBindings;
  threadBindings.reserve(static_cast<std::size_t>(*threadCount));
  for (std::uint64_t ordinal = 0; ordinal != *threadCount; ++ordinal) {
    auto rootBytes = readBlob(bytes, offset);
    if (!rootBytes)
      return rootBytes.takeError();
    auto root =
        ::dataflow::decodeDataflowReference<::dataflow::RootThreadLaunchRef>(
            *rootBytes, dataflowReference->artifact);
    if (!root)
      return root.takeError();
    auto cell = readCell(bytes, offset);
    if (!cell)
      return cell.takeError();
    auto targetBytes = readBlob(bytes, offset);
    if (!targetBytes)
      return targetBytes.takeError();
    auto target =
        fabric::decodeFabricRef<fabric::AccCoreOccurrenceRef>(*targetBytes);
    if (!target)
      return target.takeError();
    threadBindings.push_back(
        {std::move(*root), std::move(*cell), std::move(*target)});
  }
  auto graphCount = readU64(bytes, offset);
  if (!graphCount)
    return graphCount.takeError();
  if (*graphCount > bytes.size())
    return invalid("checkpoint graph count exceeds payload size");
  std::vector<SystemGraphExecutionCheckpoint> graphBindings;
  graphBindings.reserve(static_cast<std::size_t>(*graphCount));
  for (std::uint64_t ordinal = 0; ordinal != *graphCount; ++ordinal) {
    auto launchBytes = readBlob(bytes, offset);
    if (!launchBytes)
      return launchBytes.takeError();
    auto launch =
        ::dataflow::decodeDataflowReference<::dataflow::RootedGraphLaunchRef>(
            *launchBytes, dataflowReference->artifact);
    if (!launch)
      return launch.takeError();
    auto cell = readCell(bytes, offset);
    if (!cell)
      return cell.takeError();
    auto target = readRootReference(bytes, offset);
    if (!target)
      return target.takeError();
    graphBindings.push_back(
        {std::move(*launch), std::move(*cell), std::move(*target)});
  }
  if (offset != bytes.size())
    return invalid("checkpoint payload has trailing bytes");

  auto canonical = canonicalCheckpointBytes(
      *dataflowReference, *systemReference, threadBindings, graphBindings);
  if (!canonical)
    return canonical.takeError();
  if (canonical->bytes() != bytes)
    return invalid("checkpoint payload is not canonical");

  auto dataflowArtifact =
      ::dataflow::importCanonicalDataflow(*dataflowReference, store);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflowView = dataflowArtifact->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto systemArtifact = fabric::importEntireFabricRoot(*systemReference, store);
  if (!systemArtifact)
    return systemArtifact.takeError();
  auto systemView = fabric::requireSystemRoot(systemArtifact->view());
  if (!systemView)
    return systemView.takeError();
  for (const auto &binding : threadBindings) {
    auto resolved = dataflowView->resolve(binding.root);
    if (!resolved)
      return resolved.takeError();
    if (!llvm::is_contained(systemView->artifact().accCoreOccurrences(),
                            binding.target))
      return invalid("checkpoint names a foreign AccCore occurrence");
  }
  for (const auto &binding : graphBindings) {
    auto resolved = dataflowView->resolve(binding.launch);
    if (!resolved)
      return resolved.takeError();
    auto mapping = importSpatialMapping(binding.target, store);
    if (!mapping)
      return mapping.takeError();
    if (mapping->view().dataflowIdentity() != dataflowView->identity())
      return invalid("checkpoint names a foreign SpatialMapping Dataflow");
    bool importedModule = false;
    for (const fabric::FabricDirectDependency &dependency :
         systemArtifact->directDependencies())
      importedModule |=
          dependency.role == fabric::FabricDependencyRole::ImportedModule &&
          dependency.root.artifact == mapping->view().fabricIdentity();
    if (!importedModule)
      return invalid("checkpoint SpatialMapping is not imported by System");
  }
  return FinalizedSystemExecutionBindingCheckpoint(
      reference, std::move(*dataflowReference), std::move(*systemReference),
      std::move(threadBindings), std::move(graphBindings));
}

llvm::Expected<SystemAccCoreCapacityPressure>
SystemAccCoreCapacityPressure::get(
    ArtifactRootReference system, ArtifactRootReference targetModule,
    fabric::AccCoreOccurrenceRef witnessAccCore,
    std::vector<ArtifactRootReference> spatialMappings,
    std::uint64_t compatibleAccCoreCount, std::uint64_t assignmentAttempts,
    std::uint64_t witnessUsage, std::uint64_t witnessCapacity,
    ArtifactRootReference executionBindingCheckpoint) {
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
  if (executionBindingCheckpoint.schemaIdentity !=
          systemExecutionBindingCheckpointArtifactSchema.identity ||
      executionBindingCheckpoint.schemaVersion !=
          systemExecutionBindingCheckpointArtifactSchema.version)
    return invalid("capacity pressure has a foreign checkpoint schema");
  return SystemAccCoreCapacityPressure(
      std::move(system), std::move(targetModule), witnessAccCore,
      std::move(spatialMappings), compatibleAccCoreCount, assignmentAttempts,
      witnessUsage, witnessCapacity, std::move(executionBindingCheckpoint));
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
  appendBlob(bytes, fabric::canonicalFabricBytes(feedback.witnessAccCore()));
  appendU64(bytes, feedback.spatialMappings().size());
  for (const ArtifactRootReference &mapping : feedback.spatialMappings()) {
    const auto encoded = encodeArtifactRootReference(mapping);
    bytes.insert(bytes.end(), encoded.begin(), encoded.end());
  }
  appendU64(bytes, feedback.compatibleAccCoreCount());
  appendU64(bytes, feedback.assignmentAttempts());
  appendU64(bytes, feedback.witnessUsage());
  appendU64(bytes, feedback.witnessCapacity());
  const auto checkpoint =
      encodeArtifactRootReference(feedback.executionBindingCheckpoint());
  bytes.insert(bytes.end(), checkpoint.begin(), checkpoint.end());
  return bytes;
}

llvm::Expected<SystemAccCoreCapacityPressure>
adoptSystemAccCoreCapacityPressure(
    llvm::ArrayRef<std::uint8_t> bytes,
    const ArtifactRootReference &systemReference,
    const ArtifactRootReference &dataflowReference,
    llvm::ArrayRef<ArtifactRootReference> spatialMappingInputs,
    const ArtifactStore &store) {
  std::size_t offset = 0;
  auto system = readRootReference(bytes, offset);
  if (!system)
    return system.takeError();
  auto targetModule = readRootReference(bytes, offset);
  if (!targetModule)
    return targetModule.takeError();
  auto witnessAccCoreBytes = readBlob(bytes, offset);
  if (!witnessAccCoreBytes)
    return witnessAccCoreBytes.takeError();
  auto witnessAccCore = fabric::decodeFabricRef<fabric::AccCoreOccurrenceRef>(
      *witnessAccCoreBytes);
  if (!witnessAccCore)
    return witnessAccCore.takeError();
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
  auto checkpointReference = readRootReference(bytes, offset);
  if (!checkpointReference)
    return checkpointReference.takeError();
  if (offset != bytes.size())
    return invalid("payload has trailing bytes");

  auto feedback = SystemAccCoreCapacityPressure::get(
      std::move(*system), std::move(*targetModule), *witnessAccCore,
      std::move(mappings), *compatibleAccCoreCount, *assignmentAttempts,
      *witnessUsage, *witnessCapacity, std::move(*checkpointReference));
  if (!feedback)
    return feedback.takeError();
  if (feedback->system() != systemReference)
    return invalid("payload names a different System input");
  std::vector<ArtifactRootReference> canonicalInputs(
      spatialMappingInputs.begin(), spatialMappingInputs.end());
  canonicalize(canonicalInputs);
  if (!llvm::equal(feedback->spatialMappings(), canonicalInputs))
    return invalid("payload names a different SpatialMapping frontier");
  auto checkpoint = importSystemExecutionBindingCheckpoint(
      feedback->executionBindingCheckpoint(), store);
  if (!checkpoint)
    return checkpoint.takeError();
  if (checkpoint->system() != systemReference ||
      checkpoint->dataflow() != dataflowReference)
    return invalid("capacity pressure checkpoint names foreign inputs");
  if (!llvm::any_of(checkpoint->threadBindings(), [&](const auto &binding) {
        return binding.target == feedback->witnessAccCore();
      }))
    return invalid("capacity pressure witness owns no checkpoint thread");
  std::vector<ArtifactRootReference> checkpointMappings;
  checkpointMappings.reserve(checkpoint->graphBindings().size());
  for (const auto &binding : checkpoint->graphBindings())
    checkpointMappings.push_back(binding.target);
  canonicalize(checkpointMappings);
  if (!llvm::all_of(checkpointMappings, [&](const auto &mapping) {
        return llvm::is_contained(canonicalInputs, mapping);
      }))
    return invalid("checkpoint selects outside its SpatialMapping frontier");

  auto fabricArtifact = fabric::importEntireFabricRoot(systemReference, store);
  if (!fabricArtifact)
    return fabricArtifact.takeError();
  auto systemView = fabric::requireSystemRoot(fabricArtifact->view());
  if (!systemView)
    return systemView.takeError();
  if (!llvm::is_contained(systemView->artifact().accCoreOccurrences(),
                          feedback->witnessAccCore()))
    return invalid("payload witness names a foreign AccCore occurrence");
  const auto witnessTarget =
      systemView->spatialCoreTarget(feedback->witnessAccCore());
  if (!witnessTarget || witnessTarget->dependencyOrdinal >=
                            systemView->artifact().importedModules().size())
    return invalid("payload witness AccCore has no exact Module target");
  if (systemView->artifact()
          .importedModules()[witnessTarget->dependencyOrdinal]
          .identity() != feedback->targetModule().artifact)
    return invalid("payload witness AccCore targets a different Module");
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

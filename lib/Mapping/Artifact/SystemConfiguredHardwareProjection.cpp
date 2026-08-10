#include "Mapping/Artifact/ConfiguredHardwareProjection.h"

#include "Common/ArtifactLocalReference.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/Identity/FabricRefBytes.h"
#include "Mapping/Artifact/MappingArtifact.h"
#include "Mapping/Artifact/SystemMappingArtifact.h"
#include "SystemMappingExecutionProjection.h"

#include "llvm/Support/Error.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace loom::mapping {
namespace detail {

struct PhysicalConfiguredHardwareProjectionViewAccess final {
  static PhysicalConfiguredHardwareProjectionView
  create(std::vector<PhysicalConfiguredHardwareFieldValueView> fields) {
    return PhysicalConfiguredHardwareProjectionView(std::move(fields));
  }
};

} // namespace detail
namespace {

using ByteVector = std::vector<std::uint8_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "configured_hardware_projection_invalid: " +
                                     message);
}

std::string byteKey(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

llvm::Expected<std::vector<PhysicalConfiguredHardwareFieldValueView>>
qualifyFields(const ConfiguredHardwareProjectionView &projection,
              ::loom::fabric::SpatialCoreOccurrenceRef occurrence) {
  std::vector<PhysicalConfiguredHardwareFieldValueView> result;
  result.reserve(projection.fields().size());
  for (const ConfiguredHardwareFieldValueView &field : projection.fields()) {
    auto slot = ::loom::fabric::FabricPhysicalConfigurationSlotRef::create(
        ::loom::fabric::SpatialCoreInternalConfigurationSlotRef{occurrence,
                                                                field.slot});
    if (!slot)
      return slot.takeError();
    result.push_back({std::move(*slot), field.value});
  }
  return result;
}

llvm::Expected<PhysicalConfiguredHardwareProjectionView>
canonicalize(std::vector<PhysicalConfiguredHardwareFieldValueView> values) {
  std::map<ByteVector, PhysicalConfiguredHardwareFieldValueView> bySlot;
  for (PhysicalConfiguredHardwareFieldValueView &value : values) {
    ByteVector key = ::loom::fabric::canonicalFabricBytes(value.slot);
    auto found = bySlot.find(key);
    if (found == bySlot.end()) {
      bySlot.emplace(std::move(key), std::move(value));
      continue;
    }
    if (!found->second.value.bytes().equals(value.value.bytes()))
      return invalid("one occurrence-qualified slot has conflicting values");
  }

  std::vector<PhysicalConfiguredHardwareFieldValueView> ordered;
  ordered.reserve(bySlot.size());
  for (auto &[key, value] : bySlot) {
    (void)key;
    ordered.push_back(std::move(value));
  }
  return detail::PhysicalConfiguredHardwareProjectionViewAccess::create(
      std::move(ordered));
}

} // namespace

llvm::Expected<PhysicalConfiguredHardwareProjectionView>
qualifyConfiguredHardwareProjection(
    const FinalizedSpatialMapping &mapping,
    const ::loom::fabric::FabricSystemRootView &system,
    ::loom::fabric::SpatialCoreOccurrenceRef occurrence) {
  const auto target = system.spatialCoreTarget(occurrence.core);
  if (!target)
    return invalid("SpatialCore occurrence has no imported Module");
  const auto modules = system.artifact().importedModules();
  if (target->dependencyOrdinal >= modules.size())
    return invalid("SpatialCore occurrence has an invalid Module dependency");
  if (modules[target->dependencyOrdinal].identity() !=
      mapping.view().fabricIdentity())
    return invalid("SpatialMapping does not bind the occurrence's imported "
                   "Module");
  auto values = qualifyFields(mapping.view().configuredHardware(), occurrence);
  if (!values)
    return values.takeError();
  return canonicalize(std::move(*values));
}

llvm::Expected<PhysicalConfiguredHardwareProjectionView>
deriveConfiguredHardwareProjection(const FinalizedSystemMapping &mapping,
                                   const ArtifactStore &store) {
  const ArtifactRootReference dataflowReference{
      ::dataflow::canonicalDataflowSchema.identity.str(),
      ::dataflow::canonicalDataflowSchema.version,
      mapping.view().dataflowIdentity()};
  auto dataflow = ::dataflow::importCanonicalDataflow(dataflowReference, store);
  if (!dataflow)
    return dataflow.takeError();
  auto dataflowView = dataflow->view();
  if (!dataflowView)
    return dataflowView.takeError();

  auto contexts = detail::projectSystemExecutionContexts(
      *dataflowView, mapping.view().executionBindings());
  if (!contexts)
    return contexts.takeError();

  std::map<std::string, FinalizedSpatialMapping> imported;
  std::vector<PhysicalConfiguredHardwareFieldValueView> values;
  for (const detail::SystemSpatialContextDomain &domain :
       contexts->spatialDomains) {
    const std::string mappingKey =
        byteKey(encodeArtifactRootReference(domain.spatialMapping));
    auto found = imported.find(mappingKey);
    if (found == imported.end()) {
      auto spatial = importSpatialMapping(domain.spatialMapping, store);
      if (!spatial)
        return spatial.takeError();
      found = imported.emplace(mappingKey, std::move(*spatial)).first;
    }
    auto qualified = qualifyFields(
        found->second.view().configuredHardware(),
        ::loom::fabric::SpatialCoreOccurrenceRef{domain.context.accCore});
    if (!qualified)
      return qualified.takeError();
    values.insert(values.end(), std::make_move_iterator(qualified->begin()),
                  std::make_move_iterator(qualified->end()));
  }
  return canonicalize(std::move(values));
}

} // namespace loom::mapping

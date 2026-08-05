#include "ConfiguredHardwareProjectionInternal.h"

#include "Common/IndexWidth.h"
#include "Common/PointerLayout.h"
#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/Identity/FabricRefBytes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::mapping::detail {
namespace {

using ByteVector = std::vector<std::uint8_t>;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

struct SlotKey final {
  ByteVector context;
  ByteVector field;

  friend bool operator<(const SlotKey &lhs, const SlotKey &rhs) {
    return std::tie(lhs.context, lhs.field) < std::tie(rhs.context, rhs.field);
  }
};

const SpatialComputeBindingView *
findBinding(llvm::ArrayRef<SpatialComputeBindingView> bindings,
            std::uint64_t realization) {
  const SpatialComputeBindingView *result = nullptr;
  for (const SpatialComputeBindingView &binding : bindings) {
    if (binding.realization != realization)
      continue;
    if (result)
      return nullptr;
    result = &binding;
  }
  return result;
}

llvm::Expected<std::optional<::loom::PointerLayout>>
resolvePointerLayout(const ::dataflow::CanonicalDataflowProgramView &dataflow,
                     const ::dataflow::CanonicalActorSchemaProjection &actor) {
  auto addressSpace = ::dataflow::projectActorPointerAddressSpace(actor);
  if (!addressSpace)
    return addressSpace.takeError();
  if (!*addressSpace)
    return std::optional<::loom::PointerLayout>();
  auto layout = dataflow.pointerLayout(**addressSpace);
  if (!layout)
    return layout.takeError();
  return std::optional<::loom::PointerLayout>(*layout);
}

SlotKey key(const ::loom::fabric::InstructionContextRef &context,
            const ::loom::fabric::FabricSemanticConfigFieldRef &field) {
  return {::loom::fabric::canonicalFabricBytes(context),
          ::loom::fabric::canonicalFabricBytes(field)};
}

} // namespace

llvm::Expected<ConfiguredHardwareProjectionView>
canonicalizeConfiguredHardwareProjection(
    std::vector<ConfiguredHardwareFieldValueView> selectedFields) {
  std::map<SlotKey, ConfiguredHardwareFieldValueView> fields;
  for (ConfiguredHardwareFieldValueView &selected : selectedFields) {
    const SlotKey selectedKey = key(selected.slot.context, selected.slot.field);
    auto found = fields.find(selectedKey);
    if (found == fields.end()) {
      fields.emplace(selectedKey, std::move(selected));
      continue;
    }
    if (!found->second.value.bytes().equals(selected.value.bytes()))
      return invalid("one physical configuration field has conflicting "
                     "semantic values");
  }

  std::vector<ConfiguredHardwareFieldValueView> orderedFields;
  orderedFields.reserve(fields.size());
  for (auto &[slot, value] : fields) {
    (void)slot;
    orderedFields.push_back(std::move(value));
  }
  return ConfiguredHardwareProjectionViewAccess::create(
      std::move(orderedFields));
}

llvm::Expected<ConfiguredHardwareProjectionView>
deriveConfiguredHardwareProjection(
    const ::dataflow::CanonicalDataflowProgramView &dataflow,
    const TechMappingView &techMapping,
    const ::loom::fabric::FabricArtifactView &fabric,
    llvm::ArrayRef<SpatialComputeBindingView> bindings) {
  if (bindings.size() != techMapping.computeRealizations().size())
    return invalid("configured hardware projection has incomplete bindings");

  std::vector<ConfiguredHardwareFieldValueView> fields;
  for (const TechComputeRealizationView &realization :
       techMapping.computeRealizations()) {
    const SpatialComputeBindingView *binding =
        findBinding(bindings, realization.entityId);
    if (!binding)
      return invalid("configured hardware projection has a missing or "
                     "duplicate compute binding");

    for (const TechComputeActorView &actorBinding : realization.actors) {
      auto actor = dataflow.resolve(actorBinding.actor);
      if (!actor)
        return actor.takeError();
      auto actorProjection =
          ::dataflow::projectRegisteredActorSchemaProjection(actor->op);
      if (!actorProjection)
        return actorProjection.takeError();
      auto indexBitWidth = ::loom::getIndexBitWidth(actor->op);
      if (!indexBitWidth)
        return indexBitWidth.takeError();
      auto pointerLayout = resolvePointerLayout(dataflow, *actorProjection);
      if (!pointerLayout)
        return pointerLayout.takeError();

      auto occurrenceOperation = ::loom::fabric::deriveFabricFuOccurrenceNode(
          fabric, actorBinding.fabricOperation, binding->occurrence);
      if (!occurrenceOperation)
        return occurrenceOperation.takeError();
      const auto *capability =
          fabric.resolvedFabricOpCapability(*occurrenceOperation);
      if (!capability)
        return invalid("configured compute actor has no Fabric capability");

      for (const ::loom::fabric::FabricSemanticConfigFieldRef &templateField :
           capability->configurationFieldSchema) {
        auto value = capability->encodeSemanticConfiguration(
            templateField, *actorProjection, *indexBitWidth,
            actorBinding.operandPorts, actorBinding.resultPorts,
            *pointerLayout ? &**pointerLayout : nullptr);
        if (!value)
          return value.takeError();
        const ::loom::fabric::FabricSemanticConfigFieldRef occurrenceField{
            ::loom::fabric::FabricConfigurationOwnerRef(
                ::loom::fabric::FabricInventoryOwnerRef::of(
                    *occurrenceOperation)),
            templateField.ordinal};
        if (llvm::Error error =
                ::loom::fabric::validateFabricRef(fabric, occurrenceField))
          return std::move(error);

        fields.push_back(
            {{binding->context, occurrenceField}, std::move(*value)});
      }
    }
  }
  return canonicalizeConfiguredHardwareProjection(std::move(fields));
}

} // namespace loom::mapping::detail

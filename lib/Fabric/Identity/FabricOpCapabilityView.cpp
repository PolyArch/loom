#include "Fabric/Identity/FabricRefImport.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <cstddef>
#include <string>
#include <utility>

using namespace loom::fabric;

namespace {

llvm::Error capabilityRejected(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_operation_capability_rejected: " +
                                     message);
}

llvm::Error verifyPhysicalPortCapacity(
    const ResolvedFabricOpCapabilityView &capability,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    const ::loom::PointerLayout *pointerLayout) {
  llvm::SmallVector<unsigned, 4> physicalInputs;
  llvm::SmallVector<unsigned, 4> physicalResults;
  for (const ResolvedFabricOpPhysicalPortView &port :
       capability.physicalPorts) {
    if (port.reference.direction == FabricPortDirection::Input)
      physicalInputs.push_back(port.payloadWidthBits);
    else if (port.reference.direction == FabricPortDirection::Output)
      physicalResults.push_back(port.payloadWidthBits);
    else
      return capabilityRejected("physical port has an unknown direction");
  }
  if (physicalInputs.size() < actor.type.getNumInputs() ||
      physicalResults.size() < actor.type.getNumResults())
    return capabilityRejected("physical port capacity cannot cover the actor "
                              "arity");

  const auto semanticWidths = [&](mlir::TypeRange types)
      -> llvm::Expected<llvm::SmallVector<unsigned, 4>> {
    llvm::SmallVector<unsigned, 4> widths;
    widths.reserve(types.size());
    for (mlir::Type type : types) {
      std::string message;
      mlir::FailureOr<unsigned> width =
          ::fabric::getSemanticPayloadWidth(type, pointerLayout, message);
      if (mlir::failed(width))
        return capabilityRejected(message);
      widths.push_back(*width);
    }
    llvm::sort(widths);
    return widths;
  };
  auto semanticInputs = semanticWidths(actor.type.getInputs());
  if (!semanticInputs)
    return semanticInputs.takeError();
  auto semanticResults = semanticWidths(actor.type.getResults());
  if (!semanticResults)
    return semanticResults.takeError();
  llvm::sort(physicalInputs);
  llvm::sort(physicalResults);
  const auto hasWidthCapacity = [](llvm::ArrayRef<unsigned> semantic,
                                   llvm::ArrayRef<unsigned> physical) {
    std::size_t physicalPosition = 0;
    for (unsigned width : semantic) {
      while (physicalPosition < physical.size() &&
             physical[physicalPosition] < width)
        ++physicalPosition;
      if (physicalPosition == physical.size())
        return false;
      ++physicalPosition;
    }
    return true;
  };
  if (!hasWidthCapacity(*semanticInputs, physicalInputs))
    return capabilityRejected("no physical input correspondence has enough "
                              "width");
  if (!hasWidthCapacity(*semanticResults, physicalResults))
    return capabilityRejected("no physical result correspondence has enough "
                              "width");
  return llvm::Error::success();
}

struct OrderedPhysicalWidths final {
  std::vector<std::uint32_t> inputs;
  std::vector<std::uint32_t> results;
};

llvm::Expected<OrderedPhysicalWidths>
resolveOrderedPhysicalWidths(const ResolvedFabricOpCapabilityView &capability) {
  std::size_t inputCount = 0;
  std::size_t resultCount = 0;
  for (const ResolvedFabricOpPhysicalPortView &port :
       capability.physicalPorts) {
    if (port.reference.direction == FabricPortDirection::Input)
      ++inputCount;
    else if (port.reference.direction == FabricPortDirection::Output)
      ++resultCount;
    else
      return capabilityRejected("physical port has an unknown direction");
  }

  OrderedPhysicalWidths widths{std::vector<std::uint32_t>(inputCount),
                               std::vector<std::uint32_t>(resultCount)};
  std::vector<bool> inputSeen(inputCount);
  std::vector<bool> resultSeen(resultCount);
  for (const ResolvedFabricOpPhysicalPortView &port :
       capability.physicalPorts) {
    std::vector<std::uint32_t> &ordered =
        port.reference.direction == FabricPortDirection::Input ? widths.inputs
                                                               : widths.results;
    std::vector<bool> &seen =
        port.reference.direction == FabricPortDirection::Input ? inputSeen
                                                               : resultSeen;
    if (port.reference.ordinal >= ordered.size() ||
        seen[port.reference.ordinal])
      return capabilityRejected(
          "physical port ordinals are not dense and unique");
    ordered[port.reference.ordinal] = port.payloadWidthBits;
    seen[port.reference.ordinal] = true;
  }
  return widths;
}

llvm::Expected<::fabric::FabricOpSemanticFieldRelation>
resolveCurrentSemanticFieldRelation(
    const ResolvedFabricOpCapabilityView &capability,
    mlir::MLIRContext &context) {
  auto widths = resolveOrderedPhysicalWidths(capability);
  if (!widths)
    return widths.takeError();
  return ::fabric::resolveFabricOpSemanticFieldRelation(
      capability.implementationFamily, capability.parameterizedCapability,
      capability.enabledOperationSchemas, widths->inputs, widths->results,
      context);
}

} // namespace

llvm::Error ResolvedFabricOpCapabilityView::admit(
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    unsigned indexBitWidth, const ::loom::PointerLayout *pointerLayout) const {
  if (indexBitWidth == 0 || indexBitWidth > mlir::IntegerType::kMaxWidth)
    return capabilityRejected(
        "canonical index width has no fixed representation");
  if (!llvm::is_contained(enabledOperationSchemas, actor.schema))
    return capabilityRejected(
        "operation schema is not enabled by the concrete fabric.op");

  if (pointerLayout) {
    if (llvm::Error error = ::fabric::verifyImplementationFamilyAdmission(
            implementationFamily, &parameterizedCapability, actor,
            indexBitWidth, *pointerLayout))
      return error;
  } else if (llvm::Error error = ::fabric::verifyImplementationFamilyAdmission(
                 implementationFamily, &parameterizedCapability, actor,
                 indexBitWidth)) {
    return error;
  }
  auto represented = ::fabric::projectResolvedIndexTypes(actor, indexBitWidth);
  if (!represented)
    return represented.takeError();
  return verifyPhysicalPortCapacity(*this, *represented, pointerLayout);
}

llvm::Error ResolvedFabricOpCapabilityView::admitCorrespondence(
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    unsigned indexBitWidth, llvm::ArrayRef<std::uint64_t> operandPorts,
    llvm::ArrayRef<std::uint64_t> resultPorts,
    const ::loom::PointerLayout *pointerLayout) const {
  if (llvm::Error error = admit(actor, indexBitWidth, pointerLayout))
    return error;
  if (llvm::Error error =
          ::fabric::verifyImplementationFamilyPortCorrespondence(
              implementationFamily, actor, operandPorts, resultPorts))
    return error;

  auto represented = ::fabric::projectResolvedIndexTypes(actor, indexBitWidth);
  if (!represented)
    return represented.takeError();

  const auto verify = [&](mlir::TypeRange types,
                          llvm::ArrayRef<std::uint64_t> selected,
                          FabricPortDirection direction) -> llvm::Error {
    if (types.size() != selected.size())
      return capabilityRejected(
          "ordered physical-port correspondence has wrong arity");
    llvm::SmallDenseSet<std::uint64_t, 8> used;
    for (auto [softwareOrdinal, type] : llvm::enumerate(types)) {
      const std::uint64_t physicalOrdinal = selected[softwareOrdinal];
      if (!used.insert(physicalOrdinal).second)
        return capabilityRejected(
            "ordered physical-port correspondence reuses a port");
      const auto found = llvm::find_if(
          physicalPorts, [&](const ResolvedFabricOpPhysicalPortView &port) {
            return port.reference.direction == direction &&
                   port.reference.ordinal == physicalOrdinal;
          });
      if (found == physicalPorts.end())
        return capabilityRejected(
            "ordered physical-port correspondence selects a missing port");
      std::string message;
      mlir::FailureOr<unsigned> width =
          ::fabric::getSemanticPayloadWidth(type, pointerLayout, message);
      if (mlir::failed(width))
        return capabilityRejected(message);
      if (found->payloadWidthBits < *width)
        return capabilityRejected(
            "ordered physical-port correspondence selects an undersized "
            "port");
    }
    return llvm::Error::success();
  };

  if (llvm::Error error = verify(represented->type.getInputs(), operandPorts,
                                 FabricPortDirection::Input))
    return error;
  return verify(represented->type.getResults(), resultPorts,
                FabricPortDirection::Output);
}

llvm::Expected<loom::CanonicalSemanticBytes>
ResolvedFabricOpCapabilityView::encodeSemanticConfiguration(
    const FabricSemanticConfigFieldRef &field,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    unsigned indexBitWidth, llvm::ArrayRef<std::uint64_t> operandPorts,
    llvm::ArrayRef<std::uint64_t> resultPorts,
    const ::loom::PointerLayout *pointerLayout) const {
  const auto rejected = [](const llvm::Twine &message) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "fabric_operation_configuration_codec_rejected: " + message);
  };
  if (configurationFieldSchema.size() != 1 ||
      configurationFieldSchema.front() != field)
    return rejected("field is not the exact operation configuration field");
  if (llvm::Error error = admitCorrespondence(
          actor, indexBitWidth, operandPorts, resultPorts, pointerLayout))
    return std::move(error);
  auto relation =
      resolveCurrentSemanticFieldRelation(*this, *actor.type.getContext());
  if (!relation)
    return relation.takeError();
  if (!relation->hasConfigurationField())
    return rejected("operation capability has no semantic field relation");
  return relation->projectSemanticValue(
      actor, operandPorts, resultPorts,
      ::fabric::symbolizeResolvedIndexWidth(indexBitWidth));
}

llvm::Expected<::fabric::FabricOpSemanticFieldRelation>
ResolvedFabricOpCapabilityView::resolveSemanticFieldRelation(
    mlir::MLIRContext &context) const {
  return resolveCurrentSemanticFieldRelation(*this, context);
}

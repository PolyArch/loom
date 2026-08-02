#include "Fabric/Identity/FabricRefImport.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <cstddef>
#include <string>

using namespace loom::fabric;

llvm::Error ResolvedFabricOpCapabilityView::admit(
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    unsigned indexBitWidth, const ::loom::PointerLayout *pointerLayout) const {
  const auto rejected = [](const llvm::Twine &message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "fabric_operation_capability_rejected: " +
                                       message);
  };
  if (indexBitWidth == 0 || indexBitWidth > mlir::IntegerType::kMaxWidth)
    return rejected("canonical index width has no fixed representation");
  if (!llvm::is_contained(enabledOperationSchemas, actor.schema))
    return rejected(
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

  llvm::SmallVector<unsigned, 4> physicalInputs;
  llvm::SmallVector<unsigned, 4> physicalResults;
  for (const ResolvedFabricOpPhysicalPortView &port : physicalPorts)
    (port.reference.direction == FabricPortDirection::Input ? physicalInputs
                                                            : physicalResults)
        .push_back(port.payloadWidthBits);
  if (physicalInputs.size() < represented->type.getNumInputs() ||
      physicalResults.size() < represented->type.getNumResults())
    return rejected("physical port capacity cannot cover the actor arity");

  const auto semanticWidths = [&](mlir::TypeRange types)
      -> llvm::Expected<llvm::SmallVector<unsigned, 4>> {
    llvm::SmallVector<unsigned, 4> widths;
    widths.reserve(types.size());
    for (mlir::Type type : types) {
      std::string message;
      mlir::FailureOr<unsigned> width =
          ::fabric::getSemanticPayloadWidth(type, pointerLayout, message);
      if (mlir::failed(width))
        return rejected(message);
      widths.push_back(*width);
    }
    llvm::sort(widths);
    return widths;
  };
  auto semanticInputs = semanticWidths(represented->type.getInputs());
  if (!semanticInputs)
    return semanticInputs.takeError();
  auto semanticResults = semanticWidths(represented->type.getResults());
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
    return rejected("no physical input correspondence has enough width");
  if (!hasWidthCapacity(*semanticResults, physicalResults))
    return rejected("no physical result correspondence has enough width");
  return llvm::Error::success();
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

  const auto rejected = [](const llvm::Twine &message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "fabric_operation_capability_rejected: " +
                                       message);
  };
  auto represented = ::fabric::projectResolvedIndexTypes(actor, indexBitWidth);
  if (!represented)
    return represented.takeError();

  const auto verify = [&](mlir::TypeRange types,
                          llvm::ArrayRef<std::uint64_t> selected,
                          FabricPortDirection direction) -> llvm::Error {
    if (types.size() != selected.size())
      return rejected("ordered physical-port correspondence has wrong arity");
    llvm::SmallDenseSet<std::uint64_t, 8> used;
    for (auto [softwareOrdinal, type] : llvm::enumerate(types)) {
      const std::uint64_t physicalOrdinal = selected[softwareOrdinal];
      if (!used.insert(physicalOrdinal).second)
        return rejected("ordered physical-port correspondence reuses a port");
      const auto found = llvm::find_if(
          physicalPorts, [&](const ResolvedFabricOpPhysicalPortView &port) {
            return port.reference.direction == direction &&
                   port.reference.ordinal == physicalOrdinal;
          });
      if (found == physicalPorts.end())
        return rejected(
            "ordered physical-port correspondence selects a missing port");
      std::string message;
      mlir::FailureOr<unsigned> width =
          ::fabric::getSemanticPayloadWidth(type, pointerLayout, message);
      if (mlir::failed(width))
        return rejected(message);
      if (found->payloadWidthBits < *width)
        return rejected(
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

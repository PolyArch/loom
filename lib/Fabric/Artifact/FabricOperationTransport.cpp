#include "FabricOperationTransport.h"

#include "FabricMemoryEngineTemplate.h"

#include "Fabric/IR/FabricOps.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Error.h"

using namespace mlir;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

} // namespace

llvm::Expected<loom::fabric::detail::FabricOperationTransportTypes>
loom::fabric::detail::resolveFabricOperationTransportTypes(
    Operation *operation) {
  FabricOperationTransportTypes result;
  if (auto memory = dyn_cast<::fabric::MemOp>(operation)) {
    auto type = resolveFabricMemoryFunctionType(memory);
    if (!type)
      return type.takeError();
    for (Type input : type->getInputs())
      if (!isa<MemRefType>(input))
        result.inputs.push_back(input);
  } else if (auto boundary = dyn_cast<::fabric::BoundaryOp>(operation)) {
    const ArrayRef<Type> inner = boundary.getInnerInputTypes();
    if (!inner.empty())
      result.inputs.append(inner.begin(), inner.end());
    else
      for (Value input : operation->getOperands())
        result.inputs.push_back(input.getType());
  } else if (auto sw = dyn_cast<::fabric::SwitchOp>(operation)) {
    const ArrayRef<Type> inner = sw.getInnerInputTypes();
    if (!inner.empty())
      result.inputs.append(inner.begin(), inner.end());
    else
      for (Value input : operation->getOperands())
        result.inputs.push_back(input.getType());
  } else {
    for (Value input : operation->getOperands())
      result.inputs.push_back(input.getType());
  }

  for (Type output : operation->getResultTypes())
    if (!isa<MemRefType>(output))
      result.outputs.push_back(output);
  return result;
}

std::optional<loom::fabric::FabricTransportEndpointOwnerRef>
loom::fabric::detail::projectFabricTransportOwner(FabricEntityKind kind,
                                                  FabricEntityId id) {
  switch (kind) {
  case FabricEntityKind::FabricPeOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricPeOccurrenceRef(id));
  case FabricEntityKind::FabricFuOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricFuOccurrenceRef(id));
  case FabricEntityKind::FabricMemoryOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricMemoryOccurrenceRef(id));
  case FabricEntityKind::FabricSwitchOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricSwitchOccurrenceRef(id));
  case FabricEntityKind::FabricFifoOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricFifoOccurrenceRef(id));
  case FabricEntityKind::FabricBoundaryOccurrence:
    return FabricTransportEndpointOwnerRef::of(FabricBoundaryOccurrenceRef(id));
  default:
    return std::nullopt;
  }
}

llvm::Expected<std::optional<loom::fabric::FabricOrdinal>>
loom::fabric::detail::resolveFabricTokenInputOrdinal(
    Operation *operation, std::uint64_t signatureOrdinal) {
  auto memory = dyn_cast<::fabric::MemOp>(operation);
  if (!memory)
    return std::optional<FabricOrdinal>(signatureOrdinal);
  auto type = resolveFabricMemoryFunctionType(memory);
  if (!type)
    return type.takeError();
  if (signatureOrdinal >= type->getNumInputs())
    return invalid("fabric.mem input ordinal is outside its signature");
  if (isa<MemRefType>(type->getInput(signatureOrdinal)))
    return std::optional<FabricOrdinal>();
  FabricOrdinal tokenOrdinal = 0;
  for (std::uint64_t index = 0; index < signatureOrdinal; ++index)
    tokenOrdinal += !isa<MemRefType>(type->getInput(index));
  return std::optional<FabricOrdinal>(tokenOrdinal);
}

llvm::Expected<std::optional<loom::fabric::FabricOrdinal>>
loom::fabric::detail::resolveFabricTokenOutputOrdinal(
    Operation *operation, std::uint64_t signatureOrdinal) {
  auto memory = dyn_cast<::fabric::MemOp>(operation);
  if (!memory)
    return std::optional<FabricOrdinal>(operation->getNumOperands() +
                                        signatureOrdinal);
  auto type = resolveFabricMemoryFunctionType(memory);
  if (!type)
    return type.takeError();
  if (signatureOrdinal >= type->getNumResults())
    return invalid("fabric.mem result ordinal is outside its signature");
  if (isa<MemRefType>(type->getResult(signatureOrdinal)))
    return std::optional<FabricOrdinal>();
  FabricOrdinal tokenOrdinal = 0;
  for (Type input : type->getInputs())
    tokenOrdinal += !isa<MemRefType>(input);
  for (std::uint64_t index = 0; index < signatureOrdinal; ++index)
    tokenOrdinal += !isa<MemRefType>(type->getResult(index));
  return std::optional<FabricOrdinal>(tokenOrdinal);
}

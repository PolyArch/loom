#include "FabricOperationTransport.h"

#include "FabricMemoryEngineTemplate.h"

#include "Fabric/IR/FabricOps.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

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

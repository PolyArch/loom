#include "FabricMemoryEngineTemplate.h"

#include "Fabric/IR/MemoryConnectivityContract.h"
#include "Fabric/IR/MemoryOperationPort.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <cstdint>
#include <limits>
#include <system_error>
#include <utility>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "fabric_artifact_invalid: memory engine template " + message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Error appendCount(std::vector<std::uint8_t> &bytes, std::size_t count,
                        llvm::StringRef field) {
  if (count > std::numeric_limits<std::uint64_t>::max())
    return invalid(field + " count exceeds u64");
  appendU64(bytes, static_cast<std::uint64_t>(count));
  return llvm::Error::success();
}

llvm::Error appendFrame(std::vector<std::uint8_t> &bytes,
                        llvm::ArrayRef<std::uint8_t> value,
                        llvm::StringRef field) {
  if (llvm::Error error = appendCount(bytes, value.size(), field))
    return error;
  bytes.insert(bytes.end(), value.begin(), value.end());
  return llvm::Error::success();
}

std::vector<std::uint8_t> unsignedBytes(llvm::ArrayRef<std::int8_t> bytes) {
  std::vector<std::uint8_t> result;
  result.reserve(bytes.size());
  for (std::int8_t byte : bytes)
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

} // namespace

llvm::Expected<FunctionType>
resolveFabricMemoryFunctionType(::fabric::MemOp memory) {
  if (auto typeAttribute = memory.getFunctionTypeAttr()) {
    auto type = dyn_cast<FunctionType>(typeAttribute.getValue());
    if (!type)
      return invalid("function_type is not a FunctionType");
    return type;
  }

  llvm::SmallVector<Type> inputs;
  llvm::ArrayRef<Type> innerTypes = memory.getInnerInputTypes();
  if (!innerTypes.empty())
    inputs.append(innerTypes.begin(), innerTypes.end());
  else
    for (Value input : memory.getInputs())
      inputs.push_back(input.getType());
  return FunctionType::get(memory.getContext(), inputs,
                           memory.getResultTypes());
}

llvm::Expected<std::optional<DerivedFabricMemoryEngineTemplate>>
deriveFabricMemoryEngineTemplate(::fabric::MemOp memory) {
  ::fabric::MemoryContractAttr contract = memory.getMemoryContract();
  ::fabric::MemoryEngineAttr engine = contract.getEngine();
  if (!engine)
    return std::optional<DerivedFabricMemoryEngineTemplate>();

  auto functionType = resolveFabricMemoryFunctionType(memory);
  if (!functionType)
    return functionType.takeError();
  auto endpoints =
      ::fabric::deriveMemoryTransportEndpointInventory(*functionType);
  if (!endpoints)
    return endpoints.takeError();
  auto operationPorts = ::fabric::decodeMemoryOperationPortInventory(
      memory.getMemoryOperationPortsAttr(), memory.getContext(),
      engine.getSchedule(), *endpoints);
  if (!operationPorts)
    return operationPorts.takeError();

  auto connectivity = ::fabric::decodeMemoryConnectivityContractRecord(
      unsignedBytes(contract.getConnectivity().getRecord().asArrayRef()));
  if (!connectivity)
    return connectivity.takeError();

  std::optional<std::uint64_t> residentContextCount;
  if (::fabric::MemoryResidentContextsAttr contexts =
          engine.getResidentContexts())
    residentContextCount = contexts.getCount();

  FabricMemoryEngineTemplateRecord record{
      engine.getSchedule(), residentContextCount, std::move(*endpoints),
      std::move(*operationPorts),
      std::vector<::fabric::MemoryInternalConnectionDeclaration>(
          connectivity->internalConnections().begin(),
          connectivity->internalConnections().end())};

  std::vector<std::uint8_t> canonicalBytes;
  appendU32(canonicalBytes, static_cast<std::uint32_t>(record.schedule));
  appendU32(canonicalBytes, record.residentContextCount.has_value());
  if (record.residentContextCount)
    appendU64(canonicalBytes, *record.residentContextCount);

  if (llvm::Error error = appendCount(
          canonicalBytes, record.tokenEndpoints.size(), "token endpoints"))
    return std::move(error);
  for (const ::fabric::MemoryTransportEndpointDescriptor &endpoint :
       record.tokenEndpoints) {
    appendU32(canonicalBytes, static_cast<std::uint32_t>(endpoint.direction));
    appendU32(canonicalBytes, endpoint.payloadWidth);
    appendU32(canonicalBytes, endpoint.tagWidth.has_value());
    if (endpoint.tagWidth)
      appendU32(canonicalBytes, *endpoint.tagWidth);
  }

  if (llvm::Error error = appendCount(
          canonicalBytes, record.operationPorts.size(), "operation ports"))
    return std::move(error);
  for (const ::fabric::MemoryOperationPortRecord &port :
       record.operationPorts) {
    auto encoded = ::fabric::encodeMemoryOperationPortRecord(port);
    if (!encoded)
      return encoded.takeError();
    if (llvm::Error error =
            appendFrame(canonicalBytes, *encoded, "operation port"))
      return std::move(error);
  }

  if (llvm::Error error =
          appendCount(canonicalBytes, record.internalConnections.size(),
                      "internal connections"))
    return std::move(error);
  for (const ::fabric::MemoryInternalConnectionDeclaration &connection :
       record.internalConnections) {
    appendU64(canonicalBytes, connection.sourceEndpointOrdinal);
    appendU64(canonicalBytes, connection.sinkEndpointOrdinal);
  }

  return std::optional<DerivedFabricMemoryEngineTemplate>(
      DerivedFabricMemoryEngineTemplate{std::move(record),
                                        std::move(canonicalBytes)});
}

} // namespace loom::fabric::detail

#include "Fabric/IR/FabricAttrs.h"
#include "Fabric/IR/FuCapabilityDomain.h"

#include <cstdint>
#include <vector>

using namespace mlir;
using namespace fabric;

LogicalResult FuCapabilityDomainAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError,
    DenseI8ArrayAttr record) {
  if (!record)
    return emitError() << "FU capability domain requires a canonical record";
  std::vector<std::uint8_t> bytes;
  bytes.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    bytes.push_back(static_cast<std::uint8_t>(byte));
  auto decoded = decodeFuCapabilityDomainRecord(bytes);
  if (!decoded)
    return emitError() << llvm::toString(decoded.takeError());
  return success();
}

#include "FabricSystemValidation.h"

#include "Fabric/Artifact/FabricSystemContracts.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

using namespace mlir;

namespace loom::fabric::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

std::vector<std::uint8_t> unsignedBytes(DenseI8ArrayAttr attribute) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(attribute.size());
  for (std::int8_t byte : attribute.asArrayRef())
    bytes.push_back(static_cast<std::uint8_t>(byte));
  return bytes;
}

} // namespace

llvm::Error validateInstructionCoreCohort(::fabric::SystemOp root) {
  std::uint64_t hostCount = 0;
  std::uint64_t accCoreCount = 0;
  std::optional<RiscVXLen> commonXlen;
  std::optional<InstructionEndianness> commonEndianness;
  std::vector<RiscVAbi> commonAbis;

  for (Operation &operation : root.getBody().front()) {
    DenseI8ArrayAttr architectureBytes;
    if (auto host = dyn_cast<::fabric::SystemHostCoreOp>(&operation)) {
      ++hostCount;
      architectureBytes = host.getArchitectureAttr();
    } else if (auto core = dyn_cast<::fabric::SystemAccCoreOp>(&operation)) {
      ++accCoreCount;
      architectureBytes = core.getArchitectureAttr();
    } else {
      continue;
    }

    auto architecture = decodeInstructionCoreArchitecturalContract(
        unsignedBytes(architectureBytes));
    if (!architecture)
      return architecture.takeError();
    if (!commonXlen) {
      commonXlen = architecture->xlen();
      commonEndianness = architecture->endianness();
      commonAbis.assign(architecture->abiCapabilities().begin(),
                        architecture->abiCapabilities().end());
      continue;
    }
    if (architecture->xlen() != *commonXlen ||
        architecture->endianness() != *commonEndianness)
      return invalid(
          "System InstructionCores do not share a common InstructionCore "
          "XLEN and endianness");
    std::vector<RiscVAbi> intersection;
    std::set_intersection(commonAbis.begin(), commonAbis.end(),
                          architecture->abiCapabilities().begin(),
                          architecture->abiCapabilities().end(),
                          std::back_inserter(intersection));
    commonAbis = std::move(intersection);
  }

  if (hostCount != 1)
    return invalid("root-complete System requires exactly one HostCore");
  if (accCoreCount == 0)
    return invalid("root-complete System requires at least one AccCore");
  if (commonAbis.empty())
    return invalid(
        "System InstructionCores have no common InstructionCore ABI");
  return llvm::Error::success();
}

} // namespace loom::fabric::detail

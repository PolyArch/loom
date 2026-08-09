#include "Fabric/IR/Crosspoint.h"

#include "llvm/Support/Error.h"

#include <limits>

llvm::Expected<std::uint64_t>
fabric::checkedCrosspointCount(std::uint64_t inputCount,
                               std::uint64_t outputCount) {
  if (inputCount == 0 || outputCount == 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "crosspoint dimensions must be non-empty");
  if (inputCount > std::numeric_limits<std::uint64_t>::max() / outputCount)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "crosspoint product overflows u64");
  return inputCount * outputCount;
}

llvm::Expected<std::uint64_t>
fabric::validatedPeBoundaryCrosspointCount(std::uint64_t inputCount,
                                           std::uint64_t outputCount) {
  auto crosspoints = checkedCrosspointCount(inputCount, outputCount);
  if (!crosspoints)
    return crosspoints.takeError();
  if (*crosspoints > kPeCrosspointLimit)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "fabric.pe boundary selectors have %llu crosspoints, exceeding "
        "maximum %llu",
        static_cast<unsigned long long>(*crosspoints),
        static_cast<unsigned long long>(kPeCrosspointLimit));
  return *crosspoints;
}

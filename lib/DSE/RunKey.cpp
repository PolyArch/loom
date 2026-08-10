#include "DSE/InvocationManifest.h"

#include "llvm/Support/Error.h"

#include <algorithm>

namespace loom::dse {

llvm::Expected<DseRunKey>
DseRunKey::fromBytes(llvm::ArrayRef<std::uint8_t> bytes) {
  if (bytes.size() != byteSize)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "DSE run key requires exactly 32 bytes");
  Storage storage;
  std::copy(bytes.begin(), bytes.end(), storage.begin());
  return DseRunKey(storage);
}

} // namespace loom::dse

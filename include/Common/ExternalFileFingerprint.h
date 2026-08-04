#ifndef LOOM_COMMON_EXTERNALFILEFINGERPRINT_H
#define LOOM_COMMON_EXTERNALFILEFINGERPRINT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace loom {

class ExternalFileFingerprint final {
public:
  using Storage = std::array<std::uint8_t, 32>;
  static constexpr std::size_t byteSize = 32;

  static llvm::Expected<ExternalFileFingerprint>
  fromBytes(llvm::ArrayRef<std::uint8_t> bytes);

  const Storage &bytes() const { return bytes_; }

  friend bool operator==(const ExternalFileFingerprint &lhs,
                         const ExternalFileFingerprint &rhs) {
    return lhs.bytes_ == rhs.bytes_;
  }
  friend bool operator!=(const ExternalFileFingerprint &lhs,
                         const ExternalFileFingerprint &rhs) {
    return !(lhs == rhs);
  }

private:
  explicit ExternalFileFingerprint(Storage bytes) : bytes_(bytes) {}

  Storage bytes_;
};

std::string
formatExternalFileFingerprint(const ExternalFileFingerprint &fingerprint);
llvm::Expected<ExternalFileFingerprint>
parseExternalFileFingerprint(llvm::StringRef spelling);

} // namespace loom

#endif // LOOM_COMMON_EXTERNALFILEFINGERPRINT_H

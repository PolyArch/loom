#ifndef LOOM_EXTERNALTOOL_EXTERNALFILE_H
#define LOOM_EXTERNALTOOL_EXTERNALFILE_H

#include "ExternalTool/LocalConfig.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace loom::external_tool {

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

struct ExternalFileRequirement final {
  std::string providerInputSlot;
  ExternalFileFingerprint fingerprint;
};

struct ResolvedExternalFile final {
  std::string providerInputSlot;
  std::string localFileKey;
  std::string absolutePath;
  ExternalFileFingerprint fingerprint;
};

llvm::Expected<std::vector<ResolvedExternalFile>>
resolveExternalFiles(llvm::ArrayRef<ExternalFileRequirement> requirements,
                     const LocalToolConfig &config);

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_EXTERNALFILE_H

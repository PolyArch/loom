#ifndef LOOM_EXTERNALTOOL_EXTERNALFILE_H
#define LOOM_EXTERNALTOOL_EXTERNALFILE_H

#include "ExternalTool/LocalConfig.h"

#include "Common/ExternalFileFingerprint.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <vector>

namespace loom::external_tool {

using ::loom::ExternalFileFingerprint;
using ::loom::formatExternalFileFingerprint;
using ::loom::parseExternalFileFingerprint;

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

struct ExternalFileTreeMember final {
  std::string relativePath;
  ExternalFileFingerprint fingerprint;

  friend bool operator==(const ExternalFileTreeMember &lhs,
                         const ExternalFileTreeMember &rhs) {
    return lhs.relativePath == rhs.relativePath &&
           lhs.fingerprint == rhs.fingerprint;
  }
};

struct ExternalFileTreeRequirement final {
  std::string providerInputSlot;
  std::vector<ExternalFileTreeMember> members;
};

struct ResolvedExternalFileTree final {
  std::string providerInputSlot;
  std::string localFileTreeKey;
  std::string absolutePath;
  std::vector<ExternalFileTreeMember> members;
};

llvm::Expected<std::vector<ResolvedExternalFile>>
resolveExternalFiles(llvm::ArrayRef<ExternalFileRequirement> requirements,
                     const LocalToolConfig &config);

llvm::Expected<std::vector<ResolvedExternalFileTree>> resolveExternalFileTrees(
    llvm::ArrayRef<ExternalFileTreeRequirement> requirements,
    const LocalToolConfig &config);

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_EXTERNALFILE_H

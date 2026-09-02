#ifndef LOOM_EXTERNALTOOL_EXTERNALFILE_H
#define LOOM_EXTERNALTOOL_EXTERNALFILE_H

#include "ExternalTool/LocalConfig.h"

#include "Common/ExternalFileFingerprint.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
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

/// One provider-selected ordinary executable whose exact bytes and local path
/// are frozen into an invocation. Unlike an external data file, this typed
/// value may own argv[0] for a structured command.
struct ResolvedAuxiliaryToolExecutable final {
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

/// The identity under which one ordinary file was observed: device, inode,
/// mode, link count, size, and modification and change times. Two equal
/// identities denote one unchanged file object; any difference means the
/// bytes must be fingerprinted again.
struct ExternalFileIdentity final {
  std::uint64_t device = 0;
  std::uint64_t inode = 0;
  std::uint64_t mode = 0;
  std::uint64_t linkCount = 0;
  std::uint64_t size = 0;
  std::int64_t modifiedSeconds = 0;
  std::int64_t modifiedNanoseconds = 0;
  std::int64_t changedSeconds = 0;
  std::int64_t changedNanoseconds = 0;

  friend bool operator==(const ExternalFileIdentity &lhs,
                         const ExternalFileIdentity &rhs) {
    return lhs.device == rhs.device && lhs.inode == rhs.inode &&
           lhs.mode == rhs.mode && lhs.linkCount == rhs.linkCount &&
           lhs.size == rhs.size &&
           lhs.modifiedSeconds == rhs.modifiedSeconds &&
           lhs.modifiedNanoseconds == rhs.modifiedNanoseconds &&
           lhs.changedSeconds == rhs.changedSeconds &&
           lhs.changedNanoseconds == rhs.changedNanoseconds;
  }
  friend bool operator!=(const ExternalFileIdentity &lhs,
                         const ExternalFileIdentity &rhs) {
    return !(lhs == rhs);
  }
};

/// One exact fingerprint together with the identity the file had while its
/// bytes were read.
struct ExternalFileObservation final {
  ExternalFileIdentity identity;
  ExternalFileFingerprint fingerprint;
};

/// Observes the identity of one canonical ordinary file without reading its
/// bytes, rejecting symlinks.
llvm::Expected<ExternalFileIdentity>
observeExternalFileIdentity(llvm::StringRef path);

/// Computes the exact SHA-256 of one canonical ordinary file while rejecting
/// symlinks and concurrent replacement or mutation, and reports the identity
/// the file held while it was read.
llvm::Expected<ExternalFileObservation>
observeExternalFile(llvm::StringRef path);

/// Computes the exact SHA-256 of one canonical ordinary file while rejecting
/// symlinks and concurrent replacement or mutation.
llvm::Expected<ExternalFileFingerprint>
fingerprintExternalFile(llvm::StringRef path);

llvm::Error validateExternalFileTreeRequirement(
    const ExternalFileTreeRequirement &requirement);

llvm::Expected<std::vector<ResolvedExternalFile>>
resolveExternalFiles(llvm::ArrayRef<ExternalFileRequirement> requirements,
                     const LocalToolConfig &config);

llvm::Expected<std::vector<ResolvedExternalFileTree>> resolveExternalFileTrees(
    llvm::ArrayRef<ExternalFileTreeRequirement> requirements,
    const LocalToolConfig &config);

} // namespace loom::external_tool

#endif // LOOM_EXTERNALTOOL_EXTERNALFILE_H

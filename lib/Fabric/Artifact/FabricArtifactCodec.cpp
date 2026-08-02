#include "Fabric/Artifact/FabricArtifactCodec.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace loom::fabric {
namespace {

constexpr char semanticDomain[] = "loom.fabric.semantic.v1\0";
constexpr std::size_t payloadLengthFieldSize = 8;
constexpr std::size_t dependencyFixedSuffixSize =
    4 + 4 + ArtifactIdentity::byteSize;
constexpr std::size_t minimumDependencyRowSize =
    4 + 4 + 1 + dependencyFixedSuffixSize;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fabric_artifact_invalid: " + message);
}

llvm::Error implementationInputOwnerUnavailable() {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "fabric_artifact_owner_contract_unavailable: ImplementationInput has "
      "no closed artifact owner contract in loom.fabric 1.1");
}

bool dependencyRowsFit(std::uint64_t count, std::size_t remaining) {
  if (remaining < payloadLengthFieldSize)
    return false;
  return count <=
         (remaining - payloadLengthFieldSize) / minimumDependencyRowSize;
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

llvm::Expected<std::uint32_t> rootOrdinal(FabricRootKind kind) {
  switch (kind) {
  case FabricRootKind::Module:
    return 0;
  case FabricRootKind::System:
    return 1;
  case FabricRootKind::InterconnectImplementation:
    return 2;
  }
  return invalid("unknown root kind");
}

llvm::Expected<std::uint32_t> roleOrdinal(FabricDependencyRole role) {
  switch (role) {
  case FabricDependencyRole::ImportedModule:
    return 0;
  case FabricDependencyRole::RefinedSystem:
    return 1;
  case FabricDependencyRole::ImplementationInput:
    return implementationInputOwnerUnavailable();
  }
  return invalid("unknown dependency role");
}

llvm::Expected<FabricRootKind> decodeRootKind(std::uint32_t value) {
  switch (value) {
  case 0:
    return FabricRootKind::Module;
  case 1:
    return FabricRootKind::System;
  case 2:
    return FabricRootKind::InterconnectImplementation;
  default:
    return invalid("unknown root kind ordinal");
  }
}

llvm::Expected<FabricDependencyRole> decodeRole(std::uint32_t value) {
  switch (value) {
  case 0:
    return FabricDependencyRole::ImportedModule;
  case 1:
    return FabricDependencyRole::RefinedSystem;
  case 2:
    return FabricDependencyRole::ImplementationInput;
  default:
    return invalid("unknown dependency role ordinal");
  }
}

llvm::Expected<std::vector<std::uint8_t>>
encodeDependencyRow(const FabricDirectDependency &dependency) {
  auto role = roleOrdinal(dependency.role);
  if (!role)
    return role.takeError();
  if (dependency.root.schemaIdentity.empty())
    return invalid("dependency schema identity is empty");
  if (dependency.root.schemaIdentity.size() >
      std::numeric_limits<std::uint32_t>::max())
    return invalid("dependency schema identity is too large");

  std::vector<std::uint8_t> bytes;
  bytes.reserve(4 + 4 + dependency.root.schemaIdentity.size() + 8 +
                ArtifactIdentity::byteSize);
  appendU32(bytes, *role);
  appendU32(bytes,
            static_cast<std::uint32_t>(dependency.root.schemaIdentity.size()));
  bytes.insert(bytes.end(), dependency.root.schemaIdentity.begin(),
               dependency.root.schemaIdentity.end());
  appendU32(bytes, dependency.root.schemaVersion.major);
  appendU32(bytes, dependency.root.schemaVersion.minor);
  bytes.insert(bytes.end(), dependency.root.artifact.bytes().begin(),
               dependency.root.artifact.bytes().end());
  return bytes;
}

llvm::Expected<std::vector<std::vector<std::uint8_t>>>
canonicalDependencyRows(llvm::ArrayRef<FabricDirectDependency> dependencies) {
  std::vector<std::vector<std::uint8_t>> rows;
  rows.reserve(dependencies.size());
  for (const FabricDirectDependency &dependency : dependencies) {
    auto row = encodeDependencyRow(dependency);
    if (!row)
      return row.takeError();
    rows.push_back(std::move(*row));
  }
  std::sort(rows.begin(), rows.end());
  for (std::size_t index = 1; index < rows.size(); ++index)
    if (rows[index - 1] == rows[index])
      return invalid("duplicate direct dependency row");
  return rows;
}

class Reader {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : remaining_(bytes) {}

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> take(std::uint64_t count,
                                                    llvm::StringRef what) {
    if (count > remaining_.size())
      return invalid(llvm::Twine("truncated ") + what);
    llvm::ArrayRef<std::uint8_t> prefix =
        remaining_.take_front(static_cast<std::size_t>(count));
    remaining_ = remaining_.drop_front(static_cast<std::size_t>(count));
    return prefix;
  }

  llvm::Expected<std::uint32_t> u32(llvm::StringRef what) {
    auto bytes = take(4, what);
    if (!bytes)
      return bytes.takeError();
    return (static_cast<std::uint32_t>((*bytes)[0]) << 24) |
           (static_cast<std::uint32_t>((*bytes)[1]) << 16) |
           (static_cast<std::uint32_t>((*bytes)[2]) << 8) |
           static_cast<std::uint32_t>((*bytes)[3]);
  }

  llvm::Expected<std::uint64_t> u64(llvm::StringRef what) {
    auto bytes = take(8, what);
    if (!bytes)
      return bytes.takeError();
    std::uint64_t value = 0;
    for (std::uint8_t byte : *bytes)
      value = (value << 8) | byte;
    return value;
  }

  bool empty() const { return remaining_.empty(); }
  llvm::ArrayRef<std::uint8_t> remainingBytes() const { return remaining_; }
  std::size_t remainingSize() const { return remaining_.size(); }

private:
  llvm::ArrayRef<std::uint8_t> remaining_;
};

} // namespace

llvm::Expected<CanonicalSemanticBytes> encodeFabricArtifactEnvelope(
    FabricRootKind rootKind,
    llvm::ArrayRef<FabricDirectDependency> dependencies,
    llvm::ArrayRef<std::uint8_t> canonicalMlirBytecode) {
  auto root = rootOrdinal(rootKind);
  if (!root)
    return root.takeError();
  auto dependencyRows = canonicalDependencyRows(dependencies);
  if (!dependencyRows)
    return dependencyRows.takeError();

  std::vector<std::uint8_t> bytes(semanticDomain,
                                  semanticDomain + sizeof(semanticDomain) - 1);
  appendU32(bytes, *root);
  appendU64(bytes, dependencyRows->size());
  for (const std::vector<std::uint8_t> &row : *dependencyRows)
    bytes.insert(bytes.end(), row.begin(), row.end());
  appendU64(bytes, canonicalMlirBytecode.size());
  bytes.insert(bytes.end(), canonicalMlirBytecode.begin(),
               canonicalMlirBytecode.end());
  return CanonicalSemanticBytes(std::move(bytes));
}

llvm::Expected<DecodedFabricArtifact>
decodeFabricArtifactEnvelope(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  auto domain = reader.take(sizeof(semanticDomain) - 1, "semantic domain");
  if (!domain)
    return domain.takeError();
  if (!domain->equals(llvm::ArrayRef<std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(semanticDomain),
          sizeof(semanticDomain) - 1)))
    return invalid("wrong semantic domain");

  auto rootOrdinalValue = reader.u32("root kind");
  if (!rootOrdinalValue)
    return rootOrdinalValue.takeError();
  auto rootKind = decodeRootKind(*rootOrdinalValue);
  if (!rootKind)
    return rootKind.takeError();

  auto dependencyCount = reader.u64("dependency count");
  if (!dependencyCount)
    return dependencyCount.takeError();
  if (!dependencyRowsFit(*dependencyCount, reader.remainingSize()))
    return invalid("dependency count cannot fit the remaining envelope");

  std::vector<FabricDirectDependency> dependencies;
  dependencies.reserve(static_cast<std::size_t>(*dependencyCount));
  llvm::ArrayRef<std::uint8_t> previousRow;
  for (std::uint64_t index = 0; index < *dependencyCount; ++index) {
    llvm::ArrayRef<std::uint8_t> rowBytes = reader.remainingBytes();
    auto roleValue = reader.u32("dependency role");
    if (!roleValue)
      return roleValue.takeError();
    auto role = decodeRole(*roleValue);
    if (!role)
      return role.takeError();
    auto schemaLength = reader.u32("dependency schema identity length");
    if (!schemaLength)
      return schemaLength.takeError();
    if (*schemaLength == 0)
      return invalid("dependency schema identity is empty");
    if (*schemaLength > reader.remainingSize() ||
        reader.remainingSize() - *schemaLength < dependencyFixedSuffixSize ||
        !dependencyRowsFit(*dependencyCount - index - 1,
                           reader.remainingSize() - *schemaLength -
                               dependencyFixedSuffixSize))
      return invalid(
          "dependency schema identity length cannot fit the remaining "
          "envelope");
    auto schemaBytes = reader.take(*schemaLength, "dependency schema identity");
    if (!schemaBytes)
      return schemaBytes.takeError();
    auto major = reader.u32("dependency schema major version");
    if (!major)
      return major.takeError();
    auto minor = reader.u32("dependency schema minor version");
    if (!minor)
      return minor.takeError();
    auto identityBytes =
        reader.take(ArtifactIdentity::byteSize, "dependency identity");
    if (!identityBytes)
      return identityBytes.takeError();
    auto identity = ArtifactIdentity::fromBytes(*identityBytes);
    if (!identity)
      return identity.takeError();

    FabricDirectDependency dependency{
        *role,
        ArtifactRootReference{
            std::string(reinterpret_cast<const char *>(schemaBytes->data()),
                        schemaBytes->size()),
            SchemaVersion{*major, *minor}, std::move(*identity)}};
    rowBytes = rowBytes.take_front(rowBytes.size() - reader.remainingSize());
    if (!previousRow.empty() &&
        !std::lexicographical_compare(previousRow.begin(), previousRow.end(),
                                      rowBytes.begin(), rowBytes.end()))
      return invalid(previousRow == rowBytes ? "duplicate direct dependency row"
                                             : "noncanonical dependency order");
    previousRow = rowBytes;
    dependencies.push_back(std::move(dependency));
  }

  auto payloadLength = reader.u64("canonical MLIR bytecode length");
  if (!payloadLength)
    return payloadLength.takeError();
  auto payload = reader.take(*payloadLength, "canonical MLIR bytecode");
  if (!payload)
    return payload.takeError();
  if (!reader.empty())
    return invalid("trailing bytes after canonical MLIR bytecode");

  return DecodedFabricArtifact{
      *rootKind, std::move(dependencies),
      std::vector<std::uint8_t>(payload->begin(), payload->end())};
}

} // namespace loom::fabric

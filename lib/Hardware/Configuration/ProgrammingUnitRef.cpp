#include "Hardware/Configuration/ConfigurationABI.h"

#include "Common/ArtifactLocalReference.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom::hardware {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "configuration_programming_unit_ref_invalid: " + message);
}

void appendU64Be(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

llvm::Expected<std::uint64_t> readU64Be(llvm::ArrayRef<std::uint8_t> bytes,
                                        std::size_t offset) {
  if (offset > bytes.size() || bytes.size() - offset != sizeof(std::uint64_t))
    return invalid("reference does not end in one u64be unit ID");
  std::uint64_t value = 0;
  for (std::uint8_t byte : bytes.drop_front(offset))
    value = (value << 8) | byte;
  return value;
}

} // namespace

std::vector<std::uint8_t>
encodeProgrammingUnitRef(const ProgrammingUnitRef &reference) {
  std::vector<std::uint8_t> bytes =
      encodeArtifactRootReference(reference.configurationAbi);
  appendU64Be(bytes, reference.unitId);
  return bytes;
}

llvm::Expected<ProgrammingUnitRef>
detail::decodeProgrammingUnitRefFraming(llvm::ArrayRef<std::uint8_t> bytes) {
  auto root = decodeArtifactRootReferencePrefix(bytes);
  if (!root)
    return root.takeError();
  if (root->reference.schemaIdentity != configurationAbiSchema.identity ||
      root->reference.schemaVersion != configurationAbiSchema.version)
    return invalid("reference does not name loom.configuration_abi 2.0");
  auto unitId = readU64Be(bytes, root->byteCount);
  if (!unitId)
    return unitId.takeError();
  ProgrammingUnitRef reference{std::move(root->reference), *unitId};
  const std::vector<std::uint8_t> canonical =
      encodeProgrammingUnitRef(reference);
  if (llvm::ArrayRef<std::uint8_t>(canonical) != bytes)
    return invalid("reference is not canonical");
  return reference;
}

llvm::Expected<ProgrammingUnitRef>
decodeProgrammingUnitRef(llvm::ArrayRef<std::uint8_t> bytes,
                         const ArtifactStore &store) {
  auto reference = detail::decodeProgrammingUnitRefFraming(bytes);
  if (!reference)
    return reference.takeError();
  auto abi = importConfigurationABI(reference->configurationAbi, store);
  if (!abi)
    return abi.takeError();
  if (!abi->abi().findProgrammingUnit(reference->unitId))
    return invalid("reference names an unknown programming unit");
  return std::move(*reference);
}

} // namespace loom::hardware

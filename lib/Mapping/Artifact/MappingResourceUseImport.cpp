#include "MappingResourceUseImport.h"

#include "Fabric/IR/ResourceContract.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>

namespace loom::mapping::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "mapping_artifact_invalid: " + message);
}

std::vector<std::uint8_t> unsignedBytes(mlir::DenseI8ArrayAttr record) {
  std::vector<std::uint8_t> result;
  result.reserve(record.size());
  for (std::int8_t byte : record.asArrayRef())
    result.push_back(static_cast<std::uint8_t>(byte));
  return result;
}

llvm::Expected<std::vector<::fabric::UsePatternValue>>
importValues(mlir::ArrayAttr records,
             llvm::ArrayRef<::fabric::UsePatternValueSchema> schemas,
             llvm::StringRef field) {
  if (records.size() != schemas.size())
    return invalid("ResourceUse " + field +
                   " count disagrees with its Fabric use pattern schema");

  std::vector<::fabric::UsePatternValue> result;
  result.reserve(records.size());
  for (auto [record, schema] : llvm::zip_equal(records, schemas)) {
    auto typed = mlir::dyn_cast<::mapping::OwnerTypedValueAttr>(record);
    if (!typed)
      return invalid("ResourceUse " + field +
                     " contains a non-owner-typed value");
    const std::vector<std::uint8_t> bytes = unsignedBytes(typed.getRecord());
    auto value = ::fabric::decodeUsePatternValue(schema, bytes);
    if (!value)
      return invalid("ResourceUse " + field +
                     " cannot be decoded by its Fabric owner: " +
                     llvm::toString(value.takeError()));
    auto canonical = ::fabric::encodeUsePatternValue(schema, *value);
    if (!canonical)
      return invalid("ResourceUse " + field +
                     " cannot be re-encoded by its Fabric owner: " +
                     llvm::toString(canonical.takeError()));
    if (*canonical != bytes)
      return invalid("ResourceUse " + field +
                     " is not in its owner codec's canonical form");
    result.push_back(std::move(*value));
  }
  return result;
}

} // namespace

llvm::Expected<ImportedPatternValues> importResourceUsePatternValues(
    ::mapping::ResourceUseOp record,
    const ::loom::fabric::FabricArtifactView &fabric,
    const ::loom::fabric::FabricUsePatternRef &pattern) {
  const ::fabric::ResourceContract *contract =
      fabric.resourceContract(pattern.owner.catalog());
  if (!contract || pattern.ordinal >= contract->usePatternCount())
    return invalid("ResourceUse does not resolve an exact Fabric use pattern");
  const ::fabric::UsePattern declaration =
      contract->usePattern(::fabric::UsePatternKey(pattern.ordinal));
  auto parameters = importValues(record.getParameters(), declaration.parameters,
                                 "parameters");
  if (!parameters)
    return parameters.takeError();
  auto sharing =
      importValues(record.getSharingAssignments(),
                   declaration.sharingAssignments, "sharing assignments");
  if (!sharing)
    return sharing.takeError();
  return ImportedPatternValues{std::move(*parameters), std::move(*sharing)};
}

} // namespace loom::mapping::detail

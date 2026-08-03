#include "Fabric/IR/UsePatternValue.h"

#include "Fabric/IR/PhysicalTag.h"

#include "llvm/Support/Errc.h"

using namespace fabric;

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::errc::invalid_argument,
                                 "invalid Fabric use-pattern value: %s",
                                 message.str().c_str());
}

} // namespace

llvm::Expected<UsePatternValue>
fabric::decodeUsePatternValue(const UsePatternValueSchema &schema,
                              llvm::ArrayRef<std::uint8_t> bytes) {
  switch (schema.kind) {
  case UsePatternValueKind::PhysicalTag: {
    auto value = decodePhysicalTagValue(schema.bitWidth, bytes);
    if (!value)
      return value.takeError();
    return UsePatternValue(PhysicalTagPatternValue{std::move(*value)});
  }
  }
  return invalid("unknown schema kind");
}

llvm::Expected<std::vector<std::uint8_t>>
fabric::encodeUsePatternValue(const UsePatternValueSchema &schema,
                              const UsePatternValue &value) {
  switch (schema.kind) {
  case UsePatternValueKind::PhysicalTag: {
    const auto *tag = std::get_if<PhysicalTagPatternValue>(&value);
    if (!tag)
      return invalid("value kind disagrees with its schema");
    if (tag->value.getBitWidth() != schema.bitWidth)
      return invalid("Physical Tag width disagrees with its schema");
    return encodePhysicalTagValue(schema.bitWidth, tag->value);
  }
  }
  return invalid("unknown schema kind");
}

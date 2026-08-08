#ifndef LOOM_DATAFLOW_IR_OPERATION_SCHEMA_CODEC_INTERNAL_H
#define LOOM_DATAFLOW_IR_OPERATION_SCHEMA_CODEC_INTERNAL_H

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Dataflow/IR/OperationSchema.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <system_error>
#include <utility>
#include <vector>

namespace dataflow::detail {

struct CodecVersion final {
  std::uint32_t major;
  std::uint32_t minor;
};

inline constexpr CodecVersion kOperationSchemaCodecVersion{1, 0};
inline constexpr CodecVersion kActorSchemaProjectionCodecVersion{2, 0};

inline llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "operation_schema_codec_invalid: " + message);
}

class Writer {
public:
  void byte(std::uint8_t value) { bytes_.push_back(value); }

  void bytes(llvm::ArrayRef<std::uint8_t> values) {
    bytes_.insert(bytes_.end(), values.begin(), values.end());
  }

  void u32(std::uint32_t value) {
    byte(static_cast<std::uint8_t>(value >> 24));
    byte(static_cast<std::uint8_t>(value >> 16));
    byte(static_cast<std::uint8_t>(value >> 8));
    byte(static_cast<std::uint8_t>(value));
  }

  void u64(std::uint64_t value) {
    for (unsigned shift = 56; shift != 0; shift -= 8)
      byte(static_cast<std::uint8_t>(value >> shift));
    byte(static_cast<std::uint8_t>(value));
  }

  void boolean(bool value) { u32(value ? 1 : 0); }

  void string(llvm::StringRef value) {
    u64(value.size());
    bytes(llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(value.data()), value.size()));
  }

  std::vector<std::uint8_t> take() { return std::move(bytes_); }

private:
  std::vector<std::uint8_t> bytes_;
};

class Reader {
public:
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes)
      : bytes_(bytes), remaining_(bytes) {}

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> take(std::uint64_t count,
                                                    const llvm::Twine &what) {
    if (count > remaining_.size())
      return invalid(llvm::Twine("truncated ") + what);
    llvm::ArrayRef<std::uint8_t> prefix =
        remaining_.take_front(static_cast<std::size_t>(count));
    remaining_ = remaining_.drop_front(static_cast<std::size_t>(count));
    return prefix;
  }

  llvm::Expected<std::uint32_t> u32(const llvm::Twine &what) {
    auto value = take(4, what);
    if (!value)
      return value.takeError();
    return (static_cast<std::uint32_t>((*value)[0]) << 24) |
           (static_cast<std::uint32_t>((*value)[1]) << 16) |
           (static_cast<std::uint32_t>((*value)[2]) << 8) |
           static_cast<std::uint32_t>((*value)[3]);
  }

  llvm::Expected<std::uint64_t> u64(const llvm::Twine &what) {
    auto value = take(8, what);
    if (!value)
      return value.takeError();
    std::uint64_t result = 0;
    for (std::uint8_t byte : *value)
      result = (result << 8) | byte;
    return result;
  }

  llvm::Expected<bool> boolean(const llvm::Twine &what) {
    auto value = u32(what);
    if (!value)
      return value.takeError();
    if (*value > 1)
      return invalid(llvm::Twine(what) + " is not a canonical boolean");
    return *value == 1;
  }

  llvm::Expected<llvm::ArrayRef<std::uint8_t>> string(const llvm::Twine &what) {
    auto size = u64(llvm::Twine(what) + " length");
    if (!size)
      return size.takeError();
    return take(*size, what);
  }

  std::size_t remainingSize() const { return remaining_.size(); }
  std::size_t position() const { return bytes_.size() - remaining_.size(); }
  llvm::ArrayRef<std::uint8_t> bytesSince(std::size_t position) const {
    return bytes_.slice(position, this->position() - position);
  }
  bool empty() const { return remaining_.empty(); }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  llvm::ArrayRef<std::uint8_t> remaining_;
};

template <std::size_t Size>
void writeDomain(Writer &writer, const char (&domain)[Size],
                 CodecVersion version = kOperationSchemaCodecVersion) {
  writer.bytes(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(domain), Size - 1));
  writer.u32(version.major);
  writer.u32(version.minor);
}

template <std::size_t Size>
llvm::Error readDomain(Reader &reader, const char (&domain)[Size],
                       CodecVersion version = kOperationSchemaCodecVersion) {
  auto actual = reader.take(Size - 1, "semantic domain");
  if (!actual)
    return actual.takeError();
  llvm::ArrayRef<std::uint8_t> expected(
      reinterpret_cast<const std::uint8_t *>(domain), Size - 1);
  if (*actual != expected)
    return invalid("wrong semantic domain");
  auto major = reader.u32("codec major version");
  if (!major)
    return major.takeError();
  auto minor = reader.u32("codec minor version");
  if (!minor)
    return minor.takeError();
  if (*major != version.major || *minor != version.minor)
    return invalid("unsupported version");
  return llvm::Error::success();
}

inline llvm::Error writeMappedTag(Writer &writer,
                                  llvm::Expected<std::uint32_t> tag) {
  if (!tag)
    return tag.takeError();
  writer.u32(*tag);
  return llvm::Error::success();
}

inline llvm::Expected<std::uint32_t>
readClosedTag(Reader &reader, std::uint32_t maximum, llvm::StringRef what) {
  auto tag = reader.u32(what);
  if (!tag)
    return tag.takeError();
  if (*tag == 0 || *tag > maximum)
    return invalid(llvm::Twine("unknown ") + what);
  return *tag;
}

inline llvm::Expected<std::uint64_t>
readCount(Reader &reader, const llvm::Twine &what, std::size_t minimumBytes) {
  auto count = reader.u64(what);
  if (!count)
    return count.takeError();
  if (minimumBytes != 0 && *count > reader.remainingSize() / minimumBytes)
    return invalid(llvm::Twine(what) + " cannot fit remaining bytes");
  return *count;
}

enum class ScalarSummary { Other, Index, Integer, Float };

struct TypeSummary {
  ScalarSummary scalar = ScalarSummary::Other;
  std::uint32_t bitWidth = 0;
  std::uint64_t elementCount = 1;
  bool shaped = false;
};

llvm::Error encodeType(Writer &writer, ::mlir::Type type, unsigned depth);
llvm::Expected<TypeSummary> validateType(Reader &reader, unsigned depth);
llvm::Error encodeFunctionType(Writer &writer, ::mlir::FunctionType type);
llvm::Expected<std::vector<llvm::ArrayRef<std::uint8_t>>>
validateFunctionType(Reader &reader);

llvm::Expected<std::uint32_t>
integerPredicateWireTag(::mlir::arith::CmpIPredicate predicate);
llvm::Expected<::mlir::arith::CmpIPredicate>
integerPredicateFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t>
floatPredicateWireTag(::mlir::arith::CmpFPredicate predicate);
llvm::Expected<::mlir::arith::CmpFPredicate>
floatPredicateFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t>
roundingModeWireTag(::mlir::arith::RoundingMode mode);
llvm::Expected<::mlir::arith::RoundingMode>
roundingModeFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t>
serviceValueRoleWireTag(semantics::ServiceValueRole role);
llvm::Expected<semantics::ServiceValueRole>
serviceValueRoleFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t>
memoryAccessFormWireTag(semantics::MemoryAccessForm form);
llvm::Expected<semantics::MemoryAccessForm>
memoryAccessFormFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t>
memoryMaskFormWireTag(semantics::MemoryMaskForm form);
llvm::Expected<semantics::MemoryMaskForm>
memoryMaskFormFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t> atomicOrderingWireTag(AtomicOrdering ordering);
llvm::Expected<AtomicOrdering> atomicOrderingFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t> syncScopeKindWireTag(SyncScopeKind kind);
llvm::Expected<SyncScopeKind> syncScopeKindFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t>
vectorAtomicGranularityWireTag(VectorAtomicGranularity granularity);
llvm::Expected<VectorAtomicGranularity>
vectorAtomicGranularityFromWireTag(std::uint32_t wireTag);

llvm::Expected<std::uint32_t> atomicRmwKindWireTag(AtomicRmwKind kind);
llvm::Expected<AtomicRmwKind> atomicRmwKindFromWireTag(std::uint32_t wireTag);

} // namespace dataflow::detail

#endif // LOOM_DATAFLOW_IR_OPERATION_SCHEMA_CODEC_INTERNAL_H

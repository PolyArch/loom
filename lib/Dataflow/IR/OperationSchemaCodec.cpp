#include "Dataflow/IR/OperationSchemaCodec.h"

#include "OperationSchemaCodecInternal.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <variant>

using namespace mlir;
using dataflow::detail::encodeFunctionType;
using dataflow::detail::encodeType;
using dataflow::detail::invalid;
using dataflow::detail::readClosedTag;
using dataflow::detail::readCount;
using dataflow::detail::readDomain;
using dataflow::detail::Reader;
using dataflow::detail::ScalarSummary;
using dataflow::detail::TypeSummary;
using dataflow::detail::validateFunctionType;
using dataflow::detail::validateType;
using dataflow::detail::writeDomain;
using dataflow::detail::writeMappedTag;
using dataflow::detail::Writer;

namespace {

constexpr char kSchemaDomain[] = "loom.dataflow.operation-schema-id\0";
constexpr char kSemanticsDomain[] = "loom.dataflow.operation-semantics-case\0";
constexpr char kProjectionDomain[] = "loom.dataflow.actor-schema-projection\0";

static_assert(arith::getMaxEnumValForCmpFPredicate() == 15);
static_assert(arith::getMaxEnumValForRoundingMode() == 4);
static_assert(dataflow::getMaxEnumValForStreamStepKind() == 7);
static_assert(dataflow::getMaxEnumValForAtomicOrdering() == 5);
static_assert(dataflow::getMaxEnumValForSyncScopeKind() == 2);
static_assert(dataflow::getMaxEnumValForVectorAtomicGranularity() == 1);
static_assert(dataflow::getMaxEnumValForAtomicRmwKind() == 22);
static_assert(static_cast<std::uint32_t>(arith::FastMathFlags::fast) == 0x7f);
static_assert(static_cast<std::uint32_t>(arith::IntegerOverflowFlags::nsw |
                                         arith::IntegerOverflowFlags::nuw) ==
              0x3);

enum class ConstantWireTag : std::uint32_t {
  Integer = 1,
  Float = 2,
  DenseInteger = 3,
  DenseFloat = 4,
};

enum class MemoryContractWireTag : std::uint32_t {
  Plain = 1,
  Atomic = 2,
  AtomicRmw = 3,
  CompareExchange = 4,
  Fence = 5,
};

llvm::Expected<std::uint32_t>
schemaWireTag(dataflow::OperationSchemaId schema) {
  switch (schema) {
#define LOOM_OPERATION_SCHEMA(Name, Id, WireTag, OpClass, ActorKind,           \
                              SemanticsCase, SelectorKind, SelectorValue,      \
                              ElementwiseDecomposable)                         \
  case dataflow::OperationSchemaId::Name:                                      \
    return WireTag;
#include "Dataflow/IR/OperationSchemas.inc"
  }
  return invalid("unknown operation schema");
}

llvm::Expected<dataflow::OperationSchemaId>
schemaFromWireTag(std::uint32_t wireTag) {
  switch (wireTag) {
#define LOOM_OPERATION_SCHEMA(Name, Id, WireTag, OpClass, ActorKind,           \
                              SemanticsCase, SelectorKind, SelectorValue,      \
                              ElementwiseDecomposable)                         \
  case WireTag:                                                                \
    return dataflow::OperationSchemaId::Name;
#include "Dataflow/IR/OperationSchemas.inc"
  default:
    return invalid("unknown operation schema wire tag");
  }
}

llvm::Expected<std::uint32_t>
semanticsWireTag(dataflow::OperationSemanticsCase semanticCase) {
  switch (semanticCase) {
#define LOOM_OPERATION_SEMANTICS_CASE(Name, Id, WireTag)                       \
  case dataflow::OperationSemanticsCase::Name:                                 \
    return WireTag;
#include "Dataflow/IR/OperationSchemas.inc"
  }
  return invalid("unknown operation semantics case");
}

llvm::Expected<dataflow::OperationSemanticsCase>
semanticsFromWireTag(std::uint32_t wireTag) {
  switch (wireTag) {
#define LOOM_OPERATION_SEMANTICS_CASE(Name, Id, WireTag)                       \
  case WireTag:                                                                \
    return dataflow::OperationSemanticsCase::Name;
#include "Dataflow/IR/OperationSchemas.inc"
  default:
    return invalid("unknown operation semantics wire tag");
  }
}

llvm::Expected<std::uint32_t> unknownEnum(llvm::StringRef name) {
  return invalid(llvm::Twine("unknown ") + name);
}

llvm::Expected<std::uint32_t>
floatPredicateWireTag(arith::CmpFPredicate predicate) {
  using Predicate = arith::CmpFPredicate;
  switch (predicate) {
  case Predicate::AlwaysFalse:
    return 1;
  case Predicate::OEQ:
    return 2;
  case Predicate::OGT:
    return 3;
  case Predicate::OGE:
    return 4;
  case Predicate::OLT:
    return 5;
  case Predicate::OLE:
    return 6;
  case Predicate::ONE:
    return 7;
  case Predicate::ORD:
    return 8;
  case Predicate::UEQ:
    return 9;
  case Predicate::UGT:
    return 10;
  case Predicate::UGE:
    return 11;
  case Predicate::ULT:
    return 12;
  case Predicate::ULE:
    return 13;
  case Predicate::UNE:
    return 14;
  case Predicate::UNO:
    return 15;
  case Predicate::AlwaysTrue:
    return 16;
  }
  return unknownEnum("floating predicate");
}

llvm::Expected<std::uint32_t> roundingModeWireTag(arith::RoundingMode mode) {
  using Mode = arith::RoundingMode;
  switch (mode) {
  case Mode::to_nearest_even:
    return 1;
  case Mode::downward:
    return 2;
  case Mode::upward:
    return 3;
  case Mode::toward_zero:
    return 4;
  case Mode::to_nearest_away:
    return 5;
  }
  return unknownEnum("rounding mode");
}

llvm::Expected<std::uint32_t> streamStepWireTag(dataflow::StreamStepKind kind) {
  using Kind = dataflow::StreamStepKind;
  switch (kind) {
  case Kind::Add:
    return 1;
  case Kind::Sub:
    return 2;
  case Kind::Mul:
    return 3;
  case Kind::SDiv:
    return 4;
  case Kind::UDiv:
    return 5;
  case Kind::ShL:
    return 6;
  case Kind::AShr:
    return 7;
  case Kind::LShr:
    return 8;
  }
  return unknownEnum("stream step kind");
}

std::uint32_t fastMathWireBits(arith::FastMathFlags flags, bool &valid) {
  using Flags = arith::FastMathFlags;
  valid = dataflow::isValidFastMathFlags(flags);
  std::uint32_t bits = 0;
  if (arith::bitEnumContainsAny(flags, Flags::reassoc))
    bits |= 1u << 0;
  if (arith::bitEnumContainsAny(flags, Flags::nnan))
    bits |= 1u << 1;
  if (arith::bitEnumContainsAny(flags, Flags::ninf))
    bits |= 1u << 2;
  if (arith::bitEnumContainsAny(flags, Flags::nsz))
    bits |= 1u << 3;
  if (arith::bitEnumContainsAny(flags, Flags::arcp))
    bits |= 1u << 4;
  if (arith::bitEnumContainsAny(flags, Flags::contract))
    bits |= 1u << 5;
  if (arith::bitEnumContainsAny(flags, Flags::afn))
    bits |= 1u << 6;
  return bits;
}

std::uint32_t overflowWireBits(arith::IntegerOverflowFlags flags, bool &valid) {
  using Flags = arith::IntegerOverflowFlags;
  const Flags all = Flags::nsw | Flags::nuw;
  valid = (flags | all) == all;
  std::uint32_t bits = 0;
  if (arith::bitEnumContainsAny(flags, Flags::nsw))
    bits |= 1u << 0;
  if (arith::bitEnumContainsAny(flags, Flags::nuw))
    bits |= 1u << 1;
  return bits;
}

void encodeAPInt(Writer &writer, const llvm::APInt &value) {
  const unsigned width = value.getBitWidth();
  writer.u32(width);
  const std::uint64_t bytes = (static_cast<std::uint64_t>(width) + 7) / 8;
  for (std::uint64_t index = bytes; index != 0; --index) {
    const unsigned bit = static_cast<unsigned>((index - 1) * 8);
    const unsigned count = std::min(8u, width - bit);
    writer.byte(
        static_cast<std::uint8_t>(value.extractBitsAsZExtValue(count, bit)));
  }
}

llvm::Error validateAPInt(Reader &reader, std::optional<std::uint32_t> width,
                          llvm::StringRef what) {
  auto actualWidth = reader.u32(llvm::Twine(what) + " bit width");
  if (!actualWidth)
    return actualWidth.takeError();
  if (*actualWidth == 0)
    return invalid(llvm::Twine(what) + " has zero bit width");
  if (width && *actualWidth != *width)
    return invalid(llvm::Twine(what) + " bit width does not match its type");
  const std::uint64_t byteCount =
      (static_cast<std::uint64_t>(*actualWidth) + 7) / 8;
  auto bytes = reader.take(byteCount, what);
  if (!bytes)
    return bytes.takeError();
  const unsigned unused = static_cast<unsigned>(byteCount * 8 - *actualWidth);
  if (unused != 0 && ((*bytes)[0] >> (8 - unused)) != 0)
    return invalid(llvm::Twine(what) + " has nonzero unused high bits");
  return llvm::Error::success();
}

llvm::Error encodeConstant(Writer &writer,
                           dataflow::ConstantValuePayload constant) {
  if (!constant.value)
    return invalid("constant projection contains no typed value");
  if (auto integer = llvm::dyn_cast<IntegerAttr>(constant.value)) {
    writer.u32(static_cast<std::uint32_t>(ConstantWireTag::Integer));
    if (llvm::Error error = encodeType(writer, integer.getType(), 0))
      return error;
    encodeAPInt(writer, integer.getValue());
    return llvm::Error::success();
  }
  if (auto floating = llvm::dyn_cast<FloatAttr>(constant.value)) {
    writer.u32(static_cast<std::uint32_t>(ConstantWireTag::Float));
    if (llvm::Error error = encodeType(writer, floating.getType(), 0))
      return error;
    encodeAPInt(writer, floating.getValue().bitcastToAPInt());
    return llvm::Error::success();
  }
  if (auto integers = llvm::dyn_cast<DenseIntElementsAttr>(constant.value)) {
    writer.u32(static_cast<std::uint32_t>(ConstantWireTag::DenseInteger));
    if (llvm::Error error = encodeType(writer, integers.getType(), 0))
      return error;
    writer.u64(integers.getNumElements());
    for (const llvm::APInt &value : integers)
      encodeAPInt(writer, value);
    return llvm::Error::success();
  }
  if (auto floats = llvm::dyn_cast<DenseFPElementsAttr>(constant.value)) {
    writer.u32(static_cast<std::uint32_t>(ConstantWireTag::DenseFloat));
    if (llvm::Error error = encodeType(writer, floats.getType(), 0))
      return error;
    writer.u64(floats.getNumElements());
    for (const llvm::APFloat &value : floats)
      encodeAPInt(writer, value.bitcastToAPInt());
    return llvm::Error::success();
  }
  return invalid("constant value is outside the closed typed codec");
}

llvm::Error validateConstant(Reader &reader) {
  auto rawTag = reader.u32("constant value tag");
  if (!rawTag)
    return rawTag.takeError();
  auto type = validateType(reader, 0);
  if (!type)
    return type.takeError();
  switch (static_cast<ConstantWireTag>(*rawTag)) {
  case ConstantWireTag::Integer: {
    if (type->shaped || (type->scalar != ScalarSummary::Integer &&
                         type->scalar != ScalarSummary::Index))
      return invalid("integer constant carries a non-integer type");
    std::optional<std::uint32_t> width;
    if (type->scalar == ScalarSummary::Integer)
      width = type->bitWidth;
    return validateAPInt(reader, width, "integer constant");
  }
  case ConstantWireTag::Float:
    if (type->shaped || type->scalar != ScalarSummary::Float)
      return invalid("floating constant carries a non-floating type");
    return validateAPInt(reader, type->bitWidth, "floating constant");
  case ConstantWireTag::DenseInteger:
  case ConstantWireTag::DenseFloat: {
    const bool integer =
        *rawTag == static_cast<std::uint32_t>(ConstantWireTag::DenseInteger);
    const bool expectedElement =
        integer ? (type->scalar == ScalarSummary::Integer ||
                   type->scalar == ScalarSummary::Index)
                : type->scalar == ScalarSummary::Float;
    if (!type->shaped || !expectedElement)
      return invalid("dense constant element kind does not match its tag");
    auto count = reader.u64("dense constant element count");
    if (!count)
      return count.takeError();
    if (*count != type->elementCount)
      return invalid("dense constant element count does not match its type");
    const std::optional<std::uint32_t> width =
        type->scalar == ScalarSummary::Index
            ? std::nullopt
            : std::optional<std::uint32_t>(type->bitWidth);
    for (std::uint64_t index = 0; index < *count; ++index)
      if (llvm::Error error = validateAPInt(reader, width,
                                            integer ? "dense integer element"
                                                    : "dense floating element"))
        return error;
    return llvm::Error::success();
  }
  }
  return invalid("unknown constant value tag");
}

llvm::Error encodeScope(Writer &writer,
                        const dataflow::SyncScopeProjection &scope) {
  if (llvm::Error error = writeMappedTag(
          writer, dataflow::detail::syncScopeKindWireTag(scope.kind)))
    return error;
  if (scope.kind == dataflow::SyncScopeKind::Target) {
    if (!scope.targetNamespace || !scope.targetKey ||
        scope.targetNamespace.getValue().empty() ||
        scope.targetKey.getValue().empty())
      return invalid("target sync scope requires namespace and key");
    writer.string(scope.targetNamespace.getValue());
    writer.string(scope.targetKey.getValue());
  } else if (scope.targetNamespace || scope.targetKey) {
    return invalid("non-target sync scope carries target identity");
  }
  return llvm::Error::success();
}

llvm::Error validateScope(Reader &reader) {
  auto rawKind = reader.u32("sync scope kind tag");
  if (!rawKind)
    return rawKind.takeError();
  auto kind = dataflow::detail::syncScopeKindFromWireTag(*rawKind);
  if (!kind)
    return kind.takeError();
  if (*kind != dataflow::SyncScopeKind::Target)
    return llvm::Error::success();
  auto targetNamespace = reader.string("sync scope target namespace");
  if (!targetNamespace)
    return targetNamespace.takeError();
  auto targetKey = reader.string("sync scope target key");
  if (!targetKey)
    return targetKey.takeError();
  if (targetNamespace->empty() || targetKey->empty())
    return invalid("target sync scope requires namespace and key");
  return llvm::Error::success();
}

llvm::Error encodeOptionalGranularity(
    Writer &writer,
    std::optional<dataflow::VectorAtomicGranularity> granularity) {
  writer.boolean(granularity.has_value());
  if (!granularity)
    return llvm::Error::success();
  return writeMappedTag(
      writer, dataflow::detail::vectorAtomicGranularityWireTag(*granularity));
}

llvm::Error validateOptionalGranularity(Reader &reader) {
  auto present = reader.boolean("vector granularity presence");
  if (!present)
    return present.takeError();
  if (!*present)
    return llvm::Error::success();
  auto rawGranularity = reader.u32("vector atomic granularity tag");
  if (!rawGranularity)
    return rawGranularity.takeError();
  auto granularity =
      dataflow::detail::vectorAtomicGranularityFromWireTag(*rawGranularity);
  if (!granularity)
    return granularity.takeError();
  return llvm::Error::success();
}

llvm::Error encodeAlignment(Writer &writer, std::uint64_t alignment) {
  if (!llvm::isPowerOf2_64(alignment))
    return invalid("source_alignment_bytes is not a nonzero power of two");
  writer.u64(alignment);
  return llvm::Error::success();
}

llvm::Error validateAlignment(Reader &reader) {
  auto alignment = reader.u64("source_alignment_bytes");
  if (!alignment)
    return alignment.takeError();
  if (!llvm::isPowerOf2_64(*alignment))
    return invalid("source_alignment_bytes is not a nonzero power of two");
  return llvm::Error::success();
}

llvm::Error encodeAtomicAccess(Writer &writer,
                               const dataflow::AtomicAccessProjection &access) {
  if (llvm::Error error = writeMappedTag(
          writer, dataflow::detail::atomicOrderingWireTag(access.ordering)))
    return error;
  if (llvm::Error error = encodeScope(writer, access.scope))
    return error;
  if (llvm::Error error = encodeAlignment(writer, access.sourceAlignmentBytes))
    return error;
  if (llvm::Error error =
          encodeOptionalGranularity(writer, access.vectorGranularity))
    return error;
  writer.boolean(access.isVolatile);
  return llvm::Error::success();
}

llvm::Error validateAtomicAccess(Reader &reader) {
  auto rawOrdering = reader.u32("atomic ordering tag");
  if (!rawOrdering)
    return rawOrdering.takeError();
  auto ordering = dataflow::detail::atomicOrderingFromWireTag(*rawOrdering);
  if (!ordering)
    return ordering.takeError();
  if (llvm::Error error = validateScope(reader))
    return error;
  if (llvm::Error error = validateAlignment(reader))
    return error;
  if (llvm::Error error = validateOptionalGranularity(reader))
    return error;
  auto isVolatile = reader.boolean("atomic volatile flag");
  if (!isVolatile)
    return isVolatile.takeError();
  return llvm::Error::success();
}

llvm::Error
encodeMemoryContract(Writer &writer,
                     const dataflow::MemoryContractPayload &contract) {
  if (const auto *plain =
          std::get_if<dataflow::PlainAccessProjection>(&contract)) {
    writer.u32(static_cast<std::uint32_t>(MemoryContractWireTag::Plain));
    writer.boolean(plain->isVolatile);
    return llvm::Error::success();
  }
  if (const auto *atomic =
          std::get_if<dataflow::AtomicAccessProjection>(&contract)) {
    writer.u32(static_cast<std::uint32_t>(MemoryContractWireTag::Atomic));
    return encodeAtomicAccess(writer, *atomic);
  }
  if (const auto *rmw = std::get_if<dataflow::AtomicRmwProjection>(&contract)) {
    writer.u32(static_cast<std::uint32_t>(MemoryContractWireTag::AtomicRmw));
    if (llvm::Error error = writeMappedTag(
            writer, dataflow::detail::atomicRmwKindWireTag(rmw->kind)))
      return error;
    return encodeAtomicAccess(writer, rmw->access);
  }
  if (const auto *exchange =
          std::get_if<dataflow::CompareExchangeProjection>(&contract)) {
    writer.u32(
        static_cast<std::uint32_t>(MemoryContractWireTag::CompareExchange));
    if (llvm::Error error = writeMappedTag(
            writer,
            dataflow::detail::atomicOrderingWireTag(exchange->successOrdering)))
      return error;
    if (llvm::Error error = writeMappedTag(
            writer,
            dataflow::detail::atomicOrderingWireTag(exchange->failureOrdering)))
      return error;
    if (llvm::Error error = encodeScope(writer, exchange->scope))
      return error;
    if (llvm::Error error =
            encodeAlignment(writer, exchange->sourceAlignmentBytes))
      return error;
    if (llvm::Error error =
            encodeOptionalGranularity(writer, exchange->vectorGranularity))
      return error;
    writer.boolean(exchange->weak);
    writer.boolean(exchange->isVolatile);
    return llvm::Error::success();
  }
  const auto *fence = std::get_if<dataflow::FenceProjection>(&contract);
  if (!fence)
    return invalid("unknown memory contract projection");
  writer.u32(static_cast<std::uint32_t>(MemoryContractWireTag::Fence));
  if (llvm::Error error = writeMappedTag(
          writer, dataflow::detail::atomicOrderingWireTag(fence->ordering)))
    return error;
  return encodeScope(writer, fence->scope);
}

llvm::Error validateMemoryContract(Reader &reader) {
  auto rawTag = reader.u32("memory contract tag");
  if (!rawTag)
    return rawTag.takeError();
  switch (static_cast<MemoryContractWireTag>(*rawTag)) {
  case MemoryContractWireTag::Plain: {
    auto isVolatile = reader.boolean("plain volatile flag");
    if (!isVolatile)
      return isVolatile.takeError();
    return llvm::Error::success();
  }
  case MemoryContractWireTag::Atomic:
    return validateAtomicAccess(reader);
  case MemoryContractWireTag::AtomicRmw: {
    auto rawKind = reader.u32("atomic RMW kind tag");
    if (!rawKind)
      return rawKind.takeError();
    auto kind = dataflow::detail::atomicRmwKindFromWireTag(*rawKind);
    if (!kind)
      return kind.takeError();
    return validateAtomicAccess(reader);
  }
  case MemoryContractWireTag::CompareExchange: {
    auto rawSuccess = reader.u32("success ordering tag");
    if (!rawSuccess)
      return rawSuccess.takeError();
    auto success = dataflow::detail::atomicOrderingFromWireTag(*rawSuccess);
    if (!success)
      return success.takeError();
    auto rawFailure = reader.u32("failure ordering tag");
    if (!rawFailure)
      return rawFailure.takeError();
    auto failure = dataflow::detail::atomicOrderingFromWireTag(*rawFailure);
    if (!failure)
      return failure.takeError();
    if (llvm::Error error = validateScope(reader))
      return error;
    if (llvm::Error error = validateAlignment(reader))
      return error;
    if (llvm::Error error = validateOptionalGranularity(reader))
      return error;
    auto weak = reader.boolean("compare-exchange weak flag");
    if (!weak)
      return weak.takeError();
    auto isVolatile = reader.boolean("compare-exchange volatile flag");
    if (!isVolatile)
      return isVolatile.takeError();
    return llvm::Error::success();
  }
  case MemoryContractWireTag::Fence: {
    auto rawOrdering = reader.u32("fence ordering tag");
    if (!rawOrdering)
      return rawOrdering.takeError();
    auto ordering = dataflow::detail::atomicOrderingFromWireTag(*rawOrdering);
    if (!ordering)
      return ordering.takeError();
    return validateScope(reader);
  }
  }
  return invalid("unknown memory contract tag");
}

llvm::Error encodeSignedList(Writer &writer,
                             llvm::ArrayRef<std::int64_t> values,
                             std::int64_t minimum, llvm::StringRef what) {
  writer.u64(values.size());
  for (std::int64_t value : values) {
    if (value < minimum)
      return invalid(llvm::Twine(what) + " contains an invalid value");
    writer.u64(static_cast<std::uint64_t>(value));
  }
  return llvm::Error::success();
}

llvm::Error validateSignedList(Reader &reader, std::int64_t minimum,
                               llvm::StringRef what) {
  auto count = readCount(reader, llvm::Twine(what) + " count", 8);
  if (!count)
    return count.takeError();
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto raw = reader.u64(what);
    if (!raw)
      return raw.takeError();
    const bool belowMinimum =
        *raw <= static_cast<std::uint64_t>(
                    std::numeric_limits<std::int64_t>::max())
            ? static_cast<std::int64_t>(*raw) < minimum
            : minimum >= 0 || *raw < static_cast<std::uint64_t>(minimum);
    if (belowMinimum)
      return invalid(llvm::Twine(what) + " contains an invalid value");
  }
  return llvm::Error::success();
}

bool isValidVectorPosition(std::int64_t value) {
  return value == mlir::ShapedType::kDynamic || value >= 0;
}

bool isValidVectorPositionWire(std::uint64_t value) {
  return value == static_cast<std::uint64_t>(mlir::ShapedType::kDynamic) ||
         value <= static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max());
}

llvm::Error encodeVectorPositionList(Writer &writer,
                                     llvm::ArrayRef<std::int64_t> values) {
  writer.u64(values.size());
  for (std::int64_t value : values) {
    if (!isValidVectorPosition(value))
      return invalid("vector static position contains an invalid value");
    writer.u64(static_cast<std::uint64_t>(value));
  }
  return llvm::Error::success();
}

llvm::Error validateVectorPositionList(Reader &reader) {
  auto count = readCount(reader, "vector static position count", 8);
  if (!count)
    return count.takeError();
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto raw = reader.u64("vector static position");
    if (!raw)
      return raw.takeError();
    if (!isValidVectorPositionWire(*raw))
      return invalid("vector static position contains an invalid value");
  }
  return llvm::Error::success();
}

llvm::Error encodeI32List(Writer &writer, llvm::ArrayRef<std::int32_t> values,
                          llvm::StringRef what) {
  writer.u64(values.size());
  for (std::int32_t value : values)
    writer.u64(static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
  return llvm::Error::success();
}

llvm::Error validateI32List(Reader &reader, llvm::StringRef what) {
  auto count = readCount(reader, llvm::Twine(what) + " count", 8);
  if (!count)
    return count.takeError();
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto raw = reader.u64(what);
    if (!raw)
      return raw.takeError();
    const std::int64_t value = static_cast<std::int64_t>(*raw);
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max())
      return invalid(llvm::Twine(what) + " contains an invalid i32 value");
  }
  return llvm::Error::success();
}

llvm::Error encodePayload(Writer &writer,
                          dataflow::OperationSemanticsCase semanticCase,
                          const dataflow::SemanticPayload &payload) {
  using Case = dataflow::OperationSemanticsCase;
  switch (semanticCase) {
  case Case::NoSemanticPayload:
  case Case::LLVMRegisteredIntrinsic:
    if (!std::holds_alternative<dataflow::NoPayload>(payload))
      break;
    return llvm::Error::success();
  case Case::LLVMGetElementPtrSemantics: {
    const auto *gep = std::get_if<dataflow::GetElementPtrPayload>(&payload);
    if (!gep)
      break;
    if (llvm::Error error = encodeType(writer, gep->sourceElementType, 0))
      return error;
    if (llvm::Error error =
            encodeI32List(writer, gep->rawConstantIndices, "GEP index"))
      return error;
    const std::uint32_t flags = static_cast<std::uint32_t>(gep->noWrapFlags);
    if ((flags & ~std::uint32_t{0x7}) != 0)
      return invalid("GEP payload has unknown no-wrap flag bits");
    writer.u32(flags);
    return llvm::Error::success();
  }
  case Case::ArithFloatingPoint: {
    const auto *floating =
        std::get_if<dataflow::FloatingPointPayload>(&payload);
    if (!floating)
      break;
    bool valid = false;
    const std::uint32_t flags = fastMathWireBits(floating->flags, valid);
    if (!valid)
      return invalid("floating payload has unknown fast-math flags");
    writer.u32(flags);
    writer.boolean(floating->roundingMode.has_value());
    if (!floating->roundingMode)
      return llvm::Error::success();
    return writeMappedTag(writer, roundingModeWireTag(*floating->roundingMode));
  }
  case Case::SpecialMathAccuracy: {
    const auto *special = std::get_if<dataflow::SpecialMathPayload>(&payload);
    if (!special)
      break;
    bool valid = false;
    const std::uint32_t flags = fastMathWireBits(special->flags, valid);
    if (!valid)
      return invalid("special-math payload has unknown fast-math flags");
    if (llvm::Error error = loom::validateSpecialMathAccuracyContract(
            special->accuracy, (flags & (1u << 6)) != 0))
      return error;
    auto tag = loom::specialMathAccuracyWireTag(special->accuracy);
    if (!tag)
      return tag.takeError();
    writer.u32(flags);
    writer.u32(*tag);
    return llvm::Error::success();
  }
  case Case::ArithIntegerOverflow: {
    const auto *overflow =
        std::get_if<dataflow::IntegerOverflowPayload>(&payload);
    if (!overflow)
      break;
    bool valid = false;
    const std::uint32_t flags = overflowWireBits(overflow->flags, valid);
    if (!valid)
      return invalid("integer payload has unknown overflow flags");
    writer.u32(flags);
    return llvm::Error::success();
  }
  case Case::ArithIntegerCompare: {
    const auto *compare =
        std::get_if<dataflow::IntegerComparePayload>(&payload);
    if (!compare)
      break;
    return writeMappedTag(
        writer, dataflow::detail::integerPredicateWireTag(compare->predicate));
  }
  case Case::ArithFloatCompare: {
    const auto *compare = std::get_if<dataflow::FloatComparePayload>(&payload);
    if (!compare)
      break;
    if (llvm::Error error =
            writeMappedTag(writer, floatPredicateWireTag(compare->predicate)))
      return error;
    bool valid = false;
    const std::uint32_t flags = fastMathWireBits(compare->flags, valid);
    if (!valid)
      return invalid("floating compare has unknown fast-math flags");
    writer.u32(flags);
    return llvm::Error::success();
  }
  case Case::TypedConstantValue: {
    const auto *constant =
        std::get_if<dataflow::ConstantValuePayload>(&payload);
    if (!constant)
      break;
    return encodeConstant(writer, *constant);
  }
  case Case::StreamRecurrence: {
    const auto *stream =
        std::get_if<dataflow::StreamRecurrencePayload>(&payload);
    if (!stream)
      break;
    if (llvm::Error error =
            writeMappedTag(writer, streamStepWireTag(stream->stepKind)))
      return error;
    return writeMappedTag(
        writer, dataflow::detail::integerPredicateWireTag(stream->predicate));
  }
  case Case::MemoryContract: {
    const auto *memory = std::get_if<dataflow::MemoryContractPayload>(&payload);
    if (!memory)
      break;
    return encodeMemoryContract(writer, *memory);
  }
  case Case::LLVMZeroPoison: {
    const auto *poison = std::get_if<dataflow::ZeroPoisonPayload>(&payload);
    if (!poison)
      break;
    writer.boolean(poison->isZeroPoison);
    return llvm::Error::success();
  }
  case Case::LLVMIntegerMinPoison: {
    const auto *poison =
        std::get_if<dataflow::IntegerMinPoisonPayload>(&payload);
    if (!poison)
      break;
    writer.boolean(poison->isIntMinPoison);
    return llvm::Error::success();
  }
  case Case::LLVMDisjoint: {
    const auto *disjoint = std::get_if<dataflow::DisjointPayload>(&payload);
    if (!disjoint)
      break;
    writer.boolean(disjoint->isDisjoint);
    return llvm::Error::success();
  }
  case Case::LLVMAggregatePosition: {
    const auto *position =
        std::get_if<dataflow::AggregatePositionPayload>(&payload);
    if (!position)
      break;
    return encodeSignedList(writer, position->position, 0,
                            "aggregate position");
  }
  case Case::ArithExact: {
    const auto *exact = std::get_if<dataflow::ExactPayload>(&payload);
    if (!exact)
      break;
    writer.boolean(exact->isExact);
    return llvm::Error::success();
  }
  case Case::ArithNonNegative: {
    const auto *nonNegative =
        std::get_if<dataflow::NonNegativePayload>(&payload);
    if (!nonNegative)
      break;
    writer.boolean(nonNegative->isNonNegative);
    return llvm::Error::success();
  }
  case Case::VectorStaticPosition: {
    const auto *position =
        std::get_if<dataflow::VectorStaticPositionPayload>(&payload);
    if (!position)
      break;
    return encodeVectorPositionList(writer, position->position);
  }
  case Case::VectorShuffleMask: {
    const auto *mask =
        std::get_if<dataflow::VectorShuffleMaskPayload>(&payload);
    if (!mask)
      break;
    return encodeSignedList(writer, mask->mask, -1, "vector shuffle mask");
  }
  }
  return invalid("semantic payload does not match operation schema");
}

llvm::Error validatePayload(Reader &reader,
                            dataflow::OperationSemanticsCase semanticCase) {
  using Case = dataflow::OperationSemanticsCase;
  switch (semanticCase) {
  case Case::NoSemanticPayload:
  case Case::LLVMRegisteredIntrinsic:
    return llvm::Error::success();
  case Case::LLVMGetElementPtrSemantics: {
    auto element = validateType(reader, 0);
    if (!element)
      return element.takeError();
    if (llvm::Error error = validateI32List(reader, "GEP index"))
      return error;
    auto flags = reader.u32("GEP no-wrap flags");
    if (!flags)
      return flags.takeError();
    if ((*flags & ~std::uint32_t{0x7}) != 0)
      return invalid("unknown GEP no-wrap flag bits");
    return llvm::Error::success();
  }
  case Case::ArithFloatingPoint: {
    auto flags = reader.u32("fast-math flags");
    if (!flags)
      return flags.takeError();
    if ((*flags & ~std::uint32_t{0x7f}) != 0)
      return invalid("unknown fast-math flag bits");
    auto hasRounding = reader.boolean("rounding mode presence");
    if (!hasRounding)
      return hasRounding.takeError();
    if (!*hasRounding)
      return llvm::Error::success();
    auto rounding = readClosedTag(reader, 5, "rounding mode tag");
    if (!rounding)
      return rounding.takeError();
    return llvm::Error::success();
  }
  case Case::SpecialMathAccuracy: {
    auto flags = reader.u32("special-math fast-math flags");
    if (!flags)
      return flags.takeError();
    if ((*flags & ~std::uint32_t{0x7f}) != 0)
      return invalid("unknown special-math fast-math flag bits");
    auto tag = reader.u32("special-math accuracy tier");
    if (!tag)
      return tag.takeError();
    auto tier = loom::specialMathAccuracyTierFromWireTag(*tag);
    if (!tier)
      return tier.takeError();
    return loom::validateSpecialMathAccuracyContract(*tier,
                                                     (*flags & (1u << 6)) != 0);
  }
  case Case::ArithIntegerOverflow: {
    auto flags = reader.u32("integer overflow flags");
    if (!flags)
      return flags.takeError();
    if ((*flags & ~std::uint32_t{0x3}) != 0)
      return invalid("unknown integer overflow flag bits");
    return llvm::Error::success();
  }
  case Case::ArithIntegerCompare: {
    auto predicate = readClosedTag(reader, 10, "integer predicate tag");
    if (!predicate)
      return predicate.takeError();
    return llvm::Error::success();
  }
  case Case::ArithFloatCompare: {
    auto predicate = readClosedTag(reader, 16, "floating predicate tag");
    if (!predicate)
      return predicate.takeError();
    auto flags = reader.u32("fast-math flags");
    if (!flags)
      return flags.takeError();
    if ((*flags & ~std::uint32_t{0x7f}) != 0)
      return invalid("unknown fast-math flag bits");
    return llvm::Error::success();
  }
  case Case::TypedConstantValue:
    return validateConstant(reader);
  case Case::StreamRecurrence: {
    auto step = readClosedTag(reader, 8, "stream step kind tag");
    if (!step)
      return step.takeError();
    auto predicate = readClosedTag(reader, 10, "integer predicate tag");
    if (!predicate)
      return predicate.takeError();
    return llvm::Error::success();
  }
  case Case::MemoryContract:
    return validateMemoryContract(reader);
  case Case::LLVMZeroPoison: {
    auto poison = reader.boolean("zero poison flag");
    if (!poison)
      return poison.takeError();
    return llvm::Error::success();
  }
  case Case::LLVMIntegerMinPoison: {
    auto poison = reader.boolean("integer-minimum poison flag");
    if (!poison)
      return poison.takeError();
    return llvm::Error::success();
  }
  case Case::LLVMDisjoint: {
    auto disjoint = reader.boolean("disjoint flag");
    if (!disjoint)
      return disjoint.takeError();
    return llvm::Error::success();
  }
  case Case::LLVMAggregatePosition:
    return validateSignedList(reader, 0, "aggregate position");
  case Case::ArithExact: {
    auto exact = reader.boolean("exact flag");
    if (!exact)
      return exact.takeError();
    return llvm::Error::success();
  }
  case Case::ArithNonNegative: {
    auto nonNegative = reader.boolean("non-negative flag");
    if (!nonNegative)
      return nonNegative.takeError();
    return llvm::Error::success();
  }
  case Case::VectorStaticPosition:
    return validateVectorPositionList(reader);
  case Case::VectorShuffleMask:
    return validateSignedList(reader, -1, "vector shuffle mask");
  }
  return invalid("unknown operation semantics case");
}

template <typename Value, std::size_t DomainSize, typename Decode>
llvm::Expected<Value> decodeVocabulary(llvm::ArrayRef<std::uint8_t> bytes,
                                       const char (&domain)[DomainSize],
                                       llvm::StringRef what, Decode decode) {
  Reader reader(bytes);
  if (llvm::Error error = readDomain(reader, domain))
    return std::move(error);
  auto wireTag = reader.u32(llvm::Twine(what) + " wire tag");
  if (!wireTag)
    return wireTag.takeError();
  auto value = decode(*wireTag);
  if (!value)
    return value.takeError();
  if (!reader.empty())
    return invalid("trailing bytes");
  return *value;
}

} // namespace

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeOperationSchemaId(OperationSchemaId schema) {
  auto wireTag = schemaWireTag(schema);
  if (!wireTag)
    return wireTag.takeError();
  Writer writer;
  writeDomain(writer, kSchemaDomain);
  writer.u32(*wireTag);
  return loom::CanonicalSemanticBytes(writer.take());
}

llvm::Expected<dataflow::OperationSchemaId>
dataflow::decodeOperationSchemaId(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<OperationSchemaId>(
      bytes, kSchemaDomain, "operation schema", schemaFromWireTag);
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeOperationSemanticsCase(OperationSemanticsCase semanticCase) {
  auto wireTag = semanticsWireTag(semanticCase);
  if (!wireTag)
    return wireTag.takeError();
  Writer writer;
  writeDomain(writer, kSemanticsDomain);
  writer.u32(*wireTag);
  return loom::CanonicalSemanticBytes(writer.take());
}

llvm::Expected<dataflow::OperationSemanticsCase>
dataflow::decodeOperationSemanticsCase(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<OperationSemanticsCase>(
      bytes, kSemanticsDomain, "operation semantics", semanticsFromWireTag);
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeCanonicalActorSchemaProjection(
    const CanonicalActorSchemaProjection &projection) {
  auto schemaTag = schemaWireTag(projection.schema);
  if (!schemaTag)
    return schemaTag.takeError();
  const OperationSemanticsCase semanticCase = semanticsCase(projection.schema);
  auto semanticTag = semanticsWireTag(semanticCase);
  if (!semanticTag)
    return semanticTag.takeError();

  Writer writer;
  writeDomain(writer, kProjectionDomain,
              dataflow::detail::kActorSchemaProjectionCodecVersion);
  writer.u32(*schemaTag);
  writer.u32(*semanticTag);
  if (llvm::Error error = encodeFunctionType(writer, projection.type))
    return std::move(error);
  if (llvm::Error error =
          encodePayload(writer, semanticCase, projection.payload))
    return std::move(error);
  return loom::CanonicalSemanticBytes(writer.take());
}

llvm::Error dataflow::validateCanonicalActorSchemaProjectionBytes(
    llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  if (llvm::Error error =
          readDomain(reader, kProjectionDomain,
                     dataflow::detail::kActorSchemaProjectionCodecVersion))
    return error;
  auto schemaTag = reader.u32("operation schema wire tag");
  if (!schemaTag)
    return schemaTag.takeError();
  auto schema = schemaFromWireTag(*schemaTag);
  if (!schema)
    return schema.takeError();
  auto semanticTag = reader.u32("operation semantics wire tag");
  if (!semanticTag)
    return semanticTag.takeError();
  auto semanticCase = semanticsFromWireTag(*semanticTag);
  if (!semanticCase)
    return semanticCase.takeError();
  if (*semanticCase != semanticsCase(*schema))
    return invalid("semantic payload does not match operation schema");
  if (llvm::Error error = validateFunctionType(reader))
    return error;
  if (llvm::Error error = validatePayload(reader, *semanticCase))
    return error;
  if (!reader.empty())
    return invalid("trailing bytes");
  return llvm::Error::success();
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::projectRegisteredActorSchemaProjectionBytes(Operation *op) {
  auto projection = projectRegisteredActorSchemaProjection(op);
  if (!projection)
    return projection.takeError();
  return encodeCanonicalActorSchemaProjection(*projection);
}

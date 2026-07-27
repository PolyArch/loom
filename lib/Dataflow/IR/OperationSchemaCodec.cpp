#include "Dataflow/IR/OperationSchemaCodec.h"

#include "OperationSchemaCodecInternal.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

using namespace mlir;

namespace {

constexpr char kSchemaDomain[] = "loom.dataflow.operation-schema-id\0";
constexpr char kSemanticsDomain[] = "loom.dataflow.operation-semantics-case\0";
constexpr char kProjectionDomain[] = "loom.dataflow.actor-schema-projection\0";
constexpr char kCanonicalTypeDomain[] = "loom.dataflow.canonical-type\0";
constexpr std::uint32_t kCodecMajor = 1;
constexpr std::uint32_t kCodecMinor = 0;
constexpr unsigned kMaximumTypeDepth = 64;

static_assert(arith::getMaxEnumValForCmpIPredicate() == 9);
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

enum class TypeWireTag : std::uint32_t {
  None = 1,
  Index = 2,
  Integer = 3,
  Float = 4,
  Vector = 5,
  MemRef = 6,
  Tuple = 7,
  RankedTensor = 8,
  Complex = 9,
  LLVMArray = 10,
  LLVMLiteralStruct = 11,
};

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

llvm::Error invalid(const llvm::Twine &message) {
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
  explicit Reader(llvm::ArrayRef<std::uint8_t> bytes) : remaining_(bytes) {}

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
  bool empty() const { return remaining_.empty(); }

private:
  llvm::ArrayRef<std::uint8_t> remaining_;
};

template <std::size_t Size>
void writeDomain(Writer &writer, const char (&domain)[Size]) {
  writer.bytes(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(domain), Size - 1));
  writer.u32(kCodecMajor);
  writer.u32(kCodecMinor);
}

template <std::size_t Size>
llvm::Error readDomain(Reader &reader, const char (&domain)[Size]) {
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
  if (*major != kCodecMajor || *minor != kCodecMinor)
    return invalid("unsupported version");
  return llvm::Error::success();
}

llvm::Expected<std::uint32_t>
schemaWireTag(dataflow::OperationSchemaId schema) {
  switch (schema) {
#define LOOM_OPERATION_SCHEMA(Name, Id, WireTag, OpClass, ActorKind,           \
                              SemanticsCase)                                   \
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
                              SemanticsCase)                                   \
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
integerPredicateWireTag(arith::CmpIPredicate predicate) {
  using Predicate = arith::CmpIPredicate;
  switch (predicate) {
  case Predicate::eq:
    return 1;
  case Predicate::ne:
    return 2;
  case Predicate::slt:
    return 3;
  case Predicate::sle:
    return 4;
  case Predicate::sgt:
    return 5;
  case Predicate::sge:
    return 6;
  case Predicate::ult:
    return 7;
  case Predicate::ule:
    return 8;
  case Predicate::ugt:
    return 9;
  case Predicate::uge:
    return 10;
  }
  return unknownEnum("integer predicate");
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

llvm::Expected<std::uint32_t> floatFormatWireTag(FloatType type) {
  if (llvm::isa<Float8E5M2Type>(type))
    return 1;
  if (llvm::isa<Float8E4M3Type>(type))
    return 2;
  if (llvm::isa<Float8E4M3FNType>(type))
    return 3;
  if (llvm::isa<Float8E5M2FNUZType>(type))
    return 4;
  if (llvm::isa<Float8E4M3FNUZType>(type))
    return 5;
  if (llvm::isa<Float8E4M3B11FNUZType>(type))
    return 6;
  if (llvm::isa<Float8E3M4Type>(type))
    return 7;
  if (llvm::isa<Float4E2M1FNType>(type))
    return 8;
  if (llvm::isa<Float6E2M3FNType>(type))
    return 9;
  if (llvm::isa<Float6E3M2FNType>(type))
    return 10;
  if (llvm::isa<Float8E8M0FNUType>(type))
    return 11;
  if (llvm::isa<BFloat16Type>(type))
    return 12;
  if (llvm::isa<Float16Type>(type))
    return 13;
  if (llvm::isa<FloatTF32Type>(type))
    return 14;
  if (llvm::isa<Float32Type>(type))
    return 15;
  if (llvm::isa<Float64Type>(type))
    return 16;
  if (llvm::isa<Float80Type>(type))
    return 17;
  if (llvm::isa<Float128Type>(type))
    return 18;
  return invalid("unknown floating type");
}

llvm::Expected<FloatType> floatTypeFromWireTag(std::uint32_t wireTag,
                                                MLIRContext *context) {
  switch (wireTag) {
  case 1:
    return Float8E5M2Type::get(context);
  case 2:
    return Float8E4M3Type::get(context);
  case 3:
    return Float8E4M3FNType::get(context);
  case 4:
    return Float8E5M2FNUZType::get(context);
  case 5:
    return Float8E4M3FNUZType::get(context);
  case 6:
    return Float8E4M3B11FNUZType::get(context);
  case 7:
    return Float8E3M4Type::get(context);
  case 8:
    return Float4E2M1FNType::get(context);
  case 9:
    return Float6E2M3FNType::get(context);
  case 10:
    return Float6E3M2FNType::get(context);
  case 11:
    return Float8E8M0FNUType::get(context);
  case 12:
    return BFloat16Type::get(context);
  case 13:
    return Float16Type::get(context);
  case 14:
    return FloatTF32Type::get(context);
  case 15:
    return Float32Type::get(context);
  case 16:
    return Float64Type::get(context);
  case 17:
    return Float80Type::get(context);
  case 18:
    return Float128Type::get(context);
  default:
    return invalid("unknown floating type tag");
  }
}

std::uint32_t fastMathWireBits(arith::FastMathFlags flags, bool &valid) {
  using Flags = arith::FastMathFlags;
  valid = (flags | Flags::fast) == Flags::fast;
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

llvm::Error writeMappedTag(Writer &writer, llvm::Expected<std::uint32_t> tag) {
  if (!tag)
    return tag.takeError();
  writer.u32(*tag);
  return llvm::Error::success();
}

llvm::Expected<std::uint32_t>
readClosedTag(Reader &reader, std::uint32_t maximum, llvm::StringRef what) {
  auto tag = reader.u32(what);
  if (!tag)
    return tag.takeError();
  if (*tag == 0 || *tag > maximum)
    return invalid(llvm::Twine("unknown ") + what);
  return *tag;
}

llvm::Error encodeType(Writer &writer, Type type, unsigned depth);

llvm::Error encodeTypeList(Writer &writer, TypeRange types, unsigned depth) {
  writer.u64(types.size());
  for (Type type : types)
    if (llvm::Error error = encodeType(writer, type, depth))
      return error;
  return llvm::Error::success();
}

llvm::Error encodeShape(Writer &writer, llvm::ArrayRef<std::int64_t> shape,
                        bool requirePositive) {
  writer.u64(shape.size());
  for (std::int64_t dimension : shape) {
    if (ShapedType::isDynamic(dimension)) {
      if (requirePositive)
        return invalid("fixed vector shape contains a dynamic dimension");
      writer.u64(std::numeric_limits<std::uint64_t>::max());
      continue;
    }
    if (dimension < 0 || (requirePositive && dimension == 0))
      return invalid("type shape contains an invalid dimension");
    writer.u64(static_cast<std::uint64_t>(dimension));
  }
  return llvm::Error::success();
}

llvm::Error encodeType(Writer &writer, Type type, unsigned depth) {
  if (!type)
    return invalid("projection contains a null type");
  if (depth >= kMaximumTypeDepth)
    return invalid("type nesting exceeds the canonical limit");
  if (llvm::isa<NoneType>(type)) {
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::None));
    return llvm::Error::success();
  }
  if (llvm::isa<IndexType>(type)) {
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::Index));
    return llvm::Error::success();
  }
  if (auto integer = llvm::dyn_cast<IntegerType>(type)) {
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::Integer));
    switch (integer.getSignedness()) {
    case IntegerType::Signless:
      writer.u32(1);
      break;
    case IntegerType::Signed:
      writer.u32(2);
      break;
    case IntegerType::Unsigned:
      writer.u32(3);
      break;
    }
    if (integer.getWidth() == 0)
      return invalid("integer type has zero width");
    writer.u32(integer.getWidth());
    return llvm::Error::success();
  }
  if (auto floating = llvm::dyn_cast<FloatType>(type)) {
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::Float));
    return writeMappedTag(writer, floatFormatWireTag(floating));
  }
  if (auto vector = llvm::dyn_cast<VectorType>(type)) {
    if (vector.isScalable() || vector.getRank() == 0)
      return invalid("canonical vector type must be fixed and nonzero-rank");
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::Vector));
    if (llvm::Error error = encodeShape(writer, vector.getShape(), true))
      return error;
    return encodeType(writer, vector.getElementType(), depth + 1);
  }
  if (auto memory = llvm::dyn_cast<MemRefType>(type)) {
    if (!memory.getLayout().isIdentity() || memory.getMemorySpace())
      return invalid("canonical memref type requires identity layout and no "
                     "memory space");
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::MemRef));
    if (llvm::Error error = encodeShape(writer, memory.getShape(), false))
      return error;
    return encodeType(writer, memory.getElementType(), depth + 1);
  }
  if (auto tuple = llvm::dyn_cast<TupleType>(type)) {
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::Tuple));
    return encodeTypeList(writer, tuple.getTypes(), depth + 1);
  }
  if (auto tensor = llvm::dyn_cast<RankedTensorType>(type)) {
    if (tensor.getEncoding())
      return invalid("canonical ranked tensor type has no encoding attribute");
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::RankedTensor));
    if (llvm::Error error = encodeShape(writer, tensor.getShape(), false))
      return error;
    return encodeType(writer, tensor.getElementType(), depth + 1);
  }
  if (auto complex = llvm::dyn_cast<ComplexType>(type)) {
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::Complex));
    return encodeType(writer, complex.getElementType(), depth + 1);
  }
  if (auto array = llvm::dyn_cast<LLVM::LLVMArrayType>(type)) {
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::LLVMArray));
    writer.u64(array.getNumElements());
    return encodeType(writer, array.getElementType(), depth + 1);
  }
  if (auto structure = llvm::dyn_cast<LLVM::LLVMStructType>(type)) {
    if (structure.isOpaque())
      return invalid("canonical LLVM aggregate type must have a body");
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::LLVMLiteralStruct));
    writer.boolean(structure.isPacked());
    return encodeTypeList(writer, structure.getBody(), depth + 1);
  }
  return invalid("type is outside the canonical actor projection codec");
}

enum class ScalarSummary { Other, Index, Integer, Float };

struct TypeSummary {
  ScalarSummary scalar = ScalarSummary::Other;
  std::uint32_t bitWidth = 0;
  std::uint64_t elementCount = 1;
  bool shaped = false;
};

llvm::Expected<TypeSummary> validateType(Reader &reader, unsigned depth);

llvm::Expected<Type> decodeType(Reader &reader, MLIRContext *context,
                                unsigned depth);

llvm::Expected<std::uint64_t>
readCount(Reader &reader, const llvm::Twine &what,
          std::size_t minimumBytes);

llvm::Expected<SmallVector<std::int64_t>>
decodeShape(Reader &reader, bool requirePositive) {
  auto rank = readCount(reader, "type rank", 8);
  if (!rank)
    return rank.takeError();
  if (requirePositive && *rank == 0)
    return invalid("canonical vector type has zero rank");
  SmallVector<std::int64_t> shape;
  shape.reserve(static_cast<std::size_t>(*rank));
  for (std::uint64_t index = 0; index < *rank; ++index) {
    auto dimension = reader.u64("type dimension");
    if (!dimension)
      return dimension.takeError();
    if (*dimension == std::numeric_limits<std::uint64_t>::max()) {
      if (requirePositive)
        return invalid("fixed vector shape contains a dynamic dimension");
      shape.push_back(ShapedType::kDynamic);
      continue;
    }
    if (*dimension >
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      return invalid("type dimension exceeds int64_t");
    if (requirePositive && *dimension == 0)
      return invalid("canonical vector dimension is zero");
    shape.push_back(static_cast<std::int64_t>(*dimension));
  }
  return shape;
}

llvm::Expected<SmallVector<Type>> decodeTypeList(Reader &reader,
                                                 MLIRContext *context,
                                                 unsigned depth) {
  auto count = readCount(reader, "type count", 4);
  if (!count)
    return count.takeError();
  SmallVector<Type> types;
  types.reserve(static_cast<std::size_t>(*count));
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto type = decodeType(reader, context, depth);
    if (!type)
      return type.takeError();
    types.push_back(*type);
  }
  return types;
}

llvm::Expected<Type> decodeType(Reader &reader, MLIRContext *context,
                                unsigned depth) {
  if (depth >= kMaximumTypeDepth)
    return invalid("type nesting exceeds the canonical limit");
  auto rawTag = reader.u32("type tag");
  if (!rawTag)
    return rawTag.takeError();
  switch (static_cast<TypeWireTag>(*rawTag)) {
  case TypeWireTag::None:
    return NoneType::get(context);
  case TypeWireTag::Index:
    return IndexType::get(context);
  case TypeWireTag::Integer: {
    auto signedness = readClosedTag(reader, 3, "integer signedness tag");
    if (!signedness)
      return signedness.takeError();
    auto width = reader.u32("integer width");
    if (!width)
      return width.takeError();
    if (*width == 0)
      return invalid("integer type has zero width");
    using Signedness = IntegerType::SignednessSemantics;
    const Signedness semantics[] = {Signedness::Signless, Signedness::Signed,
                                    Signedness::Unsigned};
    return IntegerType::get(context, *width, semantics[*signedness - 1]);
  }
  case TypeWireTag::Float: {
    auto format = readClosedTag(reader, 18, "floating type tag");
    if (!format)
      return format.takeError();
    return floatTypeFromWireTag(*format, context);
  }
  case TypeWireTag::Vector: {
    auto shape = decodeShape(reader, true);
    if (!shape)
      return shape.takeError();
    auto element = decodeType(reader, context, depth + 1);
    if (!element)
      return element.takeError();
    if (!VectorType::isValidElementType(*element))
      return invalid("vector element type is not valid");
    return VectorType::get(*shape, *element);
  }
  case TypeWireTag::MemRef: {
    auto shape = decodeShape(reader, false);
    if (!shape)
      return shape.takeError();
    auto element = decodeType(reader, context, depth + 1);
    if (!element)
      return element.takeError();
    if (!BaseMemRefType::isValidElementType(*element))
      return invalid("memref element type is not valid");
    return MemRefType::get(*shape, *element);
  }
  case TypeWireTag::Tuple: {
    auto types = decodeTypeList(reader, context, depth + 1);
    if (!types)
      return types.takeError();
    return TupleType::get(context, *types);
  }
  case TypeWireTag::RankedTensor: {
    auto shape = decodeShape(reader, false);
    if (!shape)
      return shape.takeError();
    auto element = decodeType(reader, context, depth + 1);
    if (!element)
      return element.takeError();
    if (!TensorType::isValidElementType(*element))
      return invalid("tensor element type is not valid");
    return RankedTensorType::get(*shape, *element);
  }
  case TypeWireTag::Complex: {
    auto element = decodeType(reader, context, depth + 1);
    if (!element)
      return element.takeError();
    if (!llvm::isa<IntegerType, FloatType>(*element))
      return invalid("complex element type is not scalar integer or float");
    return ComplexType::get(*element);
  }
  case TypeWireTag::LLVMArray: {
    auto count = reader.u64("LLVM array element count");
    if (!count)
      return count.takeError();
    auto element = decodeType(reader, context, depth + 1);
    if (!element)
      return element.takeError();
    return LLVM::LLVMArrayType::get(*element, *count);
  }
  case TypeWireTag::LLVMLiteralStruct: {
    auto packed = reader.boolean("LLVM struct packed flag");
    if (!packed)
      return packed.takeError();
    auto body = decodeTypeList(reader, context, depth + 1);
    if (!body)
      return body.takeError();
    return LLVM::LLVMStructType::getLiteral(context, *body, *packed);
  }
  }
  return invalid("unknown type tag");
}

llvm::Expected<std::uint64_t> readCount(Reader &reader, const llvm::Twine &what,
                                        std::size_t minimumBytes) {
  auto count = reader.u64(what);
  if (!count)
    return count.takeError();
  if (minimumBytes != 0 && *count > reader.remainingSize() / minimumBytes)
    return invalid(llvm::Twine(what) + " cannot fit remaining bytes");
  return *count;
}

llvm::Expected<std::uint64_t> validateShape(Reader &reader,
                                            bool requirePositive) {
  auto rank = readCount(reader, "type rank", 8);
  if (!rank)
    return rank.takeError();
  if (requirePositive && *rank == 0)
    return invalid("canonical vector type has zero rank");
  std::uint64_t elements = 1;
  bool dynamic = false;
  for (std::uint64_t index = 0; index < *rank; ++index) {
    auto dimension = reader.u64("type dimension");
    if (!dimension)
      return dimension.takeError();
    if (*dimension == std::numeric_limits<std::uint64_t>::max()) {
      if (requirePositive)
        return invalid("fixed vector shape contains a dynamic dimension");
      dynamic = true;
      continue;
    }
    if (requirePositive && *dimension == 0)
      return invalid("canonical vector dimension is zero");
    if (!dynamic) {
      if (*dimension == 0) {
        elements = 0;
      } else if (elements != 0) {
        if (elements > std::numeric_limits<std::uint64_t>::max() / *dimension)
          return invalid("type element count overflows uint64_t");
        elements *= *dimension;
      }
    }
  }
  return dynamic ? 0 : elements;
}

llvm::Error validateTypeList(Reader &reader, unsigned depth) {
  auto count = readCount(reader, "type count", 4);
  if (!count)
    return count.takeError();
  for (std::uint64_t index = 0; index < *count; ++index) {
    auto type = validateType(reader, depth);
    if (!type)
      return type.takeError();
  }
  return llvm::Error::success();
}

llvm::Expected<TypeSummary> validateType(Reader &reader, unsigned depth) {
  if (depth >= kMaximumTypeDepth)
    return invalid("type nesting exceeds the canonical limit");
  auto rawTag = reader.u32("type tag");
  if (!rawTag)
    return rawTag.takeError();
  switch (static_cast<TypeWireTag>(*rawTag)) {
  case TypeWireTag::None:
    return TypeSummary{};
  case TypeWireTag::Index:
    return TypeSummary{ScalarSummary::Index, 0, 1, false};
  case TypeWireTag::Integer: {
    auto signedness = readClosedTag(reader, 3, "integer signedness tag");
    if (!signedness)
      return signedness.takeError();
    auto width = reader.u32("integer width");
    if (!width)
      return width.takeError();
    if (*width == 0)
      return invalid("integer type has zero width");
    return TypeSummary{ScalarSummary::Integer, *width, 1, false};
  }
  case TypeWireTag::Float: {
    auto format = readClosedTag(reader, 18, "floating type tag");
    if (!format)
      return format.takeError();
    constexpr std::uint32_t widths[] = {8, 8, 8,  8,  8,  8,  8,  4,  6,
                                        6, 8, 16, 16, 32, 32, 64, 80, 128};
    return TypeSummary{ScalarSummary::Float, widths[*format - 1], 1, false};
  }
  case TypeWireTag::Vector:
  case TypeWireTag::RankedTensor: {
    const bool vector =
        *rawTag == static_cast<std::uint32_t>(TypeWireTag::Vector);
    auto elements = validateShape(reader, vector);
    if (!elements)
      return elements.takeError();
    auto element = validateType(reader, depth + 1);
    if (!element)
      return element.takeError();
    if (element->shaped || element->scalar == ScalarSummary::Other)
      return invalid("shaped constant element type is not scalar");
    return TypeSummary{element->scalar, element->bitWidth, *elements, true};
  }
  case TypeWireTag::MemRef: {
    auto elements = validateShape(reader, false);
    if (!elements)
      return elements.takeError();
    auto element = validateType(reader, depth + 1);
    if (!element)
      return element.takeError();
    return TypeSummary{};
  }
  case TypeWireTag::Tuple:
    if (llvm::Error error = validateTypeList(reader, depth + 1))
      return std::move(error);
    return TypeSummary{};
  case TypeWireTag::Complex: {
    auto element = validateType(reader, depth + 1);
    if (!element)
      return element.takeError();
    if (element->shaped || (element->scalar != ScalarSummary::Integer &&
                            element->scalar != ScalarSummary::Float))
      return invalid("complex element type is not scalar integer or float");
    return TypeSummary{};
  }
  case TypeWireTag::LLVMArray: {
    auto count = reader.u64("LLVM array element count");
    if (!count)
      return count.takeError();
    auto element = validateType(reader, depth + 1);
    if (!element)
      return element.takeError();
    return TypeSummary{};
  }
  case TypeWireTag::LLVMLiteralStruct: {
    auto packed = reader.boolean("LLVM struct packed flag");
    if (!packed)
      return packed.takeError();
    if (llvm::Error error = validateTypeList(reader, depth + 1))
      return std::move(error);
    return TypeSummary{};
  }
  }
  return invalid("unknown type tag");
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

llvm::Error encodePayload(Writer &writer,
                          dataflow::OperationSemanticsCase semanticCase,
                          const dataflow::SemanticPayload &payload) {
  using Case = dataflow::OperationSemanticsCase;
  switch (semanticCase) {
  case Case::NoSemanticPayload:
    if (!std::holds_alternative<dataflow::NoPayload>(payload))
      break;
    return llvm::Error::success();
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
    return writeMappedTag(writer, integerPredicateWireTag(compare->predicate));
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
    return writeMappedTag(writer, integerPredicateWireTag(stream->predicate));
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
    return encodeSignedList(writer, position->position, 0,
                            "vector static position");
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
    return llvm::Error::success();
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
    return validateSignedList(reader, 0, "vector static position");
  case Case::VectorShuffleMask:
    return validateSignedList(reader, -1, "vector shuffle mask");
  }
  return invalid("unknown operation semantics case");
}

llvm::Error encodeFunctionType(Writer &writer, FunctionType type) {
  if (!type)
    return invalid("projection contains no function type");
  if (llvm::Error error = encodeTypeList(writer, type.getInputs(), 0))
    return error;
  return encodeTypeList(writer, type.getResults(), 0);
}

llvm::Error validateFunctionType(Reader &reader) {
  if (llvm::Error error = validateTypeList(reader, 0))
    return error;
  return validateTypeList(reader, 0);
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
dataflow::encodeCanonicalType(mlir::Type type) {
  Writer writer;
  writeDomain(writer, kCanonicalTypeDomain);
  if (llvm::Error error = encodeType(writer, type, 0))
    return std::move(error);
  return loom::CanonicalSemanticBytes(writer.take());
}

llvm::Expected<mlir::Type>
dataflow::decodeCanonicalType(llvm::ArrayRef<std::uint8_t> bytes,
                              mlir::MLIRContext *context) {
  if (!context)
    return invalid("canonical type decode requires an MLIR context");
  Reader reader(bytes);
  if (llvm::Error error = readDomain(reader, kCanonicalTypeDomain))
    return std::move(error);
  auto type = decodeType(reader, context, 0);
  if (!type)
    return type.takeError();
  if (!reader.empty())
    return invalid("trailing bytes");
  auto canonical = encodeCanonicalType(*type);
  if (!canonical)
    return canonical.takeError();
  if (canonical->bytes() != bytes)
    return invalid("canonical type bytes are not in canonical form");
  return *type;
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
  writeDomain(writer, kProjectionDomain);
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
  if (llvm::Error error = readDomain(reader, kProjectionDomain))
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

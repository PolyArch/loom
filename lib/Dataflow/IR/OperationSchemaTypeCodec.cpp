#include "Dataflow/IR/OperationSchemaCodec.h"

#include "OperationSchemaCodecInternal.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

using namespace mlir;
using dataflow::detail::invalid;
using dataflow::detail::readClosedTag;
using dataflow::detail::readCount;
using dataflow::detail::readDomain;
using dataflow::detail::Reader;
using dataflow::detail::ScalarSummary;
using dataflow::detail::TypeSummary;
using dataflow::detail::writeDomain;
using dataflow::detail::writeMappedTag;
using dataflow::detail::Writer;

namespace {

constexpr char kCanonicalTypeDomain[] = "loom.dataflow.canonical-type\0";
constexpr unsigned kMaximumTypeDepth = 64;

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
  LLVMPointer = 12,
};

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

llvm::Error encodeTypeList(Writer &writer, TypeRange types, unsigned depth) {
  writer.u64(types.size());
  for (Type type : types)
    if (llvm::Error error = dataflow::detail::encodeType(writer, type, depth))
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

llvm::Expected<Type> decodeType(Reader &reader, MLIRContext *context,
                                unsigned depth);

llvm::Expected<SmallVector<std::int64_t>> decodeShape(Reader &reader,
                                                      bool requirePositive) {
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

llvm::Expected<SmallVector<Type>>
decodeTypeList(Reader &reader, MLIRContext *context, unsigned depth) {
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
  case TypeWireTag::LLVMPointer: {
    auto addressSpace = reader.u32("LLVM pointer address space");
    if (!addressSpace)
      return addressSpace.takeError();
    return LLVM::LLVMPointerType::get(context, *addressSpace);
  }
  }
  return invalid("unknown type tag");
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
    auto type = dataflow::detail::validateType(reader, depth);
    if (!type)
      return type.takeError();
  }
  return llvm::Error::success();
}

} // namespace

llvm::Error dataflow::detail::encodeType(Writer &writer, Type type,
                                         unsigned depth) {
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
  if (auto pointer = llvm::dyn_cast<LLVM::LLVMPointerType>(type)) {
    writer.u32(static_cast<std::uint32_t>(TypeWireTag::LLVMPointer));
    writer.u32(pointer.getAddressSpace());
    return llvm::Error::success();
  }
  return invalid("type is outside the canonical actor projection codec");
}

llvm::Expected<TypeSummary> dataflow::detail::validateType(Reader &reader,
                                                           unsigned depth) {
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
  case TypeWireTag::LLVMPointer: {
    auto addressSpace = reader.u32("LLVM pointer address space");
    if (!addressSpace)
      return addressSpace.takeError();
    return TypeSummary{};
  }
  }
  return invalid("unknown type tag");
}

llvm::Error dataflow::detail::encodeFunctionType(Writer &writer,
                                                 FunctionType type) {
  if (!type)
    return invalid("projection contains no function type");
  if (llvm::Error error = encodeTypeList(writer, type.getInputs(), 0))
    return error;
  return encodeTypeList(writer, type.getResults(), 0);
}

llvm::Error dataflow::detail::validateFunctionType(Reader &reader) {
  if (llvm::Error error = validateTypeList(reader, 0))
    return error;
  return validateTypeList(reader, 0);
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeCanonicalType(mlir::Type type) {
  Writer writer;
  writeDomain(writer, kCanonicalTypeDomain);
  if (llvm::Error error = detail::encodeType(writer, type, 0))
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

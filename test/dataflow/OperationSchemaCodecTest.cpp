#include "Dataflow/IR/OperationSchemaCodec.h"

#include "Dataflow/IR/DataflowDialect.h"
#include "Dataflow/IR/DataflowServiceSchema.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>

using namespace mlir;
using namespace dataflow;

namespace {

constexpr char kSchemaDomain[] = "loom.dataflow.operation-schema-id\0";
constexpr char kSemanticsDomain[] = "loom.dataflow.operation-semantics-case\0";
constexpr char kProjectionDomain[] = "loom.dataflow.actor-schema-projection\0";
constexpr char kServiceKindDomain[] = "loom.dataflow.service-kind\0";
constexpr char kServiceRoleDomain[] = "loom.dataflow.service-value-role\0";
constexpr char kMemoryAccessFormDomain[] = "loom.dataflow.memory-access-form\0";
constexpr char kMemoryMaskFormDomain[] = "loom.dataflow.memory-mask-form\0";
constexpr char kAtomicOrderingDomain[] = "loom.dataflow.atomic-ordering\0";
constexpr char kAtomicRmwKindDomain[] = "loom.dataflow.atomic-rmw-kind\0";
constexpr char kVectorAtomicGranularityDomain[] =
    "loom.dataflow.vector-atomic-granularity\0";
constexpr char kOptionalVectorAtomicGranularityDomain[] =
    "loom.dataflow.optional-vector-atomic-granularity\0";
constexpr char kSyncScopeRefDomain[] = "loom.dataflow.sync-scope-ref\0";
constexpr char kCanonicalBooleanDomain[] = "loom.dataflow.canonical-boolean\0";

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

void appendString(std::vector<std::uint8_t> &bytes, llvm::StringRef value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.bytes_begin(), value.bytes_end());
}

std::vector<std::uint8_t> expectedVocabularyBytes(llvm::StringRef domain,
                                                  std::uint32_t wireTag) {
  std::vector<std::uint8_t> bytes(domain.bytes_begin(), domain.bytes_end());
  appendU32(bytes, 1);
  appendU32(bytes, 0);
  appendU32(bytes, wireTag);
  return bytes;
}

template <typename T>
bool expectFailure(llvm::Expected<T> value, llvm::StringRef expected) {
  if (value) {
    llvm::errs() << "expected codec rejection containing '" << expected
                 << "'\n";
    return false;
  }
  std::string message = llvm::toString(value.takeError());
  if (llvm::StringRef(message).contains(expected))
    return true;
  llvm::errs() << "codec rejection did not contain '" << expected
               << "': " << message << '\n';
  return false;
}

bool expectValidationFailure(llvm::ArrayRef<std::uint8_t> bytes,
                             llvm::StringRef expected) {
  llvm::Error error = validateCanonicalActorSchemaProjectionBytes(bytes);
  if (!error) {
    llvm::errs() << "expected projection-byte rejection containing '"
                 << expected << "'\n";
    return false;
  }
  std::string message = llvm::toString(std::move(error));
  if (llvm::StringRef(message).contains(expected))
    return true;
  llvm::errs() << "projection-byte rejection did not contain '" << expected
               << "': " << message << '\n';
  return false;
}

bool checkVocabularyCodecs() {
  constexpr OperationSchemaId schemas[] = {
#define LOOM_OPERATION_SCHEMA(Name, Id, WireTag, OpClass, ActorKind,           \
                              SemanticsCase)                                   \
  OperationSchemaId::Name,
#include "Dataflow/IR/OperationSchemas.inc"
  };
  constexpr OperationSemanticsCase semantics[] = {
#define LOOM_OPERATION_SEMANTICS_CASE(Name, Id, WireTag)                       \
  OperationSemanticsCase::Name,
#include "Dataflow/IR/OperationSchemas.inc"
  };

  bool ok = true;
  for (OperationSchemaId schema : schemas) {
    auto bytes = encodeOperationSchemaId(schema);
    if (!bytes) {
      llvm::errs() << llvm::toString(bytes.takeError()) << '\n';
      ok = false;
      continue;
    }
    auto decoded = decodeOperationSchemaId(bytes->bytes());
    if (!decoded || *decoded != schema) {
      if (!decoded)
        llvm::errs() << llvm::toString(decoded.takeError()) << '\n';
      else
        llvm::errs() << "operation schema wire roundtrip changed identity\n";
      ok = false;
    }
  }
  for (OperationSemanticsCase semanticCase : semantics) {
    auto bytes = encodeOperationSemanticsCase(semanticCase);
    if (!bytes) {
      llvm::errs() << llvm::toString(bytes.takeError()) << '\n';
      ok = false;
      continue;
    }
    auto decoded = decodeOperationSemanticsCase(bytes->bytes());
    if (!decoded || *decoded != semanticCase) {
      if (!decoded)
        llvm::errs() << llvm::toString(decoded.takeError()) << '\n';
      else
        llvm::errs() << "semantic case wire roundtrip changed identity\n";
      ok = false;
    }
  }

  auto add = encodeOperationSchemaId(OperationSchemaId::ArithAddI);
  const std::vector<std::uint8_t> expectedAdd = expectedVocabularyBytes(
      llvm::StringRef(kSchemaDomain, sizeof(kSchemaDomain) - 1), 0x4c530202);
  if (!add || add->bytes() != llvm::ArrayRef<std::uint8_t>(expectedAdd)) {
    if (!add)
      llvm::errs() << llvm::toString(add.takeError()) << '\n';
    else
      llvm::errs() << "arith.addi did not keep its explicit stable wire tag\n";
    ok = false;
  }

  auto floating =
      encodeOperationSemanticsCase(OperationSemanticsCase::ArithFloatingPoint);
  const std::vector<std::uint8_t> expectedFloating = expectedVocabularyBytes(
      llvm::StringRef(kSemanticsDomain, sizeof(kSemanticsDomain) - 1),
      0x4c530102);
  if (!floating ||
      floating->bytes() != llvm::ArrayRef<std::uint8_t>(expectedFloating)) {
    if (!floating)
      llvm::errs() << llvm::toString(floating.takeError()) << '\n';
    else
      llvm::errs() << "floating semantics did not keep its stable wire tag\n";
    ok = false;
  }

  std::vector<std::uint8_t> unknown = expectedAdd;
  std::fill(unknown.end() - 4, unknown.end(), 0xff);
  ok &= expectFailure(decodeOperationSchemaId(unknown),
                      "unknown operation schema wire tag");
  std::vector<std::uint8_t> wrongVersion = expectedAdd;
  const std::size_t versionOffset = sizeof(kSchemaDomain) - 1;
  wrongVersion[versionOffset + 3] = 2;
  ok &= expectFailure(decodeOperationSchemaId(wrongVersion),
                      "unsupported version");
  std::vector<std::uint8_t> trailing = expectedAdd;
  trailing.push_back(0);
  ok &= expectFailure(decodeOperationSchemaId(trailing), "trailing bytes");
  ok &= expectFailure(encodeOperationSchemaId(static_cast<OperationSchemaId>(
                          std::numeric_limits<std::uint32_t>::max())),
                      "unknown operation schema");
  return ok;
}

template <typename Value, typename Encoder, typename Decoder>
bool checkOwnedEnumCodec(Value value, Encoder encode, Decoder decode,
                         llvm::StringRef domain, std::uint32_t wireTag,
                         llvm::StringRef label) {
  auto bytes = encode(value);
  if (!bytes) {
    llvm::errs() << llvm::toString(bytes.takeError()) << '\n';
    return false;
  }

  bool ok = true;
  auto decoded = decode(bytes->bytes());
  if (!decoded || *decoded != value) {
    if (!decoded)
      llvm::errs() << llvm::toString(decoded.takeError()) << '\n';
    else
      llvm::errs() << label << " wire roundtrip changed identity\n";
    ok = false;
  }

  const std::vector<std::uint8_t> expected =
      expectedVocabularyBytes(domain, wireTag);
  if (bytes->bytes() != llvm::ArrayRef<std::uint8_t>(expected)) {
    llvm::errs() << label << " did not keep its explicit stable wire tag\n";
    ok = false;
  }

  std::vector<std::uint8_t> unknown = expected;
  std::fill(unknown.end() - 4, unknown.end(), 0xff);
  ok &= expectFailure(decode(unknown), "unknown");
  std::vector<std::uint8_t> trailing = expected;
  trailing.push_back(0);
  ok &= expectFailure(decode(trailing), "trailing bytes");
  return ok;
}

bool checkOwnedAtomCodecs(MLIRContext &context) {
  using semantics::MemoryAccessForm;
  using semantics::MemoryMaskForm;
  using semantics::ServiceKind;
  using semantics::ServiceValueRole;

  bool ok = true;
  ok &= checkOwnedEnumCodec(
      ServiceKind::MemoryCompareExchange,
      [](ServiceKind value) { return encodeServiceKind(value); },
      [](llvm::ArrayRef<std::uint8_t> bytes) {
        return decodeServiceKind(bytes);
      },
      llvm::StringRef(kServiceKindDomain, sizeof(kServiceKindDomain) - 1), 5,
      "service kind");
  ok &= checkOwnedEnumCodec(
      ServiceValueRole::Mask,
      [](ServiceValueRole value) { return encodeServiceValueRole(value); },
      [](llvm::ArrayRef<std::uint8_t> bytes) {
        return decodeServiceValueRole(bytes);
      },
      llvm::StringRef(kServiceRoleDomain, sizeof(kServiceRoleDomain) - 1), 7,
      "service value role");
  ok &= checkOwnedEnumCodec(
      MemoryAccessForm::Indexed,
      [](MemoryAccessForm value) { return encodeMemoryAccessForm(value); },
      [](llvm::ArrayRef<std::uint8_t> bytes) {
        return decodeMemoryAccessForm(bytes);
      },
      llvm::StringRef(kMemoryAccessFormDomain,
                      sizeof(kMemoryAccessFormDomain) - 1),
      3, "memory access form");
  ok &= checkOwnedEnumCodec(
      MemoryMaskForm::Dynamic,
      [](MemoryMaskForm value) { return encodeMemoryMaskForm(value); },
      [](llvm::ArrayRef<std::uint8_t> bytes) {
        return decodeMemoryMaskForm(bytes);
      },
      llvm::StringRef(kMemoryMaskFormDomain, sizeof(kMemoryMaskFormDomain) - 1),
      2, "memory mask form");
  ok &= checkOwnedEnumCodec(
      AtomicOrdering::AcqRel,
      [](AtomicOrdering value) { return encodeAtomicOrdering(value); },
      [](llvm::ArrayRef<std::uint8_t> bytes) {
        return decodeAtomicOrdering(bytes);
      },
      llvm::StringRef(kAtomicOrderingDomain, sizeof(kAtomicOrderingDomain) - 1),
      5, "atomic ordering");
  ok &= checkOwnedEnumCodec(
      AtomicRmwKind::FMinimumNum,
      [](AtomicRmwKind value) { return encodeAtomicRmwKind(value); },
      [](llvm::ArrayRef<std::uint8_t> bytes) {
        return decodeAtomicRmwKind(bytes);
      },
      llvm::StringRef(kAtomicRmwKindDomain, sizeof(kAtomicRmwKindDomain) - 1),
      23, "atomic RMW kind");
  ok &= checkOwnedEnumCodec(
      VectorAtomicGranularity::PerLane,
      [](VectorAtomicGranularity value) {
        return encodeVectorAtomicGranularity(value);
      },
      [](llvm::ArrayRef<std::uint8_t> bytes) {
        return decodeVectorAtomicGranularity(bytes);
      },
      llvm::StringRef(kVectorAtomicGranularityDomain,
                      sizeof(kVectorAtomicGranularityDomain) - 1),
      2, "vector atomic granularity");

  auto roleBytes = encodeServiceValueRole(ServiceValueRole::Mask);
  if (!roleBytes) {
    llvm::errs() << llvm::toString(roleBytes.takeError()) << '\n';
    ok = false;
  } else {
    ok &= expectFailure(decodeMemoryAccessForm(roleBytes->bytes()),
                        "wrong semantic domain");
  }

  Type payload = VectorType::get({4}, Float32Type::get(&context));
  auto payloadBytes = encodeCanonicalType(payload);
  if (!payloadBytes) {
    llvm::errs() << llvm::toString(payloadBytes.takeError()) << '\n';
    ok = false;
  } else {
    auto decoded = decodeCanonicalType(payloadBytes->bytes(), &context);
    if (!decoded || *decoded != payload) {
      if (!decoded)
        llvm::errs() << llvm::toString(decoded.takeError()) << '\n';
      else
        llvm::errs() << "canonical type roundtrip changed vector<4xf32>\n";
      ok = false;
    }
    std::vector<std::uint8_t> trailing(payloadBytes->bytes().begin(),
                                       payloadBytes->bytes().end());
    trailing.push_back(0);
    ok &= expectFailure(decodeCanonicalType(trailing, &context),
                        "trailing bytes");
  }

  Type aggregateBody[] = {IntegerType::get(&context, 32),
                          LLVM::LLVMArrayType::get(
                              IntegerType::get(&context, 16), 4)};
  Type namedAggregateA = LLVM::LLVMStructType::getNewIdentified(
      &context, "private.aggregate.a", aggregateBody, false);
  Type namedAggregateB = LLVM::LLVMStructType::getNewIdentified(
      &context, "private.aggregate.b", aggregateBody, false);
  auto namedBytesA = encodeCanonicalType(namedAggregateA);
  auto namedBytesB = encodeCanonicalType(namedAggregateB);
  if (!namedBytesA || !namedBytesB) {
    if (!namedBytesA)
      llvm::errs() << llvm::toString(namedBytesA.takeError()) << '\n';
    if (!namedBytesB)
      llvm::errs() << llvm::toString(namedBytesB.takeError()) << '\n';
    ok = false;
  } else {
    if (namedBytesA->bytes() != namedBytesB->bytes()) {
      llvm::errs() << "private LLVM aggregate names changed canonical type "
                      "bytes\n";
      ok = false;
    }
    auto decoded = decodeCanonicalType(namedBytesA->bytes(), &context);
    Type literalAggregate =
        LLVM::LLVMStructType::getLiteral(&context, aggregateBody, false);
    if (!decoded || *decoded != literalAggregate) {
      if (!decoded)
        llvm::errs() << llvm::toString(decoded.takeError()) << '\n';
      else
        llvm::errs() << "identified LLVM aggregate did not normalize to its "
                        "literal semantic type\n";
      ok = false;
    }
  }

  for (std::optional<VectorAtomicGranularity> value :
       {std::optional<VectorAtomicGranularity>{},
        std::optional<VectorAtomicGranularity>{
            VectorAtomicGranularity::PerLane}}) {
    auto bytes = encodeOptionalVectorAtomicGranularity(value);
    if (!bytes) {
      llvm::errs() << llvm::toString(bytes.takeError()) << '\n';
      ok = false;
      continue;
    }
    auto decoded = decodeOptionalVectorAtomicGranularity(bytes->bytes());
    if (!decoded || *decoded != value) {
      if (!decoded)
        llvm::errs() << llvm::toString(decoded.takeError()) << '\n';
      else
        llvm::errs() << "optional granularity wire roundtrip changed value\n";
      ok = false;
    }
    std::vector<std::uint8_t> expected = expectedVocabularyBytes(
        llvm::StringRef(kOptionalVectorAtomicGranularityDomain,
                        sizeof(kOptionalVectorAtomicGranularityDomain) - 1),
        value.has_value() ? 1 : 0);
    if (value)
      appendU32(expected, 2);
    if (bytes->bytes() != llvm::ArrayRef<std::uint8_t>(expected)) {
      llvm::errs() << "optional granularity did not keep stable bytes\n";
      ok = false;
    }
  }
  std::vector<std::uint8_t> invalidPresence = expectedVocabularyBytes(
      llvm::StringRef(kOptionalVectorAtomicGranularityDomain,
                      sizeof(kOptionalVectorAtomicGranularityDomain) - 1),
      2);
  ok &= expectFailure(decodeOptionalVectorAtomicGranularity(invalidPresence),
                      "not a canonical boolean");

  Builder builder(&context);
  SyncScopeProjection scope{SyncScopeKind::Target, builder.getStringAttr("gpu"),
                            builder.getStringAttr("device")};
  auto scopeBytes = encodeSyncScopeRef(scope);
  if (!scopeBytes) {
    llvm::errs() << llvm::toString(scopeBytes.takeError()) << '\n';
    ok = false;
  } else {
    auto decoded = decodeSyncScopeRef(scopeBytes->bytes(), &context);
    const bool matches = decoded && decoded->kind == scope.kind &&
                         decoded->targetNamespace == scope.targetNamespace &&
                         decoded->targetKey == scope.targetKey;
    if (!matches) {
      if (!decoded)
        llvm::errs() << llvm::toString(decoded.takeError()) << '\n';
      else
        llvm::errs() << "sync scope wire roundtrip changed identity\n";
      ok = false;
    }
    std::vector<std::uint8_t> expected = expectedVocabularyBytes(
        llvm::StringRef(kSyncScopeRefDomain, sizeof(kSyncScopeRefDomain) - 1),
        3);
    appendString(expected, "gpu");
    appendString(expected, "device");
    if (scopeBytes->bytes() != llvm::ArrayRef<std::uint8_t>(expected)) {
      llvm::errs() << "sync scope did not keep stable bytes\n";
      ok = false;
    }
    std::vector<std::uint8_t> unknown = expected;
    const std::size_t tagOffset = sizeof(kSyncScopeRefDomain) - 1 + 8;
    std::fill(unknown.begin() + tagOffset, unknown.begin() + tagOffset + 4,
              0xff);
    ok &= expectFailure(decodeSyncScopeRef(unknown, &context), "unknown");
  }

  auto booleanBytes = encodeCanonicalBoolean(true);
  const std::vector<std::uint8_t> expectedBoolean = expectedVocabularyBytes(
      llvm::StringRef(kCanonicalBooleanDomain,
                      sizeof(kCanonicalBooleanDomain) - 1),
      1);
  if (!booleanBytes ||
      booleanBytes->bytes() != llvm::ArrayRef<std::uint8_t>(expectedBoolean)) {
    if (!booleanBytes)
      llvm::errs() << llvm::toString(booleanBytes.takeError()) << '\n';
    else
      llvm::errs() << "canonical boolean did not keep stable bytes\n";
    ok = false;
  } else {
    auto decoded = decodeCanonicalBoolean(booleanBytes->bytes());
    if (!decoded || !*decoded) {
      if (!decoded)
        llvm::errs() << llvm::toString(decoded.takeError()) << '\n';
      else
        llvm::errs() << "canonical boolean wire roundtrip changed value\n";
      ok = false;
    }
  }
  std::vector<std::uint8_t> malformedBoolean = expectedVocabularyBytes(
      llvm::StringRef(kCanonicalBooleanDomain,
                      sizeof(kCanonicalBooleanDomain) - 1),
      2);
  ok &= expectFailure(decodeCanonicalBoolean(malformedBoolean),
                      "not a canonical boolean");
  return ok;
}

CanonicalActorSchemaProjection
makeFloatingProjection(MLIRContext &context, arith::FastMathFlags flags) {
  Builder builder(&context);
  Type f32 = builder.getF32Type();
  return {OperationSchemaId::ArithAddF,
          builder.getFunctionType({f32, f32}, {f32}),
          SemanticPayload{
              FloatingPointPayload{flags, arith::RoundingMode::downward}}};
}

CanonicalActorSchemaProjection
makeAtomicLoadProjection(MLIRContext &context, std::uint64_t alignment) {
  Builder builder(&context);
  Type i32 = builder.getI32Type();
  Type index = builder.getIndexType();
  Type none = builder.getNoneType();
  Type memory = MemRefType::get({ShapedType::kDynamic}, i32);
  Type lanes = VectorType::get({2}, i32);
  Type addresses = VectorType::get({2}, index);
  Type mask = VectorType::get({2}, builder.getI1Type());
  AtomicAccessProjection access{
      AtomicOrdering::Acquire,
      SyncScopeProjection{SyncScopeKind::Target, builder.getStringAttr("gpu"),
                          builder.getStringAttr("device")},
      alignment, VectorAtomicGranularity::PerLane, true};
  return {
      OperationSchemaId::DataflowLoad,
      builder.getFunctionType({memory, addresses, none, mask}, {lanes, none}),
      SemanticPayload{MemoryContractPayload{access}}};
}

bool checkProjectionCodec(MLIRContext &context) {
  bool ok = true;
  CanonicalActorSchemaProjection first =
      makeFloatingProjection(context, arith::FastMathFlags::nnan);
  auto firstBytes = encodeCanonicalActorSchemaProjection(first);
  auto firstAgain = encodeCanonicalActorSchemaProjection(first);
  if (!firstBytes || !firstAgain) {
    if (!firstBytes)
      llvm::errs() << llvm::toString(firstBytes.takeError()) << '\n';
    if (!firstAgain)
      llvm::errs() << llvm::toString(firstAgain.takeError()) << '\n';
    ok = false;
  } else {
    if (firstBytes->bytes() != firstAgain->bytes()) {
      llvm::errs() << "equal typed projections produced different bytes\n";
      ok = false;
    }
    if (llvm::Error error =
            validateCanonicalActorSchemaProjectionBytes(firstBytes->bytes())) {
      llvm::errs() << llvm::toString(std::move(error)) << '\n';
      ok = false;
    }
  }

  CanonicalActorSchemaProjection delta =
      makeFloatingProjection(context, arith::FastMathFlags::nsz);
  auto deltaBytes = encodeCanonicalActorSchemaProjection(delta);
  if (!deltaBytes) {
    llvm::errs() << llvm::toString(deltaBytes.takeError()) << '\n';
    ok = false;
  } else if (firstBytes && firstBytes->bytes() == deltaBytes->bytes()) {
    llvm::errs() << "floating semantic delta did not change projection bytes\n";
    ok = false;
  }

  CanonicalActorSchemaProjection wrongPayload = first;
  wrongPayload.payload = NoPayload{};
  ok &= expectFailure(encodeCanonicalActorSchemaProjection(wrongPayload),
                      "does not match operation schema");

  CanonicalActorSchemaProjection four = makeAtomicLoadProjection(context, 4);
  CanonicalActorSchemaProjection eight = makeAtomicLoadProjection(context, 8);
  auto fourBytes = encodeCanonicalActorSchemaProjection(four);
  auto eightBytes = encodeCanonicalActorSchemaProjection(eight);
  if (!fourBytes || !eightBytes) {
    if (!fourBytes)
      llvm::errs() << llvm::toString(fourBytes.takeError()) << '\n';
    if (!eightBytes)
      llvm::errs() << llvm::toString(eightBytes.takeError()) << '\n';
    ok = false;
  } else if (fourBytes->bytes() == eightBytes->bytes()) {
    llvm::errs() << "source_alignment_bytes did not change projection bytes\n";
    ok = false;
  }

  Builder builder(&context);
  VectorType vector = VectorType::get({2}, builder.getI32Type());
  DenseIntElementsAttr value = DenseIntElementsAttr::get(vector, {1, 2});
  CanonicalActorSchemaProjection constant{
      OperationSchemaId::ArithConstant, builder.getFunctionType({}, {vector}),
      SemanticPayload{ConstantValuePayload{value}}};
  auto constantBytes = encodeCanonicalActorSchemaProjection(constant);
  if (!constantBytes) {
    llvm::errs() << llvm::toString(constantBytes.takeError()) << '\n';
    ok = false;
  }

  RankedTensorType emptyTensor =
      RankedTensorType::get({0}, builder.getI32Type());
  DenseIntElementsAttr emptyValue =
      DenseIntElementsAttr::get(emptyTensor, llvm::ArrayRef<std::int32_t>{});
  CanonicalActorSchemaProjection emptyConstant{
      OperationSchemaId::ArithConstant,
      builder.getFunctionType({}, {emptyTensor}),
      SemanticPayload{ConstantValuePayload{emptyValue}}};
  auto emptyConstantBytes = encodeCanonicalActorSchemaProjection(emptyConstant);
  if (!emptyConstantBytes) {
    llvm::errs() << llvm::toString(emptyConstantBytes.takeError()) << '\n';
    ok = false;
  } else if (llvm::Error error = validateCanonicalActorSchemaProjectionBytes(
                 emptyConstantBytes->bytes())) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    ok = false;
  }

  RankedTensorType indexTensor =
      RankedTensorType::get({2}, builder.getIndexType());
  llvm::APInt indexElements[] = {llvm::APInt(64, 1), llvm::APInt(64, 2)};
  DenseIntElementsAttr indexValue =
      DenseIntElementsAttr::get(indexTensor, indexElements);
  CanonicalActorSchemaProjection indexConstant{
      OperationSchemaId::ArithConstant,
      builder.getFunctionType({}, {indexTensor}),
      SemanticPayload{ConstantValuePayload{indexValue}}};
  auto indexConstantBytes = encodeCanonicalActorSchemaProjection(indexConstant);
  if (!indexConstantBytes) {
    llvm::errs() << llvm::toString(indexConstantBytes.takeError()) << '\n';
    ok = false;
  } else if (llvm::Error error = validateCanonicalActorSchemaProjectionBytes(
                 indexConstantBytes->bytes())) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    ok = false;
  }

  Type aggregate = LLVM::LLVMStructType::getLiteral(
      &context, {builder.getI32Type(), builder.getI32Type()});
  CanonicalActorSchemaProjection extract{
      OperationSchemaId::LLVMExtractValue,
      builder.getFunctionType({aggregate}, {builder.getI32Type()}),
      SemanticPayload{AggregatePositionPayload{{1}}}};
  auto extractBytes = encodeCanonicalActorSchemaProjection(extract);
  if (!extractBytes) {
    llvm::errs() << llvm::toString(extractBytes.takeError()) << '\n';
    ok = false;
  }

  if (firstBytes) {
    std::vector<std::uint8_t> truncated(firstBytes->bytes().begin(),
                                        firstBytes->bytes().end() - 1);
    ok &= expectValidationFailure(truncated, "truncated");
    std::vector<std::uint8_t> trailing(firstBytes->bytes().begin(),
                                       firstBytes->bytes().end());
    trailing.push_back(0);
    ok &= expectValidationFailure(trailing, "trailing bytes");

    const std::size_t semanticOffset = sizeof(kProjectionDomain) - 1 + 12;
    std::vector<std::uint8_t> wrongCase(firstBytes->bytes().begin(),
                                        firstBytes->bytes().end());
    wrongCase[semanticOffset + 0] = 0x4c;
    wrongCase[semanticOffset + 1] = 0x53;
    wrongCase[semanticOffset + 2] = 0x01;
    wrongCase[semanticOffset + 3] = 0x01;
    ok &= expectValidationFailure(wrongCase, "does not match operation schema");
  }
  return ok;
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<dataflow::DataflowDialect, LLVM::LLVMDialect>();
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();

  bool ok = true;
  ok &= checkVocabularyCodecs();
  ok &= checkOwnedAtomCodecs(context);
  ok &= checkProjectionCodec(context);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

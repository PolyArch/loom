#include "Dataflow/IR/OperationSchemaCodec.h"

#include "Dataflow/IR/DataflowDialect.h"

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
#include <string>
#include <vector>

using namespace mlir;
using namespace dataflow;

namespace {

constexpr char kSchemaDomain[] = "loom.dataflow.operation-schema-id\0";
constexpr char kSemanticsDomain[] = "loom.dataflow.operation-semantics-case\0";
constexpr char kProjectionDomain[] = "loom.dataflow.actor-schema-projection\0";

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
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
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  bool ok = true;
  ok &= checkVocabularyCodecs();
  ok &= checkProjectionCodec(context);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "Common/SpecialMathAccuracy.h"
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

#include <array>
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
constexpr char kIntegerPredicateDomain[] =
    "loom.dataflow.integer-compare-predicate\0";
constexpr char kFloatPredicateDomain[] =
    "loom.dataflow.float-compare-predicate\0";
constexpr char kRoundingModeDomain[] = "loom.dataflow.rounding-mode\0";
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
constexpr char kSpecialMathAccuracyDomain[] =
    "loom.special-math-accuracy-tier\0";

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
                              SemanticsCase, SelectorKind, SelectorValue,      \
                              ElementwiseDecomposable)                         \
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

bool checkSpecialMathAccuracyCodec() {
  constexpr struct {
    loom::SpecialMathAccuracyTier tier;
    std::uint32_t wireTag;
  } cases[] = {
      {loom::SpecialMathAccuracyTier::CorrectlyRounded, 0x4c534101},
      {loom::SpecialMathAccuracyTier::Max1Ulp, 0x4c534102},
      {loom::SpecialMathAccuracyTier::Max2Ulp, 0x4c534103},
      {loom::SpecialMathAccuracyTier::Max4Ulp, 0x4c534104},
  };

  bool ok = true;
  for (const auto &testCase : cases) {
    auto encoded = loom::encodeSpecialMathAccuracyTier(testCase.tier);
    const std::vector<std::uint8_t> expected = expectedVocabularyBytes(
        llvm::StringRef(kSpecialMathAccuracyDomain,
                        sizeof(kSpecialMathAccuracyDomain) - 1),
        testCase.wireTag);
    if (!encoded || encoded->bytes() != llvm::ArrayRef(expected)) {
      if (!encoded)
        llvm::errs() << llvm::toString(encoded.takeError()) << '\n';
      else
        llvm::errs() << "special-math tier did not keep stable bytes\n";
      ok = false;
      continue;
    }
    auto decoded = loom::decodeSpecialMathAccuracyTier(expected);
    if (!decoded || *decoded != testCase.tier) {
      if (!decoded)
        llvm::errs() << llvm::toString(decoded.takeError()) << '\n';
      else
        llvm::errs() << "special-math tier wire roundtrip changed value\n";
      ok = false;
    }
  }

  for (std::size_t guarantee = 0; guarantee != std::size(cases); ++guarantee) {
    for (std::size_t accepted = 0; accepted != std::size(cases); ++accepted) {
      auto refines = loom::specialMathAccuracyRefines(cases[guarantee].tier,
                                                      cases[accepted].tier);
      if (!refines || *refines != (guarantee <= accepted)) {
        if (!refines)
          llvm::errs() << llvm::toString(refines.takeError()) << '\n';
        else
          llvm::errs() << "special-math refinement order is incorrect\n";
        ok = false;
      }
    }
  }
  for (std::size_t tier = 0; tier != std::size(cases); ++tier) {
    for (bool approximationPermitted : {false, true}) {
      llvm::Error validation = loom::validateSpecialMathAccuracyContract(
          cases[tier].tier, approximationPermitted);
      const bool shouldSucceed = approximationPermitted || tier == 0;
      if (validation) {
        if (shouldSucceed) {
          llvm::errs() << llvm::toString(std::move(validation)) << '\n';
          ok = false;
        } else {
          llvm::consumeError(std::move(validation));
        }
      } else if (!shouldSucceed) {
        llvm::errs() << "relaxed special-math tier was accepted without afn\n";
        ok = false;
      }
    }
  }
  constexpr auto invalidTier = static_cast<loom::SpecialMathAccuracyTier>(0xff);
  ok &= expectFailure(
      loom::specialMathAccuracyRefines(
          loom::SpecialMathAccuracyTier::CorrectlyRounded, invalidTier),
      "unknown special-math accuracy tier");
  ok &=
      expectFailure(loom::specialMathAccuracyRefines(invalidTier, invalidTier),
                    "unknown special-math accuracy tier");

  std::vector<std::uint8_t> unknown = expectedVocabularyBytes(
      llvm::StringRef(kSpecialMathAccuracyDomain,
                      sizeof(kSpecialMathAccuracyDomain) - 1),
      0x4c534105);
  ok &= expectFailure(loom::decodeSpecialMathAccuracyTier(unknown),
                      "unknown special-math accuracy tier");
  unknown.push_back(0);
  ok &= expectFailure(loom::decodeSpecialMathAccuracyTier(unknown),
                      "trailing bytes");
  return ok;
}

template <typename Value, typename Encoder, typename Decoder>
bool checkOwnedEnumMember(Value value, Encoder encode, Decoder decode,
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

  return ok;
}

template <typename Value, typename Encoder, typename Decoder>
bool checkOwnedEnumCodec(Value value, Encoder encode, Decoder decode,
                         llvm::StringRef domain, std::uint32_t wireTag,
                         llvm::StringRef label) {
  bool ok = checkOwnedEnumMember(value, encode, decode, domain, wireTag, label);
  std::vector<std::uint8_t> unknown = expectedVocabularyBytes(domain, wireTag);
  std::fill(unknown.end() - 4, unknown.end(), 0xff);
  ok &= expectFailure(decode(unknown), "unknown");
  std::vector<std::uint8_t> trailing = expectedVocabularyBytes(domain, wireTag);
  trailing.push_back(0);
  ok &= expectFailure(decode(trailing), "trailing bytes");
  return ok;
}

template <typename Value, std::size_t N, typename Encoder, typename Decoder>
bool checkOwnedEnumCodecs(
    const std::array<std::pair<Value, std::uint32_t>, N> &cases, Encoder encode,
    Decoder decode, llvm::StringRef domain, llvm::StringRef label) {
  bool ok = true;
  for (const auto &[value, wireTag] : cases)
    ok &= checkOwnedEnumMember(value, encode, decode, domain, wireTag, label);
  ok &= expectFailure(encode(static_cast<Value>(0xff)), "unknown");
  std::vector<std::uint8_t> unknown =
      expectedVocabularyBytes(domain, cases.front().second);
  std::fill(unknown.end() - 4, unknown.end(), 0xff);
  ok &= expectFailure(decode(unknown), "unknown");
  std::vector<std::uint8_t> trailing =
      expectedVocabularyBytes(domain, cases.front().second);
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
      arith::CmpIPredicate::sge,
      [](arith::CmpIPredicate value) {
        return encodeIntegerComparePredicate(value);
      },
      [](llvm::ArrayRef<std::uint8_t> bytes) {
        return decodeIntegerComparePredicate(bytes);
      },
      llvm::StringRef(kIntegerPredicateDomain,
                      sizeof(kIntegerPredicateDomain) - 1),
      6, "integer compare predicate");
  constexpr std::array floatPredicates = {
      std::pair{arith::CmpFPredicate::AlwaysFalse, 1U},
      std::pair{arith::CmpFPredicate::OEQ, 2U},
      std::pair{arith::CmpFPredicate::OGT, 3U},
      std::pair{arith::CmpFPredicate::OGE, 4U},
      std::pair{arith::CmpFPredicate::OLT, 5U},
      std::pair{arith::CmpFPredicate::OLE, 6U},
      std::pair{arith::CmpFPredicate::ONE, 7U},
      std::pair{arith::CmpFPredicate::ORD, 8U},
      std::pair{arith::CmpFPredicate::UEQ, 9U},
      std::pair{arith::CmpFPredicate::UGT, 10U},
      std::pair{arith::CmpFPredicate::UGE, 11U},
      std::pair{arith::CmpFPredicate::ULT, 12U},
      std::pair{arith::CmpFPredicate::ULE, 13U},
      std::pair{arith::CmpFPredicate::UNE, 14U},
      std::pair{arith::CmpFPredicate::UNO, 15U},
      std::pair{arith::CmpFPredicate::AlwaysTrue, 16U},
  };
  ok &= checkOwnedEnumCodecs(
      floatPredicates,
      [](arith::CmpFPredicate value) {
        return encodeFloatComparePredicate(value);
      },
      [](llvm::ArrayRef<std::uint8_t> bytes) {
        return decodeFloatComparePredicate(bytes);
      },
      llvm::StringRef(kFloatPredicateDomain, sizeof(kFloatPredicateDomain) - 1),
      "floating compare predicate");
  constexpr std::array roundingModes = {
      std::pair{arith::RoundingMode::to_nearest_even, 1U},
      std::pair{arith::RoundingMode::downward, 2U},
      std::pair{arith::RoundingMode::upward, 3U},
      std::pair{arith::RoundingMode::toward_zero, 4U},
      std::pair{arith::RoundingMode::to_nearest_away, 5U},
  };
  ok &= checkOwnedEnumCodecs(
      roundingModes,
      [](arith::RoundingMode value) { return encodeRoundingMode(value); },
      [](llvm::ArrayRef<std::uint8_t> bytes) {
        return decodeRoundingMode(bytes);
      },
      llvm::StringRef(kRoundingModeDomain, sizeof(kRoundingModeDomain) - 1),
      "rounding mode");
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

  Type aggregateBody[] = {
      IntegerType::get(&context, 32),
      LLVM::LLVMArrayType::get(IntegerType::get(&context, 16), 4)};
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
makeFloatCompareProjection(MLIRContext &context,
                           arith::CmpFPredicate predicate) {
  Builder builder(&context);
  Type f32 = builder.getF32Type();
  Type i1 = builder.getI1Type();
  return {OperationSchemaId::ArithCmpF,
          builder.getFunctionType({f32, f32}, {i1}),
          SemanticPayload{
              FloatComparePayload{predicate, arith::FastMathFlags::none}}};
}

bool checkGoldenBytes(llvm::ArrayRef<std::uint8_t> actual,
                      llvm::ArrayRef<std::uint8_t> expected,
                      llvm::StringRef label) {
  if (actual == expected)
    return true;
  llvm::errs() << label << " changed persistent bytes (actual " << actual.size()
               << ", expected " << expected.size() << "); actual = {";
  for (std::uint8_t byte : actual)
    llvm::errs() << static_cast<unsigned>(byte) << ", ";
  llvm::errs() << "}\n";
  return false;
}

CanonicalActorSchemaProjection
makeSpecialMathProjection(MLIRContext &context,
                          loom::SpecialMathAccuracyTier accuracy) {
  Builder builder(&context);
  Type f32 = builder.getF32Type();
  return {
      OperationSchemaId::MathSin, builder.getFunctionType({f32}, {f32}),
      SemanticPayload{SpecialMathPayload{arith::FastMathFlags::afn, accuracy}}};
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
    constexpr std::array<std::uint8_t, 106> expectedFloatingProjection = {
        108, 111, 111, 109, 46,  100, 97,  116, 97,  102, 108, 111, 119, 46,
        97,  99,  116, 111, 114, 45,  115, 99,  104, 101, 109, 97,  45,  112,
        114, 111, 106, 101, 99,  116, 105, 111, 110, 0,   0,   0,   0,   2,
        0,   0,   0,   0,   76,  83,  2,   27,  76,  83,  1,   2,   0,   0,
        0,   0,   0,   0,   0,   2,   0,   0,   0,   4,   0,   0,   0,   15,
        0,   0,   0,   4,   0,   0,   0,   15,  0,   0,   0,   0,   0,   0,
        0,   1,   0,   0,   0,   4,   0,   0,   0,   15,  0,   0,   0,   2,
        0,   0,   0,   1,   0,   0,   0,   2};
    ok &= checkGoldenBytes(firstBytes->bytes(), expectedFloatingProjection,
                           "floating actor projection");
    std::vector<std::uint8_t> projectionPrefix(
        reinterpret_cast<const std::uint8_t *>(kProjectionDomain),
        reinterpret_cast<const std::uint8_t *>(kProjectionDomain) +
            sizeof(kProjectionDomain) - 1);
    appendU32(projectionPrefix, 2);
    appendU32(projectionPrefix, 0);
    if (firstBytes->bytes().size() < projectionPrefix.size() ||
        firstBytes->bytes().take_front(projectionPrefix.size()) !=
            llvm::ArrayRef<std::uint8_t>(projectionPrefix)) {
      llvm::errs() << "actor projection did not use codec 2.0\n";
      ok = false;
    }
    std::vector<std::uint8_t> oldProjection(firstBytes->bytes().begin(),
                                            firstBytes->bytes().end());
    const std::size_t versionOffset = sizeof(kProjectionDomain) - 1;
    std::fill(oldProjection.begin() + versionOffset,
              oldProjection.begin() + versionOffset + 4, 0);
    oldProjection[versionOffset + 3] = 1;
    ok &= expectValidationFailure(oldProjection, "unsupported version");
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

  auto compareBytes = encodeCanonicalActorSchemaProjection(
      makeFloatCompareProjection(context, arith::CmpFPredicate::UNO));
  if (!compareBytes) {
    llvm::errs() << llvm::toString(compareBytes.takeError()) << '\n';
    ok = false;
  } else {
    constexpr std::array<std::uint8_t, 106> expectedCompareProjection = {
        108, 111, 111, 109, 46,  100, 97,  116, 97,  102, 108, 111, 119, 46,
        97,  99,  116, 111, 114, 45,  115, 99,  104, 101, 109, 97,  45,  112,
        114, 111, 106, 101, 99,  116, 105, 111, 110, 0,   0,   0,   0,   2,
        0,   0,   0,   0,   76,  83,  2,   37,  76,  83,  1,   5,   0,   0,
        0,   0,   0,   0,   0,   2,   0,   0,   0,   4,   0,   0,   0,   15,
        0,   0,   0,   4,   0,   0,   0,   15,  0,   0,   0,   0,   0,   0,
        0,   1,   0,   0,   0,   3,   0,   0,   0,   1,   0,   0,   0,   1,
        0,   0,   0,   15,  0,   0,   0,   0};
    ok &= checkGoldenBytes(compareBytes->bytes(), expectedCompareProjection,
                           "floating compare actor projection");
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

  CanonicalActorSchemaProjection oneUlp = makeSpecialMathProjection(
      context, loom::SpecialMathAccuracyTier::Max1Ulp);
  CanonicalActorSchemaProjection twoUlp = makeSpecialMathProjection(
      context, loom::SpecialMathAccuracyTier::Max2Ulp);
  auto oneUlpBytes = encodeCanonicalActorSchemaProjection(oneUlp);
  auto twoUlpBytes = encodeCanonicalActorSchemaProjection(twoUlp);
  if (!oneUlpBytes || !twoUlpBytes) {
    if (!oneUlpBytes)
      llvm::errs() << llvm::toString(oneUlpBytes.takeError()) << '\n';
    if (!twoUlpBytes)
      llvm::errs() << llvm::toString(twoUlpBytes.takeError()) << '\n';
    ok = false;
  } else if (oneUlpBytes->bytes() == twoUlpBytes->bytes()) {
    llvm::errs() << "special-math accuracy did not change projection bytes\n";
    ok = false;
  } else {
    if (llvm::Error error =
            validateCanonicalActorSchemaProjectionBytes(oneUlpBytes->bytes())) {
      llvm::errs() << llvm::toString(std::move(error)) << '\n';
      ok = false;
    }
    std::vector<std::uint8_t> unknownTier(twoUlpBytes->bytes().begin(),
                                          twoUlpBytes->bytes().end());
    std::fill(unknownTier.end() - 4, unknownTier.end(), 0xff);
    ok &= expectValidationFailure(unknownTier,
                                  "unknown special-math accuracy tier");

    std::vector<std::uint8_t> missingAfn(oneUlpBytes->bytes().begin(),
                                         oneUlpBytes->bytes().end());
    std::fill(missingAfn.end() - 8, missingAfn.end() - 4, 0);
    ok &= expectValidationFailure(missingAfn, "requires afn");
  }
  CanonicalActorSchemaProjection unauthorized = oneUlp;
  std::get<SpecialMathPayload>(unauthorized.payload).flags =
      arith::FastMathFlags::none;
  ok &= expectFailure(encodeCanonicalActorSchemaProjection(unauthorized),
                      "requires afn");
  CanonicalActorSchemaProjection invalidAccuracy = oneUlp;
  std::get<SpecialMathPayload>(invalidAccuracy.payload).accuracy =
      static_cast<loom::SpecialMathAccuracyTier>(0xff);
  ok &= expectFailure(encodeCanonicalActorSchemaProjection(invalidAccuracy),
                      "unknown special-math accuracy tier");

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

  CanonicalActorSchemaProjection scalarConstant{
      OperationSchemaId::ArithConstant,
      builder.getFunctionType({}, {builder.getI8Type()}),
      SemanticPayload{ConstantValuePayload{builder.getI8IntegerAttr(5)}}};
  auto scalarConstantBytes =
      encodeCanonicalActorSchemaProjection(scalarConstant);
  if (!scalarConstantBytes) {
    llvm::errs() << llvm::toString(scalarConstantBytes.takeError()) << '\n';
    ok = false;
  } else {
    std::vector<std::uint8_t> mismatchedBytes(
        scalarConstantBytes->bytes().begin(),
        scalarConstantBytes->bytes().end());
    mismatchedBytes[mismatchedBytes.size() - 6] = 4;
    mismatchedBytes[mismatchedBytes.size() - 2] = 4;
    ok &= expectValidationFailure(mismatchedBytes,
                                  "does not match actor result type");
  }
  CanonicalActorSchemaProjection mismatchedConstant = scalarConstant;
  std::get<ConstantValuePayload>(mismatchedConstant.payload).value =
      builder.getIntegerAttr(builder.getIntegerType(4), 5);
  ok &= expectFailure(encodeCanonicalActorSchemaProjection(mismatchedConstant),
                      "does not match actor result type");

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

  CanonicalActorSchemaProjection dynamicVectorExtract{
      OperationSchemaId::VectorExtract,
      builder.getFunctionType({vector, builder.getIndexType()},
                              {builder.getI32Type()}),
      SemanticPayload{VectorStaticPositionPayload{{ShapedType::kDynamic}}}};
  auto dynamicBytes =
      encodeCanonicalActorSchemaProjection(dynamicVectorExtract);
  if (!dynamicBytes) {
    llvm::errs() << llvm::toString(dynamicBytes.takeError()) << '\n';
    ok = false;
  } else if (llvm::Error error = validateCanonicalActorSchemaProjectionBytes(
                 dynamicBytes->bytes())) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    ok = false;
  } else {
    std::vector<std::uint8_t> malformed(dynamicBytes->bytes().begin(),
                                        dynamicBytes->bytes().end());
    std::fill(malformed.end() - sizeof(std::int64_t), malformed.end(), 0xff);
    ok &= expectValidationFailure(
        malformed, "vector static position contains an invalid value");
  }
  CanonicalActorSchemaProjection invalidVectorExtract = dynamicVectorExtract;
  invalidVectorExtract.payload = VectorStaticPositionPayload{{-1}};
  ok &=
      expectFailure(encodeCanonicalActorSchemaProjection(invalidVectorExtract),
                    "vector static position contains an invalid value");

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
  ok &= checkSpecialMathAccuracyCodec();
  ok &= checkOwnedAtomCodecs(context);
  ok &= checkProjectionCodec(context);
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

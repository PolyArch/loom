#ifndef FABRIC_IR_IMPLEMENTATIONFAMILY_H
#define FABRIC_IR_IMPLEMENTATIONFAMILY_H

#include "Dataflow/IR/OperationSchema.h"
#include "Fabric/IR/FabricEnums.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace fabric {

/// The closed typed `hw_params` record schema a family selects.
enum class CapabilityParamsSchemaId : std::uint32_t {
#define LOOM_CAPABILITY_PARAMS_SCHEMA(Name, Id) Name = Id,
#include "Fabric/IR/ImplementationFamilies.inc"
};

/// The closed typed admission rule a family selects.
enum class TypedAdmissionProviderId : std::uint32_t {
#define LOOM_TYPED_ADMISSION_PROVIDER(Name, Id) Name = Id,
#include "Fabric/IR/ImplementationFamilies.inc"
};

/// The one normative family descriptor. It owns exactly four facts: the stable
/// family identity, the admitted registered operation schemas, the closed
/// typed capability-parameter record schema, and the typed admission provider.
///
/// It carries no name, spelling, shape policy, port shape, state, timing, or
/// backend field. Diagnostic spelling is derived from the family identity.
struct ImplementationFamilyDescriptor {
  ImplementationFamilyId familyId;
  llvm::ArrayRef<::dataflow::OperationSchemaId> admittedSchemas;
  CapabilityParamsSchemaId capabilityParamsSchema;
  TypedAdmissionProviderId typedAdmissionProvider;
};

namespace detail {

template <typename Element, std::size_t DomainSize> class ClosedEnumSet {
  static_assert(DomainSize <= 64, "closed enum set exceeds its representation");

public:
  static ClosedEnumSet get(std::initializer_list<Element> elements) {
    ClosedEnumSet result;
    for (Element element : elements)
      result.insert(element);
    return result;
  }

  bool insert(Element element) {
    std::size_t index = static_cast<std::size_t>(element);
    if (index >= DomainSize) {
      valid_ = false;
      return false;
    }
    bits_ |= std::uint64_t{1} << index;
    return true;
  }

  bool contains(Element element) const {
    std::size_t index = static_cast<std::size_t>(element);
    return valid_ && index < DomainSize &&
           (bits_ & (std::uint64_t{1} << index)) != 0;
  }
  bool empty() const { return bits_ == 0; }
  std::size_t size() const {
    std::size_t result = 0;
    for (std::uint64_t remaining = bits_; remaining != 0;
         remaining &= remaining - 1)
      ++result;
    return result;
  }
  bool valid() const { return valid_; }
  bool isSubsetOf(ClosedEnumSet other) const {
    return valid_ && other.valid_ && (bits_ & ~other.bits_) == 0;
  }

private:
  std::uint64_t bits_ = 0;
  bool valid_ = true;
};

template <typename Source, std::size_t SourceDomainSize, typename Destination,
          std::size_t DestinationDomainSize>
class ClosedPairRelation {
  static_assert(SourceDomainSize * DestinationDomainSize <= 64,
                "closed pair relation exceeds its representation");

public:
  using Pair = std::pair<Source, Destination>;

  static ClosedPairRelation get(std::initializer_list<Pair> pairs) {
    ClosedPairRelation result;
    for (Pair pair : pairs)
      result.insert(pair.first, pair.second);
    return result;
  }

  bool insert(Source source, Destination destination) {
    std::size_t sourceIndex = static_cast<std::size_t>(source);
    std::size_t destinationIndex = static_cast<std::size_t>(destination);
    if (sourceIndex >= SourceDomainSize ||
        destinationIndex >= DestinationDomainSize) {
      valid_ = false;
      return false;
    }
    std::size_t index = sourceIndex * DestinationDomainSize + destinationIndex;
    bits_ |= std::uint64_t{1} << index;
    return true;
  }

  bool contains(Source source, Destination destination) const {
    std::size_t sourceIndex = static_cast<std::size_t>(source);
    std::size_t destinationIndex = static_cast<std::size_t>(destination);
    if (!valid_ || sourceIndex >= SourceDomainSize ||
        destinationIndex >= DestinationDomainSize)
      return false;
    std::size_t index = sourceIndex * DestinationDomainSize + destinationIndex;
    return (bits_ & (std::uint64_t{1} << index)) != 0;
  }
  bool empty() const { return bits_ == 0; }
  std::size_t size() const {
    std::size_t result = 0;
    for (std::uint64_t remaining = bits_; remaining != 0;
         remaining &= remaining - 1)
      ++result;
    return result;
  }
  bool valid() const { return valid_; }

private:
  std::uint64_t bits_ = 0;
  bool valid_ = true;
};

} // namespace detail

/// Closed scalar integer widths admitted by the initial family schemas.
enum class IntegerWidth : std::uint8_t { I1, I8, I16, I32, I64 };
static_assert(static_cast<std::uint8_t>(IntegerWidth::I1) == 0);
static_assert(static_cast<std::uint8_t>(IntegerWidth::I8) == 1);
static_assert(static_cast<std::uint8_t>(IntegerWidth::I16) == 2);
static_assert(static_cast<std::uint8_t>(IntegerWidth::I32) == 3);
static_assert(static_cast<std::uint8_t>(IntegerWidth::I64) == 4);
using IntegerWidthSet = detail::ClosedEnumSet<IntegerWidth, 5>;

/// Closed scalar floating-point formats admitted by the initial schemas.
enum class FloatFormat : std::uint8_t { F16, BF16, F32, F64 };
static_assert(static_cast<std::uint8_t>(FloatFormat::F16) == 0);
static_assert(static_cast<std::uint8_t>(FloatFormat::BF16) == 1);
static_assert(static_cast<std::uint8_t>(FloatFormat::F32) == 2);
static_assert(static_cast<std::uint8_t>(FloatFormat::F64) == 3);
using FloatFormatSet = detail::ClosedEnumSet<FloatFormat, 4>;

static_assert(static_cast<std::uint32_t>(
                  ::mlir::arith::RoundingMode::to_nearest_even) == 0);
static_assert(
    static_cast<std::uint32_t>(::mlir::arith::RoundingMode::downward) == 1);
static_assert(static_cast<std::uint32_t>(::mlir::arith::RoundingMode::upward) ==
              2);
static_assert(
    static_cast<std::uint32_t>(::mlir::arith::RoundingMode::toward_zero) == 3);
static_assert(static_cast<std::uint32_t>(
                  ::mlir::arith::RoundingMode::to_nearest_away) == 4);
static_assert(::mlir::arith::getMaxEnumValForRoundingMode() == 4);
using RoundingModeSet = detail::ClosedEnumSet<::mlir::arith::RoundingMode, 5>;

static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpIPredicate::eq) ==
              0);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpIPredicate::ne) ==
              1);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpIPredicate::slt) ==
              2);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpIPredicate::sle) ==
              3);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpIPredicate::sgt) ==
              4);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpIPredicate::sge) ==
              5);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpIPredicate::ult) ==
              6);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpIPredicate::ule) ==
              7);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpIPredicate::ugt) ==
              8);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpIPredicate::uge) ==
              9);
static_assert(::mlir::arith::getMaxEnumValForCmpIPredicate() == 9);
using IntegerPredicateSet =
    detail::ClosedEnumSet<::mlir::arith::CmpIPredicate, 10>;

static_assert(
    static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::AlwaysFalse) == 0);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::OEQ) ==
              1);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::OGT) ==
              2);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::OGE) ==
              3);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::OLT) ==
              4);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::OLE) ==
              5);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::ONE) ==
              6);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::ORD) ==
              7);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::UEQ) ==
              8);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::UGT) ==
              9);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::UGE) ==
              10);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::ULT) ==
              11);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::ULE) ==
              12);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::UNE) ==
              13);
static_assert(static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::UNO) ==
              14);
static_assert(
    static_cast<std::uint64_t>(::mlir::arith::CmpFPredicate::AlwaysTrue) == 15);
static_assert(::mlir::arith::getMaxEnumValForCmpFPredicate() == 15);
using FloatPredicateSet =
    detail::ClosedEnumSet<::mlir::arith::CmpFPredicate, 16>;

enum class FloatNaNBehavior : std::uint8_t { IEEE, NumberPreferred };
static_assert(static_cast<std::uint8_t>(FloatNaNBehavior::IEEE) == 0);
static_assert(static_cast<std::uint8_t>(FloatNaNBehavior::NumberPreferred) ==
              1);
using FloatNaNBehaviorSet = detail::ClosedEnumSet<FloatNaNBehavior, 2>;

enum class FloatSubnormalBehavior : std::uint8_t { Preserve, FlushToZero };
static_assert(static_cast<std::uint8_t>(FloatSubnormalBehavior::Preserve) == 0);
static_assert(static_cast<std::uint8_t>(FloatSubnormalBehavior::FlushToZero) ==
              1);
using FloatSubnormalBehaviorSet =
    detail::ClosedEnumSet<FloatSubnormalBehavior, 2>;

enum class FloatSignedZeroBehavior : std::uint8_t { Preserve, IgnoreSign };
static_assert(static_cast<std::uint8_t>(FloatSignedZeroBehavior::Preserve) ==
              0);
static_assert(static_cast<std::uint8_t>(FloatSignedZeroBehavior::IgnoreSign) ==
              1);
using FloatSignedZeroBehaviorSet =
    detail::ClosedEnumSet<FloatSignedZeroBehavior, 2>;

/// Observable floating behavior of one concrete scalar implementation.
struct FloatBehaviorProfile {
  RoundingModeSet roundingModes;
  FloatNaNBehaviorSet nanBehaviors;
  FloatSubnormalBehaviorSet subnormalBehaviors;
  FloatSignedZeroBehaviorSet signedZeroBehaviors;
  ::mlir::arith::FastMathFlags admittedFastMath =
      ::mlir::arith::FastMathFlags::none;

  static FloatBehaviorProfile strictIEEE() {
    return {
        RoundingModeSet::get({::mlir::arith::RoundingMode::to_nearest_even}),
        FloatNaNBehaviorSet::get({FloatNaNBehavior::IEEE}),
        FloatSubnormalBehaviorSet::get({FloatSubnormalBehavior::Preserve}),
        FloatSignedZeroBehaviorSet::get({FloatSignedZeroBehavior::Preserve}),
        ::mlir::arith::FastMathFlags::none};
  }
};

enum class ResolvedIndexWidth : std::uint8_t { I32, I64 };
static_assert(static_cast<std::uint8_t>(ResolvedIndexWidth::I32) == 0);
static_assert(static_cast<std::uint8_t>(ResolvedIndexWidth::I64) == 1);

using IntegerWidthRelation =
    detail::ClosedPairRelation<IntegerWidth, 5, IntegerWidth, 5>;
using FloatFormatRelation =
    detail::ClosedPairRelation<FloatFormat, 4, FloatFormat, 4>;
using IntegerFloatFormatRelation =
    detail::ClosedPairRelation<IntegerWidth, 5, FloatFormat, 4>;

/// Typed finite-domain relation for integer and resolved-index casts.
struct IntegerCastRelation {
  IntegerWidthRelation widthPairs;
  std::optional<ResolvedIndexWidth> resolvedIndexWidth;
};

struct ScalarIntegerParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::ScalarIntegerParams;
  IntegerWidthSet integerWidths;
};

struct ScalarIntegerCompareMinMaxParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::ScalarIntegerCompareMinMaxParams;
  IntegerWidthSet operandWidths;
  IntegerPredicateSet predicates;
};

struct ScalarValueSelectParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::ScalarValueSelectParams;
  IntegerWidthSet integerWidths;
  FloatFormatSet floatFormats;
};

struct ScalarIntegerCastParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::ScalarIntegerCastParams;
  IntegerCastRelation relation;
};

struct ScalarBitReinterpretParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::ScalarBitReinterpretParams;
  IntegerWidthSet integerWidths;
  FloatFormatSet floatFormats;
};

struct ScalarFloatParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::ScalarFloatParams;
  FloatFormatSet formats;
  FloatBehaviorProfile behavior;
};

struct ScalarFloatCompareMinMaxParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::ScalarFloatCompareMinMaxParams;
  FloatFormatSet formats;
  FloatBehaviorProfile behavior;
  FloatPredicateSet predicates;
};

struct ScalarFloatWidthCastParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::ScalarFloatWidthCastParams;
  FloatFormatRelation formatPairs;
  FloatBehaviorProfile behavior;
};

struct ScalarIntegerFloatConversionParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::ScalarIntegerFloatConversionParams;
  IntegerFloatFormatRelation formatPairs;
  FloatBehaviorProfile behavior;
};

struct LoopStreamParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::LoopStreamParams;
  IntegerWidthSet integerWidths;
  ::dataflow::StreamStepKind fixedStepKind;
  IntegerPredicateSet continuationPredicates;
};

/// Selects the bit-preserving scalar, float, fixed-vector, or none rule.
struct TokenPlaneParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::TokenPlaneParams;
};

struct FixedVectorIntegerParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::FixedVectorIntegerParams;
  IntegerWidthSet elementWidths;
  std::uint32_t maxPayloadBits;
};

struct FixedVectorIntegerCompareMinMaxParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::FixedVectorIntegerCompareMinMaxParams;
  IntegerWidthSet elementWidths;
  IntegerPredicateSet predicates;
  std::uint32_t maxPayloadBits;
};

struct FixedVectorValueSelectParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::FixedVectorValueSelectParams;
  IntegerWidthSet integerElementWidths;
  FloatFormatSet floatElementFormats;
  std::uint32_t maxPayloadBits;
};

struct FixedVectorFloatParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::FixedVectorFloatParams;
  FloatFormatSet elementFormats;
  FloatBehaviorProfile behavior;
  std::uint32_t maxPayloadBits;
};

struct FixedVectorFloatCompareMinMaxParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::FixedVectorFloatCompareMinMaxParams;
  FloatFormatSet elementFormats;
  FloatBehaviorProfile behavior;
  FloatPredicateSet predicates;
  std::uint32_t maxPayloadBits;
};

struct FixedVectorAdapterParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::FixedVectorAdapterParams;
  IntegerWidthSet integerElementWidths;
  FloatFormatSet floatElementFormats;
  std::uint32_t maxPayloadBits;
};

struct PayloadCapacityParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::PayloadCapacityParams;
  std::uint32_t maxPayloadBits;
};

struct RoutedTokenParams {
  static constexpr CapabilityParamsSchemaId schemaId =
      CapabilityParamsSchemaId::RoutedTokenParams;
  std::uint32_t maxPayloadBits;
  std::uint32_t maxFan;
};

using FamilyCapabilityParams =
    std::variant<ScalarIntegerParams, ScalarIntegerCompareMinMaxParams,
                 ScalarValueSelectParams, ScalarIntegerCastParams,
                 ScalarBitReinterpretParams, ScalarFloatParams,
                 ScalarFloatCompareMinMaxParams, ScalarFloatWidthCastParams,
                 ScalarIntegerFloatConversionParams, LoopStreamParams,
                 TokenPlaneParams, FixedVectorIntegerParams,
                 FixedVectorIntegerCompareMinMaxParams,
                 FixedVectorValueSelectParams, FixedVectorFloatParams,
                 FixedVectorFloatCompareMinMaxParams, FixedVectorAdapterParams,
                 PayloadCapacityParams, RoutedTokenParams>;

/// Count of registered families. Every family id is in `[0, count)`.
std::uint32_t implementationFamilyCount();

/// The descriptor of one registered family. Lookup is a dense index.
const ImplementationFamilyDescriptor &
implementationFamily(ImplementationFamilyId family);

/// The one keyword of a family, derived from its generated identity. It is the
/// only spelling the typed attribute accepts and prints, and the only spelling
/// a diagnostic uses.
llvm::StringRef implementationFamilyKeyword(ImplementationFamilyId family);

/// The family named by `keyword`, or absent when none is.
std::optional<ImplementationFamilyId>
findImplementationFamily(llvm::StringRef keyword);

/// Whether `family` admits `schema`. The admitted set of a real shared
/// datapath family is small and fixed, so this is a bounded scan of the one
/// generated relation.
bool admitsOperationSchema(ImplementationFamilyId family,
                           ::dataflow::OperationSchemaId schema);

/// The generated implementation families that admit `schema`. This is a
/// mechanical projection of the descriptor table; callers still choose an
/// exact family explicitly when more than one physical implementation exists.
llvm::SmallVector<ImplementationFamilyId, 2>
implementationFamiliesFor(::dataflow::OperationSchemaId schema);

/// Diagnostic spellings of the two closed vocabularies a descriptor selects.
llvm::StringRef capabilityParamsSchemaKeyword(CapabilityParamsSchemaId schema);
llvm::StringRef
typedAdmissionProviderKeyword(TypedAdmissionProviderId provider);

/// The generated schema identity of one closed typed capability record.
CapabilityParamsSchemaId
capabilityParamsSchema(const FamilyCapabilityParams &params);

/// Decodes the canonical closed `hw_params` record selected by `family`.
/// Unknown fields, missing fields, malformed values, and empty required
/// domains are rejected. The dictionary is serialization syntax only; the
/// returned typed sum is the semantic capability value.
llvm::Expected<FamilyCapabilityParams>
parseFamilyCapabilityParams(ImplementationFamilyId family,
                            ::mlir::DictionaryAttr params);

/// Encodes one typed capability record using its canonical field spellings.
/// The caller must pair it with a family whose generated descriptor selects
/// the same schema.
::mlir::DictionaryAttr
getFamilyCapabilityParamsAttr(::mlir::MLIRContext *context,
                              const FamilyCapabilityParams &params);

/// Verifies that one exact registered actor projection is accepted by the
/// selected family's concrete typed capability. Missing or mismatched
/// parameters and every unsupported semantic point fail closed.
llvm::Error verifyImplementationFamilyAdmission(
    ImplementationFamilyId family, const FamilyCapabilityParams *params,
    const ::dataflow::CanonicalActorSchemaProjection &actor);

/// Whether one concrete operation relation needs a semantic configuration
/// field to distinguish admitted hardware behaviors. The field is one
/// composite typed value; ConfigurationABI alone later flattens it into
/// physical leaves. This query interprets the generated family's existing
/// typed admission provider and does not define another family registry.
llvm::Expected<bool> requiresSemanticConfigurationField(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    std::uint32_t physicalInputCount, std::uint32_t physicalResultCount);

/// Exact flattened payload width used by concrete operation-resource
/// admission. Equal widths do not imply equal actor semantics.
::mlir::FailureOr<unsigned> getSemanticPayloadWidth(::mlir::Type type,
                                                    std::string &error);

} // namespace fabric

#endif // FABRIC_IR_IMPLEMENTATIONFAMILY_H

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "OperationSchemaCodecInternal.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

using dataflow::detail::invalid;
using dataflow::detail::readDomain;
using dataflow::detail::Reader;
using dataflow::detail::writeDomain;
using dataflow::detail::Writer;

namespace {

constexpr char kIntegerPredicateDomain[] =
    "loom.dataflow.integer-compare-predicate\0";
constexpr char kFloatPredicateDomain[] =
    "loom.dataflow.float-compare-predicate\0";
constexpr char kRoundingModeDomain[] = "loom.dataflow.rounding-mode\0";
constexpr char kServiceRoleDomain[] = "loom.dataflow.service-value-role\0";
constexpr char kServiceKindDomain[] = "loom.dataflow.service-kind\0";
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

static_assert(mlir::arith::getMaxEnumValForCmpIPredicate() == 9);
static_assert(mlir::arith::getMaxEnumValForCmpFPredicate() == 15);
static_assert(mlir::arith::getMaxEnumValForRoundingMode() == 4);

constexpr std::array kFloatPredicateWireRelation = {
    std::pair{mlir::arith::CmpFPredicate::AlwaysFalse, 1U},
    std::pair{mlir::arith::CmpFPredicate::OEQ, 2U},
    std::pair{mlir::arith::CmpFPredicate::OGT, 3U},
    std::pair{mlir::arith::CmpFPredicate::OGE, 4U},
    std::pair{mlir::arith::CmpFPredicate::OLT, 5U},
    std::pair{mlir::arith::CmpFPredicate::OLE, 6U},
    std::pair{mlir::arith::CmpFPredicate::ONE, 7U},
    std::pair{mlir::arith::CmpFPredicate::ORD, 8U},
    std::pair{mlir::arith::CmpFPredicate::UEQ, 9U},
    std::pair{mlir::arith::CmpFPredicate::UGT, 10U},
    std::pair{mlir::arith::CmpFPredicate::UGE, 11U},
    std::pair{mlir::arith::CmpFPredicate::ULT, 12U},
    std::pair{mlir::arith::CmpFPredicate::ULE, 13U},
    std::pair{mlir::arith::CmpFPredicate::UNE, 14U},
    std::pair{mlir::arith::CmpFPredicate::UNO, 15U},
    std::pair{mlir::arith::CmpFPredicate::AlwaysTrue, 16U},
};

constexpr std::array kRoundingModeWireRelation = {
    std::pair{mlir::arith::RoundingMode::to_nearest_even, 1U},
    std::pair{mlir::arith::RoundingMode::downward, 2U},
    std::pair{mlir::arith::RoundingMode::upward, 3U},
    std::pair{mlir::arith::RoundingMode::toward_zero, 4U},
    std::pair{mlir::arith::RoundingMode::to_nearest_away, 5U},
};

template <typename Value, std::size_t N>
llvm::Expected<std::uint32_t>
wireTagFor(Value value,
           const std::array<std::pair<Value, std::uint32_t>, N> &relation,
           llvm::StringRef name) {
  for (const auto &[member, wireTag] : relation)
    if (member == value)
      return wireTag;
  return invalid("unknown " + name);
}

template <typename Value, std::size_t N>
llvm::Expected<Value>
valueForWireTag(std::uint32_t wireTag,
                const std::array<std::pair<Value, std::uint32_t>, N> &relation,
                llvm::StringRef name) {
  for (const auto &[member, memberWireTag] : relation)
    if (memberWireTag == wireTag)
      return member;
  return invalid("unknown " + name + " wire tag");
}

llvm::Error finish(Reader &reader) {
  if (!reader.empty())
    return invalid("trailing bytes");
  return llvm::Error::success();
}

template <std::size_t Size, typename Value, typename Encode>
llvm::Expected<loom::CanonicalSemanticBytes>
encodeVocabulary(Value value, const char (&domain)[Size], Encode encode) {
  auto wireTag = encode(value);
  if (!wireTag)
    return wireTag.takeError();
  Writer writer;
  writeDomain(writer, domain);
  writer.u32(*wireTag);
  return loom::CanonicalSemanticBytes(writer.take());
}

template <std::size_t Size, typename Value, typename Decode>
llvm::Expected<Value> decodeVocabulary(llvm::ArrayRef<std::uint8_t> bytes,
                                       const char (&domain)[Size],
                                       Decode decode, llvm::StringRef what) {
  Reader reader(bytes);
  if (llvm::Error error = readDomain(reader, domain))
    return std::move(error);
  auto wireTag = reader.u32(llvm::Twine(what) + " wire tag");
  if (!wireTag)
    return wireTag.takeError();
  auto value = decode(*wireTag);
  if (!value)
    return value.takeError();
  if (llvm::Error error = finish(reader))
    return std::move(error);
  return *value;
}

llvm::StringRef asString(llvm::ArrayRef<std::uint8_t> bytes) {
  return llvm::StringRef(reinterpret_cast<const char *>(bytes.data()),
                         bytes.size());
}

} // namespace

namespace dataflow::detail {

llvm::Expected<std::uint32_t>
integerPredicateWireTag(mlir::arith::CmpIPredicate predicate) {
  using Predicate = mlir::arith::CmpIPredicate;
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
  return invalid("unknown integer predicate");
}

llvm::Expected<mlir::arith::CmpIPredicate>
integerPredicateFromWireTag(std::uint32_t wireTag) {
  using Predicate = mlir::arith::CmpIPredicate;
  switch (wireTag) {
  case 1:
    return Predicate::eq;
  case 2:
    return Predicate::ne;
  case 3:
    return Predicate::slt;
  case 4:
    return Predicate::sle;
  case 5:
    return Predicate::sgt;
  case 6:
    return Predicate::sge;
  case 7:
    return Predicate::ult;
  case 8:
    return Predicate::ule;
  case 9:
    return Predicate::ugt;
  case 10:
    return Predicate::uge;
  default:
    return invalid("unknown integer predicate wire tag");
  }
}

llvm::Expected<std::uint32_t>
floatPredicateWireTag(mlir::arith::CmpFPredicate predicate) {
  return wireTagFor(predicate, kFloatPredicateWireRelation,
                    "floating predicate");
}

llvm::Expected<mlir::arith::CmpFPredicate>
floatPredicateFromWireTag(std::uint32_t wireTag) {
  return valueForWireTag(wireTag, kFloatPredicateWireRelation,
                         "floating predicate");
}

llvm::Expected<std::uint32_t>
roundingModeWireTag(mlir::arith::RoundingMode mode) {
  return wireTagFor(mode, kRoundingModeWireRelation, "rounding mode");
}

llvm::Expected<mlir::arith::RoundingMode>
roundingModeFromWireTag(std::uint32_t wireTag) {
  return valueForWireTag(wireTag, kRoundingModeWireRelation, "rounding mode");
}

llvm::Expected<std::uint32_t> serviceKindWireTag(semantics::ServiceKind kind) {
  using Kind = semantics::ServiceKind;
  switch (kind) {
  case Kind::MessageTransfer:
    return 1;
  case Kind::MemoryRead:
    return 2;
  case Kind::MemoryWrite:
    return 3;
  case Kind::MemoryAtomicRmw:
    return 4;
  case Kind::MemoryCompareExchange:
    return 5;
  case Kind::MemoryFence:
    return 6;
  }
  return invalid("unknown service kind");
}

llvm::Expected<semantics::ServiceKind>
serviceKindFromWireTag(std::uint32_t wireTag) {
  using Kind = semantics::ServiceKind;
  switch (wireTag) {
  case 1:
    return Kind::MessageTransfer;
  case 2:
    return Kind::MemoryRead;
  case 3:
    return Kind::MemoryWrite;
  case 4:
    return Kind::MemoryAtomicRmw;
  case 5:
    return Kind::MemoryCompareExchange;
  case 6:
    return Kind::MemoryFence;
  default:
    return invalid("unknown service kind wire tag");
  }
}

llvm::Expected<std::uint32_t>
serviceValueRoleWireTag(semantics::ServiceValueRole role) {
  using Role = semantics::ServiceValueRole;
  switch (role) {
  case Role::Payload:
    return 1;
  case Role::Address:
    return 2;
  case Role::Data:
    return 3;
  case Role::Update:
    return 4;
  case Role::Expected:
    return 5;
  case Role::Desired:
    return 6;
  case Role::Mask:
    return 7;
  case Role::Control:
    return 8;
  case Role::Old:
    return 9;
  case Role::Success:
    return 10;
  case Role::Completion:
    return 11;
  }
  return invalid("unknown service value role");
}

llvm::Expected<semantics::ServiceValueRole>
serviceValueRoleFromWireTag(std::uint32_t wireTag) {
  using Role = semantics::ServiceValueRole;
  switch (wireTag) {
  case 1:
    return Role::Payload;
  case 2:
    return Role::Address;
  case 3:
    return Role::Data;
  case 4:
    return Role::Update;
  case 5:
    return Role::Expected;
  case 6:
    return Role::Desired;
  case 7:
    return Role::Mask;
  case 8:
    return Role::Control;
  case 9:
    return Role::Old;
  case 10:
    return Role::Success;
  case 11:
    return Role::Completion;
  default:
    return invalid("unknown service value role wire tag");
  }
}

llvm::Expected<std::uint32_t>
memoryAccessFormWireTag(semantics::MemoryAccessForm form) {
  using Form = semantics::MemoryAccessForm;
  switch (form) {
  case Form::Element:
    return 1;
  case Form::Contiguous:
    return 2;
  case Form::Indexed:
    return 3;
  }
  return invalid("unknown memory access form");
}

llvm::Expected<semantics::MemoryAccessForm>
memoryAccessFormFromWireTag(std::uint32_t wireTag) {
  using Form = semantics::MemoryAccessForm;
  switch (wireTag) {
  case 1:
    return Form::Element;
  case 2:
    return Form::Contiguous;
  case 3:
    return Form::Indexed;
  default:
    return invalid("unknown memory access form wire tag");
  }
}

llvm::Expected<std::uint32_t>
memoryMaskFormWireTag(semantics::MemoryMaskForm form) {
  using Form = semantics::MemoryMaskForm;
  switch (form) {
  case Form::Absent:
    return 1;
  case Form::Dynamic:
    return 2;
  }
  return invalid("unknown memory mask form");
}

llvm::Expected<semantics::MemoryMaskForm>
memoryMaskFormFromWireTag(std::uint32_t wireTag) {
  using Form = semantics::MemoryMaskForm;
  switch (wireTag) {
  case 1:
    return Form::Absent;
  case 2:
    return Form::Dynamic;
  default:
    return invalid("unknown memory mask form wire tag");
  }
}

llvm::Expected<std::uint32_t> atomicOrderingWireTag(AtomicOrdering ordering) {
  switch (ordering) {
  case AtomicOrdering::Unordered:
    return 1;
  case AtomicOrdering::Monotonic:
    return 2;
  case AtomicOrdering::Acquire:
    return 3;
  case AtomicOrdering::Release:
    return 4;
  case AtomicOrdering::AcqRel:
    return 5;
  case AtomicOrdering::SeqCst:
    return 6;
  }
  return invalid("unknown atomic ordering");
}

llvm::Expected<AtomicOrdering>
atomicOrderingFromWireTag(std::uint32_t wireTag) {
  switch (wireTag) {
  case 1:
    return AtomicOrdering::Unordered;
  case 2:
    return AtomicOrdering::Monotonic;
  case 3:
    return AtomicOrdering::Acquire;
  case 4:
    return AtomicOrdering::Release;
  case 5:
    return AtomicOrdering::AcqRel;
  case 6:
    return AtomicOrdering::SeqCst;
  default:
    return invalid("unknown atomic ordering wire tag");
  }
}

llvm::Expected<std::uint32_t> syncScopeKindWireTag(SyncScopeKind kind) {
  switch (kind) {
  case SyncScopeKind::System:
    return 1;
  case SyncScopeKind::SingleThread:
    return 2;
  case SyncScopeKind::Target:
    return 3;
  }
  return invalid("unknown sync scope kind");
}

llvm::Expected<SyncScopeKind> syncScopeKindFromWireTag(std::uint32_t wireTag) {
  switch (wireTag) {
  case 1:
    return SyncScopeKind::System;
  case 2:
    return SyncScopeKind::SingleThread;
  case 3:
    return SyncScopeKind::Target;
  default:
    return invalid("unknown sync scope kind wire tag");
  }
}

llvm::Expected<std::uint32_t>
vectorAtomicGranularityWireTag(VectorAtomicGranularity granularity) {
  switch (granularity) {
  case VectorAtomicGranularity::WholePayload:
    return 1;
  case VectorAtomicGranularity::PerLane:
    return 2;
  }
  return invalid("unknown vector atomic granularity");
}

llvm::Expected<VectorAtomicGranularity>
vectorAtomicGranularityFromWireTag(std::uint32_t wireTag) {
  switch (wireTag) {
  case 1:
    return VectorAtomicGranularity::WholePayload;
  case 2:
    return VectorAtomicGranularity::PerLane;
  default:
    return invalid("unknown vector atomic granularity wire tag");
  }
}

llvm::Expected<std::uint32_t> atomicRmwKindWireTag(AtomicRmwKind kind) {
  switch (kind) {
  case AtomicRmwKind::Xchg:
    return 1;
  case AtomicRmwKind::Add:
    return 2;
  case AtomicRmwKind::Sub:
    return 3;
  case AtomicRmwKind::And:
    return 4;
  case AtomicRmwKind::Nand:
    return 5;
  case AtomicRmwKind::Or:
    return 6;
  case AtomicRmwKind::Xor:
    return 7;
  case AtomicRmwKind::Max:
    return 8;
  case AtomicRmwKind::Min:
    return 9;
  case AtomicRmwKind::UMax:
    return 10;
  case AtomicRmwKind::UMin:
    return 11;
  case AtomicRmwKind::FAdd:
    return 12;
  case AtomicRmwKind::FSub:
    return 13;
  case AtomicRmwKind::FMax:
    return 14;
  case AtomicRmwKind::FMin:
    return 15;
  case AtomicRmwKind::UIncWrap:
    return 16;
  case AtomicRmwKind::UDecWrap:
    return 17;
  case AtomicRmwKind::USubCond:
    return 18;
  case AtomicRmwKind::USubSat:
    return 19;
  case AtomicRmwKind::FMaximum:
    return 20;
  case AtomicRmwKind::FMinimum:
    return 21;
  case AtomicRmwKind::FMaximumNum:
    return 22;
  case AtomicRmwKind::FMinimumNum:
    return 23;
  }
  return invalid("unknown atomic RMW kind");
}

llvm::Expected<AtomicRmwKind> atomicRmwKindFromWireTag(std::uint32_t wireTag) {
  switch (wireTag) {
  case 1:
    return AtomicRmwKind::Xchg;
  case 2:
    return AtomicRmwKind::Add;
  case 3:
    return AtomicRmwKind::Sub;
  case 4:
    return AtomicRmwKind::And;
  case 5:
    return AtomicRmwKind::Nand;
  case 6:
    return AtomicRmwKind::Or;
  case 7:
    return AtomicRmwKind::Xor;
  case 8:
    return AtomicRmwKind::Max;
  case 9:
    return AtomicRmwKind::Min;
  case 10:
    return AtomicRmwKind::UMax;
  case 11:
    return AtomicRmwKind::UMin;
  case 12:
    return AtomicRmwKind::FAdd;
  case 13:
    return AtomicRmwKind::FSub;
  case 14:
    return AtomicRmwKind::FMax;
  case 15:
    return AtomicRmwKind::FMin;
  case 16:
    return AtomicRmwKind::UIncWrap;
  case 17:
    return AtomicRmwKind::UDecWrap;
  case 18:
    return AtomicRmwKind::USubCond;
  case 19:
    return AtomicRmwKind::USubSat;
  case 20:
    return AtomicRmwKind::FMaximum;
  case 21:
    return AtomicRmwKind::FMinimum;
  case 22:
    return AtomicRmwKind::FMaximumNum;
  case 23:
    return AtomicRmwKind::FMinimumNum;
  default:
    return invalid("unknown atomic RMW kind wire tag");
  }
}

} // namespace dataflow::detail

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeIntegerComparePredicate(mlir::arith::CmpIPredicate predicate) {
  return encodeVocabulary(predicate, kIntegerPredicateDomain,
                          detail::integerPredicateWireTag);
}

llvm::Expected<mlir::arith::CmpIPredicate>
dataflow::decodeIntegerComparePredicate(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<sizeof(kIntegerPredicateDomain),
                          mlir::arith::CmpIPredicate>(
      bytes, kIntegerPredicateDomain, detail::integerPredicateFromWireTag,
      "integer predicate");
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeFloatComparePredicate(mlir::arith::CmpFPredicate predicate) {
  return encodeVocabulary(predicate, kFloatPredicateDomain,
                          detail::floatPredicateWireTag);
}

llvm::Expected<mlir::arith::CmpFPredicate>
dataflow::decodeFloatComparePredicate(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<sizeof(kFloatPredicateDomain),
                          mlir::arith::CmpFPredicate>(
      bytes, kFloatPredicateDomain, detail::floatPredicateFromWireTag,
      "floating predicate");
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeRoundingMode(mlir::arith::RoundingMode mode) {
  return encodeVocabulary(mode, kRoundingModeDomain,
                          detail::roundingModeWireTag);
}

llvm::Expected<mlir::arith::RoundingMode>
dataflow::decodeRoundingMode(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<sizeof(kRoundingModeDomain),
                          mlir::arith::RoundingMode>(
      bytes, kRoundingModeDomain, detail::roundingModeFromWireTag,
      "rounding mode");
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeServiceKind(semantics::ServiceKind kind) {
  return encodeVocabulary(kind, kServiceKindDomain, detail::serviceKindWireTag);
}

llvm::Expected<dataflow::semantics::ServiceKind>
dataflow::decodeServiceKind(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<sizeof(kServiceKindDomain), semantics::ServiceKind>(
      bytes, kServiceKindDomain, detail::serviceKindFromWireTag,
      "service kind");
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeServiceValueRole(semantics::ServiceValueRole role) {
  return encodeVocabulary(role, kServiceRoleDomain,
                          detail::serviceValueRoleWireTag);
}

llvm::Expected<dataflow::semantics::ServiceValueRole>
dataflow::decodeServiceValueRole(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<sizeof(kServiceRoleDomain),
                          semantics::ServiceValueRole>(
      bytes, kServiceRoleDomain, detail::serviceValueRoleFromWireTag,
      "service value role");
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeMemoryAccessForm(semantics::MemoryAccessForm form) {
  return encodeVocabulary(form, kMemoryAccessFormDomain,
                          detail::memoryAccessFormWireTag);
}

llvm::Expected<dataflow::semantics::MemoryAccessForm>
dataflow::decodeMemoryAccessForm(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<sizeof(kMemoryAccessFormDomain),
                          semantics::MemoryAccessForm>(
      bytes, kMemoryAccessFormDomain, detail::memoryAccessFormFromWireTag,
      "memory access form");
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeMemoryMaskForm(semantics::MemoryMaskForm form) {
  return encodeVocabulary(form, kMemoryMaskFormDomain,
                          detail::memoryMaskFormWireTag);
}

llvm::Expected<dataflow::semantics::MemoryMaskForm>
dataflow::decodeMemoryMaskForm(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<sizeof(kMemoryMaskFormDomain),
                          semantics::MemoryMaskForm>(
      bytes, kMemoryMaskFormDomain, detail::memoryMaskFormFromWireTag,
      "memory mask form");
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeAtomicOrdering(AtomicOrdering ordering) {
  return encodeVocabulary(ordering, kAtomicOrderingDomain,
                          detail::atomicOrderingWireTag);
}

llvm::Expected<dataflow::AtomicOrdering>
dataflow::decodeAtomicOrdering(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<sizeof(kAtomicOrderingDomain), AtomicOrdering>(
      bytes, kAtomicOrderingDomain, detail::atomicOrderingFromWireTag,
      "atomic ordering");
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeAtomicRmwKind(AtomicRmwKind kind) {
  return encodeVocabulary(kind, kAtomicRmwKindDomain,
                          detail::atomicRmwKindWireTag);
}

llvm::Expected<dataflow::AtomicRmwKind>
dataflow::decodeAtomicRmwKind(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<sizeof(kAtomicRmwKindDomain), AtomicRmwKind>(
      bytes, kAtomicRmwKindDomain, detail::atomicRmwKindFromWireTag,
      "atomic RMW kind");
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeVectorAtomicGranularity(VectorAtomicGranularity granularity) {
  return encodeVocabulary(granularity, kVectorAtomicGranularityDomain,
                          detail::vectorAtomicGranularityWireTag);
}

llvm::Expected<dataflow::VectorAtomicGranularity>
dataflow::decodeVectorAtomicGranularity(llvm::ArrayRef<std::uint8_t> bytes) {
  return decodeVocabulary<sizeof(kVectorAtomicGranularityDomain),
                          VectorAtomicGranularity>(
      bytes, kVectorAtomicGranularityDomain,
      detail::vectorAtomicGranularityFromWireTag, "vector atomic granularity");
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeOptionalVectorAtomicGranularity(
    std::optional<VectorAtomicGranularity> granularity) {
  Writer writer;
  writeDomain(writer, kOptionalVectorAtomicGranularityDomain);
  writer.boolean(granularity.has_value());
  if (granularity) {
    auto wireTag = detail::vectorAtomicGranularityWireTag(*granularity);
    if (!wireTag)
      return wireTag.takeError();
    writer.u32(*wireTag);
  }
  return loom::CanonicalSemanticBytes(writer.take());
}

llvm::Expected<std::optional<dataflow::VectorAtomicGranularity>>
dataflow::decodeOptionalVectorAtomicGranularity(
    llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  if (llvm::Error error =
          readDomain(reader, kOptionalVectorAtomicGranularityDomain))
    return std::move(error);
  auto present = reader.boolean("vector granularity presence");
  if (!present)
    return present.takeError();
  std::optional<VectorAtomicGranularity> result;
  if (*present) {
    auto wireTag = reader.u32("vector atomic granularity wire tag");
    if (!wireTag)
      return wireTag.takeError();
    auto granularity = detail::vectorAtomicGranularityFromWireTag(*wireTag);
    if (!granularity)
      return granularity.takeError();
    result = *granularity;
  }
  if (llvm::Error error = finish(reader))
    return std::move(error);
  return result;
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeSyncScopeRef(const SyncScopeProjection &scope) {
  auto wireTag = detail::syncScopeKindWireTag(scope.kind);
  if (!wireTag)
    return wireTag.takeError();
  Writer writer;
  writeDomain(writer, kSyncScopeRefDomain);
  writer.u32(*wireTag);
  if (scope.kind == SyncScopeKind::Target) {
    if (!scope.targetNamespace || !scope.targetKey ||
        scope.targetNamespace.getValue().empty() ||
        scope.targetKey.getValue().empty())
      return invalid("target sync scope requires namespace and key");
    writer.string(scope.targetNamespace.getValue());
    writer.string(scope.targetKey.getValue());
  } else if (scope.targetNamespace || scope.targetKey) {
    return invalid("non-target sync scope carries target identity");
  }
  return loom::CanonicalSemanticBytes(writer.take());
}

llvm::Expected<dataflow::SyncScopeProjection>
dataflow::decodeSyncScopeRef(llvm::ArrayRef<std::uint8_t> bytes,
                             mlir::MLIRContext *context) {
  if (!context)
    return invalid("sync scope decode requires an MLIR context");
  Reader reader(bytes);
  if (llvm::Error error = readDomain(reader, kSyncScopeRefDomain))
    return std::move(error);
  auto wireTag = reader.u32("sync scope kind wire tag");
  if (!wireTag)
    return wireTag.takeError();
  auto kind = detail::syncScopeKindFromWireTag(*wireTag);
  if (!kind)
    return kind.takeError();
  SyncScopeProjection result{*kind, {}, {}};
  if (*kind == SyncScopeKind::Target) {
    auto targetNamespace = reader.string("sync scope target namespace");
    if (!targetNamespace)
      return targetNamespace.takeError();
    auto targetKey = reader.string("sync scope target key");
    if (!targetKey)
      return targetKey.takeError();
    if (targetNamespace->empty() || targetKey->empty())
      return invalid("target sync scope requires namespace and key");
    result.targetNamespace =
        mlir::StringAttr::get(context, asString(*targetNamespace));
    result.targetKey = mlir::StringAttr::get(context, asString(*targetKey));
  }
  if (llvm::Error error = finish(reader))
    return std::move(error);
  return result;
}

llvm::Expected<loom::CanonicalSemanticBytes>
dataflow::encodeCanonicalBoolean(bool value) {
  Writer writer;
  writeDomain(writer, kCanonicalBooleanDomain);
  writer.boolean(value);
  return loom::CanonicalSemanticBytes(writer.take());
}

llvm::Expected<bool>
dataflow::decodeCanonicalBoolean(llvm::ArrayRef<std::uint8_t> bytes) {
  Reader reader(bytes);
  if (llvm::Error error = readDomain(reader, kCanonicalBooleanDomain))
    return std::move(error);
  auto value = reader.boolean("canonical boolean");
  if (!value)
    return value.takeError();
  if (llvm::Error error = finish(reader))
    return std::move(error);
  return *value;
}

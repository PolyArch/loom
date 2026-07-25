#include "BuilderInternal.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <system_error>

using namespace loom::adg;

namespace {

using ::fabric::FamilyCapabilityParams;
using ::fabric::FloatFormat;
using ::fabric::IntegerWidth;

::fabric::IntegerWidthSet ordinaryIntegerWidths() {
  return ::fabric::IntegerWidthSet::get({IntegerWidth::I8, IntegerWidth::I16,
                                         IntegerWidth::I32, IntegerWidth::I64});
}

::fabric::IntegerWidthSet allIntegerWidths() {
  return ::fabric::IntegerWidthSet::get({IntegerWidth::I1, IntegerWidth::I8,
                                         IntegerWidth::I16, IntegerWidth::I32,
                                         IntegerWidth::I64});
}

::fabric::FloatFormatSet allFloatFormats() {
  return ::fabric::FloatFormatSet::get({FloatFormat::F16, FloatFormat::BF16,
                                        FloatFormat::F32, FloatFormat::F64});
}

::fabric::IntegerPredicateSet allIntegerPredicates() {
  using Predicate = ::mlir::arith::CmpIPredicate;
  return ::fabric::IntegerPredicateSet::get(
      {Predicate::eq, Predicate::ne, Predicate::slt, Predicate::sle,
       Predicate::sgt, Predicate::sge, Predicate::ult, Predicate::ule,
       Predicate::ugt, Predicate::uge});
}

::fabric::FloatPredicateSet allFloatPredicates() {
  using Predicate = ::mlir::arith::CmpFPredicate;
  return ::fabric::FloatPredicateSet::get(
      {Predicate::AlwaysFalse, Predicate::OEQ, Predicate::OGT, Predicate::OGE,
       Predicate::OLT, Predicate::OLE, Predicate::ONE, Predicate::ORD,
       Predicate::UEQ, Predicate::UGT, Predicate::UGE, Predicate::ULT,
       Predicate::ULE, Predicate::UNE, Predicate::UNO, Predicate::AlwaysTrue});
}

::fabric::IntegerWidthRelation allIntegerCastPairs() {
  constexpr std::array<IntegerWidth, 5> widths = {
      IntegerWidth::I1, IntegerWidth::I8, IntegerWidth::I16, IntegerWidth::I32,
      IntegerWidth::I64};
  ::fabric::IntegerWidthRelation relation;
  for (IntegerWidth source : widths)
    for (IntegerWidth destination : widths)
      relation.insert(source, destination);
  return relation;
}

::fabric::FloatFormatRelation allFloatCastPairs() {
  constexpr std::array<FloatFormat, 4> formats = {
      FloatFormat::F16, FloatFormat::BF16, FloatFormat::F32, FloatFormat::F64};
  ::fabric::FloatFormatRelation relation;
  for (FloatFormat source : formats)
    for (FloatFormat destination : formats)
      if (source != destination)
        relation.insert(source, destination);
  return relation;
}

::fabric::IntegerFloatFormatRelation allIntegerFloatPairs() {
  constexpr std::array<IntegerWidth, 4> integers = {
      IntegerWidth::I8, IntegerWidth::I16, IntegerWidth::I32,
      IntegerWidth::I64};
  constexpr std::array<FloatFormat, 4> formats = {
      FloatFormat::F16, FloatFormat::BF16, FloatFormat::F32, FloatFormat::F64};
  ::fabric::IntegerFloatFormatRelation relation;
  for (IntegerWidth integer : integers)
    for (FloatFormat format : formats)
      relation.insert(integer, format);
  return relation;
}

::fabric::FloatBehaviorProfile strictFloatBehavior() {
  return ::fabric::FloatBehaviorProfile::strictIEEE();
}

::fabric::FloatBehaviorProfile compareMinMaxFloatBehavior() {
  ::fabric::FloatBehaviorProfile behavior =
      ::fabric::FloatBehaviorProfile::strictIEEE();
  behavior.nanBehaviors = ::fabric::FloatNaNBehaviorSet::get(
      {::fabric::FloatNaNBehavior::IEEE,
       ::fabric::FloatNaNBehavior::NumberPreferred});
  return behavior;
}

llvm::Error reject(llvm::StringRef message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

} // namespace

FabricOpCapability loom::adg::detail::builtinOpCapability(
    ::fabric::ImplementationFamilyId family) {
  using Family = ::fabric::ImplementationFamilyId;
  switch (family) {
  case Family::ScalarIntegerAddSub:
  case Family::ScalarIntegerShift:
  case Family::ScalarIntegerMultiply:
    return {family, ::fabric::ScalarIntegerParams{ordinaryIntegerWidths()}};
  case Family::ScalarIntegerLogic:
    return {family, ::fabric::ScalarIntegerParams{allIntegerWidths()}};
  case Family::ScalarIntegerCompareMinMax:
    return {family, ::fabric::ScalarIntegerCompareMinMaxParams{
                        ordinaryIntegerWidths(), allIntegerPredicates()}};
  case Family::ScalarValueSelect:
    return {family, ::fabric::ScalarValueSelectParams{allIntegerWidths(),
                                                      allFloatFormats()}};
  case Family::ScalarIntegerCast:
    return {family, ::fabric::ScalarIntegerCastParams{
                        {allIntegerCastPairs(), std::nullopt}}};
  case Family::ScalarBitReinterpret:
    return {family, ::fabric::ScalarBitReinterpretParams{
                        ordinaryIntegerWidths(), allFloatFormats()}};
  case Family::ScalarFloatSign:
  case Family::ScalarFloatAddSub:
  case Family::ScalarFloatMultiply:
  case Family::ScalarFloatFma:
    return {family, ::fabric::ScalarFloatParams{allFloatFormats(),
                                                strictFloatBehavior()}};
  case Family::ScalarFloatCompareMinMax:
    return {family, ::fabric::ScalarFloatCompareMinMaxParams{
                        allFloatFormats(), compareMinMaxFloatBehavior(),
                        allFloatPredicates()}};
  case Family::ScalarFloatWidthCast:
    return {family, ::fabric::ScalarFloatWidthCastParams{
                        allFloatCastPairs(), strictFloatBehavior()}};
  case Family::ScalarIntegerToFloat:
  case Family::ScalarFloatToInteger:
    return {family, ::fabric::ScalarIntegerFloatConversionParams{
                        allIntegerFloatPairs(), strictFloatBehavior()}};
  case Family::LoopCarry:
  case Family::LoopInvariant:
  case Family::LoopGate:
    return {family, ::fabric::TokenPlaneParams{}};
  case Family::LoopStream:
    llvm_unreachable("LoopStream requires an explicit fixed step kind");
  }
  llvm_unreachable("unregistered builtin implementation family");
}

FabricOpCapability loom::adg::detail::builtinIndexCastCapability(
    ::fabric::ResolvedIndexWidth resolvedIndexWidth) {
  return {::fabric::ImplementationFamilyId::ScalarIntegerCast,
          ::fabric::ScalarIntegerCastParams{
              {allIntegerCastPairs(), resolvedIndexWidth}}};
}

FabricOpCapability loom::adg::detail::loopStreamCapability(
    ::dataflow::StreamStepKind stepKind,
    std::initializer_list<::mlir::arith::CmpIPredicate> predicates) {
  ::fabric::IntegerPredicateSet predicateSet;
  for (::mlir::arith::CmpIPredicate predicate : predicates)
    predicateSet.insert(predicate);
  return {::fabric::ImplementationFamilyId::LoopStream,
          ::fabric::LoopStreamParams{ordinaryIntegerWidths(), stepKind,
                                     predicateSet}};
}

llvm::Error loom::adg::detail::validateFabricOpCapability(
    const FabricOpCapability &capability) {
  std::uint32_t familyOrdinal = static_cast<std::uint32_t>(capability.family);
  if (familyOrdinal >= ::fabric::implementationFamilyCount())
    return reject("ADG fabric.op implementation family is not registered");

  const ::fabric::ImplementationFamilyDescriptor &descriptor =
      ::fabric::implementationFamily(capability.family);
  if (::fabric::capabilityParamsSchema(capability.params) !=
      descriptor.capabilityParamsSchema)
    return reject("ADG fabric.op capability schema does not match its "
                  "implementation family");

  if (const auto *stream =
          std::get_if<::fabric::LoopStreamParams>(&capability.params)) {
    if (!::dataflow::symbolizeStreamStepKind(
            static_cast<std::uint32_t>(stream->fixedStepKind)))
      return reject("ADG stream capability has invalid fixed step kind");
  }
  if (const auto *integerCast =
          std::get_if<::fabric::ScalarIntegerCastParams>(&capability.params)) {
    if (integerCast->relation.resolvedIndexWidth) {
      ::fabric::ResolvedIndexWidth width =
          *integerCast->relation.resolvedIndexWidth;
      if (width != ::fabric::ResolvedIndexWidth::I32 &&
          width != ::fabric::ResolvedIndexWidth::I64)
        return reject("ADG integer cast capability has invalid resolved index "
                      "width");
    }
  }

  ::mlir::MLIRContext context;
  ::mlir::DictionaryAttr encoded =
      ::fabric::getFamilyCapabilityParamsAttr(&context, capability.params);
  llvm::Expected<::fabric::FamilyCapabilityParams> decoded =
      ::fabric::parseFamilyCapabilityParams(capability.family, encoded);
  if (!decoded)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "ADG fabric.op capability is invalid: %s",
                                   llvm::toString(decoded.takeError()).c_str());
  if (::fabric::getFamilyCapabilityParamsAttr(&context, *decoded) != encoded)
    return reject("ADG fabric.op capability is not canonical");
  return llvm::Error::success();
}

void loom::adg::detail::printFabricOpAttrs(
    llvm::raw_ostream &os, ::mlir::MLIRContext &context,
    const FabricOpCapability &capability) {
  os << "{implementation_family = #fabric.implementation_family<"
     << ::fabric::implementationFamilyKeyword(capability.family)
     << ">, hw_params = ";
  ::fabric::getFamilyCapabilityParamsAttr(&context, capability.params)
      .print(os);
  os << '}';
}

std::string
loom::adg::detail::fabricOpAttrsText(const FabricOpCapability &capability) {
  ::mlir::MLIRContext context;
  std::string text;
  llvm::raw_string_ostream os(text);
  printFabricOpAttrs(os, context, capability);
  return os.str();
}

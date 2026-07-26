#include "ADG/FuLibrary.h"

#include "Fabric/IR/ImplementationFamily.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace loom::adg {
namespace {

using ::dataflow::OperationSchemaId;
using ::fabric::FamilyCapabilityParams;
using ::fabric::ImplementationFamilyId;

struct SelectableResource {
  ImplementationFamilyId family;
  FamilyCapabilityParams parameters;
  std::vector<std::uint32_t> inputRoles;
};

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "adg_fu_library_invalid: " + message);
}

::fabric::IntegerWidthSet ordinaryIntegerWidths() {
  return ::fabric::IntegerWidthSet::get(
      {::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16,
       ::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I64});
}

::fabric::IntegerWidthSet logicIntegerWidths() {
  return ::fabric::IntegerWidthSet::get(
      {::fabric::IntegerWidth::I1, ::fabric::IntegerWidth::I8,
       ::fabric::IntegerWidth::I16, ::fabric::IntegerWidth::I32,
       ::fabric::IntegerWidth::I64});
}

::fabric::FloatFormatSet floatFormats() {
  return ::fabric::FloatFormatSet::get(
      {::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
       ::fabric::FloatFormat::F32, ::fabric::FloatFormat::F64});
}

::fabric::IntegerPredicateSet integerPredicates() {
  using P = mlir::arith::CmpIPredicate;
  return ::fabric::IntegerPredicateSet::get({P::eq, P::ne, P::slt, P::sle,
                                             P::sgt, P::sge, P::ult, P::ule,
                                             P::ugt, P::uge});
}

::fabric::FloatPredicateSet floatPredicates() {
  using P = mlir::arith::CmpFPredicate;
  return ::fabric::FloatPredicateSet::get(
      {P::AlwaysFalse, P::OEQ, P::OGT, P::OGE, P::OLT, P::OLE, P::ONE, P::ORD,
       P::UEQ, P::UGT, P::UGE, P::ULT, P::ULE, P::UNE, P::UNO, P::AlwaysTrue});
}

::fabric::FloatBehaviorProfile floatCompareBehavior() {
  ::fabric::FloatBehaviorProfile behavior =
      ::fabric::FloatBehaviorProfile::strictIEEE();
  behavior.nanBehaviors = ::fabric::FloatNaNBehaviorSet::get(
      {::fabric::FloatNaNBehavior::IEEE,
       ::fabric::FloatNaNBehavior::NumberPreferred});
  return behavior;
}

::fabric::IntegerCastRelation integerCastRelation() {
  ::fabric::IntegerWidthRelation relation;
  constexpr std::array<::fabric::IntegerWidth, 5> widths = {
      ::fabric::IntegerWidth::I1, ::fabric::IntegerWidth::I8,
      ::fabric::IntegerWidth::I16, ::fabric::IntegerWidth::I32,
      ::fabric::IntegerWidth::I64};
  for (::fabric::IntegerWidth source : widths)
    for (::fabric::IntegerWidth destination : widths)
      if (source != destination)
        relation.insert(source, destination);
  return {relation, ::fabric::ResolvedIndexWidth::I64};
}

::fabric::FloatFormatRelation floatCastRelation() {
  ::fabric::FloatFormatRelation relation;
  constexpr std::array<::fabric::FloatFormat, 4> formats = {
      ::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
      ::fabric::FloatFormat::F32, ::fabric::FloatFormat::F64};
  for (::fabric::FloatFormat source : formats)
    for (::fabric::FloatFormat destination : formats)
      if (source != destination)
        relation.insert(source, destination);
  return relation;
}

::fabric::IntegerFloatFormatRelation integerFloatRelation() {
  ::fabric::IntegerFloatFormatRelation relation;
  constexpr std::array<::fabric::IntegerWidth, 4> widths = {
      ::fabric::IntegerWidth::I8, ::fabric::IntegerWidth::I16,
      ::fabric::IntegerWidth::I32, ::fabric::IntegerWidth::I64};
  constexpr std::array<::fabric::FloatFormat, 4> formats = {
      ::fabric::FloatFormat::F16, ::fabric::FloatFormat::BF16,
      ::fabric::FloatFormat::F32, ::fabric::FloatFormat::F64};
  for (::fabric::IntegerWidth width : widths)
    for (::fabric::FloatFormat format : formats)
      relation.insert(width, format);
  return relation;
}

std::vector<OperationSchemaId> familyMembers(ImplementationFamilyId family) {
  llvm::ArrayRef<OperationSchemaId> members =
      ::fabric::implementationFamily(family).admittedSchemas;
  return {members.begin(), members.end()};
}

llvm::Error addSelectableFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                            std::vector<PortType> innerInputTypes,
                            PortType innerOutputType, PortType outerOutputType,
                            llvm::ArrayRef<SelectableResource> resources) {
  if (inputs.size() != innerInputTypes.size())
    return invalid("helper input count does not match its boundary");
  if (resources.size() < 2)
    return invalid("selectable FU requires at least two physical resources");

  auto fu = pe.addFu(
      inputs, FuSpec{std::move(innerInputTypes), {std::move(outerOutputType)}});
  if (!fu)
    return fu.takeError();

  std::vector<std::uint32_t> useCounts(inputs.size(), 0);
  for (const SelectableResource &resource : resources)
    for (std::uint32_t role : resource.inputRoles) {
      if (role >= useCounts.size())
        return invalid("physical resource names an unknown FU input role");
      ++useCounts[role];
    }

  std::vector<std::vector<FuValue>> routed(inputs.size());
  for (std::size_t role = 0; role != inputs.size(); ++role) {
    auto input = fu->input(role);
    if (!input)
      return input.takeError();
    if (useCounts[role] == 0)
      return invalid("FU boundary contains an unused input role");
    if (useCounts[role] == 1) {
      routed[role].push_back(*input);
      continue;
    }
    auto demux = fu->addDemux(*input, useCounts[role]);
    if (!demux)
      return demux.takeError();
    routed[role] = std::move(*demux);
  }

  std::vector<std::uint32_t> nextRoute(inputs.size(), 0);
  std::vector<FuValue> results;
  results.reserve(resources.size());
  for (const SelectableResource &resource : resources) {
    llvm::SmallVector<FuValue, 4> operationInputs;
    for (std::uint32_t role : resource.inputRoles)
      operationInputs.push_back(routed[role][nextRoute[role]++]);
    auto operationResults = fu->addOperation(
        operationInputs, OperationCapabilitySpec{resource.family,
                                                 resource.parameters,
                                                 familyMembers(resource.family),
                                                 {innerOutputType}});
    if (!operationResults)
      return operationResults.takeError();
    if (operationResults->size() != 1)
      return invalid("catalog compute resource must have one physical result");
    results.push_back(operationResults->front());
  }

  auto selected = fu->addMux(results);
  if (!selected)
    return selected.takeError();
  return fu->close({*selected});
}

SelectableResource scalarInteger(ImplementationFamilyId family,
                                 std::vector<std::uint32_t> inputs,
                                 bool logic = false) {
  return {family,
          ::fabric::ScalarIntegerParams{logic ? logicIntegerWidths()
                                              : ordinaryIntegerWidths()},
          std::move(inputs)};
}

SelectableResource scalarFloat(ImplementationFamilyId family,
                               std::vector<std::uint32_t> inputs,
                               bool comparisons = false) {
  if (comparisons)
    return {family,
            ::fabric::ScalarFloatCompareMinMaxParams{
                floatFormats(), floatCompareBehavior(), floatPredicates()},
            std::move(inputs)};
  return {family,
          ::fabric::ScalarFloatParams{
              floatFormats(), ::fabric::FloatBehaviorProfile::strictIEEE()},
          std::move(inputs)};
}

} // namespace

llvm::Error addCoreAluFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs) {
  if (inputs.size() != 3)
    return invalid("CoreAluFu requires data0, data1, and condition inputs");
  auto bits64 = PortType::bits(64);
  if (!bits64)
    return bits64.takeError();
  auto bits1 = PortType::bits(1);
  if (!bits1)
    return bits1.takeError();
  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();

  std::vector<SelectableResource> resources;
  resources.push_back(
      scalarInteger(ImplementationFamilyId::ScalarIntegerAddSub, {0, 1}));
  resources.push_back(
      scalarInteger(ImplementationFamilyId::ScalarIntegerLogic, {0, 1}, true));
  resources.push_back(
      scalarInteger(ImplementationFamilyId::ScalarIntegerShift, {0, 1}));
  resources.push_back({ImplementationFamilyId::ScalarIntegerCompareMinMax,
                       ::fabric::ScalarIntegerCompareMinMaxParams{
                           ordinaryIntegerWidths(), integerPredicates()},
                       {0, 1}});
  resources.push_back(
      {ImplementationFamilyId::ScalarValueSelect,
       ::fabric::ScalarValueSelectParams{logicIntegerWidths(), floatFormats()},
       {2, 0, 1}});
  resources.push_back({ImplementationFamilyId::ScalarIntegerCast,
                       ::fabric::ScalarIntegerCastParams{integerCastRelation()},
                       {0}});
  resources.push_back({ImplementationFamilyId::ScalarBitReinterpret,
                       ::fabric::ScalarBitReinterpretParams{
                           ordinaryIntegerWidths(), floatFormats()},
                       {0}});
  resources.push_back(
      scalarFloat(ImplementationFamilyId::ScalarFloatSign, {0}));
  resources.push_back(
      scalarFloat(ImplementationFamilyId::ScalarFloatAddSub, {0, 1}));
  resources.push_back(scalarFloat(
      ImplementationFamilyId::ScalarFloatCompareMinMax, {0, 1}, true));
  resources.push_back(
      {ImplementationFamilyId::ScalarFloatWidthCast,
       ::fabric::ScalarFloatWidthCastParams{
           floatCastRelation(), ::fabric::FloatBehaviorProfile::strictIEEE()},
       {0}});
  resources.push_back({ImplementationFamilyId::ScalarIntegerToFloat,
                       ::fabric::ScalarIntegerFloatConversionParams{
                           integerFloatRelation(),
                           ::fabric::FloatBehaviorProfile::strictIEEE()},
                       {0}});
  resources.push_back({ImplementationFamilyId::ScalarFloatToInteger,
                       ::fabric::ScalarIntegerFloatConversionParams{
                           integerFloatRelation(),
                           ::fabric::FloatBehaviorProfile::strictIEEE()},
                       {0}});
  return addSelectableFu(pe, inputs, {*bits64, *bits64, *bits1}, *bits64,
                         *bits128, resources);
}

llvm::Error addVectorComputeFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs) {
  if (inputs.size() != 4)
    return invalid(
        "VectorComputeFu requires data0, data1, data2, and condition inputs");
  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();
  const auto integer = ordinaryIntegerWidths();
  const auto floating = floatFormats();
  const auto strict = ::fabric::FloatBehaviorProfile::strictIEEE();
  std::vector<SelectableResource> resources = {
      {ImplementationFamilyId::FixedVectorIntegerAddSub,
       ::fabric::FixedVectorIntegerParams{integer, 128},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorIntegerLogic,
       ::fabric::FixedVectorIntegerParams{logicIntegerWidths(), 128},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorIntegerShift,
       ::fabric::FixedVectorIntegerParams{integer, 128},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorIntegerCompareMinMax,
       ::fabric::FixedVectorIntegerCompareMinMaxParams{
           integer, integerPredicates(), 128},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorValueSelect,
       ::fabric::FixedVectorValueSelectParams{logicIntegerWidths(), floating,
                                              128},
       {3, 0, 1}},
      {ImplementationFamilyId::FixedVectorIntegerMultiply,
       ::fabric::FixedVectorIntegerParams{integer, 128},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorFloatSign,
       ::fabric::FixedVectorFloatParams{floating, strict, 128},
       {0}},
      {ImplementationFamilyId::FixedVectorFloatAddSub,
       ::fabric::FixedVectorFloatParams{floating, strict, 128},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorFloatCompareMinMax,
       ::fabric::FixedVectorFloatCompareMinMaxParams{
           floating, floatCompareBehavior(), floatPredicates(), 128},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorFloatMultiply,
       ::fabric::FixedVectorFloatParams{floating, strict, 128},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorFloatFma,
       ::fabric::FixedVectorFloatParams{floating, strict, 128},
       {0, 1, 2}},
  };
  return addSelectableFu(pe, inputs, {*bits128, *bits128, *bits128, *bits128},
                         *bits128, *bits128, resources);
}

llvm::Error addSpecialMathFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs) {
  if (inputs.size() != 2)
    return invalid("SpecialMathFu requires two data inputs");
  auto bits64 = PortType::bits(64);
  if (!bits64)
    return bits64.takeError();
  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();

  std::vector<SelectableResource> resources;
  resources.push_back(
      scalarInteger(ImplementationFamilyId::ScalarSignedIntegerDivRem, {0, 1}));
  resources.push_back(scalarInteger(
      ImplementationFamilyId::ScalarUnsignedIntegerDivRem, {0, 1}));
  resources.push_back(
      scalarFloat(ImplementationFamilyId::ScalarFloatDivide, {0, 1}));
  resources.push_back(
      scalarFloat(ImplementationFamilyId::ScalarFloatRemainder, {0, 1}));
  constexpr std::array<ImplementationFamilyId, 21> unaryFamilies = {
      ImplementationFamilyId::ScalarMathSin,
      ImplementationFamilyId::ScalarMathCos,
      ImplementationFamilyId::ScalarMathTan,
      ImplementationFamilyId::ScalarMathSinh,
      ImplementationFamilyId::ScalarMathCosh,
      ImplementationFamilyId::ScalarMathTanh,
      ImplementationFamilyId::ScalarMathExp,
      ImplementationFamilyId::ScalarMathExp2,
      ImplementationFamilyId::ScalarMathExpM1,
      ImplementationFamilyId::ScalarMathLog,
      ImplementationFamilyId::ScalarMathLog2,
      ImplementationFamilyId::ScalarMathLog10,
      ImplementationFamilyId::ScalarMathLog1p,
      ImplementationFamilyId::ScalarMathFloor,
      ImplementationFamilyId::ScalarMathCeil,
      ImplementationFamilyId::ScalarMathRound,
      ImplementationFamilyId::ScalarMathTrunc,
      ImplementationFamilyId::ScalarMathRoundEven,
      ImplementationFamilyId::ScalarMathSqrt,
      ImplementationFamilyId::ScalarMathRsqrt,
      ImplementationFamilyId::ScalarMathErf};
  for (ImplementationFamilyId family : unaryFamilies)
    resources.push_back(scalarFloat(family, {0}));
  return addSelectableFu(pe, inputs, {*bits64, *bits64}, *bits64, *bits128,
                         resources);
}

} // namespace loom::adg

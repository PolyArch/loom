#include "ADG/FuLibrary.h"

#include "Fabric/IR/ImplementationFamily.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <array>
#include <cstdint>
#include <optional>
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

struct RoutedResource {
  ImplementationFamilyId family;
  FamilyCapabilityParams parameters;
  std::vector<std::uint32_t> inputRoles;
  std::vector<PortType> resultTypes;
  std::vector<std::uint32_t> outputRoles;
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

llvm::Error addRoutedFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                        std::vector<PortType> innerInputTypes,
                        std::vector<PortType> outerOutputTypes,
                        llvm::ArrayRef<RoutedResource> resources) {
  if (inputs.size() != innerInputTypes.size())
    return invalid("helper input count does not match its boundary");
  if (resources.empty() || outerOutputTypes.empty())
    return invalid("routed FU requires physical resources and outputs");

  auto fu =
      pe.addFu(inputs, FuSpec{std::move(innerInputTypes), outerOutputTypes});
  if (!fu)
    return fu.takeError();

  std::vector<std::uint32_t> useCounts(inputs.size(), 0);
  std::vector<std::uint32_t> producerCounts(outerOutputTypes.size(), 0);
  for (const RoutedResource &resource : resources) {
    if (resource.resultTypes.size() != resource.outputRoles.size())
      return invalid("routed resource result roles do not match its results");
    for (std::uint32_t role : resource.inputRoles) {
      if (role >= useCounts.size())
        return invalid("physical resource names an unknown FU input role");
      ++useCounts[role];
    }
    for (std::uint32_t role : resource.outputRoles) {
      if (role >= producerCounts.size())
        return invalid("physical resource names an unknown FU output role");
      ++producerCounts[role];
    }
  }

  struct RoutedInput {
    std::optional<FuNode> selector;
    std::vector<FuValue> values;
  };
  std::vector<RoutedInput> routed(inputs.size());
  for (std::size_t role = 0; role != inputs.size(); ++role) {
    auto input = fu->input(role);
    if (!input)
      return input.takeError();
    if (useCounts[role] == 0)
      return invalid("FU boundary contains an unused input role");
    if (useCounts[role] == 1) {
      routed[role].values.push_back(*input);
      continue;
    }
    auto demux = fu->addDemux(*input, useCounts[role]);
    if (!demux)
      return demux.takeError();
    routed[role].selector = *demux;
    for (std::uint32_t output = 0; output < useCounts[role]; ++output) {
      auto value = demux->output(output);
      if (!value)
        return value.takeError();
      routed[role].values.push_back(*value);
    }
  }

  struct BuiltResource {
    FuNode operation;
    std::vector<FuRouteSelection> routes;
  };
  struct OutputSource {
    FuValue value;
    std::size_t resourceOrdinal = 0;
  };
  std::vector<std::uint32_t> nextRoute(inputs.size(), 0);
  std::vector<std::vector<OutputSource>> outputSources(outerOutputTypes.size());
  std::vector<BuiltResource> builtResources;
  builtResources.reserve(resources.size());
  for (auto [resourceOrdinal, resource] : llvm::enumerate(resources)) {
    llvm::SmallVector<FuValue, 8> operationInputs;
    std::vector<FuRouteSelection> routes;
    for (std::uint32_t role : resource.inputRoles) {
      const std::uint32_t route = nextRoute[role]++;
      operationInputs.push_back(routed[role].values[route]);
      if (routed[role].selector)
        routes.push_back({*routed[role].selector, route});
    }
    auto operation = fu->addOperation(
        operationInputs,
        OperationCapabilitySpec{resource.family, resource.parameters,
                                familyMembers(resource.family),
                                resource.resultTypes});
    if (!operation)
      return operation.takeError();
    for (auto [resultOrdinal, role] : llvm::enumerate(resource.outputRoles)) {
      auto result = operation->output(resultOrdinal);
      if (!result)
        return result.takeError();
      outputSources[role].push_back({*result, resourceOrdinal});
    }
    builtResources.push_back({*operation, std::move(routes)});
  }

  std::vector<FuValue> outputs;
  outputs.reserve(outputSources.size());
  for (auto [role, sources] : llvm::enumerate(outputSources)) {
    if (sources.empty() || producerCounts[role] != sources.size())
      return invalid("FU output role has no complete physical producer set");
    if (sources.size() == 1) {
      outputs.push_back(sources.front().value);
      continue;
    }
    llvm::SmallVector<FuValue, 8> values;
    for (const OutputSource &source : sources)
      values.push_back(source.value);
    auto mux = fu->addMux(values);
    if (!mux)
      return mux.takeError();
    auto selected = mux->output(0);
    if (!selected)
      return selected.takeError();
    outputs.push_back(*selected);
    for (auto [route, source] : llvm::enumerate(sources))
      builtResources[source.resourceOrdinal].routes.push_back(
          {*mux, static_cast<std::uint32_t>(route)});
  }

  for (BuiltResource &resource : builtResources)
    if (llvm::Error error = fu->addCapabilityTemplate(
            FuCapabilityTemplateSpec{{resource.operation}, resource.routes}))
      return error;
  return fu->close(outputs);
}

llvm::Error addSelectableFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                            std::vector<PortType> innerInputTypes,
                            PortType innerOutputType, PortType outerOutputType,
                            llvm::ArrayRef<SelectableResource> resources) {
  if (inputs.size() != innerInputTypes.size())
    return invalid("helper input count does not match its boundary");
  if (resources.size() < 2)
    return invalid("selectable FU requires at least two physical resources");
  std::vector<RoutedResource> routed;
  routed.reserve(resources.size());
  for (const SelectableResource &resource : resources)
    routed.push_back({resource.family,
                      resource.parameters,
                      resource.inputRoles,
                      {innerOutputType},
                      {0}});
  return addRoutedFu(pe, inputs, std::move(innerInputTypes),
                     {std::move(outerOutputType)}, routed);
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

llvm::Error addVectorAdapterFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs) {
  if (inputs.size() != 3)
    return invalid("VectorAdapterFu requires data, mask, and phase inputs");
  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();
  auto bits1 = PortType::bits(1);
  if (!bits1)
    return bits1.takeError();
  const ::fabric::FixedVectorAdapterParams parameters{logicIntegerWidths(),
                                                      floatFormats(), 128};
  std::vector<RoutedResource> resources = {
      {ImplementationFamilyId::FixedVectorPack,
       parameters,
       {0},
       {*bits128},
       {0}},
      {ImplementationFamilyId::FixedVectorUnpack,
       parameters,
       {0},
       {*bits128},
       {0}},
      {ImplementationFamilyId::FixedVectorParallelize,
       parameters,
       {0, 2},
       {*bits128, *bits128, *bits1},
       {0, 1, 2}},
      {ImplementationFamilyId::FixedVectorSerialize,
       parameters,
       {0, 1, 2},
       {*bits128, *bits1},
       {0, 2}},
  };
  return addRoutedFu(pe, inputs, {*bits128, *bits128, *bits1},
                     {*bits128, *bits128, *bits128}, resources);
}

llvm::Error addTokenControlFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs) {
  if (inputs.size() != 5)
    return invalid(
        "TokenControlFu requires selector/control and four payload inputs");
  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();
  auto bits64 = PortType::bits(64);
  if (!bits64)
    return bits64.takeError();
  const ::fabric::RoutedTokenParams routed{128, 4};
  std::vector<RoutedResource> resources = {
      {ImplementationFamilyId::TokenConstant,
       ::fabric::PayloadCapacityParams{128},
       {0},
       {*bits128},
       {0}},
      {ImplementationFamilyId::TokenSync,
       routed,
       {1, 2, 3, 4},
       {*bits128, *bits128, *bits128, *bits128},
       {0, 1, 2, 3}},
      {ImplementationFamilyId::TokenMux,
       routed,
       {0, 1, 2, 3, 4},
       {*bits128},
       {0}},
      {ImplementationFamilyId::TokenDemux,
       routed,
       {0, 1},
       {*bits128, *bits128, *bits128, *bits128},
       {0, 1, 2, 3}},
  };
  return addRoutedFu(pe, inputs,
                     {*bits64, *bits128, *bits128, *bits128, *bits128},
                     {*bits128, *bits128, *bits128, *bits128}, resources);
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

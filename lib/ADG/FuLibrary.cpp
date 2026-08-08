#include "ADG/FuLibrary.h"

#include "CatalogCapabilities.h"

#include "Fabric/IR/ImplementationFamily.h"
#include "Fabric/IR/OperationResourceContract.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

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

const ::fabric::ResourceContract &
loopControlResourceContract(ImplementationFamilyId family) {
  switch (family) {
  case ImplementationFamilyId::LoopStream:
    return ::fabric::loopStreamOperationResourceContract();
  case ImplementationFamilyId::LoopCarry:
    return ::fabric::loopCarryOperationResourceContract();
  case ImplementationFamilyId::LoopInvariant:
    return ::fabric::loopInvariantOperationResourceContract();
  case ImplementationFamilyId::LoopGate:
    return ::fabric::loopGateOperationResourceContract();
  default:
    llvm_unreachable("non-loop family requested a loop-control contract");
  }
}

struct SelectableResource {
  SelectableResource(ImplementationFamilyId family,
                     FamilyCapabilityParams parameters,
                     std::vector<std::uint32_t> inputRoles,
                     std::vector<OperationSchemaId> enabledOperations = {})
      : family(family), parameters(std::move(parameters)),
        inputRoles(std::move(inputRoles)),
        enabledOperations(std::move(enabledOperations)) {}

  ImplementationFamilyId family;
  FamilyCapabilityParams parameters;
  std::vector<std::uint32_t> inputRoles;
  std::vector<OperationSchemaId> enabledOperations;
};

struct RoutedResource {
  RoutedResource(ImplementationFamilyId family,
                 FamilyCapabilityParams parameters,
                 std::vector<std::uint32_t> inputRoles,
                 std::vector<PortType> resultTypes,
                 std::vector<std::uint32_t> outputRoles,
                 std::vector<OperationSchemaId> enabledOperations = {})
      : family(family), parameters(std::move(parameters)),
        inputRoles(std::move(inputRoles)), resultTypes(std::move(resultTypes)),
        outputRoles(std::move(outputRoles)),
        enabledOperations(std::move(enabledOperations)) {}

  ImplementationFamilyId family;
  FamilyCapabilityParams parameters;
  std::vector<std::uint32_t> inputRoles;
  std::vector<PortType> resultTypes;
  std::vector<std::uint32_t> outputRoles;
  std::vector<OperationSchemaId> enabledOperations;
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

::fabric::IntegerCastRelation
integerCastRelation(::fabric::ResolvedIndexWidthSet resolvedIndexWidths) {
  ::fabric::IntegerWidthRelation relation;
  constexpr std::array<::fabric::IntegerWidth, 5> widths = {
      ::fabric::IntegerWidth::I1, ::fabric::IntegerWidth::I8,
      ::fabric::IntegerWidth::I16, ::fabric::IntegerWidth::I32,
      ::fabric::IntegerWidth::I64};
  for (::fabric::IntegerWidth source : widths) {
    for (::fabric::IntegerWidth destination : widths) {
      const bool resolvedIndexIdentity =
          source == destination &&
          ((source == ::fabric::IntegerWidth::I32 &&
            resolvedIndexWidths.contains(::fabric::ResolvedIndexWidth::I32)) ||
           (source == ::fabric::IntegerWidth::I64 &&
            resolvedIndexWidths.contains(::fabric::ResolvedIndexWidth::I64)));
      if (source != destination || resolvedIndexIdentity)
        relation.insert(source, destination);
    }
  }
  return {relation, resolvedIndexWidths};
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

std::vector<OperationSchemaId> familyMembers(ImplementationFamilyId family,
                                             bool includePointer = false) {
  llvm::ArrayRef<OperationSchemaId> members =
      ::fabric::implementationFamily(family).admittedSchemas;
  std::vector<OperationSchemaId> enabled;
  for (OperationSchemaId member : members) {
    if (member == OperationSchemaId::LLVMGetElementPtr && !includePointer)
      continue;
    enabled.push_back(member);
  }
  return enabled;
}

SelectableResource
pointerCapableIntegerAddSub(std::vector<std::uint32_t> inputs) {
  return {ImplementationFamilyId::ScalarIntegerAddSub,
          ::fabric::ScalarIntegerParams{
              ordinaryIntegerWidths(),
              ::loom::adg::detail::catalogPointerFormats()},
          std::move(inputs),
          familyMembers(ImplementationFamilyId::ScalarIntegerAddSub, true)};
}

llvm::Expected<std::vector<FuValue>> nodeOutputs(const FuNode &node,
                                                 std::uint32_t count) {
  std::vector<FuValue> outputs;
  outputs.reserve(count);
  for (std::uint32_t ordinal = 0; ordinal != count; ++ordinal) {
    auto output = node.output(ordinal);
    if (!output)
      return output.takeError();
    outputs.push_back(*output);
  }
  return outputs;
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
        OperationCapabilitySpec{
            resource.family, resource.parameters,
            resource.enabledOperations.empty() ? familyMembers(resource.family)
                                               : resource.enabledOperations,
            resource.resultTypes,
            ::fabric::oneCycleElasticOperationResourceContract()});
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
                      {0},
                      resource.enabledOperations});
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

SelectableResource scalarSpecialMath(ImplementationFamilyId family,
                                     std::vector<std::uint32_t> inputs) {
  return {family,
          ::fabric::ScalarSpecialMathParams{
              floatFormats(), ::fabric::FloatBehaviorProfile::strictIEEE(),
              SpecialMathAccuracyTier::CorrectlyRounded},
          std::move(inputs)};
}

} // namespace

llvm::Error addCoreAluFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                         ::fabric::ResolvedIndexWidthSet resolvedIndexWidths) {
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
  resources.push_back(pointerCapableIntegerAddSub({0, 1}));
  resources.push_back(scalarInteger(
      ImplementationFamilyId::ScalarIntegerSaturatingAddSub, {0, 1}));
  resources.push_back(
      scalarInteger(ImplementationFamilyId::ScalarIntegerCountZeros, {0}));
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
                       ::fabric::ScalarIntegerCastParams{
                           integerCastRelation(resolvedIndexWidths)},
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

llvm::Error addMacFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs) {
  if (inputs.size() != 4)
    return invalid("MacFu requires data0, data1, data2, and phase inputs");
  auto bits64 = PortType::bits(64);
  if (!bits64)
    return bits64.takeError();
  auto bits1 = PortType::bits(1);
  if (!bits1)
    return bits1.takeError();
  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();

  auto fu =
      pe.addFu(inputs, FuSpec{{*bits64, *bits64, *bits64, *bits1}, {*bits128}});
  if (!fu)
    return fu.takeError();
  llvm::SmallVector<FuValue, 4> boundary;
  for (std::size_t ordinal = 0; ordinal != 4; ++ordinal) {
    auto value = fu->input(ordinal);
    if (!value)
      return value.takeError();
    boundary.push_back(*value);
  }

  auto d0 = fu->addDemux(boundary[0], 3);
  if (!d0)
    return d0.takeError();
  auto d1 = fu->addDemux(boundary[1], 3);
  if (!d1)
    return d1.takeError();
  auto d2 = fu->addDemux(boundary[2], 4);
  if (!d2)
    return d2.takeError();
  auto d0Values = nodeOutputs(*d0, 3);
  if (!d0Values)
    return d0Values.takeError();
  auto d1Values = nodeOutputs(*d1, 3);
  if (!d1Values)
    return d1Values.takeError();
  auto d2Values = nodeOutputs(*d2, 4);
  if (!d2Values)
    return d2Values.takeError();
  auto carryBackedge = fu->createBackedge(*bits64);
  if (!carryBackedge)
    return carryBackedge.takeError();
  auto carryFeedback = fu->addDemux(carryBackedge->value(), 3);
  if (!carryFeedback)
    return carryFeedback.takeError();
  auto carryFeedbackValues = nodeOutputs(*carryFeedback, 3);
  if (!carryFeedbackValues)
    return carryFeedbackValues.takeError();

  auto integerAddRhs = fu->addMux({(*d2Values)[1], (*carryFeedbackValues)[0]});
  if (!integerAddRhs)
    return integerAddRhs.takeError();
  auto floatAddRhs = fu->addMux({(*d2Values)[2], (*carryFeedbackValues)[1]});
  if (!floatAddRhs)
    return floatAddRhs.takeError();
  auto fmaAddend = fu->addMux({(*d2Values)[0], (*carryFeedbackValues)[2]});
  if (!fmaAddend)
    return fmaAddend.takeError();
  auto integerAddRhsValue = integerAddRhs->output(0);
  if (!integerAddRhsValue)
    return integerAddRhsValue.takeError();
  auto floatAddRhsValue = floatAddRhs->output(0);
  if (!floatAddRhsValue)
    return floatAddRhsValue.takeError();
  auto fmaAddendValue = fmaAddend->output(0);
  if (!fmaAddendValue)
    return fmaAddendValue.takeError();

  const auto integerParams =
      ::fabric::ScalarIntegerParams{ordinaryIntegerWidths()};
  const auto floatParams = ::fabric::ScalarFloatParams{
      floatFormats(), ::fabric::FloatBehaviorProfile::strictIEEE()};
  const auto operationSpec = [&](ImplementationFamilyId family,
                                 const FamilyCapabilityParams &parameters) {
    return OperationCapabilitySpec{
        family,
        parameters,
        familyMembers(family),
        {*bits64},
        ::fabric::oneCycleElasticOperationResourceContract()};
  };

  auto integerMultiply = fu->addOperation(
      {(*d0Values)[0], (*d1Values)[0]},
      operationSpec(ImplementationFamilyId::ScalarIntegerMultiply,
                    integerParams));
  if (!integerMultiply)
    return integerMultiply.takeError();
  auto floatMultiply = fu->addOperation(
      {(*d0Values)[1], (*d1Values)[1]},
      operationSpec(ImplementationFamilyId::ScalarFloatMultiply, floatParams));
  if (!floatMultiply)
    return floatMultiply.takeError();

  auto integerMultiplyValue = integerMultiply->output(0);
  if (!integerMultiplyValue)
    return integerMultiplyValue.takeError();
  auto floatMultiplyValue = floatMultiply->output(0);
  if (!floatMultiplyValue)
    return floatMultiplyValue.takeError();
  auto integerMultiplyRoutes = fu->addDemux(*integerMultiplyValue, 2);
  if (!integerMultiplyRoutes)
    return integerMultiplyRoutes.takeError();
  auto floatMultiplyRoutes = fu->addDemux(*floatMultiplyValue, 2);
  if (!floatMultiplyRoutes)
    return floatMultiplyRoutes.takeError();
  auto integerMultiplyRouteValues = nodeOutputs(*integerMultiplyRoutes, 2);
  if (!integerMultiplyRouteValues)
    return integerMultiplyRouteValues.takeError();
  auto floatMultiplyRouteValues = nodeOutputs(*floatMultiplyRoutes, 2);
  if (!floatMultiplyRouteValues)
    return floatMultiplyRouteValues.takeError();

  auto integerAdd = fu->addOperation(
      {(*integerMultiplyRouteValues)[1], *integerAddRhsValue},
      operationSpec(ImplementationFamilyId::ScalarIntegerAddSub,
                    integerParams));
  if (!integerAdd)
    return integerAdd.takeError();
  auto floatAdd = fu->addOperation(
      {(*floatMultiplyRouteValues)[1], *floatAddRhsValue},
      operationSpec(ImplementationFamilyId::ScalarFloatAddSub, floatParams));
  if (!floatAdd)
    return floatAdd.takeError();
  auto fusedFma = fu->addOperation(
      {(*d0Values)[2], (*d1Values)[2], *fmaAddendValue},
      operationSpec(ImplementationFamilyId::ScalarFloatFma, floatParams));
  if (!fusedFma)
    return fusedFma.takeError();

  auto integerAddValue = integerAdd->output(0);
  if (!integerAddValue)
    return integerAddValue.takeError();
  auto floatAddValue = floatAdd->output(0);
  if (!floatAddValue)
    return floatAddValue.takeError();
  auto fusedFmaValue = fusedFma->output(0);
  if (!fusedFmaValue)
    return fusedFmaValue.takeError();
  auto integerAddRoutes = fu->addDemux(*integerAddValue, 2);
  if (!integerAddRoutes)
    return integerAddRoutes.takeError();
  auto floatAddRoutes = fu->addDemux(*floatAddValue, 2);
  if (!floatAddRoutes)
    return floatAddRoutes.takeError();
  auto fmaRoutes = fu->addDemux(*fusedFmaValue, 2);
  if (!fmaRoutes)
    return fmaRoutes.takeError();
  auto integerAddRouteValues = nodeOutputs(*integerAddRoutes, 2);
  if (!integerAddRouteValues)
    return integerAddRouteValues.takeError();
  auto floatAddRouteValues = nodeOutputs(*floatAddRoutes, 2);
  if (!floatAddRouteValues)
    return floatAddRouteValues.takeError();
  auto fmaRouteValues = nodeOutputs(*fmaRoutes, 2);
  if (!fmaRouteValues)
    return fmaRouteValues.takeError();

  auto carryNext =
      fu->addMux({(*integerAddRouteValues)[1], (*floatAddRouteValues)[1],
                  (*fmaRouteValues)[1]});
  if (!carryNext)
    return carryNext.takeError();
  auto carryNextValue = carryNext->output(0);
  if (!carryNextValue)
    return carryNextValue.takeError();
  auto carry = fu->addOperation(
      {boundary[3], (*d2Values)[3], *carryNextValue},
      OperationCapabilitySpec{ImplementationFamilyId::LoopCarry,
                              ::fabric::TokenPlaneParams{},
                              familyMembers(ImplementationFamilyId::LoopCarry),
                              {*bits64},
                              ::fabric::loopCarryOperationResourceContract()});
  if (!carry)
    return carry.takeError();
  auto carryOutput = carry->output(0);
  if (!carryOutput)
    return carryOutput.takeError();
  if (llvm::Error error =
          fu->resolveBackedge(std::move(*carryBackedge), *carryOutput))
    return error;

  auto result = fu->addMux({(*integerMultiplyRouteValues)[0],
                            (*floatMultiplyRouteValues)[0],
                            (*fmaRouteValues)[0], (*integerAddRouteValues)[0],
                            (*floatAddRouteValues)[0], *carryOutput});
  if (!result)
    return result.takeError();

  auto addTemplate = [&](std::vector<FuNode> operations,
                         std::vector<FuRouteSelection> routes) {
    return fu->addCapabilityTemplate(
        FuCapabilityTemplateSpec{std::move(operations), std::move(routes)});
  };
  if (llvm::Error error = addTemplate(
          {*integerMultiply},
          {{*d0, 0}, {*d1, 0}, {*integerMultiplyRoutes, 0}, {*result, 0}}))
    return error;
  if (llvm::Error error = addTemplate(
          {*floatMultiply},
          {{*d0, 1}, {*d1, 1}, {*floatMultiplyRoutes, 0}, {*result, 1}}))
    return error;
  if (llvm::Error error = addTemplate({*fusedFma}, {{*d0, 2},
                                                    {*d1, 2},
                                                    {*d2, 0},
                                                    {*fmaAddend, 0},
                                                    {*fmaRoutes, 0},
                                                    {*result, 2}}))
    return error;
  if (llvm::Error error = addTemplate({*integerMultiply, *integerAdd},
                                      {{*d0, 0},
                                       {*d1, 0},
                                       {*d2, 1},
                                       {*integerMultiplyRoutes, 1},
                                       {*integerAddRhs, 0},
                                       {*integerAddRoutes, 0},
                                       {*result, 3}}))
    return error;
  if (llvm::Error error =
          addTemplate({*floatMultiply, *floatAdd}, {{*d0, 1},
                                                    {*d1, 1},
                                                    {*d2, 2},
                                                    {*floatMultiplyRoutes, 1},
                                                    {*floatAddRhs, 0},
                                                    {*floatAddRoutes, 0},
                                                    {*result, 4}}))
    return error;
  if (llvm::Error error = addTemplate({*integerMultiply, *integerAdd, *carry},
                                      {{*d0, 0},
                                       {*d1, 0},
                                       {*d2, 3},
                                       {*integerMultiplyRoutes, 1},
                                       {*carryFeedback, 0},
                                       {*integerAddRhs, 1},
                                       {*integerAddRoutes, 1},
                                       {*carryNext, 0},
                                       {*result, 5}}))
    return error;
  if (llvm::Error error = addTemplate({*floatMultiply, *floatAdd, *carry},
                                      {{*d0, 1},
                                       {*d1, 1},
                                       {*d2, 3},
                                       {*floatMultiplyRoutes, 1},
                                       {*carryFeedback, 1},
                                       {*floatAddRhs, 1},
                                       {*floatAddRoutes, 1},
                                       {*carryNext, 1},
                                       {*result, 5}}))
    return error;
  if (llvm::Error error = addTemplate({*fusedFma, *carry}, {{*d0, 2},
                                                            {*d1, 2},
                                                            {*d2, 3},
                                                            {*carryFeedback, 2},
                                                            {*fmaAddend, 1},
                                                            {*fmaRoutes, 1},
                                                            {*carryNext, 2},
                                                            {*result, 5}}))
    return error;

  auto output = result->output(0);
  if (!output)
    return output.takeError();
  return fu->close({*output});
}

llvm::Error addLoopControlFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                             ::dataflow::StreamStepKind firstStep,
                             ::dataflow::StreamStepKind secondStep) {
  if (inputs.size() != 4)
    return invalid(
        "LoopControlFu requires data0, data1, data2, and phase inputs");
  if (firstStep == secondStep)
    return invalid("LoopControlFu requires distinct step kinds");

  auto bits128 = PortType::bits(128);
  if (!bits128)
    return bits128.takeError();
  auto bits1 = PortType::bits(1);
  if (!bits1)
    return bits1.takeError();

  auto fu = pe.addFu(inputs, FuSpec{{*bits128, *bits128, *bits128, *bits1},
                                    {*bits128, *bits128, *bits128}});
  if (!fu)
    return fu.takeError();
  llvm::SmallVector<FuValue, 4> boundary;
  for (std::size_t ordinal = 0; ordinal != 4; ++ordinal) {
    auto value = fu->input(ordinal);
    if (!value)
      return value.takeError();
    boundary.push_back(*value);
  }

  auto d0 = fu->addDemux(boundary[0], 5);
  if (!d0)
    return d0.takeError();
  auto d1 = fu->addDemux(boundary[1], 3);
  if (!d1)
    return d1.takeError();
  auto d2 = fu->addDemux(boundary[2], 2);
  if (!d2)
    return d2.takeError();
  auto phase = fu->addDemux(boundary[3], 5);
  if (!phase)
    return phase.takeError();
  auto d0Values = nodeOutputs(*d0, 5);
  if (!d0Values)
    return d0Values.takeError();
  auto d1Values = nodeOutputs(*d1, 3);
  if (!d1Values)
    return d1Values.takeError();
  auto d2Values = nodeOutputs(*d2, 2);
  if (!d2Values)
    return d2Values.takeError();
  auto phaseValues = nodeOutputs(*phase, 5);
  if (!phaseValues)
    return phaseValues.takeError();

  auto carryPhase = fu->addMux({(*phaseValues)[0], (*phaseValues)[3]});
  if (!carryPhase)
    return carryPhase.takeError();
  auto invariantPhase = fu->addMux({(*phaseValues)[1], (*phaseValues)[4]});
  if (!invariantPhase)
    return invariantPhase.takeError();
  auto gatePhase =
      fu->addMux({(*phaseValues)[2], (*phaseValues)[3], (*phaseValues)[4]});
  if (!gatePhase)
    return gatePhase.takeError();
  auto carryPhaseValue = carryPhase->output(0);
  if (!carryPhaseValue)
    return carryPhaseValue.takeError();
  auto invariantPhaseValue = invariantPhase->output(0);
  if (!invariantPhaseValue)
    return invariantPhaseValue.takeError();
  auto gatePhaseValue = gatePhase->output(0);
  if (!gatePhaseValue)
    return gatePhaseValue.takeError();

  const auto operationSpec = [&](ImplementationFamilyId family,
                                 const FamilyCapabilityParams &parameters,
                                 std::vector<PortType> results) {
    return OperationCapabilitySpec{family, parameters, familyMembers(family),
                                   std::move(results),
                                   loopControlResourceContract(family)};
  };
  const auto streamSpec = [&](::dataflow::StreamStepKind step) {
    return operationSpec(ImplementationFamilyId::LoopStream,
                         ::fabric::LoopStreamParams{ordinaryIntegerWidths(),
                                                    step, integerPredicates()},
                         {*bits128, *bits1});
  };
  const auto tokenSpec = [&](ImplementationFamilyId family,
                             std::vector<PortType> results) {
    return operationSpec(family, ::fabric::TokenPlaneParams{},
                         std::move(results));
  };

  auto firstStream = fu->addOperation(
      {(*d0Values)[0], (*d1Values)[0], (*d2Values)[0]}, streamSpec(firstStep));
  if (!firstStream)
    return firstStream.takeError();
  auto secondStream = fu->addOperation(
      {(*d0Values)[1], (*d1Values)[1], (*d2Values)[1]}, streamSpec(secondStep));
  if (!secondStream)
    return secondStream.takeError();
  auto carry = fu->addOperation(
      {*carryPhaseValue, (*d0Values)[2], (*d1Values)[2]},
      tokenSpec(ImplementationFamilyId::LoopCarry, {*bits128}));
  if (!carry)
    return carry.takeError();
  auto invariant = fu->addOperation(
      {*invariantPhaseValue, (*d0Values)[3]},
      tokenSpec(ImplementationFamilyId::LoopInvariant, {*bits128}));
  if (!invariant)
    return invariant.takeError();

  auto firstStreamValues = nodeOutputs(*firstStream, 2);
  if (!firstStreamValues)
    return firstStreamValues.takeError();
  auto secondStreamValues = nodeOutputs(*secondStream, 2);
  if (!secondStreamValues)
    return secondStreamValues.takeError();
  auto carryValue = carry->output(0);
  if (!carryValue)
    return carryValue.takeError();
  auto invariantValue = invariant->output(0);
  if (!invariantValue)
    return invariantValue.takeError();
  auto carryRoutes = fu->addDemux(*carryValue, 2);
  if (!carryRoutes)
    return carryRoutes.takeError();
  auto invariantRoutes = fu->addDemux(*invariantValue, 2);
  if (!invariantRoutes)
    return invariantRoutes.takeError();
  auto carryRouteValues = nodeOutputs(*carryRoutes, 2);
  if (!carryRouteValues)
    return carryRouteValues.takeError();
  auto invariantRouteValues = nodeOutputs(*invariantRoutes, 2);
  if (!invariantRouteValues)
    return invariantRouteValues.takeError();

  auto gateValue = fu->addMux(
      {(*d0Values)[4], (*carryRouteValues)[1], (*invariantRouteValues)[1]});
  if (!gateValue)
    return gateValue.takeError();
  auto selectedGateValue = gateValue->output(0);
  if (!selectedGateValue)
    return selectedGateValue.takeError();
  auto gate = fu->addOperation(
      {*gatePhaseValue, *selectedGateValue},
      tokenSpec(ImplementationFamilyId::LoopGate, {*bits1, *bits128}));
  if (!gate)
    return gate.takeError();
  auto gateValues = nodeOutputs(*gate, 2);
  if (!gateValues)
    return gateValues.takeError();
  auto gatePhaseRoutes = fu->addDemux((*gateValues)[0], 3);
  if (!gatePhaseRoutes)
    return gatePhaseRoutes.takeError();
  auto gateValueRoutes = fu->addDemux((*gateValues)[1], 3);
  if (!gateValueRoutes)
    return gateValueRoutes.takeError();
  auto gatePhaseRouteValues = nodeOutputs(*gatePhaseRoutes, 3);
  if (!gatePhaseRouteValues)
    return gatePhaseRouteValues.takeError();
  auto gateValueRouteValues = nodeOutputs(*gateValueRoutes, 3);
  if (!gateValueRouteValues)
    return gateValueRouteValues.takeError();

  auto r0 = fu->addMux({(*firstStreamValues)[0], (*secondStreamValues)[0],
                        (*carryRouteValues)[0], (*invariantRouteValues)[0],
                        (*gateValueRouteValues)[0], (*carryRouteValues)[1],
                        (*invariantRouteValues)[1]});
  if (!r0)
    return r0.takeError();
  auto r1 =
      fu->addMux({(*gateValueRouteValues)[1], (*gateValueRouteValues)[2]});
  if (!r1)
    return r1.takeError();
  auto p0 = fu->addMux({(*firstStreamValues)[1], (*secondStreamValues)[1],
                        (*gatePhaseRouteValues)[0], (*gatePhaseRouteValues)[1],
                        (*gatePhaseRouteValues)[2]});
  if (!p0)
    return p0.takeError();

  auto addTemplate = [&](std::vector<FuNode> operations,
                         std::vector<FuRouteSelection> routes) {
    return fu->addCapabilityTemplate(
        FuCapabilityTemplateSpec{std::move(operations), std::move(routes)});
  };
  if (llvm::Error error = addTemplate(
          {*firstStream}, {{*d0, 0}, {*d1, 0}, {*d2, 0}, {*r0, 0}, {*p0, 0}}))
    return error;
  if (llvm::Error error = addTemplate(
          {*secondStream}, {{*d0, 1}, {*d1, 1}, {*d2, 1}, {*r0, 1}, {*p0, 1}}))
    return error;
  if (llvm::Error error = addTemplate({*carry}, {{*phase, 0},
                                                 {*carryPhase, 0},
                                                 {*d0, 2},
                                                 {*d1, 2},
                                                 {*carryRoutes, 0},
                                                 {*r0, 2}}))
    return error;
  if (llvm::Error error = addTemplate({*invariant}, {{*phase, 1},
                                                     {*invariantPhase, 0},
                                                     {*d0, 3},
                                                     {*invariantRoutes, 0},
                                                     {*r0, 3}}))
    return error;
  if (llvm::Error error = addTemplate({*gate}, {{*phase, 2},
                                                {*gatePhase, 0},
                                                {*d0, 4},
                                                {*gateValue, 0},
                                                {*gatePhaseRoutes, 0},
                                                {*gateValueRoutes, 0},
                                                {*r0, 4},
                                                {*p0, 2}}))
    return error;
  if (llvm::Error error = addTemplate({*carry, *gate}, {{*phase, 3},
                                                        {*carryPhase, 1},
                                                        {*gatePhase, 1},
                                                        {*d0, 2},
                                                        {*d1, 2},
                                                        {*carryRoutes, 1},
                                                        {*gateValue, 1},
                                                        {*gatePhaseRoutes, 1},
                                                        {*gateValueRoutes, 1},
                                                        {*r0, 5},
                                                        {*r1, 0},
                                                        {*p0, 3}}))
    return error;
  if (llvm::Error error =
          addTemplate({*invariant, *gate}, {{*phase, 4},
                                            {*invariantPhase, 1},
                                            {*gatePhase, 2},
                                            {*d0, 3},
                                            {*invariantRoutes, 1},
                                            {*gateValue, 2},
                                            {*gatePhaseRoutes, 2},
                                            {*gateValueRoutes, 2},
                                            {*r0, 6},
                                            {*r1, 1},
                                            {*p0, 4}}))
    return error;

  auto r0Value = r0->output(0);
  if (!r0Value)
    return r0Value.takeError();
  auto r1Value = r1->output(0);
  if (!r1Value)
    return r1Value.takeError();
  auto p0Value = p0->output(0);
  if (!p0Value)
    return p0Value.takeError();
  return fu->close({*r0Value, *r1Value, *p0Value});
}

llvm::Error addVectorComputeFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                               VectorComputeFuParameters parameters) {
  if (inputs.size() != 4)
    return invalid(
        "VectorComputeFu requires data0, data1, data2, and condition inputs");
  if (parameters.outerPayloadBits == 0 || parameters.vectorPayloadBits == 0)
    return invalid("VectorComputeFu widths must be positive");
  if (parameters.vectorPayloadBits > parameters.outerPayloadBits)
    return invalid("VectorComputeFu vector width exceeds its outer width");
  auto outer = PortType::bits(parameters.outerPayloadBits);
  if (!outer)
    return outer.takeError();
  auto vector = PortType::bits(parameters.vectorPayloadBits);
  if (!vector)
    return vector.takeError();
  const auto integer = ordinaryIntegerWidths();
  const auto floating = floatFormats();
  const auto strict = ::fabric::FloatBehaviorProfile::strictIEEE();
  const std::uint32_t capacity = parameters.vectorPayloadBits;
  std::vector<SelectableResource> resources = {
      {ImplementationFamilyId::FixedVectorIntegerAddSub,
       ::fabric::FixedVectorIntegerParams{integer, capacity},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorIntegerSaturatingAddSub,
       ::fabric::FixedVectorIntegerParams{integer, capacity},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorIntegerCountZeros,
       ::fabric::FixedVectorIntegerParams{integer, capacity},
       {0}},
      {ImplementationFamilyId::FixedVectorIntegerLogic,
       ::fabric::FixedVectorIntegerParams{logicIntegerWidths(), capacity},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorIntegerShift,
       ::fabric::FixedVectorIntegerParams{integer, capacity},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorIntegerCompareMinMax,
       ::fabric::FixedVectorIntegerCompareMinMaxParams{
           integer, integerPredicates(), capacity},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorValueSelect,
       ::fabric::FixedVectorValueSelectParams{logicIntegerWidths(), floating,
                                              capacity},
       {3, 0, 1}},
      {ImplementationFamilyId::FixedVectorIntegerMultiply,
       ::fabric::FixedVectorIntegerParams{integer, capacity},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorFloatSign,
       ::fabric::FixedVectorFloatParams{floating, strict, capacity},
       {0}},
      {ImplementationFamilyId::FixedVectorFloatAddSub,
       ::fabric::FixedVectorFloatParams{floating, strict, capacity},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorFloatCompareMinMax,
       ::fabric::FixedVectorFloatCompareMinMaxParams{
           floating, floatCompareBehavior(), floatPredicates(), capacity},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorFloatMultiply,
       ::fabric::FixedVectorFloatParams{floating, strict, capacity},
       {0, 1}},
      {ImplementationFamilyId::FixedVectorFloatFma,
       ::fabric::FixedVectorFloatParams{floating, strict, capacity},
       {0, 1, 2}},
  };
  return addSelectableFu(pe, inputs, {*vector, *vector, *vector, *vector},
                         *vector, *outer, resources);
}

llvm::Error
addVectorStructuralFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                      const VectorStructuralFuParameters &parameters) {
  const std::uint32_t dynamicRank =
      parameters.sliceCapability.maxDynamicPositionRank;
  if (inputs.size() != 2 + static_cast<std::size_t>(dynamicRank))
    return invalid("VectorStructuralFu input count does not match its dynamic "
                   "position rank");
  if (parameters.outerPayloadBits == 0 || parameters.vectorPayloadBits == 0 ||
      parameters.indexPayloadBits == 0)
    return invalid("VectorStructuralFu widths must be positive");
  if (parameters.vectorPayloadBits > parameters.outerPayloadBits ||
      parameters.indexPayloadBits > parameters.outerPayloadBits)
    return invalid("VectorStructuralFu inner width exceeds its outer width");
  if (parameters.sliceCapability.maxContainerPayloadBits >
          parameters.vectorPayloadBits ||
      parameters.sliceCapability.maxSlicePayloadBits >
          parameters.vectorPayloadBits ||
      parameters.shuffleCapability.maxOperandPayloadBits >
          parameters.vectorPayloadBits ||
      parameters.shuffleCapability.maxResultPayloadBits >
          parameters.vectorPayloadBits ||
      parameters.shuffleCapability.maxBlockPayloadBits >
          parameters.vectorPayloadBits)
    return invalid("VectorStructuralFu capability exceeds its vector width");
  if ((parameters.sliceCapability.resolvedIndexWidths.contains(
           ::fabric::ResolvedIndexWidth::I32) &&
       parameters.indexPayloadBits < 32) ||
      (parameters.sliceCapability.resolvedIndexWidths.contains(
           ::fabric::ResolvedIndexWidth::I64) &&
       parameters.indexPayloadBits < 64))
    return invalid("VectorStructuralFu index width cannot carry its resolved "
                   "index domain");

  auto sliceLayout =
      ::fabric::resolveFixedVectorSliceAlignMergeConfigurationLayout(
          parameters.sliceCapability,
          {OperationSchemaId::VectorExtract, OperationSchemaId::VectorInsert});
  if (!sliceLayout)
    return sliceLayout.takeError();
  auto shuffleLayout = ::fabric::resolveFixedVectorShuffleConfigurationLayout(
      parameters.shuffleCapability);
  if (!shuffleLayout)
    return shuffleLayout.takeError();

  auto outer = PortType::bits(parameters.outerPayloadBits);
  if (!outer)
    return outer.takeError();
  auto vector = PortType::bits(parameters.vectorPayloadBits);
  if (!vector)
    return vector.takeError();
  auto index = PortType::bits(parameters.indexPayloadBits);
  if (!index)
    return index.takeError();

  std::vector<PortType> innerInputs = {*vector, *vector};
  std::vector<std::uint32_t> sliceRoles = {0, 1};
  innerInputs.reserve(inputs.size());
  sliceRoles.reserve(inputs.size());
  for (std::uint32_t role = 0; role != dynamicRank; ++role) {
    innerInputs.push_back(*index);
    sliceRoles.push_back(role + 2);
  }
  std::vector<RoutedResource> resources = {
      {ImplementationFamilyId::FixedVectorSliceAlignMerge,
       parameters.sliceCapability,
       std::move(sliceRoles),
       {*vector},
       {0}},
      {ImplementationFamilyId::FixedVectorShuffle,
       parameters.shuffleCapability,
       {0, 1},
       {*vector},
       {0}},
  };
  return addRoutedFu(pe, inputs, std::move(innerInputs), {*outer}, resources);
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

llvm::Error addTokenControlFu(PeBuilder &pe, llvm::ArrayRef<PeValue> inputs,
                              TokenControlFuParameters parameters) {
  if (parameters.outerPayloadBits == 0 || parameters.selectorPayloadBits == 0)
    return invalid("TokenControlFu widths must be positive");
  if (parameters.selectorPayloadBits > parameters.outerPayloadBits)
    return invalid("TokenControlFu selector width exceeds its outer width");
  if (inputs.size() != 5)
    return invalid(
        "TokenControlFu requires selector/control and four payload inputs");
  auto outer = PortType::bits(parameters.outerPayloadBits);
  if (!outer)
    return outer.takeError();
  auto selector = PortType::bits(parameters.selectorPayloadBits);
  if (!selector)
    return selector.takeError();

  const std::vector<std::uint32_t> payloadRoles = {1, 2, 3, 4};
  const std::vector<std::uint32_t> outputRoles = {0, 1, 2, 3};
  const std::vector<PortType> payloadTypes(4, *outer);
  const ::fabric::RoutedTokenParams routed{parameters.outerPayloadBits, 4};
  std::vector<RoutedResource> resources = {
      {ImplementationFamilyId::TokenConstant,
       ::fabric::PayloadCapacityParams{parameters.outerPayloadBits},
       {0},
       {*outer},
       {0}},
      {ImplementationFamilyId::TokenSync, routed, payloadRoles, payloadTypes,
       outputRoles},
      {ImplementationFamilyId::TokenMux,
       routed,
       {0, 1, 2, 3, 4},
       {*outer},
       {0}},
      {ImplementationFamilyId::TokenDemux,
       routed,
       {0, 1},
       payloadTypes,
       outputRoles},
  };
  std::vector<PortType> innerInputs = {*selector};
  innerInputs.insert(innerInputs.end(), payloadTypes.begin(),
                     payloadTypes.end());
  return addRoutedFu(pe, inputs, std::move(innerInputs), payloadTypes,
                     resources);
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
  resources.push_back(
      scalarSpecialMath(ImplementationFamilyId::ScalarMathPow, {0, 1}));
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
    resources.push_back(scalarSpecialMath(family, {0}));
  return addSelectableFu(pe, inputs, {*bits64, *bits64}, *bits64, *bits128,
                         resources);
}

} // namespace loom::adg

//===- ImplementationFamilyControlBehavior.cpp --------------------------===//
//
// Owns the finite behavior quotients of loop, vector-adapter, and routed
// token implementation families.
//
//===----------------------------------------------------------------------===//

#include "ImplementationFamilyBehaviorInternal.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace fabric;
using ::dataflow::OperationSchemaId;
using Component = detail::ImplementationFamilyBehaviorKeyComponent;

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

struct Candidate final {
  FiniteImplementationFamilyBehaviorPoint point;
  std::vector<Component> components;
};

bool componentEqual(const Component &lhs, const Component &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *width = std::get_if<std::uint32_t>(&lhs))
    return *width == std::get<std::uint32_t>(rhs);
  if (const auto *predicate = std::get_if<::loom::CanonicalSemanticBytes>(&lhs))
    return predicate->bytes().equals(
        std::get<::loom::CanonicalSemanticBytes>(rhs).bytes());
  const auto &lhsImage =
      std::get<detail::ImplementationFamilyBehaviorLaneImage>(lhs);
  const auto &rhsImage =
      std::get<detail::ImplementationFamilyBehaviorLaneImage>(rhs);
  return lhsImage.bound == rhsImage.bound &&
         lhsImage.ordinals == rhsImage.ordinals;
}

bool behaviorEqual(const Candidate &lhs, const Candidate &rhs) {
  return lhs.components.size() == rhs.components.size() &&
         llvm::all_of(
             llvm::zip(lhs.components, rhs.components), [](const auto &pair) {
               return componentEqual(std::get<0>(pair), std::get<1>(pair));
             });
}

bool sameLaneSet(llvm::ArrayRef<std::uint64_t> lhs,
                 llvm::ArrayRef<std::uint64_t> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  std::vector<std::uint64_t> sortedLhs(lhs.begin(), lhs.end());
  std::vector<std::uint64_t> sortedRhs(rhs.begin(), rhs.end());
  llvm::sort(sortedLhs);
  llvm::sort(sortedRhs);
  return sortedLhs == sortedRhs;
}

llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>>
finalizeDomain(ImplementationFamilyId family, std::vector<Candidate> candidates,
               std::string firstRejection) {
  std::vector<Candidate> unique;
  for (Candidate &candidate : candidates) {
    if (llvm::none_of(unique, [&](const Candidate &existing) {
          return behaviorEqual(existing, candidate);
        }))
      unique.push_back(std::move(candidate));
  }
  if (unique.empty())
    return reject(firstRejection.empty()
                      ? "concrete capability has no reachable behavior"
                      : firstRejection);
  if (unique.size() == 1) {
    unique.front().point.semanticConfiguration = std::nullopt;
    std::vector<FiniteImplementationFamilyBehaviorPoint> singleton;
    singleton.push_back(std::move(unique.front().point));
    return singleton;
  }

  const std::size_t componentCount = unique.front().components.size();
  if (llvm::any_of(unique, [&](const Candidate &candidate) {
        return candidate.components.size() != componentCount;
      }))
    return reject("behavior quotient has inconsistent component arity");
  std::vector<bool> varying(componentCount, false);
  for (std::size_t component = 0; component != componentCount; ++component)
    varying[component] = llvm::any_of(
        llvm::ArrayRef<Candidate>(unique).drop_front(),
        [&](const Candidate &candidate) {
          return !componentEqual(unique.front().components[component],
                                 candidate.components[component]);
        });

  std::vector<FiniteImplementationFamilyBehaviorPoint> result;
  result.reserve(unique.size());
  for (Candidate &candidate : unique) {
    std::vector<Component> encodedComponents;
    for (std::size_t component = 0; component != componentCount; ++component)
      if (varying[component])
        encodedComponents.push_back(candidate.components[component]);
    auto key = detail::encodeImplementationFamilyBehaviorKey(family, {},
                                                             encodedComponents);
    if (!key)
      return key.takeError();
    candidate.point.semanticConfiguration = std::move(*key);
    result.push_back(std::move(candidate.point));
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return std::lexicographical_compare(
        lhs.semanticConfiguration->bytes().begin(),
        lhs.semanticConfiguration->bytes().end(),
        rhs.semanticConfiguration->bytes().begin(),
        rhs.semanticConfiguration->bytes().end());
  });
  return result;
}

llvm::Error
requireSingletonSchema(ImplementationFamilyId family,
                       llvm::ArrayRef<OperationSchemaId> enabledSchemas,
                       OperationSchemaId expected) {
  if (enabledSchemas.size() != 1 || enabledSchemas.front() != expected)
    return reject(implementationFamilyKeyword(family) +
                  " must enable exactly its registered schema");
  return llvm::Error::success();
}

mlir::Type floatType(mlir::MLIRContext &context, FloatFormat format) {
  switch (format) {
  case FloatFormat::F16:
    return mlir::Float16Type::get(&context);
  case FloatFormat::BF16:
    return mlir::BFloat16Type::get(&context);
  case FloatFormat::F32:
    return mlir::Float32Type::get(&context);
  case FloatFormat::F64:
    return mlir::Float64Type::get(&context);
  }
  llvm_unreachable("unknown floating-point format");
}

mlir::Type payloadType(mlir::MLIRContext &context, std::uint32_t width) {
  if (width == 0)
    return mlir::NoneType::get(&context);
  return mlir::IntegerType::get(&context, width);
}

void noteRejection(std::string &firstRejection, llvm::Error error) {
  if (firstRejection.empty())
    firstRejection = llvm::toString(std::move(error));
  else
    llvm::consumeError(std::move(error));
}

void appendReachableCandidate(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths, Candidate candidate,
    std::vector<Candidate> &candidates, std::string &firstRejection) {
  if (llvm::Error error = detail::validateImplementationFamilyBehaviorPoint(
          family, params, candidate.point.representativeActor,
          candidate.point.operandPorts, candidate.point.resultPorts,
          physicalInputWidths, physicalResultWidths,
          candidate.point.resolvedIndexWidth)) {
    noteRejection(firstRejection, std::move(error));
    return;
  }
  candidates.push_back(std::move(candidate));
}

llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>>
resolveLoopStream(const FamilyCapabilityParams &params,
                  llvm::ArrayRef<OperationSchemaId> enabledSchemas,
                  llvm::ArrayRef<std::uint32_t> physicalInputWidths,
                  llvm::ArrayRef<std::uint32_t> physicalResultWidths,
                  mlir::MLIRContext &context) {
  constexpr ImplementationFamilyId family = ImplementationFamilyId::LoopStream;
  if (llvm::Error error = requireSingletonSchema(
          family, enabledSchemas, OperationSchemaId::DataflowStream))
    return std::move(error);
  const auto *typed = std::get_if<LoopStreamParams>(&params);
  if (!typed)
    return reject("LoopStream has the wrong parameter schema");

  std::vector<Candidate> candidates;
  std::string firstRejection;
  for (IntegerWidth width : integerWidthDomain) {
    if (!typed->integerWidths.contains(width))
      continue;
    const std::uint32_t bits = getBitWidth(width);
    mlir::Type integer = mlir::IntegerType::get(&context, bits);
    for (std::uint32_t ordinal = 0;
         ordinal <= mlir::arith::getMaxEnumValForCmpIPredicate(); ++ordinal) {
      const auto predicate = static_cast<mlir::arith::CmpIPredicate>(ordinal);
      if (!typed->continuationPredicates.contains(predicate))
        continue;
      auto predicateBytes =
          ::dataflow::encodeIntegerComparePredicate(predicate);
      if (!predicateBytes)
        return predicateBytes.takeError();
      ::dataflow::CanonicalActorSchemaProjection actor{
          OperationSchemaId::DataflowStream,
          mlir::FunctionType::get(
              &context, {integer, integer, integer},
              {integer, mlir::IntegerType::get(&context, 1)}),
          ::dataflow::StreamRecurrencePayload{typed->fixedStepKind, predicate}};
      appendReachableCandidate(
          family, params, physicalInputWidths, physicalResultWidths,
          Candidate{
              {std::move(actor), std::nullopt, std::nullopt, {0, 1, 2}, {0, 1}},
              {bits, std::move(*predicateBytes)}},
          candidates, firstRejection);
    }
  }
  return finalizeDomain(family, std::move(candidates),
                        std::move(firstRejection));
}

llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>>
resolveAdapter(ImplementationFamilyId family,
               const FamilyCapabilityParams &params,
               llvm::ArrayRef<OperationSchemaId> enabledSchemas,
               llvm::ArrayRef<std::uint32_t> physicalInputWidths,
               llvm::ArrayRef<std::uint32_t> physicalResultWidths,
               mlir::MLIRContext &context) {
  const OperationSchemaId schema =
      family == ImplementationFamilyId::FixedVectorParallelize
          ? OperationSchemaId::DataflowParallelize
          : OperationSchemaId::DataflowSerialize;
  if (llvm::Error error =
          requireSingletonSchema(family, enabledSchemas, schema))
    return std::move(error);
  const auto *typed = std::get_if<FixedVectorAdapterParams>(&params);
  if (!typed)
    return reject("fixed-vector adapter has the wrong parameter schema");

  std::vector<mlir::Type> elements;
  for (IntegerWidth width : integerWidthDomain)
    if (typed->integerElementWidths.contains(width))
      elements.push_back(mlir::IntegerType::get(&context, getBitWidth(width)));
  for (FloatFormat format : floatFormatDomain)
    if (typed->floatElementFormats.contains(format))
      elements.push_back(floatType(context, format));

  std::vector<Candidate> candidates;
  std::string firstRejection;
  const mlir::Type i1 = mlir::IntegerType::get(&context, 1);
  for (mlir::Type element : elements) {
    const std::uint32_t elementWidth = element.getIntOrFloatBitWidth();
    if (elementWidth == 0)
      continue;
    const std::uint32_t maxLanes = typed->maxPayloadBits / elementWidth;
    for (std::uint64_t lanes = 1; lanes <= maxLanes; ++lanes) {
      const auto laneCount = static_cast<std::int64_t>(lanes);
      const auto vector = mlir::VectorType::get({laneCount}, element);
      const auto mask = mlir::VectorType::get({laneCount}, i1);
      ::dataflow::CanonicalActorSchemaProjection actor{
          schema,
          family == ImplementationFamilyId::FixedVectorParallelize
              ? mlir::FunctionType::get(&context, {element, i1},
                                        {vector, mask, i1})
              : mlir::FunctionType::get(&context, {vector, mask, i1},
                                        {element, i1}),
          ::dataflow::NoPayload{}};
      const bool parallelize =
          family == ImplementationFamilyId::FixedVectorParallelize;
      appendReachableCandidate(
          family, params, physicalInputWidths, physicalResultWidths,
          Candidate{{std::move(actor), std::nullopt, std::nullopt,
                     parallelize ? std::vector<std::uint64_t>{0, 1}
                                 : std::vector<std::uint64_t>{0, 1, 2},
                     parallelize ? std::vector<std::uint64_t>{0, 1, 2}
                                 : std::vector<std::uint64_t>{0, 1}},
                    {elementWidth, static_cast<std::uint32_t>(lanes)}},
          candidates, firstRejection);
    }
  }
  return finalizeDomain(family, std::move(candidates),
                        std::move(firstRejection));
}

llvm::Expected<std::vector<FiniteImplementationFamilyBehaviorPoint>>
resolveRoutedToken(ImplementationFamilyId family,
                   const FamilyCapabilityParams &params,
                   llvm::ArrayRef<OperationSchemaId> enabledSchemas,
                   llvm::ArrayRef<std::uint32_t> physicalInputWidths,
                   llvm::ArrayRef<std::uint32_t> physicalResultWidths,
                   mlir::MLIRContext &context) {
  if (physicalInputWidths.size() > std::numeric_limits<std::uint32_t>::max() ||
      physicalResultWidths.size() > std::numeric_limits<std::uint32_t>::max())
    return reject("routed-token physical port inventory exceeds uint32");
  const auto *typed = std::get_if<RoutedTokenParams>(&params);
  if (!typed)
    return reject("routed token has the wrong parameter schema");
  if (llvm::Error error = verifyRoutedTokenParams(*typed))
    return std::move(error);
  OperationSchemaId schema;
  if (family == ImplementationFamilyId::TokenSync)
    schema = OperationSchemaId::DataflowSync;
  else if (family == ImplementationFamilyId::TokenMux)
    schema = OperationSchemaId::DataflowMux;
  else
    schema = OperationSchemaId::DataflowDemux;
  if (llvm::Error error =
          requireSingletonSchema(family, enabledSchemas, schema))
    return std::move(error);

  std::vector<Candidate> candidates;
  std::string firstRejection;
  if (family == ImplementationFamilyId::TokenSync) {
    const std::uint32_t portCount = static_cast<std::uint32_t>(
        std::min(physicalInputWidths.size(), physicalResultWidths.size()));
    for (std::uint32_t laneCount = 1;
         laneCount <= std::min(typed->maxFan, portCount); ++laneCount) {
      if (llvm::Error error = detail::forEachCanonicalRoutedTokenLaneImage(
              family, params, physicalInputWidths, physicalResultWidths,
              laneCount,
              [&](const detail::CanonicalRoutedTokenLaneImage &image)
                  -> llvm::Expected<bool> {
                std::vector<std::uint64_t> ports = image.operandPorts;
                llvm::sort(ports);
                std::vector<mlir::Type> laneTypes;
                laneTypes.reserve(ports.size());
                for (std::uint64_t port : ports) {
                  const std::uint32_t width = std::min(
                      {typed->maxPayloadBits, physicalInputWidths[port],
                       physicalResultWidths[port]});
                  laneTypes.push_back(payloadType(context, width));
                }
                ::dataflow::CanonicalActorSchemaProjection actor{
                    schema,
                    mlir::FunctionType::get(&context, laneTypes, laneTypes),
                    ::dataflow::NoPayload{}};
                appendReachableCandidate(
                    family, params, physicalInputWidths, physicalResultWidths,
                    Candidate{{std::move(actor), std::nullopt, std::nullopt,
                               ports, ports},
                              {detail::ImplementationFamilyBehaviorLaneImage{
                                  ports, portCount}}},
                    candidates, firstRejection);
                return true;
              }))
        return std::move(error);
    }
  } else if (family == ImplementationFamilyId::TokenMux) {
    if (physicalInputWidths.empty() || physicalResultWidths.empty())
      return reject("token mux physical role inventory is incomplete");
    const std::uint32_t dataPortCount =
        static_cast<std::uint32_t>(physicalInputWidths.size() - 1);
    for (std::uint32_t laneCount = 2;
         laneCount <= std::min(typed->maxFan, dataPortCount); ++laneCount) {
      if (llvm::Error error = detail::forEachCanonicalRoutedTokenLaneImage(
              family, params, physicalInputWidths, physicalResultWidths,
              laneCount,
              [&](const detail::CanonicalRoutedTokenLaneImage &image)
                  -> llvm::Expected<bool> {
                const std::uint32_t width =
                    *std::min_element(image.effectivePayloadWidths.begin(),
                                      image.effectivePayloadWidths.end());
                const bool indexSelector = laneCount > 2;
                const mlir::Type selector =
                    indexSelector
                        ? mlir::Type(mlir::IndexType::get(&context))
                        : mlir::Type(mlir::IntegerType::get(&context, 1));
                const mlir::Type payload = payloadType(context, width);
                std::vector<mlir::Type> inputs(1, selector);
                inputs.insert(inputs.end(), laneCount, payload);
                ::dataflow::CanonicalActorSchemaProjection actor{
                    schema,
                    mlir::FunctionType::get(&context, inputs, {payload}),
                    ::dataflow::NoPayload{}};
                std::vector<std::uint64_t> dataImage(
                    image.operandPorts.begin() + 1, image.operandPorts.end());
                appendReachableCandidate(
                    family, params, physicalInputWidths, physicalResultWidths,
                    Candidate{
                        {std::move(actor), std::nullopt,
                         indexSelector ? std::optional<ResolvedIndexWidth>(
                                             ResolvedIndexWidth::I32)
                                       : std::nullopt,
                         image.operandPorts, image.resultPorts},
                        {detail::ImplementationFamilyBehaviorLaneImage{
                            std::move(dataImage), physicalInputWidths.size()}}},
                    candidates, firstRejection);
                return true;
              }))
        return std::move(error);
    }
  } else {
    if (physicalInputWidths.size() < 2)
      return reject("token demux physical role inventory is incomplete");
    const std::uint32_t resultPortCount =
        static_cast<std::uint32_t>(physicalResultWidths.size());
    for (std::uint32_t laneCount = 2;
         laneCount <= std::min(typed->maxFan, resultPortCount); ++laneCount) {
      if (llvm::Error error = detail::forEachCanonicalRoutedTokenLaneImage(
              family, params, physicalInputWidths, physicalResultWidths,
              laneCount,
              [&](const detail::CanonicalRoutedTokenLaneImage &image)
                  -> llvm::Expected<bool> {
                const std::uint32_t width =
                    *std::min_element(image.effectivePayloadWidths.begin(),
                                      image.effectivePayloadWidths.end());
                const bool indexSelector = laneCount > 2;
                const mlir::Type selector =
                    indexSelector
                        ? mlir::Type(mlir::IndexType::get(&context))
                        : mlir::Type(mlir::IntegerType::get(&context, 1));
                const mlir::Type payload = payloadType(context, width);
                std::vector<mlir::Type> results(laneCount, payload);
                ::dataflow::CanonicalActorSchemaProjection actor{
                    schema,
                    mlir::FunctionType::get(&context, {selector, payload},
                                            results),
                    ::dataflow::NoPayload{}};
                appendReachableCandidate(
                    family, params, physicalInputWidths, physicalResultWidths,
                    Candidate{
                        {std::move(actor), std::nullopt,
                         indexSelector ? std::optional<ResolvedIndexWidth>(
                                             ResolvedIndexWidth::I32)
                                       : std::nullopt,
                         image.operandPorts, image.resultPorts},
                        {detail::ImplementationFamilyBehaviorLaneImage{
                            image.resultPorts, physicalResultWidths.size()}}},
                    candidates, firstRejection);
                return true;
              }))
        return std::move(error);
    }
  }
  return finalizeDomain(family, std::move(candidates),
                        std::move(firstRejection));
}

std::optional<std::pair<std::uint32_t, std::uint32_t>>
adapterBehavior(ImplementationFamilyId family,
                const ::dataflow::CanonicalActorSchemaProjection &actor) {
  mlir::Type vectorType =
      family == ImplementationFamilyId::FixedVectorParallelize
          ? actor.type.getResult(0)
          : actor.type.getInput(0);
  auto vector = llvm::dyn_cast<mlir::VectorType>(vectorType);
  if (!vector || vector.getRank() != 1 || vector.getDimSize(0) <= 0)
    return std::nullopt;
  return std::pair<std::uint32_t, std::uint32_t>{
      vector.getElementTypeBitWidth(),
      static_cast<std::uint32_t>(vector.getDimSize(0))};
}

bool sameProjectedBehavior(
    ImplementationFamilyId family,
    const FiniteImplementationFamilyBehaviorPoint &point,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint64_t> operandPorts,
    llvm::ArrayRef<std::uint64_t> resultPorts) {
  if (family == ImplementationFamilyId::LoopStream) {
    const auto *lhs = std::get_if<::dataflow::StreamRecurrencePayload>(
        &point.representativeActor.payload);
    const auto *rhs =
        std::get_if<::dataflow::StreamRecurrencePayload>(&actor.payload);
    return lhs && rhs && lhs->predicate == rhs->predicate &&
           point.representativeActor.type.getInput(0).getIntOrFloatBitWidth() ==
               actor.type.getInput(0).getIntOrFloatBitWidth();
  }
  if (family == ImplementationFamilyId::FixedVectorParallelize ||
      family == ImplementationFamilyId::FixedVectorSerialize)
    return adapterBehavior(family, point.representativeActor) ==
           adapterBehavior(family, actor);
  if (family == ImplementationFamilyId::TokenSync)
    return sameLaneSet(point.operandPorts, operandPorts);
  if (family == ImplementationFamilyId::TokenMux)
    return llvm::ArrayRef<std::uint64_t>(point.operandPorts).drop_front() ==
           operandPorts.drop_front();
  return llvm::ArrayRef<std::uint64_t>(point.resultPorts) == resultPorts;
}

} // namespace

bool fabric::detail::ownsControlBehaviorRelation(
    ImplementationFamilyId family) {
  switch (family) {
  case ImplementationFamilyId::LoopStream:
  case ImplementationFamilyId::FixedVectorParallelize:
  case ImplementationFamilyId::FixedVectorSerialize:
  case ImplementationFamilyId::TokenSync:
  case ImplementationFamilyId::TokenMux:
  case ImplementationFamilyId::TokenDemux:
    return true;
  default:
    return false;
  }
}

llvm::Expected<std::vector<fabric::FiniteImplementationFamilyBehaviorPoint>>
fabric::detail::resolveControlBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    ::mlir::MLIRContext &context) {
  switch (family) {
  case ImplementationFamilyId::LoopStream:
    return resolveLoopStream(params, enabledSchemas, physicalInputWidths,
                             physicalResultWidths, context);
  case ImplementationFamilyId::FixedVectorParallelize:
  case ImplementationFamilyId::FixedVectorSerialize:
    return resolveAdapter(family, params, enabledSchemas, physicalInputWidths,
                          physicalResultWidths, context);
  case ImplementationFamilyId::TokenSync:
  case ImplementationFamilyId::TokenMux:
  case ImplementationFamilyId::TokenDemux:
    return resolveRoutedToken(family, params, enabledSchemas,
                              physicalInputWidths, physicalResultWidths,
                              context);
  default:
    return reject("implementation family has no control behavior quotient");
  }
}

llvm::Expected<::loom::CanonicalSemanticBytes>
fabric::detail::projectControlBehaviorKey(
    ImplementationFamilyId family,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<std::uint64_t> operandPorts,
    llvm::ArrayRef<std::uint64_t> resultPorts) {
  const auto point = llvm::find_if(domain, [&](const auto &candidate) {
    return sameProjectedBehavior(family, candidate, actor, operandPorts,
                                 resultPorts);
  });
  if (point == domain.end() || !point->semanticConfiguration)
    return reject("actor does not project into the finite control domain");
  return ::loom::CanonicalSemanticBytes(
      std::vector<std::uint8_t>(point->semanticConfiguration->bytes().begin(),
                                point->semanticConfiguration->bytes().end()));
}

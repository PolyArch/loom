//===- ImplementationFamilyScalarFloatBehavior.cpp ----------------------===//
//
// Owns the closed finite behavior quotients of scalar floating families.
//
//===----------------------------------------------------------------------===//

#include "ImplementationFamilyScalarFloatBehavior.h"

#include "ImplementationFamilyBehaviorInternal.h"

#include "Dataflow/IR/OperationSchemaCodec.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace fabric;

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

enum class BehaviorComponentSlot : std::uint8_t {
  RepresentationWidth,
  Format,
  SourceFormat,
  DestinationFormat,
  SourceWidth,
  DestinationWidth,
  Rounding,
};

using BehaviorComponentValue =
    std::variant<std::uint32_t, ::loom::CanonicalSemanticBytes>;

struct BehaviorComponent final {
  BehaviorComponentSlot slot;
  BehaviorComponentValue value;
};

struct ScalarFloatBehaviorCandidate final {
  ::dataflow::CanonicalActorSchemaProjection actor;
  llvm::StringRef role;
  std::vector<BehaviorComponent> components;
  std::vector<std::uint64_t> operandPorts;
  std::vector<std::uint64_t> resultPorts;
  std::vector<std::uint8_t> canonicalActorBytes;
  std::optional<::mlir::arith::RoundingMode> selectedRounding;
  unsigned refinementStrength = 0;
  bool nonNegativeRelaxation = false;
};

bool equalComponentValue(const BehaviorComponentValue &lhs,
                         const BehaviorComponentValue &rhs) {
  if (lhs.index() != rhs.index())
    return false;
  if (const auto *left = std::get_if<std::uint32_t>(&lhs))
    return *left == std::get<std::uint32_t>(rhs);
  return std::get<::loom::CanonicalSemanticBytes>(lhs).bytes().equals(
      std::get<::loom::CanonicalSemanticBytes>(rhs).bytes());
}

const BehaviorComponent *
findComponent(const ScalarFloatBehaviorCandidate &candidate,
              BehaviorComponentSlot slot) {
  auto found = llvm::find_if(candidate.components,
                             [&](const BehaviorComponent &component) {
                               return component.slot == slot;
                             });
  return found == candidate.components.end() ? nullptr : &*found;
}

bool componentVaries(llvm::ArrayRef<ScalarFloatBehaviorCandidate> candidates,
                     llvm::StringRef role, BehaviorComponentSlot slot) {
  const BehaviorComponentValue *first = nullptr;
  for (const ScalarFloatBehaviorCandidate &candidate : candidates) {
    if (candidate.role != role)
      continue;
    const BehaviorComponent *component = findComponent(candidate, slot);
    if (!component)
      continue;
    if (!first) {
      first = &component->value;
      continue;
    }
    if (!equalComponentValue(*first, component->value))
      return true;
  }
  return false;
}

std::vector<std::uint64_t> identityPorts(unsigned count) {
  std::vector<std::uint64_t> ports(count);
  std::iota(ports.begin(), ports.end(), 0);
  return ports;
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
  llvm_unreachable("unknown floating format");
}

mlir::Type integerType(mlir::MLIRContext &context, IntegerWidth width) {
  return mlir::IntegerType::get(&context, getBitWidth(width));
}

llvm::Expected<FloatFormat> scalarFloatFormat(mlir::Type type) {
  auto floating = llvm::dyn_cast<mlir::FloatType>(type);
  if (!floating)
    return reject("scalar floating behavior has a non-floating endpoint");
  if (floating.isF16())
    return FloatFormat::F16;
  if (floating.isBF16())
    return FloatFormat::BF16;
  if (floating.isF32())
    return FloatFormat::F32;
  if (floating.isF64())
    return FloatFormat::F64;
  return reject("scalar floating behavior has an unsupported format");
}

llvm::Expected<IntegerWidth> scalarIntegerWidth(mlir::Type type) {
  auto integer = llvm::dyn_cast<mlir::IntegerType>(type);
  if (!integer || !integer.isSignless())
    return reject("floating conversion has a non-signless integer endpoint");
  switch (integer.getWidth()) {
  case 8:
    return IntegerWidth::I8;
  case 16:
    return IntegerWidth::I16;
  case 32:
    return IntegerWidth::I32;
  case 64:
    return IntegerWidth::I64;
  default:
    return reject("floating conversion has an unsupported integer width");
  }
}

llvm::Expected<::loom::CanonicalSemanticBytes>
encodeFormat(mlir::MLIRContext &context, FloatFormat format) {
  return ::dataflow::encodeCanonicalType(floatType(context, format));
}

llvm::Error appendCandidate(
    std::vector<ScalarFloatBehaviorCandidate> &candidates,
    ::dataflow::CanonicalActorSchemaProjection actor, llvm::StringRef role,
    std::vector<BehaviorComponent> components,
    std::optional<::mlir::arith::RoundingMode> selectedRounding = std::nullopt,
    unsigned refinementStrength = 0, bool nonNegativeRelaxation = false) {
  auto canonical = ::dataflow::encodeCanonicalActorSchemaProjection(actor);
  if (!canonical)
    return canonical.takeError();
  std::vector<std::uint8_t> canonicalActorBytes(canonical->bytes().begin(),
                                                canonical->bytes().end());
  std::vector<std::uint64_t> operandPorts =
      identityPorts(actor.type.getNumInputs());
  std::vector<std::uint64_t> resultPorts =
      identityPorts(actor.type.getNumResults());
  candidates.push_back({std::move(actor), role, std::move(components),
                        std::move(operandPorts), std::move(resultPorts),
                        std::move(canonicalActorBytes), selectedRounding,
                        refinementStrength, nonNegativeRelaxation});
  return llvm::Error::success();
}

llvm::Error validateFloatParameterDomain(const FloatFormatSet &formats,
                                         const FloatBehaviorProfile &behavior) {
  if (!formats.valid() || formats.empty())
    return reject("scalar floating format domain is invalid");
  if (!behavior.roundingModes.valid() || behavior.roundingModes.empty())
    return reject("scalar floating rounding domain is invalid");
  if (!behavior.nanBehaviors.valid() || behavior.nanBehaviors.empty())
    return reject("scalar floating NaN domain is invalid");
  if (behavior.nanBehaviors.size() != 1)
    return reject("scalar non-compare floating NaN behavior has no projector "
                  "image");
  if (!behavior.subnormalBehaviors.valid() ||
      behavior.subnormalBehaviors.empty())
    return reject("scalar floating subnormal domain is invalid");
  if (!behavior.signedZeroBehaviors.valid() ||
      behavior.signedZeroBehaviors.empty())
    return reject("scalar floating signed-zero domain is invalid");
  return llvm::Error::success();
}

struct UniformBehaviorShape final {
  llvm::StringRef role;
  unsigned inputCount = 0;
  bool representationWidth = false;
  bool exactFormat = false;
  bool rounding = false;
};

llvm::Expected<UniformBehaviorShape>
describeUniformShape(ImplementationFamilyId family,
                     ::dataflow::OperationSchemaId schema) {
  using Family = ImplementationFamilyId;
  using Schema = ::dataflow::OperationSchemaId;
  switch (family) {
  case Family::ScalarFloatSign:
    if (schema == Schema::ArithNegF)
      return UniformBehaviorShape{"Negate", 1, true, false, false};
    if (schema == Schema::MathAbsF)
      return UniformBehaviorShape{"Absolute", 1, true, false, false};
    return reject("scalar float sign capability contains a foreign schema");
  case Family::ScalarFloatAddSub:
    if (schema == Schema::ArithAddF)
      return UniformBehaviorShape{"Add", 2, false, true, true};
    if (schema == Schema::ArithSubF)
      return UniformBehaviorShape{"Sub", 2, false, true, true};
    return reject("scalar float add/sub capability contains a foreign schema");
  case Family::ScalarFloatMultiply:
    if (schema == Schema::ArithMulF)
      return UniformBehaviorShape{"", 2, false, true, true};
    return reject("scalar float multiply capability contains a foreign schema");
  case Family::ScalarFloatFma:
    if (schema == Schema::MathFma)
      return UniformBehaviorShape{"", 3, false, true, true};
    return reject("scalar float FMA capability contains a foreign schema");
  case Family::ScalarFloatDivide:
    if (schema == Schema::ArithDivF)
      return UniformBehaviorShape{"", 2, false, true, true};
    return reject("scalar float divide capability contains a foreign schema");
  case Family::ScalarFloatRemainder:
    if (schema == Schema::ArithRemF)
      return UniformBehaviorShape{"", 2, false, true, false};
    return reject("scalar float remainder capability contains a foreign "
                  "schema");
  default:
    return reject("family is not a uniform scalar floating quotient");
  }
}

llvm::Error appendUniformCandidates(
    ImplementationFamilyId family, const ScalarFloatParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> orderedSchemas,
    mlir::MLIRContext &context,
    std::vector<ScalarFloatBehaviorCandidate> &candidates) {
  if (llvm::Error error =
          validateFloatParameterDomain(params.formats, params.behavior))
    return error;
  const mlir::arith::FastMathFlags actorFlags =
      detail::minimalFloatingActorPermissions(params.behavior);

  bool selectsRounding = false;
  for (::dataflow::OperationSchemaId schema : orderedSchemas) {
    auto shape = describeUniformShape(family, schema);
    if (!shape)
      return shape.takeError();
    selectsRounding |= shape->rounding;
    const std::size_t begin = candidates.size();
    for (FloatFormat format : floatFormatDomain) {
      if (!params.formats.contains(format))
        continue;
      auto formatBytes = encodeFormat(context, format);
      if (!formatBytes)
        return formatBytes.takeError();

      const auto append =
          [&](std::optional<mlir::arith::RoundingMode> actorRounding,
              std::optional<mlir::arith::RoundingMode> selectedRounding)
          -> llvm::Error {
        mlir::Type type = floatType(context, format);
        std::vector<mlir::Type> inputs(shape->inputCount, type);
        std::vector<BehaviorComponent> components;
        if (shape->representationWidth)
          components.push_back({BehaviorComponentSlot::RepresentationWidth,
                                getBitWidth(format)});
        if (shape->exactFormat)
          components.push_back({BehaviorComponentSlot::Format, *formatBytes});
        if (selectedRounding) {
          auto encoded = ::dataflow::encodeRoundingMode(*selectedRounding);
          if (!encoded)
            return encoded.takeError();
          components.push_back(
              {BehaviorComponentSlot::Rounding, std::move(*encoded)});
        }
        return appendCandidate(
            candidates,
            {schema, mlir::FunctionType::get(&context, inputs, {type}),
             ::dataflow::FloatingPointPayload{actorFlags, actorRounding}},
            shape->role, std::move(components), selectedRounding, 1);
      };

      if (!shape->rounding) {
        if (llvm::Error error = append(std::nullopt, std::nullopt))
          return error;
        continue;
      }
      for (std::uint32_t ordinal = 0;
           ordinal <= mlir::arith::getMaxEnumValForRoundingMode(); ++ordinal) {
        auto mode = static_cast<mlir::arith::RoundingMode>(ordinal);
        if (!params.behavior.roundingModes.contains(mode))
          continue;
        std::optional<mlir::arith::RoundingMode> actorRounding = mode;
        if (mode == mlir::arith::RoundingMode::to_nearest_even)
          actorRounding = std::nullopt;
        if (llvm::Error error = append(actorRounding, mode))
          return error;
      }
    }
    if (candidates.size() == begin)
      return reject("enabled scalar floating schema has no admitted behavior");
  }
  if (!selectsRounding && params.behavior.roundingModes.size() != 1)
    return reject("scalar floating rounding behavior has no projector image");
  return llvm::Error::success();
}

llvm::Error appendWidthCastCandidates(
    const ScalarFloatWidthCastParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> orderedSchemas,
    mlir::MLIRContext &context,
    std::vector<ScalarFloatBehaviorCandidate> &candidates) {
  if (!params.formatPairs.valid() || params.formatPairs.empty())
    return reject("scalar floating cast relation is invalid");
  FloatFormatSet formats;
  for (FloatFormat source : floatFormatDomain)
    for (FloatFormat destination : floatFormatDomain)
      if (params.formatPairs.contains(source, destination)) {
        formats.insert(source);
        formats.insert(destination);
      }
  if (llvm::Error error =
          validateFloatParameterDomain(formats, params.behavior))
    return error;
  const mlir::arith::FastMathFlags actorFlags =
      detail::minimalFloatingActorPermissions(params.behavior);

  std::vector<std::pair<FloatFormat, FloatFormat>> usedPairs;
  bool selectsRounding = false;
  for (::dataflow::OperationSchemaId schema : orderedSchemas) {
    const bool extension = schema == ::dataflow::OperationSchemaId::ArithExtF;
    const bool truncation =
        schema == ::dataflow::OperationSchemaId::ArithTruncF;
    if (!extension && !truncation)
      return reject("scalar floating cast capability contains a foreign "
                    "schema");
    selectsRounding |= truncation;
    const std::size_t begin = candidates.size();
    for (FloatFormat source : floatFormatDomain) {
      for (FloatFormat destination : floatFormatDomain) {
        if (!params.formatPairs.contains(source, destination))
          continue;
        const bool direction =
            extension ? getBitWidth(source) < getBitWidth(destination)
                      : getBitWidth(source) > getBitWidth(destination);
        if (!direction)
          continue;
        usedPairs.emplace_back(source, destination);
        auto sourceBytes = encodeFormat(context, source);
        if (!sourceBytes)
          return sourceBytes.takeError();
        auto destinationBytes = encodeFormat(context, destination);
        if (!destinationBytes)
          return destinationBytes.takeError();

        const auto append =
            [&](std::optional<mlir::arith::RoundingMode> actorRounding,
                std::optional<mlir::arith::RoundingMode> selectedRounding)
            -> llvm::Error {
          std::vector<BehaviorComponent> components;
          components.push_back(
              {BehaviorComponentSlot::SourceFormat, *sourceBytes});
          components.push_back(
              {BehaviorComponentSlot::DestinationFormat, *destinationBytes});
          if (selectedRounding) {
            auto encoded = ::dataflow::encodeRoundingMode(*selectedRounding);
            if (!encoded)
              return encoded.takeError();
            components.push_back(
                {BehaviorComponentSlot::Rounding, std::move(*encoded)});
          }
          return appendCandidate(
              candidates,
              {schema,
               mlir::FunctionType::get(&context, {floatType(context, source)},
                                       {floatType(context, destination)}),
               ::dataflow::FloatingPointPayload{actorFlags, actorRounding}},
              "", std::move(components), selectedRounding, 1);
        };

        if (extension) {
          if (llvm::Error error = append(std::nullopt, std::nullopt))
            return error;
          continue;
        }
        for (std::uint32_t ordinal = 0;
             ordinal <= mlir::arith::getMaxEnumValForRoundingMode();
             ++ordinal) {
          auto mode = static_cast<mlir::arith::RoundingMode>(ordinal);
          if (!params.behavior.roundingModes.contains(mode))
            continue;
          std::optional<mlir::arith::RoundingMode> actorRounding = mode;
          if (mode == mlir::arith::RoundingMode::to_nearest_even)
            actorRounding = std::nullopt;
          if (llvm::Error error = append(actorRounding, mode))
            return error;
        }
      }
    }
    if (candidates.size() == begin)
      return reject("enabled scalar floating cast schema has no admitted "
                    "behavior");
  }
  for (FloatFormat source : floatFormatDomain)
    for (FloatFormat destination : floatFormatDomain)
      if (params.formatPairs.contains(source, destination) &&
          !llvm::is_contained(usedPairs, std::pair{source, destination}))
        return reject("scalar floating cast relation contains an orphan "
                      "format pair");
  if (!selectsRounding && params.behavior.roundingModes.size() != 1)
    return reject("scalar floating cast rounding behavior has no projector "
                  "image");
  return llvm::Error::success();
}

llvm::Error appendConversionCandidates(
    ImplementationFamilyId family,
    const ScalarIntegerFloatConversionParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> orderedSchemas,
    mlir::MLIRContext &context,
    std::vector<ScalarFloatBehaviorCandidate> &candidates) {
  if (!params.formatPairs.valid() || params.formatPairs.empty())
    return reject("scalar floating conversion relation is invalid");

  using Family = ImplementationFamilyId;
  using Schema = ::dataflow::OperationSchemaId;
  for (Schema schema : orderedSchemas) {
    const std::size_t begin = candidates.size();
    for (IntegerWidth integer : integerWidthDomain) {
      if (integer == IntegerWidth::I1)
        continue;
      for (FloatFormat format : floatFormatDomain) {
        if (!params.formatPairs.contains(integer, format))
          continue;
        auto formatBytes = encodeFormat(context, format);
        if (!formatBytes)
          return formatBytes.takeError();
        mlir::Type integerEndpoint = integerType(context, integer);
        mlir::Type floatEndpoint = floatType(context, format);
        std::vector<BehaviorComponent> components;
        llvm::StringRef role;
        ::dataflow::SemanticPayload payload = ::dataflow::NoPayload{};
        mlir::Type input;
        mlir::Type result;
        unsigned strength = 1;

        if (family == Family::ScalarIntegerToFloat) {
          input = integerEndpoint;
          result = floatEndpoint;
          components.push_back(
              {BehaviorComponentSlot::SourceWidth, getBitWidth(integer)});
          components.push_back(
              {BehaviorComponentSlot::DestinationFormat, *formatBytes});
          if (schema == Schema::ArithSIToFP) {
            role = "Signed";
          } else if (schema == Schema::ArithUIToFP) {
            role = "Unsigned";
            payload = ::dataflow::NonNegativePayload{false};
          } else {
            return reject("integer-to-float capability contains a foreign "
                          "schema");
          }
        } else {
          input = floatEndpoint;
          result = integerEndpoint;
          components.push_back(
              {BehaviorComponentSlot::SourceFormat, *formatBytes});
          components.push_back(
              {BehaviorComponentSlot::DestinationWidth, getBitWidth(integer)});
          if (schema == Schema::ArithFPToSI || schema == Schema::LLVMFPToSISat)
            role = "Signed";
          else if (schema == Schema::ArithFPToUI ||
                   schema == Schema::LLVMFPToUISat)
            role = "Unsigned";
          else
            return reject("float-to-integer capability contains a foreign "
                          "schema");
          const bool saturating = schema == Schema::LLVMFPToSISat ||
                                  schema == Schema::LLVMFPToUISat;
          strength = saturating ? 3 : 1;
        }

        if (llvm::Error error = appendCandidate(
                candidates,
                {schema, mlir::FunctionType::get(&context, {input}, {result}),
                 payload},
                role, components, std::nullopt, strength))
          return error;
        if (schema == Schema::ArithUIToFP) {
          payload = ::dataflow::NonNegativePayload{true};
          if (llvm::Error error = appendCandidate(
                  candidates,
                  {schema, mlir::FunctionType::get(&context, {input}, {result}),
                   payload},
                  role, std::move(components), std::nullopt, 0, true))
            return error;
        }
      }
    }
    if (candidates.size() == begin)
      return reject("enabled scalar floating conversion schema has no "
                    "admitted behavior");
  }
  return llvm::Error::success();
}

llvm::Expected<bool>
isPhysicallyReachable(const ScalarFloatBehaviorCandidate &candidate,
                      llvm::ArrayRef<std::uint32_t> physicalInputWidths,
                      llvm::ArrayRef<std::uint32_t> physicalResultWidths) {
  const auto fits =
      [](llvm::ArrayRef<mlir::Type> types,
         llvm::ArrayRef<std::uint32_t> widths) -> llvm::Expected<bool> {
    if (types.size() > widths.size())
      return false;
    for (auto [type, width] : llvm::zip(types, widths)) {
      std::string message;
      auto semanticWidth = getSemanticPayloadWidth(type, message);
      if (mlir::failed(semanticWidth))
        return reject(message);
      if (*semanticWidth > width)
        return false;
    }
    return true;
  };
  auto inputs = fits(candidate.actor.type.getInputs(), physicalInputWidths);
  if (!inputs || !*inputs)
    return inputs;
  return fits(candidate.actor.type.getResults(), physicalResultWidths);
}

llvm::Expected<::loom::CanonicalSemanticBytes>
encodeCandidate(ImplementationFamilyId family,
                llvm::ArrayRef<ScalarFloatBehaviorCandidate> candidates,
                const ScalarFloatBehaviorCandidate &candidate) {
  std::vector<detail::ImplementationFamilyBehaviorKeyComponent> components;
  for (const BehaviorComponent &component : candidate.components) {
    if (!componentVaries(candidates, candidate.role, component.slot))
      continue;
    if (const auto *width = std::get_if<std::uint32_t>(&component.value))
      components.emplace_back(*width);
    else
      components.emplace_back(
          std::get<::loom::CanonicalSemanticBytes>(component.value));
  }
  return detail::encodeImplementationFamilyBehaviorKey(family, candidate.role,
                                                       components);
}

llvm::Expected<const ::dataflow::FloatingPointPayload *> requireFloatingPayload(
    const ::dataflow::CanonicalActorSchemaProjection &actor) {
  const auto *payload =
      std::get_if<::dataflow::FloatingPointPayload>(&actor.payload);
  if (!payload)
    return reject("scalar floating actor has the wrong semantic payload");
  if (!::dataflow::isValidFastMathFlags(payload->flags))
    return reject("scalar floating actor has invalid fast-math flags");
  return payload;
}

llvm::Expected<ScalarFloatBehaviorCandidate>
describeActorBehavior(ImplementationFamilyId family,
                      const ::dataflow::CanonicalActorSchemaProjection &actor) {
  std::vector<BehaviorComponent> components;
  llvm::StringRef role;
  bool nonNegativeRelaxation = false;

  if (family == ImplementationFamilyId::ScalarFloatWidthCast) {
    if (actor.type.getNumInputs() != 1 || actor.type.getNumResults() != 1)
      return reject("scalar floating cast behavior has the wrong arity");
    auto payload = requireFloatingPayload(actor);
    if (!payload)
      return payload.takeError();
    auto source = scalarFloatFormat(actor.type.getInput(0));
    if (!source)
      return source.takeError();
    auto destination = scalarFloatFormat(actor.type.getResult(0));
    if (!destination)
      return destination.takeError();
    const bool extension =
        actor.schema == ::dataflow::OperationSchemaId::ArithExtF;
    const bool truncation =
        actor.schema == ::dataflow::OperationSchemaId::ArithTruncF;
    if ((!extension && !truncation) ||
        (extension && getBitWidth(*source) >= getBitWidth(*destination)) ||
        (truncation && getBitWidth(*source) <= getBitWidth(*destination)))
      return reject("actor has no scalar floating cast behavior");
    auto sourceBytes = encodeFormat(*actor.type.getContext(), *source);
    if (!sourceBytes)
      return sourceBytes.takeError();
    auto destinationBytes =
        encodeFormat(*actor.type.getContext(), *destination);
    if (!destinationBytes)
      return destinationBytes.takeError();
    components.push_back(
        {BehaviorComponentSlot::SourceFormat, std::move(*sourceBytes)});
    components.push_back({BehaviorComponentSlot::DestinationFormat,
                          std::move(*destinationBytes)});
    if (truncation) {
      auto rounding = (*payload)->roundingMode.value_or(
          mlir::arith::RoundingMode::to_nearest_even);
      auto encoded = ::dataflow::encodeRoundingMode(rounding);
      if (!encoded)
        return encoded.takeError();
      components.push_back(
          {BehaviorComponentSlot::Rounding, std::move(*encoded)});
    }
  } else if (family == ImplementationFamilyId::ScalarIntegerToFloat ||
             family == ImplementationFamilyId::ScalarFloatToInteger) {
    if (actor.type.getNumInputs() != 1 || actor.type.getNumResults() != 1)
      return reject("scalar floating conversion behavior has the wrong arity");
    mlir::Type integerEndpoint;
    mlir::Type floatEndpoint;
    using Schema = ::dataflow::OperationSchemaId;
    if (family == ImplementationFamilyId::ScalarIntegerToFloat) {
      integerEndpoint = actor.type.getInput(0);
      floatEndpoint = actor.type.getResult(0);
      if (actor.schema == Schema::ArithSIToFP) {
        role = "Signed";
        if (!std::holds_alternative<::dataflow::NoPayload>(actor.payload))
          return reject("signed conversion has the wrong semantic payload");
      } else if (actor.schema == Schema::ArithUIToFP) {
        role = "Unsigned";
        const auto *payload =
            std::get_if<::dataflow::NonNegativePayload>(&actor.payload);
        if (!payload)
          return reject("unsigned conversion has the wrong semantic payload");
        nonNegativeRelaxation = payload->isNonNegative;
      } else {
        return reject("actor has no integer-to-float behavior");
      }
    } else {
      integerEndpoint = actor.type.getResult(0);
      floatEndpoint = actor.type.getInput(0);
      if (!std::holds_alternative<::dataflow::NoPayload>(actor.payload))
        return reject("float-to-integer conversion has the wrong semantic "
                      "payload");
      if (actor.schema == Schema::ArithFPToSI ||
          actor.schema == Schema::LLVMFPToSISat)
        role = "Signed";
      else if (actor.schema == Schema::ArithFPToUI ||
               actor.schema == Schema::LLVMFPToUISat)
        role = "Unsigned";
      else
        return reject("actor has no float-to-integer behavior");
    }
    auto integer = scalarIntegerWidth(integerEndpoint);
    if (!integer)
      return integer.takeError();
    auto format = scalarFloatFormat(floatEndpoint);
    if (!format)
      return format.takeError();
    auto formatBytes = encodeFormat(*actor.type.getContext(), *format);
    if (!formatBytes)
      return formatBytes.takeError();
    if (family == ImplementationFamilyId::ScalarIntegerToFloat) {
      components.push_back(
          {BehaviorComponentSlot::SourceWidth, getBitWidth(*integer)});
      components.push_back(
          {BehaviorComponentSlot::DestinationFormat, std::move(*formatBytes)});
    } else {
      components.push_back(
          {BehaviorComponentSlot::SourceFormat, std::move(*formatBytes)});
      components.push_back(
          {BehaviorComponentSlot::DestinationWidth, getBitWidth(*integer)});
    }
  } else {
    auto shape = describeUniformShape(family, actor.schema);
    if (!shape)
      return shape.takeError();
    if (actor.type.getNumInputs() != shape->inputCount ||
        actor.type.getNumResults() != 1)
      return reject("uniform scalar floating behavior has the wrong arity");
    mlir::Type type = actor.type.getInput(0);
    for (mlir::Type input : actor.type.getInputs())
      if (input != type)
        return reject("uniform scalar floating inputs do not agree");
    if (actor.type.getResult(0) != type)
      return reject("uniform scalar floating result does not agree");
    auto format = scalarFloatFormat(type);
    if (!format)
      return format.takeError();
    auto payload = requireFloatingPayload(actor);
    if (!payload)
      return payload.takeError();
    role = shape->role;
    if (shape->representationWidth)
      components.push_back(
          {BehaviorComponentSlot::RepresentationWidth, getBitWidth(*format)});
    if (shape->exactFormat) {
      auto encoded = encodeFormat(*actor.type.getContext(), *format);
      if (!encoded)
        return encoded.takeError();
      components.push_back(
          {BehaviorComponentSlot::Format, std::move(*encoded)});
    }
    if (shape->rounding) {
      auto rounding = (*payload)->roundingMode.value_or(
          mlir::arith::RoundingMode::to_nearest_even);
      auto encoded = ::dataflow::encodeRoundingMode(rounding);
      if (!encoded)
        return encoded.takeError();
      components.push_back(
          {BehaviorComponentSlot::Rounding, std::move(*encoded)});
    }
  }

  return ScalarFloatBehaviorCandidate{
      actor,        role, std::move(components), {}, {}, {},
      std::nullopt, 0,    nonNegativeRelaxation};
}

bool sameBehavior(const ScalarFloatBehaviorCandidate &lhs,
                  const ScalarFloatBehaviorCandidate &rhs) {
  if (lhs.role != rhs.role || lhs.components.size() != rhs.components.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs.components, rhs.components))
    if (left.slot != right.slot ||
        !equalComponentValue(left.value, right.value))
      return false;
  return true;
}

bool lessKey(const FiniteImplementationFamilyBehaviorPoint &lhs,
             const FiniteImplementationFamilyBehaviorPoint &rhs) {
  return std::lexicographical_compare(
      lhs.semanticConfiguration->bytes().begin(),
      lhs.semanticConfiguration->bytes().end(),
      rhs.semanticConfiguration->bytes().begin(),
      rhs.semanticConfiguration->bytes().end());
}

} // namespace

bool fabric::detail::ownsScalarFloatBehaviorRelation(
    ImplementationFamilyId family) {
  switch (family) {
  case ImplementationFamilyId::ScalarFloatSign:
  case ImplementationFamilyId::ScalarFloatAddSub:
  case ImplementationFamilyId::ScalarFloatWidthCast:
  case ImplementationFamilyId::ScalarIntegerToFloat:
  case ImplementationFamilyId::ScalarFloatToInteger:
  case ImplementationFamilyId::ScalarFloatMultiply:
  case ImplementationFamilyId::ScalarFloatFma:
  case ImplementationFamilyId::ScalarFloatDivide:
  case ImplementationFamilyId::ScalarFloatRemainder:
    return true;
  default:
    return false;
  }
}

llvm::Expected<std::vector<fabric::FiniteImplementationFamilyBehaviorPoint>>
fabric::detail::resolveScalarFloatBehaviorDomain(
    ImplementationFamilyId family, const FamilyCapabilityParams &params,
    llvm::ArrayRef<::dataflow::OperationSchemaId> enabledSchemas,
    llvm::ArrayRef<std::uint32_t> physicalInputWidths,
    llvm::ArrayRef<std::uint32_t> physicalResultWidths,
    mlir::MLIRContext &context) {
  if (!ownsScalarFloatBehaviorRelation(family))
    return reject("family has no scalar floating behavior relation");
  if (enabledSchemas.empty())
    return reject("scalar floating capability has no enabled schema");
  const ImplementationFamilyDescriptor &descriptor =
      implementationFamily(family);
  if (capabilityParamsSchema(params) != descriptor.capabilityParamsSchema)
    return reject("capability parameter schema does not match its family");
  for (auto [ordinal, schema] : llvm::enumerate(enabledSchemas)) {
    if (!llvm::is_contained(descriptor.admittedSchemas, schema))
      return reject("scalar floating capability enables a foreign schema");
    if (llvm::is_contained(enabledSchemas.take_front(ordinal), schema))
      return reject("scalar floating capability enables a schema twice");
  }

  std::vector<::dataflow::OperationSchemaId> orderedSchemas;
  for (::dataflow::OperationSchemaId schema : descriptor.admittedSchemas)
    if (llvm::is_contained(enabledSchemas, schema))
      orderedSchemas.push_back(schema);

  std::vector<ScalarFloatBehaviorCandidate> candidates;
  if (family == ImplementationFamilyId::ScalarFloatWidthCast) {
    const auto *typed = std::get_if<ScalarFloatWidthCastParams>(&params);
    if (!typed)
      return reject("capability has the wrong scalar floating cast parameter "
                    "schema");
    if (llvm::Error error = appendWidthCastCandidates(*typed, orderedSchemas,
                                                      context, candidates))
      return std::move(error);
  } else if (family == ImplementationFamilyId::ScalarIntegerToFloat ||
             family == ImplementationFamilyId::ScalarFloatToInteger) {
    const auto *typed =
        std::get_if<ScalarIntegerFloatConversionParams>(&params);
    if (!typed)
      return reject("capability has the wrong scalar floating conversion "
                    "parameter schema");
    if (llvm::Error error = appendConversionCandidates(
            family, *typed, orderedSchemas, context, candidates))
      return std::move(error);
  } else {
    const auto *typed = std::get_if<ScalarFloatParams>(&params);
    if (!typed)
      return reject("capability has the wrong scalar floating parameter "
                    "schema");
    if (llvm::Error error = appendUniformCandidates(
            family, *typed, orderedSchemas, context, candidates))
      return std::move(error);
  }
  if (candidates.empty())
    return reject("scalar floating capability has no behavior candidate");

  std::vector<ScalarFloatBehaviorCandidate> reachable;
  for (ScalarFloatBehaviorCandidate &candidate : candidates) {
    if (llvm::Error error = verifyImplementationFamilyAdmission(
            family, &params, candidate.actor))
      return std::move(error);
    if (llvm::Error error = verifyImplementationFamilyPortCorrespondence(
            family, candidate.actor, candidate.operandPorts,
            candidate.resultPorts))
      return std::move(error);
    auto physical = isPhysicallyReachable(candidate, physicalInputWidths,
                                          physicalResultWidths);
    if (!physical)
      return physical.takeError();
    if (*physical)
      reachable.push_back(std::move(candidate));
  }
  if (reachable.empty())
    return reject(
        "scalar floating capability has no physically reachable behavior");
  for (::dataflow::OperationSchemaId schema : orderedSchemas)
    if (!llvm::any_of(reachable, [&](const auto &candidate) {
          return candidate.actor.schema == schema;
        }))
      return reject("enabled scalar floating schema has no physically "
                    "reachable behavior");

  const FloatBehaviorProfile *behavior = nullptr;
  if (const auto *typed = std::get_if<ScalarFloatParams>(&params))
    behavior = &typed->behavior;
  else if (const auto *typed = std::get_if<ScalarFloatWidthCastParams>(&params))
    behavior = &typed->behavior;
  if (behavior && behavior->roundingModes.size() > 1) {
    for (std::uint32_t ordinal = 0;
         ordinal <= mlir::arith::getMaxEnumValForRoundingMode(); ++ordinal) {
      auto mode = static_cast<mlir::arith::RoundingMode>(ordinal);
      if (!behavior->roundingModes.contains(mode))
        continue;
      if (!llvm::any_of(reachable, [&](const auto &candidate) {
            return candidate.selectedRounding == mode;
          }))
        return reject("scalar floating rounding behavior has no reachable "
                      "projector image");
    }
  }

  llvm::sort(reachable, [](const auto &lhs, const auto &rhs) {
    if (lhs.refinementStrength != rhs.refinementStrength)
      return lhs.refinementStrength > rhs.refinementStrength;
    return std::lexicographical_compare(
        lhs.canonicalActorBytes.begin(), lhs.canonicalActorBytes.end(),
        rhs.canonicalActorBytes.begin(), rhs.canonicalActorBytes.end());
  });

  std::vector<FiniteImplementationFamilyBehaviorPoint> points;
  for (ScalarFloatBehaviorCandidate &candidate : reachable) {
    auto key = encodeCandidate(family, reachable, candidate);
    if (!key)
      return key.takeError();
    if (llvm::any_of(points, [&](const auto &point) {
          return point.semanticConfiguration &&
                 point.semanticConfiguration->bytes().equals(key->bytes());
        }))
      continue;
    points.push_back({std::move(candidate.actor), std::move(*key), std::nullopt,
                      std::move(candidate.operandPorts),
                      std::move(candidate.resultPorts)});
  }
  llvm::sort(points, lessKey);
  if (points.size() == 1)
    points.front().semanticConfiguration = std::nullopt;
  return points;
}

llvm::Expected<::loom::CanonicalSemanticBytes>
fabric::detail::projectScalarFloatBehavior(
    ImplementationFamilyId family,
    const ::dataflow::CanonicalActorSchemaProjection &actor,
    llvm::ArrayRef<FiniteImplementationFamilyBehaviorPoint> domain) {
  if (!ownsScalarFloatBehaviorRelation(family))
    return reject("capability family has no scalar floating projector");
  auto projected = describeActorBehavior(family, actor);
  if (!projected)
    return projected.takeError();

  const auto find = [&](const ScalarFloatBehaviorCandidate &behavior)
      -> llvm::Expected<std::optional<::loom::CanonicalSemanticBytes>> {
    for (const FiniteImplementationFamilyBehaviorPoint &point : domain) {
      auto witness = describeActorBehavior(family, point.representativeActor);
      if (!witness)
        return witness.takeError();
      if (!sameBehavior(behavior, *witness))
        continue;
      if (!point.semanticConfiguration)
        return reject("scalar floating relation has no semantic field");
      return *point.semanticConfiguration;
    }
    return std::nullopt;
  };

  auto matched = find(*projected);
  if (!matched)
    return matched.takeError();
  if (*matched)
    return std::move(**matched);
  if (projected->nonNegativeRelaxation && projected->role == "Unsigned") {
    projected->role = "Signed";
    matched = find(*projected);
    if (!matched)
      return matched.takeError();
    if (*matched)
      return std::move(**matched);
  }
  return reject("actor is outside the scalar floating behavior image");
}

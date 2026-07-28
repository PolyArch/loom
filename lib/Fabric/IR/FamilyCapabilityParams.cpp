//===- FamilyCapabilityParams.cpp - Closed Fabric capability records ------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <array>
#include <optional>

using namespace fabric;
using namespace mlir;

namespace {

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(message);
}

llvm::Error checkFields(DictionaryAttr params,
                        llvm::ArrayRef<llvm::StringRef> required,
                        llvm::ArrayRef<llvm::StringRef> optional = {}) {
  llvm::StringSet<> allowed;
  for (llvm::StringRef field : required)
    allowed.insert(field);
  for (llvm::StringRef field : optional)
    allowed.insert(field);
  for (NamedAttribute field : params) {
    llvm::StringRef name = field.getName().getValue();
    if (!allowed.contains(name))
      return reject("hw_params contains unknown field '" + name + "'");
  }
  for (llvm::StringRef field : required) {
    if (!params.get(field))
      return reject("hw_params is missing required field '" + field + "'");
  }
  return llvm::Error::success();
}

llvm::Expected<ArrayAttr> requireArray(DictionaryAttr params,
                                       llvm::StringRef field) {
  auto value = dyn_cast_or_null<ArrayAttr>(params.get(field));
  if (!value)
    return reject("hw_params field '" + field + "' must be an array");
  return value;
}

llvm::Expected<StringAttr> requireString(DictionaryAttr params,
                                         llvm::StringRef field) {
  auto value = dyn_cast_or_null<StringAttr>(params.get(field));
  if (!value)
    return reject("hw_params field '" + field + "' must be a string");
  return value;
}

llvm::Expected<std::uint32_t> requirePositiveU32(DictionaryAttr params,
                                                 llvm::StringRef field) {
  auto value = dyn_cast_or_null<IntegerAttr>(params.get(field));
  if (!value || value.getValue().isNegative() ||
      value.getValue().getActiveBits() > 32 || value.getValue().isZero())
    return reject("hw_params field '" + field + "' must be a positive uint32");
  return static_cast<std::uint32_t>(value.getValue().getZExtValue());
}

llvm::Expected<IntegerWidth> parseIntegerWidth(Attribute attr,
                                               llvm::StringRef field) {
  auto integer = dyn_cast<IntegerAttr>(attr);
  if (!integer || integer.getValue().isNegative() ||
      integer.getValue().getActiveBits() > 32)
    return reject("hw_params field '" + field +
                  "' must contain integer widths");
  switch (integer.getValue().getZExtValue()) {
  case 1:
    return IntegerWidth::I1;
  case 8:
    return IntegerWidth::I8;
  case 16:
    return IntegerWidth::I16;
  case 32:
    return IntegerWidth::I32;
  case 64:
    return IntegerWidth::I64;
  default:
    return reject("hw_params field '" + field +
                  "' contains an unsupported integer width");
  }
}

llvm::Expected<IntegerWidthSet> parseIntegerWidths(DictionaryAttr params,
                                                   llvm::StringRef field,
                                                   bool allowEmpty = false) {
  auto values = requireArray(params, field);
  if (!values)
    return values.takeError();
  IntegerWidthSet widths;
  for (Attribute value : *values) {
    auto width = parseIntegerWidth(value, field);
    if (!width)
      return width.takeError();
    if (widths.contains(*width))
      return reject("hw_params field '" + field +
                    "' contains a duplicate integer width");
    widths.insert(*width);
  }
  if (widths.empty() && !allowEmpty)
    return reject("hw_params field '" + field + "' must not be empty");
  return widths;
}

llvm::Expected<FloatFormat> parseFloatFormat(Attribute attr,
                                             llvm::StringRef field) {
  auto spelling = dyn_cast<StringAttr>(attr);
  if (!spelling)
    return reject("hw_params field '" + field +
                  "' must contain floating-format strings");
  std::optional<FloatFormat> format =
      llvm::StringSwitch<std::optional<FloatFormat>>(spelling.getValue())
          .Case("f16", FloatFormat::F16)
          .Case("bf16", FloatFormat::BF16)
          .Case("f32", FloatFormat::F32)
          .Case("f64", FloatFormat::F64)
          .Default(std::nullopt);
  if (!format)
    return reject("hw_params field '" + field +
                  "' contains an unsupported floating format");
  return *format;
}

llvm::Expected<FloatFormatSet> parseFloatFormats(DictionaryAttr params,
                                                 llvm::StringRef field) {
  auto values = requireArray(params, field);
  if (!values)
    return values.takeError();
  FloatFormatSet formats;
  for (Attribute value : *values) {
    auto format = parseFloatFormat(value, field);
    if (!format)
      return format.takeError();
    if (formats.contains(*format))
      return reject("hw_params field '" + field +
                    "' contains a duplicate floating format");
    formats.insert(*format);
  }
  return formats;
}

template <typename Enum, typename Set, typename Symbolizer>
llvm::Expected<Set> parseEnumSet(DictionaryAttr params, llvm::StringRef field,
                                 Symbolizer symbolize) {
  auto values = requireArray(params, field);
  if (!values)
    return values.takeError();
  Set result;
  for (Attribute value : *values) {
    auto spelling = dyn_cast<StringAttr>(value);
    if (!spelling)
      return reject("hw_params field '" + field +
                    "' must contain enum strings");
    std::optional<Enum> parsed = symbolize(spelling.getValue());
    if (!parsed)
      return reject("hw_params field '" + field +
                    "' contains an unsupported enum value");
    if (result.contains(*parsed))
      return reject("hw_params field '" + field +
                    "' contains a duplicate enum value");
    result.insert(*parsed);
  }
  if (result.empty())
    return reject("hw_params field '" + field + "' must not be empty");
  return result;
}

llvm::Expected<FloatNaNBehaviorSet> parseNaNBehaviors(DictionaryAttr params,
                                                      llvm::StringRef field) {
  return parseEnumSet<FloatNaNBehavior, FloatNaNBehaviorSet>(
      params, field, [](llvm::StringRef spelling) {
        return llvm::StringSwitch<std::optional<FloatNaNBehavior>>(spelling)
            .Case("ieee", FloatNaNBehavior::IEEE)
            .Case("number_preferred", FloatNaNBehavior::NumberPreferred)
            .Default(std::nullopt);
      });
}

llvm::Expected<FloatSubnormalBehaviorSet>
parseSubnormalBehaviors(DictionaryAttr params, llvm::StringRef field) {
  return parseEnumSet<FloatSubnormalBehavior, FloatSubnormalBehaviorSet>(
      params, field, [](llvm::StringRef spelling) {
        return llvm::StringSwitch<std::optional<FloatSubnormalBehavior>>(
                   spelling)
            .Case("preserve", FloatSubnormalBehavior::Preserve)
            .Case("flush_to_zero", FloatSubnormalBehavior::FlushToZero)
            .Default(std::nullopt);
      });
}

llvm::Expected<FloatSignedZeroBehaviorSet>
parseSignedZeroBehaviors(DictionaryAttr params, llvm::StringRef field) {
  return parseEnumSet<FloatSignedZeroBehavior, FloatSignedZeroBehaviorSet>(
      params, field, [](llvm::StringRef spelling) {
        return llvm::StringSwitch<std::optional<FloatSignedZeroBehavior>>(
                   spelling)
            .Case("preserve", FloatSignedZeroBehavior::Preserve)
            .Case("ignore_sign", FloatSignedZeroBehavior::IgnoreSign)
            .Default(std::nullopt);
      });
}

llvm::Expected<FloatBehaviorProfile> parseFloatBehavior(DictionaryAttr params) {
  auto behavior = dyn_cast_or_null<DictionaryAttr>(params.get("behavior"));
  if (!behavior)
    return reject("hw_params field 'behavior' must be a dictionary");
  if (llvm::Error error = checkFields(
          behavior, {"rounding_modes", "nan_behaviors", "subnormal_behaviors",
                     "signed_zero_behaviors", "fastmath"}))
    return std::move(error);

  auto roundingModes = parseEnumSet<arith::RoundingMode, RoundingModeSet>(
      behavior, "rounding_modes", [](llvm::StringRef spelling) {
        return arith::symbolizeRoundingMode(spelling);
      });
  if (!roundingModes)
    return roundingModes.takeError();
  auto nanBehaviors = parseNaNBehaviors(behavior, "nan_behaviors");
  if (!nanBehaviors)
    return nanBehaviors.takeError();
  auto subnormalBehaviors =
      parseSubnormalBehaviors(behavior, "subnormal_behaviors");
  if (!subnormalBehaviors)
    return subnormalBehaviors.takeError();
  auto signedZeroBehaviors =
      parseSignedZeroBehaviors(behavior, "signed_zero_behaviors");
  if (!signedZeroBehaviors)
    return signedZeroBehaviors.takeError();
  auto fastmath = requireString(behavior, "fastmath");
  if (!fastmath)
    return fastmath.takeError();
  std::optional<arith::FastMathFlags> flags =
      arith::symbolizeFastMathFlags(fastmath->getValue());
  if (!flags)
    return reject("hw_params field 'behavior.fastmath' is invalid");

  return FloatBehaviorProfile{*roundingModes, *nanBehaviors,
                              *subnormalBehaviors, *signedZeroBehaviors,
                              *flags};
}

llvm::Expected<IntegerWidthRelation>
parseIntegerWidthRelation(DictionaryAttr params, llvm::StringRef field) {
  auto pairs = requireArray(params, field);
  if (!pairs)
    return pairs.takeError();
  IntegerWidthRelation relation;
  for (Attribute value : *pairs) {
    auto pair = dyn_cast<ArrayAttr>(value);
    if (!pair || pair.size() != 2)
      return reject("hw_params field '" + field +
                    "' must contain two-element width pairs");
    auto source = parseIntegerWidth(pair[0], field);
    if (!source)
      return source.takeError();
    auto destination = parseIntegerWidth(pair[1], field);
    if (!destination)
      return destination.takeError();
    if (relation.contains(*source, *destination))
      return reject("hw_params field '" + field +
                    "' contains a duplicate width pair");
    relation.insert(*source, *destination);
  }
  if (relation.empty())
    return reject("hw_params field '" + field + "' must not be empty");
  return relation;
}

llvm::Expected<FloatFormatRelation>
parseFloatFormatRelation(DictionaryAttr params, llvm::StringRef field) {
  auto pairs = requireArray(params, field);
  if (!pairs)
    return pairs.takeError();
  FloatFormatRelation relation;
  for (Attribute value : *pairs) {
    auto pair = dyn_cast<ArrayAttr>(value);
    if (!pair || pair.size() != 2)
      return reject("hw_params field '" + field +
                    "' must contain two-element format pairs");
    auto source = parseFloatFormat(pair[0], field);
    if (!source)
      return source.takeError();
    auto destination = parseFloatFormat(pair[1], field);
    if (!destination)
      return destination.takeError();
    if (relation.contains(*source, *destination))
      return reject("hw_params field '" + field +
                    "' contains a duplicate format pair");
    relation.insert(*source, *destination);
  }
  if (relation.empty())
    return reject("hw_params field '" + field + "' must not be empty");
  return relation;
}

llvm::Expected<IntegerFloatFormatRelation>
parseIntegerFloatRelation(DictionaryAttr params, llvm::StringRef field) {
  auto pairs = requireArray(params, field);
  if (!pairs)
    return pairs.takeError();
  IntegerFloatFormatRelation relation;
  for (Attribute value : *pairs) {
    auto pair = dyn_cast<ArrayAttr>(value);
    if (!pair || pair.size() != 2)
      return reject("hw_params field '" + field +
                    "' must contain integer/float pairs");
    auto integer = parseIntegerWidth(pair[0], field);
    if (!integer)
      return integer.takeError();
    auto format = parseFloatFormat(pair[1], field);
    if (!format)
      return format.takeError();
    if (relation.contains(*integer, *format))
      return reject("hw_params field '" + field +
                    "' contains a duplicate integer/float pair");
    relation.insert(*integer, *format);
  }
  if (relation.empty())
    return reject("hw_params field '" + field + "' must not be empty");
  return relation;
}

llvm::Expected<ResolvedIndexWidth>
parseResolvedIndexWidth(Attribute attr, llvm::StringRef field) {
  auto integer = dyn_cast<IntegerAttr>(attr);
  if (!integer || integer.getValue().isNegative() ||
      integer.getValue().getActiveBits() > 32)
    return reject("hw_params field '" + field + "' must contain 32 or 64");
  std::optional<ResolvedIndexWidth> width = symbolizeResolvedIndexWidth(
      static_cast<unsigned>(integer.getValue().getZExtValue()));
  if (!width)
    return reject("hw_params field '" + field + "' must contain 32 or 64");
  return *width;
}

llvm::Expected<ResolvedIndexWidthSet>
parseResolvedIndexWidths(DictionaryAttr params, llvm::StringRef field) {
  auto values = requireArray(params, field);
  if (!values)
    return values.takeError();
  ResolvedIndexWidthSet widths;
  for (Attribute value : *values) {
    auto width = parseResolvedIndexWidth(value, field);
    if (!width)
      return width.takeError();
    if (widths.contains(*width))
      return reject("hw_params field '" + field +
                    "' contains a duplicate index width");
    widths.insert(*width);
  }
  return widths;
}

llvm::Expected<FamilyCapabilityParams>
parseParams(CapabilityParamsSchemaId schema, DictionaryAttr params) {
  using Schema = CapabilityParamsSchemaId;
  switch (schema) {
  case Schema::ScalarIntegerParams: {
    if (llvm::Error error = checkFields(params, {"integer_widths"}))
      return std::move(error);
    auto widths = parseIntegerWidths(params, "integer_widths");
    if (!widths)
      return widths.takeError();
    return FamilyCapabilityParams(ScalarIntegerParams{*widths});
  }
  case Schema::ScalarIntegerCompareMinMaxParams: {
    if (llvm::Error error =
            checkFields(params, {"integer_widths", "predicates"}))
      return std::move(error);
    auto widths = parseIntegerWidths(params, "integer_widths");
    if (!widths)
      return widths.takeError();
    auto predicates = parseEnumSet<arith::CmpIPredicate, IntegerPredicateSet>(
        params, "predicates", [](llvm::StringRef spelling) {
          return arith::symbolizeCmpIPredicate(spelling);
        });
    if (!predicates)
      return predicates.takeError();
    return FamilyCapabilityParams(
        ScalarIntegerCompareMinMaxParams{*widths, *predicates});
  }
  case Schema::ScalarValueSelectParams: {
    if (llvm::Error error =
            checkFields(params, {"integer_widths", "float_formats"}))
      return std::move(error);
    auto integerWidths =
        parseIntegerWidths(params, "integer_widths", /*allowEmpty=*/true);
    if (!integerWidths)
      return integerWidths.takeError();
    auto floatFormats = parseFloatFormats(params, "float_formats");
    if (!floatFormats)
      return floatFormats.takeError();
    if (integerWidths->empty() && floatFormats->empty())
      return reject("hw_params scalar value domain must not be empty");
    return FamilyCapabilityParams(
        ScalarValueSelectParams{*integerWidths, *floatFormats});
  }
  case Schema::ScalarIntegerCastParams: {
    if (llvm::Error error =
            checkFields(params, {"width_pairs", "resolved_index_widths"}))
      return std::move(error);
    auto pairs = parseIntegerWidthRelation(params, "width_pairs");
    if (!pairs)
      return pairs.takeError();
    auto indexWidths =
        parseResolvedIndexWidths(params, "resolved_index_widths");
    if (!indexWidths)
      return indexWidths.takeError();
    return FamilyCapabilityParams(
        ScalarIntegerCastParams{IntegerCastRelation{*pairs, *indexWidths}});
  }
  case Schema::ScalarBitReinterpretParams: {
    if (llvm::Error error =
            checkFields(params, {"integer_widths", "float_formats"}))
      return std::move(error);
    auto integerWidths =
        parseIntegerWidths(params, "integer_widths", /*allowEmpty=*/true);
    if (!integerWidths)
      return integerWidths.takeError();
    auto floatFormats = parseFloatFormats(params, "float_formats");
    if (!floatFormats)
      return floatFormats.takeError();
    if (integerWidths->empty() && floatFormats->empty())
      return reject("hw_params bit reinterpretation domain must not be empty");
    return FamilyCapabilityParams(
        ScalarBitReinterpretParams{*integerWidths, *floatFormats});
  }
  case Schema::ScalarFloatParams: {
    if (llvm::Error error = checkFields(params, {"float_formats", "behavior"}))
      return std::move(error);
    auto formats = parseFloatFormats(params, "float_formats");
    if (!formats)
      return formats.takeError();
    if (formats->empty())
      return reject("hw_params field 'float_formats' must not be empty");
    auto behavior = parseFloatBehavior(params);
    if (!behavior)
      return behavior.takeError();
    return FamilyCapabilityParams(ScalarFloatParams{*formats, *behavior});
  }
  case Schema::ScalarFloatCompareMinMaxParams: {
    if (llvm::Error error =
            checkFields(params, {"float_formats", "behavior", "predicates"}))
      return std::move(error);
    auto formats = parseFloatFormats(params, "float_formats");
    if (!formats)
      return formats.takeError();
    if (formats->empty())
      return reject("hw_params field 'float_formats' must not be empty");
    auto behavior = parseFloatBehavior(params);
    if (!behavior)
      return behavior.takeError();
    auto predicates = parseEnumSet<arith::CmpFPredicate, FloatPredicateSet>(
        params, "predicates", [](llvm::StringRef spelling) {
          return arith::symbolizeCmpFPredicate(spelling);
        });
    if (!predicates)
      return predicates.takeError();
    return FamilyCapabilityParams(
        ScalarFloatCompareMinMaxParams{*formats, *behavior, *predicates});
  }
  case Schema::ScalarFloatWidthCastParams: {
    if (llvm::Error error = checkFields(params, {"format_pairs", "behavior"}))
      return std::move(error);
    auto pairs = parseFloatFormatRelation(params, "format_pairs");
    if (!pairs)
      return pairs.takeError();
    auto behavior = parseFloatBehavior(params);
    if (!behavior)
      return behavior.takeError();
    return FamilyCapabilityParams(
        ScalarFloatWidthCastParams{*pairs, *behavior});
  }
  case Schema::ScalarIntegerFloatConversionParams: {
    if (llvm::Error error = checkFields(params, {"format_pairs", "behavior"}))
      return std::move(error);
    auto pairs = parseIntegerFloatRelation(params, "format_pairs");
    if (!pairs)
      return pairs.takeError();
    auto behavior = parseFloatBehavior(params);
    if (!behavior)
      return behavior.takeError();
    return FamilyCapabilityParams(
        ScalarIntegerFloatConversionParams{*pairs, *behavior});
  }
  case Schema::LoopStreamParams: {
    if (llvm::Error error =
            checkFields(params, {"integer_widths", "step_kind", "predicates"}))
      return std::move(error);
    auto widths = parseIntegerWidths(params, "integer_widths");
    if (!widths)
      return widths.takeError();
    auto stepKind = requireString(params, "step_kind");
    if (!stepKind)
      return stepKind.takeError();
    std::optional<dataflow::StreamStepKind> parsedStep =
        dataflow::symbolizeStreamStepKind(stepKind->getValue());
    if (!parsedStep)
      return reject("hw_params field 'step_kind' is invalid");
    auto predicates = parseEnumSet<arith::CmpIPredicate, IntegerPredicateSet>(
        params, "predicates", [](llvm::StringRef spelling) {
          return arith::symbolizeCmpIPredicate(spelling);
        });
    if (!predicates)
      return predicates.takeError();
    return FamilyCapabilityParams(
        LoopStreamParams{*widths, *parsedStep, *predicates});
  }
  case Schema::TokenPlaneParams:
    if (llvm::Error error = checkFields(params, {}))
      return std::move(error);
    return FamilyCapabilityParams(TokenPlaneParams{});
  case Schema::FixedVectorIntegerParams: {
    if (llvm::Error error =
            checkFields(params, {"element_widths", "max_payload_bits"}))
      return std::move(error);
    auto widths = parseIntegerWidths(params, "element_widths");
    if (!widths)
      return widths.takeError();
    auto capacity = requirePositiveU32(params, "max_payload_bits");
    if (!capacity)
      return capacity.takeError();
    return FamilyCapabilityParams(FixedVectorIntegerParams{*widths, *capacity});
  }
  case Schema::FixedVectorIntegerCompareMinMaxParams: {
    if (llvm::Error error = checkFields(
            params, {"element_widths", "predicates", "max_payload_bits"}))
      return std::move(error);
    auto widths = parseIntegerWidths(params, "element_widths");
    if (!widths)
      return widths.takeError();
    auto predicates = parseEnumSet<arith::CmpIPredicate, IntegerPredicateSet>(
        params, "predicates", [](llvm::StringRef spelling) {
          return arith::symbolizeCmpIPredicate(spelling);
        });
    if (!predicates)
      return predicates.takeError();
    auto capacity = requirePositiveU32(params, "max_payload_bits");
    if (!capacity)
      return capacity.takeError();
    return FamilyCapabilityParams(
        FixedVectorIntegerCompareMinMaxParams{*widths, *predicates, *capacity});
  }
  case Schema::FixedVectorValueSelectParams: {
    if (llvm::Error error =
            checkFields(params, {"integer_element_widths",
                                 "float_element_formats", "max_payload_bits"}))
      return std::move(error);
    auto widths = parseIntegerWidths(params, "integer_element_widths",
                                     /*allowEmpty=*/true);
    if (!widths)
      return widths.takeError();
    auto formats = parseFloatFormats(params, "float_element_formats");
    if (!formats)
      return formats.takeError();
    if (widths->empty() && formats->empty())
      return reject("hw_params fixed-vector value domain must not be empty");
    auto capacity = requirePositiveU32(params, "max_payload_bits");
    if (!capacity)
      return capacity.takeError();
    return FamilyCapabilityParams(
        FixedVectorValueSelectParams{*widths, *formats, *capacity});
  }
  case Schema::FixedVectorFloatParams: {
    if (llvm::Error error = checkFields(
            params, {"element_formats", "behavior", "max_payload_bits"}))
      return std::move(error);
    auto formats = parseFloatFormats(params, "element_formats");
    if (!formats)
      return formats.takeError();
    if (formats->empty())
      return reject("hw_params field 'element_formats' must not be empty");
    auto behavior = parseFloatBehavior(params);
    if (!behavior)
      return behavior.takeError();
    auto capacity = requirePositiveU32(params, "max_payload_bits");
    if (!capacity)
      return capacity.takeError();
    return FamilyCapabilityParams(
        FixedVectorFloatParams{*formats, *behavior, *capacity});
  }
  case Schema::FixedVectorFloatCompareMinMaxParams: {
    if (llvm::Error error =
            checkFields(params, {"element_formats", "behavior", "predicates",
                                 "max_payload_bits"}))
      return std::move(error);
    auto formats = parseFloatFormats(params, "element_formats");
    if (!formats)
      return formats.takeError();
    if (formats->empty())
      return reject("hw_params field 'element_formats' must not be empty");
    auto behavior = parseFloatBehavior(params);
    if (!behavior)
      return behavior.takeError();
    auto predicates = parseEnumSet<arith::CmpFPredicate, FloatPredicateSet>(
        params, "predicates", [](llvm::StringRef spelling) {
          return arith::symbolizeCmpFPredicate(spelling);
        });
    if (!predicates)
      return predicates.takeError();
    auto capacity = requirePositiveU32(params, "max_payload_bits");
    if (!capacity)
      return capacity.takeError();
    return FamilyCapabilityParams(FixedVectorFloatCompareMinMaxParams{
        *formats, *behavior, *predicates, *capacity});
  }
  case Schema::FixedVectorAdapterParams: {
    if (llvm::Error error =
            checkFields(params, {"integer_element_widths",
                                 "float_element_formats", "max_payload_bits"}))
      return std::move(error);
    auto widths = parseIntegerWidths(params, "integer_element_widths",
                                     /*allowEmpty=*/true);
    if (!widths)
      return widths.takeError();
    auto formats = parseFloatFormats(params, "float_element_formats");
    if (!formats)
      return formats.takeError();
    if (widths->empty() && formats->empty())
      return reject("hw_params fixed-vector adapter domain must not be empty");
    auto capacity = requirePositiveU32(params, "max_payload_bits");
    if (!capacity)
      return capacity.takeError();
    return FamilyCapabilityParams(
        FixedVectorAdapterParams{*widths, *formats, *capacity});
  }
  case Schema::PayloadCapacityParams: {
    if (llvm::Error error = checkFields(params, {"max_payload_bits"}))
      return std::move(error);
    auto capacity = requirePositiveU32(params, "max_payload_bits");
    if (!capacity)
      return capacity.takeError();
    return FamilyCapabilityParams(PayloadCapacityParams{*capacity});
  }
  case Schema::RoutedTokenParams: {
    if (llvm::Error error =
            checkFields(params, {"max_payload_bits", "max_fan"}))
      return std::move(error);
    auto capacity = requirePositiveU32(params, "max_payload_bits");
    if (!capacity)
      return capacity.takeError();
    auto fan = requirePositiveU32(params, "max_fan");
    if (!fan)
      return fan.takeError();
    if (*fan < 2)
      return reject("hw_params field 'max_fan' must be at least two");
    return FamilyCapabilityParams(RoutedTokenParams{*capacity, *fan});
  }
  }
  llvm_unreachable("unregistered capability parameter schema");
}

IntegerAttr integerAttr(OpBuilder &builder, unsigned value) {
  return builder.getI32IntegerAttr(value);
}

ArrayAttr integerWidthsAttr(OpBuilder &builder, IntegerWidthSet widths) {
  constexpr std::array<std::pair<IntegerWidth, unsigned>, 5> values = {{
      {IntegerWidth::I1, 1},
      {IntegerWidth::I8, 8},
      {IntegerWidth::I16, 16},
      {IntegerWidth::I32, 32},
      {IntegerWidth::I64, 64},
  }};
  llvm::SmallVector<Attribute, 5> encoded;
  for (auto [width, bits] : values)
    if (widths.contains(width))
      encoded.push_back(integerAttr(builder, bits));
  return builder.getArrayAttr(encoded);
}

llvm::StringRef floatFormatSpelling(FloatFormat format) {
  switch (format) {
  case FloatFormat::F16:
    return "f16";
  case FloatFormat::BF16:
    return "bf16";
  case FloatFormat::F32:
    return "f32";
  case FloatFormat::F64:
    return "f64";
  }
  llvm_unreachable("unknown floating format");
}

ArrayAttr floatFormatsAttr(OpBuilder &builder, FloatFormatSet formats) {
  constexpr std::array<FloatFormat, 4> values = {
      FloatFormat::F16, FloatFormat::BF16, FloatFormat::F32, FloatFormat::F64};
  llvm::SmallVector<Attribute, 4> encoded;
  for (FloatFormat format : values)
    if (formats.contains(format))
      encoded.push_back(builder.getStringAttr(floatFormatSpelling(format)));
  return builder.getArrayAttr(encoded);
}

template <typename Enum, typename Set, std::size_t Size, typename Stringifier>
ArrayAttr enumSetAttr(OpBuilder &builder, Set values,
                      const std::array<Enum, Size> &domain,
                      Stringifier stringify) {
  llvm::SmallVector<Attribute, Size> encoded;
  for (Enum value : domain)
    if (values.contains(value))
      encoded.push_back(builder.getStringAttr(stringify(value)));
  return builder.getArrayAttr(encoded);
}

DictionaryAttr floatBehaviorAttr(OpBuilder &builder,
                                 const FloatBehaviorProfile &behavior) {
  constexpr std::array<arith::RoundingMode, 5> roundingModes = {
      arith::RoundingMode::to_nearest_even, arith::RoundingMode::downward,
      arith::RoundingMode::upward, arith::RoundingMode::toward_zero,
      arith::RoundingMode::to_nearest_away};
  constexpr std::array<FloatNaNBehavior, 2> nanBehaviors = {
      FloatNaNBehavior::IEEE, FloatNaNBehavior::NumberPreferred};
  constexpr std::array<FloatSubnormalBehavior, 2> subnormalBehaviors = {
      FloatSubnormalBehavior::Preserve, FloatSubnormalBehavior::FlushToZero};
  constexpr std::array<FloatSignedZeroBehavior, 2> signedZeroBehaviors = {
      FloatSignedZeroBehavior::Preserve, FloatSignedZeroBehavior::IgnoreSign};
  auto nanSpelling = [](FloatNaNBehavior value) -> llvm::StringRef {
    return value == FloatNaNBehavior::IEEE ? "ieee" : "number_preferred";
  };
  auto subnormalSpelling = [](FloatSubnormalBehavior value) -> llvm::StringRef {
    return value == FloatSubnormalBehavior::Preserve ? "preserve"
                                                     : "flush_to_zero";
  };
  auto signedZeroSpelling =
      [](FloatSignedZeroBehavior value) -> llvm::StringRef {
    return value == FloatSignedZeroBehavior::Preserve ? "preserve"
                                                      : "ignore_sign";
  };
  return builder.getDictionaryAttr({
      builder.getNamedAttr("rounding_modes",
                           enumSetAttr(builder, behavior.roundingModes,
                                       roundingModes,
                                       arith::stringifyRoundingMode)),
      builder.getNamedAttr("nan_behaviors",
                           enumSetAttr(builder, behavior.nanBehaviors,
                                       nanBehaviors, nanSpelling)),
      builder.getNamedAttr("subnormal_behaviors",
                           enumSetAttr(builder, behavior.subnormalBehaviors,
                                       subnormalBehaviors, subnormalSpelling)),
      builder.getNamedAttr("signed_zero_behaviors",
                           enumSetAttr(builder, behavior.signedZeroBehaviors,
                                       signedZeroBehaviors,
                                       signedZeroSpelling)),
      builder.getNamedAttr("fastmath",
                           builder.getStringAttr(arith::stringifyFastMathFlags(
                               behavior.requiredFastMath))),
  });
}

ArrayAttr integerPairsAttr(OpBuilder &builder,
                           const IntegerWidthRelation &relation) {
  constexpr std::array<std::pair<IntegerWidth, unsigned>, 5> values = {{
      {IntegerWidth::I1, 1},
      {IntegerWidth::I8, 8},
      {IntegerWidth::I16, 16},
      {IntegerWidth::I32, 32},
      {IntegerWidth::I64, 64},
  }};
  llvm::SmallVector<Attribute, 25> encoded;
  for (auto [source, sourceBits] : values)
    for (auto [destination, destinationBits] : values)
      if (relation.contains(source, destination))
        encoded.push_back(
            builder.getArrayAttr({integerAttr(builder, sourceBits),
                                  integerAttr(builder, destinationBits)}));
  return builder.getArrayAttr(encoded);
}

ArrayAttr floatPairsAttr(OpBuilder &builder,
                         const FloatFormatRelation &relation) {
  constexpr std::array<FloatFormat, 4> values = {
      FloatFormat::F16, FloatFormat::BF16, FloatFormat::F32, FloatFormat::F64};
  llvm::SmallVector<Attribute, 16> encoded;
  for (FloatFormat source : values)
    for (FloatFormat destination : values)
      if (relation.contains(source, destination))
        encoded.push_back(builder.getArrayAttr(
            {builder.getStringAttr(floatFormatSpelling(source)),
             builder.getStringAttr(floatFormatSpelling(destination))}));
  return builder.getArrayAttr(encoded);
}

ArrayAttr integerFloatPairsAttr(OpBuilder &builder,
                                const IntegerFloatFormatRelation &relation) {
  constexpr std::array<std::pair<IntegerWidth, unsigned>, 5> integers = {{
      {IntegerWidth::I1, 1},
      {IntegerWidth::I8, 8},
      {IntegerWidth::I16, 16},
      {IntegerWidth::I32, 32},
      {IntegerWidth::I64, 64},
  }};
  constexpr std::array<FloatFormat, 4> formats = {
      FloatFormat::F16, FloatFormat::BF16, FloatFormat::F32, FloatFormat::F64};
  llvm::SmallVector<Attribute, 20> encoded;
  for (auto [integer, bits] : integers)
    for (FloatFormat format : formats)
      if (relation.contains(integer, format))
        encoded.push_back(builder.getArrayAttr(
            {integerAttr(builder, bits),
             builder.getStringAttr(floatFormatSpelling(format))}));
  return builder.getArrayAttr(encoded);
}

} // namespace

llvm::Expected<FamilyCapabilityParams>
fabric::parseFamilyCapabilityParams(ImplementationFamilyId family,
                                    DictionaryAttr params) {
  if (!params)
    return reject("fabric.op requires typed hw_params");
  return parseParams(implementationFamily(family).capabilityParamsSchema,
                     params);
}

DictionaryAttr
fabric::getFamilyCapabilityParamsAttr(MLIRContext *context,
                                      const FamilyCapabilityParams &params) {
  OpBuilder builder(context);
  return std::visit(
      [&](const auto &typed) -> DictionaryAttr {
        using T = std::decay_t<decltype(typed)>;
        if constexpr (std::is_same_v<T, ScalarIntegerParams>) {
          return builder.getDictionaryAttr({builder.getNamedAttr(
              "integer_widths",
              integerWidthsAttr(builder, typed.integerWidths))});
        } else if constexpr (std::is_same_v<T,
                                            ScalarIntegerCompareMinMaxParams>) {
          constexpr std::array<arith::CmpIPredicate, 10> predicates = {
              arith::CmpIPredicate::eq,  arith::CmpIPredicate::ne,
              arith::CmpIPredicate::slt, arith::CmpIPredicate::sle,
              arith::CmpIPredicate::sgt, arith::CmpIPredicate::sge,
              arith::CmpIPredicate::ult, arith::CmpIPredicate::ule,
              arith::CmpIPredicate::ugt, arith::CmpIPredicate::uge};
          return builder.getDictionaryAttr({
              builder.getNamedAttr(
                  "integer_widths",
                  integerWidthsAttr(builder, typed.operandWidths)),
              builder.getNamedAttr("predicates",
                                   enumSetAttr(builder, typed.predicates,
                                               predicates,
                                               arith::stringifyCmpIPredicate)),
          });
        } else if constexpr (std::is_same_v<T, ScalarValueSelectParams>) {
          return builder.getDictionaryAttr({
              builder.getNamedAttr(
                  "integer_widths",
                  integerWidthsAttr(builder, typed.integerWidths)),
              builder.getNamedAttr(
                  "float_formats",
                  floatFormatsAttr(builder, typed.floatFormats)),
          });
        } else if constexpr (std::is_same_v<T, ScalarIntegerCastParams>) {
          llvm::SmallVector<NamedAttribute, 2> fields;
          fields.push_back(builder.getNamedAttr(
              "width_pairs",
              integerPairsAttr(builder, typed.relation.widthPairs)));
          llvm::SmallVector<Attribute, 2> indexWidths;
          for (ResolvedIndexWidth width : resolvedIndexWidthDomain) {
            if (typed.relation.resolvedIndexWidths.contains(width))
              indexWidths.push_back(
                  integerAttr(builder, getResolvedIndexBitWidth(width)));
          }
          fields.push_back(builder.getNamedAttr(
              "resolved_index_widths", builder.getArrayAttr(indexWidths)));
          return builder.getDictionaryAttr(fields);
        } else if constexpr (std::is_same_v<T, ScalarBitReinterpretParams>) {
          return builder.getDictionaryAttr({
              builder.getNamedAttr(
                  "integer_widths",
                  integerWidthsAttr(builder, typed.integerWidths)),
              builder.getNamedAttr(
                  "float_formats",
                  floatFormatsAttr(builder, typed.floatFormats)),
          });
        } else if constexpr (std::is_same_v<T, ScalarFloatParams>) {
          return builder.getDictionaryAttr({
              builder.getNamedAttr("float_formats",
                                   floatFormatsAttr(builder, typed.formats)),
              builder.getNamedAttr("behavior",
                                   floatBehaviorAttr(builder, typed.behavior)),
          });
        } else if constexpr (std::is_same_v<T,
                                            ScalarFloatCompareMinMaxParams>) {
          constexpr std::array<arith::CmpFPredicate, 16> predicates = {
              arith::CmpFPredicate::AlwaysFalse,
              arith::CmpFPredicate::OEQ,
              arith::CmpFPredicate::OGT,
              arith::CmpFPredicate::OGE,
              arith::CmpFPredicate::OLT,
              arith::CmpFPredicate::OLE,
              arith::CmpFPredicate::ONE,
              arith::CmpFPredicate::ORD,
              arith::CmpFPredicate::UEQ,
              arith::CmpFPredicate::UGT,
              arith::CmpFPredicate::UGE,
              arith::CmpFPredicate::ULT,
              arith::CmpFPredicate::ULE,
              arith::CmpFPredicate::UNE,
              arith::CmpFPredicate::UNO,
              arith::CmpFPredicate::AlwaysTrue};
          return builder.getDictionaryAttr({
              builder.getNamedAttr("float_formats",
                                   floatFormatsAttr(builder, typed.formats)),
              builder.getNamedAttr("behavior",
                                   floatBehaviorAttr(builder, typed.behavior)),
              builder.getNamedAttr("predicates",
                                   enumSetAttr(builder, typed.predicates,
                                               predicates,
                                               arith::stringifyCmpFPredicate)),
          });
        } else if constexpr (std::is_same_v<T, ScalarFloatWidthCastParams>) {
          return builder.getDictionaryAttr({
              builder.getNamedAttr("format_pairs",
                                   floatPairsAttr(builder, typed.formatPairs)),
              builder.getNamedAttr("behavior",
                                   floatBehaviorAttr(builder, typed.behavior)),
          });
        } else if constexpr (std::is_same_v<
                                 T, ScalarIntegerFloatConversionParams>) {
          return builder.getDictionaryAttr({
              builder.getNamedAttr(
                  "format_pairs",
                  integerFloatPairsAttr(builder, typed.formatPairs)),
              builder.getNamedAttr("behavior",
                                   floatBehaviorAttr(builder, typed.behavior)),
          });
        } else if constexpr (std::is_same_v<T, LoopStreamParams>) {
          constexpr std::array<arith::CmpIPredicate, 10> predicates = {
              arith::CmpIPredicate::eq,  arith::CmpIPredicate::ne,
              arith::CmpIPredicate::slt, arith::CmpIPredicate::sle,
              arith::CmpIPredicate::sgt, arith::CmpIPredicate::sge,
              arith::CmpIPredicate::ult, arith::CmpIPredicate::ule,
              arith::CmpIPredicate::ugt, arith::CmpIPredicate::uge};
          return builder.getDictionaryAttr({
              builder.getNamedAttr(
                  "integer_widths",
                  integerWidthsAttr(builder, typed.integerWidths)),
              builder.getNamedAttr(
                  "step_kind",
                  builder.getStringAttr(
                      dataflow::stringifyStreamStepKind(typed.fixedStepKind))),
              builder.getNamedAttr(
                  "predicates",
                  enumSetAttr(builder, typed.continuationPredicates, predicates,
                              arith::stringifyCmpIPredicate)),
          });
        } else if constexpr (std::is_same_v<T, TokenPlaneParams>) {
          return builder.getDictionaryAttr({});
        } else if constexpr (std::is_same_v<T, FixedVectorIntegerParams>) {
          return builder.getDictionaryAttr({
              builder.getNamedAttr(
                  "element_widths",
                  integerWidthsAttr(builder, typed.elementWidths)),
              builder.getNamedAttr("max_payload_bits",
                                   integerAttr(builder, typed.maxPayloadBits)),
          });
        } else if constexpr (std::is_same_v<
                                 T, FixedVectorIntegerCompareMinMaxParams>) {
          constexpr std::array<arith::CmpIPredicate, 10> predicates = {
              arith::CmpIPredicate::eq,  arith::CmpIPredicate::ne,
              arith::CmpIPredicate::slt, arith::CmpIPredicate::sle,
              arith::CmpIPredicate::sgt, arith::CmpIPredicate::sge,
              arith::CmpIPredicate::ult, arith::CmpIPredicate::ule,
              arith::CmpIPredicate::ugt, arith::CmpIPredicate::uge};
          return builder.getDictionaryAttr({
              builder.getNamedAttr(
                  "element_widths",
                  integerWidthsAttr(builder, typed.elementWidths)),
              builder.getNamedAttr("predicates",
                                   enumSetAttr(builder, typed.predicates,
                                               predicates,
                                               arith::stringifyCmpIPredicate)),
              builder.getNamedAttr("max_payload_bits",
                                   integerAttr(builder, typed.maxPayloadBits)),
          });
        } else if constexpr (std::is_same_v<T, FixedVectorValueSelectParams>) {
          return builder.getDictionaryAttr({
              builder.getNamedAttr(
                  "integer_element_widths",
                  integerWidthsAttr(builder, typed.integerElementWidths)),
              builder.getNamedAttr(
                  "float_element_formats",
                  floatFormatsAttr(builder, typed.floatElementFormats)),
              builder.getNamedAttr("max_payload_bits",
                                   integerAttr(builder, typed.maxPayloadBits)),
          });
        } else if constexpr (std::is_same_v<T, FixedVectorFloatParams>) {
          return builder.getDictionaryAttr({
              builder.getNamedAttr(
                  "element_formats",
                  floatFormatsAttr(builder, typed.elementFormats)),
              builder.getNamedAttr("behavior",
                                   floatBehaviorAttr(builder, typed.behavior)),
              builder.getNamedAttr("max_payload_bits",
                                   integerAttr(builder, typed.maxPayloadBits)),
          });
        } else if constexpr (std::is_same_v<
                                 T, FixedVectorFloatCompareMinMaxParams>) {
          constexpr std::array<arith::CmpFPredicate, 16> predicates = {
              arith::CmpFPredicate::AlwaysFalse,
              arith::CmpFPredicate::OEQ,
              arith::CmpFPredicate::OGT,
              arith::CmpFPredicate::OGE,
              arith::CmpFPredicate::OLT,
              arith::CmpFPredicate::OLE,
              arith::CmpFPredicate::ONE,
              arith::CmpFPredicate::ORD,
              arith::CmpFPredicate::UEQ,
              arith::CmpFPredicate::UGT,
              arith::CmpFPredicate::UGE,
              arith::CmpFPredicate::ULT,
              arith::CmpFPredicate::ULE,
              arith::CmpFPredicate::UNE,
              arith::CmpFPredicate::UNO,
              arith::CmpFPredicate::AlwaysTrue};
          return builder.getDictionaryAttr({
              builder.getNamedAttr(
                  "element_formats",
                  floatFormatsAttr(builder, typed.elementFormats)),
              builder.getNamedAttr("behavior",
                                   floatBehaviorAttr(builder, typed.behavior)),
              builder.getNamedAttr("predicates",
                                   enumSetAttr(builder, typed.predicates,
                                               predicates,
                                               arith::stringifyCmpFPredicate)),
              builder.getNamedAttr("max_payload_bits",
                                   integerAttr(builder, typed.maxPayloadBits)),
          });
        } else if constexpr (std::is_same_v<T, FixedVectorAdapterParams>) {
          return builder.getDictionaryAttr({
              builder.getNamedAttr(
                  "integer_element_widths",
                  integerWidthsAttr(builder, typed.integerElementWidths)),
              builder.getNamedAttr(
                  "float_element_formats",
                  floatFormatsAttr(builder, typed.floatElementFormats)),
              builder.getNamedAttr("max_payload_bits",
                                   integerAttr(builder, typed.maxPayloadBits)),
          });
        } else if constexpr (std::is_same_v<T, PayloadCapacityParams>) {
          return builder.getDictionaryAttr({builder.getNamedAttr(
              "max_payload_bits", integerAttr(builder, typed.maxPayloadBits))});
        } else {
          static_assert(std::is_same_v<T, RoutedTokenParams>);
          return builder.getDictionaryAttr({
              builder.getNamedAttr("max_payload_bits",
                                   integerAttr(builder, typed.maxPayloadBits)),
              builder.getNamedAttr("max_fan",
                                   integerAttr(builder, typed.maxFan)),
          });
        }
      },
      params);
}

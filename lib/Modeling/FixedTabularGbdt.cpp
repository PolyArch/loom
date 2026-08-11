#include "FixedTabularGbdt.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace loom::evaluation::models::detail {
namespace {

constexpr std::int64_t kMagnitudeLimit = std::int64_t{1} << 40;
constexpr std::uint32_t kMaximumFieldCount = 1024;
constexpr std::uint64_t kMaximumRows = std::uint64_t{1} << 20;
constexpr std::uint32_t kWireVersion = 1;

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "fixed_tabular_gbdt_invalid: " + message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void appendI64(std::vector<std::uint8_t> &bytes, std::int64_t value) {
  appendU64(bytes, static_cast<std::uint64_t>(value));
}

void appendFramed(std::vector<std::uint8_t> &bytes,
                  llvm::ArrayRef<std::uint8_t> value) {
  appendU64(bytes, value.size());
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void appendDecimal(std::vector<std::uint8_t> &bytes, DecimalValue value) {
  appendI64(bytes, value.coefficient());
  appendI64(bytes, value.base10Exponent());
}

class Decoder final {
public:
  explicit Decoder(llvm::ArrayRef<std::uint8_t> bytes) : bytes_(bytes) {}

  llvm::Expected<std::uint32_t> u32(const llvm::Twine &field) {
    if (remaining() < 4)
      return invalid(field + " is truncated");
    std::uint32_t value = 0;
    for (unsigned index = 0; index != 4; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::uint64_t> u64(const llvm::Twine &field) {
    if (remaining() < 8)
      return invalid(field + " is truncated");
    std::uint64_t value = 0;
    for (unsigned index = 0; index != 8; ++index)
      value = (value << 8) | bytes_[offset_++];
    return value;
  }

  llvm::Expected<std::int64_t> i64(const llvm::Twine &field) {
    auto value = u64(field);
    if (!value)
      return value.takeError();
    return static_cast<std::int64_t>(*value);
  }

  llvm::Expected<std::vector<std::uint8_t>> framed(const llvm::Twine &field) {
    auto size = u64(field);
    if (!size)
      return size.takeError();
    if (*size > remaining())
      return invalid(field + " is truncated");
    std::vector<std::uint8_t> value(
        bytes_.begin() + static_cast<std::ptrdiff_t>(offset_),
        bytes_.begin() + static_cast<std::ptrdiff_t>(offset_ + *size));
    offset_ += static_cast<std::size_t>(*size);
    return value;
  }

  llvm::Expected<DecimalValue> decimal(const llvm::Twine &field) {
    auto coefficient = i64(field + " coefficient");
    if (!coefficient)
      return coefficient.takeError();
    auto exponent = i64(field + " exponent");
    if (!exponent)
      return exponent.takeError();
    return DecimalValue::get(*coefficient, *exponent);
  }

  std::size_t remaining() const { return bytes_.size() - offset_; }

private:
  llvm::ArrayRef<std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

llvm::Expected<std::int64_t> divideRoundTiesToEven(__int128 numerator,
                                                   std::uint64_t denominator) {
  if (denominator == 0)
    return invalid("decimal scaling has a zero denominator");
  const bool negative = numerator < 0;
  const unsigned __int128 magnitude =
      negative ? static_cast<unsigned __int128>(-numerator)
               : static_cast<unsigned __int128>(numerator);
  unsigned __int128 quotient = magnitude / denominator;
  const unsigned __int128 remainder = magnitude % denominator;
  if (remainder * 2 > denominator ||
      (remainder * 2 == denominator && (quotient & 1) != 0))
    ++quotient;
  const __int128 result = negative ? -static_cast<__int128>(quotient)
                                   : static_cast<__int128>(quotient);
  if (result < std::numeric_limits<std::int64_t>::min() ||
      result > std::numeric_limits<std::int64_t>::max())
    return invalid("scaled decimal exceeds int64");
  return static_cast<std::int64_t>(result);
}

llvm::Expected<std::int64_t> scaleDecimal(DecimalValue value,
                                          std::int64_t targetExponent) {
  if (value.coefficient() == 0)
    return std::int64_t{0};
  const __int128 shift = static_cast<__int128>(value.base10Exponent()) -
                         static_cast<__int128>(targetExponent);
  if (shift >= 0) {
    __int128 result = value.coefficient();
    for (__int128 index = 0; index != shift; ++index) {
      result *= 10;
      if (result < -static_cast<__int128>(kMagnitudeLimit) ||
          result > static_cast<__int128>(kMagnitudeLimit))
        return invalid("scaled decimal exceeds the admitted magnitude");
    }
    return static_cast<std::int64_t>(result);
  }
  const __int128 divisorDigits = -shift;
  if (divisorDigits > 19)
    return std::int64_t{0};
  std::uint64_t divisor = 1;
  for (__int128 index = 0; index != divisorDigits; ++index)
    divisor *= 10;
  return divideRoundTiesToEven(value.coefficient(), divisor);
}

llvm::Expected<std::int64_t> selectScale(llvm::ArrayRef<DecimalValue> values) {
  if (values.empty())
    return invalid("decimal scale requires observations");
  std::int64_t minimumExponent = values.front().base10Exponent();
  std::int64_t maximumExponent = minimumExponent;
  for (DecimalValue value : values) {
    minimumExponent = std::min(minimumExponent, value.base10Exponent());
    maximumExponent = std::max(maximumExponent, value.base10Exponent());
  }
  std::int64_t selected = minimumExponent;
  if (maximumExponent > std::numeric_limits<std::int64_t>::min() + 18)
    selected = std::max(selected, maximumExponent - 18);
  for (unsigned attempt = 0; attempt != 64; ++attempt) {
    bool admitted = true;
    for (DecimalValue value : values) {
      auto scaled = scaleDecimal(value, selected);
      if (!scaled || *scaled < -kMagnitudeLimit || *scaled > kMagnitudeLimit) {
        if (!scaled)
          llvm::consumeError(scaled.takeError());
        admitted = false;
        break;
      }
    }
    if (admitted)
      return selected;
    if (selected == std::numeric_limits<std::int64_t>::max())
      break;
    ++selected;
  }
  return invalid("decimal observations cannot share an admitted integer scale");
}

llvm::Error validateShape(const FixedTabularFeatureView &features,
                          std::size_t integral, std::size_t decimal,
                          std::size_t categorical, std::size_t presence) {
  if (features.integral.size() != integral ||
      features.decimal.size() != decimal ||
      features.categorical.size() != categorical ||
      features.presence.size() != presence)
    return invalid("feature view has a different fixed field shape");
  for (std::int64_t value : features.integral)
    if (value < -kMagnitudeLimit || value > kMagnitudeLimit)
      return invalid("integral feature exceeds the admitted magnitude");
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::int64_t>>
flatten(const FixedTabularFeatureView &features,
        const FixedTabularGbdtParameters &parameters, bool enforceSupport) {
  if (llvm::Error error =
          validateShape(features, parameters.integralMinimum.size(),
                        parameters.decimalSupport.size(),
                        parameters.categoricalSupport.size(),
                        parameters.presenceSupport.size()))
    return std::move(error);
  std::vector<std::int64_t> flattened;
  flattened.reserve(features.integral.size() + features.decimal.size() +
                    features.categorical.size() + features.presence.size());
  for (std::size_t index = 0; index != features.integral.size(); ++index) {
    const std::int64_t value = features.integral[index];
    if (enforceSupport && (value < parameters.integralMinimum[index] ||
                           value > parameters.integralMaximum[index]))
      return std::vector<std::int64_t>{};
    flattened.push_back(value);
  }
  for (std::size_t index = 0; index != features.decimal.size(); ++index) {
    const DecimalValue value = features.decimal[index];
    const DecimalSupportInterval &support = parameters.decimalSupport[index];
    if (enforceSupport && (compareDecimalValue(value, support.minimum) < 0 ||
                           compareDecimalValue(value, support.maximum) > 0))
      return std::vector<std::int64_t>{};
    auto scaled = scaleDecimal(value, support.scaleBase10Exponent);
    if (!scaled)
      return scaled.takeError();
    flattened.push_back(*scaled);
  }
  for (std::size_t index = 0; index != features.categorical.size(); ++index) {
    const auto &support = parameters.categoricalSupport[index];
    auto found = llvm::lower_bound(support, features.categorical[index]);
    if (found == support.end() || *found != features.categorical[index]) {
      if (enforceSupport)
        return std::vector<std::int64_t>{};
      return invalid("Training categorical support lost a row value");
    }
    flattened.push_back(static_cast<std::int64_t>(found - support.begin()));
  }
  for (std::size_t index = 0; index != features.presence.size(); ++index) {
    const std::uint8_t bit = features.presence[index] ? 2 : 1;
    if (enforceSupport && (parameters.presenceSupport[index] & bit) == 0)
      return std::vector<std::int64_t>{};
    flattened.push_back(features.presence[index] ? 1 : 0);
  }
  return flattened;
}

llvm::Error validateParameters(const FixedTabularGbdtParameters &parameters) {
  if (parameters.groundTruthTargetKey.empty())
    return invalid("ground-truth target key is empty");
  if (parameters.integralMinimum.size() != parameters.integralMaximum.size())
    return invalid("integral support widths differ");
  const std::uint64_t featureCount =
      parameters.integralMinimum.size() + parameters.decimalSupport.size() +
      parameters.categoricalSupport.size() + parameters.presenceSupport.size();
  if (featureCount == 0 || featureCount > kMaximumFieldCount ||
      featureCount != parameters.ensemble.featureCount)
    return invalid("parameter feature count is inconsistent");
  if (parameters.targetBase10Exponents.empty() ||
      parameters.targetBase10Exponents.size() > kMaximumFieldCount ||
      parameters.targetBase10Exponents.size() != parameters.ensemble.headCount)
    return invalid("parameter target count is inconsistent");
  for (std::size_t index = 0; index != parameters.integralMinimum.size();
       ++index)
    if (parameters.integralMinimum[index] < -kMagnitudeLimit ||
        parameters.integralMinimum[index] > kMagnitudeLimit ||
        parameters.integralMaximum[index] < -kMagnitudeLimit ||
        parameters.integralMaximum[index] > kMagnitudeLimit)
      return invalid("integral support exceeds the admitted magnitude");
    else if (parameters.integralMinimum[index] >
             parameters.integralMaximum[index])
      return invalid("integral support interval is inverted");
  for (const DecimalSupportInterval &support : parameters.decimalSupport)
    if (compareDecimalValue(support.minimum, support.maximum) > 0)
      return invalid("decimal support interval is inverted");
  for (const auto &support : parameters.categoricalSupport)
    if (support.empty() || !llvm::is_sorted(support) ||
        std::adjacent_find(support.begin(), support.end()) != support.end())
      return invalid("categorical support is not canonical and nonempty");
  for (std::uint8_t support : parameters.presenceSupport)
    if (support == 0 || support > 3)
      return invalid("presence support is invalid");
  auto encoded = encodeDeterministicGbdt(parameters.ensemble);
  if (!encoded)
    return encoded.takeError();
  return llvm::Error::success();
}

bool hasSameSupport(const FixedTabularGbdtParameters &lhs,
                    const FixedTabularGbdtParameters &rhs) {
  if (lhs.groundTruthTargetKey != rhs.groundTruthTargetKey ||
      lhs.integralMinimum != rhs.integralMinimum ||
      lhs.integralMaximum != rhs.integralMaximum ||
      lhs.categoricalSupport != rhs.categoricalSupport ||
      lhs.presenceSupport != rhs.presenceSupport ||
      lhs.targetBase10Exponents != rhs.targetBase10Exponents ||
      lhs.decimalSupport.size() != rhs.decimalSupport.size())
    return false;
  for (std::size_t index = 0; index != lhs.decimalSupport.size(); ++index) {
    const DecimalSupportInterval &left = lhs.decimalSupport[index];
    const DecimalSupportInterval &right = rhs.decimalSupport[index];
    if (left.minimum != right.minimum || left.maximum != right.maximum ||
        left.scaleBase10Exponent != right.scaleBase10Exponent)
      return false;
  }
  return true;
}

} // namespace

llvm::Expected<FixedTabularGbdtParameters>
trainFixedTabularGbdt(llvm::ArrayRef<FixedTabularTrainingRow> rows,
                      llvm::ArrayRef<std::uint8_t> groundTruthTargetKey,
                      const DeterministicGbdtConfig &config,
                      const FixedTabularGbdtParameters *initial) {
  if (rows.empty() || rows.size() > kMaximumRows)
    return invalid("Training row count is invalid");
  if (groundTruthTargetKey.empty())
    return invalid("ground-truth target key is empty");
  const std::size_t integralCount = rows.front().features.integral.size();
  const std::size_t decimalCount = rows.front().features.decimal.size();
  const std::size_t categoricalCount = rows.front().features.categorical.size();
  const std::size_t presenceCount = rows.front().features.presence.size();
  const std::size_t targetCount = rows.front().targets.size();
  const std::size_t featureCount =
      integralCount + decimalCount + categoricalCount + presenceCount;
  if (featureCount == 0 || featureCount > kMaximumFieldCount ||
      targetCount == 0 || targetCount > kMaximumFieldCount)
    return invalid("Training field count is invalid");

  FixedTabularGbdtParameters parameters;
  parameters.groundTruthTargetKey = groundTruthTargetKey.vec();
  parameters.integralMinimum = rows.front().features.integral;
  parameters.integralMaximum = rows.front().features.integral;
  parameters.decimalSupport.reserve(decimalCount);
  parameters.categoricalSupport.resize(categoricalCount);
  parameters.presenceSupport.assign(presenceCount, 0);
  for (const FixedTabularTrainingRow &row : rows) {
    if (llvm::Error error =
            validateShape(row.features, integralCount, decimalCount,
                          categoricalCount, presenceCount))
      return std::move(error);
    if (row.targets.size() != targetCount)
      return invalid("Training target shape changed between rows");
    for (std::size_t index = 0; index != integralCount; ++index) {
      parameters.integralMinimum[index] = std::min(
          parameters.integralMinimum[index], row.features.integral[index]);
      parameters.integralMaximum[index] = std::max(
          parameters.integralMaximum[index], row.features.integral[index]);
    }
    for (std::size_t index = 0; index != categoricalCount; ++index)
      parameters.categoricalSupport[index].push_back(
          row.features.categorical[index]);
    for (std::size_t index = 0; index != presenceCount; ++index)
      parameters.presenceSupport[index] |= row.features.presence[index] ? 2 : 1;
  }
  for (auto &support : parameters.categoricalSupport) {
    llvm::sort(support);
    support.erase(std::unique(support.begin(), support.end()), support.end());
  }

  for (std::size_t index = 0; index != decimalCount; ++index) {
    std::vector<DecimalValue> values;
    values.reserve(rows.size());
    for (const FixedTabularTrainingRow &row : rows)
      values.push_back(row.features.decimal[index]);
    auto scale = selectScale(values);
    if (!scale)
      return scale.takeError();
    DecimalValue minimum = values.front();
    DecimalValue maximum = values.front();
    for (DecimalValue value : values) {
      if (compareDecimalValue(value, minimum) < 0)
        minimum = value;
      if (compareDecimalValue(value, maximum) > 0)
        maximum = value;
    }
    parameters.decimalSupport.push_back({minimum, maximum, *scale});
  }

  parameters.targetBase10Exponents.reserve(targetCount);
  for (std::size_t index = 0; index != targetCount; ++index) {
    std::vector<DecimalValue> values;
    values.reserve(rows.size());
    for (const FixedTabularTrainingRow &row : rows)
      values.push_back(row.targets[index]);
    auto scale = selectScale(values);
    if (!scale)
      return scale.takeError();
    parameters.targetBase10Exponents.push_back(*scale);
  }

  if (initial) {
    if (llvm::Error error = validateParameters(*initial))
      return std::move(error);
    if (!hasSameSupport(parameters, *initial))
      return invalid(
          "prior parameters do not match the exact Training support");
  }

  std::vector<DeterministicGbdtTrainingRow> fixedRows;
  fixedRows.reserve(rows.size());
  for (const FixedTabularTrainingRow &row : rows) {
    auto features = flatten(row.features, parameters, false);
    if (!features)
      return features.takeError();
    std::vector<std::int64_t> targets;
    targets.reserve(targetCount);
    for (std::size_t index = 0; index != targetCount; ++index) {
      auto value = scaleDecimal(row.targets[index],
                                parameters.targetBase10Exponents[index]);
      if (!value)
        return value.takeError();
      targets.push_back(*value);
    }
    fixedRows.push_back({std::move(*features), std::move(targets)});
  }
  auto ensemble = trainDeterministicGbdt(
      fixedRows, config, initial ? &initial->ensemble : nullptr);
  if (!ensemble)
    return ensemble.takeError();
  parameters.ensemble = std::move(*ensemble);
  if (llvm::Error error = validateParameters(parameters))
    return std::move(error);
  return parameters;
}

llvm::Expected<std::optional<std::vector<DecimalValue>>>
inferFixedTabularGbdt(const FixedTabularGbdtParameters &parameters,
                      const FixedTabularFeatureView &features) {
  if (llvm::Error error = validateParameters(parameters))
    return std::move(error);
  auto fixed = flatten(features, parameters, true);
  if (!fixed)
    return fixed.takeError();
  if (fixed->empty())
    return std::optional<std::vector<DecimalValue>>{};
  auto prediction = inferDeterministicGbdt(parameters.ensemble, *fixed);
  if (!prediction)
    return prediction.takeError();
  std::vector<DecimalValue> values;
  values.reserve(prediction->size());
  for (std::size_t index = 0; index != prediction->size(); ++index) {
    auto value = DecimalValue::get((*prediction)[index],
                                   parameters.targetBase10Exponents[index]);
    if (!value)
      return value.takeError();
    values.push_back(*value);
  }
  return std::optional<std::vector<DecimalValue>>(std::move(values));
}

llvm::Expected<std::vector<std::uint8_t>> encodeFixedTabularGbdt(
    const FixedTabularGbdtParameters &parameters,
    llvm::ArrayRef<std::uint8_t> ownerSchemaDescriptorBytes) {
  if (ownerSchemaDescriptorBytes.empty())
    return invalid("owner schema descriptor is empty");
  if (llvm::Error error = validateParameters(parameters))
    return std::move(error);
  auto ensemble = encodeDeterministicGbdt(parameters.ensemble);
  if (!ensemble)
    return ensemble.takeError();
  std::vector<std::uint8_t> bytes;
  appendFramed(bytes, ownerSchemaDescriptorBytes);
  appendU32(bytes, kWireVersion);
  appendFramed(bytes, parameters.groundTruthTargetKey);
  appendU32(bytes, parameters.integralMinimum.size());
  appendU32(bytes, parameters.decimalSupport.size());
  appendU32(bytes, parameters.categoricalSupport.size());
  appendU32(bytes, parameters.presenceSupport.size());
  appendU32(bytes, parameters.targetBase10Exponents.size());
  for (std::size_t index = 0; index != parameters.integralMinimum.size();
       ++index) {
    appendI64(bytes, parameters.integralMinimum[index]);
    appendI64(bytes, parameters.integralMaximum[index]);
  }
  for (const DecimalSupportInterval &support : parameters.decimalSupport) {
    appendDecimal(bytes, support.minimum);
    appendDecimal(bytes, support.maximum);
    appendI64(bytes, support.scaleBase10Exponent);
  }
  for (const auto &support : parameters.categoricalSupport) {
    appendU64(bytes, support.size());
    for (const auto &value : support)
      appendFramed(bytes, value);
  }
  for (std::uint8_t support : parameters.presenceSupport)
    appendU32(bytes, support);
  for (std::int64_t exponent : parameters.targetBase10Exponents)
    appendI64(bytes, exponent);
  appendFramed(bytes, *ensemble);
  return bytes;
}

llvm::Expected<FixedTabularGbdtParameters> decodeFixedTabularGbdt(
    llvm::ArrayRef<std::uint8_t> bytes,
    llvm::ArrayRef<std::uint8_t> ownerSchemaDescriptorBytes,
    std::uint32_t integralFeatureCount, std::uint32_t decimalFeatureCount,
    std::uint32_t categoricalFeatureCount, std::uint32_t presenceFeatureCount,
    std::uint32_t targetCount) {
  Decoder decoder(bytes);
  auto schema = decoder.framed("owner schema descriptor");
  if (!schema)
    return schema.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*schema) != ownerSchemaDescriptorBytes)
    return invalid("payload has a foreign owner schema descriptor");
  auto version = decoder.u32("wire version");
  if (!version)
    return version.takeError();
  if (*version != kWireVersion)
    return invalid("payload wire version is unsupported");
  auto targetKey = decoder.framed("ground-truth target key");
  if (!targetKey)
    return targetKey.takeError();
  auto integralCount = decoder.u32("integral feature count");
  if (!integralCount)
    return integralCount.takeError();
  auto decimalCount = decoder.u32("decimal feature count");
  if (!decimalCount)
    return decimalCount.takeError();
  auto categoricalCount = decoder.u32("categorical feature count");
  if (!categoricalCount)
    return categoricalCount.takeError();
  auto presenceCount = decoder.u32("presence feature count");
  if (!presenceCount)
    return presenceCount.takeError();
  auto heads = decoder.u32("target count");
  if (!heads)
    return heads.takeError();
  if (*integralCount != integralFeatureCount ||
      *decimalCount != decimalFeatureCount ||
      *categoricalCount != categoricalFeatureCount ||
      *presenceCount != presenceFeatureCount || *heads != targetCount)
    return invalid("payload field counts do not match the owner schema");

  FixedTabularGbdtParameters parameters;
  parameters.groundTruthTargetKey = std::move(*targetKey);
  parameters.integralMinimum.reserve(*integralCount);
  parameters.integralMaximum.reserve(*integralCount);
  for (std::uint32_t index = 0; index != *integralCount; ++index) {
    auto minimum = decoder.i64("integral support minimum");
    if (!minimum)
      return minimum.takeError();
    auto maximum = decoder.i64("integral support maximum");
    if (!maximum)
      return maximum.takeError();
    parameters.integralMinimum.push_back(*minimum);
    parameters.integralMaximum.push_back(*maximum);
  }
  parameters.decimalSupport.reserve(*decimalCount);
  for (std::uint32_t index = 0; index != *decimalCount; ++index) {
    auto minimum = decoder.decimal("decimal support minimum");
    if (!minimum)
      return minimum.takeError();
    auto maximum = decoder.decimal("decimal support maximum");
    if (!maximum)
      return maximum.takeError();
    auto scale = decoder.i64("decimal support scale");
    if (!scale)
      return scale.takeError();
    parameters.decimalSupport.push_back({*minimum, *maximum, *scale});
  }
  parameters.categoricalSupport.resize(*categoricalCount);
  for (auto &support : parameters.categoricalSupport) {
    auto count = decoder.u64("categorical support count");
    if (!count)
      return count.takeError();
    if (*count == 0 || *count > decoder.remaining() / 8)
      return invalid("categorical support count is invalid");
    support.reserve(static_cast<std::size_t>(*count));
    for (std::uint64_t index = 0; index != *count; ++index) {
      auto value = decoder.framed("categorical support value");
      if (!value)
        return value.takeError();
      support.push_back(std::move(*value));
    }
  }
  parameters.presenceSupport.reserve(*presenceCount);
  for (std::uint32_t index = 0; index != *presenceCount; ++index) {
    auto support = decoder.u32("presence support");
    if (!support)
      return support.takeError();
    if (*support > std::numeric_limits<std::uint8_t>::max())
      return invalid("presence support exceeds uint8");
    parameters.presenceSupport.push_back(static_cast<std::uint8_t>(*support));
  }
  parameters.targetBase10Exponents.reserve(*heads);
  for (std::uint32_t index = 0; index != *heads; ++index) {
    auto exponent = decoder.i64("target exponent");
    if (!exponent)
      return exponent.takeError();
    parameters.targetBase10Exponents.push_back(*exponent);
  }
  auto ensembleBytes = decoder.framed("GBDT ensemble");
  if (!ensembleBytes)
    return ensembleBytes.takeError();
  if (decoder.remaining() != 0)
    return invalid("payload has trailing bytes");
  auto ensemble = decodeDeterministicGbdt(*ensembleBytes);
  if (!ensemble)
    return ensemble.takeError();
  parameters.ensemble = std::move(*ensemble);
  if (llvm::Error error = validateParameters(parameters))
    return std::move(error);
  auto canonical =
      encodeFixedTabularGbdt(parameters, ownerSchemaDescriptorBytes);
  if (!canonical)
    return canonical.takeError();
  if (llvm::ArrayRef<std::uint8_t>(*canonical) != bytes)
    return invalid("payload does not re-encode canonically");
  return parameters;
}

} // namespace loom::evaluation::models::detail

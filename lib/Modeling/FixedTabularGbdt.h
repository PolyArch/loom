#ifndef LOOM_LIB_MODELING_FIXEDTABULARGBDT_H
#define LOOM_LIB_MODELING_FIXEDTABULARGBDT_H

#include "DeterministicGbdt.h"

#include "Evaluation/NumericValue.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace loom::evaluation::models::detail {

struct FixedTabularFeatureView final {
  std::vector<std::int64_t> integral;
  std::vector<DecimalValue> decimal;
  std::vector<std::vector<std::uint8_t>> categorical;
  std::vector<bool> presence;
};

struct FixedTabularTrainingRow final {
  FixedTabularFeatureView features;
  std::vector<DecimalValue> targets;
};

struct DecimalSupportInterval final {
  DecimalValue minimum;
  DecimalValue maximum;
  std::int64_t scaleBase10Exponent = 0;
};

struct FixedTabularGbdtParameters final {
  std::vector<std::uint8_t> groundTruthTargetKey;
  std::vector<std::int64_t> integralMinimum;
  std::vector<std::int64_t> integralMaximum;
  std::vector<DecimalSupportInterval> decimalSupport;
  std::vector<std::vector<std::vector<std::uint8_t>>> categoricalSupport;
  std::vector<std::uint8_t> presenceSupport;
  std::vector<std::int64_t> targetBase10Exponents;
  DeterministicGbdtEnsemble ensemble;
};

llvm::Expected<FixedTabularGbdtParameters>
trainFixedTabularGbdt(llvm::ArrayRef<FixedTabularTrainingRow> rows,
                      llvm::ArrayRef<std::uint8_t> groundTruthTargetKey,
                      const DeterministicGbdtConfig &config,
                      const FixedTabularGbdtParameters *initial = nullptr);

/// A disengaged result is a structurally valid feature view outside the exact
/// Training envelope. Errors denote malformed or arithmetically invalid data.
llvm::Expected<std::optional<std::vector<DecimalValue>>>
inferFixedTabularGbdt(const FixedTabularGbdtParameters &parameters,
                      const FixedTabularFeatureView &features);

llvm::Expected<std::vector<std::uint8_t>>
encodeFixedTabularGbdt(const FixedTabularGbdtParameters &parameters,
                       llvm::ArrayRef<std::uint8_t> ownerSchemaDescriptorBytes);

llvm::Expected<FixedTabularGbdtParameters> decodeFixedTabularGbdt(
    llvm::ArrayRef<std::uint8_t> bytes,
    llvm::ArrayRef<std::uint8_t> ownerSchemaDescriptorBytes,
    std::uint32_t integralFeatureCount, std::uint32_t decimalFeatureCount,
    std::uint32_t categoricalFeatureCount, std::uint32_t presenceFeatureCount,
    std::uint32_t targetCount);

} // namespace loom::evaluation::models::detail

#endif // LOOM_LIB_MODELING_FIXEDTABULARGBDT_H

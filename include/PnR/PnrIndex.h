#ifndef LOOM_PNR_PNRINDEX_H
#define LOOM_PNR_PNRINDEX_H

#include "PnR/BuildConfig.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace loom::pnr {

static_assert(LOOM_PNR_INDEX_BITS == 32 || LOOM_PNR_INDEX_BITS == 64,
              "LOOM_PNR_INDEX_BITS must be 32 or 64");

using PnrIndex =
    std::conditional_t<LOOM_PNR_INDEX_BITS == 32, std::uint32_t, std::uint64_t>;

inline constexpr unsigned getPnrIndexBits() { return LOOM_PNR_INDEX_BITS; }

inline constexpr std::uint64_t getPnrIndexMax() {
  return std::numeric_limits<PnrIndex>::max();
}

inline constexpr std::string_view getPnrIndexBuildIdentity() {
  return getPnrIndexBits() == 32 ? "LOOM_PNR_INDEX_BITS=32"
                                 : "LOOM_PNR_INDEX_BITS=64";
}

enum class PnrCapacityMeasure {
  Count,
  Index,
  Offset,
};

struct PnrCapacityContext {
  llvm::StringRef artifact;
  llvm::StringRef table;
  llvm::StringRef domain;
  PnrCapacityMeasure measure;
};

class PnrIndexCapacityError final
    : public llvm::ErrorInfo<PnrIndexCapacityError> {
public:
  static char ID;

  explicit PnrIndexCapacityError(std::string message)
      : message_(std::move(message)) {}

  void log(llvm::raw_ostream &stream) const override;
  std::error_code convertToErrorCode() const override;

private:
  std::string message_;
};

llvm::Error preflightPnrIndexCapacity(PnrCapacityContext context,
                                      std::uint64_t requiredMaximum);

llvm::Expected<PnrIndex> checkedPnrIndex(PnrCapacityContext context,
                                         std::uint64_t required);

llvm::Expected<PnrIndex> checkedPnrIndexAdd(PnrCapacityContext context,
                                            std::uint64_t lhs,
                                            std::uint64_t rhs);

llvm::Expected<PnrIndex> checkedPnrIndexMultiply(PnrCapacityContext context,
                                                 std::uint64_t lhs,
                                                 std::uint64_t rhs);

} // namespace loom::pnr

#endif // LOOM_PNR_PNRINDEX_H

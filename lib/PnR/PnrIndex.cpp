#include "PnR/PnrIndex.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <string>
#include <system_error>
#include <utility>

using namespace loom::pnr;

char PnrIndexCapacityError::ID;

void PnrIndexCapacityError::log(llvm::raw_ostream &stream) const {
  stream << message_;
}

std::error_code PnrIndexCapacityError::convertToErrorCode() const {
  return std::make_error_code(std::errc::value_too_large);
}

namespace {

llvm::StringRef capacityMeasureName(PnrCapacityMeasure measure) {
  switch (measure) {
  case PnrCapacityMeasure::Count:
    return "required_max_count";
  case PnrCapacityMeasure::Index:
    return "required_max_index";
  case PnrCapacityMeasure::Offset:
    return "required_max_offset";
  }
  llvm_unreachable("invalid PnR capacity measure");
}

std::string decimalString(const llvm::APInt &value) {
  llvm::SmallString<40> buffer;
  value.toString(buffer, 10, false);
  return std::string(buffer);
}

llvm::Error capacityError(PnrCapacityContext context,
                          const llvm::APInt &requiredMaximum) {
  std::string message;
  llvm::raw_string_ostream stream(message);
  stream << "PnR native index capacity exceeded for artifact '"
         << context.artifact << "', table '" << context.table << "', domain '"
         << context.domain << "': " << capacityMeasureName(context.measure)
         << '=' << decimalString(requiredMaximum) << "; active build "
         << getPnrIndexBuildIdentity() << " (PnrIndex maximum "
         << getPnrIndexMax() << "); ";
  const llvm::APInt maximum64(requiredMaximum.getBitWidth(),
                              std::numeric_limits<std::uint64_t>::max());
  if (requiredMaximum.ugt(maximum64)) {
    stream << "requirement exceeds the supported "
              "LOOM_PNR_INDEX_BITS=64 contract";
  } else {
    stream << "reconfigure and rebuild Loom with "
              "-DLOOM_PNR_INDEX_BITS=64";
  }
  stream.flush();
  return llvm::make_error<PnrIndexCapacityError>(std::move(message));
}

llvm::Error validatePnrIndexCapacity(PnrCapacityContext context,
                                     const llvm::APInt &requiredMaximum) {
  const llvm::APInt activeMaximum(requiredMaximum.getBitWidth(),
                                  getPnrIndexMax());
  if (requiredMaximum.ugt(activeMaximum))
    return capacityError(context, requiredMaximum);
  return llvm::Error::success();
}

llvm::Expected<PnrIndex>
checkedPnrIndexValue(PnrCapacityContext context,
                     const llvm::APInt &requiredMaximum) {
  if (llvm::Error error = validatePnrIndexCapacity(context, requiredMaximum))
    return std::move(error);
  return static_cast<PnrIndex>(requiredMaximum.getZExtValue());
}

} // namespace

llvm::Error
loom::pnr::preflightPnrIndexCapacity(PnrCapacityContext context,
                                     std::uint64_t requiredMaximum) {
  return validatePnrIndexCapacity(context, llvm::APInt(64, requiredMaximum));
}

llvm::Expected<PnrIndex> loom::pnr::checkedPnrIndex(PnrCapacityContext context,
                                                    std::uint64_t required) {
  return checkedPnrIndexValue(context, llvm::APInt(64, required));
}

llvm::Expected<PnrIndex>
loom::pnr::checkedPnrIndexAdd(PnrCapacityContext context, std::uint64_t lhs,
                              std::uint64_t rhs) {
  llvm::APInt requiredMaximum(128, lhs);
  requiredMaximum += llvm::APInt(128, rhs);
  return checkedPnrIndexValue(context, requiredMaximum);
}

llvm::Expected<PnrIndex>
loom::pnr::checkedPnrIndexMultiply(PnrCapacityContext context,
                                   std::uint64_t lhs, std::uint64_t rhs) {
  llvm::APInt requiredMaximum(128, lhs);
  requiredMaximum *= llvm::APInt(128, rhs);
  return checkedPnrIndexValue(context, requiredMaximum);
}

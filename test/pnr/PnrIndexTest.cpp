#include "PnR/PnrIndex.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

using namespace loom::pnr;

namespace {

static_assert(std::is_unsigned_v<PnrIndex>);
static_assert(sizeof(PnrIndex) * 8 == getPnrIndexBits());

constexpr PnrCapacityContext countContext{"artifact:demo", "candidate.entries",
                                          "candidate_entities",
                                          PnrCapacityMeasure::Count};
constexpr PnrCapacityContext indexContext{"artifact:demo", "routing.vertices",
                                          "routing_vertices",
                                          PnrCapacityMeasure::Index};
constexpr PnrCapacityContext offsetContext{
    "artifact:demo", "routing.csr_offsets", "routing_arcs",
    PnrCapacityMeasure::Offset};

void fail(const char *test, const char *message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(1);
}

void requireContains(const char *test, std::string_view text,
                     std::string_view expected) {
  if (text.find(expected) == std::string_view::npos)
    fail(test, ("missing diagnostic text: " + std::string(expected)).c_str());
}

void requireNotContains(const char *test, std::string_view text,
                        std::string_view unexpected) {
  if (text.find(unexpected) != std::string_view::npos)
    fail(test,
         ("unexpected diagnostic text: " + std::string(unexpected)).c_str());
}

template <typename T> T takeValue(const char *test, llvm::Expected<T> result) {
  if (!result)
    fail(test, llvm::toString(result.takeError()).c_str());
  return *result;
}

std::string takeCapacityError(const char *test, llvm::Error error) {
  if (!error)
    fail(test, "expected PnR index capacity failure");

  bool sawCapacityError = false;
  std::string message;
  llvm::handleAllErrors(std::move(error),
                        [&](const PnrIndexCapacityError &capacityError) {
                          sawCapacityError = true;
                          llvm::raw_string_ostream stream(message);
                          capacityError.log(stream);
                        });
  if (!sawCapacityError)
    fail(test, "received a different error category");
  return message;
}

template <typename T>
std::string takeCapacityError(const char *test, llvm::Expected<T> result) {
  if (result)
    fail(test, "expected PnR index capacity failure");
  return takeCapacityError(test, result.takeError());
}

void expectPreflightSuccess(const char *test, PnrCapacityContext context,
                            std::uint64_t requiredMaximum) {
  if (llvm::Error error = preflightPnrIndexCapacity(context, requiredMaximum))
    fail(test, llvm::toString(std::move(error)).c_str());
}

void exposesConfiguredWidthAndIdentity() {
  const unsigned bits = getPnrIndexBits();
  if (bits != 32 && bits != 64)
    fail(__func__, "configured width is neither 32 nor 64 bits");
  if (sizeof(PnrIndex) * 8 != bits)
    fail(__func__, "PnrIndex does not match the configured width");

  const std::string_view expectedIdentity =
      bits == 32 ? "LOOM_PNR_INDEX_BITS=32" : "LOOM_PNR_INDEX_BITS=64";
  if (getPnrIndexBuildIdentity() != expectedIdentity)
    fail(__func__, "native index build identity is inconsistent");
}

void acceptsActiveWidthBoundary() {
  const std::uint64_t maximum = getPnrIndexMax();
  if (takeValue(__func__, checkedPnrIndex(indexContext, maximum)) != maximum)
    fail(__func__, "checked conversion changed the maximum value");
  if (takeValue(__func__, checkedPnrIndexAdd(countContext, maximum - 1, 1)) !=
      maximum)
    fail(__func__, "checked addition changed the maximum value");
  if (takeValue(__func__, checkedPnrIndexMultiply(offsetContext, maximum, 1)) !=
      maximum)
    fail(__func__, "checked multiplication changed the maximum value");
}

void preflightsCapacityWithoutEncoding() {
  const std::uint64_t maximum = getPnrIndexMax();
  expectPreflightSuccess(__func__, countContext, maximum);
  expectPreflightSuccess(__func__, indexContext, maximum);
  expectPreflightSuccess(__func__, offsetContext, maximum);

  if (getPnrIndexBits() == 32) {
    std::string message = takeCapacityError(
        __func__, preflightPnrIndexCapacity(countContext, maximum + 1));
    requireContains(__func__, message, "required_max_count=4294967296");
    requireContains(__func__, message, "-DLOOM_PNR_INDEX_BITS=64");
    requireNotContains(__func__, message,
                       "exceeds the supported LOOM_PNR_INDEX_BITS=64 contract");
  }
}

void reportsTypedCapacityMeasures() {
  const std::uint64_t maximum = getPnrIndexMax();
  const std::string_view maximumPlusOne =
      getPnrIndexBits() == 32 ? "4294967296" : "18446744073709551616";
  const std::string_view maximumTimesTwo =
      getPnrIndexBits() == 32 ? "8589934590" : "36893488147419103230";

  std::string message =
      takeCapacityError(__func__, checkedPnrIndexAdd(countContext, maximum, 1));

  requireContains(__func__, message, "PnR native index capacity exceeded");
  requireContains(__func__, message, "artifact 'artifact:demo'");
  requireContains(__func__, message, "table 'candidate.entries'");
  requireContains(__func__, message, "domain 'candidate_entities'");
  requireContains(__func__, message,
                  "required_max_count=" + std::string(maximumPlusOne));
  requireContains(__func__, message, getPnrIndexBuildIdentity());
  if (getPnrIndexBits() == 32) {
    requireContains(__func__, message, "-DLOOM_PNR_INDEX_BITS=64");
  } else {
    requireContains(__func__, message,
                    "exceeds the supported LOOM_PNR_INDEX_BITS=64 contract");
  }

  message =
      takeCapacityError(__func__, checkedPnrIndexAdd(indexContext, maximum, 1));
  requireContains(__func__, message,
                  "required_max_index=" + std::string(maximumPlusOne));

  message = takeCapacityError(
      __func__,
      checkedPnrIndexMultiply(offsetContext, maximum, std::uint64_t{2}));
  requireContains(__func__, message,
                  "required_max_offset=" + std::string(maximumTimesTwo));
  requireNotContains(__func__, message, "required value");
}

void rejectsRequirementsBeyondThe64BitContract() {
  const std::uint64_t maximum64 = std::numeric_limits<std::uint64_t>::max();
  std::string message = takeCapacityError(
      __func__, checkedPnrIndexAdd(indexContext, maximum64, 1));

  requireContains(__func__, message, "required_max_index=18446744073709551616");
  requireContains(__func__, message,
                  "exceeds the supported LOOM_PNR_INDEX_BITS=64 contract");
  requireNotContains(__func__, message, "-DLOOM_PNR_INDEX_BITS=64");
}

} // namespace

int main() {
  exposesConfiguredWidthAndIdentity();
  acceptsActiveWidthBoundary();
  preflightsCapacityWithoutEncoding();
  reportsTypedCapacityMeasures();
  rejectsRequirementsBeyondThe64BitContract();
  return 0;
}

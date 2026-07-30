#include "DeterministicTranscendental.h"

#include "llvm/ADT/APFloat.h"

#include <limits>
#include <mpfr.h>
#include <system_error>

namespace loom::sim::detail {
namespace {

class MpfrValue final {
public:
  explicit MpfrValue(mpfr_prec_t precision) { mpfr_init2(value_, precision); }
  ~MpfrValue() { mpfr_clear(value_); }

  MpfrValue(const MpfrValue &) = delete;
  MpfrValue &operator=(const MpfrValue &) = delete;

  mpfr_ptr get() { return value_; }
  mpfr_srcptr get() const { return value_; }

private:
  mpfr_t value_;
};

bool isSupportedSemantic(const llvm::fltSemantics &semantics) {
  return &semantics == &llvm::APFloat::IEEEhalf() ||
         &semantics == &llvm::APFloat::BFloat() ||
         &semantics == &llvm::APFloat::IEEEsingle() ||
         &semantics == &llvm::APFloat::IEEEdouble();
}

} // namespace

llvm::Expected<llvm::APFloat>
evaluateDeterministicCosine(const llvm::APFloat &operand) {
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53);

  const llvm::fltSemantics &semantics = operand.getSemantics();
  if (!isSupportedSemantic(semantics))
    return llvm::createStringError(
        std::errc::not_supported,
        "deterministic cosine supports only f16, bf16, f32, and f64");
  if (operand.isNaN())
    return operand.makeQuiet();
  if (operand.isInfinity())
    return llvm::APFloat::getNaN(semantics);

  MpfrValue input(
      llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEdouble()));
  MpfrValue result(llvm::APFloat::semanticsPrecision(semantics));
  mpfr_set_d(input.get(), operand.convertToDouble(), MPFR_RNDN);
  mpfr_cos(result.get(), input.get(), MPFR_RNDN);

  llvm::APFloat rounded(mpfr_get_d(result.get(), MPFR_RNDN));
  if (&semantics != &llvm::APFloat::IEEEdouble()) {
    bool losesInformation = false;
    (void)rounded.convert(semantics, llvm::RoundingMode::NearestTiesToEven,
                          &losesInformation);
  }
  return rounded;
}

} // namespace loom::sim::detail

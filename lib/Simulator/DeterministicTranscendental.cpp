#include "DeterministicTranscendental.h"

#include "llvm/ADT/APFloat.h"

#include <limits>
#include <mpfr.h>
#include <mutex>
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

std::mutex &mpfrEnvironmentMutex() {
  static std::mutex mutex;
  return mutex;
}

class ScopedMpfrEnvironment final {
public:
  ScopedMpfrEnvironment() : lock_(mpfrEnvironmentMutex(), std::defer_lock) {
    if (!mpfr_buildopt_tls_p())
      lock_.lock();
    previousMinimumExponent_ = mpfr_get_emin();
    previousMaximumExponent_ = mpfr_get_emax();
    previousFlags_ = mpfr_flags_save();
  }

  ~ScopedMpfrEnvironment() {
    (void)mpfr_set_emax(previousMaximumExponent_);
    (void)mpfr_set_emin(previousMinimumExponent_);
    mpfr_flags_restore(previousFlags_, MPFR_FLAGS_ALL);
  }

  ScopedMpfrEnvironment(const ScopedMpfrEnvironment &) = delete;
  ScopedMpfrEnvironment &operator=(const ScopedMpfrEnvironment &) = delete;

  llvm::Error select(const llvm::fltSemantics &semantics) {
    const mpfr_exp_t precision = llvm::APFloat::semanticsPrecision(semantics);
    const mpfr_exp_t minimumExponent =
        llvm::APFloat::semanticsMinExponent(semantics) - precision + 2;
    const mpfr_exp_t maximumExponent =
        llvm::APFloat::semanticsMaxExponent(semantics) + 1;
    if (mpfr_set_emin(minimumExponent) != 0 ||
        mpfr_set_emax(maximumExponent) != 0)
      return llvm::createStringError(
          std::errc::not_supported,
          "cannot select the target IEEE exponent range");
    mpfr_clear_flags();
    return llvm::Error::success();
  }

private:
  std::unique_lock<std::mutex> lock_;
  mpfr_exp_t previousMinimumExponent_ = 0;
  mpfr_exp_t previousMaximumExponent_ = 0;
  mpfr_flags_t previousFlags_ = 0;
};

bool isSupportedSemantic(const llvm::fltSemantics &semantics) {
  return &semantics == &llvm::APFloat::IEEEhalf() ||
         &semantics == &llvm::APFloat::BFloat() ||
         &semantics == &llvm::APFloat::IEEEsingle() ||
         &semantics == &llvm::APFloat::IEEEdouble();
}

using MpfrUnaryOperation = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
using MpfrBinaryOperation = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr,
                                    mpfr_rnd_t);

MpfrUnaryOperation mpfrOperation(dataflow::OperationSchemaId schema) {
  using Schema = dataflow::OperationSchemaId;
  switch (schema) {
  case Schema::MathSin:
    return &mpfr_sin;
  case Schema::MathCos:
    return &mpfr_cos;
  case Schema::MathTan:
    return &mpfr_tan;
  case Schema::MathSinh:
    return &mpfr_sinh;
  case Schema::MathCosh:
    return &mpfr_cosh;
  case Schema::MathTanh:
    return &mpfr_tanh;
  case Schema::MathExp:
    return &mpfr_exp;
  case Schema::MathExp2:
    return &mpfr_exp2;
  case Schema::MathExpM1:
    return &mpfr_expm1;
  case Schema::MathLog:
    return &mpfr_log;
  case Schema::MathLog2:
    return &mpfr_log2;
  case Schema::MathLog10:
    return &mpfr_log10;
  case Schema::MathLog1p:
    return &mpfr_log1p;
  case Schema::MathSqrt:
    return &mpfr_sqrt;
  case Schema::MathRsqrt:
    return &mpfr_rec_sqrt;
  case Schema::MathErf:
    return &mpfr_erf;
  default:
    return nullptr;
  }
}

MpfrBinaryOperation mpfrBinaryOperation(dataflow::OperationSchemaId schema) {
  switch (schema) {
  case dataflow::OperationSchemaId::MathPowF:
    return &mpfr_pow;
  default:
    return nullptr;
  }
}

llvm::APFloat roundedResult(mpfr_srcptr result,
                            const llvm::fltSemantics &semantics) {
  llvm::APFloat rounded(mpfr_get_d(result, MPFR_RNDN));
  if (&semantics != &llvm::APFloat::IEEEdouble()) {
    bool losesInformation = false;
    (void)rounded.convert(semantics, llvm::RoundingMode::NearestTiesToEven,
                          &losesInformation);
  }
  return rounded;
}

} // namespace

llvm::Expected<llvm::APFloat>
evaluateDeterministicUnaryMath(dataflow::OperationSchemaId schema,
                               const llvm::APFloat &operand) {
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53);

  MpfrUnaryOperation operation = mpfrOperation(schema);
  if (!operation)
    return llvm::createStringError(
        std::errc::not_supported, "%s is not deterministic unary math",
        dataflow::operationSchemaSpelling(schema).str().c_str());
  const llvm::fltSemantics &semantics = operand.getSemantics();
  if (!isSupportedSemantic(semantics))
    return llvm::createStringError(
        std::errc::not_supported,
        "deterministic unary math supports only f16, bf16, f32, and f64");
  if (operand.isNaN())
    return operand.makeQuiet();

  ScopedMpfrEnvironment environment;
  if (llvm::Error error = environment.select(semantics))
    return std::move(error);
  MpfrValue input(
      llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEdouble()));
  MpfrValue result(llvm::APFloat::semanticsPrecision(semantics));
  mpfr_set_d(input.get(), operand.convertToDouble(), MPFR_RNDN);
  const int ternary = operation(result.get(), input.get(), MPFR_RNDN);
  (void)mpfr_subnormalize(result.get(), ternary, MPFR_RNDN);

  return roundedResult(result.get(), semantics);
}

llvm::Expected<llvm::APFloat>
evaluateDeterministicBinaryMath(dataflow::OperationSchemaId schema,
                                const llvm::APFloat &lhs,
                                const llvm::APFloat &rhs) {
  static_assert(std::numeric_limits<double>::is_iec559 &&
                std::numeric_limits<double>::digits == 53);

  MpfrBinaryOperation operation = mpfrBinaryOperation(schema);
  if (!operation)
    return llvm::createStringError(
        std::errc::not_supported, "%s is not deterministic binary math",
        dataflow::operationSchemaSpelling(schema).str().c_str());
  const llvm::fltSemantics &semantics = lhs.getSemantics();
  if (&rhs.getSemantics() != &semantics)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "deterministic binary math requires identical operand semantics");
  if (!isSupportedSemantic(semantics))
    return llvm::createStringError(
        std::errc::not_supported,
        "deterministic binary math supports only f16, bf16, f32, and f64");

  ScopedMpfrEnvironment environment;
  if (llvm::Error error = environment.select(semantics))
    return std::move(error);
  const mpfr_prec_t inputPrecision =
      llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEdouble());
  MpfrValue left(inputPrecision);
  MpfrValue right(inputPrecision);
  MpfrValue result(llvm::APFloat::semanticsPrecision(semantics));
  mpfr_set_d(left.get(), lhs.convertToDouble(), MPFR_RNDN);
  mpfr_set_d(right.get(), rhs.convertToDouble(), MPFR_RNDN);
  const int ternary =
      operation(result.get(), left.get(), right.get(), MPFR_RNDN);
  (void)mpfr_subnormalize(result.get(), ternary, MPFR_RNDN);
  return roundedResult(result.get(), semantics);
}

} // namespace loom::sim::detail

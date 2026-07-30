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

using MpfrUnaryOperation = int (*)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);

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

  MpfrValue input(
      llvm::APFloat::semanticsPrecision(llvm::APFloat::IEEEdouble()));
  MpfrValue result(llvm::APFloat::semanticsPrecision(semantics));
  mpfr_set_d(input.get(), operand.convertToDouble(), MPFR_RNDN);
  operation(result.get(), input.get(), MPFR_RNDN);

  llvm::APFloat rounded(mpfr_get_d(result.get(), MPFR_RNDN));
  if (&semantics != &llvm::APFloat::IEEEdouble()) {
    bool losesInformation = false;
    (void)rounded.convert(semantics, llvm::RoundingMode::NearestTiesToEven,
                          &losesInformation);
  }
  return rounded;
}

} // namespace loom::sim::detail

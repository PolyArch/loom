//===- FloatBehaviorProfileTest.cpp - Floating capability invariants ------===//

#include "Fabric/IR/ImplementationFamily.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

namespace {

using namespace fabric;

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

::dataflow::CanonicalActorSchemaProjection
addActor(mlir::MLIRContext &context, mlir::arith::FastMathFlags flags) {
  mlir::Type type = mlir::Float32Type::get(&context);
  return {::dataflow::OperationSchemaId::ArithAddF,
          mlir::FunctionType::get(&context, {type, type}, {type}),
          ::dataflow::FloatingPointPayload{flags, std::nullopt}};
}

llvm::Error admit(const FloatBehaviorProfile &behavior,
                  const ::dataflow::CanonicalActorSchemaProjection &actor) {
  const FamilyCapabilityParams params =
      ScalarFloatParams{FloatFormatSet::get({FloatFormat::F32}), behavior};
  return verifyImplementationFamilyAdmission(
      ImplementationFamilyId::ScalarFloatAddSub, &params, actor);
}

void expectRejected(const char *test, FloatBehaviorProfile behavior,
                    const ::dataflow::CanonicalActorSchemaProjection &actor,
                    llvm::StringRef fragment) {
  llvm::Error error = admit(behavior, actor);
  if (!error)
    fail(test, "invalid floating behavior profile was admitted");
  const std::string message = llvm::toString(std::move(error));
  if (!llvm::StringRef(message).contains(fragment))
    fail(test, "unexpected rejection: " + message);
}

void expectAdmitted(const char *test, FloatBehaviorProfile behavior,
                    const ::dataflow::CanonicalActorSchemaProjection &actor) {
  if (llvm::Error error = admit(behavior, actor))
    fail(test, llvm::toString(std::move(error)));
}

void subnormalBehaviorHasNoOrphanValues() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const auto actor = addActor(context, mlir::arith::FastMathFlags::none);

  FloatBehaviorProfile multiple = FloatBehaviorProfile::strictIEEE();
  multiple.subnormalBehaviors = FloatSubnormalBehaviorSet::get(
      {FloatSubnormalBehavior::Preserve, FloatSubnormalBehavior::FlushToZero});
  expectRejected(test, multiple, actor, "subnormal");

  FloatBehaviorProfile flushOnly = FloatBehaviorProfile::strictIEEE();
  flushOnly.subnormalBehaviors =
      FloatSubnormalBehaviorSet::get({FloatSubnormalBehavior::FlushToZero});
  expectRejected(test, flushOnly, actor, "subnormal");
}

void signedZeroBehaviorIsOneConcreteChoice() {
  const char *test = __func__;
  mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
  const auto strictActor = addActor(context, mlir::arith::FastMathFlags::none);

  FloatBehaviorProfile multiple = FloatBehaviorProfile::strictIEEE();
  multiple.signedZeroBehaviors = FloatSignedZeroBehaviorSet::get(
      {FloatSignedZeroBehavior::Preserve, FloatSignedZeroBehavior::IgnoreSign});
  expectRejected(test, multiple, strictActor, "signed-zero");

  FloatBehaviorProfile ignored = FloatBehaviorProfile::strictIEEE();
  ignored.signedZeroBehaviors =
      FloatSignedZeroBehaviorSet::get({FloatSignedZeroBehavior::IgnoreSign});
  const auto noSignedZeros = addActor(context, mlir::arith::FastMathFlags::nsz);
  expectAdmitted(test, ignored, noSignedZeros);
}

} // namespace

int main() {
  subnormalBehaviorHasNoOrphanValues();
  signedZeroBehaviorIsOneConcreteChoice();
  return EXIT_SUCCESS;
}

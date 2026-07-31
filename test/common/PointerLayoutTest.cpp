#include "Common/PointerLayout.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace {

bool expectFailure(llvm::Expected<loom::PointerLayout> layout,
                   llvm::StringRef reason) {
  if (layout) {
    llvm::errs() << "pointer layout unexpectedly succeeded\n";
    return false;
  }
  std::string message = llvm::toString(layout.takeError());
  if (llvm::StringRef(message).contains(reason))
    return true;
  llvm::errs() << "pointer layout failed for the wrong reason: " << message
               << '\n';
  return false;
}

} // namespace

int main() {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  module->setAttr(
      "llvm.data_layout",
      builder.getStringAttr("e-p:64:64-p1:128:128:128:64-p2:64:64:64:32-ni:1"));

  bool ok = true;
  auto defaultPointer = loom::resolvePointerLayout(module, 0);
  if (!defaultPointer ||
      *defaultPointer !=
          loom::PointerLayout{0, 64, 64,
                              loom::PointerLayoutKind::StableIntegral}) {
    llvm::errs() << "default pointer layout was not derived exactly\n";
    if (defaultPointer)
      llvm::errs() << "actual layout: address-space="
                   << defaultPointer->addressSpace
                   << " representation=" << defaultPointer->representationBits
                   << " address=" << defaultPointer->addressBits
                   << " kind=" << static_cast<unsigned>(defaultPointer->kind)
                   << '\n';
    else
      llvm::consumeError(defaultPointer.takeError());
    ok = false;
  }

  auto nonIntegral = loom::resolvePointerLayout(module, 1);
  if (!nonIntegral ||
      *nonIntegral !=
          loom::PointerLayout{1, 128, 64, loom::PointerLayoutKind::Unstable}) {
    llvm::errs()
        << "non-integral pointer layout lost its representation kind\n";
    if (!nonIntegral)
      llvm::consumeError(nonIntegral.takeError());
    ok = false;
  }

  auto narrowAddress = loom::resolvePointerLayout(module, 2);
  if (!narrowAddress ||
      *narrowAddress != loom::PointerLayout{
                            2, 64, 32, loom::PointerLayoutKind::NonIntegral}) {
    llvm::errs() << "narrow pointer address lost its representation kind\n";
    if (!narrowAddress)
      llvm::consumeError(narrowAddress.takeError());
    ok = false;
  }

  auto missing = mlir::ModuleOp::create(builder.getUnknownLoc());
  ok &= expectFailure(loom::resolvePointerLayout(missing, 0),
                      "nonempty LLVM DataLayout");
  missing->setAttr("llvm.data_layout", builder.getStringAttr("not-a-layout"));
  ok &= expectFailure(loom::resolvePointerLayout(missing, 0),
                      "cannot parse LLVM DataLayout");
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

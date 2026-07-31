#include "Common/PointerLayout.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"

#include <system_error>

namespace loom {

namespace {

llvm::Expected<unsigned> checkWidth(unsigned width, llvm::StringRef role) {
  if (width == 0 || width > mlir::IntegerType::kMaxWidth)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "pointer %s width %u has no fixed integer representation",
        role.str().c_str(), width);
  return width;
}

mlir::ModuleOp findEnclosingModule(mlir::Operation *op) {
  if (!op)
    return {};
  if (auto module = mlir::dyn_cast<mlir::ModuleOp>(op))
    return module;
  return op->getParentOfType<mlir::ModuleOp>();
}

} // namespace

llvm::Expected<llvm::DataLayout> resolveLLVMDataLayout(mlir::Operation *op) {
  mlir::ModuleOp module = findEnclosingModule(op);
  auto spelling =
      module ? module->getAttrOfType<mlir::StringAttr>("llvm.data_layout")
             : mlir::StringAttr{};
  if (!spelling || spelling.getValue().empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "pointer layout requires a nonempty LLVM DataLayout");

  llvm::Expected<llvm::DataLayout> parsed =
      llvm::DataLayout::parse(spelling.getValue());
  if (!parsed)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot parse LLVM DataLayout: %s",
                                   llvm::toString(parsed.takeError()).c_str());
  return parsed;
}

llvm::Expected<PointerLayout> resolvePointerLayout(mlir::Operation *op,
                                                   std::uint32_t addressSpace) {
  llvm::Expected<llvm::DataLayout> parsed = resolveLLVMDataLayout(op);
  if (!parsed)
    return parsed.takeError();

  llvm::Expected<unsigned> representationBits =
      checkWidth(parsed->getPointerSizeInBits(addressSpace), "representation");
  if (!representationBits)
    return representationBits.takeError();
  llvm::Expected<unsigned> addressBits =
      checkWidth(parsed->getIndexSizeInBits(addressSpace), "address");
  if (!addressBits)
    return addressBits.takeError();
  if (*addressBits > *representationBits)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "pointer address width %u exceeds representation width %u",
        *addressBits, *representationBits);

  PointerLayoutKind kind = PointerLayoutKind::StableIntegral;
  if (parsed->hasExternalState(addressSpace))
    kind = PointerLayoutKind::ExternalState;
  else if (parsed->hasUnstableRepresentation(addressSpace))
    kind = PointerLayoutKind::Unstable;
  else if (parsed->isNonIntegralAddressSpace(addressSpace))
    kind = PointerLayoutKind::NonIntegral;

  return PointerLayout{addressSpace, *representationBits, *addressBits, kind};
}

} // namespace loom

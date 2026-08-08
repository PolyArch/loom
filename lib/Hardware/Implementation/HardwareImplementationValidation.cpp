#include "HardwareImplementationInternal.h"

#include "llvm/ADT/Twine.h"

#include <cctype>

namespace loom::hardware::detail {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "hardware_implementation_invalid: " + message);
}

} // namespace

llvm::Error
validateRepresentationLocator(const RepresentationLocator &locator,
                              const ImplementationRepresentationRoot &root) {
  if (locator.canonicalName.empty())
    return invalid("representation locator name must be nonempty");
  for (char character : locator.canonicalName) {
    const unsigned char byte = static_cast<unsigned char>(character);
    if (byte < 0x21 || byte > 0x7e)
      return invalid("representation locator name must be printable ASCII");
  }
  const bool rtlObject = locator.kind == RepresentationObjectKind::Module ||
                         locator.kind == RepresentationObjectKind::Instance ||
                         locator.kind == RepresentationObjectKind::Port ||
                         locator.kind == RepresentationObjectKind::Net ||
                         locator.kind == RepresentationObjectKind::Register ||
                         locator.kind == RepresentationObjectKind::Memory;
  if (root.variant == RepresentationRootVariant::Rtl && !rtlObject)
    return invalid("RTL representation uses an incompatible locator kind");
  return llvm::Error::success();
}

} // namespace loom::hardware::detail

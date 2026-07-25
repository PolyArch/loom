#include "Frontend/Payload/PayloadCarrier.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#include <memory>
#include <utility>

namespace loom {
namespace {

llvm::Error rejected(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

} // namespace

llvm::StringRef relocatablePayloadCarrierSection() {
  return ".loom.relocatable_accelerator_payload";
}

void embedRelocatablePayloadCarrier(
    llvm::Module &module, llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  llvm::Constant *bytes =
      llvm::ConstantDataArray::get(module.getContext(), canonicalBytes);
  auto *carrier = new llvm::GlobalVariable(
      module, bytes->getType(), /*isConstant=*/true,
      llvm::GlobalValue::PrivateLinkage, bytes,
      "loom.relocatable_accelerator_payload");
  carrier->setSection(relocatablePayloadCarrierSection());

  // Byte alignment keeps the emitted section exactly the payload bytes, with no
  // padding a reader would have to strip back off.
  carrier->setAlignment(llvm::Align(1));

  // The carrier has no in-module use, so it is anchored the way LLVM anchors
  // every other compiler-generated section: nothing else may drop it.
  llvm::appendToCompilerUsed(module, carrier);
}

llvm::Expected<std::optional<std::vector<std::uint8_t>>>
readRelocatablePayloadCarrier(llvm::MemoryBufferRef object) {
  llvm::Expected<std::unique_ptr<llvm::object::ObjectFile>> parsed =
      llvm::object::ObjectFile::createObjectFile(object);
  if (!parsed)
    return rejected("payload_carrier_unreadable: " +
                    llvm::toString(parsed.takeError()));

  std::optional<std::vector<std::uint8_t>> carried;
  for (const llvm::object::SectionRef &section : (*parsed)->sections()) {
    llvm::Expected<llvm::StringRef> name = section.getName();
    if (!name)
      return rejected("payload_carrier_unreadable: " +
                      llvm::toString(name.takeError()));
    if (*name != relocatablePayloadCarrierSection())
      continue;
    if (carried)
      return rejected("payload_carrier_ambiguous: the object carries section " +
                      relocatablePayloadCarrierSection() +
                      " more than once, so it delivers no single payload");
    llvm::Expected<llvm::StringRef> contents = section.getContents();
    if (!contents)
      return rejected("payload_carrier_unreadable: " +
                      llvm::toString(contents.takeError()));
    carried.emplace(contents->bytes_begin(), contents->bytes_end());
  }
  return carried;
}

} // namespace loom

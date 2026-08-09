#include "Hardware/Implementation/RepresentationIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace loom::hardware {
namespace detail {

llvm::Error invalidIndex(const llvm::Twine &reason) {
  return llvm::make_error<RepresentationIndexFailure>(
      RepresentationIndexFailureKind::Invalid, reason.str());
}

llvm::Error unsupportedIndex(const llvm::Twine &reason) {
  return llvm::make_error<RepresentationIndexFailure>(
      RepresentationIndexFailureKind::Unsupported, reason.str());
}

llvm::Error
validateRepresentationTextPolicy(RepresentationTextPolicy policy,
                                 const ImplementationPayload &payload,
                                 llvm::ArrayRef<std::uint8_t> contents) {
  switch (policy) {
  case RepresentationTextPolicy::Opaque:
    return llvm::Error::success();
  case RepresentationTextPolicy::Utf8LfNoNul:
    break;
  }
  const llvm::StringRef text(reinterpret_cast<const char *>(contents.data()),
                             contents.size());
  if (text.contains('\0'))
    return invalidIndex("text payload '" + payload.canonicalLogicalName +
                        "' contains a NUL byte");
  if (text.contains('\r'))
    return invalidIndex("text payload '" + payload.canonicalLogicalName +
                        "' does not use LF line endings");
  if (!llvm::json::isUTF8(text))
    return invalidIndex("text payload '" + payload.canonicalLogicalName +
                        "' is not valid UTF-8");
  return llvm::Error::success();
}

} // namespace detail

char RepresentationIndexFailure::ID;

void RepresentationIndexFailure::log(llvm::raw_ostream &stream) const {
  stream << (kind_ == RepresentationIndexFailureKind::Invalid
                 ? "representation_index_invalid: "
                 : "representation_index_unsupported: ")
         << reason_;
}

std::error_code RepresentationIndexFailure::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}

llvm::Expected<std::optional<RepresentationObjectFacts>>
RepresentationIndex::lookup(const RepresentationLocator &locator) const {
  const RepresentationFormatDescriptor &descriptor =
      getRepresentationFormatDescriptor(formatRef_);
  const RepresentationRootAdmission *admission =
      findRepresentationRootAdmission(descriptor, rootVariant_, stage_);
  if (!admission)
    return detail::invalidIndex(
        "indexed root claim has no exact format admission");
  if (!llvm::is_contained(admission->admittedObjectKinds, locator.kind))
    return detail::invalidIndex(
        "lookup locator object kind is not admitted by the indexed root");
  if (llvm::Error error =
          validateRepresentationLocatorSyntax(formatRef_, locator))
    return detail::invalidIndex("lookup locator is invalid: " +
                                llvm::toString(std::move(error)));
  const llvm::StringRef name(locator.canonicalName);
  const llvm::StringRef root(exactRoot_.canonicalName);
  const bool rooted = locator == exactRoot_ ||
                      (name.starts_with(root) && name.size() > root.size() &&
                       name[root.size()] == '.');
  const bool unresolved =
      llvm::is_contained(unresolvedExternalDefinitions_, locator);
  const bool unrootedModule =
      exactRoot_.kind == RepresentationObjectKind::Module &&
      locator.kind == RepresentationObjectKind::Module;
  if (!rooted && !unresolved && !unrootedModule)
    return detail::invalidIndex(
        "lookup locator is not rooted at the indexed exact root");
  const auto found = llvm::lower_bound(
      entries_, locator,
      [](const Entry &entry, const RepresentationLocator &key) {
        return representationLocatorCanonicalLess(entry.locator, key);
      });
  if (found == entries_.end() || !(found->locator == locator))
    return std::optional<RepresentationObjectFacts>();
  return std::optional<RepresentationObjectFacts>(found->facts);
}

} // namespace loom::hardware

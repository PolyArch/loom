#include "Hardware/Implementation/RepresentationIndex.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <utility>

namespace loom::hardware {
namespace {

llvm::Error invalidIndex(const llvm::Twine &reason) {
  return llvm::make_error<RepresentationIndexFailure>(
      RepresentationIndexFailureKind::Invalid, reason.str());
}

} // namespace

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
  if (llvm::Error error =
          validateRepresentationLocatorSyntax(formatRef_, locator))
    return invalidIndex("lookup locator is invalid: " +
                        llvm::toString(std::move(error)));
  const llvm::StringRef name(locator.canonicalName);
  const llvm::StringRef root(exactRoot_.canonicalName);
  if (locator.kind != exactRoot_.kind &&
      !(name.starts_with(root) && name.size() > root.size() &&
        name[root.size()] == '.'))
    return invalidIndex(
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

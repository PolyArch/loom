#ifndef LOOM_FABRIC_IDENTITY_FABRICREFTEXT_H
#define LOOM_FABRIC_IDENTITY_FABRICREFTEXT_H

#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace loom {
namespace fabric {

/// The accepted language is exactly the canonical typed projection of the
/// reference catalog: one family keyword per named family, one closed keyword
/// per variant, unsigned decimal fields in declaration order, and `, ` between
/// fields. There is no optional whitespace, symbol form, attribute alias,
/// printer position, path or property escape, or numeric alias, so parsing
/// followed by canonical printing always reproduces the accepted spelling.
class FabricRefScanner {
public:
  explicit FabricRefScanner(llvm::StringRef text) : rest_(text) {}

  /// Consumes `literal` or reports malformed input.
  llvm::Error expect(llvm::StringRef literal);
  /// Consumes one canonical unsigned 64-bit decimal field.
  llvm::Expected<std::uint64_t> unsignedField();
  /// Consumes the next family or variant keyword without committing to it.
  llvm::StringRef peekKeyword() const;
  /// Consumes a keyword previously matched with `peekKeyword`.
  void take(std::size_t size) { rest_ = rest_.drop_front(size); }
  /// Reports anything left after a complete reference.
  llvm::Error finish() const;

  llvm::StringRef rest() const { return rest_; }

private:
  llvm::StringRef rest_;
};

/// Classifies a spelling that is not the canonical typed language. Retired
/// spellings and generic path, property, symbol, attribute, and printer
/// position escapes are reported as such instead of as syntax noise.
llvm::Error fabricRefTextError(const llvm::Twine &context,
                               llvm::StringRef rest);

//===---------------------------------------------------------------------===//
// Printing
//===---------------------------------------------------------------------===//

/// Consumes exactly `family`. A different catalog family, a retired spelling,
/// or a generic escape is classified rather than reported as syntax noise.
llvm::Error fabricExpectFamily(FabricRefScanner &scanner,
                               llvm::StringRef family);

template <FabricEntityKind Kind>
void printFabricRef(llvm::raw_ostream &os,
                    const FabricTypedEntityRef<Kind> &ref);
template <typename Ref>
void printFabricRef(llvm::raw_ostream &os, const Ref &ref);

void printFabricRef(llvm::raw_ostream &os,
                    const FabricTransportEndpointOwnerRef &owner);
void printFabricRef(llvm::raw_ostream &os,
                    const FabricMemoryEndpointOwnerRef &owner);
void printFabricRef(llvm::raw_ostream &os,
                    const FabricInventoryOwnerRef &owner);
void printFabricRef(llvm::raw_ostream &os,
                    const FabricMemoryServiceRef &service);
void printFabricRef(llvm::raw_ostream &os,
                    const FabricPhysicalTraversalRef &traversal);

/// A projection and a refinement print exactly their underlying reference.
template <FabricInventoryKind Inventory>
void printFabricRef(llvm::raw_ostream &os,
                    const FabricOwnerProjection<Inventory> &owner) {
  printFabricRef(os, owner.catalog());
}
template <FabricRefinementKind Refinement, typename Underlying>
void printFabricRef(llvm::raw_ostream &os,
                    const FabricRefinedRef<Refinement, Underlying> &ref) {
  printFabricRef(os, ref.underlying());
}

struct FabricPrintVisitor {
  llvm::raw_ostream &os;
  bool started = false;

  void separate() {
    if (started)
      os << ", ";
    started = true;
  }
  template <typename Enum> void tag(const Enum &value) {
    separate();
    os << fabricRefKeyword(value);
  }
  void ordinal(const FabricOrdinal &value) {
    separate();
    os << value;
  }
  template <typename Ref> void ref(const Ref &value) {
    separate();
    printFabricRef(os, value);
  }
};

template <FabricEntityKind Kind>
void printFabricRef(llvm::raw_ostream &os,
                    const FabricTypedEntityRef<Kind> &ref) {
  os << fabricRefKeyword(Kind) << '<' << ref.id() << '>';
}

template <typename Ref>
void printFabricRef(llvm::raw_ostream &os, const Ref &ref) {
  os << Ref::familyKeyword << '<';
  FabricPrintVisitor visitor{os};
  Ref::visitFields(ref, visitor);
  os << '>';
}

template <typename Ref> std::string printFabricRef(const Ref &ref) {
  std::string text;
  llvm::raw_string_ostream os(text);
  printFabricRef(os, ref);
  return os.str();
}

//===---------------------------------------------------------------------===//
// Parsing
//===---------------------------------------------------------------------===//

template <FabricEntityKind Kind>
llvm::Error parseFabricRefInto(FabricRefScanner &scanner,
                               FabricTypedEntityRef<Kind> &ref);
template <typename Ref>
llvm::Error parseFabricRefInto(FabricRefScanner &scanner, Ref &ref);

llvm::Error parseFabricRefInto(FabricRefScanner &scanner,
                               FabricTransportEndpointOwnerRef &owner);
llvm::Error parseFabricRefInto(FabricRefScanner &scanner,
                               FabricMemoryEndpointOwnerRef &owner);
llvm::Error parseFabricRefInto(FabricRefScanner &scanner,
                               FabricInventoryOwnerRef &owner);
llvm::Error parseFabricRefInto(FabricRefScanner &scanner,
                               FabricMemoryServiceRef &service);
llvm::Error parseFabricRefInto(FabricRefScanner &scanner,
                               FabricPhysicalTraversalRef &traversal);

/// Consumes one closed keyword of `Enum` or reports the offending spelling.
template <typename Enum>
llvm::Error parseFabricKeyword(FabricRefScanner &scanner, Enum &value,
                               std::uint32_t bound, llvm::StringRef what) {
  llvm::StringRef keyword = scanner.peekKeyword();
  for (std::uint32_t index = 0; index < bound; ++index) {
    Enum candidate = static_cast<Enum>(index);
    if (keyword == fabricRefKeyword(candidate)) {
      scanner.take(keyword.size());
      value = candidate;
      return llvm::Error::success();
    }
  }
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            llvm::Twine("expected a canonical ") + what +
                                " keyword before '" + scanner.rest() + "'");
}

template <FabricInventoryKind Inventory>
llvm::Error parseFabricRefInto(FabricRefScanner &scanner,
                               FabricOwnerProjection<Inventory> &owner) {
  FabricInventoryOwnerRef catalog;
  if (llvm::Error error = parseFabricRefInto(scanner, catalog))
    return error;
  owner = FabricOwnerProjection<Inventory>(std::move(catalog));
  return llvm::Error::success();
}
template <FabricRefinementKind Refinement, typename Underlying>
llvm::Error parseFabricRefInto(FabricRefScanner &scanner,
                               FabricRefinedRef<Refinement, Underlying> &ref) {
  Underlying underlying;
  if (llvm::Error error = parseFabricRefInto(scanner, underlying))
    return error;
  ref = FabricRefinedRef<Refinement, Underlying>(std::move(underlying));
  return llvm::Error::success();
}

struct FabricParseVisitor {
  FabricRefScanner &scanner;
  bool started = false;
  llvm::Error error = llvm::Error::success();

  void separate() {
    if (error)
      return;
    if (started)
      error = scanner.expect(", ");
    started = true;
  }
  template <typename Enum> void tag(Enum &value) {
    separate();
    if (error)
      return;
    error = parseFabricKeyword(scanner, value, fabricClosedBound(value),
                               fabricClosedName(value));
  }
  void ordinal(FabricOrdinal &value) {
    separate();
    if (error)
      return;
    llvm::Expected<std::uint64_t> parsed = scanner.unsignedField();
    if (!parsed)
      error = parsed.takeError();
    else
      value = *parsed;
  }
  template <typename Ref> void ref(Ref &value) {
    separate();
    if (error)
      return;
    error = parseFabricRefInto(scanner, value);
  }
};

template <FabricEntityKind Kind>
llvm::Error parseFabricRefInto(FabricRefScanner &scanner,
                               FabricTypedEntityRef<Kind> &ref) {
  if (llvm::Error error = fabricExpectFamily(scanner, fabricRefKeyword(Kind)))
    return error;
  if (llvm::Error error = scanner.expect("<"))
    return error;
  llvm::Expected<std::uint64_t> id = scanner.unsignedField();
  if (!id)
    return id.takeError();
  if (llvm::Error error = scanner.expect(">"))
    return error;
  ref = FabricTypedEntityRef<Kind>(*id);
  return llvm::Error::success();
}

template <typename Ref>
llvm::Error parseFabricRefInto(FabricRefScanner &scanner, Ref &ref) {
  if (llvm::Error error = fabricExpectFamily(scanner, Ref::familyKeyword))
    return error;
  if (llvm::Error error = scanner.expect("<"))
    return error;
  FabricParseVisitor visitor{scanner};
  Ref::visitFields(ref, visitor);
  if (visitor.error)
    return std::move(visitor.error);
  return scanner.expect(">");
}

/// Parses one complete reference of the statically required family.
template <typename Ref>
llvm::Expected<Ref> parseFabricRef(llvm::StringRef text) {
  FabricRefScanner scanner(text);
  Ref ref;
  if (llvm::Error error = parseFabricRefInto(scanner, ref))
    return std::move(error);
  if (llvm::Error error = scanner.finish())
    return std::move(error);
  return ref;
}

} // namespace fabric
} // namespace loom

#endif // LOOM_FABRIC_IDENTITY_FABRICREFTEXT_H

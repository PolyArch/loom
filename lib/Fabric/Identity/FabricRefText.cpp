#include "Fabric/Identity/FabricRefText.h"

#include "llvm/ADT/StringExtras.h"

#include <limits>

using namespace loom;
using namespace loom::fabric;

namespace {

bool isKeywordByte(char byte) {
  return llvm::isAlnum(byte) || byte == '_' || byte == '.';
}

/// Retired spellings named by the catalog.
bool isDeprecatedFamily(llvm::StringRef keyword) {
#define LOOM_FABRIC_DEPRECATED_REF(Keyword)                                    \
  if (keyword == Keyword)                                                      \
    return true;
#include "Fabric/Identity/FabricRefs.def"
  return false;
}

/// Escapes that would reintroduce an untyped identity: symbols, attribute
/// aliases, quoted paths, SSA values, printer positions, and path or property
/// continuations after a complete reference.
bool isGenericEscape(llvm::StringRef rest) {
  if (rest.empty())
    return false;
  if (rest.starts_with("loc("))
    return true;
  switch (rest.front()) {
  case '@':
  case '#':
  case '"':
  case '%':
  case '.':
  case '[':
    return true;
  default:
    return false;
  }
}

/// True when `keyword` names a family of the closed entity catalog.
bool isEntityFamily(llvm::StringRef keyword) {
#define LOOM_FABRIC_ENTITY(Name, Keyword)                                      \
  if (keyword == Keyword)                                                      \
    return true;
#include "Fabric/Identity/FabricRefs.def"
  return false;
}

bool isTransportOwnerFamily(llvm::StringRef keyword) {
#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type)                       \
  if (keyword == Type::familyKeyword)                                          \
    return true;
#include "Fabric/Identity/FabricRefs.def"
  return false;
}

bool isMemoryOwnerFamily(llvm::StringRef keyword) {
#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type)                          \
  if (keyword == Type::familyKeyword)                                          \
    return true;
#include "Fabric/Identity/FabricRefs.def"
  return false;
}

} // namespace

/// An owner of the opposite plane is plane misuse; an owner of neither plane
/// is an invalid owner family. Both stay distinct from ordinary syntax noise.
static llvm::Error fabricOwnerFamilyError(llvm::StringRef keyword,
                                          llvm::StringRef rest,
                                          bool isTransportPlane) {
  const bool inOtherPlane = isTransportPlane ? isMemoryOwnerFamily(keyword)
                                             : isTransportOwnerFamily(keyword);
  if (inOtherPlane)
    return makeFabricRefError(FabricRefErrorKind::PlaneMisuse,
                              llvm::Twine("'") + keyword +
                                  "' owns the other endpoint plane");
  if (isDeprecatedFamily(keyword) || isGenericEscape(rest))
    return fabricRefTextError("an endpoint owner", rest);
  return makeFabricRefError(FabricRefErrorKind::InvalidOwnerFamily,
                            llvm::Twine("'") + keyword +
                                "' exposes no endpoint inventory");
}

llvm::Error loom::fabric::fabricRefTextError(const llvm::Twine &context,
                                             llvm::StringRef rest) {
  const llvm::StringRef trimmed = rest.ltrim(' ');
  if (isGenericEscape(trimmed) ||
      isDeprecatedFamily(rest.take_while(isKeywordByte)))
    return makeFabricRefError(FabricRefErrorKind::DeprecatedAlias,
                              llvm::Twine("deprecated or generic reference "
                                          "escape at '") +
                                  rest + "'");
  return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                            llvm::Twine("expected ") + context + " at '" +
                                rest + "'");
}

llvm::Error FabricRefScanner::expect(llvm::StringRef literal) {
  if (!rest_.starts_with(literal))
    return fabricRefTextError(llvm::Twine("'") + literal + "'", rest_);
  rest_ = rest_.drop_front(literal.size());
  return llvm::Error::success();
}

llvm::StringRef FabricRefScanner::peekKeyword() const {
  return rest_.take_while(isKeywordByte);
}

llvm::Expected<std::uint64_t> FabricRefScanner::unsignedField() {
  llvm::StringRef digits = rest_.take_while(llvm::isDigit);
  if (digits.empty())
    return fabricRefTextError("an unsigned decimal field", rest_);
  // A canonical field has no sign, radix prefix, or leading zero.
  if (digits.size() > 1 && digits.front() == '0')
    return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                              llvm::Twine("noncanonical numeric alias '") +
                                  digits + "'");
  std::uint64_t value = 0;
  for (char digit : digits) {
    const std::uint64_t next = static_cast<std::uint64_t>(digit - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - next) / 10)
      return makeFabricRefError(FabricRefErrorKind::MalformedSyntax,
                                llvm::Twine("unsigned 64-bit overflow in '") +
                                    digits + "'");
    value = value * 10 + next;
  }
  rest_ = rest_.drop_front(digits.size());
  return value;
}

llvm::Error FabricRefScanner::finish() const {
  if (rest_.empty())
    return llvm::Error::success();
  return fabricRefTextError("end of reference", rest_);
}

llvm::Error loom::fabric::fabricExpectFamily(FabricRefScanner &scanner,
                                             llvm::StringRef family) {
  const llvm::StringRef keyword = scanner.peekKeyword();
  if (keyword != family) {
    // Naming another entity of the closed catalog is a kind failure, not a
    // spelling failure.
    if (isEntityFamily(keyword) && isEntityFamily(family))
      return makeFabricRefError(FabricRefErrorKind::WrongEntityKind,
                                llvm::Twine("'") + keyword + "' is not '" +
                                    family + "'");
    return fabricRefTextError(llvm::Twine("reference family '") + family + "'",
                              scanner.rest());
  }
  scanner.take(keyword.size());
  return llvm::Error::success();
}

//===---------------------------------------------------------------------===//
// Closed unions
//
// A union is spelled as its selected payload's own canonical reference, so the
// constructor is recovered from the payload family rather than written twice.
//===---------------------------------------------------------------------===//

void loom::fabric::printFabricRef(
    llvm::raw_ostream &os, const FabricTransportEndpointOwnerRef &owner) {
  switch (owner.kind()) {
#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type)                       \
  case FabricTransportEndpointOwnerKind::Name:                                 \
    return printFabricRef(os, std::get<Type>(owner.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::printFabricRef(llvm::raw_ostream &os,
                                  const FabricMemoryEndpointOwnerRef &owner) {
  switch (owner.kind()) {
#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type)                          \
  case FabricMemoryEndpointOwnerKind::Name:                                    \
    return printFabricRef(os, std::get<Type>(owner.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::printFabricRef(llvm::raw_ostream &os,
                                  const FabricInventoryOwnerRef &owner) {
  switch (owner.kind()) {
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Type)                                \
  case FabricInventoryOwnerKind::Name:                                         \
    return printFabricRef(os, std::get<Type>(owner.payload));
#include "Fabric/Identity/FabricRefs.def"
  }
}

void loom::fabric::printFabricRef(llvm::raw_ostream &os,
                                  const FabricMemoryServiceRef &service) {
  os << FabricMemoryServiceRef::familyKeyword << '<'
     << fabricRefKeyword(service.kind()) << ", ";
  switch (service.kind()) {
#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Type)                        \
  case FabricMemoryServiceKind::Name:                                          \
    printFabricRef(os, std::get<Type>(service.payload));                       \
    break;
#include "Fabric/Identity/FabricRefs.def"
  }
  os << '>';
}

void loom::fabric::printFabricRef(llvm::raw_ostream &os,
                                  const FabricPhysicalTraversalRef &traversal) {
  os << FabricPhysicalTraversalRef::familyKeyword << '<'
     << fabricRefKeyword(traversal.kind());
  FabricPrintVisitor visitor{os, /*started=*/true};
  switch (traversal.kind()) {
#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Type)                             \
  case FabricPhysicalTraversalKind::Name:                                      \
    Type::visitFields(std::get<Type>(traversal.payload), visitor);             \
    break;
#include "Fabric/Identity/FabricRefs.def"
  }
  os << '>';
}

llvm::Error
loom::fabric::parseFabricRefInto(FabricRefScanner &scanner,
                                 FabricTransportEndpointOwnerRef &owner) {
  const llvm::StringRef keyword = scanner.peekKeyword();
#define LOOM_FABRIC_TRANSPORT_OWNER(Ordinal, Name, Type)                       \
  if (keyword == Type::familyKeyword)                                          \
    return parseFabricRefInto(scanner, owner.payload.emplace<Type>());
#include "Fabric/Identity/FabricRefs.def"
  return fabricOwnerFamilyError(keyword, scanner.rest(),
                                /*isTransportPlane=*/true);
}

llvm::Error
loom::fabric::parseFabricRefInto(FabricRefScanner &scanner,
                                 FabricMemoryEndpointOwnerRef &owner) {
  const llvm::StringRef keyword = scanner.peekKeyword();
#define LOOM_FABRIC_MEMORY_OWNER(Ordinal, Name, Type)                          \
  if (keyword == Type::familyKeyword)                                          \
    return parseFabricRefInto(scanner, owner.payload.emplace<Type>());
#include "Fabric/Identity/FabricRefs.def"
  return fabricOwnerFamilyError(keyword, scanner.rest(),
                                /*isTransportPlane=*/false);
}

llvm::Error loom::fabric::parseFabricRefInto(FabricRefScanner &scanner,
                                             FabricInventoryOwnerRef &owner) {
  const llvm::StringRef keyword = scanner.peekKeyword();
#define LOOM_FABRIC_INVENTORY_OWNER(Name, Type)                                \
  if (keyword == Type::familyKeyword)                                          \
    return parseFabricRefInto(scanner, owner.payload.emplace<Type>());
#include "Fabric/Identity/FabricRefs.def"
  if (isDeprecatedFamily(keyword) || isGenericEscape(scanner.rest()))
    return fabricRefTextError("an inventory owner", scanner.rest());
  return makeFabricRefError(FabricRefErrorKind::InvalidOwnerFamily,
                            llvm::Twine("'") + keyword +
                                "' is not an inventory owner constructor");
}

llvm::Error loom::fabric::parseFabricRefInto(FabricRefScanner &scanner,
                                             FabricMemoryServiceRef &service) {
  if (llvm::Error error =
          fabricExpectFamily(scanner, FabricMemoryServiceRef::familyKeyword))
    return error;
  if (llvm::Error error = scanner.expect("<"))
    return error;
  FabricMemoryServiceKind kind = FabricMemoryServiceKind();
  if (llvm::Error error = parseFabricKeyword(
          scanner, kind, fabricClosedBound(kind), fabricClosedName(kind)))
    return error;
  if (llvm::Error error = scanner.expect(", "))
    return error;
  switch (kind) {
#define LOOM_FABRIC_MEMORY_SERVICE(Name, Keyword, Type)                        \
  case FabricMemoryServiceKind::Name:                                          \
    if (llvm::Error error =                                                    \
            parseFabricRefInto(scanner, service.payload.emplace<Type>()))      \
      return error;                                                            \
    break;
#include "Fabric/Identity/FabricRefs.def"
  }
  return scanner.expect(">");
}

llvm::Error
loom::fabric::parseFabricRefInto(FabricRefScanner &scanner,
                                 FabricPhysicalTraversalRef &traversal) {
  if (llvm::Error error = fabricExpectFamily(
          scanner, FabricPhysicalTraversalRef::familyKeyword))
    return error;
  if (llvm::Error error = scanner.expect("<"))
    return error;
  FabricPhysicalTraversalKind kind = FabricPhysicalTraversalKind();
  if (llvm::Error error = parseFabricKeyword(
          scanner, kind, fabricClosedBound(kind), fabricClosedName(kind)))
    return error;
  FabricParseVisitor visitor{scanner, /*started=*/true};
  switch (kind) {
#define LOOM_FABRIC_TRAVERSAL(Name, Keyword, Type)                             \
  case FabricPhysicalTraversalKind::Name:                                      \
    Type::visitFields(traversal.payload.emplace<Type>(), visitor);             \
    break;
#include "Fabric/Identity/FabricRefs.def"
  }
  if (visitor.error)
    return std::move(visitor.error);
  return scanner.expect(">");
}

//===- ImplementationFamilyBehaviorKey.cpp -------------------------------===//
//
// Owns the canonical framing shared by finite operation behavior keys.
//
//===----------------------------------------------------------------------===//

#include "ImplementationFamilyBehaviorInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr char kDomain[] = "loom.fabric.operation-behavior-key\0";

llvm::Error reject(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

void appendU32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  bytes.push_back(static_cast<std::uint8_t>(value >> 24));
  bytes.push_back(static_cast<std::uint8_t>(value >> 16));
  bytes.push_back(static_cast<std::uint8_t>(value >> 8));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

void appendU64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 56; shift != 0; shift -= 8)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  bytes.push_back(static_cast<std::uint8_t>(value));
}

llvm::Error appendFrame(std::vector<std::uint8_t> &bytes,
                        llvm::ArrayRef<std::uint8_t> value) {
  if (value.size() > std::numeric_limits<std::uint32_t>::max())
    return reject("operation behavior key frame exceeds uint32");
  appendU32(bytes, static_cast<std::uint32_t>(value.size()));
  bytes.insert(bytes.end(), value.begin(), value.end());
  return llvm::Error::success();
}

llvm::Error appendFrame(std::vector<std::uint8_t> &bytes,
                        llvm::StringRef value) {
  return appendFrame(bytes, llvm::ArrayRef<std::uint8_t>(value.bytes_begin(),
                                                         value.bytes_end()));
}

} // namespace

llvm::Expected<::loom::CanonicalSemanticBytes>
fabric::detail::encodeImplementationFamilyBehaviorKey(
    ImplementationFamilyId family, llvm::StringRef role,
    llvm::ArrayRef<ImplementationFamilyBehaviorKeyComponent> components) {
  if (static_cast<std::uint32_t>(family) >= implementationFamilyCount())
    return reject("implementation behavior key names an unknown family");

  std::vector<std::uint8_t> bytes(std::begin(kDomain), std::end(kDomain) - 1);
  appendU32(bytes, 1);
  appendU32(bytes, 0);
  if (llvm::Error error =
          appendFrame(bytes, implementationFamilyKeyword(family)))
    return std::move(error);
  if (llvm::Error error = appendFrame(bytes, role))
    return std::move(error);

  for (const ImplementationFamilyBehaviorKeyComponent &component : components) {
    if (const auto *width = std::get_if<std::uint32_t>(&component)) {
      if (*width == 0)
        return reject("operation behavior key contains a zero width");
      appendU32(bytes, *width);
      continue;
    }
    if (const auto *predicate =
            std::get_if<::loom::CanonicalSemanticBytes>(&component)) {
      if (predicate->bytes().empty())
        return reject("operation behavior key contains an empty predicate");
      if (llvm::Error error = appendFrame(bytes, predicate->bytes()))
        return std::move(error);
      continue;
    }
    const auto &image =
        std::get<ImplementationFamilyBehaviorLaneImage>(component);
    if (image.ordinals.size() > std::numeric_limits<std::uint32_t>::max())
      return reject("operation behavior lane image exceeds uint32");
    for (auto [ordinal, port] : llvm::enumerate(image.ordinals)) {
      if (port >= image.bound)
        return reject("operation behavior lane image selects a missing port");
      if (llvm::is_contained(
              llvm::ArrayRef<std::uint64_t>(image.ordinals).take_front(ordinal),
              port))
        return reject("operation behavior lane image contains a duplicate");
    }
    appendU32(bytes, static_cast<std::uint32_t>(image.ordinals.size()));
    for (std::uint64_t port : image.ordinals)
      appendU64(bytes, port);
  }
  return ::loom::CanonicalSemanticBytes(std::move(bytes));
}

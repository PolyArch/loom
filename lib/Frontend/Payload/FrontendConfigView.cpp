#include "Frontend/Payload/FrontendConfigView.h"

#include "Common/ComponentViewDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom {
namespace {

/// Exact ASCII schema descriptor bytes of ResolvedFrontendConfigView 1.0. The
/// descriptor has no trailing zero byte, so it is spelled as an explicit byte
/// sequence rather than as a NUL-terminated literal.
constexpr llvm::StringRef schemaDescriptor = "loom.config.view.frontend.1.0";

llvm::ArrayRef<std::uint8_t> descriptorBytes() {
  return {reinterpret_cast<const std::uint8_t *>(schemaDescriptor.data()),
          schemaDescriptor.size()};
}

} // namespace

llvm::ArrayRef<std::uint8_t>
ResolvedFrontendConfigView::schemaDescriptorBytes() const {
  return descriptorBytes();
}

llvm::ArrayRef<std::uint8_t>
ResolvedFrontendConfigView::canonicalViewBytes() const {
  return {};
}

ComponentViewDigest ResolvedFrontendConfigView::digest() const {
  // The descriptor length is fixed and small, so the framed length can always
  // represent it and the Common digest cannot fail for this view.
  return llvm::cantFail(computeComponentViewDigest(schemaDescriptorBytes(),
                                                   canonicalViewBytes()));
}

ResolvedFrontendConfigView
projectResolvedFrontendConfigView(const ResolvedConfig &) {
  // Version 1.0 declares an empty field set. The parameter is deliberately
  // unnamed so no config field can reach the projection.
  return ResolvedFrontendConfigView();
}

llvm::Expected<ResolvedFrontendConfigView> adoptResolvedFrontendConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest) {
  if (schemaDescriptorBytes != descriptorBytes())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "frontend_config_view_descriptor_mismatch: the stored schema "
        "descriptor is not the exact loom.config.view.frontend.1.0 descriptor");
  if (!canonicalViewBytes.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "frontend_config_view_bytes_not_empty: the zero-field frontend view "
        "has empty canonical bytes");
  if (llvm::Error error = validateComponentViewDigest(
          schemaDescriptorBytes, canonicalViewBytes, digest))
    return std::move(error);
  return ResolvedFrontendConfigView();
}

} // namespace loom

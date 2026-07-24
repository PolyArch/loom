#ifndef LOOM_FRONTEND_PAYLOAD_FRONTENDCONFIGVIEW_H
#define LOOM_FRONTEND_PAYLOAD_FRONTENDCONFIGVIEW_H

#include "Common/ComponentViewDigest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace loom {

struct ResolvedConfig;

/// The typed `ResolvedFrontendConfigView` version 1.0 value.
///
/// Version 1.0 declares an explicitly empty field set: its canonical view bytes
/// are empty and its sole projector reads no ResolvedConfig field. That empty
/// set is a closed dependency declaration, not permission for Part 1 to reach
/// into the full ResolvedConfig. Consuming the first real frontend config field
/// changes this view schema version and this projector together.
///
/// Every 1.0 value is therefore the same value. A view can only be obtained by
/// projecting a ResolvedConfig or by adopting decoded fields that were checked
/// against this closed contract.
class ResolvedFrontendConfigView {
public:
  /// Exact ASCII schema descriptor bytes, without a trailing zero byte.
  llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes() const;

  /// The canonical view bytes, empty at version 1.0.
  llvm::ArrayRef<std::uint8_t> canonicalViewBytes() const;

  /// The Common component-view digest over exactly those two byte sequences.
  ComponentViewDigest digest() const;

private:
  ResolvedFrontendConfigView() = default;

  friend ResolvedFrontendConfigView
  projectResolvedFrontendConfigView(const ResolvedConfig &config);
  friend llvm::Expected<ResolvedFrontendConfigView>
  adoptResolvedFrontendConfigView(
      llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
      llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
      const ComponentViewDigest &digest);
};

/// The sole deterministic projector from an exact ResolvedConfig. Version 1.0
/// consumes no field, so no config value can reach the projected view.
ResolvedFrontendConfigView
projectResolvedFrontendConfigView(const ResolvedConfig &config);

/// Adopts decoded view fields after checking them against the closed 1.0
/// contract: the exact schema descriptor, empty canonical bytes, and the Common
/// digest of both. Disagreement is a typed error; nothing is repaired.
llvm::Expected<ResolvedFrontendConfigView> adoptResolvedFrontendConfigView(
    llvm::ArrayRef<std::uint8_t> schemaDescriptorBytes,
    llvm::ArrayRef<std::uint8_t> canonicalViewBytes,
    const ComponentViewDigest &digest);

} // namespace loom

#endif // LOOM_FRONTEND_PAYLOAD_FRONTENDCONFIGVIEW_H

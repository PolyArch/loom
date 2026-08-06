#ifndef LOOM_COMMON_PROVIDERFORM_H
#define LOOM_COMMON_PROVIDERFORM_H

#include <cstdint>

namespace loom {

/// The one closed provider form shared by the DSE provider boundaries. An
/// InProcess provider executes inside the invoking process; an
/// ExternalPrepareImport provider materializes one deterministic invocation
/// bundle and strictly imports it as two descriptor-owned calls.
enum class ProviderForm : std::uint32_t {
  InProcess = 0,
  ExternalPrepareImport = 1,
};

} // namespace loom

#endif // LOOM_COMMON_PROVIDERFORM_H

#ifndef LOOM_HARDWARE_CONFIGURATION_CONFIGURATIONDIAGNOSTICS_H
#define LOOM_HARDWARE_CONFIGURATION_CONFIGURATIONDIAGNOSTICS_H

#include "Hardware/Configuration/ConfigurationABI.h"
#include "Hardware/Configuration/PackedConfigurationABI.h"

namespace loom::hardware {

enum class ConfigurationABIImportVerificationDomain : std::uint8_t {
  SourceInvocation,
  IndependentReplay,
};

/// Emits the owner-typed packed ABI derivation payload through the Common
/// invocation diagnostic envelope when Summary diagnostics are enabled.
void emitPackedConfigurationABIDerivationStatistics(
    const PackedConfigurationABIDerivationStatistics &statistics);

/// Emits the owner-typed ABI finalization payload through the Common invocation
/// diagnostic envelope when Summary diagnostics are enabled.
void emitConfigurationABIConstructionStatistics(
    const ConfigurationABIConstructionStatistics &statistics);

void emitConfigurationABIImportSessionStatistics(
    ConfigurationABIImportVerificationDomain domain,
    const ConfigurationABIImportSessionStatistics &statistics);

} // namespace loom::hardware

#endif // LOOM_HARDWARE_CONFIGURATION_CONFIGURATIONDIAGNOSTICS_H

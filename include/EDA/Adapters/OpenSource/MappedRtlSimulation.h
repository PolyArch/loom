#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLSIMULATION_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLSIMULATION_H

#include "Simulator/SimulationArtifacts.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::eda::open_source {

inline constexpr llvm::StringLiteral mappedRtlResultSchema =
    "loom.mapped_rtl_result";
inline constexpr llvm::StringLiteral mappedRtlResultVersion = "1.0";
inline constexpr llvm::StringLiteral mappedRtlHarnessTop =
    "loom_mapped_rtl_testbench";
inline constexpr llvm::StringLiteral mappedRtlTestbenchPath =
    "drivers/testbench.sv";
inline constexpr llvm::StringLiteral mappedRtlVerilatorDriverPath =
    "drivers/verilator.args";
inline constexpr llvm::StringLiteral mappedRtlBridgedVerilatorDriverPath =
    "drivers/verilator-bridge.args";
inline constexpr llvm::StringLiteral mappedRtlBridgeEngineSourcePath =
    "drivers/loom-gem5-rtl-engine.cpp";
inline constexpr llvm::StringLiteral mappedRtlSimulatorExecutablePath =
    "work/verilator/simulation";
inline constexpr llvm::StringLiteral mappedRtlResultPath =
    "outputs/mapped-rtl-result.txt";

enum class MappedRtlTerminalStatus : std::uint8_t {
  Retired,
  StoppedByLimit,
};

struct MappedRtlValueObservation final {
  std::optional<llvm::APInt> token;
};

struct MappedRtlStreamObservation final {
  std::uint32_t tokenBitWidth = 0;
  std::vector<llvm::APInt> tokens;
  sim::StreamTermination termination = sim::StreamTermination::ClosedAfterLast;
};

struct MappedRtlMemoryObservation final {
  std::vector<sim::SemanticMemoryByte> bytes;
};

/// The provider-neutral output of one independently executed RTL harness.
/// The exact Evaluation Request supplies all semantic identities and lane
/// geometry; this value carries observations and reference-cycle coordinates
/// only.
struct MappedRtlSimulationResult final {
  MappedRtlTerminalStatus terminal = MappedRtlTerminalStatus::StoppedByLimit;
  std::uint64_t launchCycle = 0;
  std::optional<std::uint64_t> retirementCycle;
  std::uint64_t terminalCycle = 0;
  std::vector<MappedRtlValueObservation> valueResults;
  std::vector<MappedRtlStreamObservation> streamOutputs;
  std::vector<MappedRtlMemoryObservation> memories;
};

/// Canonical enum spellings shared by the host parser and generated HDL
/// harness. These are the only text codecs for their semantic domains.
llvm::StringRef
mappedRtlTerminalStatusSpelling(MappedRtlTerminalStatus status);
llvm::StringRef mappedRtlStreamTerminationSpelling(
    sim::StreamTermination termination);

/// Canonical host-side renderer and strict parser for the authored text
/// protocol written by generated HDL harnesses. Unknown states, widths,
/// ordinals, fields, or trailing tokens are rejected.
llvm::Expected<std::string>
renderMappedRtlSimulationResult(const MappedRtlSimulationResult &result);
llvm::Expected<MappedRtlSimulationResult>
parseMappedRtlSimulationResult(llvm::StringRef contents);

/// Registers the exact ExternalPrepareImport provider for the production
/// mapped-RTL Evaluation descriptor. Repeated registration is idempotent.
llvm::Error registerMappedRtlSimulationProvider();

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_MAPPEDRTLSIMULATION_H

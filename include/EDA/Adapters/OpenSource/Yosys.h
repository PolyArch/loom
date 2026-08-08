#ifndef LOOM_EDA_ADAPTERS_OPENSOURCE_YOSYS_H
#define LOOM_EDA_ADAPTERS_OPENSOURCE_YOSYS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace loom::eda::open_source {

/// One signal bit in a Yosys write_json document: either a net ordinal or
/// one constant bit.
struct YosysSignalBit final {
  std::variant<std::uint64_t, char> value;

  friend bool operator==(const YosysSignalBit &lhs, const YosysSignalBit &rhs) {
    return lhs.value == rhs.value;
  }
};

/// The exact geometry of one module port in a Yosys structure document.
struct YosysPortGeometry final {
  enum class Direction : std::uint8_t { Input = 0, Output = 1, Inout = 2 };

  Direction direction;
  /// Exact signal bits, always nonempty. Net ordinals inside are relative to
  /// one produced document, so geometry equality compares direction, width,
  /// offset, and flags only.
  std::vector<YosysSignalBit> bits;
  std::int64_t offset;
  bool upto;
  bool isSigned;

  friend bool operator==(const YosysPortGeometry &lhs,
                         const YosysPortGeometry &rhs) {
    return lhs.direction == rhs.direction && lhs.bits.size() == rhs.bits.size() &&
           lhs.offset == rhs.offset && lhs.upto == rhs.upto &&
           lhs.isSigned == rhs.isSigned;
  }
};

struct YosysCellFacts final {
  std::string type;
  std::map<std::string, YosysPortGeometry::Direction> portDirections;
  std::map<std::string, std::vector<YosysSignalBit>> connections;
};

struct YosysModuleFacts final {
  /// The module carries a blackbox or whitebox attribute.
  bool declaredBox = false;
  /// Residual unmapped process or memory sections.
  bool hasProcesses = false;
  bool hasMemories = false;
  std::map<std::string, YosysPortGeometry> ports;
  std::map<std::string, YosysCellFacts> cells;
};

/// The ephemeral typed facts view of one Yosys write_json document. It is
/// never persisted: the pinned Slang gate index remains the sole netlist
/// representation authority.
struct YosysStructureFacts final {
  std::map<std::string, YosysModuleFacts> modules;
};

/// Renders the byte-deterministic Yosys 0.67 synthesis driver. The script
/// consumes only the fixed bundle-relative paths inputs/design.sv and
/// inputs/library.lib, writes only outputs/..., and embeds the exact
/// portable top identifier. A top that is not a portable HDL identifier is
/// rejected rather than quoted.
llvm::Expected<std::string> renderYosysSynthesisDriver(llvm::StringRef topModule);

/// Renders the same driver for an exact ordered RTL payload closure and one
/// resolved standard-cell Liberty file. Each source remains an independent
/// compilation unit. RTL paths may use Yosys quoting; the Liberty path must be
/// one bare token because the downstream ABC script cannot preserve quoting.
llvm::Expected<std::string> renderYosysSynthesisDriver(
    llvm::StringRef topModule, llvm::ArrayRef<std::string> rtlSources,
    llvm::StringRef standardCellLiberty);

/// Parses one write_json document into the typed facts view. Malformed JSON,
/// wrong field types, unknown port directions, and invalid signal bits are
/// rejected here.
llvm::Expected<YosysStructureFacts>
parseYosysStructureFacts(llvm::StringRef contents);

/// Validates a synthesized structure: exact non-blackbox top, only declared
/// blackbox/whitebox cells besides it, no residual processes, memories, or
/// generic $ cells, declared cell types and connection ports, and exactly one
/// defined driver for every required top output. A zero-cell constant-only
/// design is potentially valid.
llvm::Error
validateYosysSynthesizedStructure(const YosysStructureFacts &structure,
                                  llvm::StringRef topModule);

/// The canonical top port names, directions, and widths must be identical
/// before and after synthesis. Provider-local range metadata is not a second
/// representation-fact authority.
llvm::Error
compareYosysTopPortGeometry(const YosysStructureFacts &preSynthesis,
                            const YosysStructureFacts &postSynthesis,
                            llvm::StringRef topModule);

} // namespace loom::eda::open_source

#endif // LOOM_EDA_ADAPTERS_OPENSOURCE_YOSYS_H

#ifndef LOOM_HARDWARE_RTL_RTLBLOCKCLOSURE_H
#define LOOM_HARDWARE_RTL_RTLBLOCKCLOSURE_H

#include "Common/BlobDigest.h"
#include "Hardware/RTL/RtlModuleGraph.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace loom::hardware::rtl {

/// The port names under which the generated hierarchy threads the
/// implementation clock and reset by name into every clocked definition.
struct RtlDomainPortNames final {
  std::optional<std::string> clock;
  std::optional<std::string> reset;
};

struct RtlBlockClosureChild final {
  std::size_t member = 0;
  std::uint64_t multiplicity = 0;
};

/// One structurally distinct definition of a block closure. Every projection
/// definition whose content identity is equal is an alias of this member.
struct RtlBlockClosureMember final {
  BlobDigest identity;
  /// Projection ordinals sharing this identity, sorted; the first is the
  /// representative whose emission is rendered.
  std::vector<std::size_t> definitions;
  /// Direct children by member ordinal with merged multiplicity, sorted by
  /// member ordinal.
  std::vector<RtlBlockClosureChild> children;
  /// Instances of this member inside the root, weighted through every parent
  /// multiplicity; the root itself counts one.
  std::uint64_t instanceCount = 0;
};

/// The exact dependency closure of one definition as content-addressed,
/// occurrence-free derived evidence. A member's identity is the SHA-256 of
/// the exact preamble, its interface, parameters, direct child identities with
/// multiplicity, and
/// its emission bytes with its own emitted name and every concrete child's
/// emitted name replaced by content-derived block names. The identity does not
/// depend on the occurrence subject, the emitted occurrence names, or the top
/// module; it is validated against the RTL payload bytes and never authored.
struct RtlBlockClosure final {
  /// Members in dependency order: every child precedes its parents and the
  /// root is the last member.
  std::vector<RtlBlockClosureMember> members;
  /// The root definition's input clock and reset ports, present only when the
  /// root carries the implementation's domain port by exact name and shape.
  std::optional<std::string> clockPort;
  std::optional<std::string> resetPort;

  std::size_t root() const { return members.size() - 1; }
  const BlobDigest &identity() const { return members.back().identity; }
};

/// The occurrence-free module name of one content identity.
std::string rtlBlockName(const BlobDigest &identity);

/// Derives the complete dependency closure of rootModule. Concrete children
/// remain present; external definitions retain their unresolved symbol
/// contracts.
llvm::Expected<RtlBlockClosure>
deriveRtlBlockClosure(const RtlModuleGraphProjection &graph,
                      const RtlModuleGraphSourceBinding &source,
                      std::size_t rootModule,
                      const RtlDomainPortNames &domainPorts);

/// A mechanically reframed block source and its normalized definition DAG.
/// The source contains the exact preamble and every concrete dependency. The
/// graph uses content names and has no occurrence ordinals or source aliases.
struct RtlBlockSourceProjection final {
  std::string source;
  RtlModuleGraphProjection graph;
};

llvm::Expected<RtlBlockSourceProjection>
projectRtlBlockClosureSource(const RtlBlockClosure &closure,
                             const RtlModuleGraphProjection &graph,
                             const RtlModuleGraphSourceBinding &source);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_RTLBLOCKCLOSURE_H

#ifndef LOOM_HARDWARE_RTL_RTLMODULEGRAPH_H
#define LOOM_HARDWARE_RTL_RTLMODULEGRAPH_H

#include "Common/BlobDigest.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace loom::hardware::rtl {

/// Closed definition kinds admitted by the generated portable RTL hierarchy.
/// Concrete definitions own emitted bytes; external definitions own only a
/// symbol contract and remain in the unresolved implementation closure.
enum class RtlModuleDefinitionKind : std::uint8_t {
  Concrete = 0,
  External = 1,
};

enum class RtlModulePortDirection : std::uint8_t {
  Input = 0,
  Output = 1,
  Inout = 2,
};

struct RtlModulePortProjection final {
  std::string name;
  std::string type;
  std::string attributes;
  RtlModulePortDirection direction = RtlModulePortDirection::Input;

  friend bool operator==(const RtlModulePortProjection &lhs,
                         const RtlModulePortProjection &rhs) {
    return lhs.name == rhs.name && lhs.type == rhs.type &&
           lhs.attributes == rhs.attributes && lhs.direction == rhs.direction;
  }
};

/// One direct definition dependency. Multiplicity counts instance operations
/// in the parent definition, never expanded top-rooted occurrence paths.
struct RtlModuleDependency final {
  std::size_t targetModule = 0;
  std::uint64_t multiplicity = 0;

  friend bool operator==(RtlModuleDependency lhs, RtlModuleDependency rhs) {
    return lhs.targetModule == rhs.targetModule &&
           lhs.multiplicity == rhs.multiplicity;
  }
};

/// Exact content bytes emitted for one CIRCT output-file partition. Framing
/// bytes in the monolithic source are outside this range and are accounted by
/// RtlModuleGraphProjection::framingByteCount.
struct RtlModuleEmissionRange final {
  std::uint64_t offset = 0;
  std::uint64_t byteCount = 0;
  BlobDigest digest;

  RtlModuleEmissionRange(std::uint64_t offset, std::uint64_t byteCount,
                         BlobDigest digest)
      : offset(offset), byteCount(byteCount), digest(std::move(digest)) {}

  friend bool operator==(const RtlModuleEmissionRange &lhs,
                         const RtlModuleEmissionRange &rhs) {
    return lhs.offset == rhs.offset && lhs.byteCount == rhs.byteCount &&
           lhs.digest == rhs.digest;
  }
};

struct RtlModuleProjection final {
  std::string irSymbol;
  std::string emittedName;
  RtlModuleDefinitionKind kind = RtlModuleDefinitionKind::Concrete;
  bool reachable = false;
  std::vector<RtlModulePortProjection> ports;
  std::string parameters;
  std::vector<RtlModuleDependency> dependencies;
  std::optional<RtlModuleEmissionRange> emission;
};

/// Invocation-local projection of one post-lowering CIRCT/HW symbol graph.
/// The CIRCT module is the authority while alive. This value is a frozen cache
/// bound to the exact emitted source bytes and is never independently authored.
struct RtlModuleGraphProjection final {
  std::size_t topModule = 0;
  std::vector<RtlModuleProjection> modules;
  std::optional<RtlModuleEmissionRange> preamble;
  std::uint64_t framingByteCount = 0;
  std::uint64_t sourceByteCount = 0;
  std::optional<BlobDigest> sourceDigest;
};

/// Derives the complete canonical definition catalog and direct instance DAG
/// from CIRCT/HW IR. Reachability is rooted only at exactTopModule.
llvm::Expected<RtlModuleGraphProjection>
projectRtlModuleGraph(mlir::ModuleOp module, llvm::StringRef exactTopModule);

/// Assigns one deterministic CIRCT output file per concrete definition,
/// streams a monolithic SystemVerilog source, records exact file-content byte
/// ranges from CIRCT's own framing, and cold-compares the post-export graph.
llvm::Expected<RtlModuleGraphProjection>
exportFramedRtlModuleGraph(mlir::ModuleOp module,
                           const RtlModuleGraphProjection &before,
                           llvm::raw_ostream &output);

/// The exact per-definition byte views of one framed RTL source. A binding
/// exists only after the source identity, every recorded range digest, the
/// preamble, the framing byte count, and the complete coverage were validated
/// against the payload bytes, so no consumer rediscovers modules from text.
class RtlModuleGraphSourceBinding final {
public:
  llvm::StringRef source() const { return source_; }
  llvm::StringRef preamble() const { return preamble_; }
  /// Indexed by module ordinal; empty for external definitions.
  llvm::ArrayRef<llvm::StringRef> moduleBytes() const { return moduleBytes_; }

private:
  RtlModuleGraphSourceBinding(llvm::StringRef source, llvm::StringRef preamble,
                              std::vector<llvm::StringRef> moduleBytes)
      : source_(source), preamble_(preamble),
        moduleBytes_(std::move(moduleBytes)) {}
  llvm::StringRef source_;
  llvm::StringRef preamble_;
  std::vector<llvm::StringRef> moduleBytes_;

  friend llvm::Expected<RtlModuleGraphSourceBinding>
  bindRtlModuleGraphSource(const RtlModuleGraphProjection &, llvm::StringRef);
};

llvm::Expected<RtlModuleGraphSourceBinding>
bindRtlModuleGraphSource(const RtlModuleGraphProjection &graph,
                         llvm::StringRef source);

} // namespace loom::hardware::rtl

#endif // LOOM_HARDWARE_RTL_RTLMODULEGRAPH_H

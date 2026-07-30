#ifndef LOOM_FRONTEND_IR_STRUCTUREDPROGRAMARTIFACT_H
#define LOOM_FRONTEND_IR_STRUCTUREDPROGRAMARTIFACT_H

#include "Common/Artifact.h"
#include "Common/ArtifactStore.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace mlir {
class Block;
class MLIRContext;
class Operation;
class Region;
} // namespace mlir

namespace loom::frontend {

class StructuredProgramCandidate;
class StructuredProgramCandidateView;

inline constexpr ArtifactSchemaDescriptor structuredProgramArtifactSchema{
    "loom.structured_program", {1, 0}};

enum class StructuredEntityKind : std::uint32_t {
  Operation = 0,
  Region = 1,
  Block = 2,
  Value = 3,
};

struct StructuredEntityRef {
  ArtifactIdentity parent;
  StructuredEntityKind kind;
  std::uint64_t ordinal;

  friend bool operator==(const StructuredEntityRef &lhs,
                         const StructuredEntityRef &rhs) {
    return lhs.parent == rhs.parent && lhs.kind == rhs.kind &&
           lhs.ordinal == rhs.ordinal;
  }
  friend bool operator!=(const StructuredEntityRef &lhs,
                         const StructuredEntityRef &rhs) {
    return !(lhs == rhs);
  }
};

inline constexpr std::size_t structuredEntityRefWireSize =
    ArtifactIdentity::byteSize + sizeof(std::uint32_t) + sizeof(std::uint64_t);

std::vector<std::uint8_t>
encodeStructuredEntityRef(const StructuredEntityRef &reference);
llvm::Expected<StructuredEntityRef>
decodeStructuredEntityRef(llvm::ArrayRef<std::uint8_t> bytes);

llvm::Expected<StructuredProgramCandidateView>
buildStructuredProgramCandidateView(mlir::ModuleOp, const ArtifactIdentity &);

struct StructuredEntity {
  StructuredEntityRef reference;
  mlir::Operation *operation = nullptr;
  mlir::Region *region = nullptr;
  mlir::Block *block = nullptr;
  mlir::Value value;
};

/// The read-only owner projection for one exact immutable Structured Program.
/// Entity ordinals are derived from the schema-owned canonical relation graph;
/// native pointers are only local lookup accelerators.
class StructuredProgramCandidateView final {
public:
  const ArtifactIdentity &identity() const { return identity_; }
  llvm::ArrayRef<StructuredEntity> entities(StructuredEntityKind kind) const;
  llvm::Expected<StructuredEntity>
  resolve(const StructuredEntityRef &reference) const;

private:
  explicit StructuredProgramCandidateView(ArtifactIdentity identity)
      : identity_(std::move(identity)) {}

  ArtifactIdentity identity_;
  std::array<std::vector<StructuredEntity>, 4> entities_;

  friend class StructuredProgramCandidate;
  friend llvm::Expected<StructuredProgramCandidateView>
  buildStructuredProgramCandidateView(mlir::ModuleOp, const ArtifactIdentity &);
  friend llvm::Expected<StructuredProgramCandidate>
      finalizeStructuredProgram(mlir::ModuleOp);
};

/// A complete immutable S0/Sn snapshot. The family owns one canonical MLIR
/// bytecode writer and strict importer; it has no stage-state bit, direct
/// dependency table, analysis cache, or copied Fabric facts.
class StructuredProgramCandidate final {
public:
  StructuredProgramCandidate(const StructuredProgramCandidate &) = delete;
  StructuredProgramCandidate &
  operator=(const StructuredProgramCandidate &) = delete;
  StructuredProgramCandidate(StructuredProgramCandidate &&) = default;
  StructuredProgramCandidate &
  operator=(StructuredProgramCandidate &&) = default;

  const ArtifactIdentity &identity() const { return identity_; }
  const CanonicalSemanticBytes &canonicalBytes() const {
    return canonicalBytes_;
  }
  mlir::ModuleOp module() const { return *module_; }
  llvm::Expected<StructuredProgramCandidateView> view() const;

private:
  StructuredProgramCandidate(ArtifactIdentity identity,
                             CanonicalSemanticBytes canonicalBytes,
                             std::unique_ptr<mlir::MLIRContext> context,
                             mlir::OwningOpRef<mlir::ModuleOp> module,
                             StructuredProgramCandidateView view)
      : identity_(std::move(identity)),
        canonicalBytes_(std::move(canonicalBytes)),
        context_(std::move(context)), module_(std::move(module)),
        view_(std::move(view)) {}

  ArtifactIdentity identity_;
  CanonicalSemanticBytes canonicalBytes_;
  std::unique_ptr<mlir::MLIRContext> context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  StructuredProgramCandidateView view_;

  friend llvm::Expected<StructuredProgramCandidate>
      finalizeStructuredProgram(mlir::ModuleOp);
  friend llvm::Expected<StructuredProgramCandidate>
  importStructuredProgram(const ArtifactIdentity &,
                          const CanonicalSemanticBytes &);
  friend llvm::Expected<StructuredProgramCandidate>
  importStructuredProgram(const ArtifactRootReference &, const ArtifactStore &);
};

/// Finalizes a private clone of one complete mixed-dialect S0/Sn module.
/// Source locations and consumed hints do not affect identity; ordered program
/// semantics, ABI facts, and selected structured decisions do.
llvm::Expected<StructuredProgramCandidate>
finalizeStructuredProgram(mlir::ModuleOp source);

/// Strictly imports one exact family payload. The supplied identity must match
/// the Common identity of the canonical semantic bytes, and re-encoding must
/// reproduce those bytes exactly.
llvm::Expected<StructuredProgramCandidate>
importStructuredProgram(const ArtifactIdentity &identity,
                        const CanonicalSemanticBytes &canonicalBytes);

llvm::Expected<ArtifactRootReference>
publishStructuredProgram(const StructuredProgramCandidate &candidate,
                         const ArtifactStore &store);

llvm::Expected<StructuredProgramCandidate>
importStructuredProgram(const ArtifactRootReference &reference,
                        const ArtifactStore &store);

} // namespace loom::frontend

#endif // LOOM_FRONTEND_IR_STRUCTUREDPROGRAMARTIFACT_H

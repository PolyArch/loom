#ifndef FABRIC_TECH_TEMPLATELIBRARY_H
#define FABRIC_TECH_TEMPLATELIBRARY_H

#include "Dataflow/IR/DataflowOps.h"
#include "Fabric/IR/FabricOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace fabric {

// One (FU, configuration) entry in the TemplateLibrary.
//
// Lifetime: the wrapping `dataflow.subgraph` lives inside the library's
// owned ModuleOp. The pointer is stable for the library's lifetime and may
// be read concurrently from multiple threads. `swConfigsByOp` keys point
// into the originating FU body and are owned by the user's IR; the values
// (DictionaryAttr) are SSA-immutable MLIR attributes safe to share.
struct FuTemplate {
  // Index into TemplateLibrary::templates(); also used as a stable
  // deterministic ordering key.
  unsigned id = 0;
  // The originating FU op in the user's module.
  FuOp fu;
  // The canonical dataflow.subgraph for this configuration, hosted inside
  // the library's owned module.
  ::dataflow::SubgraphOp subgraph;
  // Number of body ops (excluding terminator). Equal to the candidate's
  // "size" used by the cost model.
  unsigned bodyOpCount = 0;
  // Top op kind name (the root op for matching). For a subgraph with a
  // single non-terminator op this is just that op's name; for multi-op
  // subgraphs it is the op feeding the yield.
  std::string rootOpName;
  // Configuration description (e.g. "op#0{op_sel=arith.muli}; mux#0{sel=0,...}").
  std::string configDescription;
  // The sw_configs dictionary to write back to each configurable op in the
  // originating `fu`'s body to realize this configuration. Keys point at
  // ops inside `fu` (not the library copy).
  ::llvm::DenseMap<::mlir::Operation *, ::mlir::DictionaryAttr> swConfigsByOp;
};

// Immutable, thread-safe library of FU templates.
//
// Building the library runs the existing SubgraphEnumerator against each
// FU exactly once into a single owned ModuleOp. After construction the
// library is read-only and its `templates()` view is safe to scan from
// multiple threads in parallel.
class TemplateLibrary {
public:
  TemplateLibrary(const TemplateLibrary &) = delete;
  TemplateLibrary &operator=(const TemplateLibrary &) = delete;
  TemplateLibrary(TemplateLibrary &&) = default;
  TemplateLibrary &operator=(TemplateLibrary &&) = default;
  ~TemplateLibrary();

  // Build a library covering every FuOp in `fus`. `ctx` is borrowed; the
  // library does not extend its lifetime.
  static std::unique_ptr<TemplateLibrary>
  build(::mlir::MLIRContext *ctx, ::llvm::ArrayRef<FuOp> fus);

  // Templates indexed by stable id. Iteration order matches the order
  // returned here and is deterministic across runs given the same input.
  ::llvm::ArrayRef<FuTemplate> templates() const { return entries; }

  // Index of templates whose root op kind name matches a given key. The
  // partitioner uses this to short-circuit candidate enumeration: only
  // templates whose root op kind matches the user op are tried.
  ::llvm::ArrayRef<unsigned> templatesByRootOp(::llvm::StringRef rootName) const;

  // Borrowed module (debugging / dumps only).
  ::mlir::ModuleOp module() const { return moduleRef.get(); }

private:
  TemplateLibrary() = default;

  ::mlir::OwningOpRef<::mlir::ModuleOp> moduleRef;
  ::llvm::SmallVector<FuTemplate> entries;
  ::llvm::StringMap<::llvm::SmallVector<unsigned>> rootIndex;
};

} // namespace fabric

#endif // FABRIC_TECH_TEMPLATELIBRARY_H

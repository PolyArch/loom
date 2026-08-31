#include "Mapping/Artifact/MappingConstraintSetMigration.h"

#include "Common/ArtifactFinalizer.h"
#include "Mapping/Artifact/SystemMappingConstraintSet.h"
#include "Mapping/IR/MappingDialect.h"
#include "Mapping/IR/MappingOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/Twine.h"

#include <memory>
#include <string>
#include <system_error>
#include <utility>

using namespace mlir;

namespace loom::mapping {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "mapping_constraint_set_migration_invalid: " + message);
}

/// Reads one 1.0 payload by its own 1.0 reference and proves the reference
/// identity really is the identity of those bytes under the 1.0 descriptor.
/// Migration never trusts a caller-supplied identity.
llvm::Expected<CanonicalSemanticBytes>
readLegacyPayload(const ArtifactRootReference &reference,
                  const ArtifactStore &store) {
  if (reference.schemaIdentity != mappingConstraintSetSchemaV1_0.identity ||
      reference.schemaVersion != mappingConstraintSetSchemaV1_0.version)
    return invalid("migration requires an exact loom.mapping_constraints 1.0 "
                   "root reference");
  auto canonical = store.get(reference);
  if (!canonical)
    return canonical.takeError();
  if (finalizeArtifactIdentity(mappingConstraintSetSchemaV1_0, *canonical) !=
      reference.artifact)
    return invalid("loom.mapping_constraints 1.0 reference identity does not "
                   "match its canonical bytes");
  return std::move(*canonical);
}

struct ParsedLegacyRoot final {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
};

llvm::Expected<ParsedLegacyRoot>
parseLegacyRoot(const CanonicalSemanticBytes &canonicalBytes) {
  std::string wrapped = "module {\n";
  wrapped.append(reinterpret_cast<const char *>(canonicalBytes.bytes().data()),
                 canonicalBytes.bytes().size());
  wrapped += "}\n";

  DialectRegistry registry;
  registry.insert<::mapping::MappingDialect>();
  auto context =
      std::make_unique<MLIRContext>(registry, MLIRContext::Threading::DISABLED);
  auto module = parseSourceString<ModuleOp>(wrapped, context.get());
  if (!module)
    return invalid("loom.mapping_constraints 1.0 payload cannot be parsed");
  return ParsedLegacyRoot{std::move(context), std::move(module)};
}

/// A 1.0 payload cannot contain the 1.1-only clause kind. Finding one means the
/// reference was mislabelled, not that an upgrade is due.
llvm::Error rejectExtensionClauses(Operation *root) {
  llvm::Error result = llvm::Error::success();
  root->walk([&](::mapping::ConstraintRuntimeCounterexampleNoGoodOp) {
    result = llvm::joinErrors(
        std::move(result),
        invalid("a loom.mapping_constraints 1.0 payload holds a 1.1-only "
                "runtime-counterexample no-good clause"));
  });
  return result;
}

/// Migration accepts only what the superseded strict 1.0 importer would have
/// accepted. Cold-canonicalizing the parsed root with the same writer the 1.0
/// family used must reproduce the stored bytes exactly; a payload that only
/// becomes canonical by being rewritten was never a valid 1.0 artifact, and
/// normalizing it here would silently mint a valid 1.1 artifact from an
/// invalid input.
template <typename Writer>
llvm::Error requireStoredPayloadIsCanonical(
    Writer &&write, const CanonicalSemanticBytes &stored) {
  auto rewritten = write();
  if (!rewritten)
    return rewritten.takeError();
  if (!rewritten->bytes().equals(stored.bytes()))
    return invalid("loom.mapping_constraints 1.0 payload is not canonical "
                   "under its own family and cannot be migrated");
  return llvm::Error::success();
}

template <typename RootOp>
llvm::Expected<RootOp> singleRoot(ModuleOp module, llvm::StringRef spelling) {
  RootOp root;
  unsigned count = 0;
  for (Operation &operation : module.getBody()->without_terminator()) {
    auto candidate = dyn_cast<RootOp>(operation);
    if (!candidate)
      return invalid("loom.mapping_constraints 1.0 payload is not a " +
                     spelling + " root");
    root = candidate;
    ++count;
  }
  if (count != 1)
    return invalid("loom.mapping_constraints 1.0 payload must hold exactly one "
                   + spelling + " root");
  if (failed(verify(root)))
    return invalid("loom.mapping_constraints 1.0 " + spelling +
                   " root is structurally invalid");
  return root;
}

} // namespace

llvm::Expected<ArtifactRootReference>
migrateSpatialConstraintRootV1_0ToV1_1(const ArtifactRootReference &reference,
                                       const ArtifactStore &store) {
  auto canonical = readLegacyPayload(reference, store);
  if (!canonical)
    return canonical.takeError();
  auto parsed = parseLegacyRoot(*canonical);
  if (!parsed)
    return parsed.takeError();
  auto root = singleRoot<::mapping::ConstraintsSpatialOp>(*parsed->module,
                                                          "Spatial");
  if (!root)
    return root.takeError();
  if (llvm::Error error = rejectExtensionClauses(*root))
    return std::move(error);
  if (llvm::Error error = requireStoredPayloadIsCanonical(
          [&] { return writeCanonicalSpatialConstraintAssembly(*root); },
          *canonical))
    return std::move(error);
  // Re-finalization is the publication authority: it re-canonicalizes,
  // strictly re-imports through the ordinary 1.1 path, and republishes.
  auto finalized = finalizeSpatialMappingConstraintSet(*root, store);
  if (!finalized)
    return finalized.takeError();
  return finalized->reference();
}

llvm::Expected<ArtifactRootReference>
migrateSystemConstraintRootV1_0ToV1_1(const ArtifactRootReference &reference,
                                      const ArtifactStore &store) {
  auto canonical = readLegacyPayload(reference, store);
  if (!canonical)
    return canonical.takeError();
  auto parsed = parseLegacyRoot(*canonical);
  if (!parsed)
    return parsed.takeError();
  auto root =
      singleRoot<::mapping::ConstraintsSystemOp>(*parsed->module, "System");
  if (!root)
    return root.takeError();
  if (llvm::Error error = rejectExtensionClauses(*root))
    return std::move(error);
  if (llvm::Error error = requireStoredPayloadIsCanonical(
          [&] { return writeCanonicalSystemConstraintAssembly(*root); },
          *canonical))
    return std::move(error);
  auto finalized = finalizeSystemMappingConstraintSet(*root, store);
  if (!finalized)
    return finalized.takeError();
  return finalized->reference();
}

} // namespace loom::mapping

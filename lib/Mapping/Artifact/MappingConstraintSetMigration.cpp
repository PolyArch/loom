#include "Mapping/Artifact/MappingConstraintSetMigration.h"

#include "Common/ArtifactFinalizer.h"
#include "Common/ArtifactText.h"
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
#include "llvm/ADT/STLExtras.h"

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

/// Reads one legacy payload by its own exact reference and proves the reference
/// identity really is the identity of those bytes under that descriptor.
/// Migration never trusts a caller-supplied identity.
llvm::Expected<CanonicalSemanticBytes>
readLegacyPayload(const ArtifactRootReference &reference,
                  const ArtifactSchemaDescriptor &schema,
                  const ArtifactStore &store) {
  if (reference.schemaIdentity != schema.identity ||
      reference.schemaVersion != schema.version)
    return invalid("migration input has the wrong exact " +
                   formatSchemaVersion(schema.version) + " root reference");
  auto canonical = store.get(reference);
  if (!canonical)
    return canonical.takeError();
  if (finalizeArtifactIdentity(schema, *canonical) != reference.artifact)
    return invalid("loom.mapping_constraints reference identity does not "
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
    return invalid("loom.mapping_constraints legacy payload cannot be parsed");
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

llvm::Error rejectPhysicalTagLiterals(Operation *root) {
  llvm::Error result = llvm::Error::success();
  root->walk([&](::mapping::ConstraintRuntimeCounterexampleNoGoodOp clause) {
    if (llvm::any_of(clause.getLiterals(),
                     [](Attribute literal) {
                       return isa<::mapping::NetTagEqualsAttr>(literal);
                     }))
      result = llvm::joinErrors(
          std::move(result),
          invalid("a loom.mapping_constraints 1.1 payload holds a 1.2-only "
                  "Physical Tag no-good literal"));
  });
  return result;
}

llvm::Error rejectSpatialMappingIdentityLiterals(Operation *root) {
  llvm::Error result = llvm::Error::success();
  root->walk([&](::mapping::ConstraintRuntimeCounterexampleNoGoodOp clause) {
    if (llvm::any_of(clause.getLiterals(), [](Attribute literal) {
          return isa<::mapping::SpatialMappingIdentityEqualsAttr>(literal);
        }))
      result = llvm::joinErrors(
          std::move(result),
          invalid("a loom.mapping_constraints 1.2 payload holds a 1.3-only "
                  "SpatialMapping identity no-good literal"));
  });
  return result;
}

llvm::Expected<ArtifactRootReference>
publishUnderSchema(const CanonicalSemanticBytes &canonical,
                   const ArtifactSchemaDescriptor &schema,
                   const ArtifactStore &store) {
  auto identity = store.put(schema, canonical);
  if (!identity)
    return identity.takeError();
  return ArtifactRootReference{schema.identity.str(), schema.version,
                               std::move(*identity)};
}

/// Migration accepts only what the superseded strict importer would have
/// accepted. Cold-canonicalizing the parsed root with the same family writer
/// must reproduce the stored bytes exactly; a payload that only becomes
/// canonical by being rewritten was never a valid artifact under its claimed
/// descriptor, and normalizing it here would silently mint a valid successor
/// from invalid input.
template <typename Writer>
llvm::Error requireStoredPayloadIsCanonical(
    Writer &&write, const CanonicalSemanticBytes &stored) {
  auto rewritten = write();
  if (!rewritten)
    return rewritten.takeError();
  if (!rewritten->bytes().equals(stored.bytes()))
    return invalid("loom.mapping_constraints legacy payload is not canonical "
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
      return invalid("loom.mapping_constraints legacy payload is not a " +
                     spelling + " root");
    root = candidate;
    ++count;
  }
  if (count != 1)
    return invalid("loom.mapping_constraints legacy payload must hold exactly one "
                   + spelling + " root");
  if (failed(verify(root)))
    return invalid("loom.mapping_constraints legacy " + spelling +
                   " root is structurally invalid");
  return root;
}

} // namespace

llvm::Expected<ArtifactRootReference>
migrateSpatialConstraintRootV1_0ToV1_1(const ArtifactRootReference &reference,
                                       const ArtifactStore &store) {
  auto canonical =
      readLegacyPayload(reference, mappingConstraintSetSchemaV1_0, store);
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
  return publishUnderSchema(*canonical, mappingConstraintSetSchemaV1_1, store);
}

llvm::Expected<ArtifactRootReference>
migrateSystemConstraintRootV1_0ToV1_1(const ArtifactRootReference &reference,
                                      const ArtifactStore &store) {
  auto canonical =
      readLegacyPayload(reference, mappingConstraintSetSchemaV1_0, store);
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
  return publishUnderSchema(*canonical, mappingConstraintSetSchemaV1_1, store);
}

llvm::Expected<ArtifactRootReference>
migrateSpatialConstraintRootV1_1ToV1_2(const ArtifactRootReference &reference,
                                       const ArtifactStore &store) {
  auto canonical =
      readLegacyPayload(reference, mappingConstraintSetSchemaV1_1, store);
  if (!canonical)
    return canonical.takeError();
  auto parsed = parseLegacyRoot(*canonical);
  if (!parsed)
    return parsed.takeError();
  auto root = singleRoot<::mapping::ConstraintsSpatialOp>(*parsed->module,
                                                          "Spatial");
  if (!root)
    return root.takeError();
  if (llvm::Error error = rejectPhysicalTagLiterals(*root))
    return error;
  if (llvm::Error error = requireStoredPayloadIsCanonical(
          [&] { return writeCanonicalSpatialConstraintAssembly(*root); },
          *canonical))
    return error;
  return publishUnderSchema(*canonical, mappingConstraintSetSchemaV1_2,
                            store);
}

llvm::Expected<ArtifactRootReference>
migrateSystemConstraintRootV1_1ToV1_2(const ArtifactRootReference &reference,
                                      const ArtifactStore &store) {
  auto canonical =
      readLegacyPayload(reference, mappingConstraintSetSchemaV1_1, store);
  if (!canonical)
    return canonical.takeError();
  auto parsed = parseLegacyRoot(*canonical);
  if (!parsed)
    return parsed.takeError();
  auto root =
      singleRoot<::mapping::ConstraintsSystemOp>(*parsed->module, "System");
  if (!root)
    return root.takeError();
  if (llvm::Error error = requireStoredPayloadIsCanonical(
          [&] { return writeCanonicalSystemConstraintAssembly(*root); },
          *canonical))
    return error;
  return publishUnderSchema(*canonical, mappingConstraintSetSchemaV1_2,
                            store);
}

llvm::Expected<ArtifactRootReference>
migrateSpatialConstraintRootV1_2ToV1_3(const ArtifactRootReference &reference,
                                       const ArtifactStore &store) {
  auto canonical =
      readLegacyPayload(reference, mappingConstraintSetSchemaV1_2, store);
  if (!canonical)
    return canonical.takeError();
  auto parsed = parseLegacyRoot(*canonical);
  if (!parsed)
    return parsed.takeError();
  auto root = singleRoot<::mapping::ConstraintsSpatialOp>(*parsed->module,
                                                          "Spatial");
  if (!root)
    return root.takeError();
  if (llvm::Error error = rejectSpatialMappingIdentityLiterals(*root))
    return error;
  if (llvm::Error error = requireStoredPayloadIsCanonical(
          [&] { return writeCanonicalSpatialConstraintAssembly(*root); },
          *canonical))
    return error;
  auto finalized = finalizeSpatialMappingConstraintSet(*root, store);
  if (!finalized)
    return finalized.takeError();
  return finalized->reference();
}

llvm::Expected<ArtifactRootReference>
migrateSystemConstraintRootV1_2ToV1_3(const ArtifactRootReference &reference,
                                      const ArtifactStore &store) {
  auto canonical =
      readLegacyPayload(reference, mappingConstraintSetSchemaV1_2, store);
  if (!canonical)
    return canonical.takeError();
  auto parsed = parseLegacyRoot(*canonical);
  if (!parsed)
    return parsed.takeError();
  auto root =
      singleRoot<::mapping::ConstraintsSystemOp>(*parsed->module, "System");
  if (!root)
    return root.takeError();
  if (llvm::Error error = requireStoredPayloadIsCanonical(
          [&] { return writeCanonicalSystemConstraintAssembly(*root); },
          *canonical))
    return error;
  auto finalized = finalizeSystemMappingConstraintSet(*root, store);
  if (!finalized)
    return finalized.takeError();
  return finalized->reference();
}

llvm::Expected<ArtifactRootReference>
migrateSpatialConstraintRootV1_0ToV1_2(const ArtifactRootReference &reference,
                                       const ArtifactStore &store) {
  auto intermediate = migrateSpatialConstraintRootV1_0ToV1_1(reference, store);
  if (!intermediate)
    return intermediate.takeError();
  return migrateSpatialConstraintRootV1_1ToV1_2(*intermediate, store);
}

llvm::Expected<ArtifactRootReference>
migrateSystemConstraintRootV1_0ToV1_2(const ArtifactRootReference &reference,
                                      const ArtifactStore &store) {
  auto intermediate = migrateSystemConstraintRootV1_0ToV1_1(reference, store);
  if (!intermediate)
    return intermediate.takeError();
  return migrateSystemConstraintRootV1_1ToV1_2(*intermediate, store);
}

llvm::Expected<ArtifactRootReference>
migrateSpatialConstraintRootV1_0ToV1_3(const ArtifactRootReference &reference,
                                       const ArtifactStore &store) {
  auto intermediate = migrateSpatialConstraintRootV1_0ToV1_2(reference, store);
  if (!intermediate)
    return intermediate.takeError();
  return migrateSpatialConstraintRootV1_2ToV1_3(*intermediate, store);
}

llvm::Expected<ArtifactRootReference>
migrateSystemConstraintRootV1_0ToV1_3(const ArtifactRootReference &reference,
                                      const ArtifactStore &store) {
  auto intermediate = migrateSystemConstraintRootV1_0ToV1_2(reference, store);
  if (!intermediate)
    return intermediate.takeError();
  return migrateSystemConstraintRootV1_2ToV1_3(*intermediate, store);
}

} // namespace loom::mapping

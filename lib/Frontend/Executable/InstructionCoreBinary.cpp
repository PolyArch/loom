#include "InstructionCoreBinaryInternal.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Frontend/Executable/CompilerTargetBinding.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom {
InstructionCoreBinary detail::InstructionCoreBinaryBuilder::create(
    ArtifactRootReference canonicalDataflow,
    ArtifactRootReference compilerTargetBinding, BlobDigest codeBlob,
    std::vector<InstructionLoadSegment> loadSegments,
    std::vector<ThreadEntryBinding> threadEntryTable,
    std::vector<RuntimeImport> runtimeImports) {
  return InstructionCoreBinary(
      std::move(canonicalDataflow), std::move(compilerTargetBinding), codeBlob,
      std::move(loadSegments), std::move(threadEntryTable),
      std::move(runtimeImports));
}

namespace {

llvm::Error binaryError(llvm::StringRef kind, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 kind + ": " + message);
}

llvm::StringRef asText(llvm::ArrayRef<std::uint8_t> bytes) {
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

llvm::Error
validateRootSchemas(const ArtifactRootReference &dataflow,
                    const ArtifactRootReference &compilerTargetBinding) {
  if (dataflow.schemaIdentity != dataflow::canonicalDataflowSchema.identity ||
      dataflow.schemaVersion != dataflow::canonicalDataflowSchema.version)
    return binaryError("instruction_core_binary_dataflow_schema_unsupported",
                       "canonical_dataflow_ref is not the exact Canonical "
                       "Dataflow schema");
  if (compilerTargetBinding.schemaIdentity !=
          compilerTargetBindingSchema.identity ||
      compilerTargetBinding.schemaVersion !=
          compilerTargetBindingSchema.version)
    return binaryError("instruction_core_binary_target_schema_unsupported",
                       "compiler_target_binding_ref has the wrong schema");
  return llvm::Error::success();
}

llvm::Error
validateThreadEntries(llvm::ArrayRef<ThreadEntryBinding> entries,
                      const dataflow::CanonicalDataflowProgramView &view,
                      std::uint64_t executableEntryCount) {
  for (const ThreadEntryBinding &entry : entries) {
    if (auto resolved = view.resolve(entry.rootThreadLaunch)) {
      (void)*resolved;
    } else {
      return binaryError("instruction_core_binary_invalid_root",
                         llvm::toString(resolved.takeError()));
    }
    if (entry.entryOrdinal >= executableEntryCount)
      return binaryError("instruction_core_binary_missing_entry",
                         "root launch selects absent binary entry ordinal " +
                             llvm::Twine(entry.entryOrdinal));
  }
  return llvm::Error::success();
}

llvm::Error validateRuntimeImportProjection(
    llvm::ArrayRef<RuntimeImport> stored,
    llvm::ArrayRef<std::pair<std::string, std::optional<std::string>>>
        unresolved) {
  std::vector<std::pair<std::string, std::optional<std::string>>> declared;
  declared.reserve(stored.size());
  for (const RuntimeImport &entry : stored)
    declared.emplace_back(entry.abiSymbol, entry.abiSymbolVersion);
  llvm::sort(declared);
  if (!std::equal(declared.begin(), declared.end(), unresolved.begin(),
                  unresolved.end()))
    return binaryError("instruction_core_binary_import_mismatch",
                       "runtime_imports are not the exact unresolved ELF "
                       "dynamic-symbol set");
  return llvm::Error::success();
}

llvm::Error
validateRuntimeImportProviders(llvm::ArrayRef<RuntimeImport> imports,
                               const CompilerTargetBinding &target,
                               const BlobStore &blobs) {
  if (imports.empty())
    return llvm::Error::success();
  using Symbol = std::pair<std::string, std::optional<std::string>>;
  std::vector<std::vector<Symbol>> exports(target.supportComponents().size());
  for (std::size_t ordinal = 0; ordinal < target.supportComponents().size();
       ++ordinal) {
    const CompilerSupportComponent &component =
        target.supportComponents()[ordinal];
    if (component.linkMode != CompilerSupportLinkMode::Dynamic)
      continue;
    auto bytes = blobs.get(component.contentBlob);
    if (!bytes)
      return bytes.takeError();
    auto parsed = detail::parseInstructionDynamicExports(*bytes, target);
    if (!parsed)
      return parsed.takeError();
    exports[ordinal] = std::move(*parsed);
  }
  for (const RuntimeImport &entry : imports) {
    const Symbol selected{entry.abiSymbol, entry.abiSymbolVersion};
    std::size_t providerCount = 0;
    std::size_t providerOrdinal = 0;
    for (std::size_t ordinal = 0; ordinal < exports.size(); ++ordinal) {
      if (!std::binary_search(exports[ordinal].begin(), exports[ordinal].end(),
                              selected))
        continue;
      ++providerCount;
      providerOrdinal = ordinal;
    }
    if (providerCount != 1)
      return binaryError("instruction_core_binary_import_provider_mismatch",
                         "runtime import must resolve to exactly one dynamic "
                         "support component");
    if (providerOrdinal != entry.supportComponentOrdinal)
      return binaryError("instruction_core_binary_import_provider_mismatch",
                         "runtime import selects the wrong support component");
  }
  return llvm::Error::success();
}

llvm::Expected<InstructionCoreBinary>
buildValidatedBinary(detail::DecodedInstructionCoreBinaryFields fields,
                     llvm::ArrayRef<std::uint8_t> executableBytes,
                     const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (llvm::Error error = validateRootSchemas(fields.canonicalDataflow,
                                              fields.compilerTargetBinding))
    return std::move(error);
  auto dataflowArtifact =
      dataflow::importCanonicalDataflow(fields.canonicalDataflow, artifacts);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflowView = dataflowArtifact->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto target =
      importCompilerTargetBinding(fields.compilerTargetBinding, artifacts);
  if (!target)
    return target.takeError();
  if (target->binding().processorArchitecture().isHost())
    return binaryError("instruction_core_binary_host_target",
                       "InstructionCoreBinary requires an InstructionCore "
                       "CompilerTargetBinding");

  auto parsedElf =
      detail::parseInstructionElf(executableBytes, target->binding());
  if (!parsedElf)
    return parsedElf.takeError();
  if (fields.loadSegments != parsedElf->loadSegments)
    return binaryError("instruction_core_binary_segment_mismatch",
                       "load_segments are not the exact code_blob projection");

  auto entries = detail::canonicalizeThreadEntries(
      fields.threadEntryTable, fields.canonicalDataflow.artifact);
  if (!entries)
    return entries.takeError();
  if (*entries != fields.threadEntryTable)
    return binaryError("instruction_core_binary_not_canonical",
                       "thread_entry_table is not in canonical key order");
  if (llvm::Error error =
          validateThreadEntries(*entries, *dataflowView, parsedElf->entryCount))
    return std::move(error);

  auto imports = detail::canonicalizeRuntimeImports(fields.runtimeImports,
                                                    target->binding());
  if (!imports)
    return imports.takeError();
  if (*imports != fields.runtimeImports)
    return binaryError("instruction_core_binary_not_canonical",
                       "runtime_imports are not in canonical key order");
  if (llvm::Error error = validateRuntimeImportProjection(
          *imports, parsedElf->unresolvedImports))
    return std::move(error);
  if (llvm::Error error =
          validateRuntimeImportProviders(*imports, target->binding(), blobs))
    return std::move(error);

  return detail::InstructionCoreBinaryBuilder::create(
      std::move(fields.canonicalDataflow),
      std::move(fields.compilerTargetBinding), fields.codeBlob,
      std::move(fields.loadSegments), std::move(*entries), std::move(*imports));
}

} // namespace

llvm::Expected<std::uint64_t>
InstructionCoreBinary::threadEntry(dataflow::RootThreadLaunchRef root) const {
  if (root.artifact != canonicalDataflow_.artifact)
    return binaryError("instruction_core_binary_foreign_root",
                       "root launch belongs to another Dataflow artifact");
  auto found = llvm::lower_bound(
      threadEntryTable_, root.entity.value(),
      [](const ThreadEntryBinding &entry, std::uint64_t entity) {
        return entry.rootThreadLaunch.entity.value() < entity;
      });
  if (found == threadEntryTable_.end() ||
      found->rootThreadLaunch.entity != root.entity)
    return binaryError("instruction_core_binary_root_unbound",
                       "binary does not implement the root launch");
  return found->entryOrdinal;
}

llvm::Expected<InstructionCoreBinary>
decodeInstructionCoreBinary(llvm::StringRef canonicalJson,
                            const ArtifactStore &artifacts,
                            const BlobStore &blobs) {
  auto fields = detail::parseInstructionCoreBinaryFields(canonicalJson);
  if (!fields)
    return fields.takeError();
  auto executable = blobs.get(fields->codeBlob);
  if (!executable)
    return executable.takeError();
  auto binary =
      buildValidatedBinary(std::move(*fields), *executable, artifacts, blobs);
  if (!binary)
    return binary.takeError();
  if (detail::serializeInstructionCoreBinary(*binary) != canonicalJson)
    return binaryError("instruction_core_binary_not_canonical",
                       "stored JSON is not the production canonical encoding");
  return binary;
}

llvm::Expected<FinalizedInstructionCoreBinary>
finalizeInstructionCoreBinary(InstructionCoreBinaryDraft draft,
                              const ArtifactStore &artifacts,
                              const BlobStore &blobs) {
  if (llvm::Error error = validateRootSchemas(draft.canonicalDataflow,
                                              draft.compilerTargetBinding))
    return std::move(error);
  auto dataflowArtifact =
      dataflow::importCanonicalDataflow(draft.canonicalDataflow, artifacts);
  if (!dataflowArtifact)
    return dataflowArtifact.takeError();
  auto dataflowView = dataflowArtifact->view();
  if (!dataflowView)
    return dataflowView.takeError();
  auto target =
      importCompilerTargetBinding(draft.compilerTargetBinding, artifacts);
  if (!target)
    return target.takeError();
  if (target->binding().processorArchitecture().isHost())
    return binaryError("instruction_core_binary_host_target",
                       "InstructionCoreBinary requires an InstructionCore "
                       "CompilerTargetBinding");

  auto parsedElf =
      detail::parseInstructionElf(draft.executableBytes, target->binding());
  if (!parsedElf)
    return parsedElf.takeError();
  auto entries = detail::canonicalizeThreadEntries(
      draft.threadEntryTable, draft.canonicalDataflow.artifact);
  if (!entries)
    return entries.takeError();
  if (llvm::Error error =
          validateThreadEntries(*entries, *dataflowView, parsedElf->entryCount))
    return std::move(error);
  auto imports = detail::canonicalizeRuntimeImports(draft.runtimeImports,
                                                    target->binding());
  if (!imports)
    return imports.takeError();
  if (llvm::Error error = validateRuntimeImportProjection(
          *imports, parsedElf->unresolvedImports))
    return std::move(error);
  if (llvm::Error error =
          validateRuntimeImportProviders(*imports, target->binding(), blobs))
    return std::move(error);

  auto codeBlob = blobs.put(draft.executableBytes);
  if (!codeBlob)
    return codeBlob.takeError();
  InstructionCoreBinary binary = detail::InstructionCoreBinaryBuilder::create(
      std::move(draft.canonicalDataflow),
      std::move(draft.compilerTargetBinding), *codeBlob,
      std::move(parsedElf->loadSegments), std::move(*entries),
      std::move(*imports));
  const std::string json = detail::serializeInstructionCoreBinary(binary);
  auto strict = decodeInstructionCoreBinary(json, artifacts, blobs);
  if (!strict)
    return strict.takeError();
  CanonicalSemanticBytes canonicalBytes(
      std::vector<std::uint8_t>(json.begin(), json.end()));
  auto identity = artifacts.put(instructionCoreBinarySchema, canonicalBytes);
  if (!identity)
    return identity.takeError();
  return importInstructionCoreBinary(
      {instructionCoreBinarySchema.identity.str(),
       instructionCoreBinarySchema.version, *identity},
      artifacts, blobs);
}

llvm::Expected<FinalizedInstructionCoreBinary>
importInstructionCoreBinary(const ArtifactRootReference &reference,
                            const ArtifactStore &artifacts,
                            const BlobStore &blobs) {
  if (reference.schemaIdentity != instructionCoreBinarySchema.identity ||
      reference.schemaVersion != instructionCoreBinarySchema.version)
    return binaryError("instruction_core_binary_schema_unsupported",
                       "reference is not loom.instruction_core_binary 1.0");
  auto bytes = artifacts.get(reference);
  if (!bytes)
    return bytes.takeError();
  auto binary =
      decodeInstructionCoreBinary(asText(bytes->bytes()), artifacts, blobs);
  if (!binary)
    return binary.takeError();
  return FinalizedInstructionCoreBinary(reference, std::move(*bytes),
                                        std::move(*binary));
}

} // namespace loom

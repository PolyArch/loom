#include "Deployment/ExecutableLeaves.h"

#include "Common/ArtifactStore.h"
#include "Common/BlobStore.h"
#include "Dataflow/IR/DataflowCanonicalArtifact.h"
#include "Dataflow/IR/OperationSchemaCodec.h"
#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/ExecutableElf.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::deployment {

class ExecutableLeafBuilder final {
public:
  static HostProgramLeaf
  host(ArtifactRootReference compilerTargetBinding, BlobDigest programBlob,
       std::vector<HostProgramEntry> programEntries,
       std::vector<HostExternalInterface> externalInterfaces,
       BlobDigest registrationTableDigest,
       std::vector<std::uint64_t> supportComponentOrdinals) {
    return HostProgramLeaf(
        std::move(compilerTargetBinding), programBlob,
        std::move(programEntries), std::move(externalInterfaces),
        registrationTableDigest, std::move(supportComponentOrdinals));
  }

  static StaticMemoryImageLeaf
  staticMemory(ArtifactRootReference canonicalDataflow,
               dataflow::RootedGraphLaunchRef rootedGraphLaunch,
               dataflow::LogicalMemoryRootRef logicalMemoryRoot,
               ArtifactRootReference layoutBinding, std::uint64_t sizeBytes,
               std::uint64_t alignmentBytes,
               frontend::StaticMemoryPermissions permissions,
               std::vector<StaticMemoryInitializedChunk> initializedChunks,
               std::vector<StaticMemoryZeroFillRange> zeroFillRanges) {
    return StaticMemoryImageLeaf(
        std::move(canonicalDataflow), rootedGraphLaunch, logicalMemoryRoot,
        std::move(layoutBinding), sizeBytes, alignmentBytes, permissions,
        std::move(initializedChunks), std::move(zeroFillRanges));
  }
};

namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "deployment_executable_leaf_invalid: " + message);
}

mlir::MLIRContext &typeContext() {
  static thread_local mlir::MLIRContext *context = [] {
    mlir::DialectRegistry registry;
    registry.insert<mlir::LLVM::LLVMDialect>();
    auto *created =
        new mlir::MLIRContext(registry, mlir::MLIRContext::Threading::DISABLED);
    created->loadAllAvailableDialects();
    return created;
  }();
  return *context;
}

llvm::Expected<mlir::Type>
validateCanonicalType(llvm::ArrayRef<std::uint8_t> bytes,
                      llvm::StringRef context) {
  auto type = dataflow::decodeCanonicalType(bytes, &typeContext());
  if (!type)
    return invalid(context + " semantic type is not canonical: " +
                   llvm::toString(type.takeError()));
  return *type;
}

llvm::Error validateValueType(llvm::ArrayRef<std::uint8_t> bytes,
                              llvm::StringRef context) {
  auto type = validateCanonicalType(bytes, context);
  if (!type)
    return type.takeError();
  if (mlir::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(*type))
    return invalid(context + " requires a non-memory value type");
  return llvm::Error::success();
}

llvm::Error validateInterfaceType(const HostExternalInterface &interface) {
  bool knownKind = false;
  switch (interface.kind) {
  case HostExternalInterfaceKind::Value:
  case HostExternalInterfaceKind::Stream:
  case HostExternalInterfaceKind::Memory:
    knownKind = true;
  }
  if (!knownKind)
    return invalid("external interface kind is unknown");
  bool knownDirection = false;
  switch (interface.direction) {
  case HostExternalInterfaceDirection::Input:
  case HostExternalInterfaceDirection::Output:
  case HostExternalInterfaceDirection::InOut:
    knownDirection = true;
  }
  if (!knownDirection)
    return invalid("external interface direction is unknown");
  auto type = validateCanonicalType(interface.semanticType,
                                    "external interface");
  if (!type)
    return type.takeError();
  const bool isMemory =
      mlir::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(*type);
  if ((interface.kind == HostExternalInterfaceKind::Memory) != isMemory)
    return invalid("external interface kind disagrees with semantic type");
  return llvm::Error::success();
}

llvm::Expected<std::vector<HostExternalInterface>>
canonicalizeInterfaces(std::vector<HostExternalInterface> interfaces) {
  llvm::sort(interfaces, [](const auto &lhs, const auto &rhs) {
    return lhs.interfaceOrdinal < rhs.interfaceOrdinal;
  });
  if (std::adjacent_find(
          interfaces.begin(), interfaces.end(),
          [](const auto &lhs, const auto &rhs) {
            return lhs.interfaceOrdinal == rhs.interfaceOrdinal;
          }) != interfaces.end())
    return invalid("duplicate external interface ordinal");
  for (std::size_t ordinal = 0; ordinal < interfaces.size(); ++ordinal) {
    if (interfaces[ordinal].interfaceOrdinal != ordinal)
      return invalid("external interface ordinals are not dense from zero");
    if (llvm::Error error = validateInterfaceType(interfaces[ordinal]))
      return std::move(error);
  }
  return interfaces;
}

llvm::Expected<std::vector<HostProgramEntry>>
canonicalizeEntries(std::vector<HostProgramEntry> entries,
                    std::size_t interfaceCount) {
  if (entries.empty())
    return invalid("host program has no executable entry");
  llvm::sort(entries, [](const auto &lhs, const auto &rhs) {
    return lhs.entryOrdinal < rhs.entryOrdinal;
  });
  if (std::adjacent_find(entries.begin(), entries.end(),
                         [](const auto &lhs, const auto &rhs) {
                           return lhs.entryOrdinal == rhs.entryOrdinal;
                         }) != entries.end())
    return invalid("duplicate program entry ordinal");
  std::set<std::string> symbols;
  std::vector<bool> usedInterfaces(interfaceCount, false);
  for (std::size_t ordinal = 0; ordinal < entries.size(); ++ordinal) {
    HostProgramEntry &entry = entries[ordinal];
    if (entry.entryOrdinal != ordinal)
      return invalid("program entry ordinals are not dense from zero");
    if (entry.abiSymbol.empty() || entry.abiSymbol.find('\0') != std::string::npos)
      return invalid("program entry ABI symbol is empty or contains NUL");
    if (!symbols.insert(entry.abiSymbol).second)
      return invalid("duplicate program entry ABI symbol");
    for (const CanonicalTypeBytes &type : entry.valueArgumentTypes)
      if (llvm::Error error = validateValueType(type, "program argument"))
        return std::move(error);
    for (const CanonicalTypeBytes &type : entry.valueResultTypes)
      if (llvm::Error error = validateValueType(type, "program result"))
        return std::move(error);
    llvm::sort(entry.externalInterfaceOrdinals);
    if (std::adjacent_find(entry.externalInterfaceOrdinals.begin(),
                           entry.externalInterfaceOrdinals.end()) !=
        entry.externalInterfaceOrdinals.end())
      return invalid("program entry repeats an external interface ordinal");
    for (std::uint64_t interfaceOrdinal :
         entry.externalInterfaceOrdinals) {
      if (interfaceOrdinal >= interfaceCount)
        return invalid("external interface ordinal is out of range");
      usedInterfaces[interfaceOrdinal] = true;
    }
  }
  if (llvm::any_of(usedInterfaces, [](bool used) { return !used; }))
    return invalid("external interface catalog contains an unused entry");
  return entries;
}

llvm::StringRef interfaceKindName(HostExternalInterfaceKind kind) {
  switch (kind) {
  case HostExternalInterfaceKind::Value:
    return "value";
  case HostExternalInterfaceKind::Stream:
    return "stream";
  case HostExternalInterfaceKind::Memory:
    return "memory";
  }
  llvm_unreachable("unknown host external interface kind");
}

llvm::StringRef interfaceDirectionName(HostExternalInterfaceDirection direction) {
  switch (direction) {
  case HostExternalInterfaceDirection::Input:
    return "input";
  case HostExternalInterfaceDirection::Output:
    return "output";
  case HostExternalInterfaceDirection::InOut:
    return "inout";
  }
  llvm_unreachable("unknown host external interface direction");
}

BlobDigest registrationDigest(
    llvm::ArrayRef<HostProgramEntry> entries,
    llvm::ArrayRef<HostExternalInterface> interfaces) {
  llvm::SmallString<1024> storage;
  llvm::raw_svector_ostream output(storage);
  llvm::json::OStream json(output);
  auto writeTypes = [&](llvm::StringRef name,
                        llvm::ArrayRef<CanonicalTypeBytes> types) {
    json.attributeArray(name, [&] {
      for (const CanonicalTypeBytes &type : types)
        json.value(llvm::toHex(type, true));
    });
  };
  json.object([&] {
    json.attribute("schema", "loom.host_registration_table");
    json.attribute("schema_version", "1.0");
    json.attributeArray("program_entries", [&] {
      for (const HostProgramEntry &entry : entries)
        json.object([&] {
          json.attribute("entry_ordinal", entry.entryOrdinal);
          json.attribute("abi_symbol", entry.abiSymbol);
          writeTypes("value_argument_types", entry.valueArgumentTypes);
          writeTypes("value_result_types", entry.valueResultTypes);
          json.attributeArray("external_interface_ordinals", [&] {
            for (std::uint64_t ordinal : entry.externalInterfaceOrdinals)
              json.value(ordinal);
          });
        });
    });
    json.attributeArray("external_interfaces", [&] {
      for (const HostExternalInterface &interface : interfaces)
        json.object([&] {
          json.attribute("interface_ordinal", interface.interfaceOrdinal);
          json.attribute("kind", interfaceKindName(interface.kind));
          json.attribute("direction",
                         interfaceDirectionName(interface.direction));
          json.attribute("semantic_type",
                         llvm::toHex(interface.semanticType, true));
        });
    });
  });
  return computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(storage.data()), storage.size()));
}

template <typename ELFT>
llvm::Expected<std::vector<std::pair<std::uint64_t, std::uint64_t>>>
executableRanges(const llvm::object::ELFObjectFile<ELFT> &object) {
  auto headers = object.getELFFile().program_headers();
  if (!headers)
    return headers.takeError();
  std::vector<std::pair<std::uint64_t, std::uint64_t>> result;
  for (const auto &header : *headers) {
    if (header.p_type != llvm::ELF::PT_LOAD ||
        (header.p_flags & llvm::ELF::PF_X) == 0)
      continue;
    if (header.p_memsz == 0 ||
        header.p_vaddr > std::numeric_limits<std::uint64_t>::max() -
                             header.p_memsz)
      return invalid("host executable has an invalid executable segment");
    result.emplace_back(header.p_vaddr, header.p_vaddr + header.p_memsz);
  }
  if (result.empty())
    return invalid("host executable has no executable load segment");
  return result;
}

llvm::Expected<std::vector<std::pair<std::uint64_t, std::uint64_t>>>
executableRanges(const llvm::object::ELFObjectFileBase &object) {
  if (const auto *typed =
          llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(&object))
    return executableRanges(*typed);
  if (const auto *typed =
          llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(&object))
    return executableRanges(*typed);
  if (const auto *typed =
          llvm::dyn_cast<llvm::object::ELF32BEObjectFile>(&object))
    return executableRanges(*typed);
  return executableRanges(
      *llvm::cast<llvm::object::ELF64BEObjectFile>(&object));
}

llvm::Error validateHostExecutable(
    llvm::ArrayRef<std::uint8_t> bytes, const CompilerTargetBinding &target,
    llvm::ArrayRef<HostProgramEntry> entries) {
  if (bytes.empty())
    return invalid("host program blob is empty");
  llvm::MemoryBufferRef buffer(
      llvm::StringRef(reinterpret_cast<const char *>(bytes.data()), bytes.size()),
      "host-program.elf");
  auto object = llvm::object::ObjectFile::createObjectFile(buffer);
  if (!object)
    return invalid("host program blob is not a valid object: " +
                   llvm::toString(object.takeError()));
  const auto *elf =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(object->get());
  if (!elf)
    return invalid("host program blob is not ELF");
  if (elf->getEType() != llvm::ELF::ET_EXEC &&
      elf->getEType() != llvm::ELF::ET_DYN)
    return invalid("host program blob is not an ELF executable or PIE");
  if (llvm::Error error = validateElfTarget(*elf, target))
    return invalid(llvm::toString(std::move(error)));
  auto ranges = executableRanges(*elf);
  if (!ranges)
    return ranges.takeError();

  std::set<std::string> required;
  for (const HostProgramEntry &entry : entries)
    required.insert(entry.abiSymbol);
  for (const llvm::object::ELFSymbolRef &symbol : elf->symbols()) {
    auto name = symbol.getName();
    if (!name)
      return name.takeError();
    auto found = required.find(name->str());
    if (found == required.end())
      continue;
    auto flags = symbol.getFlags();
    auto address = symbol.getAddress();
    if (!flags)
      return flags.takeError();
    if (!address)
      return address.takeError();
    const bool executable = llvm::any_of(*ranges, [&](const auto &range) {
      return *address >= range.first && *address < range.second;
    });
    if ((*flags & llvm::object::SymbolRef::SF_Undefined) != 0 ||
        symbol.getELFType() != llvm::ELF::STT_FUNC ||
        (symbol.getBinding() != llvm::ELF::STB_GLOBAL &&
         symbol.getBinding() != llvm::ELF::STB_WEAK) ||
        !executable)
      return invalid("host program entry is not a defined executable global "
                     "function");
    required.erase(found);
  }
  if (!required.empty())
    return invalid("host program blob is missing ABI symbol '" +
                   *required.begin() + "'");
  return llvm::Error::success();
}

llvm::Expected<FinalizedCompilerTargetBinding>
importTarget(const ArtifactRootReference &reference,
             const ArtifactStore &artifacts) {
  if (reference.schemaIdentity != compilerTargetBindingSchema.identity ||
      reference.schemaVersion != compilerTargetBindingSchema.version)
    return invalid("layout binding has the wrong schema descriptor");
  return importCompilerTargetBinding(reference, artifacts);
}

llvm::Error validateLayoutCompatibility(llvm::StringRef moduleLayout,
                                        const CompilerTargetBinding &binding) {
  auto module = llvm::DataLayout::parse(moduleLayout);
  if (!module)
    return invalid("linked module DataLayout is invalid: " +
                   llvm::toString(module.takeError()));
  auto target = llvm::DataLayout::parse(binding.dataLayout());
  if (!target)
    return invalid("binding DataLayout is invalid: " +
                   llvm::toString(target.takeError()));
  if (*module != *target)
    return invalid("linked module DataLayout is not structurally compatible "
                   "with layout binding");
  return llvm::Error::success();
}

llvm::Expected<dataflow::CanonicalDataflowArtifact>
importMemoryOwner(const StaticMemoryImageLeaf &leaf,
                  const ArtifactStore &artifacts) {
  if (leaf.canonicalDataflow().schemaIdentity !=
          dataflow::canonicalDataflowSchema.identity ||
      leaf.canonicalDataflow().schemaVersion !=
          dataflow::canonicalDataflowSchema.version)
    return invalid("static memory canonical_dataflow_ref has the wrong schema");
  return dataflow::importCanonicalDataflow(leaf.canonicalDataflow(), artifacts);
}

struct ByteRange final {
  std::uint64_t offset;
  std::uint64_t count;
  bool initialized;
};

llvm::Error validateStaticRanges(const StaticMemoryImageLeaf &leaf,
                                 const BlobStore &blobs) {
  std::vector<ByteRange> ranges;
  ranges.reserve(leaf.initializedChunks().size() +
                 leaf.zeroFillRanges().size());
  std::uint64_t previousOffset = 0;
  bool first = true;
  for (const StaticMemoryInitializedChunk &chunk : leaf.initializedChunks()) {
    if (!first && chunk.byteOffset <= previousOffset)
      return invalid("initialized chunks are not in canonical order");
    first = false;
    previousOffset = chunk.byteOffset;
    if (chunk.byteCount == 0 || chunk.byteOffset > leaf.sizeBytes() ||
        chunk.byteCount > leaf.sizeBytes() - chunk.byteOffset)
      return invalid("initialized chunk is empty or out of bounds");
    auto bytes = blobs.get(chunk.blobDigest);
    if (!bytes)
      return invalid("initialized chunk blob is unavailable: " +
                     llvm::toString(bytes.takeError()));
    if (bytes->size() != chunk.byteCount)
      return invalid("initialized chunk blob size disagrees with byte_count");
    ranges.push_back({chunk.byteOffset, chunk.byteCount, true});
  }
  previousOffset = 0;
  first = true;
  for (const StaticMemoryZeroFillRange &range : leaf.zeroFillRanges()) {
    if (!first && range.byteOffset <= previousOffset)
      return invalid("zero-fill ranges are not in canonical order");
    first = false;
    previousOffset = range.byteOffset;
    if (range.byteCount == 0 || range.byteOffset > leaf.sizeBytes() ||
        range.byteCount > leaf.sizeBytes() - range.byteOffset)
      return invalid("zero-fill range is empty or out of bounds");
    ranges.push_back({range.byteOffset, range.byteCount, false});
  }
  llvm::sort(ranges, [](const auto &lhs, const auto &rhs) {
    return std::tie(lhs.offset, lhs.count, lhs.initialized) <
           std::tie(rhs.offset, rhs.count, rhs.initialized);
  });
  std::uint64_t cursor = 0;
  for (const ByteRange &range : ranges) {
    if (range.offset != cursor)
      return invalid(range.offset < cursor
                         ? "static memory ranges overlap"
                         : "static memory ranges do not cover the full image");
    cursor += range.count;
  }
  if (cursor != leaf.sizeBytes())
    return invalid("static memory ranges do not cover the full image");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<HostProgramLeaf>
finalizeHostProgramLeaf(HostProgramLeafDraft draft,
                        const ArtifactStore &artifacts,
                        const BlobStore &blobs) {
  auto target = importTarget(draft.compilerTargetBinding, artifacts);
  if (!target)
    return target.takeError();
  if (!target->binding().processorArchitecture().isHost())
    return invalid("HostProgramLeaf requires a HostCore CompilerTargetBinding");
  auto interfaces =
      canonicalizeInterfaces(std::move(draft.externalInterfaces));
  if (!interfaces)
    return interfaces.takeError();
  auto entries = canonicalizeEntries(std::move(draft.programEntries),
                                     interfaces->size());
  if (!entries)
    return entries.takeError();
  llvm::sort(draft.supportComponentOrdinals);
  if (std::adjacent_find(draft.supportComponentOrdinals.begin(),
                         draft.supportComponentOrdinals.end()) !=
      draft.supportComponentOrdinals.end())
    return invalid("support component ordinal is duplicated");
  for (std::uint64_t ordinal : draft.supportComponentOrdinals)
    if (ordinal >= target->binding().supportComponents().size())
      return invalid("support component ordinal is out of range");
  if (llvm::Error error = validateHostExecutable(
          draft.programBytes, target->binding(), *entries))
    return std::move(error);
  auto programBlob = blobs.put(draft.programBytes);
  if (!programBlob)
    return programBlob.takeError();
  const BlobDigest registrationTableDigest =
      registrationDigest(*entries, *interfaces);
  HostProgramLeaf leaf = ExecutableLeafBuilder::host(
      std::move(draft.compilerTargetBinding), *programBlob, std::move(*entries),
      std::move(*interfaces), registrationTableDigest,
      std::move(draft.supportComponentOrdinals));
  if (llvm::Error error = validateHostProgramLeaf(leaf, artifacts, blobs))
    return std::move(error);
  return leaf;
}

llvm::Error validateHostProgramLeaf(const HostProgramLeaf &leaf,
                                    const ArtifactStore &artifacts,
                                    const BlobStore &blobs) {
  auto target = importTarget(leaf.compilerTargetBinding(), artifacts);
  if (!target)
    return target.takeError();
  if (!target->binding().processorArchitecture().isHost())
    return invalid("HostProgramLeaf requires a HostCore CompilerTargetBinding");
  auto bytes = blobs.get(leaf.programBlob());
  if (!bytes)
    return invalid("host program blob is unavailable: " +
                   llvm::toString(bytes.takeError()));
  std::vector<HostExternalInterface> interfaces(
      leaf.externalInterfaces().begin(), leaf.externalInterfaces().end());
  auto canonicalInterfaces = canonicalizeInterfaces(std::move(interfaces));
  if (!canonicalInterfaces)
    return canonicalInterfaces.takeError();
  if (llvm::ArrayRef<HostExternalInterface>(*canonicalInterfaces) !=
      leaf.externalInterfaces())
    return invalid("external interface catalog is not canonical");
  std::vector<HostProgramEntry> entries(leaf.programEntries().begin(),
                                        leaf.programEntries().end());
  auto canonicalEntries =
      canonicalizeEntries(std::move(entries), canonicalInterfaces->size());
  if (!canonicalEntries)
    return canonicalEntries.takeError();
  if (llvm::ArrayRef<HostProgramEntry>(*canonicalEntries) !=
      leaf.programEntries())
    return invalid("program entry catalog is not canonical");
  if (registrationDigest(*canonicalEntries, *canonicalInterfaces) !=
      leaf.registrationTableDigest())
    return invalid("registration table digest disagrees with executable "
                   "catalogs");
  if (!llvm::is_sorted(leaf.supportComponentOrdinals()) ||
      std::adjacent_find(leaf.supportComponentOrdinals().begin(),
                         leaf.supportComponentOrdinals().end()) !=
          leaf.supportComponentOrdinals().end())
    return invalid("support component ordinals are not sorted and unique");
  for (std::uint64_t ordinal : leaf.supportComponentOrdinals())
    if (ordinal >= target->binding().supportComponents().size())
      return invalid("support component ordinal is out of range");
  return validateHostExecutable(*bytes, target->binding(), *canonicalEntries);
}

llvm::Expected<StaticMemoryImageLeaf>
buildStaticMemoryImageLeaf(const ArtifactRootReference &canonicalDataflow,
                           dataflow::RootedGraphLaunchRef rootedGraphLaunch,
                           dataflow::LogicalMemoryRootRef logicalMemoryRoot,
                           const ArtifactRootReference &layoutBinding,
                           const frontend::StaticGlobalMemoryCatalog &catalog,
                           std::uint64_t globalOrdinal,
                           const ArtifactStore &artifacts,
                           const BlobStore &blobs) {
  auto target = importTarget(layoutBinding, artifacts);
  if (!target)
    return target.takeError();
  if (llvm::Error error =
          validateLayoutCompatibility(catalog.dataLayout, target->binding()))
    return std::move(error);
  if (globalOrdinal >= catalog.globals.size())
    return invalid("static global ordinal is out of range");
  const frontend::StaticGlobalMemory &global = catalog.globals[globalOrdinal];
  if (global.provision != frontend::StaticGlobalProvision::Image)
    return invalid("static global is runtime-provided rather than an image");
  if (global.sizeBytes == 0 || global.bytes.size() != global.sizeBytes)
    return invalid("static global image size disagrees with its bytes");
  if (global.alignmentBytes == 0 ||
      !llvm::isPowerOf2_64(global.alignmentBytes))
    return invalid("static global alignment is not a positive power of two");

  std::vector<StaticMemoryInitializedChunk> chunks;
  std::vector<StaticMemoryZeroFillRange> zeroFill;
  if (llvm::all_of(global.bytes,
                   [](std::uint8_t byte) { return byte == 0; })) {
    zeroFill.push_back({0, global.sizeBytes});
  } else {
    auto digest = blobs.put(global.bytes);
    if (!digest)
      return digest.takeError();
    chunks.push_back({0, global.sizeBytes, *digest});
  }
  StaticMemoryImageLeaf leaf = ExecutableLeafBuilder::staticMemory(
      canonicalDataflow, rootedGraphLaunch, logicalMemoryRoot, layoutBinding,
      global.sizeBytes, global.alignmentBytes, global.permissions,
      std::move(chunks), std::move(zeroFill));
  if (llvm::Error error =
          validateStaticMemoryImageLeaf(leaf, artifacts, blobs))
    return std::move(error);
  return leaf;
}

llvm::Error validateStaticMemoryImageLeaf(const StaticMemoryImageLeaf &leaf,
                                          const ArtifactStore &artifacts,
                                          const BlobStore &blobs) {
  if (leaf.sizeBytes() == 0)
    return invalid("static memory image size must be positive");
  if (leaf.alignmentBytes() == 0 ||
      !llvm::isPowerOf2_64(leaf.alignmentBytes()))
    return invalid("static memory alignment is not a positive power of two");
  bool knownPermissions = false;
  switch (leaf.permissions()) {
  case frontend::StaticMemoryPermissions::ReadOnly:
  case frontend::StaticMemoryPermissions::ReadWrite:
    knownPermissions = true;
  }
  if (!knownPermissions)
    return invalid("static memory permissions are unknown");
  auto target = importTarget(leaf.layoutBinding(), artifacts);
  if (!target)
    return target.takeError();
  auto program = importMemoryOwner(leaf, artifacts);
  if (!program)
    return program.takeError();
  auto dataflowLayout = program->module()->getAttrOfType<mlir::StringAttr>(
      "llvm.data_layout");
  if (!dataflowLayout)
    return invalid("Canonical Dataflow has no LLVM DataLayout projection");
  if (llvm::Error error =
          validateLayoutCompatibility(dataflowLayout.getValue(),
                                      target->binding()))
    return error;
  auto view = program->view();
  if (!view)
    return view.takeError();
  auto memoryInputs = view->graphMemoryInputs(leaf.rootedGraphLaunch());
  if (!memoryInputs)
    return invalid("rooted graph launch is invalid: " +
                   llvm::toString(memoryInputs.takeError()));
  if (leaf.logicalMemoryRoot().artifact !=
      leaf.canonicalDataflow().artifact)
    return invalid("logical memory root has a foreign Dataflow owner");
  auto root = view->resolve(leaf.logicalMemoryRoot());
  if (!root)
    return invalid("logical memory root is invalid: " +
                   llvm::toString(root.takeError()));
  const bool isSource = llvm::any_of(
      *memoryInputs, [&](const dataflow::LogicalMemoryRootOrViewRef &input) {
        if (const auto *inputRoot =
                std::get_if<dataflow::LogicalMemoryRootRef>(&input))
          return *inputRoot == leaf.logicalMemoryRoot();
        return std::get<dataflow::LogicalMemoryViewRef>(input).root ==
               leaf.logicalMemoryRoot();
      });
  if (!isSource)
    return invalid("logical memory root is not a source of the rooted graph "
                   "launch");
  auto extent = view->staticMemoryByteExtent(
      dataflow::LogicalMemoryRootOrViewRef{leaf.logicalMemoryRoot()});
  if (!extent)
    return extent.takeError();
  if (*extent && **extent != leaf.sizeBytes())
    return invalid("logical memory extent disagrees with static image size");
  if (llvm::Error error = validateStaticRanges(leaf, blobs))
    return error;
  return llvm::Error::success();
}

} // namespace loom::deployment

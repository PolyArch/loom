#include "InstructionCoreBinaryInternal.h"

#include "Frontend/Executable/CompilerTargetBinding.h"
#include "Frontend/Executable/ExecutableElf.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::detail {
namespace {

constexpr llvm::StringLiteral entrySymbolPrefix = "__loom_thread_entry_";

llvm::Error elfError(llvm::StringRef kind, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 kind + ": " + message);
}

llvm::Expected<std::uint64_t> parseEntryOrdinal(llvm::StringRef name) {
  llvm::StringRef suffix = name.drop_front(entrySymbolPrefix.size());
  if (suffix.empty())
    return elfError("instruction_core_binary_invalid_entry_symbol",
                    "thread entry symbol has no ordinal");
  std::uint64_t ordinal = 0;
  if (suffix.getAsInteger(10, ordinal) || suffix != std::to_string(ordinal))
    return elfError("instruction_core_binary_invalid_entry_symbol",
                    "thread entry symbol ordinal is not canonical unsigned "
                    "decimal");
  return ordinal;
}

bool addressInExecutableSegment(
    std::uint64_t address, llvm::ArrayRef<InstructionLoadSegment> segments) {
  return llvm::any_of(segments, [&](const auto &segment) {
    return segment.executable && address >= segment.virtualAddress &&
           address - segment.virtualAddress < segment.memorySize;
  });
}

llvm::Expected<std::uint64_t>
parseEntryCatalog(const llvm::object::ELFObjectFileBase &object,
                  llvm::ArrayRef<InstructionLoadSegment> segments) {
  std::map<std::uint64_t, std::uint64_t> addresses;
  for (const llvm::object::ELFSymbolRef &symbol : object.symbols()) {
    auto name = symbol.getName();
    if (!name)
      return name.takeError();
    if (!name->starts_with(entrySymbolPrefix))
      continue;
    auto ordinal = parseEntryOrdinal(*name);
    if (!ordinal)
      return ordinal.takeError();
    auto flags = symbol.getFlags();
    if (!flags)
      return flags.takeError();
    if ((*flags & llvm::object::SymbolRef::SF_Undefined) != 0 ||
        symbol.getELFType() != llvm::ELF::STT_FUNC ||
        symbol.getBinding() != llvm::ELF::STB_GLOBAL)
      return elfError("instruction_core_binary_invalid_entry_symbol",
                      "thread entry must be a defined global ELF function");
    auto address = symbol.getAddress();
    if (!address)
      return address.takeError();
    if (!addressInExecutableSegment(*address, segments))
      return elfError("instruction_core_binary_invalid_entry_symbol",
                      "thread entry is outside executable PT_LOAD segments");
    if (!addresses.emplace(*ordinal, *address).second)
      return elfError("instruction_core_binary_duplicate_entry",
                      "ELF contains duplicate thread entry ordinal " +
                          llvm::Twine(*ordinal));
  }
  if (addresses.empty())
    return elfError("instruction_core_binary_missing_entry",
                    "ELF contains no generated thread entry symbol");
  std::uint64_t expected = 0;
  for (const auto &[ordinal, address] : addresses) {
    (void)address;
    if (ordinal != expected)
      return elfError("instruction_core_binary_noncanonical_entry_catalog",
                      "thread entry ordinals are not dense from zero");
    ++expected;
  }
  return expected;
}

llvm::Expected<std::vector<std::pair<std::string, std::optional<std::string>>>>
parseDynamicSymbols(const llvm::object::ELFObjectFileBase &object,
                    bool undefined) {
  auto versions = object.readDynsymVersions();
  if (!versions)
    return versions.takeError();
  std::vector<std::pair<std::string, std::optional<std::string>>> result;
  std::size_t index = 0;
  const auto symbols = object.getDynamicSymbolIterators();
  for (const llvm::object::ELFSymbolRef &symbol : symbols) {
    auto flags = symbol.getFlags();
    if (!flags)
      return flags.takeError();
    auto name = symbol.getName();
    if (!name)
      return name.takeError();
    const bool isUndefined =
        (*flags & llvm::object::SymbolRef::SF_Undefined) != 0;
    const bool externallyVisible =
        symbol.getBinding() == llvm::ELF::STB_GLOBAL ||
        symbol.getBinding() == llvm::ELF::STB_WEAK;
    if (isUndefined == undefined && externallyVisible && !name->empty()) {
      std::optional<std::string> version;
      if (!versions->empty()) {
        if (index >= versions->size())
          return elfError("instruction_core_binary_invalid_imports",
                          "dynamic symbol version table is truncated");
        if (!(*versions)[index].Name.empty())
          version = (*versions)[index].Name;
      }
      result.emplace_back(name->str(), std::move(version));
    }
    ++index;
  }
  if (!versions->empty() && index != versions->size())
    return elfError("instruction_core_binary_invalid_imports",
                    "dynamic symbol and version table sizes disagree");
  llvm::sort(result);
  if (std::adjacent_find(result.begin(), result.end()) != result.end())
    return elfError("instruction_core_binary_duplicate_import",
                    "ELF contains duplicate unresolved dynamic import");
  return result;
}

llvm::Error validateNoHiddenUndefinedSymbols(
    const llvm::object::ELFObjectFileBase &object,
    llvm::ArrayRef<std::pair<std::string, std::optional<std::string>>>
        dynamicImports) {
  std::set<std::string> dynamicNames;
  for (const auto &entry : dynamicImports)
    dynamicNames.insert(entry.first);
  for (const llvm::object::ELFSymbolRef &symbol : object.symbols()) {
    auto flags = symbol.getFlags();
    if (!flags)
      return flags.takeError();
    if ((*flags & llvm::object::SymbolRef::SF_Undefined) == 0)
      continue;
    auto name = symbol.getName();
    if (!name)
      return name.takeError();
    if (!name->empty() && !dynamicNames.count(name->str()))
      return elfError("instruction_core_binary_undeclared_import",
                      "undefined ELF symbol '" + *name +
                          "' is not a dynamic runtime import");
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<ParsedInstructionElf>
parseInstructionElf(llvm::ArrayRef<std::uint8_t> bytes,
                    const CompilerTargetBinding &target) {
  if (bytes.empty())
    return elfError("instruction_core_binary_invalid_elf",
                    "code_blob is empty");
  llvm::MemoryBufferRef buffer(
      llvm::StringRef(reinterpret_cast<const char *>(bytes.data()),
                      bytes.size()),
      "instruction-core.elf");
  auto object = llvm::object::ObjectFile::createObjectFile(buffer);
  if (!object)
    return elfError("instruction_core_binary_invalid_elf",
                    llvm::toString(object.takeError()));
  const auto *elf =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(object->get());
  if (!elf)
    return elfError("instruction_core_binary_invalid_elf",
                    "code_blob is not ELF");
  if (elf->getEType() != llvm::ELF::ET_EXEC &&
      elf->getEType() != llvm::ELF::ET_DYN)
    return elfError("instruction_core_binary_not_executable",
                    "code_blob is not an ELF executable or PIE");
  if (llvm::Error error = validateElfTarget(*elf, target))
    return elfError("instruction_core_binary_target_mismatch",
                    llvm::toString(std::move(error)));
  auto segments = projectExecutableLoadSegments(*elf, bytes.size());
  if (!segments)
    return segments.takeError();
  auto entryCount = parseEntryCatalog(*elf, *segments);
  if (!entryCount)
    return entryCount.takeError();
  auto imports = parseDynamicSymbols(*elf, /*undefined=*/true);
  if (!imports)
    return imports.takeError();
  if (llvm::Error error = validateNoHiddenUndefinedSymbols(*elf, *imports))
    return std::move(error);
  return ParsedInstructionElf{std::move(*segments), *entryCount,
                              std::move(*imports)};
}

llvm::Expected<std::vector<std::pair<std::string, std::optional<std::string>>>>
parseInstructionDynamicExports(llvm::ArrayRef<std::uint8_t> bytes,
                               const CompilerTargetBinding &target) {
  if (bytes.empty())
    return elfError("instruction_core_binary_invalid_support_component",
                    "dynamic support component blob is empty");
  llvm::MemoryBufferRef buffer(
      llvm::StringRef(reinterpret_cast<const char *>(bytes.data()),
                      bytes.size()),
      "instruction-support.elf");
  auto object = llvm::object::ObjectFile::createObjectFile(buffer);
  if (!object)
    return elfError("instruction_core_binary_invalid_support_component",
                    llvm::toString(object.takeError()));
  const auto *elf =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(object->get());
  if (!elf || elf->getEType() != llvm::ELF::ET_DYN)
    return elfError("instruction_core_binary_invalid_support_component",
                    "dynamic support component is not an ELF shared object");
  if (llvm::Error error = validateElfTarget(*elf, target))
    return elfError("instruction_core_binary_target_mismatch",
                    llvm::toString(std::move(error)));
  return parseDynamicSymbols(*elf, /*undefined=*/false);
}

} // namespace loom::detail

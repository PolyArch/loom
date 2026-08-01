#include "InstructionCoreBinaryInternal.h"

#include "Frontend/Executable/CompilerTargetBinding.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/TargetParser/Triple.h"

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

template <typename ELFT>
llvm::Expected<std::vector<InstructionLoadSegment>>
parseLoadSegments(const llvm::object::ELFObjectFile<ELFT> &object,
                  std::size_t blobSize) {
  auto headers = object.getELFFile().program_headers();
  if (!headers)
    return headers.takeError();

  std::vector<InstructionLoadSegment> result;
  for (const auto &header : *headers) {
    if (header.p_type != llvm::ELF::PT_LOAD)
      continue;
    const std::uint64_t offset = header.p_offset;
    const std::uint64_t fileSize = header.p_filesz;
    const std::uint64_t memorySize = header.p_memsz;
    const std::uint64_t alignment = header.p_align;
    if (fileSize > memorySize)
      return elfError("instruction_core_binary_invalid_segment",
                      "PT_LOAD file size exceeds memory size");
    if (offset > blobSize || fileSize > blobSize - offset)
      return elfError("instruction_core_binary_invalid_segment",
                      "PT_LOAD file range escapes code_blob");
    if (alignment != 0 && !llvm::isPowerOf2_64(alignment))
      return elfError("instruction_core_binary_invalid_segment",
                      "PT_LOAD alignment is not zero or a power of two");
    if (alignment != 0 && header.p_vaddr % alignment != offset % alignment)
      return elfError("instruction_core_binary_invalid_segment",
                      "PT_LOAD virtual address and file offset are not "
                      "alignment-congruent");
    result.push_back({0, header.p_vaddr, offset, fileSize, memorySize,
                      alignment, (header.p_flags & llvm::ELF::PF_R) != 0,
                      (header.p_flags & llvm::ELF::PF_W) != 0,
                      (header.p_flags & llvm::ELF::PF_X) != 0});
  }
  llvm::sort(result, [](const auto &lhs, const auto &rhs) {
    return std::tuple{lhs.virtualAddress, lhs.fileOffset, lhs.readable,
                      lhs.writable,       lhs.executable, lhs.fileSize,
                      lhs.memorySize,     lhs.alignment} <
           std::tuple{rhs.virtualAddress, rhs.fileOffset, rhs.readable,
                      rhs.writable,       rhs.executable, rhs.fileSize,
                      rhs.memorySize,     rhs.alignment};
  });
  if (result.empty())
    return elfError("instruction_core_binary_missing_load_segment",
                    "ELF executable has no PT_LOAD segment");
  if (!llvm::any_of(result,
                    [](const auto &segment) { return segment.executable; }))
    return elfError("instruction_core_binary_missing_executable_segment",
                    "ELF executable has no executable PT_LOAD segment");
  for (std::uint64_t ordinal = 0; ordinal < result.size(); ++ordinal)
    result[ordinal].ordinal = ordinal;
  return result;
}

llvm::Expected<std::vector<InstructionLoadSegment>>
parseLoadSegments(const llvm::object::ELFObjectFileBase &object,
                  std::size_t blobSize) {
  if (const auto *value =
          llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(&object))
    return parseLoadSegments(*value, blobSize);
  if (const auto *value =
          llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(&object))
    return parseLoadSegments(*value, blobSize);
  if (const auto *value =
          llvm::dyn_cast<llvm::object::ELF32BEObjectFile>(&object))
    return parseLoadSegments(*value, blobSize);
  return parseLoadSegments(
      *llvm::cast<llvm::object::ELF64BEObjectFile>(&object), blobSize);
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

unsigned expectedRiscVFloatAbi(fabric::RiscVAbi abi) {
  switch (abi) {
  case fabric::RiscVAbi::Ilp32:
  case fabric::RiscVAbi::Ilp32e:
  case fabric::RiscVAbi::Lp64:
    return llvm::ELF::EF_RISCV_FLOAT_ABI_SOFT;
  case fabric::RiscVAbi::Ilp32f:
  case fabric::RiscVAbi::Lp64f:
    return llvm::ELF::EF_RISCV_FLOAT_ABI_SINGLE;
  case fabric::RiscVAbi::Ilp32d:
  case fabric::RiscVAbi::Lp64d:
    return llvm::ELF::EF_RISCV_FLOAT_ABI_DOUBLE;
  }
  llvm_unreachable("unknown RISC-V ABI");
}

llvm::Error validateElfTarget(const llvm::object::ELFObjectFileBase &object,
                              const CompilerTargetBinding &target) {
  if (target.objectFormat() != CompilerObjectFormat::Elf)
    return elfError("instruction_core_binary_target_mismatch",
                    "CompilerTargetBinding does not select ELF");
  if (object.getEType() != llvm::ELF::ET_EXEC &&
      object.getEType() != llvm::ELF::ET_DYN)
    return elfError("instruction_core_binary_not_executable",
                    "code_blob is not an ELF executable or PIE");
  if (object.getEMachine() != llvm::ELF::EM_RISCV)
    return elfError("instruction_core_binary_target_mismatch",
                    "ELF machine is not RISC-V");
  const llvm::Triple targetTriple(target.targetTriple());
  if (object.getArch() != targetTriple.getArch())
    return elfError("instruction_core_binary_target_mismatch",
                    "ELF class does not match CompilerTargetBinding triple");
  const unsigned flags = object.getPlatformFlags();
  if ((flags & llvm::ELF::EF_RISCV_FLOAT_ABI) !=
      expectedRiscVFloatAbi(target.backendAbi()))
    return elfError("instruction_core_binary_target_mismatch",
                    "ELF floating-point ABI does not match target binding");
  const bool expectsEmbedded = target.backendAbi() == fabric::RiscVAbi::Ilp32e;
  if (((flags & llvm::ELF::EF_RISCV_RVE) != 0) != expectsEmbedded)
    return elfError("instruction_core_binary_target_mismatch",
                    "ELF base integer ABI does not match target binding");

  auto observedFeatures = object.getFeatures();
  if (!observedFeatures)
    return observedFeatures.takeError();
  std::vector<std::string> observed;
  for (const std::string &feature : observedFeatures->getFeatures())
    if (feature != "+64bit" && feature != "-64bit")
      observed.push_back(feature);
  const unsigned xlen = targetTriple.isArch64Bit() ? 64 : 32;
  auto expectedIsa =
      llvm::RISCVISAInfo::parseFeatures(xlen, target.backendFeatures().vec());
  if (!expectedIsa)
    return expectedIsa.takeError();
  auto observedIsa = llvm::RISCVISAInfo::parseFeatures(xlen, observed);
  if (!observedIsa)
    return observedIsa.takeError();
  if ((*expectedIsa)->toString() != (*observedIsa)->toString())
    return elfError("instruction_core_binary_target_mismatch",
                    "ELF ISA attributes do not match target binding features");
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
  if (llvm::Error error = validateElfTarget(*elf, target))
    return std::move(error);
  auto segments = parseLoadSegments(*elf, bytes.size());
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
    return std::move(error);
  return parseDynamicSymbols(*elf, /*undefined=*/false);
}

} // namespace loom::detail

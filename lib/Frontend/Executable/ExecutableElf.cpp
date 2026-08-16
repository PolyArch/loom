#include "Frontend/Executable/ExecutableElf.h"

#include "Frontend/Executable/CompilerTargetBinding.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace loom {
namespace {

llvm::Error elfError(llvm::StringRef kind, const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 kind + ": " + message);
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

template <typename ELFT>
llvm::Expected<std::vector<ExecutableLoadSegment>>
projectLoadSegments(const llvm::object::ELFObjectFile<ELFT> &object,
                    std::size_t blobSize) {
  auto headers = object.getELFFile().program_headers();
  if (!headers)
    return headers.takeError();

  std::vector<ExecutableLoadSegment> result;
  for (const auto &header : *headers) {
    if (header.p_type != llvm::ELF::PT_LOAD)
      continue;
    const std::uint64_t offset = header.p_offset;
    const std::uint64_t fileSize = header.p_filesz;
    const std::uint64_t memorySize = header.p_memsz;
    const std::uint64_t alignment = header.p_align;
    if (memorySize == 0 || fileSize > memorySize)
      return elfError("executable_elf_invalid_segment",
                      "PT_LOAD is empty or its file size exceeds memory size");
    if (offset > blobSize || fileSize > blobSize - offset)
      return elfError("executable_elf_invalid_segment",
                      "PT_LOAD file range escapes the executable blob");
    if (alignment != 0 && !llvm::isPowerOf2_64(alignment))
      return elfError("executable_elf_invalid_segment",
                      "PT_LOAD alignment is not zero or a power of two");
    if (alignment != 0 && header.p_vaddr % alignment != offset % alignment)
      return elfError("executable_elf_invalid_segment",
                      "PT_LOAD address and file offset are not congruent");
    if (header.p_vaddr > std::numeric_limits<std::uint64_t>::max() - memorySize)
      return elfError("executable_elf_invalid_segment",
                      "PT_LOAD virtual address range overflows");
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
    return elfError("executable_elf_missing_load_segment",
                    "ELF executable has no PT_LOAD segment");
  if (!llvm::any_of(result,
                    [](const auto &segment) { return segment.executable; }))
    return elfError("executable_elf_missing_executable_segment",
                    "ELF executable has no executable PT_LOAD segment");
  for (std::uint64_t ordinal = 0; ordinal != result.size(); ++ordinal)
    result[ordinal].ordinal = ordinal;
  return result;
}

llvm::Expected<const llvm::object::ELFObjectFileBase *>
parseExecutable(llvm::ArrayRef<std::uint8_t> bytes,
                std::unique_ptr<llvm::object::Binary> &owner) {
  if (bytes.empty())
    return elfError("executable_elf_invalid", "executable blob is empty");
  llvm::MemoryBufferRef buffer(
      llvm::StringRef(reinterpret_cast<const char *>(bytes.data()),
                      bytes.size()),
      "compiler-target.elf");
  auto object = llvm::object::ObjectFile::createObjectFile(buffer);
  if (!object)
    return elfError("executable_elf_invalid",
                    llvm::toString(object.takeError()));
  owner = std::move(*object);
  const auto *elf =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(owner.get());
  if (!elf || (elf->getEType() != llvm::ELF::ET_EXEC &&
               elf->getEType() != llvm::ELF::ET_DYN))
    return elfError("executable_elf_invalid",
                    "blob is not an ELF executable or PIE");
  return elf;
}

} // namespace

llvm::Error validateElfTarget(const llvm::object::ELFObjectFileBase &object,
                              const CompilerTargetBinding &target) {
  if (target.objectFormat() != CompilerObjectFormat::Elf)
    return elfError("executable_elf_target_mismatch",
                    "CompilerTargetBinding does not select ELF");
  if (object.getEMachine() != llvm::ELF::EM_RISCV)
    return elfError("executable_elf_target_mismatch",
                    "ELF machine is not RISC-V");
  const llvm::Triple targetTriple(target.targetTriple());
  if (object.getArch() != targetTriple.getArch())
    return elfError("executable_elf_target_mismatch",
                    "ELF class does not match CompilerTargetBinding triple");
  const unsigned flags = object.getPlatformFlags();
  if ((flags & llvm::ELF::EF_RISCV_FLOAT_ABI) !=
      expectedRiscVFloatAbi(target.backendAbi()))
    return elfError("executable_elf_target_mismatch",
                    "ELF floating-point ABI does not match target binding");
  const bool expectsEmbedded = target.backendAbi() == fabric::RiscVAbi::Ilp32e;
  if (((flags & llvm::ELF::EF_RISCV_RVE) != 0) != expectsEmbedded)
    return elfError("executable_elf_target_mismatch",
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
    return elfError("executable_elf_target_mismatch",
                    "ELF ISA attributes do not match target binding features");
  return llvm::Error::success();
}

llvm::Expected<std::vector<ExecutableLoadSegment>>
projectExecutableLoadSegments(const llvm::object::ELFObjectFileBase &object,
                              std::size_t blobSize) {
  if (const auto *value =
          llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(&object))
    return projectLoadSegments(*value, blobSize);
  if (const auto *value =
          llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(&object))
    return projectLoadSegments(*value, blobSize);
  if (const auto *value =
          llvm::dyn_cast<llvm::object::ELF32BEObjectFile>(&object))
    return projectLoadSegments(*value, blobSize);
  return projectLoadSegments(
      *llvm::cast<llvm::object::ELF64BEObjectFile>(&object), blobSize);
}

llvm::Expected<std::vector<ExecutableLoadSegment>>
projectCompilerTargetExecutableLoadSegments(
    llvm::ArrayRef<std::uint8_t> bytes, const CompilerTargetBinding &target) {
  std::unique_ptr<llvm::object::Binary> owner;
  auto elf = parseExecutable(bytes, owner);
  if (!elf)
    return elf.takeError();
  if (llvm::Error error = validateElfTarget(**elf, target))
    return std::move(error);
  return projectExecutableLoadSegments(**elf, bytes.size());
}

llvm::Expected<ExecutableLoadRange>
projectCompilerTargetExecutableLoadRange(llvm::ArrayRef<std::uint8_t> bytes,
                                         const CompilerTargetBinding &target) {
  auto segments = projectCompilerTargetExecutableLoadSegments(bytes, target);
  if (!segments)
    return segments.takeError();
  std::uint64_t begin = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t end = 0;
  for (const ExecutableLoadSegment &segment : *segments) {
    begin = std::min(begin, segment.virtualAddress);
    end = std::max(end, segment.virtualAddress + segment.memorySize);
  }
  return ExecutableLoadRange{begin, end};
}

} // namespace loom

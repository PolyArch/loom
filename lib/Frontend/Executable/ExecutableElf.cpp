#include "Frontend/Executable/ExecutableElf.h"

#include "Frontend/Executable/CompilerTargetBinding.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/TargetParser/RISCVISAInfo.h"
#include "llvm/TargetParser/Triple.h"

#include <string>
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

} // namespace loom

#include "Frontend/Executable/CompilerTargetLinker.h"

#include "Frontend/Executable/ExecutableElf.h"

#include "lld/Common/Driver.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

LLD_HAS_DRIVER(elf)

namespace loom {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "compiler_target_link_invalid: " + message);
}

llvm::Expected<llvm::StringRef>
elfEmulation(const CompilerTargetBinding &binding) {
  const llvm::Triple triple(binding.targetTriple());
  switch (triple.getArch()) {
  case llvm::Triple::riscv32:
    return llvm::StringRef("elf32lriscv");
  case llvm::Triple::riscv64:
    return llvm::StringRef("elf64lriscv");
  default:
    return invalid("target has no admitted static ELF emulation");
  }
}

llvm::Expected<llvm::SmallString<128>>
createWorkspaceFile(llvm::StringRef directory, llvm::StringRef model,
                    int &descriptor) {
  if (directory.empty())
    return invalid("temporary directory is empty");
  if (std::error_code error = llvm::sys::fs::create_directories(directory))
    return invalid("cannot create temporary directory: " + error.message());
  llvm::SmallString<128> pattern(directory);
  llvm::sys::path::append(pattern, model);
  llvm::SmallString<128> path;
  if (std::error_code error =
          llvm::sys::fs::createUniqueFile(pattern, descriptor, path))
    return invalid("cannot create temporary file: " + error.message());
  return path;
}

} // namespace

llvm::Expected<std::vector<std::uint8_t>> linkCompilerTargetExecutable(
    llvm::ArrayRef<std::uint8_t> objectBytes,
    const CompilerTargetBinding &binding, llvm::StringRef entrySymbol,
    std::uint64_t imageBase, const CompilerTargetLinkWorkspace &workspace) {
  if (objectBytes.empty())
    return invalid("target object is empty");
  if (entrySymbol.empty() || entrySymbol.contains('\0'))
    return invalid("entry symbol is empty or contains NUL");
  if (imageBase == 0 || imageBase % 4096 != 0)
    return invalid("image base is zero or not page-aligned");
  auto emulation = elfEmulation(binding);
  if (!emulation)
    return emulation.takeError();

  int objectDescriptor = -1;
  auto objectPath = createWorkspaceFile(
      workspace.temporaryDirectory, "loom-target-%%%%%%.o", objectDescriptor);
  if (!objectPath)
    return objectPath.takeError();
  llvm::FileRemover removeObject(*objectPath);
  {
    llvm::raw_fd_ostream output(objectDescriptor, true);
    output.write(reinterpret_cast<const char *>(objectBytes.data()),
                 objectBytes.size());
    output.close();
    if (output.has_error())
      return invalid("cannot write target object");
  }

  int executableDescriptor = -1;
  auto executablePath =
      createWorkspaceFile(workspace.temporaryDirectory,
                          "loom-target-%%%%%%.elf", executableDescriptor);
  if (!executablePath)
    return executablePath.takeError();
  if (std::error_code error = llvm::sys::fs::closeFile(executableDescriptor))
    return invalid("cannot close target executable placeholder: " +
                   error.message());
  llvm::FileRemover removeExecutable(*executablePath);

  const std::string entryArgument = "--entry=" + entrySymbol.str();
  const std::string imageBaseArgument =
      "--image-base=0x" + llvm::utohexstr(imageBase);
  const std::string outputPath = executablePath->str().str();
  const std::string inputPath = objectPath->str().str();
  const std::string emulationArgument = emulation->str();
  const std::vector<const char *> arguments{"ld.lld",
                                            "-m",
                                            emulationArgument.c_str(),
                                            entryArgument.c_str(),
                                            imageBaseArgument.c_str(),
                                            "--no-dynamic-linker",
                                            "--build-id=none",
                                            "--no-relax",
                                            "-z",
                                            "max-page-size=4096",
                                            "-o",
                                            outputPath.c_str(),
                                            inputPath.c_str()};
  llvm::SmallString<256> stdoutStorage;
  llvm::SmallString<1024> stderrStorage;
  llvm::raw_svector_ostream stdoutStream(stdoutStorage);
  llvm::raw_svector_ostream stderrStream(stderrStorage);
  const lld::Result result = lld::lldMain(arguments, stdoutStream, stderrStream,
                                          {{lld::Gnu, &lld::elf::link}});
  if (result.retCode != 0)
    return invalid("LLD failed: " + stderrStream.str());
  if (!result.canRunAgain)
    return invalid("LLD completed without preserving reentrant state");

  auto executable = llvm::MemoryBuffer::getFile(outputPath, false, false);
  if (!executable)
    return invalid("cannot read linked executable: " +
                   executable.getError().message());
  std::vector<std::uint8_t> resultBytes(
      (*executable)->getBuffer().bytes_begin(),
      (*executable)->getBuffer().bytes_end());
  auto range = projectCompilerTargetExecutableLoadRange(resultBytes, binding);
  if (!range)
    return range.takeError();
  if (range->begin != imageBase)
    return invalid("linked PT_LOAD range does not begin at the image base");
  return resultBytes;
}

} // namespace loom

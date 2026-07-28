#include "Frontend/Payload/AcceleratorFinalLink.h"
#include "Frontend/Payload/PayloadCarrier.h"
#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<relocatable object>"),
                                         llvm::cl::init(""));

llvm::cl::opt<std::string>
    resolutionInput("resolution",
                    llvm::cl::desc("pinned LLD resolution report"),
                    llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<std::string>
    linkedBitcodeInput("linked-bitcode",
                       llvm::cl::desc("pinned LLD pre-code-generation bitcode"),
                       llvm::cl::value_desc("filename"), llvm::cl::init(""));

llvm::cl::opt<std::string>
    bitcodeOutput("bitcode-output",
                  llvm::cl::desc("write the validated normalized LLVM bitcode"),
                  llvm::cl::value_desc("filename"), llvm::cl::init(""));

int report(llvm::Error error) {
  llvm::errs() << "loom-payload: " << llvm::toString(std::move(error)) << '\n';
  return 1;
}

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
readInput(llvm::StringRef path, llvm::StringRef role) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> input =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!input)
    return llvm::createStringError(input.getError(), "cannot read %s '%s'",
                                   role.str().c_str(), path.str().c_str());
  return std::move(*input);
}

llvm::Error writeBitcode(llvm::StringRef bytes) {
  if (bitcodeOutput.empty())
    return llvm::Error::success();
  std::error_code error;
  llvm::ToolOutputFile output(bitcodeOutput, error, llvm::sys::fs::OF_None);
  if (error)
    return llvm::createStringError(error, "cannot open bitcode output '%s'",
                                   bitcodeOutput.c_str());
  output.os().write(bytes.data(), bytes.size());
  output.keep();
  return llvm::Error::success();
}

llvm::Error verifyObjectCarrier() {
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> object =
      readInput(inputFilename, "relocatable object");
  if (!object)
    return object.takeError();

  llvm::Expected<std::optional<std::vector<std::uint8_t>>> carrier =
      loom::readRelocatablePayloadCarrier((*object)->getMemBufferRef());
  if (!carrier)
    return carrier.takeError();
  if (!*carrier)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "relocatable object has no accelerator payload carrier");

  llvm::Expected<loom::RelocatableAcceleratorPayload> payload =
      loom::decodeRelocatableAcceleratorPayload(
          loom::RelocatableAcceleratorPayload::artifactSchema, **carrier);
  if (!payload)
    return payload.takeError();
  const llvm::ArrayRef<std::uint8_t> bitcode = payload->normalizedBitcode();
  return writeBitcode(llvm::StringRef(
      reinterpret_cast<const char *>(bitcode.data()), bitcode.size()));
}

llvm::Error verifyFinalLink() {
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> resolution =
      readInput(resolutionInput, "LLD resolution report");
  if (!resolution)
    return resolution.takeError();
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> linked =
      readInput(linkedBitcodeInput, "LLD linked bitcode");
  if (!linked)
    return linked.takeError();

  llvm::LLVMContext context;
  llvm::Expected<std::unique_ptr<llvm::Module>> module =
      loom::importLldAcceleratorFinalLink((*resolution)->getMemBufferRef(),
                                          (*linked)->getMemBufferRef(),
                                          context);
  if (!module)
    return module.takeError();
  if (!*module)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "selected final link has no accelerator payload cohort");

  llvm::SmallVector<char, 0> bytes;
  llvm::raw_svector_ostream stream(bytes);
  llvm::WriteBitcodeToFile(**module, stream,
                           /*ShouldPreserveUseListOrder=*/false,
                           /*Index=*/nullptr, /*GenerateHash=*/false);
  return writeBitcode(llvm::StringRef(bytes.data(), bytes.size()));
}

llvm::Error run() {
  const bool finalLinkMode =
      !resolutionInput.empty() || !linkedBitcodeInput.empty();
  if (finalLinkMode) {
    if (resolutionInput.empty() || linkedBitcodeInput.empty())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "final-link mode requires both --resolution and --linked-bitcode");
    if (!inputFilename.empty())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "final-link mode does not accept a relocatable object operand");
    return verifyFinalLink();
  }
  if (inputFilename.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "object mode requires one relocatable object operand");
  return verifyObjectCarrier();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Loom relocatable payload and final-link "
                                    "verifier\n");
  if (llvm::Error error = run())
    return report(std::move(error));
  return 0;
}

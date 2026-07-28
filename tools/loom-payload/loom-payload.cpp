#include "Frontend/Payload/PayloadCarrier.h"
#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

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
                                         llvm::cl::Required);

llvm::cl::opt<std::string>
    bitcodeOutput("bitcode-output",
                  llvm::cl::desc("write the validated normalized LLVM bitcode"),
                  llvm::cl::value_desc("filename"), llvm::cl::init(""));

int report(llvm::Error error) {
  llvm::errs() << "loom-payload: " << llvm::toString(std::move(error)) << '\n';
  return 1;
}

llvm::Error run() {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> object =
      llvm::MemoryBuffer::getFile(inputFilename, /*IsText=*/false);
  if (!object)
    return llvm::createStringError(object.getError(),
                                   "cannot read relocatable object '%s'",
                                   inputFilename.c_str());

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
  if (bitcodeOutput.empty())
    return llvm::Error::success();

  std::error_code error;
  llvm::ToolOutputFile output(bitcodeOutput, error, llvm::sys::fs::OF_None);
  if (error)
    return llvm::createStringError(error, "cannot open bitcode output '%s'",
                                   bitcodeOutput.c_str());
  const llvm::ArrayRef<std::uint8_t> bitcode = payload->normalizedBitcode();
  output.os().write(reinterpret_cast<const char *>(bitcode.data()),
                    bitcode.size());
  output.keep();
  return llvm::Error::success();
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM init(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Loom relocatable payload verifier\n");
  if (llvm::Error error = run())
    return report(std::move(error));
  return 0;
}

#include "Frontend/Payload/AcceleratorFinalLink.h"
#include "Common/Artifact.h"
#include "Common/ResolvedConfig.h"
#include "Frontend/Payload/FrontendConfigView.h"
#include "Frontend/Payload/PayloadCarrier.h"
#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace loom;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << "\n";
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(const char *test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

template <typename T>
std::string rejectionMessage(const char *test, llvm::Expected<T> value) {
  if (value)
    fail(test, "a selected input that must fail closed was accepted");
  return llvm::toString(value.takeError());
}

void requireMentions(const char *test, const std::string &message,
                     llvm::StringRef marker, const std::string &what) {
  if (!llvm::StringRef(message).contains(marker))
    fail(test, what + ": " + message);
}

const llvm::Triple &hostTriple() {
  static const llvm::Triple triple(
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple()));
  return triple;
}

std::string assemblyWithDataLayout(const llvm::Triple &triple,
                                   llvm::StringRef dataLayout,
                                   const std::string &body) {
  return "target datalayout = \"" + dataLayout.str() +
         "\"\ntarget triple = \"" + triple.str() + "\"\n\n" + body;
}

std::string assemblyFor(const llvm::Triple &triple, const std::string &body) {
  return assemblyWithDataLayout(triple, triple.computeDataLayout(), body);
}

std::string reorderedDataLayout(llvm::StringRef dataLayout) {
  llvm::SmallVector<llvm::StringRef, 16> components;
  dataLayout.split(components, '-');
  std::reverse(components.begin(), components.end());
  return llvm::join(components, "-");
}

std::string definitionOf(llvm::StringRef function) {
  return "define i32 @" + function.str() +
         "(i32 %value) {\n"
         "entry:\n"
         "  %doubled = add nsw i32 %value, %value\n"
         "  ret i32 %doubled\n"
         "}\n";
}

std::string translationUnitAssembly(llvm::StringRef function) {
  return assemblyFor(hostTriple(), definitionOf(function));
}

std::unique_ptr<llvm::Module> parseAssembly(const char *test,
                                            llvm::StringRef assembly,
                                            llvm::LLVMContext &context) {
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(assembly, diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, "test module assembly did not parse: " + message);
  }
  return module;
}

std::vector<std::uint8_t> bitcodeOf(const llvm::Module &module) {
  llvm::SmallVector<char, 0> buffer;
  llvm::raw_svector_ostream stream(buffer);
  llvm::WriteBitcodeToFile(module, stream);
  return std::vector<std::uint8_t>(buffer.begin(), buffer.end());
}

RelocatableAcceleratorPayload payloadFor(const char *test,
                                         llvm::StringRef assembly) {
  llvm::LLVMContext context;
  const std::unique_ptr<llvm::Module> module =
      parseAssembly(test, assembly, context);
  return takeExpected(
      test, RelocatableAcceleratorPayload::create(
                bitcodeOf(*module),
                projectResolvedFrontendConfigView(defaultResolvedConfig())));
}

std::vector<std::uint8_t> payloadBytesFor(const char *test,
                                          llvm::StringRef assembly) {
  const CanonicalSemanticBytes canonical =
      payloadFor(test, assembly).canonicalSemanticBytes();
  return std::vector<std::uint8_t>(canonical.bytes().begin(),
                                   canonical.bytes().end());
}

std::vector<char> emitObject(const char *test, llvm::StringRef assembly,
                             llvm::ArrayRef<std::uint8_t> canonicalBytes) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = parseAssembly(test, assembly, context);
  if (!canonicalBytes.empty())
    embedRelocatablePayloadCarrier(*module, canonicalBytes);

  const llvm::Triple triple = module->getTargetTriple();
  std::string lookupError;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, lookupError);
  if (!target)
    fail(test, "no registered target for " + triple.str() + ": " + lookupError);
  const std::unique_ptr<llvm::TargetMachine> machine(
      target->createTargetMachine(triple, "generic", "", llvm::TargetOptions(),
                                  llvm::Reloc::Model::PIC_));
  require(test, machine != nullptr, "the target created no target machine");

  llvm::SmallVector<char, 0> object;
  llvm::raw_svector_ostream stream(object);
  llvm::legacy::PassManager passes;
  require(test,
          !machine->addPassesToEmitFile(passes, stream, nullptr,
                                        llvm::CodeGenFileType::ObjectFile),
          "the target cannot emit relocatable objects");
  passes.run(*module);
  return std::vector<char>(object.begin(), object.end());
}

class TempTree {
public:
  explicit TempTree(const char *test) : test_(test) {
    llvm::SmallString<128> path;
    requireSuccess(test_,
                   llvm::errorCodeToError(llvm::sys::fs::createUniqueDirectory(
                       "loom-final-link", path)));
    root_ = path.str().str();
  }

  ~TempTree() { llvm::sys::fs::remove_directories(root_); }

  TempTree(const TempTree &) = delete;
  TempTree &operator=(const TempTree &) = delete;

  std::string writeFile(llvm::StringRef name, llvm::ArrayRef<char> bytes) {
    const std::string path = root_ + "/" + name.str();
    std::error_code code;
    llvm::raw_fd_ostream stream(path, code);
    requireSuccess(test_, llvm::errorCodeToError(code));
    stream.write(bytes.data(), bytes.size());
    stream.close();
    requireSuccess(test_, llvm::errorCodeToError(stream.error()));
    return path;
  }

  std::string writeText(llvm::StringRef name, llvm::StringRef contents) {
    return writeFile(name, llvm::ArrayRef(contents.data(), contents.size()));
  }

  std::string path(llvm::StringRef name) const {
    return root_ + "/" + name.str();
  }

  std::string writeArchive(
      llvm::StringRef name,
      llvm::ArrayRef<std::pair<std::string, std::vector<char>>> entries) {
    std::vector<std::unique_ptr<llvm::MemoryBuffer>> buffers;
    std::vector<llvm::NewArchiveMember> members;
    for (const auto &entry : entries) {
      buffers.push_back(llvm::MemoryBuffer::getMemBufferCopy(
          llvm::StringRef(entry.second.data(), entry.second.size()),
          entry.first));
      members.emplace_back(buffers.back()->getMemBufferRef());
    }

    const std::string path = root_ + "/" + name.str();
    requireSuccess(test_,
                   llvm::writeArchive(path, members,
                                      llvm::SymtabWritingMode::NormalSymtab,
                                      llvm::object::Archive::K_GNU,
                                      /*Deterministic=*/true, /*Thin=*/false));
    return path;
  }

private:
  const char *test_;
  std::string root_;
};

void runProgram(const char *test, llvm::StringRef program,
                llvm::ArrayRef<llvm::StringRef> arguments) {
  std::string error;
  bool executionFailed = false;
  const int result = llvm::sys::ExecuteAndWait(
      program, arguments, std::nullopt, {}, /*SecondsToWait=*/60,
      /*MemoryLimit=*/1024, &error, &executionFailed);
  require(test, !executionFailed,
          "failed to execute " + program.str() + ": " + error);
  require(test, result == 0,
          program.str() + " returned " + std::to_string(result) +
              (error.empty() ? "" : ": " + error));
}

std::unique_ptr<llvm::MemoryBuffer> readBuffer(const char *test,
                                               llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  requireSuccess(test, llvm::errorCodeToError(buffer.getError()));
  return std::move(*buffer);
}

std::vector<char> readBytes(const char *test, llvm::StringRef path) {
  const std::unique_ptr<llvm::MemoryBuffer> buffer = readBuffer(test, path);
  return std::vector<char>(buffer->getBufferStart(), buffer->getBufferEnd());
}

std::string compileFatObject(const char *test, TempTree &tree,
                             llvm::StringRef stem, llvm::StringRef source) {
  const std::string sourceName = stem.str() + ".c";
  const std::string objectName = stem.str() + ".o";
  const std::string sourcePath = tree.writeText(sourceName, source);
  const std::string objectPath = tree.path(objectName);
  const std::string compiler = LOOM_TEST_CC_PATH;
  const llvm::SmallVector<llvm::StringRef, 10> arguments = {
      compiler, "-O1",      "-flto=full", "-ffat-lto-objects",
      "-c",     sourcePath, "-o",         objectPath};
  runProgram(test, compiler, arguments);
  return objectPath;
}

void pinnedLldSelectionProducesUniqueLinkedModule() {
  TempTree tree(__func__);
  const std::string mainObject =
      compileFatObject(__func__, tree, "main",
                       "extern int loom_first(int);\n"
                       "extern int loom_selected(int);\n"
                       "__attribute__((noinline, used))\n"
                       "int loom_entry(int value) {\n"
                       "  return loom_selected(loom_first(value));\n"
                       "}\n");
  const std::string firstObject =
      compileFatObject(__func__, tree, "first",
                       "__attribute__((noinline))\n"
                       "int loom_first(int value) { return value + 1; }\n");
  const std::string selectedObject =
      compileFatObject(__func__, tree, "selected",
                       "__attribute__((noinline))\n"
                       "int loom_selected(int value) { return value * 3; }\n");
  const std::string unselectedObject = compileFatObject(
      __func__, tree, "unselected",
      "__attribute__((noinline))\n"
      "int loom_unselected(int value) { return value - 7; }\n");
  const std::string archive = tree.writeArchive(
      "libmembers.a",
      {{"selected.o", readBytes(__func__, selectedObject)},
       {"unselected.o", readBytes(__func__, unselectedObject)}});

  const std::string linkedBitcode = tree.path("linked.bc");
  const std::string linker = LOOM_TEST_LLD_PATH;
  const llvm::SmallVector<llvm::StringRef, 12> arguments = {
      linker,
      "-r",
      "--fat-lto-objects",
      "--lto-emit-llvm",
      "--save-temps=resolution",
      "-o",
      linkedBitcode,
      mainObject,
      firstObject,
      archive};
  runProgram(__func__, linker, arguments);

  const std::unique_ptr<llvm::MemoryBuffer> resolution =
      readBuffer(__func__, linkedBitcode + ".resolution.txt");
  const std::unique_ptr<llvm::MemoryBuffer> linked =
      readBuffer(__func__, linkedBitcode);
  llvm::LLVMContext context;
  const std::unique_ptr<llvm::Module> module = takeExpected(
      __func__,
      importLldAcceleratorFinalLink(resolution->getMemBufferRef(),
                                    linked->getMemBufferRef(), context));
  require(__func__, module != nullptr,
          "the selected payload cohort produced no linked module");
  require(__func__, module->getFunction("loom_entry") != nullptr,
          "the first selected object did not reach the linked module");
  require(__func__, module->getFunction("loom_first") != nullptr,
          "the second selected object did not reach the linked module");
  require(__func__, module->getFunction("loom_selected") != nullptr,
          "the selected archive member did not reach the linked module");
  require(__func__, module->getFunction("loom_unselected") == nullptr,
          "an unselected archive member entered the linked module");
  require(
      __func__,
      !takeExpected(__func__, hasGeneratedRelocatablePayloadCarrier(*module)),
      "the Part 1 hand-off retained a relocatable carrier projection");
}

std::string selectedInputReport(llvm::ArrayRef<std::string> paths) {
  std::string report;
  for (const std::string &path : paths) {
    report += path;
    report += '\n';
  }
  return report;
}

std::vector<std::uint8_t>
linkedCarrierBitcode(const char *test, llvm::StringRef dataLayout,
                     llvm::ArrayRef<std::vector<std::uint8_t>> carriers) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module = parseAssembly(
      test, assemblyWithDataLayout(hostTriple(), dataLayout, ""), context);
  for (const std::vector<std::uint8_t> &carrier : carriers)
    embedRelocatablePayloadCarrier(*module, carrier);
  return bitcodeOf(*module);
}

void equivalentDataLayoutSpellingsLinkWithDistinctPayloadIdentities() {
  const std::string exactLayout = hostTriple().computeDataLayout();
  const std::string equivalentLayout = reorderedDataLayout(exactLayout);
  require(__func__, equivalentLayout != exactLayout,
          "the equivalent layout fixture did not change spelling");

  const std::string body =
      "define linkonce_odr i32 @loom_equivalent_layout(i32 %value) {\n"
      "entry:\n"
      "  ret i32 %value\n"
      "}\n";
  const std::string exactAssembly =
      assemblyWithDataLayout(hostTriple(), exactLayout, body);
  const std::string equivalentAssembly =
      assemblyWithDataLayout(hostTriple(), equivalentLayout, body);
  const RelocatableAcceleratorPayload exactPayload =
      payloadFor(__func__, exactAssembly);
  const RelocatableAcceleratorPayload equivalentPayload =
      payloadFor(__func__, equivalentAssembly);
  require(__func__, exactPayload.identity() != equivalentPayload.identity(),
          "equivalent layout spellings collapsed payload identity");
  require(__func__,
          exactPayload.abiCompatibilityKey() ==
              equivalentPayload.abiCompatibilityKey(),
          "layout spelling entered the ABI compatibility key");

  TempTree tree(__func__);
  const std::string exactObject = tree.writeFile(
      "exact.o",
      emitObject(__func__, translationUnitAssembly("loom_exact_carrier"),
                 payloadBytesFor(__func__, exactAssembly)));
  const std::string equivalentObject = tree.writeFile(
      "equivalent.o",
      emitObject(__func__, translationUnitAssembly("loom_equivalent_carrier"),
                 payloadBytesFor(__func__, equivalentAssembly)));
  const std::vector<std::uint8_t> exactBytes =
      payloadBytesFor(__func__, exactAssembly);
  const std::vector<std::uint8_t> equivalentBytes =
      payloadBytesFor(__func__, equivalentAssembly);
  const std::string report =
      selectedInputReport({exactObject, equivalentObject});
  const std::vector<std::uint8_t> linkedBytes = linkedCarrierBitcode(
      __func__, exactLayout, {exactBytes, equivalentBytes});

  llvm::LLVMContext context;
  const std::unique_ptr<llvm::Module> linked = takeExpected(
      __func__,
      importLldAcceleratorFinalLink(
          llvm::MemoryBufferRef(report, "layout.resolution.txt"),
          llvm::MemoryBufferRef(llvm::StringRef(reinterpret_cast<const char *>(
                                                    linkedBytes.data()),
                                                linkedBytes.size()),
                                "layout.bc"),
          context));
  require(__func__, linked != nullptr,
          "structurally equivalent layouts produced no linked module");
}

void structurallyDifferentDataLayoutsFailFinalLink() {
  const std::string exactLayout = hostTriple().computeDataLayout();
  const std::string differentLayout = exactLayout + "-i7:8:8";
  const std::string exactAssembly = assemblyWithDataLayout(
      hostTriple(), exactLayout, definitionOf("loom_layout_exact"));
  const std::string differentAssembly = assemblyWithDataLayout(
      hostTriple(), differentLayout, definitionOf("loom_layout_different"));
  const RelocatableAcceleratorPayload exactPayload =
      payloadFor(__func__, exactAssembly);
  const RelocatableAcceleratorPayload differentPayload =
      payloadFor(__func__, differentAssembly);
  require(__func__,
          exactPayload.abiCompatibilityKey() ==
              differentPayload.abiCompatibilityKey(),
          "a structural layout mismatch changed the necessary ABI key");

  TempTree tree(__func__);
  const std::string exactObject = tree.writeFile(
      "exact.o",
      emitObject(__func__, translationUnitAssembly("loom_exact_carrier"),
                 payloadBytesFor(__func__, exactAssembly)));
  const std::string differentObject = tree.writeFile(
      "different.o",
      emitObject(__func__, translationUnitAssembly("loom_different_carrier"),
                 payloadBytesFor(__func__, differentAssembly)));
  const std::string report =
      selectedInputReport({exactObject, differentObject});

  llvm::LLVMContext context;
  const std::string message = rejectionMessage(
      __func__, importLldAcceleratorFinalLink(
                    llvm::MemoryBufferRef(report, "layout.resolution.txt"),
                    llvm::MemoryBufferRef("", "layout.bc"), context));
  requireMentions(__func__, message, "selected_payload_incompatible",
                  "a structural layout mismatch was not typed");
  requireMentions(__func__, message, "data layout",
                  "a structural layout mismatch did not name the field");
}

} // namespace

int main() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  pinnedLldSelectionProducesUniqueLinkedModule();
  equivalentDataLayoutSpellingsLinkWithDistinctPayloadIdentities();
  structurallyDifferentDataLayoutsFailFinalLink();
  return 0;
}

#include "Frontend/Payload/AcceleratorFinalLink.h"
#include "Common/Artifact.h"
#include "Common/ResolvedConfig.h"
#include "Frontend/Payload/FrontendConfigView.h"
#include "Frontend/Payload/PayloadCarrier.h"
#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#include <cstddef>
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

template <typename T> T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

void requireSuccess(const char *test, llvm::Error error) {
  if (error)
    fail(test, llvm::toString(std::move(error)));
}

/// Returns the typed rejection message, failing the test when the input that
/// must fail closed was accepted instead.
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

/// The host target the payload modules and the emitted relocatable objects
/// share. Deriving both from the pinned provider keeps the test modules exactly
/// as canonical as normalization requires.
const llvm::Triple &hostTriple() {
  static const llvm::Triple triple(
      llvm::Triple::normalize(llvm::sys::getDefaultTargetTriple()));
  return triple;
}

/// A second real target differing from the host in exactly one canonical triple
/// component, so two payloads land in different ABI cohorts while both stay
/// buildable by the one code generator this build has.
const llvm::Triple &foreignCohortTriple() {
  static const llvm::Triple triple = [] {
    llvm::Triple other = hostTriple();
    other.setEnvironment(hostTriple().getEnvironment() == llvm::Triple::Musl
                             ? llvm::Triple::GNU
                             : llvm::Triple::Musl);
    return llvm::Triple(llvm::Triple::normalize(other.str()));
  }();
  return triple;
}

std::string assemblyFor(const llvm::Triple &triple, const std::string &body) {
  return "target datalayout = \"" + triple.computeDataLayout() +
         "\"\ntarget triple = \"" + triple.str() + "\"\n\n" + body;
}

/// One externally visible definition, so a linked module states which selected
/// inputs were collected into it.
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

/// The complete canonical payload bytes production payload creation derives for
/// one translation unit.
std::vector<std::uint8_t> payloadBytesFor(const char *test,
                                          llvm::StringRef assembly) {
  llvm::LLVMContext context;
  const std::unique_ptr<llvm::Module> module =
      parseAssembly(test, assembly, context);
  const RelocatableAcceleratorPayload payload = takeExpected(
      test, RelocatableAcceleratorPayload::create(
                bitcodeOf(*module),
                projectResolvedFrontendConfigView(defaultResolvedConfig())));
  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();
  return std::vector<std::uint8_t>(canonical.bytes().begin(),
                                   canonical.bytes().end());
}

/// Emits the ordinary relocatable object an ordinary compiler produces for one
/// translation unit. Empty payload bytes emit an ordinary payload-free object.
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

std::vector<char> emitCarrierObject(const char *test,
                                    llvm::StringRef assembly) {
  return emitObject(test, assembly, payloadBytesFor(test, assembly));
}

std::size_t definedFunctions(const llvm::Module &module) {
  std::size_t defined = 0;
  for (const llvm::Function &function : module)
    if (!function.isDeclaration())
      ++defined;
  return defined;
}

/// A directory holding the real objects and archives one case links.
class TempTree {
public:
  explicit TempTree(const char *test) : test_(test) {
    llvm::SmallString<128> path;
    requireSuccess(test_, llvm::errorCodeToError(
                              llvm::sys::fs::createUniqueDirectory(
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

  /// Writes one real `ar` archive through the pinned LLVM archive writer.
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

/// Stands in for the ordinary linker reporting which archive member it selected.
/// A member is identified by its exact child offset because archive member
/// names are not unique.
std::uint64_t archiveMemberOffset(const char *test, llvm::StringRef archivePath,
                                  llvm::StringRef memberName) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile(archivePath, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  requireSuccess(test, llvm::errorCodeToError(file.getError()));
  const std::unique_ptr<llvm::object::Archive> archive = takeExpected(
      test, llvm::object::Archive::create((*file)->getMemBufferRef()));

  std::optional<std::uint64_t> offset;
  llvm::Error iteration = llvm::Error::success();
  for (const llvm::object::Archive::Child &child :
       archive->children(iteration)) {
    if (takeExpected(test, child.getName()) != memberName)
      continue;
    offset = child.getChildOffset();
    break;
  }
  requireSuccess(test, std::move(iteration));
  if (!offset)
    fail(test, "the archive has no member named " + memberName.str());
  return *offset;
}

/// Two selected ordinary objects plus one archive in which the ordinary linker
/// selected exactly one of two payload-bearing members. The linked module must
/// hold every selected translation unit and nothing from the unselected member.
void selectedObjectsAndArchiveMemberLinkOnce() {
  const char *test = __func__;
  TempTree tree(test);

  const std::string firstObject = tree.writeFile(
      "first.o", emitCarrierObject(test, translationUnitAssembly("loom_first")));
  const std::string secondObject = tree.writeFile(
      "second.o",
      emitCarrierObject(test, translationUnitAssembly("loom_second")));
  const std::string archive = tree.writeArchive(
      "libloom.a",
      {{"selected.o",
        emitCarrierObject(test, translationUnitAssembly("loom_selected"))},
       {"unselected.o",
        emitCarrierObject(test, translationUnitAssembly("loom_unselected"))}});

  LinkerSelectedInputs selected;
  selected.selectObjectFile(firstObject);
  selected.selectObjectFile(secondObject);
  selected.selectArchiveMember(
      archive, archiveMemberOffset(test, archive, "selected.o"));

  llvm::LLVMContext context;
  const std::unique_ptr<llvm::Module> linked =
      takeExpected(test, linkSelectedAcceleratorPayloads(selected, context));
  require(test, linked != nullptr,
          "the selected payload cohort produced no linked module");
  require(test, linked->getFunction("loom_first") != nullptr,
          "the first selected object did not reach the linked module");
  require(test, linked->getFunction("loom_second") != nullptr,
          "the second selected object did not reach the linked module");
  require(test, linked->getFunction("loom_selected") != nullptr,
          "the selected archive member did not reach the linked module");
  require(test, linked->getFunction("loom_unselected") == nullptr,
          "an archive member the ordinary linker did not select entered the "
          "linked module");
}

/// The complement selection over the same archive. Which member is excluded
/// follows the ordinary linker's result alone, not archive order or content.
void archiveMemberExclusionFollowsSelection() {
  const char *test = __func__;
  TempTree tree(test);

  const std::string archive = tree.writeArchive(
      "libloom.a",
      {{"first.o",
        emitCarrierObject(test, translationUnitAssembly("loom_first"))},
       {"second.o",
        emitCarrierObject(test, translationUnitAssembly("loom_second"))}});

  LinkerSelectedInputs selected;
  selected.selectArchiveMember(archive,
                               archiveMemberOffset(test, archive, "second.o"));

  llvm::LLVMContext context;
  const std::unique_ptr<llvm::Module> linked =
      takeExpected(test, linkSelectedAcceleratorPayloads(selected, context));
  require(test, linked != nullptr,
          "the selected archive member produced no linked module");
  require(test, linked->getFunction("loom_second") != nullptr,
          "the selected archive member did not reach the linked module");
  require(test, linked->getFunction("loom_first") == nullptr,
          "the unselected archive member entered the linked module");
}

/// A real object carries the complete canonical payload bytes back out
/// unchanged, and the carrier itself contributes nothing to payload identity.
void carrierRoundTripPreservesPayloadIdentity() {
  const char *test = __func__;
  const std::string assembly = translationUnitAssembly("loom_round_trip");
  const std::vector<std::uint8_t> canonicalBytes =
      payloadBytesFor(test, assembly);

  TempTree tree(test);
  const std::string objectPath =
      tree.writeFile("round_trip.o", emitObject(test, assembly, canonicalBytes));
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> object =
      llvm::MemoryBuffer::getFile(objectPath, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  requireSuccess(test, llvm::errorCodeToError(object.getError()));

  const std::optional<std::vector<std::uint8_t>> carried = takeExpected(
      test, readRelocatablePayloadCarrier((*object)->getMemBufferRef()));
  require(test, carried.has_value(), "the emitted object carries no payload");
  require(test, *carried == canonicalBytes,
          "the object carrier did not return the complete canonical payload "
          "bytes unchanged");

  const RelocatableAcceleratorPayload decoded = takeExpected(
      test, decodeRelocatableAcceleratorPayload(
                RelocatableAcceleratorPayload::artifactSchema, *carried));
  const RelocatableAcceleratorPayload direct = takeExpected(
      test, decodeRelocatableAcceleratorPayload(
                RelocatableAcceleratorPayload::artifactSchema, canonicalBytes));
  require(test, decoded.identity() == direct.identity(),
          "riding inside an object changed the payload's ArtifactIdentity");
}

/// Objects without a payload stay valid link inputs, and a selection carrying
/// no payload at all implies no accelerator compilation.
void objectsWithoutPayloadStayValid() {
  const char *test = __func__;
  TempTree tree(test);

  const std::string plainObject = tree.writeFile(
      "plain.o", emitObject(test, translationUnitAssembly("loom_plain"), {}));
  const std::string carrierObject = tree.writeFile(
      "carrier.o",
      emitCarrierObject(test, translationUnitAssembly("loom_carrier")));

  llvm::LLVMContext context;
  LinkerSelectedInputs onlyPlain;
  onlyPlain.selectObjectFile(plainObject);
  require(test,
          takeExpected(test,
                       linkSelectedAcceleratorPayloads(onlyPlain, context)) ==
              nullptr,
          "a selection with no payload implied accelerator compilation");

  const LinkerSelectedInputs nothingSelected;
  require(test,
          takeExpected(test, linkSelectedAcceleratorPayloads(nothingSelected,
                                                             context)) ==
              nullptr,
          "an empty selection implied accelerator compilation");

  LinkerSelectedInputs mixed;
  mixed.selectObjectFile(plainObject);
  mixed.selectObjectFile(carrierObject);
  const std::unique_ptr<llvm::Module> linked =
      takeExpected(test, linkSelectedAcceleratorPayloads(mixed, context));
  require(test, linked != nullptr,
          "a payload-free input suppressed the payload beside it");
  require(test, linked->getFunction("loom_carrier") != nullptr,
          "the payload-bearing object did not reach the linked module");
  require(test, linked->getFunction("loom_plain") == nullptr,
          "a payload-free object contributed to the linked module");
}

/// A selected member whose payload is malformed names that exact member and
/// publishes nothing, instead of being quietly dropped from the cohort.
void malformedSelectedMemberFailsWithAttribution() {
  const char *test = __func__;
  TempTree tree(test);

  const std::string brokenAssembly = translationUnitAssembly("loom_broken");
  std::vector<std::uint8_t> brokenBytes = payloadBytesFor(test, brokenAssembly);
  brokenBytes.back() ^= 0xFF;

  const std::string archive = tree.writeArchive(
      "libloom.a",
      {{"intact.o",
        emitCarrierObject(test, translationUnitAssembly("loom_intact"))},
       {"broken.o", emitObject(test, brokenAssembly, brokenBytes)}});
  const std::uint64_t intactOffset =
      archiveMemberOffset(test, archive, "intact.o");
  const std::uint64_t brokenOffset =
      archiveMemberOffset(test, archive, "broken.o");

  llvm::LLVMContext context;
  LinkerSelectedInputs withBroken;
  withBroken.selectArchiveMember(archive, intactOffset);
  withBroken.selectArchiveMember(archive, brokenOffset);
  const std::string message = rejectionMessage(
      test, linkSelectedAcceleratorPayloads(withBroken, context));
  requireMentions(test, message, "selected_payload_rejected",
                  "a malformed selected payload was not typed");
  requireMentions(test, message, "broken.o",
                  "a malformed selected payload did not name its member");

  // The intact member on its own still links, so the rejection came from the
  // malformed member rather than from the archive as a whole.
  LinkerSelectedInputs onlyIntact;
  onlyIntact.selectArchiveMember(archive, intactOffset);
  const std::unique_ptr<llvm::Module> linked =
      takeExpected(test, linkSelectedAcceleratorPayloads(onlyIntact, context));
  require(test, linked != nullptr && linked->getFunction("loom_intact"),
          "the intact archive member did not link on its own");
}

/// Payloads whose raw cohort fields disagree are a typed link error naming the
/// member that does not join, with no implicit merge or precedence rule.
void incompatibleRawCohortFieldsFail() {
  const char *test = __func__;
  require(test, foreignCohortTriple().str() != hostTriple().str(),
          "the test could not derive a second canonical target triple");

  TempTree tree(test);
  const std::string hostObject = tree.writeFile(
      "host.o", emitCarrierObject(test, translationUnitAssembly("loom_host")));
  const std::string foreignObject = tree.writeFile(
      "foreign.o",
      emitCarrierObject(test, assemblyFor(foreignCohortTriple(),
                                          definitionOf("loom_foreign"))));

  LinkerSelectedInputs selected;
  selected.selectObjectFile(hostObject);
  selected.selectObjectFile(foreignObject);

  llvm::LLVMContext context;
  const std::string message =
      rejectionMessage(test, linkSelectedAcceleratorPayloads(selected, context));
  requireMentions(test, message, "selected_payload_incompatible",
                  "a raw cohort disagreement was not typed");
  requireMentions(test, message, "foreign.o",
                  "a raw cohort disagreement did not name its member");
  requireMentions(test, message, "the canonical target triple",
                  "a raw cohort disagreement did not name the field");
}

/// COMDAT and ODR resolution across selected payloads is the pinned LLVM
/// Linker's decision. The duplicate definition merges instead of being renamed
/// or duplicated, and nothing here restates that rule.
void llvmLinkerResolvesComdatOdr() {
  const char *test = __func__;

  // Both translation units define and use the same COMDAT definition, which is
  // exactly the duplicate an ODR link has to resolve.
  const auto sharingUnit = [](llvm::StringRef function) {
    return assemblyFor(hostTriple(),
                       "$shared = comdat any\n"
                       "\n"
                       "define linkonce_odr i32 @shared() comdat {\n"
                       "entry:\n"
                       "  ret i32 11\n"
                       "}\n"
                       "\n"
                       "define i32 @" +
                           function.str() +
                           "(i32 %value) {\n"
                           "entry:\n"
                           "  %constant = call i32 @shared()\n"
                           "  %sum = add nsw i32 %value, %constant\n"
                           "  ret i32 %sum\n"
                           "}\n");
  };

  TempTree tree(test);
  const std::string firstObject =
      tree.writeFile("first.o", emitCarrierObject(test, sharingUnit("loom_first")));
  const std::string secondObject = tree.writeFile(
      "second.o", emitCarrierObject(test, sharingUnit("loom_second")));

  LinkerSelectedInputs selected;
  selected.selectObjectFile(firstObject);
  selected.selectObjectFile(secondObject);

  llvm::LLVMContext context;
  const std::unique_ptr<llvm::Module> linked =
      takeExpected(test, linkSelectedAcceleratorPayloads(selected, context));
  require(test, linked != nullptr, "the ODR cohort produced no linked module");
  require(test, linked->getFunction("shared") != nullptr,
          "the shared COMDAT definition did not survive the link");
  require(test, definedFunctions(*linked) == 3,
          "the linked module does not hold exactly the two unique definitions "
          "and one merged COMDAT definition");
}

/// Module-flag validation is the pinned LLVM Linker's too. Loom surfaces its
/// rejection as a typed error naming the member that failed to link.
void llvmLinkerRejectsConflictingModuleFlags() {
  const char *test = __func__;
  const auto withWcharSize = [](llvm::StringRef function, int size) {
    return assemblyFor(hostTriple(),
                       definitionOf(function) +
                           "\n!llvm.module.flags = !{!0}\n"
                           "!0 = !{i32 1, !\"wchar_size\", i32 " +
                           std::to_string(size) + "}\n");
  };

  TempTree tree(test);
  const std::string firstObject = tree.writeFile(
      "first.o", emitCarrierObject(test, withWcharSize("loom_first", 4)));
  const std::string secondObject = tree.writeFile(
      "second.o", emitCarrierObject(test, withWcharSize("loom_second", 2)));

  LinkerSelectedInputs selected;
  selected.selectObjectFile(firstObject);
  selected.selectObjectFile(secondObject);

  llvm::LLVMContext context;
  const std::string message =
      rejectionMessage(test, linkSelectedAcceleratorPayloads(selected, context));
  requireMentions(test, message, "accelerator_link_failed",
                  "an LLVM Linker rejection was not typed");
  requireMentions(test, message, "second.o",
                  "an LLVM Linker rejection did not name its member");
  requireMentions(test, message, "wchar_size",
                  "an LLVM Linker rejection dropped the reason LLVM reported");
}

} // namespace

int main() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  selectedObjectsAndArchiveMemberLinkOnce();
  archiveMemberExclusionFollowsSelection();
  carrierRoundTripPreservesPayloadIdentity();
  objectsWithoutPayloadStayValid();
  malformedSelectedMemberFailsWithAttribution();
  incompatibleRawCohortFieldsFail();
  llvmLinkerResolvesComdatOdr();
  llvmLinkerRejectsConflictingModuleFlags();
  return 0;
}

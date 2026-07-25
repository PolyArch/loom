#include "Frontend/Payload/AcceleratorFinalLink.h"

#include "Frontend/Payload/LlvmModuleNormalization.h"
#include "Frontend/Payload/PayloadCarrier.h"
#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DiagnosticHandler.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <cstddef>
#include <memory>
#include <utility>

namespace loom {
namespace {

llvm::Error rejected(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// Names the exact object or archive member a rejection came from, so a driver
/// can point at the selected input that failed instead of reporting that some
/// accelerator payload somewhere was bad.
llvm::Error attributed(llvm::StringRef carrier, llvm::Error cause) {
  return rejected("selected_payload_rejected: '" + carrier +
                  "': " + llvm::toString(std::move(cause)));
}

/// The complete canonical payload bytes one selected carrier delivers, together
/// with the exact name every diagnostic about it must use. No bytes means the
/// selected input simply carries no payload, which stays valid.
struct CarrierContents {
  std::string carrier;
  std::optional<std::vector<std::uint8_t>> canonicalBytes;
};

llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
openSelectedInput(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> file =
      llvm::MemoryBuffer::getFile(path, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (!file)
    return rejected("selected_input_unreadable: cannot read selected input '" +
                    path + "': " + file.getError().message());
  return std::move(*file);
}

llvm::Expected<CarrierContents> readObjectFileCarrier(llvm::StringRef path) {
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> file =
      openSelectedInput(path);
  if (!file)
    return file.takeError();

  llvm::Expected<std::optional<std::vector<std::uint8_t>>> carried =
      readRelocatablePayloadCarrier((*file)->getMemBufferRef());
  if (!carried)
    return attributed(path, carried.takeError());
  return CarrierContents{path.str(), std::move(*carried)};
}

llvm::Expected<CarrierContents>
readArchiveMemberCarrier(llvm::StringRef archivePath,
                         std::uint64_t memberChildOffset) {
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> file =
      openSelectedInput(archivePath);
  if (!file)
    return file.takeError();
  llvm::Expected<std::unique_ptr<llvm::object::Archive>> archive =
      llvm::object::Archive::create((*file)->getMemBufferRef());
  if (!archive)
    return rejected("selected_input_unreadable: cannot read selected archive '" +
                    archivePath + "': " + llvm::toString(archive.takeError()));

  // The ordinary linker's selection is the only thing that names a member here.
  // Every other child is stepped over without its contents being looked at, so
  // a member the ordinary linker did not select cannot reach the link.
  std::optional<llvm::object::Archive::Child> selectedMember;
  llvm::Error iteration = llvm::Error::success();
  for (const llvm::object::Archive::Child &child :
       (*archive)->children(iteration)) {
    if (child.getChildOffset() != memberChildOffset)
      continue;
    selectedMember = child;
    break;
  }
  if (iteration)
    return rejected("selected_input_unreadable: cannot read selected archive '" +
                    archivePath + "': " + llvm::toString(std::move(iteration)));
  if (!selectedMember)
    return rejected("selected_input_unreadable: selected archive '" +
                    archivePath + "' has no member at child offset " +
                    llvm::Twine(memberChildOffset));

  llvm::Expected<llvm::StringRef> memberName = selectedMember->getName();
  if (!memberName)
    return rejected("selected_input_unreadable: selected archive '" +
                    archivePath + "' has no readable name for its member at "
                                  "child offset " +
                    llvm::Twine(memberChildOffset) + ": " +
                    llvm::toString(memberName.takeError()));
  const std::string carrier =
      archivePath.str() + "(" + memberName->str() + ")";

  llvm::Expected<llvm::MemoryBufferRef> member =
      selectedMember->getMemoryBufferRef();
  if (!member)
    return attributed(carrier, member.takeError());
  llvm::Expected<std::optional<std::vector<std::uint8_t>>> carried =
      readRelocatablePayloadCarrier(*member);
  if (!carried)
    return attributed(carrier, carried.takeError());
  return CarrierContents{carrier, std::move(*carried)};
}

llvm::Expected<CarrierContents>
readSelectedCarrier(const LinkerSelectedInputs::Selection &selection) {
  if (selection.memberChildOffset)
    return readArchiveMemberCarrier(selection.path,
                                    *selection.memberChildOffset);
  return readObjectFileCarrier(selection.path);
}

/// Routes the diagnostics the pinned LLVM Linker owns into one typed error.
/// Symbol, COMDAT, ODR, and module-flag rejections are its decisions; capturing
/// them keeps them exactly as reported rather than restating them here.
class CapturedLinkDiagnostics {
public:
  CapturedLinkDiagnostics(llvm::LLVMContext &context, std::string &report)
      : context_(context),
        previousCallback_(context.getDiagnosticHandlerCallBack()),
        previousContext_(context.getDiagnosticContext()) {
    context_.setDiagnosticHandlerCallBack(append, &report);
  }

  ~CapturedLinkDiagnostics() {
    context_.setDiagnosticHandlerCallBack(previousCallback_, previousContext_);
  }

  CapturedLinkDiagnostics(const CapturedLinkDiagnostics &) = delete;
  CapturedLinkDiagnostics &operator=(const CapturedLinkDiagnostics &) = delete;

private:
  static void append(const llvm::DiagnosticInfo *info, void *report) {
    if (info->getSeverity() != llvm::DS_Error)
      return;
    auto &sink = *static_cast<std::string *>(report);
    llvm::raw_string_ostream stream(sink);
    if (!sink.empty())
      stream << "; ";
    llvm::DiagnosticPrinterRawOStream printer(stream);
    info->print(printer);
  }

  llvm::LLVMContext &context_;
  llvm::DiagnosticHandler::DiagnosticHandlerTy previousCallback_;
  void *previousContext_;
};

/// One validated payload and the selected carrier that delivered it.
struct CollectedPayload {
  std::string carrier;
  RelocatableAcceleratorPayload payload;
};

/// The identifier of the one module a final link produces. It names the linked
/// program, not any translation unit that entered it.
constexpr llvm::StringLiteral linkedModuleIdentifier =
    "loom.accelerator.final_link";

} // namespace

void LinkerSelectedInputs::selectObjectFile(llvm::StringRef objectPath) {
  selections_.push_back(Selection{objectPath.str(), std::nullopt});
}

void LinkerSelectedInputs::selectArchiveMember(
    llvm::StringRef archivePath, std::uint64_t memberChildOffset) {
  selections_.push_back(Selection{archivePath.str(), memberChildOffset});
}

llvm::Expected<std::unique_ptr<llvm::Module>>
linkSelectedAcceleratorPayloads(const LinkerSelectedInputs &selected,
                                llvm::LLVMContext &context) {
  std::vector<CollectedPayload> collected;
  for (const LinkerSelectedInputs::Selection &selection :
       selected.selections()) {
    llvm::Expected<CarrierContents> contents = readSelectedCarrier(selection);
    if (!contents)
      return contents.takeError();

    // A selected input without a payload is an ordinary external or
    // InstructionCore-only link input and stays valid.
    if (!contents->canonicalBytes)
      continue;

    // A carrier delivers the complete canonical payload bytes and nothing else,
    // so the payload family this build supports is the schema the decoder is
    // held to. Bytes that are not exactly a valid payload of that schema are
    // rejected below rather than read under some other reading.
    llvm::Expected<RelocatableAcceleratorPayload> payload =
        decodeRelocatableAcceleratorPayload(
            RelocatableAcceleratorPayload::artifactSchema,
            *contents->canonicalBytes);
    if (!payload)
      return attributed(contents->carrier, payload.takeError());
    collected.push_back(
        CollectedPayload{std::move(contents->carrier), std::move(*payload)});
  }

  // No selected member carries a payload, so no accelerator compilation is
  // implied. That is a complete and valid result, not a failure.
  if (collected.empty())
    return std::unique_ptr<llvm::Module>();

  // The raw cohort preflight runs entirely through the production comparison,
  // so no ABI-key formula or view rule is restated here.
  const CollectedPayload &cohort = collected.front();
  for (std::size_t index = 1; index < collected.size(); ++index) {
    const CollectedPayload &member = collected[index];
    if (llvm::Error error = requireRelocatablePayloadCompatibility(
            cohort.payload, member.payload))
      return rejected("selected_payload_incompatible: '" + member.carrier +
                      "' does not join the cohort established by '" +
                      cohort.carrier + "': " + llvm::toString(std::move(error)));
  }

  auto linked = std::make_unique<llvm::Module>(linkedModuleIdentifier, context);
  linked->setTargetTriple(llvm::Triple(cohort.payload.targetTriple()));
  linked->setDataLayout(cohort.payload.dataLayout());

  for (const CollectedPayload &member : collected) {
    llvm::Expected<std::unique_ptr<llvm::Module>> module =
        parseCompleteLlvmModule(member.payload.normalizedBitcode(), context);
    if (!module)
      return attributed(member.carrier, module.takeError());

    std::string report;
    const CapturedLinkDiagnostics diagnostics(context, report);
    if (llvm::Linker::linkModules(*linked, std::move(*module)))
      return rejected(
          "accelerator_link_failed: the pinned LLVM Linker rejected '" +
          member.carrier + "': " + report);
  }

  std::string verifierReport;
  llvm::raw_string_ostream verifierStream(verifierReport);
  if (llvm::verifyModule(*linked, &verifierStream))
    return rejected("linked_module_invalid: " + verifierReport);
  return linked;
}

} // namespace loom

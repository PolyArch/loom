#include "Frontend/Payload/AcceleratorFinalLink.h"

#include "Frontend/Payload/LlvmModuleNormalization.h"
#include "Frontend/Payload/PayloadCarrier.h"
#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Module.h"
#include "llvm/LTO/Config.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/Caching.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <map>
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
llvm::Error attributed(llvm::StringRef label, llvm::Error cause) {
  return rejected("selected_payload_rejected: '" + label +
                  "': " + llvm::toString(std::move(cause)));
}

/// The complete canonical payload bytes one selected carrier delivers, together
/// with the label every diagnostic about it must use. The label is for reading,
/// never for identity. No bytes means the selected input simply carries no
/// payload, which stays valid.
struct CarrierContents {
  std::string label;
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
                         const llvm::object::Archive::Child &member) {
  llvm::Expected<llvm::StringRef> memberName = member.getName();
  if (!memberName)
    return rejected("selected_input_unreadable: selected archive '" +
                    archivePath +
                    "' has no readable name for its member at "
                    "child offset " +
                    llvm::Twine(member.getChildOffset()) + ": " +
                    llvm::toString(memberName.takeError()));
  // Same-named members are legal, so the child offset is what makes the label
  // identify one exact member; the name only makes it readable.
  const std::string label = archivePath.str() + "(" + memberName->str() +
                            " at child offset " +
                            std::to_string(member.getChildOffset()) + ")";

  llvm::Expected<llvm::MemoryBufferRef> buffer = member.getMemoryBufferRef();
  if (!buffer)
    return attributed(label, buffer.takeError());
  llvm::Expected<std::optional<std::vector<std::uint8_t>>> carried =
      readRelocatablePayloadCarrier(*buffer);
  if (!carried)
    return attributed(label, carried.takeError());
  return CarrierContents{label, std::move(*carried)};
}

/// Reads every member one archive was selected from, opening and traversing
/// that archive exactly once no matter how many of its members were selected.
///
/// The traversal indexes only the child offsets the ordinary linker selected,
/// so an unselected member is stepped over and its contents are never read.
llvm::Error
readArchiveCarriers(llvm::StringRef archivePath,
                    llvm::ArrayRef<LinkerSelectedInputs::Selection> selections,
                    llvm::ArrayRef<std::size_t> selectedIndices,
                    std::vector<CarrierContents> &carriers) {
  llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>> file =
      openSelectedInput(archivePath);
  if (!file)
    return file.takeError();
  llvm::Expected<std::unique_ptr<llvm::object::Archive>> archive =
      llvm::object::Archive::create((*file)->getMemBufferRef());
  if (!archive)
    return rejected(
        "selected_input_unreadable: cannot read selected archive '" +
        archivePath + "': " + llvm::toString(archive.takeError()));

  llvm::DenseSet<std::uint64_t> selectedOffsets;
  for (std::size_t index : selectedIndices)
    selectedOffsets.insert(*selections[index].memberChildOffset);

  std::map<std::uint64_t, llvm::object::Archive::Child> selectedMembers;
  llvm::Error iteration = llvm::Error::success();
  for (const llvm::object::Archive::Child &child :
       (*archive)->children(iteration))
    if (selectedOffsets.contains(child.getChildOffset()))
      selectedMembers.emplace(child.getChildOffset(), child);
  if (iteration)
    return rejected(
        "selected_input_unreadable: cannot read selected archive '" +
        archivePath + "': " + llvm::toString(std::move(iteration)));

  for (std::size_t index : selectedIndices) {
    const std::uint64_t offset = *selections[index].memberChildOffset;
    const auto member = selectedMembers.find(offset);
    if (member == selectedMembers.end())
      return rejected("selected_input_unreadable: selected archive '" +
                      archivePath + "' has no member at child offset " +
                      llvm::Twine(offset));
    llvm::Expected<CarrierContents> contents =
        readArchiveMemberCarrier(archivePath, member->second);
    if (!contents)
      return contents.takeError();
    carriers[index] = std::move(*contents);
  }
  return llvm::Error::success();
}

/// Reads the carrier of every selected input, keeping the ordinary linker's
/// selection order.
llvm::Expected<std::vector<CarrierContents>>
readSelectedCarriers(const LinkerSelectedInputs &selected) {
  const llvm::ArrayRef<LinkerSelectedInputs::Selection> selections =
      selected.selections();
  std::vector<CarrierContents> carriers(selections.size());

  llvm::MapVector<llvm::StringRef, llvm::SmallVector<std::size_t, 4>>
      archiveSelections;
  for (std::size_t index = 0; index < selections.size(); ++index) {
    if (selections[index].memberChildOffset) {
      archiveSelections[selections[index].path].push_back(index);
      continue;
    }
    llvm::Expected<CarrierContents> contents =
        readObjectFileCarrier(selections[index].path);
    if (!contents)
      return contents.takeError();
    carriers[index] = std::move(*contents);
  }

  for (const auto &archive : archiveSelections)
    if (llvm::Error error = readArchiveCarriers(archive.first, selections,
                                                archive.second, carriers))
      return std::move(error);
  return carriers;
}

/// One validated payload, the selection that identifies the input it came from,
/// and the label diagnostics about it use.
struct CollectedPayload {
  std::size_t selectionIndex;
  std::string label;
  RelocatableAcceleratorPayload payload;
};

llvm::MemoryBufferRef payloadBuffer(const CollectedPayload &member) {
  const llvm::ArrayRef<std::uint8_t> bitcode =
      member.payload.normalizedBitcode();
  return llvm::MemoryBufferRef(
      llvm::StringRef(reinterpret_cast<const char *>(bitcode.data()),
                      bitcode.size()),
      member.label);
}

std::vector<std::uint8_t> writeTransportBitcode(const llvm::Module &module) {
  llvm::SmallVector<char, 0> written;
  llvm::raw_svector_ostream stream(written);
  llvm::WriteBitcodeToFile(module, stream);
  return std::vector<std::uint8_t>(written.begin(), written.end());
}

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
                                LinkerSymbolResolver resolveSymbol,
                                llvm::LLVMContext &context) {
  llvm::Expected<std::vector<CarrierContents>> carriers =
      readSelectedCarriers(selected);
  if (!carriers)
    return carriers.takeError();

  std::vector<CollectedPayload> collected;
  for (std::size_t index = 0; index < carriers->size(); ++index) {
    CarrierContents &contents = (*carriers)[index];
    // A selected input without a payload is an ordinary external or
    // InstructionCore-only link input and stays valid.
    if (!contents.canonicalBytes)
      continue;

    llvm::Expected<RelocatableAcceleratorPayload> payload =
        decodeRelocatableAcceleratorPayload(
            RelocatableAcceleratorPayload::artifactSchema,
            *contents.canonicalBytes);
    if (!payload)
      return attributed(contents.label, payload.takeError());
    collected.push_back(CollectedPayload{index, std::move(contents.label),
                                         std::move(*payload)});
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
      return rejected("selected_payload_incompatible: '" + member.label +
                      "' does not join the cohort established by '" +
                      cohort.label + "': " + llvm::toString(std::move(error)));
  }

  // Diagnostics LTO reports are kept exactly as reported: they state decisions
  // this code does not make.
  std::string report;
  llvm::lto::Config config;
  config.DiagHandler = [&report](const llvm::DiagnosticInfo &info) {
    llvm::raw_string_ostream stream(report);
    if (!report.empty())
      stream << "; ";
    llvm::DiagnosticPrinterRawOStream printer(stream);
    info.print(printer);
  };

  // The last hook before code generation yields the unique post-LTO module.
  // Refusing to continue there is how the pinned pipeline is asked for a linked
  // module instead of native objects.
  std::vector<std::uint8_t> linkedBitcode;
  config.PreCodeGenModuleHook = [&linkedBitcode](unsigned,
                                                 const llvm::Module &module) {
    linkedBitcode = writeTransportBitcode(module);
    return false;
  };

  llvm::lto::LTO lto(std::move(config));
  for (const CollectedPayload &member : collected) {
    llvm::Expected<std::unique_ptr<llvm::lto::InputFile>> input =
        llvm::lto::InputFile::create(payloadBuffer(member));
    if (!input)
      return attributed(member.label, input.takeError());

    // Every symbol of this input is resolved by the ordinary linker, in the
    // order the pinned LTO API enumerates them. The exact selection identifies
    // the input, so same-named archive members are resolved separately.
    const LinkerSelectedInputs::Selection &selection =
        selected.selections()[member.selectionIndex];
    std::vector<llvm::lto::SymbolResolution> resolutions;
    resolutions.reserve((*input)->symbols().size());
    for (const llvm::lto::InputFile::Symbol &symbol : (*input)->symbols())
      resolutions.push_back(resolveSymbol(selection, symbol));

    // Admitting an input is where LTO merges it, so a rejection here is a link
    // failure attributed to that member rather than a bad payload.
    if (llvm::Error error = lto.add(std::move(*input), resolutions))
      return rejected("accelerator_link_failed: the pinned LTO pipeline "
                      "rejected '" +
                      member.label + "': " + llvm::toString(std::move(error)));
  }

  const llvm::AddStreamFn refuseNativeOutput = [](unsigned, const llvm::Twine &)
      -> llvm::Expected<std::unique_ptr<llvm::CachedFileStream>> {
    return rejected("accelerator_link_unexpected_codegen: the final link asked "
                    "for native output instead of a linked module");
  };
  if (llvm::Error error = lto.run(refuseNativeOutput))
    return rejected("accelerator_link_failed: the pinned LTO pipeline rejected "
                    "the selected payload cohort: " +
                    llvm::toString(std::move(error)) +
                    (report.empty() ? "" : "; " + report));
  if (linkedBitcode.empty())
    return rejected("accelerator_link_incomplete: the pinned LTO pipeline "
                    "produced no linked module for the selected payload "
                    "cohort" +
                    (report.empty() ? "" : ": " + report));

  return parseCompleteLlvmModule(linkedBitcode, context);
}

} // namespace loom

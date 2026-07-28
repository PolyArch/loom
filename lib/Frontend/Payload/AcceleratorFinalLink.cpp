#include "Frontend/Payload/AcceleratorFinalLink.h"

#include "Frontend/Payload/LlvmModuleNormalization.h"
#include "Frontend/Payload/PayloadCarrier.h"
#include "Frontend/Payload/RelocatableAcceleratorPayload.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

struct SelectedInput {
  std::string path;
  std::optional<std::uint64_t> memberChildOffset;
};

using SelectedInputs = std::vector<SelectedInput>;

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
llvm::Error readArchiveCarriers(llvm::StringRef archivePath,
                                llvm::ArrayRef<SelectedInput> selections,
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
readSelectedCarriers(llvm::ArrayRef<SelectedInput> selections) {
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

/// One validated payload and the label diagnostics about it use.
struct CollectedPayload {
  std::string label;
  RelocatableAcceleratorPayload payload;
};

llvm::Expected<std::vector<CollectedPayload>>
collectSelectedPayloads(llvm::ArrayRef<SelectedInput> selected) {
  llvm::Expected<std::vector<CarrierContents>> carriers =
      readSelectedCarriers(selected);
  if (!carriers)
    return carriers.takeError();

  std::vector<CollectedPayload> collected;
  for (std::size_t index = 0; index < carriers->size(); ++index) {
    CarrierContents &contents = (*carriers)[index];
    if (!contents.canonicalBytes)
      continue;
    llvm::Expected<RelocatableAcceleratorPayload> payload =
        decodeRelocatableAcceleratorPayload(
            RelocatableAcceleratorPayload::artifactSchema,
            *contents.canonicalBytes);
    if (!payload)
      return attributed(contents.label, payload.takeError());
    collected.push_back(
        CollectedPayload{std::move(contents.label), std::move(*payload)});
  }
  return collected;
}

llvm::Error
requireCompatiblePayloadCohort(llvm::ArrayRef<CollectedPayload> collected) {
  if (collected.empty())
    return llvm::Error::success();
  const CollectedPayload &cohort = collected.front();
  for (const CollectedPayload &member : collected.drop_front())
    if (llvm::Error error = requireRelocatablePayloadCompatibility(
            cohort.payload, member.payload))
      return rejected("selected_payload_incompatible: '" + member.label +
                      "' does not join the cohort established by '" +
                      cohort.label + "': " + llvm::toString(std::move(error)));
  return llvm::Error::success();
}

llvm::Expected<SelectedInput> decodeLldInputName(llvm::StringRef inputName) {
  std::vector<SelectedInput> candidates;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> direct =
      llvm::MemoryBuffer::getFile(inputName, /*IsText=*/false,
                                  /*RequiresNullTerminator=*/false);
  if (direct &&
      llvm::identify_magic((*direct)->getBuffer()) != llvm::file_magic::archive)
    candidates.push_back({inputName.str(), std::nullopt});

  for (std::size_t open = inputName.find('('); open != llvm::StringRef::npos;
       open = inputName.find('(', open + 1)) {
    if (!inputName.ends_with(")"))
      break;
    const llvm::StringRef archivePath = inputName.take_front(open);
    const llvm::StringRef memberDescription =
        inputName.slice(open + 1, inputName.size() - 1);
    const std::size_t at = memberDescription.rfind(" at ");
    if (archivePath.empty() || at == llvm::StringRef::npos)
      continue;
    std::uint64_t offset = 0;
    if (memberDescription.drop_front(at + 4).getAsInteger(10, offset))
      continue;

    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> archiveFile =
        llvm::MemoryBuffer::getFile(archivePath, /*IsText=*/false,
                                    /*RequiresNullTerminator=*/false);
    if (!archiveFile)
      continue;
    llvm::Expected<std::unique_ptr<llvm::object::Archive>> archive =
        llvm::object::Archive::create((*archiveFile)->getMemBufferRef());
    if (!archive) {
      llvm::consumeError(archive.takeError());
      continue;
    }

    llvm::Error iteration = llvm::Error::success();
    for (const llvm::object::Archive::Child &child :
         (*archive)->children(iteration)) {
      if (child.getChildOffset() != offset)
        continue;
      llvm::Expected<llvm::MemoryBufferRef> member = child.getMemoryBufferRef();
      if (!member) {
        llvm::consumeError(member.takeError());
        break;
      }
      const llvm::StringRef memberName =
          llvm::sys::path::filename(member->getBufferIdentifier());
      const std::string reconstructed = archivePath.str() + "(" +
                                        memberName.str() + " at " +
                                        std::to_string(offset) + ")";
      if (reconstructed == inputName)
        candidates.push_back({archivePath.str(), offset});
      break;
    }
    if (iteration)
      llvm::consumeError(std::move(iteration));
  }

  if (candidates.empty())
    return rejected("lld_resolution_input_unreadable: cannot resolve selected "
                    "LTO input '" +
                    inputName + "' to an object or exact archive child");
  if (candidates.size() != 1)
    return rejected("lld_resolution_input_ambiguous: selected LTO input '" +
                    inputName + "' has more than one filesystem meaning");
  return std::move(candidates.front());
}

llvm::Error validateLldResolutionRow(llvm::StringRef inputName,
                                     llvm::StringRef row) {
  const std::string prefix = "-r=" + inputName.str() + ",";
  if (!row.starts_with(prefix))
    return rejected("lld_resolution_malformed: symbol row does not belong to "
                    "its preceding input '" +
                    inputName + "'");
  const llvm::StringRef payload = row.drop_front(prefix.size());
  const std::size_t comma = payload.rfind(',');
  if (comma == llvm::StringRef::npos || comma == 0)
    return rejected("lld_resolution_malformed: symbol row for '" + inputName +
                    "' has no symbol or flag field");
  const llvm::StringRef flags = payload.drop_front(comma + 1);
  constexpr llvm::StringLiteral canonicalFlags = "plxr";
  std::size_t previous = 0;
  bool first = true;
  for (char flag : flags) {
    const std::size_t position = canonicalFlags.find(flag);
    if (position == llvm::StringRef::npos || (!first && position <= previous))
      return rejected("lld_resolution_malformed: symbol row for '" + inputName +
                      "' has unknown, duplicate, or noncanonical "
                      "flags");
    previous = position;
    first = false;
  }
  return llvm::Error::success();
}

llvm::Expected<SelectedInputs>
parseLldSelectedInputs(llvm::MemoryBufferRef resolution) {
  SelectedInputs selected;
  std::optional<std::string> currentInput;
  llvm::SmallVector<llvm::StringRef, 32> lines;
  resolution.getBuffer().split(lines, '\n', /*MaxSplit=*/-1,
                               /*KeepEmpty=*/true);
  for (std::size_t index = 0; index < lines.size(); ++index) {
    llvm::StringRef line = lines[index];
    line.consume_back("\r");
    if (line.empty()) {
      if (index + 1 == lines.size())
        continue;
      return rejected("lld_resolution_malformed: empty line before end of "
                      "resolution report");
    }
    if (line.starts_with("-r=")) {
      if (!currentInput)
        return rejected("lld_resolution_malformed: symbol row precedes its "
                        "selected input");
      if (llvm::Error error = validateLldResolutionRow(*currentInput, line))
        return std::move(error);
      continue;
    }

    llvm::Expected<SelectedInput> selection = decodeLldInputName(line);
    if (!selection)
      return selection.takeError();
    selected.push_back(std::move(*selection));
    currentInput = line.str();
  }
  return selected;
}

std::vector<std::string>
canonicalCarrierMultiset(llvm::ArrayRef<CollectedPayload> payloads) {
  std::vector<std::string> carriers;
  carriers.reserve(payloads.size());
  for (const CollectedPayload &payload : payloads) {
    const CanonicalSemanticBytes bytes =
        payload.payload.canonicalSemanticBytes();
    carriers.emplace_back(reinterpret_cast<const char *>(bytes.bytes().data()),
                          bytes.bytes().size());
  }
  std::sort(carriers.begin(), carriers.end());
  return carriers;
}

std::vector<std::string> canonicalCarrierMultiset(
    llvm::ArrayRef<std::vector<std::uint8_t>> carrierBytes) {
  std::vector<std::string> carriers;
  carriers.reserve(carrierBytes.size());
  for (const std::vector<std::uint8_t> &bytes : carrierBytes)
    carriers.emplace_back(reinterpret_cast<const char *>(bytes.data()),
                          bytes.size());
  std::sort(carriers.begin(), carriers.end());
  return carriers;
}

llvm::Error
verifyLinkedModuleCohort(const llvm::Module &module,
                         llvm::ArrayRef<CollectedPayload> collected) {
  if (collected.empty())
    return llvm::Error::success();
  const RelocatableAcceleratorPayload &cohort = collected.front().payload;
  if (llvm::Triple::normalize(module.getTargetTriple().str()) !=
      cohort.targetTriple())
    return rejected("lld_linked_module_mismatch: target triple differs from "
                    "the selected payload cohort");
  llvm::Expected<llvm::DataLayout> linkedLayout =
      llvm::DataLayout::parse(module.getDataLayoutStr());
  llvm::Expected<llvm::DataLayout> cohortLayout =
      llvm::DataLayout::parse(cohort.dataLayout());
  if (!linkedLayout || !cohortLayout) {
    if (!linkedLayout)
      return rejected("lld_linked_module_mismatch: invalid linked data "
                      "layout: " +
                      llvm::toString(linkedLayout.takeError()));
    return rejected("lld_linked_module_mismatch: invalid cohort data layout: " +
                    llvm::toString(cohortLayout.takeError()));
  }
  if (*linkedLayout != *cohortLayout)
    return rejected("lld_linked_module_mismatch: data layout differs from the "
                    "selected payload cohort");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<std::unique_ptr<llvm::Module>>
importLldAcceleratorFinalLink(llvm::MemoryBufferRef resolution,
                              llvm::MemoryBufferRef linkedBitcode,
                              llvm::LLVMContext &context) {
  llvm::Expected<SelectedInputs> selected = parseLldSelectedInputs(resolution);
  if (!selected)
    return selected.takeError();
  llvm::Expected<std::vector<CollectedPayload>> collectedOrError =
      collectSelectedPayloads(*selected);
  if (!collectedOrError)
    return collectedOrError.takeError();
  std::vector<CollectedPayload> collected = std::move(*collectedOrError);
  if (llvm::Error error = requireCompatiblePayloadCohort(collected))
    return std::move(error);

  const llvm::StringRef linkedBytes = linkedBitcode.getBuffer();
  const llvm::ArrayRef<std::uint8_t> bitcode(
      reinterpret_cast<const std::uint8_t *>(linkedBytes.data()),
      linkedBytes.size());
  llvm::Expected<std::unique_ptr<llvm::Module>> module =
      parseCompleteLlvmModule(bitcode, context);
  if (!module)
    return rejected("lld_linked_module_invalid: " +
                    llvm::toString(module.takeError()));
  if (llvm::Error error = verifyLinkedModuleCohort(**module, collected))
    return std::move(error);

  llvm::Expected<std::vector<std::vector<std::uint8_t>>> linkedCarriers =
      removeGeneratedRelocatablePayloadCarriers(**module);
  if (!linkedCarriers)
    return linkedCarriers.takeError();
  if (canonicalCarrierMultiset(collected) !=
      canonicalCarrierMultiset(*linkedCarriers))
    return rejected("lld_linked_module_stale: linked carrier projections do "
                    "not equal the selected payload cohort");

  if (collected.empty())
    return std::unique_ptr<llvm::Module>();
  std::string verifierReport;
  llvm::raw_string_ostream verifierStream(verifierReport);
  if (llvm::verifyModule(**module, &verifierStream))
    return rejected("lld_linked_module_invalid_after_carrier_removal: " +
                    verifierReport);
  return std::move(*module);
}

} // namespace loom

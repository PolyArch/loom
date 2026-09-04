#include "Hardware/RTL/BlockGateNetlist.h"

#include "Common/ArtifactStore.h"
#include "Common/ArtifactText.h"
#include "Common/BlobStore.h"
#include "Hardware/Implementation/RepresentationIndex.h"
#include "ImplementationPlatform/ImplementationPlatform.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "block_gate_netlist_invalid: " + message);
}

llvm::ArrayRef<std::uint8_t> bytesOf(llvm::StringRef text) {
  return {reinterpret_cast<const std::uint8_t *>(text.data()), text.size()};
}

llvm::Expected<std::string> encode(const BlockGateNetlistDraft &draft) {
  auto representation =
      serializeImplementationRepresentationRootJson(draft.representation);
  if (!representation)
    return representation.takeError();
  auto root = llvm::json::parse(*representation);
  if (!root)
    return root.takeError();
  std::string result;
  llvm::raw_string_ostream stream(result);
  llvm::json::OStream json(stream);
  json.object([&] {
    json.attributeObject("source", [&] {
      writeArtifactRootReferenceJsonFields(json, draft.source);
    });
    json.attributeObject("platform", [&] {
      writeArtifactRootReferenceJsonFields(json, draft.implementationPlatform);
    });
    json.attribute("corner", formatArtifactLocalPayloadHex(
                                 platform::encodeTechnologyCornerPayload(
                                     draft.corner.entity)));
    json.attribute("standard_cell_contract", draft.standardCellContract);
    json.attribute("standard_cell_library",
                   formatExternalFileFingerprint(draft.standardCellLibrary));
    json.attribute("representation", *root);
  });
  stream.flush();
  return result;
}

llvm::Expected<BlockGateNetlistDraft> decode(llvm::StringRef text) {
  auto value = llvm::json::parse(text);
  if (!value)
    return value.takeError();
  const auto *object = value->getAsObject();
  if (!object)
    return invalid("root is not an object");
  const auto *sourceObject = object->getObject("source");
  const auto *platformObject = object->getObject("platform");
  const auto *representationObject = object->getObject("representation");
  const auto cornerText = object->getString("corner");
  const auto contract = object->getString("standard_cell_contract");
  const auto libraryText = object->getString("standard_cell_library");
  if (!sourceObject || !platformObject || !representationObject ||
      !cornerText || !contract || !libraryText)
    return invalid("source, target, library or representation is absent");
  auto source = parseArtifactRootReferenceJson(*sourceObject);
  if (!source)
    return source.takeError();
  auto target = parseArtifactRootReferenceJson(*platformObject);
  if (!target)
    return target.takeError();
  auto cornerBytes = parseArtifactLocalPayloadHex(*cornerText);
  if (!cornerBytes)
    return cornerBytes.takeError();
  auto corner = platform::decodeTechnologyCornerPayload(*cornerBytes);
  if (!corner)
    return corner.takeError();
  auto library = parseExternalFileFingerprint(*libraryText);
  if (!library)
    return library.takeError();
  auto representation =
      parseImplementationRepresentationRootJsonValue(*representationObject);
  if (!representation)
    return representation.takeError();
  platform::TechnologyCornerRef cornerRef{target->artifact, *corner};
  return BlockGateNetlistDraft{std::move(*source), std::move(*target),
                               cornerRef,          contract->str(),
                               *library,           std::move(*representation)};
}

llvm::Error validate(const BlockGateNetlistDraft &draft,
                     const ExternalImplementationContractCatalog &contracts,
                     const ArtifactStore &artifacts, const BlobStore &blobs) {
  auto source = importRtlBlockSource(draft.source, artifacts, blobs);
  if (!source)
    return source.takeError();
  auto target = platform::importImplementationPlatform(
      draft.implementationPlatform, artifacts);
  if (!target)
    return target.takeError();
  if (!std::holds_alternative<platform::AsicTarget>(
          target->platform().target()) ||
      draft.corner.artifact != target->reference().artifact ||
      !target->platform().findTechnologyCorner(draft.corner.entity))
    return invalid("corner does not belong to the exact ASIC platform");
  const auto &root = draft.representation;
  if (root.variant != RepresentationRootVariant::GateNetlist || root.stage ||
      root.formatRef.kind() !=
          RepresentationFormatKind::StructuralVerilogGateNetlist ||
      root.top.kind != RepresentationObjectKind::Module ||
      root.top.canonicalName != source->top())
    return invalid("representation is not the exact source block gate netlist");
  auto contract = contracts.find(draft.standardCellContract);
  if (!contract || contract->inputSlots.size() != 1)
    return invalid("standard-cell contract must bind one exact library file");
  auto inputs = contracts.canonicalizeAndValidateInputs(
      draft.standardCellContract,
      {{contract->inputSlots.front().providerInputSlotRef,
        ExplicitFileDependency{draft.standardCellLibrary}}},
      RepresentationRootVariant::GateNetlist);
  if (!inputs)
    return inputs.takeError();
  std::optional<ImplementationPayloadKey> blackBox;
  std::size_t netlists = 0, constraints = 0;
  const std::string constraint = source->generationConstraint();
  for (const auto &payload : root.payloads) {
    switch (payload.role) {
    case PayloadRole::Netlist:
      ++netlists;
      break;
    case PayloadRole::GenerationConstraint:
      ++constraints;
      if (constraint.empty() ||
          payload.blobDigest != computeBlobDigest(bytesOf(constraint)))
        return invalid(
            "generation constraint differs from the source clock contract");
      break;
    case PayloadRole::BlackBoxContract:
      if (blackBox)
        return invalid("mapped library has multiple black-box contracts");
      blackBox =
          ImplementationPayloadKey{payload.role, payload.canonicalLogicalName};
      break;
    default:
      return invalid("payload is outside the block gate-netlist closure");
    }
    auto verified = blobs.verify(payload.blobDigest);
    if (!verified)
      return verified.takeError();
  }
  if (netlists != 1 || constraints != (constraint.empty() ? 0u : 1u) ||
      (contract->blackBoxContractRequired && !blackBox))
    return invalid(
        "netlist, constraint or mapped-library payload closure is incomplete");
  auto output = indexRepresentationRoot(root, blobs);
  if (!output)
    return output.takeError();
  auto sourceFormat = RepresentationFormatDescriptorRef::get(
      RepresentationFormatKind::SystemVerilogRtl);
  if (!sourceFormat)
    return sourceFormat.takeError();
  auto input =
      indexProspectiveRepresentation(*sourceFormat, root.top,
                                     {{PayloadRole::RtlSource, "rtl/block.sv",
                                       bytesOf(source->projection().source)}});
  if (!input)
    return input.takeError();
  if (input->rootBoundaryPorts() != output->rootBoundaryPorts())
    return invalid("synthesis changed the block root interface");
  if (contract->validator) {
    ExternalImplementationBindingDraft binding{
        draft.standardCellContract,
        std::move(*inputs),
        {},
        output->unresolvedExternalDefinitions().vec(),
        std::move(blackBox)};
    if (llvm::Error error =
            contract->validator(binding, root, &target->platform()))
      return error;
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<FinalizedBlockGateNetlist>
importBlockGateNetlist(const ArtifactRootReference &reference,
                       const ExternalImplementationContractCatalog &contracts,
                       const ArtifactStore &artifacts, const BlobStore &blobs) {
  if (reference.schemaIdentity != blockGateNetlistSchema.identity ||
      reference.schemaVersion != blockGateNetlistSchema.version)
    return invalid("root reference has another schema");
  auto canonical = artifacts.get(blockGateNetlistSchema, reference.artifact);
  if (!canonical)
    return canonical.takeError();
  llvm::StringRef text(
      reinterpret_cast<const char *>(canonical->bytes().data()),
      canonical->bytes().size());
  auto draft = decode(text);
  if (!draft)
    return draft.takeError();
  if (llvm::Error error = validate(*draft, contracts, artifacts, blobs))
    return std::move(error);
  auto encoded = encode(*draft);
  if (!encoded)
    return encoded.takeError();
  if (*encoded != text)
    return invalid("stored block gate netlist is not canonical");
  return FinalizedBlockGateNetlist(reference, std::move(*draft));
}

llvm::Expected<FinalizedBlockGateNetlist>
finalizeBlockGateNetlist(BlockGateNetlistDraft draft,
                         const ExternalImplementationContractCatalog &contracts,
                         const ArtifactStore &artifacts,
                         const BlobStore &blobs) {
  if (llvm::Error error = validate(draft, contracts, artifacts, blobs))
    return std::move(error);
  auto encoded = encode(draft);
  if (!encoded)
    return encoded.takeError();
  auto identity = artifacts.put(
      blockGateNetlistSchema, CanonicalSemanticBytes(std::vector<std::uint8_t>(
                                  encoded->begin(), encoded->end())));
  if (!identity)
    return identity.takeError();
  return importBlockGateNetlist({blockGateNetlistSchema.identity.str(),
                                 blockGateNetlistSchema.version, *identity},
                                contracts, artifacts, blobs);
}

} // namespace loom::hardware::rtl

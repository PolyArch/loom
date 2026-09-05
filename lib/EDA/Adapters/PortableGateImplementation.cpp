#include "EDA/Adapters/PortableGateImplementation.h"

#include "EDA/Adapters/AsicStandardCellContracts.h"
#include "Hardware/Implementation/RepresentationIndex.h"

namespace loom::eda {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "portable_gate_implementation_invalid: " +
                                     message);
}

} // namespace

llvm::Expected<hardware::FinalizedHardwareImplementation>
associatePortableBlockGateNetlist(
    const hardware::FinalizedHardwareImplementation &implementation,
    const hardware::rtl::FinalizedBlockGateNetlist &block,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs) {
  using namespace hardware;
  using namespace hardware::rtl;
  const auto &original = implementation.implementation();
  const auto &mapped = block.netlist();
  if (original.implementationPlatform() &&
      *original.implementationPlatform() != mapped.implementationPlatform)
    return invalid("portable RTL and mapped block select different platforms");
  auto abi = importConfigurationABI(original.configurationAbi(), artifacts);
  if (!abi)
    return abi.takeError();
  auto source = importRtlBlockSource(mapped.source, artifacts, blobs);
  if (!source)
    return source.takeError();
  if (llvm::Error error = verifyPortableRtlBlockSourceRootDerivation(
          *source, *abi, implementation, blobs))
    return std::move(error);
  // Exact portable replay above owns the admitted empty activity, memory-macro
  // and external-binding catalogs. Specialized RTL requires its own association
  // owner; no synthesis correspondence is inferred for such bindings here.
  auto before = indexRepresentationRoot(original.representationRoot(), blobs);
  if (!before)
    return before.takeError();
  auto after = indexRepresentationRoot(mapped.representation, blobs);
  if (!after)
    return after.takeError();
  std::vector<ImplementationInterface> interfaces = original.interfaces().vec();
  const std::string prefix =
      original.representationRoot().top.canonicalName + ".";
  for (auto &interface : interfaces) {
    auto sourceFacts = before->lookup(interface.representationLocator);
    if (!sourceFacts)
      return sourceFacts.takeError();
    llvm::StringRef port = interface.representationLocator.canonicalName;
    if (interface.representationLocator.kind !=
            RepresentationObjectKind::Port ||
        !port.consume_front(prefix) || port.empty() || port.contains('.'))
      return invalid("portable interface is not a direct root port");
    interface.representationLocator.canonicalName =
        mapped.representation.top.canonicalName + "." + port.str();
    auto mappedFacts = after->lookup(interface.representationLocator);
    if (!mappedFacts)
      return mappedFacts.takeError();
    if (!*sourceFacts || !*mappedFacts || !(**sourceFacts == **mappedFacts))
      return invalid("mapped root changed a public interface geometry");
  }
  std::optional<ImplementationPayloadKey> blackBox;
  for (const auto &payload : mapped.representation.payloads)
    if (payload.role == PayloadRole::BlackBoxContract)
      blackBox = {payload.role, payload.canonicalLogicalName};
  HardwareImplementationDraft draft{
      original.fabric(),
      original.subject(),
      original.configurationAbi(),
      mapped.representation,
      mapped.implementationPlatform,
      std::move(interfaces),
      {},
      {},
      {{mapped.standardCellContract,
        {{asicStandardCellLibertyInputSlot.str(),
          ExplicitFileDependency{mapped.standardCellLibrary}}},
        {},
        after->unresolvedExternalDefinitions().vec(),
        std::move(blackBox)}}};
  return finalizeHardwareImplementation(std::move(draft), contracts, artifacts,
                                        blobs);
}

} // namespace loom::eda

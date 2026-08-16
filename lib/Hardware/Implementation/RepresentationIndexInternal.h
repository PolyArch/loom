#ifndef LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONINDEXINTERNAL_H
#define LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONINDEXINTERNAL_H

#include "Hardware/Implementation/RepresentationFormat.h"
#include "Hardware/Implementation/RepresentationIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace slang::ast {
enum class ArgumentDirection;
class InstanceSymbol;
class Type;
} // namespace slang::ast

namespace loom::hardware::detail {

enum class BuiltinRepresentationIndexer {
  SystemVerilogRtl,
  StructuralVerilogGateNetlist,
  IndexedPhysical,
  FabricModel,
};

struct StaticRepresentationFormatEntry final {
  RepresentationFormatDescriptor descriptor;
  BuiltinRepresentationIndexer indexer;
};

const StaticRepresentationFormatEntry &
getStaticRepresentationFormatEntry(RepresentationFormatDescriptorRef reference);

llvm::Error invalidIndex(const llvm::Twine &reason);
llvm::Error unsupportedIndex(const llvm::Twine &reason);
llvm::Error
validateRepresentationTextPolicy(RepresentationTextPolicy policy,
                                 const ImplementationPayload &payload,
                                 llvm::ArrayRef<std::uint8_t> contents);

struct RawIndexEntry final {
  RepresentationLocator locator;
  RepresentationObjectFacts facts;
};

struct RawIndex final {
  std::optional<RepresentationRootVariant> rootVariant;
  std::optional<RepresentationPhysicalStage> stage;
  std::vector<RawIndexEntry> entries;
  std::vector<RepresentationLocator> unresolved;
};

class RawIndexBuilder final {
public:
  explicit RawIndexBuilder(RepresentationFormatDescriptorRef formatRef)
      : formatRef_(formatRef) {}

  llvm::Error addEntry(RepresentationLocator locator,
                       RepresentationObjectFacts facts);
  llvm::Error addUnresolvedModule(std::string_view definitionName);
  llvm::Expected<RawIndex> finish();

private:
  RepresentationFormatDescriptorRef formatRef_;
  RawIndex raw_;
};

std::string childPath(llvm::StringRef parent, std::string_view child);

llvm::Expected<RepresentationSignalDirection>
signalDirection(slang::ast::ArgumentDirection direction,
                llvm::StringRef description);

llvm::Expected<std::uint64_t> packedIntegralWidth(const slang::ast::Type &type,
                                                  llvm::StringRef description);

llvm::Expected<RawIndex>
indexSystemVerilogRtl(RepresentationFormatDescriptorRef formatRef,
                      const slang::ast::InstanceSymbol &top,
                      const RepresentationLocator &exactRoot);

llvm::Expected<RawIndex>
indexStructuralVerilogGateNetlist(RepresentationFormatDescriptorRef formatRef,
                                  const slang::ast::InstanceSymbol &top,
                                  const RepresentationLocator &exactRoot);

llvm::Expected<RawIndex>
indexHdlRepresentation(RepresentationFormatDescriptorRef formatRef,
                       const RepresentationLocator &exactRoot,
                       llvm::ArrayRef<ImplementationPayload> canonicalPayloads,
                       const BlobStore &blobs);

llvm::Expected<RawIndex>
indexHdlRepresentation(RepresentationFormatDescriptorRef formatRef,
                       const RepresentationLocator &exactRoot,
                       llvm::ArrayRef<ImplementationPayloadBytes> payloads);

llvm::Expected<RawIndex> indexPhysicalRepresentation(
    RepresentationFormatDescriptorRef formatRef,
    const RepresentationLocator &exactRoot,
    llvm::ArrayRef<ImplementationPayload> canonicalPayloads,
    const BlobStore &blobs);

} // namespace loom::hardware::detail

#endif // LOOM_HARDWARE_IMPLEMENTATION_REPRESENTATIONINDEXINTERNAL_H

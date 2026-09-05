#ifndef LOOM_EDA_ADAPTERS_PORTABLEGATEIMPLEMENTATION_H
#define LOOM_EDA_ADAPTERS_PORTABLEGATEIMPLEMENTATION_H

#include "Hardware/RTL/BlockGateNetlist.h"

namespace loom::eda {

/// Replays the complete portable root association and publishes its unchanged
/// mapped payloads and exact ASIC contract. The caller owns vendor-specific
/// block import and supplies that vendor's matching standard-cell catalog.
llvm::Expected<hardware::FinalizedHardwareImplementation>
associatePortableBlockGateNetlist(
    const hardware::FinalizedHardwareImplementation &implementation,
    const hardware::rtl::FinalizedBlockGateNetlist &block,
    const hardware::ExternalImplementationContractCatalog &contracts,
    const ArtifactStore &artifacts, const BlobStore &blobs);

} // namespace loom::eda

#endif // LOOM_EDA_ADAPTERS_PORTABLEGATEIMPLEMENTATION_H

#ifndef LOOM_LIB_APPLICATION_RUNTIMEMANIFESTVALIDATION_H
#define LOOM_LIB_APPLICATION_RUNTIMEMANIFESTVALIDATION_H

#include "Application/RuntimeManifest.h"

namespace loom::application::detail {

llvm::Error canonicalizeDigestSet(std::vector<ComponentViewDigest> &digests,
                                  llvm::StringRef name);
llvm::Error verifyManifestDraft(ApplicationRuntimeManifestDraft &draft,
                                const ArtifactStore &artifacts,
                                const BlobStore &blobs);

} // namespace loom::application::detail

#endif // LOOM_LIB_APPLICATION_RUNTIMEMANIFESTVALIDATION_H

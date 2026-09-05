#ifndef LOOM_APPLICATION_ACTIVATIONREPAIRLINEAGE_H
#define LOOM_APPLICATION_ACTIVATIONREPAIRLINEAGE_H

#include "llvm/Support/Error.h"

namespace loom {
class ArtifactStore;
}

namespace loom::application {
struct ApplicationActivationDecisionDraft;

namespace activation_detail {

llvm::Error
validateHardwareMutationRepairs(const ApplicationActivationDecisionDraft &draft,
                                const ArtifactStore &artifacts);

} // namespace activation_detail
} // namespace loom::application

#endif // LOOM_APPLICATION_ACTIVATIONREPAIRLINEAGE_H

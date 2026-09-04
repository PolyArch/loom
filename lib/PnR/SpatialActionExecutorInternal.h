#ifndef LOOM_LIB_PNR_SPATIALACTIONEXECUTORINTERNAL_H
#define LOOM_LIB_PNR_SPATIALACTIONEXECUTORINTERNAL_H

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

namespace loom::pnr::detail {

llvm::Error executorError(const llvm::Twine &message);
llvm::Error intrinsicTransitionFailure(const llvm::Twine &message);

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_SPATIALACTIONEXECUTORINTERNAL_H

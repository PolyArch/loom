#ifndef LOOM_PNR_INITIALIZERCHOICEORDER_H
#define LOOM_PNR_INITIALIZERCHOICEORDER_H

#include "PnR/DeterministicSearchProtocol.h"
#include "PnR/PnrIndex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

namespace loom::pnr::detail {

/// Builds the exact initializer choice order into caller-owned storage.
/// Canonical choices must already be sorted by their typed choice key.
llvm::Error
buildInitializerChoiceOrder(llvm::ArrayRef<PnrIndex> canonicalChoices,
                            DeterministicPnrRandomStream *diversificationStream,
                            llvm::MutableArrayRef<PnrIndex> choiceOrder,
                            llvm::MutableArrayRef<PnrIndex> fenwickScratch);

} // namespace loom::pnr::detail

#endif // LOOM_PNR_INITIALIZERCHOICEORDER_H

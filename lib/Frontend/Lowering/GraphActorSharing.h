#ifndef LOOM_FRONTEND_LOWERING_GRAPH_ACTOR_SHARING_H
#define LOOM_FRONTEND_LOWERING_GRAPH_ACTOR_SHARING_H

#include "mlir/IR/Block.h"

namespace loom::lowering {

// Publish one canonical actor per structural identity in a lowered graph body
// and reduce every frontier join whose prerequisites one of its own inputs
// already implies. Recursive region lowering emits one actor per role, and
// distinct roles routinely resolve to the same replication, selection, or
// rendezvous of the same inputs; without this step the published control
// density records how the lowering was written rather than what the source
// region means.
void shareCanonicalControlActors(::mlir::Block &body);

} // namespace loom::lowering

#endif // LOOM_FRONTEND_LOWERING_GRAPH_ACTOR_SHARING_H

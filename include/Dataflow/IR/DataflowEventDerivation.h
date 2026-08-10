#ifndef LOOM_DATAFLOW_IR_DATAFLOW_EVENT_DERIVATION_H
#define LOOM_DATAFLOW_IR_DATAFLOW_EVENT_DERIVATION_H

#include "Dataflow/IR/DataflowStructuralRefs.h"

#include <utility>

namespace dataflow {

inline EventFamilyKey graphLaunchStartEventFamily(RootedGraphLaunchRef launch) {
  return EventFamilyKey{StaticTransferEventRef{
      ProducedTransferEventRef{CanonicalProducerTerminalRef{
          GraphLaunchBoundarySourceRef{GraphLaunchBoundaryTransferRef{
              GraphLaunchStartTransferRef{std::move(launch)}}}}}}};
}

} // namespace dataflow

#endif // LOOM_DATAFLOW_IR_DATAFLOW_EVENT_DERIVATION_H

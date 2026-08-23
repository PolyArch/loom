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

inline EventFamilyKey
rootThreadStartEventFamily(RootThreadLaunchRef launch) {
  return EventFamilyKey{StaticTransferEventRef{
      ConsumedTransferEventRef{CanonicalSinkTerminalRef{
          RootThreadBoundarySinkRef{RootThreadBoundaryTransferRef{
              RootThreadStartTransferRef{std::move(launch)}}}}}}};
}

inline EventFamilyKey
rootThreadCompletionEventFamily(RootThreadLaunchRef launch) {
  return EventFamilyKey{StaticTransferEventRef{
      ProducedTransferEventRef{CanonicalProducerTerminalRef{
          RootThreadBoundarySourceRef{RootThreadBoundaryTransferRef{
              RootThreadCompletionTransferRef{std::move(launch)}}}}}}};
}

} // namespace dataflow

#endif // LOOM_DATAFLOW_IR_DATAFLOW_EVENT_DERIVATION_H

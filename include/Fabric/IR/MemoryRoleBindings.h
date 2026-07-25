#ifndef LOOM_FABRIC_IR_MEMORY_ROLE_BINDINGS_H
#define LOOM_FABRIC_IR_MEMORY_ROLE_BINDINGS_H

#include "Dataflow/IR/DataflowServiceSchema.h"
#include "Fabric/IR/FabricEnums.h"
#include "Fabric/Identity/FabricRefs.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <utility>
#include <vector>

namespace fabric {

struct MemoryRoleBinding {
  dataflow::semantics::ServiceValueRole role;
  loom::fabric::FabricTransportEndpointRef endpoint;
};

class TemporalMemoryInputMatcherQueue {
public:
  dataflow::semantics::ServiceValueRole role() const { return role_; }
  const loom::fabric::FabricTransportEndpointRef &endpoint() const {
    return endpoint_;
  }

private:
  TemporalMemoryInputMatcherQueue(
      dataflow::semantics::ServiceValueRole role,
      loom::fabric::FabricTransportEndpointRef endpoint)
      : role_(role), endpoint_(std::move(endpoint)) {}

  dataflow::semantics::ServiceValueRole role_;
  loom::fabric::FabricTransportEndpointRef endpoint_;

  friend class MemoryRoleBindingView;
};

/// One selected actor's active role relation. This derived view is
/// nonpersistent: canonical role bytes and the complete capability alternative
/// remain blocked on their Dataflow-owned codecs.
class MemoryRoleBindingView {
public:
  static llvm::Expected<MemoryRoleBindingView>
  create(Schedule schedule,
         const dataflow::semantics::CanonicalService &service,
         llvm::ArrayRef<MemoryRoleBinding> bindings);

  llvm::ArrayRef<MemoryRoleBinding> activeBindings() const {
    return activeBindings_;
  }
  llvm::ArrayRef<TemporalMemoryInputMatcherQueue>
  temporalInputMatcherQueues() const {
    return temporalInputMatcherQueues_;
  }

private:
  MemoryRoleBindingView(
      std::vector<MemoryRoleBinding> activeBindings,
      std::vector<TemporalMemoryInputMatcherQueue> temporalInputMatcherQueues)
      : activeBindings_(std::move(activeBindings)),
        temporalInputMatcherQueues_(std::move(temporalInputMatcherQueues)) {}

  std::vector<MemoryRoleBinding> activeBindings_;
  std::vector<TemporalMemoryInputMatcherQueue> temporalInputMatcherQueues_;
};

} // namespace fabric

#endif // LOOM_FABRIC_IR_MEMORY_ROLE_BINDINGS_H

#include "Fabric/IR/MemoryRoleBindings.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <system_error>

using namespace dataflow::semantics;

namespace fabric {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(std::errc::invalid_argument, "%s",
                                 message.str().c_str());
}

bool isKnownSchedule(Schedule schedule) {
  return schedule == Schedule::Spatial || schedule == Schedule::Temporal;
}

} // namespace

llvm::Expected<MemoryRoleBindingView>
MemoryRoleBindingView::create(Schedule schedule,
                              const CanonicalService &service,
                              llvm::ArrayRef<MemoryRoleBinding> bindings) {
  if (!isKnownSchedule(schedule))
    return invalid("memory role binding has an unknown schedule");

  const ServiceValues arguments = service.arguments();
  const ServiceValues results = service.results();
  if (bindings.size() != arguments.size() + results.size())
    return invalid("active memory role binding is not total");

  std::vector<MemoryRoleBinding> ordered;
  ordered.reserve(bindings.size());
  const auto appendRole = [&](ServiceValueRole role) -> llvm::Error {
    const MemoryRoleBinding *selected = nullptr;
    for (const MemoryRoleBinding &binding : bindings) {
      if (binding.role != role)
        continue;
      if (selected)
        return invalid("active memory role binding repeats one service role");
      selected = &binding;
    }
    if (!selected)
      return invalid("active memory role binding omits one service role");
    ordered.push_back(*selected);
    return llvm::Error::success();
  };

  for (const ServiceValue argument : arguments)
    if (llvm::Error error = appendRole(argument.role))
      return std::move(error);
  for (const ServiceValue result : results)
    if (llvm::Error error = appendRole(result.role))
      return std::move(error);

  for (std::size_t right = 0; right < ordered.size(); ++right) {
    const bool rightIsOutput = right >= arguments.size();
    for (std::size_t left = 0; left < right; ++left) {
      if (ordered[left].endpoint != ordered[right].endpoint)
        continue;
      if (schedule == Schedule::Spatial || rightIsOutput)
        return invalid("active memory output or Spatial role bindings must be "
                       "injective");
    }
  }

  std::vector<TemporalMemoryInputMatcherQueue> matcherQueues;
  if (schedule == Schedule::Temporal) {
    matcherQueues.reserve(arguments.size());
    for (std::size_t index = 0; index < arguments.size(); ++index)
      matcherQueues.push_back(TemporalMemoryInputMatcherQueue(
          ordered[index].role, ordered[index].endpoint));
  }

  return MemoryRoleBindingView(std::move(ordered), std::move(matcherQueues));
}

} // namespace fabric

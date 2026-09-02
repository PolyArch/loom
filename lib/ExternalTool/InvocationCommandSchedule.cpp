#include "InvocationBundleInternal.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <cstdint>
#include <set>
#include <string>
#include <system_error>
#include <vector>

namespace loom::external_tool {
namespace {

llvm::Error scheduleError(const llvm::Twine &detail) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "invocation_bundle_invalid: " + detail);
}

llvm::Expected<std::uint64_t> requiredUnsigned(const llvm::json::Object &object,
                                               llvm::StringRef field) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return scheduleError("parallel command group omits '" + field + "'");
  const std::optional<std::uint64_t> result = value->getAsUINT64();
  if (!result)
    return scheduleError("parallel command group field '" + field +
                         "' is not an unsigned integer");
  return *result;
}

} // namespace

llvm::Error validateParallelCommandGroups(
    llvm::ArrayRef<ExternalToolParallelCommandGroup> groups,
    llvm::ArrayRef<std::vector<std::string>> commands,
    llvm::ArrayRef<std::string> toolExecutables,
    llvm::ArrayRef<std::string> toolProducedExecutables) {
  std::uint64_t previousEnd = 0;
  bool hasPrevious = false;
  const std::set<std::string> produced(toolProducedExecutables.begin(),
                                       toolProducedExecutables.end());
  for (const ExternalToolParallelCommandGroup &group : groups) {
    if (group.beginCommandOrdinal >= group.endCommandOrdinal ||
        group.endCommandOrdinal > commands.size())
      return scheduleError("parallel command group range is invalid");
    const std::uint64_t commandCount =
        group.endCommandOrdinal - group.beginCommandOrdinal;
    if (commandCount < 2 || group.workerLimit < 2 ||
        group.workerLimit > commandCount)
      return scheduleError(
          "parallel command group worker limit is outside its range");
    if (hasPrevious && previousEnd > group.beginCommandOrdinal)
      return scheduleError(
          "parallel command groups are not canonical nonoverlapping ranges");
    previousEnd = group.endCommandOrdinal;
    hasPrevious = true;
    for (std::uint64_t ordinal = group.beginCommandOrdinal;
         ordinal != group.endCommandOrdinal; ++ordinal) {
      const std::vector<std::string> &command = commands[ordinal];
      if (command.empty() ||
          !llvm::is_contained(toolExecutables, command.front()))
        return scheduleError(
            "parallel command group contains a generated controller");
      if (llvm::any_of(command, [&](const std::string &argument) {
            return produced.count(argument) != 0;
          }))
        return scheduleError(
            "parallel command group consumes a tool-produced executable");
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<ExternalToolParallelCommandGroup>>
parseParallelCommandGroups(const llvm::json::Object &manifest) {
  const llvm::json::Value *value = manifest.get("parallel_command_groups");
  if (!value)
    return std::vector<ExternalToolParallelCommandGroup>{};
  const llvm::json::Array *array = value->getAsArray();
  if (!array)
    return scheduleError("parallel_command_groups is not an array");
  std::vector<ExternalToolParallelCommandGroup> groups;
  groups.reserve(array->size());
  for (const llvm::json::Value &value : *array) {
    const llvm::json::Object *object = value.getAsObject();
    if (!object)
      return scheduleError("parallel command group is not an object");
    for (const auto &[field, ignored] : *object)
      if (field != "begin_command_ordinal" && field != "end_command_ordinal" &&
          field != "worker_limit")
        return scheduleError("parallel command group contains unknown field '" +
                             llvm::StringRef(field) + "'");
    auto begin = requiredUnsigned(*object, "begin_command_ordinal");
    auto end = requiredUnsigned(*object, "end_command_ordinal");
    auto workers = requiredUnsigned(*object, "worker_limit");
    if (!begin)
      return begin.takeError();
    if (!end)
      return end.takeError();
    if (!workers)
      return workers.takeError();
    groups.push_back({*begin, *end, *workers});
  }
  return groups;
}

void writeParallelCommandGroups(
    llvm::json::OStream &json,
    llvm::ArrayRef<ExternalToolParallelCommandGroup> groups) {
  json.attributeArray("parallel_command_groups", [&] {
    for (const ExternalToolParallelCommandGroup &group : groups)
      json.object([&] {
        json.attribute("begin_command_ordinal", group.beginCommandOrdinal);
        json.attribute("end_command_ordinal", group.endCommandOrdinal);
        json.attribute("worker_limit", group.workerLimit);
      });
  });
}

} // namespace loom::external_tool

#include "fcc/Simulator/StaticModel.h"

#include <limits>

namespace fcc {
namespace sim {

const StaticConfigSlice *
StaticConfigImage::findSliceByNameAndKind(const std::string &name,
                                          const std::string &kind) const {
  for (const auto &slice : slices) {
    if (slice.name == name && slice.kind == kind)
      return &slice;
  }
  return nullptr;
}

std::optional<unsigned>
StaticMappedModel::getBoundaryInputOrdinal(IdIndex hwNodeId) const {
  if (hwNodeId == INVALID_ID ||
      hwNodeId >= static_cast<IdIndex>(boundaryInputOrdinals_.size()))
    return std::nullopt;
  unsigned ordinal = boundaryInputOrdinals_[hwNodeId];
  if (ordinal == std::numeric_limits<unsigned>::max())
    return std::nullopt;
  return ordinal;
}

std::optional<unsigned>
StaticMappedModel::getBoundaryOutputOrdinal(IdIndex hwNodeId) const {
  if (hwNodeId == INVALID_ID ||
      hwNodeId >= static_cast<IdIndex>(boundaryOutputOrdinals_.size()))
    return std::nullopt;
  unsigned ordinal = boundaryOutputOrdinals_[hwNodeId];
  if (ordinal == std::numeric_limits<unsigned>::max())
    return std::nullopt;
  return ordinal;
}

const StaticModuleDesc *StaticMappedModel::findModule(IdIndex hwNodeId) const {
  for (const auto &module : modules_) {
    if (module.hwNodeId == static_cast<uint32_t>(hwNodeId))
      return &module;
  }
  return nullptr;
}

const StaticPortDesc *StaticMappedModel::findPort(IdIndex portId) const {
  for (const auto &port : ports_) {
    if (port.portId == static_cast<uint32_t>(portId))
      return &port;
  }
  return nullptr;
}

} // namespace sim
} // namespace fcc

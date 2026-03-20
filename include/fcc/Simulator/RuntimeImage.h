#ifndef FCC_SIMULATOR_RUNTIMEIMAGE_H
#define FCC_SIMULATOR_RUNTIMEIMAGE_H

#include "fcc/Simulator/StaticModel.h"

#include <string>

namespace fcc {
namespace sim {

struct RuntimeScalarSlotBinding {
  uint32_t slot = 0;
  int64_t argIndex = -1;
  uint32_t portIdx = 0;
};

struct RuntimeMemorySlotBinding {
  uint32_t slot = 0;
  int64_t memrefArgIndex = -1;
  uint32_t regionId = 0;
  uint32_t elemSizeLog2 = 0;
};

struct RuntimeOutputSlotBinding {
  uint32_t slot = 0;
  int64_t resultIndex = -1;
  uint32_t portIdx = 0;
};

struct RuntimeControlImage {
  int32_t startTokenPort = -1;
  std::vector<RuntimeScalarSlotBinding> scalarBindings;
  std::vector<RuntimeMemorySlotBinding> memoryBindings;
  std::vector<RuntimeOutputSlotBinding> outputBindings;
};

struct RuntimeImage {
  StaticMappedModel staticModel;
  StaticConfigImage configImage;
  RuntimeControlImage controlImage;
};

bool writeRuntimeImageBinary(const RuntimeImage &image, const std::string &path,
                             std::string &error);

bool loadRuntimeImageBinary(const std::string &path, RuntimeImage &image,
                            std::string &error);

bool writeRuntimeImageJson(const RuntimeImage &image, const std::string &path,
                           std::string &error);

bool loadRuntimeImageJson(const std::string &path, RuntimeImage &image,
                          std::string &error);

} // namespace sim
} // namespace fcc

#endif // FCC_SIMULATOR_RUNTIMEIMAGE_H

#ifndef LOOM_TOOLS_LOOM_MAPPER_CONFIG_H
#define LOOM_TOOLS_LOOM_MAPPER_CONFIG_H

#include "loom/Mapper/MapperOptions.h"

#include <string>

namespace loom {

std::string getDefaultMapperBaseConfigPath();

bool loadMapperBaseConfig(const std::string &path, MapperOptions &opts,
                         std::string &error);

} // namespace loom

#endif

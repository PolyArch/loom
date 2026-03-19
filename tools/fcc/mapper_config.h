#ifndef FCC_TOOLS_FCC_MAPPER_CONFIG_H
#define FCC_TOOLS_FCC_MAPPER_CONFIG_H

#include "fcc/Mapper/MapperOptions.h"

#include <string>

namespace fcc {

std::string getDefaultMapperBaseConfigPath();

bool loadMapperBaseConfig(const std::string &path, MapperOptions &opts,
                         std::string &error);

} // namespace fcc

#endif

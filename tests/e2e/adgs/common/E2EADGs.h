#ifndef FCC_TESTS_E2E_ADGS_COMMON_E2EADGS_H
#define FCC_TESTS_E2E_ADGS_COMMON_E2EADGS_H

#include <string>

namespace fcc {
namespace e2e {

void buildTinyAdd1PE(const std::string &outputPath);
void buildMesh6x6Extmem1(const std::string &outputPath);
void buildMesh6x6Extmem2(const std::string &outputPath);

} // namespace e2e
} // namespace fcc

#endif

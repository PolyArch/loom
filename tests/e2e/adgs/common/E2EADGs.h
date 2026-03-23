#ifndef LOOM_TESTS_E2E_ADGS_COMMON_E2EADGS_H
#define LOOM_TESTS_E2E_ADGS_COMMON_E2EADGS_H

#include <string>

namespace loom {
namespace e2e {

void buildTinyAdd1PE(const std::string &outputPath);
void buildMesh6x6Extmem1(const std::string &outputPath);
void buildMesh6x6Extmem2(const std::string &outputPath);

/// Build parameterized mesh ADGs with full i32/i64/f32 FU complement.
/// Side-wired external memories with tagged bridges.
void buildMesh8x8Extmem2(const std::string &outputPath);
void buildMesh10x10Extmem2(const std::string &outputPath);
void buildMesh8x8Extmem4(const std::string &outputPath);
void buildMesh10x10Extmem4(const std::string &outputPath);

} // namespace e2e
} // namespace loom

#endif

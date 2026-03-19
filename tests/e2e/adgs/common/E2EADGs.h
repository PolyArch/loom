#ifndef FCC_TESTS_E2E_ADGS_COMMON_E2EADGS_H
#define FCC_TESTS_E2E_ADGS_COMMON_E2EADGS_H

#include <string>

namespace fcc {
namespace e2e {

void buildTinyAdd1PE(const std::string &outputPath);
void buildSumArrayDemoChess4x4(const std::string &outputPath);
void buildSumArrayDemoChess5x5(const std::string &outputPath);
void buildSumArrayDemoChess6x6(const std::string &outputPath);
void buildSumArrayDemoChess7x7(const std::string &outputPath);
void buildTinyStar4PE(const std::string &outputPath);
void buildMediumStar8PE(const std::string &outputPath);
void buildWideStar16PE(const std::string &outputPath);
void buildVecaddDemoChess6x6(const std::string &outputPath);
void buildMediumChess6x6(const std::string &outputPath);
void buildMediumChess10x10(const std::string &outputPath);

} // namespace e2e
} // namespace fcc

#endif

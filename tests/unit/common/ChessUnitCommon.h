//===-- ChessUnitCommon.h - Shared chessboard ADG test helpers -*- C++ -*-===//
//
// Part of the fcc project.
//
//===----------------------------------------------------------------------===//

#ifndef FCC_TESTS_UNIT_COMMON_CHESSUNITCOMMON_H
#define FCC_TESTS_UNIT_COMMON_CHESSUNITCOMMON_H

#include "fcc/ADG/ADGBuilder.h"

#include <cassert>
#include <string>

namespace fcc::unit {

inline void buildChessUnitADG(const std::string &outputPath,
                              const std::string &moduleName, unsigned rows,
                              unsigned cols) {
  assert(rows >= 1 && cols >= 2 &&
         "chess unit tests expect at least one row and two columns");

  fcc::adg::ADGBuilder builder(moduleName);
  constexpr unsigned dataWidth = 64;

  auto fuAdd = builder.defineFU("fu_add", {"i32", "i32"}, {"i32"},
                                {"arith.addi"});
  auto pe = builder.defineSpatialPE("chess_pe",
                                    /*numInputs=*/4,
                                    /*numOutputs=*/4,
                                    dataWidth, {fuAdd});

  constexpr unsigned topLeftBoundaryInputs = 3;
  constexpr unsigned bottomRightBoundaryOutputs = 1;
  auto mesh = builder.buildChessMesh(rows, cols, pe,
                                     /*decomposableBits=*/-1,
                                     topLeftBoundaryInputs,
                                     bottomRightBoundaryOutputs);

  auto in0 = builder.addScalarInput("a", dataWidth);
  auto in1 = builder.addScalarInput("b", dataWidth);
  auto in2 = builder.addScalarInput("c", dataWidth);
  auto out0 = builder.addScalarOutput("result", dataWidth);

  auto swTopLeft = mesh.swGrid[0][0];
  auto swBottomRight = mesh.swGrid[rows][cols];
  constexpr unsigned topLeftBaseDegree = 3;
  constexpr unsigned bottomRightBaseDegree = 3;

  builder.connectScalarInputToInstance(in0, swTopLeft, topLeftBaseDegree + 0);
  builder.connectScalarInputToInstance(in1, swTopLeft, topLeftBaseDegree + 1);
  builder.connectScalarInputToInstance(in2, swTopLeft, topLeftBaseDegree + 2);
  builder.connectInstanceToScalarOutput(swBottomRight, bottomRightBaseDegree,
                                        out0);

  builder.exportMLIR(outputPath);
}

} // namespace fcc::unit

#endif // FCC_TESTS_UNIT_COMMON_CHESSUNITCOMMON_H

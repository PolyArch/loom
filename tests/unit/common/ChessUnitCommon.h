//===-- ChessUnitCommon.h - Shared chessboard ADG test helpers -*- C++ -*-===//
//
// Part of the loom project.
//
//===----------------------------------------------------------------------===//

#ifndef LOOM_TESTS_UNIT_COMMON_CHESSUNITCOMMON_H
#define LOOM_TESTS_UNIT_COMMON_CHESSUNITCOMMON_H

#include "loom/ADG/ADGBuilder.h"

#include <cassert>
#include <string>

namespace loom::unit {

inline void buildChessUnitADG(const std::string &outputPath,
                              const std::string &moduleName, unsigned rows,
                              unsigned cols) {
  assert(rows >= 1 && cols >= 2 &&
         "chess unit tests expect at least one row and two columns");

  loom::adg::ADGBuilder builder(moduleName);
  constexpr unsigned dataWidth = 64;

  auto fuAdd = builder.defineBinaryFU("fu_add", "arith.addi", "i32", "i32");

  auto pe = builder.defineSingleFUSpatialPE("chess_pe", 4, 4, dataWidth, fuAdd);

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

  builder.connectInputToPort(in0, mesh.ingressPorts[0]);
  builder.connectInputToPort(in1, mesh.ingressPorts[1]);
  builder.connectInputToPort(in2, mesh.ingressPorts[2]);
  builder.connectPortToOutput(mesh.egressPorts[0], out0);

  builder.exportMLIR(outputPath);
}

} // namespace loom::unit

#endif // LOOM_TESTS_UNIT_COMMON_CHESSUNITCOMMON_H

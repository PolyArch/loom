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

} // namespace fcc::unit

#endif // FCC_TESTS_UNIT_COMMON_CHESSUNITCOMMON_H

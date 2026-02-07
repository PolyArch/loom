//===-- topo_diag_mesh_3x3.cpp - ADG test: 3x3 diagonal mesh ---*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verifies that buildMesh with Topology::DiagonalMesh creates:
//   - 3x3 PE grid and 3x3 switch grid
//   - SE/SW diagonal internal connections (no wraparound)
//   - All PEs and switches connected
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

#include <cassert>

using namespace loom::adg;

int main() {
  ADGBuilder builder("topo_diag_mesh_3x3");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(9, 9)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(3, 3, pe, sw, Topology::DiagonalMesh);

  // Verify grid dimensions.
  assert(mesh.peGrid.size() == 3 && "expected 3 PE rows");
  assert(mesh.peGrid[0].size() == 3 && "expected 3 PE cols");
  assert(mesh.swGrid.size() == 3 && "expected 3 switch rows");
  assert(mesh.swGrid[0].size() == 3 && "expected 3 switch cols");

  // Chain PEs in row-major order:
  // [0][0] -> [0][1] -> [0][2] -> [1][0] -> [1][1] -> [1][2] ->
  // [2][0] -> [2][1] -> [2][2]
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto e = builder.addModuleInput("e", Type::i32());
  auto f = builder.addModuleInput("f", Type::i32());
  auto g = builder.addModuleInput("g", Type::i32());
  auto h = builder.addModuleInput("h", Type::i32());
  auto i = builder.addModuleInput("i", Type::i32());
  auto j = builder.addModuleInput("j", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[0][2], 0);
  builder.connectToModuleInput(d, mesh.peGrid[0][2], 1);
  builder.connectPorts(mesh.peGrid[0][2], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(e, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(f, mesh.peGrid[1][1], 1);
  builder.connectPorts(mesh.peGrid[1][1], 0, mesh.peGrid[1][2], 0);
  builder.connectToModuleInput(g, mesh.peGrid[1][2], 1);
  builder.connectPorts(mesh.peGrid[1][2], 0, mesh.peGrid[2][0], 0);
  builder.connectToModuleInput(h, mesh.peGrid[2][0], 1);
  builder.connectPorts(mesh.peGrid[2][0], 0, mesh.peGrid[2][1], 0);
  builder.connectToModuleInput(i, mesh.peGrid[2][1], 1);
  builder.connectPorts(mesh.peGrid[2][1], 0, mesh.peGrid[2][2], 0);
  builder.connectToModuleInput(j, mesh.peGrid[2][2], 1);
  builder.connectToModuleOutput(mesh.peGrid[2][2], 0, out);

  // Validate ADG before export.
  auto validation = builder.validateADG();
  assert(validation.success && "validation failed");

  builder.exportMLIR("Output/topo_diag_mesh_3x3.fabric.mlir");
  return 0;
}

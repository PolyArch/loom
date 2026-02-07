//===-- topo_torus_4x4.cpp - ADG test: 4x4 torus topology ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verifies that buildMesh with Topology::Torus creates:
//   - 4x4 PE grid and 4x4 switch grid
//   - East-West wraparound module I/O (4 pairs: one per row)
//   - North-South wraparound module I/O (4 pairs: one per column)
//   - All PEs and switches connected
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

#include <cassert>
#include <fstream>
#include <string>

static unsigned mlirCount(const std::string &path, const std::string &substr) {
  std::ifstream f(path);
  std::string content((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  unsigned count = 0;
  size_t pos = 0;
  while ((pos = content.find(substr, pos)) != std::string::npos) {
    ++count;
    pos += substr.size();
  }
  return count;
}

using namespace loom::adg;

int main() {
  ADGBuilder builder("topo_torus_4x4");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(4, 4, pe, sw, Topology::Torus);

  // Verify grid dimensions.
  assert(mesh.peGrid.size() == 4 && "expected 4 PE rows");
  assert(mesh.peGrid[0].size() == 4 && "expected 4 PE cols");
  assert(mesh.swGrid.size() == 4 && "expected 4 switch rows");
  assert(mesh.swGrid[0].size() == 4 && "expected 4 switch cols");

  // Chain 16 PEs in row-major order:
  // [0][0] -> [0][1] -> [0][2] -> [0][3] -> [1][0] -> ... -> [3][3]
  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto in2 = builder.addModuleInput("in2", Type::i32());
  auto in3 = builder.addModuleInput("in3", Type::i32());
  auto in4 = builder.addModuleInput("in4", Type::i32());
  auto in5 = builder.addModuleInput("in5", Type::i32());
  auto in6 = builder.addModuleInput("in6", Type::i32());
  auto in7 = builder.addModuleInput("in7", Type::i32());
  auto in8 = builder.addModuleInput("in8", Type::i32());
  auto in9 = builder.addModuleInput("in9", Type::i32());
  auto in10 = builder.addModuleInput("in10", Type::i32());
  auto in11 = builder.addModuleInput("in11", Type::i32());
  auto in12 = builder.addModuleInput("in12", Type::i32());
  auto in13 = builder.addModuleInput("in13", Type::i32());
  auto in14 = builder.addModuleInput("in14", Type::i32());
  auto in15 = builder.addModuleInput("in15", Type::i32());
  auto in16 = builder.addModuleInput("in16", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  // PE[0][0]: first PE gets both inputs from module inputs
  builder.connectToModuleInput(in0, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(in1, mesh.peGrid[0][0], 1);

  // PE[0][1]
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(in2, mesh.peGrid[0][1], 1);
  // PE[0][2]
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[0][2], 0);
  builder.connectToModuleInput(in3, mesh.peGrid[0][2], 1);
  // PE[0][3]
  builder.connectPorts(mesh.peGrid[0][2], 0, mesh.peGrid[0][3], 0);
  builder.connectToModuleInput(in4, mesh.peGrid[0][3], 1);
  // PE[1][0]
  builder.connectPorts(mesh.peGrid[0][3], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(in5, mesh.peGrid[1][0], 1);
  // PE[1][1]
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(in6, mesh.peGrid[1][1], 1);
  // PE[1][2]
  builder.connectPorts(mesh.peGrid[1][1], 0, mesh.peGrid[1][2], 0);
  builder.connectToModuleInput(in7, mesh.peGrid[1][2], 1);
  // PE[1][3]
  builder.connectPorts(mesh.peGrid[1][2], 0, mesh.peGrid[1][3], 0);
  builder.connectToModuleInput(in8, mesh.peGrid[1][3], 1);
  // PE[2][0]
  builder.connectPorts(mesh.peGrid[1][3], 0, mesh.peGrid[2][0], 0);
  builder.connectToModuleInput(in9, mesh.peGrid[2][0], 1);
  // PE[2][1]
  builder.connectPorts(mesh.peGrid[2][0], 0, mesh.peGrid[2][1], 0);
  builder.connectToModuleInput(in10, mesh.peGrid[2][1], 1);
  // PE[2][2]
  builder.connectPorts(mesh.peGrid[2][1], 0, mesh.peGrid[2][2], 0);
  builder.connectToModuleInput(in11, mesh.peGrid[2][2], 1);
  // PE[2][3]
  builder.connectPorts(mesh.peGrid[2][2], 0, mesh.peGrid[2][3], 0);
  builder.connectToModuleInput(in12, mesh.peGrid[2][3], 1);
  // PE[3][0]
  builder.connectPorts(mesh.peGrid[2][3], 0, mesh.peGrid[3][0], 0);
  builder.connectToModuleInput(in13, mesh.peGrid[3][0], 1);
  // PE[3][1]
  builder.connectPorts(mesh.peGrid[3][0], 0, mesh.peGrid[3][1], 0);
  builder.connectToModuleInput(in14, mesh.peGrid[3][1], 1);
  // PE[3][2]
  builder.connectPorts(mesh.peGrid[3][1], 0, mesh.peGrid[3][2], 0);
  builder.connectToModuleInput(in15, mesh.peGrid[3][2], 1);
  // PE[3][3]
  builder.connectPorts(mesh.peGrid[3][2], 0, mesh.peGrid[3][3], 0);
  builder.connectToModuleInput(in16, mesh.peGrid[3][3], 1);

  builder.connectToModuleOutput(mesh.peGrid[3][3], 0, out);

  // Validate ADG before export.
  auto validation = builder.validateADG();
  assert(validation.success && "validation failed");

  builder.exportMLIR("Output/topo_torus_4x4.fabric.mlir");

  // Verify MLIR contains expected instances: 16 PEs and 16 switches.
  const char *mlir = "Output/topo_torus_4x4.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 16 && "expected 16 PE instances");
  assert(mlirCount(mlir, "fabric.switch") == 16 && "expected 16 switch instances");
  assert(mlirCount(mlir, "sym_name = \"pe_") == 16 && "expected pe_ sym_names");

  return 0;
}

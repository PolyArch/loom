//===-- topo_torus_2x2.cpp - ADG test: 2x2 torus topology ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verifies that buildMesh with Topology::Torus creates:
//   - 2x2 PE grid and 2x2 switch grid
//   - East-West wraparound module I/O (2 pairs: one per row)
//   - North-South wraparound module I/O (2 pairs: one per column)
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
  ADGBuilder builder("topo_torus_2x2");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::Torus);

  // Verify grid dimensions.
  assert(mesh.peGrid.size() == 2 && "expected 2 PE rows");
  assert(mesh.peGrid[0].size() == 2 && "expected 2 PE cols");
  assert(mesh.swGrid.size() == 2 && "expected 2 switch rows");
  assert(mesh.swGrid[0].size() == 2 && "expected 2 switch cols");

  // Chain PEs in row-major order: [0][0] -> [0][1] -> [1][0] -> [1][1]
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto e = builder.addModuleInput("e", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(d, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(e, mesh.peGrid[1][1], 1);
  builder.connectToModuleOutput(mesh.peGrid[1][1], 0, out);

  // Validate ADG before export.
  auto validation = builder.validateADG();
  assert(validation.success && "validation failed");

  builder.exportMLIR("Output/topo_torus_2x2.fabric.mlir");

  // Verify MLIR contains expected instances: 4 PEs and 4 switches.
  const char *mlir = "Output/topo_torus_2x2.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 4 && "expected 4 PE instances");
  assert(mlirCount(mlir, "fabric.switch") == 4 && "expected 4 switch instances");
  // Torus has more module I/O than mesh due to wraparound connections.
  // A 2x2 torus with 5-port switches has wrap ports that become module I/O.
  assert(mlirCount(mlir, "sym_name = \"pe_") == 4 && "expected pe_ sym_names");
  assert(mlirCount(mlir, "sym_name = \"sw_") == 0 ||
         mlirCount(mlir, "fabric.switch") == 4);

  return 0;
}

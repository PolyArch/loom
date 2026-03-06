//===-- topo_lattice_1x1.cpp - ADG test: 1x1 lattice mesh ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

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
  ADGBuilder builder("topo_lattice_1x1");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(8, 8)
      .setType(Type::i32());

  auto lattice = builder.latticeMesh(1, 1, sw);

  // 1x1 PE grid -> 2x2 switch grid
  assert(lattice.peRows == 1);
  assert(lattice.peCols == 1);

  // Place PE
  builder.placePEInLattice(lattice, 0, 0, pe, "pe_0_0");

  builder.finalizeLattice(lattice);
  builder.exportMLIR("Output/topo_lattice_1x1.fabric.mlir");

  const char *mlir = "Output/topo_lattice_1x1.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 1 && "expected 1 PE instance");
  assert(mlirCount(mlir, "fabric.switch") == 4 && "expected 4 switch instances");

  return 0;
}

//===-- topo_lattice_hetero.cpp - ADG test: heterogeneous PEs ---*- C++ -*-===//
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
  ADGBuilder builder("topo_lattice_hetero");

  // 1-in 1-out PE (unary negation)
  auto pe1 = builder.newPE("pe_1i1o")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32()})
      .setOutputPorts({Type::i32()})
      .setBodyMLIR(
          "  %c0 = arith.constant 0 : i32\n"
          "  %r = arith.subi %c0, %arg0 : i32\n"
          "  fabric.yield %r : i32\n");

  // 2-in 1-out PE
  auto pe2 = builder.newPE("pe_2i1o")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // 3-in 2-out PE (select + passthrough)
  auto pe3 = builder.newPE("pe_3i2o")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32(), Type::i32()})
      .setBodyMLIR(
          "  %r0 = arith.addi %arg0, %arg1 : i32\n"
          "  %r1 = arith.addi %arg1, %arg2 : i32\n"
          "  fabric.yield %r0, %r1 : i32, i32\n");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(8, 8)
      .setType(Type::i32());

  auto lattice = builder.latticeMesh(2, 2, sw);

  builder.placePEInLattice(lattice, 0, 0, pe1, "pe_1i1o_0");
  builder.placePEInLattice(lattice, 0, 1, pe2, "pe_2i1o_0");
  builder.placePEInLattice(lattice, 1, 0, pe3, "pe_3i2o_0");
  builder.placePEInLattice(lattice, 1, 1, pe2, "pe_2i1o_1");

  builder.finalizeLattice(lattice);
  builder.exportMLIR("Output/topo_lattice_hetero.fabric.mlir");

  const char *mlir = "Output/topo_lattice_hetero.fabric.mlir";
  assert(mlirCount(mlir, "fabric.instance") == 4 && "expected 4 PE instances");
  assert(mlirCount(mlir, "fabric.switch") == 9 && "expected 9 switch instances");

  return 0;
}

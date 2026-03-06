//===-- topo_lattice_multitype.cpp - ADG test: two type planes --*- C++ -*-===//
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
  ADGBuilder builder("topo_lattice_multitype");

  // i32 PE
  auto peI32 = builder.newPE("pe_addi")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // f32 PE
  auto peF32 = builder.newPE("pe_addf")
      .setLatency(1, 1, 3)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::f32(), Type::f32()})
      .setOutputPorts({Type::f32()})
      .addOp("arith.addf");

  // Type conversion PE: i32 -> f32
  auto peCvt = builder.newPE("pe_sitofp")
      .setLatency(1, 1, 3)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32()})
      .setOutputPorts({Type::f32()})
      .setBodyMLIR(
          "  %r = arith.sitofp %arg0 : i32 to f32\n"
          "  fabric.yield %r : f32\n");

  // i32 switch
  auto swI32 = builder.newSwitch("sw_i32")
      .setPortCount(8, 8)
      .setType(Type::i32());

  // f32 switch
  auto swF32 = builder.newSwitch("sw_f32")
      .setPortCount(8, 8)
      .setType(Type::f32());

  // i32 lattice: 2x2
  auto latticeI32 = builder.latticeMesh(2, 2, swI32);
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 2; ++c)
      builder.placePEInLattice(latticeI32, r, c, peI32,
          "pe_i32_" + std::to_string(r) + "_" + std::to_string(c));

  // f32 lattice: 1x1
  auto latticeF32 = builder.latticeMesh(1, 1, swF32);
  builder.placePEInLattice(latticeF32, 0, 0, peF32, "pe_f32_0_0");

  // Conversion PE bridging the two lattices (connected manually)
  auto cvt = builder.clone(peCvt, "cvt_0");
  // Connect cvt input from i32 lattice switch output (port 5; port 4 is
  // already used by PE(0,0) input 0).
  builder.connectPorts(latticeI32.swGrid[0][0], 5, cvt, 0);
  // Connect cvt output to f32 lattice switch input
  builder.connectPorts(cvt, 0, latticeF32.swGrid[0][0], 4);

  builder.finalizeLattice(latticeI32);
  builder.finalizeLattice(latticeF32);
  builder.exportMLIR("Output/topo_lattice_multitype.fabric.mlir");

  const char *mlir = "Output/topo_lattice_multitype.fabric.mlir";
  // 4 i32 PEs + 1 f32 PE + 1 conversion PE = 6 instances
  assert(mlirCount(mlir, "fabric.instance") == 6 && "expected 6 PE instances");
  // 9 i32 switches + 4 f32 switches = 13
  assert(mlirCount(mlir, "fabric.switch") == 13 && "expected 13 switch instances");

  return 0;
}

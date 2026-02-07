//===-- mixed_heterogeneous_4x4.cpp - ADG test: 4x4 mesh, 4 PE types -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_heterogeneous_4x4");

  // Default PE for mesh construction
  auto pe_add = builder.newPE("pe_add")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  // Additional PE types
  auto pe_mul = builder.newPE("pe_mul")
      .setLatency(2, 2, 2)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.muli");

  auto pe_sub = builder.newPE("pe_sub")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.subi");

  auto pe_and = builder.newPE("pe_and")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.andi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  auto mesh = builder.buildMesh(4, 4, pe_add, sw, Topology::Mesh);

  // Clone additional PE types alongside the mesh PEs
  auto mul_0 = builder.clone(pe_mul, "mul_0");
  auto sub_0 = builder.clone(pe_sub, "sub_0");
  auto and_0 = builder.clone(pe_and, "and_0");
  auto mul_1 = builder.clone(pe_mul, "mul_1");

  // Module I/O
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
  auto k = builder.addModuleInput("k", Type::i32());
  auto l = builder.addModuleInput("l", Type::i32());
  auto m = builder.addModuleInput("m", Type::i32());
  auto n = builder.addModuleInput("n", Type::i32());
  auto o = builder.addModuleInput("o", Type::i32());
  auto p = builder.addModuleInput("p", Type::i32());
  auto q = builder.addModuleInput("q", Type::i32());
  auto mesh_out = builder.addModuleOutput("mesh_result", Type::i32());
  auto hetero_out = builder.addModuleOutput("hetero_result", Type::i32());

  // Chain PEs in row-major order: [0][0] -> [0][1] -> ... -> [3][3]
  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[0][2], 0);
  builder.connectToModuleInput(d, mesh.peGrid[0][2], 1);
  builder.connectPorts(mesh.peGrid[0][2], 0, mesh.peGrid[0][3], 0);
  builder.connectToModuleInput(e, mesh.peGrid[0][3], 1);
  builder.connectPorts(mesh.peGrid[0][3], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(f, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(g, mesh.peGrid[1][1], 1);
  builder.connectPorts(mesh.peGrid[1][1], 0, mesh.peGrid[1][2], 0);
  builder.connectToModuleInput(h, mesh.peGrid[1][2], 1);
  builder.connectPorts(mesh.peGrid[1][2], 0, mesh.peGrid[1][3], 0);
  builder.connectToModuleInput(i, mesh.peGrid[1][3], 1);
  builder.connectPorts(mesh.peGrid[1][3], 0, mesh.peGrid[2][0], 0);
  builder.connectToModuleInput(j, mesh.peGrid[2][0], 1);
  builder.connectPorts(mesh.peGrid[2][0], 0, mesh.peGrid[2][1], 0);
  builder.connectToModuleInput(k, mesh.peGrid[2][1], 1);
  builder.connectPorts(mesh.peGrid[2][1], 0, mesh.peGrid[2][2], 0);
  builder.connectToModuleInput(l, mesh.peGrid[2][2], 1);
  builder.connectPorts(mesh.peGrid[2][2], 0, mesh.peGrid[2][3], 0);
  builder.connectToModuleInput(m, mesh.peGrid[2][3], 1);
  builder.connectPorts(mesh.peGrid[2][3], 0, mesh.peGrid[3][0], 0);
  builder.connectToModuleInput(n, mesh.peGrid[3][0], 1);
  builder.connectPorts(mesh.peGrid[3][0], 0, mesh.peGrid[3][1], 0);
  builder.connectToModuleInput(o, mesh.peGrid[3][1], 1);
  builder.connectPorts(mesh.peGrid[3][1], 0, mesh.peGrid[3][2], 0);
  builder.connectToModuleInput(p, mesh.peGrid[3][2], 1);
  builder.connectPorts(mesh.peGrid[3][2], 0, mesh.peGrid[3][3], 0);
  builder.connectToModuleInput(q, mesh.peGrid[3][3], 1);
  builder.connectToModuleOutput(mesh.peGrid[3][3], 0, mesh_out);

  // Heterogeneous chain: mul_0 -> sub_0 -> and_0 -> mul_1
  builder.connectToModuleInput(a, mul_0, 0);
  builder.connectToModuleInput(b, mul_0, 1);
  builder.connectPorts(mul_0, 0, sub_0, 0);
  builder.connectToModuleInput(c, sub_0, 1);
  builder.connectPorts(sub_0, 0, and_0, 0);
  builder.connectToModuleInput(d, and_0, 1);
  builder.connectPorts(and_0, 0, mul_1, 0);
  builder.connectToModuleInput(c, mul_1, 1);
  builder.connectToModuleOutput(mul_1, 0, hetero_out);

  builder.exportMLIR("Output/mixed_heterogeneous_4x4.fabric.mlir");
  return 0;
}

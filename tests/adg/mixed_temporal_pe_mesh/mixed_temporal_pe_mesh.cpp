//===-- mixed_temporal_pe_mesh.cpp - ADG test: 2x2 mesh + temporal PEs -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_temporal_pe_mesh");

  auto tagType = Type::iN(4);
  auto taggedType = Type::tagged(Type::i32(), tagType);

  // Native PE for mesh
  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(Type::i32());

  // Temporal PE alongside
  auto fu = builder.newPE("fu_mul")
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.muli");

  auto tpe = builder.newTemporalPE("tpe_mul")
      .setNumRegisters(0)
      .setNumInstructions(4)
      .setRegFifoDepth(0)
      .setInterface(taggedType)
      .addFU(fu);

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::Mesh);
  auto tpe0 = builder.clone(tpe, "tpe0");
  InstanceHandle at0 = builder.newAddTag("tag0")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle at1 = builder.newAddTag("tag1")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle dt0 = builder.newDelTag("untag0")
      .setInputType(taggedType);

  // Module I/O
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto e = builder.addModuleInput("e", Type::i32());
  auto mesh_out = builder.addModuleOutput("mesh_result", Type::i32());
  auto tpe_out = builder.addModuleOutput("tpe_result", Type::i32());

  // Mesh path: chain PEs in row-major order [0][0] -> [0][1] -> [1][0] -> [1][1]
  builder.connectToModuleInput(a, mesh.peGrid[0][0], 0);
  builder.connectToModuleInput(b, mesh.peGrid[0][0], 1);
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(d, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(e, mesh.peGrid[1][1], 1);
  builder.connectToModuleOutput(mesh.peGrid[1][1], 0, mesh_out);

  // Temporal PE path: b, c -> add_tag -> tpe -> del_tag -> output
  builder.connectToModuleInput(b, at0, 0);
  builder.connectToModuleInput(c, at1, 0);
  builder.connectPorts(at0, 0, tpe0, 0);
  builder.connectPorts(at1, 0, tpe0, 1);
  builder.connectPorts(tpe0, 0, dt0, 0);
  builder.connectToModuleOutput(dt0, 0, tpe_out);

  builder.exportMLIR("Output/mixed_temporal_pe_mesh.fabric.mlir");
  return 0;
}

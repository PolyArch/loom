//===-- mixed_full_system.cpp - ADG test: full system with all components -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_full_system");

  auto tagType = Type::iN(4);
  auto taggedType = Type::tagged(Type::i32(), tagType);

  // Tagged PE for mesh
  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setInputPorts({taggedType, taggedType})
      .setOutputPorts({taggedType})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(5, 5)
      .setType(taggedType);

  auto mem = builder.newMemory("sram")
      .setLoadPorts(1)
      .setStorePorts(0)
      .setShape(MemrefType::static1D(128, Type::i32()));

  auto cpe = builder.newConstantPE("const_gen")
      .setOutputType(Type::i32());

  // Build 2x2 mesh (tagged)
  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::Mesh);
  auto mem0 = builder.clone(mem, "mem0");
  auto c0 = builder.clone(cpe, "c0");
  InstanceHandle at0 = builder.newAddTag("tag0")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle at1 = builder.newAddTag("tag1")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle dt0 = builder.newDelTag("untag0")
      .setInputType(taggedType);

  // Module I/O
  auto ctrl = builder.addModuleInput("ctrl", Type::none());
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto ld_addr = builder.addModuleInput("ld_addr", Type::index());
  auto mesh_result = builder.addModuleOutput("mesh_result", Type::i32());
  auto lddata = builder.addModuleOutput("lddata", Type::i32());
  auto lddone = builder.addModuleOutput("lddone", Type::none());

  // Additional add_tag instances for chaining
  InstanceHandle at2 = builder.newAddTag("tag2")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle at3 = builder.newAddTag("tag3")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle at4 = builder.newAddTag("tag4")
      .setValueType(Type::i32()).setTagType(tagType);

  // Constant PE: ctrl -> const value
  builder.connectToModuleInput(ctrl, c0, 0);

  // add_tag: native values -> tagged
  builder.connectPorts(c0, 0, at0, 0);
  builder.connectToModuleInput(a, at1, 0);

  // Tagged mesh: at0 -> PE[0][0] port 0, at1 -> PE[0][0] port 1
  builder.connectPorts(at0, 0, mesh.peGrid[0][0], 0);
  builder.connectPorts(at1, 0, mesh.peGrid[0][0], 1);

  // Chain PEs in row-major order: [0][0] -> [0][1] -> [1][0] -> [1][1]
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(b, at2, 0);
  builder.connectPorts(at2, 0, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(b, at3, 0);
  builder.connectPorts(at3, 0, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(c, at4, 0);
  builder.connectPorts(at4, 0, mesh.peGrid[1][1], 1);

  // del_tag: mesh output -> native
  builder.connectPorts(mesh.peGrid[1][1], 0, dt0, 0);
  builder.connectToModuleOutput(dt0, 0, mesh_result);

  // Memory subsystem (independent path)
  builder.connectToModuleInput(ld_addr, mem0, 0);
  builder.connectToModuleOutput(mem0, 0, lddata);
  builder.connectToModuleOutput(mem0, 1, lddone);

  builder.exportMLIR("Output/mixed_full_system.fabric.mlir");
  return 0;
}

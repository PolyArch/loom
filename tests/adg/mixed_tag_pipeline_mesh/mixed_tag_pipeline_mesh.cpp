//===-- mixed_tag_pipeline_mesh.cpp - ADG test: 2x2 mesh + add/del tag -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_tag_pipeline_mesh");

  auto tagType = Type::iN(4);
  auto taggedType = Type::tagged(Type::i32(), tagType);

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

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::Mesh);
  InstanceHandle at0 = builder.newAddTag("tag0")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle at1 = builder.newAddTag("tag1")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle at2 = builder.newAddTag("tag2")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle at3 = builder.newAddTag("tag3")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle at4 = builder.newAddTag("tag4")
      .setValueType(Type::i32()).setTagType(tagType);
  InstanceHandle dt0 = builder.newDelTag("untag0")
      .setInputType(taggedType);

  // Module I/O: native inputs, native output
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto result = builder.addModuleOutput("result", Type::i32());

  // add_tag at input
  builder.connectToModuleInput(a, at0, 0);
  builder.connectToModuleInput(b, at1, 0);
  builder.connectPorts(at0, 0, mesh.peGrid[0][0], 0);
  builder.connectPorts(at1, 0, mesh.peGrid[0][0], 1);

  // Chain PEs in row-major order: [0][0] -> [0][1] -> [1][0] -> [1][1]
  // Each subsequent PE gets input 0 from previous PE, input 1 from tagged module input
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, at2, 0);
  builder.connectPorts(at2, 0, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(c, at3, 0);
  builder.connectPorts(at3, 0, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(d, at4, 0);
  builder.connectPorts(at4, 0, mesh.peGrid[1][1], 1);

  // del_tag at output
  builder.connectPorts(mesh.peGrid[1][1], 0, dt0, 0);
  builder.connectToModuleOutput(dt0, 0, result);

  builder.exportMLIR("Output/mixed_tag_pipeline_mesh.fabric.mlir");
  return 0;
}

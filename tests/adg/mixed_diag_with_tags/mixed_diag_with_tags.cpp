//===-- mixed_diag_with_tags.cpp - ADG test: 2x2 diagonal mesh + tags -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("mixed_diag_with_tags");

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
      .setPortCount(9, 9)
      .setType(taggedType);

  auto at = builder.newAddTag("tagger")
      .setValueType(Type::i32())
      .setTagType(tagType);

  auto dt = builder.newDelTag("untagger")
      .setInputType(taggedType);

  auto mesh = builder.buildMesh(2, 2, pe, sw, Topology::DiagonalMesh);
  auto at0 = builder.clone(at, "tag0");
  auto at1 = builder.clone(at, "tag1");
  auto at2 = builder.clone(at, "tag2");
  auto at3 = builder.clone(at, "tag3");
  auto at4 = builder.clone(at, "tag4");
  auto dt0 = builder.clone(dt, "untag0");

  // Module I/O: native in/out
  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto result = builder.addModuleOutput("result", Type::i32());

  // add_tag at input -> mesh -> del_tag at output
  builder.connectToModuleInput(a, at0, 0);
  builder.connectToModuleInput(b, at1, 0);
  builder.connectPorts(at0, 0, mesh.peGrid[0][0], 0);
  builder.connectPorts(at1, 0, mesh.peGrid[0][0], 1);

  // Chain PEs in row-major order: [0][0] -> [0][1] -> [1][0] -> [1][1]
  builder.connectPorts(mesh.peGrid[0][0], 0, mesh.peGrid[0][1], 0);
  builder.connectToModuleInput(c, at2, 0);
  builder.connectPorts(at2, 0, mesh.peGrid[0][1], 1);
  builder.connectPorts(mesh.peGrid[0][1], 0, mesh.peGrid[1][0], 0);
  builder.connectToModuleInput(c, at3, 0);
  builder.connectPorts(at3, 0, mesh.peGrid[1][0], 1);
  builder.connectPorts(mesh.peGrid[1][0], 0, mesh.peGrid[1][1], 0);
  builder.connectToModuleInput(d, at4, 0);
  builder.connectPorts(at4, 0, mesh.peGrid[1][1], 1);

  builder.connectPorts(mesh.peGrid[1][1], 0, dt0, 0);
  builder.connectToModuleOutput(dt0, 0, result);

  builder.exportMLIR("Output/mixed_diag_with_tags.fabric.mlir");
  return 0;
}

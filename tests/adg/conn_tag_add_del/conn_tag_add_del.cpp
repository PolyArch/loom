//===-- conn_tag_add_del.cpp - ADG test: add_tag -> PE -> del_tag -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_tag_add_del");

  auto at = builder.newAddTag("tagger")
      .setValueType(Type::i32())
      .setTagType(Type::iN(4));

  auto dt = builder.newDelTag("untagger")
      .setInputType(Type::tagged(Type::i32(), Type::iN(4)));

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setInputPorts({taggedType, taggedType})
      .setOutputPorts({taggedType})
      .addOp("arith.addi");

  auto at0 = builder.clone(at, "tag0");
  auto at1 = builder.clone(at, "tag1");
  auto add0 = builder.clone(adder, "add0");
  auto dt0 = builder.clone(dt, "untag0");

  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(a, at0, 0);
  builder.connectToModuleInput(b, at1, 0);
  builder.connectPorts(at0, 0, add0, 0);
  builder.connectPorts(at1, 0, add0, 1);
  builder.connectPorts(add0, 0, dt0, 0);
  builder.connectToModuleOutput(dt0, 0, out);

  builder.exportMLIR("Output/conn_tag_add_del.fabric.mlir");
  return 0;
}

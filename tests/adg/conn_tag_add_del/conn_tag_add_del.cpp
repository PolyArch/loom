//===-- conn_tag_add_del.cpp - ADG test: add_tag -> PE -> del_tag -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_tag_add_del");

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setInputPorts({taggedType, taggedType})
      .setOutputPorts({taggedType})
      .addOp("arith.addi");

  InstanceHandle at0 = builder.newAddTag("tag0")
      .setValueType(Type::i32()).setTagType(Type::iN(4));
  InstanceHandle at1 = builder.newAddTag("tag1")
      .setValueType(Type::i32()).setTagType(Type::iN(4));
  auto add0 = builder.clone(adder, "add0");
  InstanceHandle dt0 = builder.newDelTag("untag0")
      .setInputType(taggedType);

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

//===-- conn_tag_map_chain.cpp - ADG test: add -> map -> PE -> del -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_tag_map_chain");

  auto taggedI2 = Type::tagged(Type::i32(), Type::iN(2));

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setInputPorts({taggedI2, taggedI2})
      .setOutputPorts({taggedI2})
      .addOp("arith.addi");

  InstanceHandle at0 = builder.newAddTag("tag0")
      .setValueType(Type::i32()).setTagType(Type::iN(4));
  InstanceHandle at1 = builder.newAddTag("tag1")
      .setValueType(Type::i32()).setTagType(Type::iN(4));
  InstanceHandle mt0 = builder.newMapTag("map0")
      .setValueType(Type::i32()).setInputTagType(Type::iN(4))
      .setOutputTagType(Type::iN(2)).setTableSize(16);
  InstanceHandle mt1 = builder.newMapTag("map1")
      .setValueType(Type::i32()).setInputTagType(Type::iN(4))
      .setOutputTagType(Type::iN(2)).setTableSize(16);
  auto add0 = builder.clone(adder, "add0");
  InstanceHandle dt0 = builder.newDelTag("untag0")
      .setInputType(taggedI2);

  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(a, at0, 0);
  builder.connectToModuleInput(b, at1, 0);
  builder.connectPorts(at0, 0, mt0, 0);
  builder.connectPorts(at1, 0, mt1, 0);
  builder.connectPorts(mt0, 0, add0, 0);
  builder.connectPorts(mt1, 0, add0, 1);
  builder.connectPorts(add0, 0, dt0, 0);
  builder.connectToModuleOutput(dt0, 0, out);

  builder.exportMLIR("Output/conn_tag_map_chain.fabric.mlir");
  return 0;
}

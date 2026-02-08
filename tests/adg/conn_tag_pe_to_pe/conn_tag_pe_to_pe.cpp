//===-- conn_tag_pe_to_pe.cpp - ADG test: tagged PE chain -----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_tag_pe_to_pe");

  auto tagType = Type::iN(4);
  auto taggedI32 = Type::tagged(Type::i32(), tagType);

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setInputPorts({taggedI32, taggedI32})
      .setOutputPorts({taggedI32})
      .addOp("arith.addi");

  auto mul = builder.newPE("mul")
      .setLatency(2, 2, 2)
      .setInterval(1, 1, 1)
      .setInterfaceCategory(InterfaceCategory::Tagged)
      .setInputPorts({taggedI32, taggedI32})
      .setOutputPorts({taggedI32})
      .addOp("arith.muli");

  auto add0 = builder.clone(adder, "add0");
  auto mul0 = builder.clone(mul, "mul0");

  auto a = builder.addModuleInput("a", taggedI32);
  auto b = builder.addModuleInput("b", taggedI32);
  auto c = builder.addModuleInput("c", taggedI32);
  auto out = builder.addModuleOutput("result", taggedI32);

  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);
  builder.connectPorts(add0, 0, mul0, 0);
  builder.connectToModuleInput(c, mul0, 1);
  builder.connectToModuleOutput(mul0, 0, out);

  builder.exportMLIR("Output/conn_tag_pe_to_pe.fabric.mlir");
  return 0;
}

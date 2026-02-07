//===-- conn_pe_to_pe.cpp - ADG test: PE -> PE connection ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_pe_to_pe");

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto mul = builder.newPE("mul")
      .setLatency(2, 2, 2)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.muli");

  auto add0 = builder.clone(adder, "add0");
  auto mul0 = builder.clone(mul, "mul0");

  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);
  builder.connectPorts(add0, 0, mul0, 0);
  builder.connectToModuleInput(c, mul0, 1);
  builder.connectToModuleOutput(mul0, 0, out);

  builder.exportMLIR("Output/conn_pe_to_pe.fabric.mlir");
  return 0;
}

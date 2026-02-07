//===-- conn_dataflow_stream.cpp - ADG test: 3-PE arithmetic chain -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_dataflow_stream");

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

  auto sub = builder.newPE("sub")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.subi");

  // Chain: (a + b) * c - d
  auto add0 = builder.clone(adder, "add0");
  auto mul0 = builder.clone(mul, "mul0");
  auto sub0 = builder.clone(sub, "sub0");

  auto a = builder.addModuleInput("a", Type::i32());
  auto b = builder.addModuleInput("b", Type::i32());
  auto c = builder.addModuleInput("c", Type::i32());
  auto d = builder.addModuleInput("d", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  // PE0: a + b
  builder.connectToModuleInput(a, add0, 0);
  builder.connectToModuleInput(b, add0, 1);
  // PE1: (a+b) * c
  builder.connectPorts(add0, 0, mul0, 0);
  builder.connectToModuleInput(c, mul0, 1);
  // PE2: ((a+b)*c) - d
  builder.connectPorts(mul0, 0, sub0, 0);
  builder.connectToModuleInput(d, sub0, 1);
  builder.connectToModuleOutput(sub0, 0, out);

  builder.exportMLIR("Output/conn_dataflow_stream.fabric.mlir");
  return 0;
}

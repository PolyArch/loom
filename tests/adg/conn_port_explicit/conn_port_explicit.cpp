//===-- conn_port_explicit.cpp - ADG test: connectPorts non-zero -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_port_explicit");

  auto sw = builder.newSwitch("sw_3x3")
      .setPortCount(3, 3)
      .setType(Type::i32());

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw0 = builder.clone(sw, "sw0");
  auto add0 = builder.clone(adder, "add0");

  auto x0 = builder.addModuleInput("x0", Type::i32());
  auto x1 = builder.addModuleInput("x1", Type::i32());
  auto x2 = builder.addModuleInput("x2", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  builder.connectToModuleInput(x0, sw0, 0);
  builder.connectToModuleInput(x1, sw0, 1);
  builder.connectToModuleInput(x2, sw0, 2);
  // Use non-zero output ports
  builder.connectPorts(sw0, 1, add0, 0);
  builder.connectPorts(sw0, 2, add0, 1);
  builder.connectToModuleOutput(add0, 0, out);

  builder.exportMLIR("Output/conn_port_explicit.fabric.mlir");
  return 0;
}

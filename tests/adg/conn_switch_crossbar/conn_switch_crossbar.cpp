//===-- conn_switch_crossbar.cpp - ADG test: full crossbar ----*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_switch_crossbar");

  auto adder = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("sw_3x3")
      .setPortCount(3, 3)
      .setType(Type::i32());

  auto a0 = builder.clone(adder, "a0");
  auto a1 = builder.clone(adder, "a1");
  auto sw0 = builder.clone(sw, "sw0");
  auto a2 = builder.clone(adder, "a2");

  auto x0 = builder.addModuleInput("x0", Type::i32());
  auto x1 = builder.addModuleInput("x1", Type::i32());
  auto x2 = builder.addModuleInput("x2", Type::i32());
  auto x3 = builder.addModuleInput("x3", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  auto bcast_0_sw_def = builder.newSwitch("bcast_0_sw")
      .setPortCount(1, 2)
      .setType(Type::i32());
  auto bcast_0 = builder.clone(bcast_0_sw_def, "bcast_0");
  builder.connectToModuleInput(x0, bcast_0, 0);
  builder.connectPorts(bcast_0, 0, a0, 0);
  builder.connectPorts(bcast_0, 1, sw0, 2);
  builder.connectToModuleInput(x1, a0, 1);
  builder.connectToModuleInput(x2, a1, 0);
  builder.connectToModuleInput(x3, a1, 1);
  builder.connectPorts(a0, 0, sw0, 0);
  builder.connectPorts(a1, 0, sw0, 1);
  builder.connectPorts(sw0, 0, a2, 0);
  builder.connectPorts(sw0, 1, a2, 1);
  builder.connectToModuleOutput(a2, 0, out);
  // Connect remaining switch output port to avoid dangling.
  auto sink = builder.addModuleOutput("sw0_out2", Type::i32());
  builder.connectToModuleOutput(sw0, 2, sink);

  builder.exportMLIR("Output/conn_switch_crossbar.fabric.mlir");
  return 0;
}

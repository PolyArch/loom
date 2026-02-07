//===-- conn_temporal_sw_io.cpp - ADG test: temporal switch between PEs -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_temporal_sw_io");

  auto tagType = Type::iN(4);
  auto taggedType = Type::tagged(Type::i32(), tagType);

  // Native PE as FU for temporal PEs
  auto fu = builder.newPE("fu_add")
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto tpe = builder.newTemporalPE("tpe_add")
      .setNumRegisters(0)
      .setNumInstructions(2)
      .setRegFifoDepth(0)
      .setInterface(taggedType)
      .addFU(fu);

  auto tsw = builder.newTemporalSwitch("tsw_2x2")
      .setNumRouteTable(2)
      .setPortCount(2, 2)
      .setInterface(taggedType);

  // Two temporal PEs with a temporal switch in between
  auto tpe0 = builder.clone(tpe, "tpe0");
  auto tsw0 = builder.clone(tsw, "tsw0");
  auto tpe1 = builder.clone(tpe, "tpe1");

  auto a = builder.addModuleInput("a", taggedType);
  auto b = builder.addModuleInput("b", taggedType);
  auto c = builder.addModuleInput("c", taggedType);
  auto out = builder.addModuleOutput("result", taggedType);

  // tpe0: a + b
  builder.connectToModuleInput(a, tpe0, 0);
  builder.connectToModuleInput(b, tpe0, 1);
  // tpe0 output -> tsw0 input 0, c -> tsw0 input 1
  builder.connectPorts(tpe0, 0, tsw0, 0);
  builder.connectToModuleInput(c, tsw0, 1);
  // tsw0 output 0 -> tpe1 input 0, tsw0 output 1 -> tpe1 input 1
  builder.connectPorts(tsw0, 0, tpe1, 0);
  builder.connectPorts(tsw0, 1, tpe1, 1);
  builder.connectToModuleOutput(tpe1, 0, out);

  builder.exportMLIR("Output/conn_temporal_sw_io.fabric.mlir");
  return 0;
}

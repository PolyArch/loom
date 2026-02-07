//===-- conn_fifo_in_cycle.cpp - ADG test: fifo breaks switch cycle -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Two switches with a feedback loop. Without a fifo the combinational loop
// check would fail. Inserting a fifo on the back edge breaks the loop.
//
//===----------------------------------------------------------------------===//

#include <loom/Hardware/adg.h>

#include <cassert>
#include <fstream>
#include <string>

static unsigned mlirCount(const std::string &path, const std::string &substr) {
  std::ifstream f(path);
  std::string content((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  unsigned count = 0;
  size_t pos = 0;
  while ((pos = content.find(substr, pos)) != std::string::npos) {
    ++count;
    pos += substr.size();
  }
  return count;
}

using namespace loom::adg;

int main() {
  ADGBuilder builder("conn_fifo_in_cycle");

  // 2-port switches: 2 inputs, 2 outputs.
  auto sw = builder.newSwitch("xbar")
      .setPortCount(2, 2)
      .setType(Type::i32());

  auto fifo = builder.newFifo("buf");
  fifo.setDepth(2).setType(Type::i32());

  auto sw0 = builder.clone(sw, "sw0");
  auto sw1 = builder.clone(sw, "sw1");
  auto f0 = builder.clone(fifo, "f0");

  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto out0 = builder.addModuleOutput("out0", Type::i32());

  // sw0(in0, fifo_out) -> (to_sw1, out0)
  // sw1(from_sw0, in_dummy) -> (to_fifo, out_dummy)
  // fifo: sw1 out 0 -> fifo -> sw0 in 1 (back edge, broken by fifo)
  builder.connectToModuleInput(in0, sw0, 0);     // module in -> sw0 in 0
  builder.connectPorts(f0, 0, sw0, 1);            // fifo out -> sw0 in 1
  builder.connectPorts(sw0, 0, sw1, 0);           // sw0 out 0 -> sw1 in 0
  builder.connectToModuleOutput(sw0, 1, out0);    // sw0 out 1 -> module out

  // sw1 needs both inputs connected.
  auto in1 = builder.addModuleInput("in1", Type::i32());
  builder.connectToModuleInput(in1, sw1, 1);      // module in -> sw1 in 1
  builder.connectPorts(sw1, 0, f0, 0);            // sw1 out 0 -> fifo (back edge)
  auto out1 = builder.addModuleOutput("out1", Type::i32());
  builder.connectToModuleOutput(sw1, 1, out1);    // sw1 out 1 -> module out

  auto validation = builder.validateADG();
  assert(validation.success && "validation should pass with fifo breaking cycle");

  builder.exportMLIR("Output/conn_fifo_in_cycle.fabric.mlir");

  const char *mlir = "Output/conn_fifo_in_cycle.fabric.mlir";
  assert(mlirCount(mlir, "fabric.switch") == 2 && "expected 2 switch instances");
  assert(mlirCount(mlir, "fabric.fifo") == 1 && "expected 1 fifo instance");

  return 0;
}

//===-- fifo_depth_zero_validation.cpp - ADG test: fifo depth=0 fails -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Validates that ADG builder rejects a fifo with depth=0, then builds a valid
// ADG with depth=1 for MLIR export.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

#include <cassert>
#include <string>

using namespace loom::adg;

int main() {
  // Part 1: Verify depth=0 is rejected by validateADG.
  {
    ADGBuilder builder("bad_fifo_depth_zero");

    auto pe = builder.newPE("adder")
        .setLatency(1, 1, 1)
        .setInterval(1, 1, 1)
        .setInputPorts({Type::i32(), Type::i32()})
        .setOutputPorts({Type::i32()})
        .addOp("arith.addi");

    auto fifo = builder.newFifo("bad_fifo");
    fifo.setDepth(0).setType(Type::i32());

    auto p0 = builder.clone(pe, "p0");
    auto f0 = builder.clone(fifo, "f0");

    auto in0 = builder.addModuleInput("in0", Type::i32());
    auto in1 = builder.addModuleInput("in1", Type::i32());
    auto out = builder.addModuleOutput("result", Type::i32());

    builder.connectToModuleInput(in0, p0, 0);
    builder.connectToModuleInput(in1, p0, 1);
    builder.connectPorts(p0, 0, f0, 0);
    builder.connectToModuleOutput(f0, 0, out);

    auto validation = builder.validateADG();
    assert(!validation.success && "validation should fail for depth=0 fifo");

    bool foundDepthZero = false;
    for (const auto &err : validation.errors) {
      if (err.code == "CPL_FIFO_DEPTH_ZERO") {
        foundDepthZero = true;
        break;
      }
    }
    assert(foundDepthZero && "expected CPL_FIFO_DEPTH_ZERO error");
  }

  // Part 2: Build a valid ADG with depth=1 and export MLIR.
  {
    ADGBuilder builder("fifo_depth_zero_validation");

    auto pe = builder.newPE("adder")
        .setLatency(1, 1, 1)
        .setInterval(1, 1, 1)
        .setInputPorts({Type::i32(), Type::i32()})
        .setOutputPorts({Type::i32()})
        .addOp("arith.addi");

    auto fifo = builder.newFifo("good_fifo");
    fifo.setDepth(1).setType(Type::i32());

    auto p0 = builder.clone(pe, "p0");
    auto f0 = builder.clone(fifo, "f0");

    auto in0 = builder.addModuleInput("in0", Type::i32());
    auto in1 = builder.addModuleInput("in1", Type::i32());
    auto out = builder.addModuleOutput("result", Type::i32());

    builder.connectToModuleInput(in0, p0, 0);
    builder.connectToModuleInput(in1, p0, 1);
    builder.connectPorts(p0, 0, f0, 0);
    builder.connectToModuleOutput(f0, 0, out);

    auto validation = builder.validateADG();
    assert(validation.success && "validation should pass for depth=1 fifo");

    builder.exportMLIR("Output/fifo_depth_zero_validation.fabric.mlir");
  }

  return 0;
}

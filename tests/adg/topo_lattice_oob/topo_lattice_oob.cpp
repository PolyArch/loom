//===-- topo_lattice_oob.cpp - ADG test: out-of-bounds placement --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
// Negative test: out-of-bounds cell placement should trigger error.

#include <loom/adg.h>

#include <cassert>
#include <sys/wait.h>
#include <unistd.h>

using namespace loom::adg;

static bool expectError(void (*fn)()) {
  pid_t pid = fork();
  if (pid == 0) {
    fn();
    _exit(0);
  }
  int status = 0;
  waitpid(pid, &status, 0);
  return WIFEXITED(status) && WEXITSTATUS(status) != 0;
}

static void outOfBoundsRow() {
  ADGBuilder builder("test");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(8, 8)
      .setType(Type::i32());

  auto lattice = builder.latticeMesh(2, 2, sw);
  builder.placePEInLattice(lattice, 5, 0, pe, "pe_oob");
}

static void outOfBoundsCol() {
  ADGBuilder builder("test");

  auto pe = builder.newPE("alu")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto sw = builder.newSwitch("xbar")
      .setPortCount(8, 8)
      .setType(Type::i32());

  auto lattice = builder.latticeMesh(2, 2, sw);
  builder.placePEInLattice(lattice, 0, 10, pe, "pe_oob");
}

int main() {
  assert(expectError(outOfBoundsRow) && "row out-of-bounds should fail");
  assert(expectError(outOfBoundsCol) && "col out-of-bounds should fail");

  // Build a minimal valid ADG so the test harness MLIR check passes.
  {
    ADGBuilder b("topo_lattice_oob");
    auto pe = b.newPE("p")
        .setLatency(1, 1, 1).setInterval(1, 1, 1)
        .setInputPorts({Type::i32(), Type::i32()})
        .setOutputPorts({Type::i32()})
        .addOp("arith.addi");
    auto inst = b.clone(pe, "i0");
    auto a = b.addModuleInput("a", Type::i32());
    auto c = b.addModuleInput("c", Type::i32());
    auto out = b.addModuleOutput("out", Type::i32());
    b.connectToModuleInput(a, inst, 0);
    b.connectToModuleInput(c, inst, 1);
    b.connectToModuleOutput(inst, 0, out);
    b.exportMLIR("Output/topo_lattice_oob.fabric.mlir");
  }
  return 0;
}

//===-- fifo_tagged.cpp - ADG test: fifo with tagged type ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include <loom/adg.h>

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
  ADGBuilder builder("fifo_tagged");

  auto taggedType = Type::tagged(Type::i32(), Type::iN(4));

  auto fifo = builder.newFifo("tbuf");
  fifo.setDepth(2).setType(taggedType);

  auto addTag = builder.newAddTag("tagger");
  addTag.setValueType(Type::i32()).setTagType(Type::iN(4));

  auto delTag = builder.newDelTag("detagger");
  delTag.setInputType(taggedType);

  auto pe = builder.newPE("adder")
      .setLatency(1, 1, 1)
      .setInterval(1, 1, 1)
      .setInputPorts({Type::i32(), Type::i32()})
      .setOutputPorts({Type::i32()})
      .addOp("arith.addi");

  auto p0 = builder.clone(pe, "p0");
  InstanceHandle at = addTag;
  auto f0 = builder.clone(fifo, "f0");
  InstanceHandle dt = delTag;
  auto p1 = builder.clone(pe, "p1");

  auto in0 = builder.addModuleInput("in0", Type::i32());
  auto in1 = builder.addModuleInput("in1", Type::i32());
  auto in2 = builder.addModuleInput("in2", Type::i32());
  auto out = builder.addModuleOutput("result", Type::i32());

  // Pipeline: p0 -> add_tag -> fifo(tagged) -> del_tag -> p1
  builder.connectToModuleInput(in0, p0, 0);
  builder.connectToModuleInput(in1, p0, 1);
  builder.connectPorts(p0, 0, at, 0);    // value -> add_tag (tag is hardware param)
  builder.connectPorts(at, 0, f0, 0);    // tagged -> fifo
  builder.connectPorts(f0, 0, dt, 0);    // fifo -> del_tag
  builder.connectPorts(dt, 0, p1, 0);    // value -> p1
  builder.connectToModuleInput(in2, p1, 1);
  builder.connectToModuleOutput(p1, 0, out);

  auto validation = builder.validateADG();
  assert(validation.success && "validation failed");

  builder.exportMLIR("Output/fifo_tagged.fabric.mlir");

  const char *mlir = "Output/fifo_tagged.fabric.mlir";
  assert(mlirCount(mlir, "fabric.fifo") == 1 && "expected 1 fifo instance");
  assert(mlirCount(mlir, "!dataflow.tagged<i32, i4>") >= 1 &&
         "expected tagged type in fifo");

  return 0;
}

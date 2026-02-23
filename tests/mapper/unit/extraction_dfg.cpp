//===-- extraction_dfg.cpp - DFGBuilder extraction tests ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Exercises the real DFGBuilder entry point by parsing minimal handshake MLIR
// strings and running DFGBuilder::build(), then asserting invariants on the
// resulting Graph.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/DFGBuilder.h"
#include "loom/Mapper/Graph.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

using namespace loom;

namespace {

/// Parse an inline MLIR module string and extract the first handshake.func.
/// Returns nullptr on failure.
circt::handshake::FuncOp parseFuncOp(mlir::MLIRContext &ctx,
                                     llvm::StringRef mlirSource,
                                     mlir::OwningOpRef<mlir::ModuleOp> &module) {
  module = mlir::parseSourceString<mlir::ModuleOp>(mlirSource,
                                                   mlir::ParserConfig(&ctx));
  if (!module)
    return nullptr;

  circt::handshake::FuncOp funcOp;
  module->walk([&](circt::handshake::FuncOp f) {
    if (!funcOp)
      funcOp = f;
  });
  return funcOp;
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.allowUnregisteredDialects();

  // Test 1: Simple 2-op chain DFG (arith.addi -> arith.addi)
  //
  // Graph structure:
  //   arg0 (ModuleInputNode) --\
  //                             +-> addi0 (OperationNode) --\
  //   arg1 (ModuleInputNode) --/                            +-> addi1 ---> return
  //                                                         |
  //   ctrl (ModuleInputNode) --------------------------------+-----------> return
  //
  // Expected nodes: 2 ops + 3 input sentinels + 2 output sentinels = 7
  // Expected edges: arg0->addi0, arg1->addi0, addi0->addi1, arg0->addi1,
  //                 addi1->out0, ctrl->out1 = 6
  {
    llvm::StringRef mlirSource = R"MLIR(
      module {
        handshake.func @test(%arg0: i32, %arg1: i32, %ctrl: none) -> (i32, none)
            attributes {argNames = ["arg0", "arg1", "ctrl"],
                        resNames = ["out0", "ctrl"]} {
          %0 = arith.addi %arg0, %arg1 : i32
          %1 = arith.addi %0, %arg0 : i32
          handshake.return %1, %ctrl : i32, none
        }
      }
    )MLIR";

    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto funcOp = parseFuncOp(ctx, mlirSource, module);
    TEST_ASSERT(funcOp != nullptr);

    DFGBuilder builder;
    Graph dfg = builder.build(funcOp);

    // Count nodes by kind.
    int opCount = 0, inputCount = 0, outputCount = 0;
    for (auto *node : dfg.nodeRange()) {
      if (node->kind == Node::OperationNode)
        opCount++;
      else if (node->kind == Node::ModuleInputNode)
        inputCount++;
      else if (node->kind == Node::ModuleOutputNode)
        outputCount++;
    }

    // 2 arith.addi ops + 3 block args (arg0, arg1, ctrl) + 2 return operands.
    TEST_ASSERT(opCount == 2);
    TEST_ASSERT(inputCount == 3);
    TEST_ASSERT(outputCount == 2);
    TEST_ASSERT(dfg.countNodes() == 7);

    // Verify edge direction invariant: srcPort is Output, dstPort is Input.
    for (auto *edge : dfg.edgeRange()) {
      Port *srcPort = dfg.getPort(edge->srcPort);
      Port *dstPort = dfg.getPort(edge->dstPort);
      TEST_ASSERT(srcPort != nullptr);
      TEST_ASSERT(dstPort != nullptr);
      TEST_ASSERT(srcPort->direction == Port::Output);
      TEST_ASSERT(dstPort->direction == Port::Input);
    }

    // Edges: arg0->addi0, arg1->addi0, addi0->addi1, arg0->addi1,
    //        addi1->out0, ctrl->out1 = 6 edges.
    TEST_ASSERT(dfg.countEdges() == 6);
  }

  // Test 2: Fan-out DFG
  //
  // One operation result feeds two consumers.
  //
  // Graph structure:
  //   arg0, arg1 -> addi -> {muli(addi, addi), return}
  //   The addi result is used twice by muli (both operands) = fan-out of 3
  //   (2 to muli inputs + 1 to return via muli).
  {
    llvm::StringRef mlirSource = R"MLIR(
      module {
        handshake.func @fanout(%arg0: i32, %arg1: i32, %ctrl: none) -> (i32, none)
            attributes {argNames = ["arg0", "arg1", "ctrl"],
                        resNames = ["out0", "ctrl"]} {
          %0 = arith.addi %arg0, %arg1 : i32
          %1 = arith.muli %0, %0 : i32
          handshake.return %1, %ctrl : i32, none
        }
      }
    )MLIR";

    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto funcOp = parseFuncOp(ctx, mlirSource, module);
    TEST_ASSERT(funcOp != nullptr);

    DFGBuilder builder;
    Graph dfg = builder.build(funcOp);

    // Find the addi operation node (producer) by checking op_name attribute.
    Node *addiNode = nullptr;
    for (auto *node : dfg.nodeRange()) {
      if (node->kind != Node::OperationNode)
        continue;
      for (auto &attr : node->attributes) {
        if (attr.getName().str() == "op_name" &&
            mlir::dyn_cast<mlir::StringAttr>(attr.getValue())
                .getValue() == "arith.addi") {
          addiNode = node;
          break;
        }
      }
      if (addiNode)
        break;
    }
    TEST_ASSERT(addiNode != nullptr);

    // addi has 1 output port. That port feeds muli's 2 inputs = fan-out of 2.
    TEST_ASSERT(addiNode->outputPorts.size() == 1);
    IdIndex addiOutPort = addiNode->outputPorts[0];
    Port *outPort = dfg.getPort(addiOutPort);
    TEST_ASSERT(outPort != nullptr);
    TEST_ASSERT(outPort->connectedEdges.size() == 2);

    // Each connected edge's dstPort must belong to the muli node.
    for (IdIndex edgeId : outPort->connectedEdges) {
      Edge *edge = dfg.getEdge(edgeId);
      TEST_ASSERT(edge != nullptr);
      Port *dstPort = dfg.getPort(edge->dstPort);
      TEST_ASSERT(dstPort != nullptr);
      Node *consumer = dfg.getNode(dstPort->parentNode);
      TEST_ASSERT(consumer != nullptr);
      TEST_ASSERT(consumer->kind == Node::OperationNode);
    }
  }

  // Test 3: Sentinel port directions
  //
  // After building any DFG, verify:
  //   - Every ModuleInputNode has NO input ports and at least 1 output port
  //   - Every ModuleOutputNode has at least 1 input port and NO output ports
  {
    llvm::StringRef mlirSource = R"MLIR(
      module {
        handshake.func @sentinel(%a: i32, %b: i32, %c: i32, %ctrl: none) -> (i32, i32, none)
            attributes {argNames = ["a", "b", "c", "ctrl"],
                        resNames = ["out0", "out1", "ctrl"]} {
          %0 = arith.addi %a, %b : i32
          %1 = arith.muli %b, %c : i32
          handshake.return %0, %1, %ctrl : i32, i32, none
        }
      }
    )MLIR";

    mlir::OwningOpRef<mlir::ModuleOp> module;
    auto funcOp = parseFuncOp(ctx, mlirSource, module);
    TEST_ASSERT(funcOp != nullptr);

    DFGBuilder builder;
    Graph dfg = builder.build(funcOp);

    int inputSentinelCount = 0;
    int outputSentinelCount = 0;

    for (auto *node : dfg.nodeRange()) {
      if (node->kind == Node::ModuleInputNode) {
        inputSentinelCount++;
        // Must have NO input ports.
        TEST_ASSERT(node->inputPorts.empty());
        // Must have at least 1 output port.
        TEST_ASSERT(!node->outputPorts.empty());
        // All ports must be Output direction.
        for (IdIndex pid : node->outputPorts) {
          Port *p = dfg.getPort(pid);
          TEST_ASSERT(p != nullptr);
          TEST_ASSERT(p->direction == Port::Output);
          TEST_ASSERT(p->parentNode != INVALID_ID);
        }
      } else if (node->kind == Node::ModuleOutputNode) {
        outputSentinelCount++;
        // Must have at least 1 input port.
        TEST_ASSERT(!node->inputPorts.empty());
        // Must have NO output ports.
        TEST_ASSERT(node->outputPorts.empty());
        // All ports must be Input direction.
        for (IdIndex pid : node->inputPorts) {
          Port *p = dfg.getPort(pid);
          TEST_ASSERT(p != nullptr);
          TEST_ASSERT(p->direction == Port::Input);
          TEST_ASSERT(p->parentNode != INVALID_ID);
        }
      }
    }

    // 4 block args (a, b, c, ctrl) -> 4 input sentinels.
    TEST_ASSERT(inputSentinelCount == 4);
    // 3 return operands (out0, out1, ctrl) -> 3 output sentinels.
    TEST_ASSERT(outputSentinelCount == 3);
  }

  return 0;
}

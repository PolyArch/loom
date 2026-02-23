//===-- extraction_adg.cpp - ADGFlattener real-template test --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Exercises the real ADGFlattener entry point by parsing the actual
// loom_cgra_small.fabric.mlir template, running ADGFlattener::flatten(), and
// asserting structural invariants on the resulting Graph and ConnectivityMatrix.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/ADGFlattener.h"
#include "loom/Mapper/ConnectivityMatrix.h"
#include "loom/Mapper/Graph.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"
#include "loom/Dialect/Fabric/FabricOps.h"
#include "circt/Dialect/Handshake/HandshakeDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

using namespace loom;

namespace {

/// Helper to retrieve a string attribute from a node by name.
llvm::StringRef getAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

} // namespace

#ifndef LOOM_SRC_DIR
#define LOOM_SRC_DIR "."
#endif

int main() {
  // --- Setup: parse the real template file ---
  mlir::MLIRContext ctx;
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx.allowUnregisteredDialects();

  std::string templatePath =
      std::string(LOOM_SRC_DIR) +
      "/tests/mapper-app/templates/loom_cgra_small.fabric.mlir";
  mlir::ParserConfig parserConfig(&ctx);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(templatePath,
                                                       parserConfig);
  TEST_ASSERT(module);

  // Find the first fabric.module inside the parsed builtin module.
  loom::fabric::ModuleOp fabricMod;
  module->walk([&](loom::fabric::ModuleOp m) {
    if (!fabricMod)
      fabricMod = m;
  });
  TEST_ASSERT(fabricMod);

  // Run ADGFlattener on the real template.
  ADGFlattener flattener;
  Graph adg = flattener.flatten(fabricMod);
  const ConnectivityMatrix &matrix = flattener.getConnectivityMatrix();

  // Test 1: Parse real template and verify node counts.
  // The template must produce at least one OperationNode (PEs/switches exist)
  // and at least some edges (hardware connections).
  {
    size_t opNodeCount = 0;
    size_t inputSentinelCount = 0;
    size_t outputSentinelCount = 0;
    size_t functionalCount = 0;
    size_t routingCount = 0;

    for (auto *node : adg.nodeRange()) {
      if (node->kind == Node::OperationNode) {
        ++opNodeCount;
        llvm::StringRef resClass = getAttr(node, "resource_class");
        if (resClass == "functional")
          ++functionalCount;
        else if (resClass == "routing")
          ++routingCount;
      } else if (node->kind == Node::ModuleInputNode) {
        ++inputSentinelCount;
      } else if (node->kind == Node::ModuleOutputNode) {
        ++outputSentinelCount;
      }
    }

    // At least one OperationNode must exist.
    TEST_ASSERT(opNodeCount > 0);

    // At least some edges must exist.
    TEST_ASSERT(adg.countEdges() > 0);

    // Both functional and routing nodes must be present.
    TEST_ASSERT(functionalCount > 0);
    TEST_ASSERT(routingCount > 0);

    // The template has block arguments, so input sentinels must exist.
    TEST_ASSERT(inputSentinelCount > 0);

    // The template has fabric.yield operands, so output sentinels must exist.
    TEST_ASSERT(outputSentinelCount > 0);
  }

  // Test 2: ConnectivityMatrix outToIn is non-empty.
  // Every hardware edge produces an outToIn entry.
  {
    TEST_ASSERT(matrix.outToIn.size() > 0);
  }

  // Test 3: ConnectivityMatrix inToOut exists for routing nodes.
  // Find a routing node, get one of its input ports, and verify that
  // inToOut has entries for it.
  {
    bool foundRoutingWithInToOut = false;

    for (auto *node : adg.nodeRange()) {
      if (node->kind != Node::OperationNode)
        continue;

      llvm::StringRef resClass = getAttr(node, "resource_class");
      if (resClass != "routing")
        continue;

      // Routing nodes with input ports should have inToOut entries.
      if (node->inputPorts.empty())
        continue;

      IdIndex firstInputPort = node->inputPorts[0];
      if (matrix.inToOut.count(firstInputPort)) {
        TEST_ASSERT(!matrix.inToOut.find(firstInputPort)->second.empty());
        foundRoutingWithInToOut = true;
        break;
      }
    }

    TEST_ASSERT(foundRoutingWithInToOut);
  }

  // Test 4: Sentinel node invariants.
  // ModuleInputNode sentinels have output ports only (no input ports).
  // ModuleOutputNode sentinels have input ports only (no output ports).
  {
    for (auto *node : adg.nodeRange()) {
      if (node->kind == Node::ModuleInputNode) {
        TEST_ASSERT(node->inputPorts.empty());
        TEST_ASSERT(!node->outputPorts.empty());
        for (IdIndex portId : node->outputPorts) {
          Port *port = adg.getPort(portId);
          TEST_ASSERT(port != nullptr);
          TEST_ASSERT(port->direction == Port::Output);
        }
      } else if (node->kind == Node::ModuleOutputNode) {
        TEST_ASSERT(node->outputPorts.empty());
        TEST_ASSERT(!node->inputPorts.empty());
        for (IdIndex portId : node->inputPorts) {
          Port *port = adg.getPort(portId);
          TEST_ASSERT(port != nullptr);
          TEST_ASSERT(port->direction == Port::Input);
        }
      }
    }
  }

  // Test 5: Resource class coverage.
  // The small CGRA template must contain both "functional" and "routing"
  // resource classes among its OperationNodes.
  {
    bool hasFunctional = false;
    bool hasRouting = false;

    for (auto *node : adg.nodeRange()) {
      if (node->kind != Node::OperationNode)
        continue;

      llvm::StringRef resClass = getAttr(node, "resource_class");
      if (resClass == "functional")
        hasFunctional = true;
      if (resClass == "routing")
        hasRouting = true;
    }

    TEST_ASSERT(hasFunctional);
    TEST_ASSERT(hasRouting);
  }

  return 0;
}

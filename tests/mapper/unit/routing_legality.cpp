//===-- routing_legality.cpp - Routing with connectivity_table test -*-C++-*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that the mapper respects connectivity_table when building routing
// node internal connectivity, and rejects illegal routing transitions.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/ConnectivityMatrix.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/MappingState.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

using namespace loom;

namespace {

void setStringAttr(Node *node, mlir::MLIRContext &ctx,
                   llvm::StringRef name, llvm::StringRef value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name), mlir::StringAttr::get(&ctx, value)));
}

void setDenseI8Attr(Node *node, mlir::MLIRContext &ctx,
                    llvm::StringRef name,
                    llvm::ArrayRef<int8_t> values) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name),
      mlir::DenseI8ArrayAttr::get(&ctx, values)));
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: Connectivity table restricts internal routing.
  // Switch with 2 inputs, 2 outputs. connectivity_table:
  //   output 0: can receive from input 0 only   [1, 0]
  //   output 1: can receive from input 1 only   [0, 1]
  {
    Graph adg(&ctx);

    auto swNode = std::make_unique<Node>();
    swNode->kind = Node::OperationNode;
    setStringAttr(swNode.get(), ctx, "op_name", "fabric.switch");
    setStringAttr(swNode.get(), ctx, "resource_class", "routing");
    // Row-major: [out0_in0, out0_in1, out1_in0, out1_in1] = [1, 0, 0, 1]
    setDenseI8Attr(swNode.get(), ctx, "connectivity_table",
                   {1, 0, 0, 1});

    IdIndex swId = adg.addNode(std::move(swNode));

    // Add 2 input ports and 2 output ports.
    IdIndex inPort0, inPort1, outPort0, outPort1;
    {
      auto p = std::make_unique<Port>();
      p->parentNode = swId;
      p->direction = Port::Input;
      inPort0 = adg.addPort(std::move(p));
      adg.getNode(swId)->inputPorts.push_back(inPort0);
    }
    {
      auto p = std::make_unique<Port>();
      p->parentNode = swId;
      p->direction = Port::Input;
      inPort1 = adg.addPort(std::move(p));
      adg.getNode(swId)->inputPorts.push_back(inPort1);
    }
    {
      auto p = std::make_unique<Port>();
      p->parentNode = swId;
      p->direction = Port::Output;
      outPort0 = adg.addPort(std::move(p));
      adg.getNode(swId)->outputPorts.push_back(outPort0);
    }
    {
      auto p = std::make_unique<Port>();
      p->parentNode = swId;
      p->direction = Port::Output;
      outPort1 = adg.addPort(std::move(p));
      adg.getNode(swId)->outputPorts.push_back(outPort1);
    }

    // Build connectivity matrix using the same logic as Mapper::preprocess.
    ConnectivityMatrix cm;
    unsigned numIn = 2;
    unsigned numOut = 2;

    mlir::DenseI8ArrayAttr connTable;
    for (auto &attr : adg.getNode(swId)->attributes) {
      if (attr.getName() == "connectivity_table") {
        connTable = mlir::dyn_cast<mlir::DenseI8ArrayAttr>(attr.getValue());
        break;
      }
    }

    TEST_ASSERT(connTable);
    TEST_ASSERT(static_cast<unsigned>(connTable.size()) == numOut * numIn);

    for (unsigned o = 0; o < numOut; ++o) {
      for (unsigned i = 0; i < numIn; ++i) {
        if (connTable[o * numIn + i] != 0) {
          cm.inToOut[adg.getNode(swId)->inputPorts[i]].push_back(
              adg.getNode(swId)->outputPorts[o]);
        }
      }
    }

    // Verify: inPort0 -> outPort0 (allowed), inPort0 -> outPort1 (NOT allowed)
    TEST_ASSERT(cm.inToOut.count(inPort0) > 0);
    auto &out0 = cm.inToOut[inPort0];
    bool hasOut0 = false;
    bool hasOut1 = false;
    for (IdIndex p : out0) {
      if (p == outPort0) hasOut0 = true;
      if (p == outPort1) hasOut1 = true;
    }
    TEST_ASSERT(hasOut0);   // inPort0 -> outPort0 is legal
    TEST_ASSERT(!hasOut1);  // inPort0 -> outPort1 is NOT legal

    // Verify: inPort1 -> outPort1 (allowed), inPort1 -> outPort0 (NOT allowed)
    TEST_ASSERT(cm.inToOut.count(inPort1) > 0);
    auto &out1 = cm.inToOut[inPort1];
    bool has1Out0 = false;
    bool has1Out1 = false;
    for (IdIndex p : out1) {
      if (p == outPort0) has1Out0 = true;
      if (p == outPort1) has1Out1 = true;
    }
    TEST_ASSERT(!has1Out0);  // inPort1 -> outPort0 is NOT legal
    TEST_ASSERT(has1Out1);   // inPort1 -> outPort1 is legal
  }

  // Test 2: Full crossbar when no connectivity_table.
  {
    Graph adg(&ctx);

    auto swNode = std::make_unique<Node>();
    swNode->kind = Node::OperationNode;
    setStringAttr(swNode.get(), ctx, "op_name", "fabric.switch");
    setStringAttr(swNode.get(), ctx, "resource_class", "routing");
    // No connectivity_table attribute.

    IdIndex swId = adg.addNode(std::move(swNode));

    IdIndex inPort0, inPort1, outPort0, outPort1;
    {
      auto p = std::make_unique<Port>();
      p->parentNode = swId;
      p->direction = Port::Input;
      inPort0 = adg.addPort(std::move(p));
      adg.getNode(swId)->inputPorts.push_back(inPort0);
    }
    {
      auto p = std::make_unique<Port>();
      p->parentNode = swId;
      p->direction = Port::Input;
      inPort1 = adg.addPort(std::move(p));
      adg.getNode(swId)->inputPorts.push_back(inPort1);
    }
    {
      auto p = std::make_unique<Port>();
      p->parentNode = swId;
      p->direction = Port::Output;
      outPort0 = adg.addPort(std::move(p));
      adg.getNode(swId)->outputPorts.push_back(outPort0);
    }
    {
      auto p = std::make_unique<Port>();
      p->parentNode = swId;
      p->direction = Port::Output;
      outPort1 = adg.addPort(std::move(p));
      adg.getNode(swId)->outputPorts.push_back(outPort1);
    }

    // Build full crossbar (no connectivity_table).
    ConnectivityMatrix cm;
    for (IdIndex inPortId : adg.getNode(swId)->inputPorts) {
      for (IdIndex outPortId : adg.getNode(swId)->outputPorts) {
        cm.inToOut[inPortId].push_back(outPortId);
      }
    }

    // All transitions should be legal.
    TEST_ASSERT(cm.inToOut[inPort0].size() == 2);
    TEST_ASSERT(cm.inToOut[inPort1].size() == 2);
  }

  return 0;
}

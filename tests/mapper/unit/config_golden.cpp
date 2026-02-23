//===-- config_golden.cpp - Golden config bit-level semantics test -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verifies golden config semantics at the bit level. Uses manually-constructed
// graphs and MappingState to exercise ConfigGen and verify the actual binary
// config output.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/ConfigGen.h"
#include "loom/Mapper/Graph.h"
#include "loom/Mapper/Mapper.h"
#include "loom/Mapper/MappingState.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <fstream>
#include <vector>

using namespace loom;

namespace {

void setStringAttr(Node *node, mlir::MLIRContext &ctx,
                   llvm::StringRef name, llvm::StringRef value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name), mlir::StringAttr::get(&ctx, value)));
}

void setIntAttr(Node *node, mlir::MLIRContext &ctx,
                llvm::StringRef name, int64_t value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name),
      mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 64), value)));
}

IdIndex addOpNode(Graph &g, mlir::MLIRContext &ctx,
                  llvm::StringRef opName, llvm::StringRef resClass,
                  unsigned numIn = 1, unsigned numOut = 1) {
  auto node = std::make_unique<Node>();
  node->kind = Node::OperationNode;
  setStringAttr(node.get(), ctx, "op_name", opName);
  setStringAttr(node.get(), ctx, "resource_class", resClass);
  IdIndex nodeId = g.addNode(std::move(node));

  for (unsigned i = 0; i < numIn; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Input;
    IdIndex pid = g.addPort(std::move(port));
    g.getNode(nodeId)->inputPorts.push_back(pid);
  }
  for (unsigned i = 0; i < numOut; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = nodeId;
    port->direction = Port::Output;
    IdIndex pid = g.addPort(std::move(port));
    g.getNode(nodeId)->outputPorts.push_back(pid);
  }
  return nodeId;
}

IdIndex addPENode(Graph &adg, mlir::MLIRContext &ctx,
                  llvm::StringRef opName, llvm::StringRef resClass,
                  unsigned numIn, unsigned numOut) {
  auto hwNode = std::make_unique<Node>();
  hwNode->kind = Node::OperationNode;
  setStringAttr(hwNode.get(), ctx, "op_name", opName);
  setStringAttr(hwNode.get(), ctx, "resource_class", resClass);
  IdIndex hwId = adg.addNode(std::move(hwNode));

  for (unsigned i = 0; i < numIn; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = hwId;
    port->direction = Port::Input;
    IdIndex pid = adg.addPort(std::move(port));
    adg.getNode(hwId)->inputPorts.push_back(pid);
  }
  for (unsigned i = 0; i < numOut; ++i) {
    auto port = std::make_unique<Port>();
    port->parentNode = hwId;
    port->direction = Port::Output;
    IdIndex pid = adg.addPort(std::move(port));
    adg.getNode(hwId)->outputPorts.push_back(pid);
  }
  return hwId;
}

IdIndex addEdgeBetween(Graph &g, IdIndex srcNode, unsigned srcPortIdx,
                       IdIndex dstNode, unsigned dstPortIdx) {
  IdIndex srcPort = g.getNode(srcNode)->outputPorts[srcPortIdx];
  IdIndex dstPort = g.getNode(dstNode)->inputPorts[dstPortIdx];

  auto edge = std::make_unique<Edge>();
  edge->srcPort = srcPort;
  edge->dstPort = dstPort;
  IdIndex eid = g.addEdge(std::move(edge));
  g.getPort(srcPort)->connectedEdges.push_back(eid);
  g.getPort(dstPort)->connectedEdges.push_back(eid);
  return eid;
}

void addADGEdge(Graph &adg, IdIndex srcPort, IdIndex dstPort) {
  auto e = std::make_unique<Edge>();
  e->srcPort = srcPort;
  e->dstPort = dstPort;
  IdIndex eid = adg.addEdge(std::move(e));
  adg.getPort(srcPort)->connectedEdges.push_back(eid);
  adg.getPort(dstPort)->connectedEdges.push_back(eid);
}

std::vector<uint32_t> readConfigWords(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  std::vector<uint32_t> words;
  uint32_t w;
  while (in.read(reinterpret_cast<char *>(&w), 4))
    words.push_back(w);
  return words;
}

void cleanupConfigFiles(const std::string &basePath) {
  llvm::sys::fs::remove(basePath + ".config.bin");
  llvm::sys::fs::remove(basePath + "_addr.h");
  llvm::sys::fs::remove(basePath + ".mapping.json");
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Create a temp directory for config output files.
  llvm::SmallString<128> tmpDir;
  std::error_code ec = llvm::sys::fs::createUniqueDirectory(
      "loom-config-golden-test", tmpDir);
  TEST_ASSERT(!ec);

  // Test 1: Switch routing and PE config generation.
  // DFG: 2-node chain (arith.cmpi -> arith.addi).
  // ADG: 2 functional PEs + 1 switch (2-in, 2-out, full crossbar).
  // Verify:
  //   - Mapper routes the edge through the switch (swEdgeToHwPaths non-empty).
  //   - ConfigGen produces a non-empty binary with correct structure.
  //   - arith.cmpi PE config has a non-zero predicate field.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // Use arith.cmpi so genPEConfig emits a 4-bit predicate (non-zero config).
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.cmpi", "functional", 2, 1);
    // Set a predicate attribute on the DFG cmp node.
    setIntAttr(dfg.getNode(sw0), ctx, "predicate", 2); // "slt"

    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: 2 PEs + 1 switch (2 in, 2 out).
    IdIndex hw0 = addPENode(adg, ctx, "arith.cmpi", "functional", 2, 1);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex csw = addPENode(adg, ctx, "fabric.switch", "routing", 2, 2);

    // PE0.out -> switch.in0
    addADGEdge(adg, adg.getNode(hw0)->outputPorts[0],
               adg.getNode(csw)->inputPorts[0]);
    // PE1.out -> switch.in1
    addADGEdge(adg, adg.getNode(hw1)->outputPorts[0],
               adg.getNode(csw)->inputPorts[1]);
    // switch.out0 -> PE0.in0
    addADGEdge(adg, adg.getNode(csw)->outputPorts[0],
               adg.getNode(hw0)->inputPorts[0]);
    // switch.out1 -> PE1.in
    addADGEdge(adg, adg.getNode(csw)->outputPorts[1],
               adg.getNode(hw1)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 10.0;
    opts.seed = 42;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);
    TEST_ASSERT(result.success);

    // Verify routing paths exist.
    bool hasRoutingPath = false;
    for (const auto &pathVec : result.state.swEdgeToHwPaths) {
      if (!pathVec.empty()) {
        hasRoutingPath = true;
        break;
      }
    }
    TEST_ASSERT(hasRoutingPath);

    // Invoke ConfigGen on the successful mapping result.
    llvm::SmallString<256> basePath(tmpDir);
    llvm::sys::path::append(basePath, "test1_switch");
    std::string basePathStr = std::string(basePath);

    ConfigGen configGen;
    bool genOk = configGen.generate(result.state, dfg, adg, basePathStr,
                                    false, opts.profile, opts.seed);
    TEST_ASSERT(genOk);

    // Read the .config.bin and verify it is non-empty with at least one
    // non-zero word (the arith.cmpi PE should have predicate bits set).
    std::vector<uint32_t> words = readConfigWords(basePathStr + ".config.bin");
    TEST_ASSERT(!words.empty());

    bool hasNonZero = false;
    for (uint32_t w : words) {
      if (w != 0) {
        hasNonZero = true;
        break;
      }
    }
    TEST_ASSERT(hasNonZero);

    cleanupConfigFiles(basePathStr);
  }

  // Test 2: PE config for arith.addi.
  // DFG: 1-node arith.addi (2 inputs, 1 output).
  // ADG: 1 functional PE (arith.addi, 2 in, 1 out) -- direct binding.
  // Verify that ConfigGen produces at least 1 config word for the PE.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 2, 1);

    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 2, 1);

    // Manual mapping (no routing needed for a single node).
    MappingState state;
    state.init(dfg, adg);
    state.mapNode(sw0, hw0, dfg, adg);
    state.mapPort(dfg.getNode(sw0)->inputPorts[0],
                  adg.getNode(hw0)->inputPorts[0], dfg, adg);
    state.mapPort(dfg.getNode(sw0)->inputPorts[1],
                  adg.getNode(hw0)->inputPorts[1], dfg, adg);
    state.mapPort(dfg.getNode(sw0)->outputPorts[0],
                  adg.getNode(hw0)->outputPorts[0], dfg, adg);

    TEST_ASSERT(state.swNodeToHwNode[sw0] == hw0);

    llvm::SmallString<256> basePath(tmpDir);
    llvm::sys::path::append(basePath, "test2_pe");
    std::string basePathStr = std::string(basePath);

    ConfigGen configGen;
    bool genOk = configGen.generate(state, dfg, adg, basePathStr,
                                    false, "balanced", 0);
    TEST_ASSERT(genOk);

    // PE config should produce at least 1 word (minimum PE config).
    std::vector<uint32_t> words = readConfigWords(basePathStr + ".config.bin");
    TEST_ASSERT(words.size() >= 1);

    cleanupConfigFiles(basePathStr);
  }

  // Test 3: Memory addr_offset_table.
  // DFG: 2 memory ops (handshake.load).
  // ADG: 1 memory node with numRegion=2, tag_width=4.
  // Verify region entries in the binary config:
  //   entry_width = 1(valid) + 4(start_tag) + 5(end_tag) + 16(addr_offset) = 26 bits
  //   Region 0 valid bit at bit 0 should be 1
  //   Region 1 valid bit at bit 26 should be 1
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // DFG: 2 memory operations.
    auto swNode0 = std::make_unique<Node>();
    swNode0->kind = Node::OperationNode;
    setStringAttr(swNode0.get(), ctx, "op_name", "handshake.load");
    setStringAttr(swNode0.get(), ctx, "resource_class", "functional");
    IdIndex swId0 = dfg.addNode(std::move(swNode0));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = swId0;
      port->direction = Port::Input;
      IdIndex pid = dfg.addPort(std::move(port));
      dfg.getNode(swId0)->inputPorts.push_back(pid);
    }

    auto swNode1 = std::make_unique<Node>();
    swNode1->kind = Node::OperationNode;
    setStringAttr(swNode1.get(), ctx, "op_name", "handshake.load");
    setStringAttr(swNode1.get(), ctx, "resource_class", "functional");
    IdIndex swId1 = dfg.addNode(std::move(swNode1));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = swId1;
      port->direction = Port::Input;
      IdIndex pid = dfg.addPort(std::move(port));
      dfg.getNode(swId1)->inputPorts.push_back(pid);
    }

    // ADG: 1 memory node with numRegion=2, tag_width=4.
    auto hwNode = std::make_unique<Node>();
    hwNode->kind = Node::OperationNode;
    setStringAttr(hwNode.get(), ctx, "op_name", "fabric.memory");
    setStringAttr(hwNode.get(), ctx, "resource_class", "memory");
    setStringAttr(hwNode.get(), ctx, "sym_name", "mem0");
    setIntAttr(hwNode.get(), ctx, "numRegion", 2);
    setIntAttr(hwNode.get(), ctx, "tag_width", 4);
    IdIndex hwId = adg.addNode(std::move(hwNode));
    {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Input;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->inputPorts.push_back(pid);
    }
    {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Output;
      IdIndex pid = adg.addPort(std::move(port));
      adg.getNode(hwId)->outputPorts.push_back(pid);
    }

    // Map both SW nodes to the memory HW node.
    MappingState state;
    state.init(dfg, adg);
    state.mapNode(swId0, hwId, dfg, adg);
    state.mapNode(swId1, hwId, dfg, adg);

    TEST_ASSERT(state.hwNodeToSwNodes[hwId].size() == 2);

    llvm::SmallString<256> basePath(tmpDir);
    llvm::sys::path::append(basePath, "test3_memory");
    std::string basePathStr = std::string(basePath);

    ConfigGen configGen;
    bool genOk = configGen.generate(state, dfg, adg, basePathStr,
                                    false, "balanced", 0);
    TEST_ASSERT(genOk);

    // Read the binary config.
    std::string binPath = basePathStr + ".config.bin";
    std::ifstream binFile(binPath, std::ios::binary);
    TEST_ASSERT(binFile.good());

    std::vector<uint8_t> binData(
        (std::istreambuf_iterator<char>(binFile)),
        std::istreambuf_iterator<char>());
    binFile.close();

    // With 2 regions of 26 bits each (52 bits total), need at least 2 words
    // (8 bytes).
    TEST_ASSERT(binData.size() >= 8);

    // Reconstruct words (little-endian).
    uint32_t word0 = binData[0] | (binData[1] << 8) |
                      (binData[2] << 16) | (binData[3] << 24);

    // Region 0: valid bit at bit 0 should be 1.
    TEST_ASSERT((word0 & 0x1) == 1);

    // Region 1: valid bit at bit 26 should be 1.
    // entry_width = 1 + 4 + 5 + 16 = 26
    TEST_ASSERT((word0 & (1U << 26)) != 0);

    cleanupConfigFiles(basePathStr);
  }

  // Clean up temp directory.
  llvm::sys::fs::remove(tmpDir);

  return 0;
}

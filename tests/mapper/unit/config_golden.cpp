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

  // Test 1: Switch routing and PE predicate config.
  // DFG: 2-node chain (arith.cmpi -> arith.addi).
  // ADG: 2 functional PEs + 1 switch (2-in, 2-out, full crossbar).
  // Verify:
  //   - Mapper routes the edge through the switch.
  //   - PE config for arith.cmpi encodes predicate=2 as the value 0x2 in the
  //     bottom 4 bits (LSB-first packing of the 4-bit predicate field).
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    // Use arith.cmpi so genPEConfig emits a 4-bit predicate.
    IdIndex sw0 = addOpNode(dfg, ctx, "arith.cmpi", "functional", 2, 1);
    setIntAttr(dfg.getNode(sw0), ctx, "predicate", 2); // "slt"

    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    // ADG: 2 PEs + 1 switch (2 in, 2 out).
    IdIndex hw0 = addPENode(adg, ctx, "arith.cmpi", "functional", 2, 1);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex csw = addPENode(adg, ctx, "fabric.switch", "routing", 2, 2);

    addADGEdge(adg, adg.getNode(hw0)->outputPorts[0],
               adg.getNode(csw)->inputPorts[0]);
    addADGEdge(adg, adg.getNode(hw1)->outputPorts[0],
               adg.getNode(csw)->inputPorts[1]);
    addADGEdge(adg, adg.getNode(csw)->outputPorts[0],
               adg.getNode(hw0)->inputPorts[0]);
    addADGEdge(adg, adg.getNode(csw)->outputPorts[1],
               adg.getNode(hw1)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 10.0;
    opts.seed = 42;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);
    TEST_ASSERT(result.success);

    // Verify routing paths exist (edge must route through the switch).
    bool hasRoutingPath = false;
    for (const auto &pathVec : result.state.swEdgeToHwPaths) {
      if (!pathVec.empty()) {
        hasRoutingPath = true;
        break;
      }
    }
    TEST_ASSERT(hasRoutingPath);

    // Invoke ConfigGen.
    llvm::SmallString<256> basePath(tmpDir);
    llvm::sys::path::append(basePath, "test1_switch");
    std::string basePathStr = std::string(basePath);

    ConfigGen configGen;
    bool genOk = configGen.generate(result.state, dfg, adg, basePathStr,
                                    false, opts.profile, opts.seed);
    TEST_ASSERT(genOk);

    // The config binary must contain at least one word with the predicate
    // value (0x2) in bits [0:3]. genPEConfig encodes the predicate field
    // for arith.cmpi as 4 LSB bits.
    std::vector<uint32_t> words = readConfigWords(basePathStr + ".config.bin");
    TEST_ASSERT(!words.empty());
    bool foundPredicate = false;
    for (uint32_t w : words) {
      if ((w & 0xF) == 2) {
        foundPredicate = true;
        break;
      }
    }
    TEST_ASSERT(foundPredicate);

    cleanupConfigFiles(basePathStr);
  }

  // Test 2: PE config with deterministic manual mapping.
  // DFG: 1-node arith.cmpi with predicate=5.
  // ADG: 1 functional PE (arith.cmpi, 2 in, 1 out).
  // Manual binding -> ConfigGen -> verify exact config word == 0x5.
  // genPEConfig packs: no tags (no output_tag attr), predicate(4 bits)=5.
  // LSB-first: 5 = 0b0101 -> bits[0:3] = 0x5.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.cmpi", "functional", 2, 1);
    setIntAttr(dfg.getNode(sw0), ctx, "predicate", 5);

    IdIndex hw0 = addPENode(adg, ctx, "arith.cmpi", "functional", 2, 1);

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

    std::vector<uint32_t> words = readConfigWords(basePathStr + ".config.bin");
    TEST_ASSERT(words.size() == 1);
    // Exact golden value: predicate=5 packed in 4 LSB bits.
    TEST_ASSERT(words[0] == 0x5);

    cleanupConfigFiles(basePathStr);
  }

  // Test 3: Memory addr_offset_table with exact field encoding.
  // DFG: 2 memory ops (handshake.load).
  // ADG: 1 memory node with numRegion=2, tag_width=4.
  // Region entry layout (LSB-first, packed contiguously):
  //   valid(1) + start_tag(4) + end_tag(5) + addr_offset(16) = 26 bits
  //
  // Region 0 (r=0): valid=1, start_tag=0, end_tag=1, addr_offset=0
  //   Bits 0-25:  bit0=1(valid), bits1-4=0(start_tag),
  //               bits5-9: end_tag=1 -> bit5=1 rest=0,
  //               bits10-25=0(addr_offset)
  //
  // Region 1 (r=1): valid=1, start_tag=1, end_tag=2, addr_offset=0
  //   Bits 26-51: bit26=1(valid), bits27-30: start_tag=1 -> bit27=1 rest=0,
  //               bits31-35: end_tag=2 -> 0b10 LSB-first -> bit31=0, bit32=1
  //               bits36-51=0(addr_offset)
  //
  // Expected word0: (1<<0) | (1<<5) | (1<<26) | (1<<27) = 0x0C000021
  // Expected word1: (1<<0) = 0x00000001  (end_tag bit 1 of value 2)
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex swId0 = addOpNode(dfg, ctx, "handshake.load", "functional", 1, 0);
    IdIndex swId1 = addOpNode(dfg, ctx, "handshake.load", "functional", 1, 0);

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

    std::vector<uint32_t> words = readConfigWords(basePathStr + ".config.bin");
    // 2 regions * 26 bits = 52 bits -> 2 words (64 bits).
    TEST_ASSERT(words.size() == 2);

    // Exact golden values computed from the LSB-first packing of:
    //   Region 0: valid=1, start_tag=0, end_tag=1, addr_offset=0
    //   Region 1: valid=1, start_tag=1, end_tag=2, addr_offset=0
    TEST_ASSERT(words[0] == 0x0C000021U);
    TEST_ASSERT(words[1] == 0x00000001U);

    cleanupConfigFiles(basePathStr);
  }

  // Clean up temp directory.
  llvm::sys::fs::remove(tmpDir);

  return 0;
}

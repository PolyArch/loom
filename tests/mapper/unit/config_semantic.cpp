//===-- config_semantic.cpp - Config generation semantic test ------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Test config generation semantic properties: verify that ConfigGen produces
// non-empty output after a successful mapping, output size is reasonable,
// and failed mappings do not produce valid config output.
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

#include <cstdio>

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

/// Helper to get file size. Returns -1 on error.
int64_t getFileSize(const std::string &path) {
  uint64_t size = 0;
  if (llvm::sys::fs::file_size(path, size))
    return -1;
  return static_cast<int64_t>(size);
}

/// Helper to remove generated config files.
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
      "loom-config-semantic-test", tmpDir);
  TEST_ASSERT(!ec);

  // Test 1: After a successful mapping (2-node chain), invoke ConfigGen
  // and verify it produces non-empty config output.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    addADGEdge(adg, adg.getNode(hw0)->outputPorts[0],
               adg.getNode(hw1)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 42;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);
    TEST_ASSERT(result.success);

    // Invoke ConfigGen on the successful mapping result.
    llvm::SmallString<256> basePath(tmpDir);
    llvm::sys::path::append(basePath, "test1_kernel");
    std::string basePathStr = std::string(basePath);

    ConfigGen configGen;
    bool genOk = configGen.generate(result.state, dfg, adg, basePathStr,
                                    true, opts.profile, opts.seed);
    TEST_ASSERT(genOk);

    // Verify .config.bin exists and is non-empty.
    int64_t binSize = getFileSize(basePathStr + ".config.bin");
    TEST_ASSERT(binSize > 0);

    // Verify _addr.h exists and is non-empty.
    int64_t addrSize = getFileSize(basePathStr + "_addr.h");
    TEST_ASSERT(addrSize > 0);

    // Verify .mapping.json exists and is non-empty.
    int64_t jsonSize = getFileSize(basePathStr + ".mapping.json");
    TEST_ASSERT(jsonSize > 0);

    cleanupConfigFiles(basePathStr);
  }

  // Test 2: Config output size is reasonable (non-zero bytes, not
  // unreasonably large). Use the same 2-node chain pattern.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    IdIndex sw0 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex sw1 = addOpNode(dfg, ctx, "arith.addi", "functional", 1, 1);
    addEdgeBetween(dfg, sw0, 0, sw1, 0);

    IdIndex hw0 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    IdIndex hw1 = addPENode(adg, ctx, "arith.addi", "functional", 1, 1);
    addADGEdge(adg, adg.getNode(hw0)->outputPorts[0],
               adg.getNode(hw1)->inputPorts[0]);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 42;
    opts.profile = "balanced";

    Mapper::Result result = mapper.run(dfg, adg, opts);
    TEST_ASSERT(result.success);

    llvm::SmallString<256> basePath(tmpDir);
    llvm::sys::path::append(basePath, "test2_kernel");
    std::string basePathStr = std::string(basePath);

    ConfigGen configGen;
    bool genOk = configGen.generate(result.state, dfg, adg, basePathStr,
                                    true, opts.profile, opts.seed);
    TEST_ASSERT(genOk);

    // Binary config should be non-zero but reasonable for 2 nodes.
    // For 2 PE nodes each with 1 config word (32 bits = 4 bytes), the
    // binary should be at most a few hundred bytes.
    int64_t binSize = getFileSize(basePathStr + ".config.bin");
    TEST_ASSERT(binSize > 0);
    TEST_ASSERT(binSize < 4096); // Reasonable upper bound for 2-node config.

    // Address header should be reasonable text size.
    int64_t addrSize = getFileSize(basePathStr + "_addr.h");
    TEST_ASSERT(addrSize > 0);
    TEST_ASSERT(addrSize < 8192);

    // JSON should be reasonable for 2-node mapping report.
    int64_t jsonSize = getFileSize(basePathStr + ".mapping.json");
    TEST_ASSERT(jsonSize > 0);
    TEST_ASSERT(jsonSize < 65536); // 64KB upper bound.

    cleanupConfigFiles(basePathStr);
  }

  // Test 3: After a failed mapping, ConfigGen should produce empty/error
  // output. Use incompatible DFG/ADG types.
  {
    Graph dfg(&ctx);
    Graph adg(&ctx);

    addOpNode(dfg, ctx, "arith.muli", "functional", 1, 1);
    addPENode(adg, ctx, "arith.addi", "functional", 1, 1);

    Mapper mapper;
    Mapper::Options opts;
    opts.budgetSeconds = 5.0;
    opts.seed = 0;

    Mapper::Result result = mapper.run(dfg, adg, opts);
    TEST_ASSERT(!result.success);

    // On a failed mapping, the MappingState has no valid placement.
    // ConfigGen::generate should either produce empty files or return false.
    llvm::SmallString<256> basePath(tmpDir);
    llvm::sys::path::append(basePath, "test3_failed");
    std::string basePathStr = std::string(basePath);

    ConfigGen configGen;
    bool genOk = configGen.generate(result.state, dfg, adg, basePathStr,
                                    false, opts.profile, opts.seed);

    if (genOk) {
      // If generate returned true, the binary should be empty or zero-size
      // because no nodes were successfully placed.
      int64_t binSize = getFileSize(basePathStr + ".config.bin");
      TEST_ASSERT(binSize >= 0);
      // For a failed mapping with no placed nodes, config should be minimal.
      TEST_ASSERT(binSize <= 16);
    }
    // If genOk is false, that is also acceptable for a failed mapping.

    cleanupConfigFiles(basePathStr);
  }

  // Clean up temp directory.
  llvm::sys::fs::remove(tmpDir);

  return 0;
}

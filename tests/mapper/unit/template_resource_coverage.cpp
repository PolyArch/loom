//===-- template_resource_coverage.cpp - Template PE coverage test -*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Verify that CGRA template fabric files declare PE instances for all
// required operation classes. This catches template regressions where
// PE definitions exist but are never instantiated in the module body.
//
// The test constructs an ADG graph (simulating a parsed template) and
// checks that required operation classes have at least one instance.
//
//===----------------------------------------------------------------------===//

#include "TestUtil.h"
#include "loom/Mapper/Graph.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"

using namespace loom;

namespace {

void setStringAttr(Node *node, mlir::MLIRContext &ctx,
                   llvm::StringRef name, llvm::StringRef value) {
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name), mlir::StringAttr::get(&ctx, value)));
}

void setArrayStrAttr(Node *node, mlir::MLIRContext &ctx,
                     llvm::StringRef name,
                     llvm::ArrayRef<llvm::StringRef> values) {
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  for (auto v : values)
    attrs.push_back(mlir::StringAttr::get(&ctx, v));
  node->attributes.push_back(mlir::NamedAttribute(
      mlir::StringAttr::get(&ctx, name),
      mlir::ArrayAttr::get(&ctx, attrs)));
}

/// Get a string attribute from a node.
llvm::StringRef getStrAttr(const Node *node, llvm::StringRef name) {
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr.getValue()))
        return strAttr.getValue();
    }
  }
  return "";
}

/// Get an ArrayAttr of strings from a node.
llvm::SmallVector<std::string, 4> getArrayStrAttr(const Node *node,
                                                    llvm::StringRef name) {
  llvm::SmallVector<std::string, 4> result;
  for (auto &attr : node->attributes) {
    if (attr.getName() == name) {
      if (auto arrAttr = mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue())) {
        for (auto elem : arrAttr) {
          if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(elem))
            result.push_back(strAttr.getValue().str());
        }
      }
    }
  }
  return result;
}

IdIndex addPEInstance(Graph &adg, mlir::MLIRContext &ctx,
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

/// Classify an op_name into a high-level operation class.
/// Returns one of: "constant", "arithmetic", "control", "dataflow", "memory",
/// or empty string for unknown/routing.
llvm::StringRef classifyOp(llvm::StringRef opName) {
  if (opName.contains("constant"))
    return "constant";
  if (opName.contains("addi") || opName.contains("subi") ||
      opName.contains("muli") || opName.contains("addf") ||
      opName.contains("subf") || opName.contains("mulf") ||
      opName.contains("divf") || opName.contains("negf") ||
      opName.contains("cmpi") || opName.contains("cmpf") ||
      opName.contains("select") || opName.contains("andi") ||
      opName.contains("ori") || opName.contains("xori") ||
      opName.contains("shli") || opName.contains("shrui") ||
      opName.contains("shrsi"))
    return "arithmetic";
  if (opName.contains("cond_br") || opName.contains("mux") ||
      opName.contains("join"))
    return "control";
  if (opName.contains("invariant") || opName.contains("carry") ||
      opName.contains("gate") || opName.contains("stream"))
    return "dataflow";
  if (opName.contains("load") || opName.contains("store") ||
      opName.contains("memory"))
    return "memory";
  return "";
}

/// Count instances per operation class in an ADG.
/// Checks both op_name and body_ops (for multi-op PEs).
llvm::StringMap<unsigned> countClassInstances(const Graph &adg) {
  llvm::StringMap<unsigned> counts;

  for (IdIndex i = 0; i < static_cast<IdIndex>(adg.nodes.size()); ++i) {
    const Node *node = adg.getNode(i);
    if (!node || node->kind != Node::OperationNode)
      continue;

    llvm::StringRef resClass = getStrAttr(node, "resource_class");
    if (resClass != "functional" && resClass != "memory")
      continue;

    // Check direct op_name.
    llvm::StringRef opName = getStrAttr(node, "op_name");
    llvm::StringRef opClass = classifyOp(opName);
    if (!opClass.empty())
      counts[opClass]++;

    // Check body_ops for multi-op PEs.
    auto bodyOps = getArrayStrAttr(node, "body_ops");
    for (const auto &bodyOp : bodyOps) {
      llvm::StringRef bodyClass = classifyOp(bodyOp);
      if (!bodyClass.empty())
        counts[bodyClass]++;
    }
  }

  return counts;
}

} // namespace

int main() {
  mlir::MLIRContext ctx;
  ctx.loadAllAvailableDialects();

  // Test 1: A well-formed template has instances for all required classes.
  {
    Graph adg(&ctx);

    // Simulate a small CGRA template with representative PEs.
    // Constant PEs.
    addPEInstance(adg, ctx, "handshake.constant", "functional", 1, 1);
    addPEInstance(adg, ctx, "handshake.constant", "functional", 1, 1);

    // Arithmetic PEs.
    addPEInstance(adg, ctx, "arith.addi", "functional", 2, 1);
    addPEInstance(adg, ctx, "arith.muli", "functional", 2, 1);
    addPEInstance(adg, ctx, "arith.cmpi", "functional", 2, 1);

    // Control flow PEs.
    addPEInstance(adg, ctx, "handshake.cond_br", "functional", 2, 2);
    addPEInstance(adg, ctx, "handshake.mux", "functional", 3, 1);

    // Dataflow PEs.
    addPEInstance(adg, ctx, "dataflow.invariant", "functional", 1, 1);
    addPEInstance(adg, ctx, "dataflow.carry", "functional", 2, 1);

    // Memory PEs.
    addPEInstance(adg, ctx, "handshake.load", "memory", 2, 1);

    // Routing PEs (should not count toward operation classes).
    addPEInstance(adg, ctx, "fabric.switch", "routing", 4, 4);

    auto counts = countClassInstances(adg);

    // All required classes must have at least 1 instance.
    TEST_ASSERT(counts.count("constant") && counts["constant"] >= 1);
    TEST_ASSERT(counts.count("arithmetic") && counts["arithmetic"] >= 1);
    TEST_ASSERT(counts.count("control") && counts["control"] >= 1);
    TEST_ASSERT(counts.count("dataflow") && counts["dataflow"] >= 1);
    TEST_ASSERT(counts.count("memory") && counts["memory"] >= 1);
  }

  // Test 2: Template missing memory instances should fail coverage check.
  {
    Graph adg(&ctx);

    // Only arithmetic and constant PEs, no memory.
    addPEInstance(adg, ctx, "handshake.constant", "functional", 1, 1);
    addPEInstance(adg, ctx, "arith.addi", "functional", 2, 1);
    addPEInstance(adg, ctx, "handshake.cond_br", "functional", 2, 2);
    addPEInstance(adg, ctx, "dataflow.invariant", "functional", 1, 1);

    auto counts = countClassInstances(adg);

    // Memory class should be missing.
    TEST_ASSERT(!counts.count("memory") || counts["memory"] == 0);
    // Other classes should be present.
    TEST_ASSERT(counts.count("constant") && counts["constant"] >= 1);
    TEST_ASSERT(counts.count("arithmetic") && counts["arithmetic"] >= 1);
  }

  // Test 3: Multi-op PE body_ops contribute to class counts.
  {
    Graph adg(&ctx);

    // A multi-op PE that contains arithmetic operations in body_ops.
    auto multiPE = std::make_unique<Node>();
    multiPE->kind = Node::OperationNode;
    setStringAttr(multiPE.get(), ctx, "op_name", "fabric.pe");
    setStringAttr(multiPE.get(), ctx, "resource_class", "functional");
    setArrayStrAttr(multiPE.get(), ctx, "body_ops",
                    {"arith.addi", "arith.muli"});
    IdIndex hwId = adg.addNode(std::move(multiPE));
    for (int i = 0; i < 2; ++i) {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Input;
      adg.addPort(std::move(port));
    }
    {
      auto port = std::make_unique<Port>();
      port->parentNode = hwId;
      port->direction = Port::Output;
      adg.addPort(std::move(port));
    }

    auto counts = countClassInstances(adg);

    // body_ops should contribute to arithmetic class.
    TEST_ASSERT(counts.count("arithmetic") && counts["arithmetic"] >= 2);
  }

  // Test 4: Empty template has zero instances for all classes.
  {
    Graph adg(&ctx);

    auto counts = countClassInstances(adg);

    TEST_ASSERT(!counts.count("constant") || counts["constant"] == 0);
    TEST_ASSERT(!counts.count("arithmetic") || counts["arithmetic"] == 0);
    TEST_ASSERT(!counts.count("control") || counts["control"] == 0);
    TEST_ASSERT(!counts.count("dataflow") || counts["dataflow"] == 0);
    TEST_ASSERT(!counts.count("memory") || counts["memory"] == 0);
  }

  return 0;
}

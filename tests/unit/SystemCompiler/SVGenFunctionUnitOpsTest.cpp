// Function unit support regression for rsqrt and minimumf across simulator
// and SV generation helpers.

#include "SVGenInternal.h"

#include "loom/SVGen/SVModuleRegistry.h"
#include "loom/Simulator/SimFunctionUnit.h"
#include "loom/Simulator/SimModule.h"
#include "loom/Simulator/StaticModel.h"
#include "loom/Simulator/StaticModelTypes.h"

#include "loom/Dialect/Fabric/FabricDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using loom::sim::SimChannel;
using loom::sim::StaticMappedModel;
using loom::sim::StaticModuleDesc;
using loom::sim::StaticModuleKind;
using loom::sim::StaticPortDesc;
using loom::sim::StaticPortDirection;

struct BuiltFunctionUnit {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  loom::fabric::FunctionUnitOp fu;
};

static uint64_t toBits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static uint64_t toBits(double value) {
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static StaticPortDesc makePort(uint32_t portId, uint32_t parentNodeId,
                               StaticPortDirection direction,
                               unsigned valueWidth) {
  StaticPortDesc port;
  port.portId = portId;
  port.parentNodeId = parentNodeId;
  port.direction = direction;
  port.valueWidth = valueWidth;
  return port;
}

static std::string typeToText(mlir::Type type) {
  if (type.isF64())
    return "f64";
  if (type.isF32())
    return "f32";
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  os.flush();
  return text;
}

static BuiltFunctionUnit buildInlineFunctionUnit(mlir::MLIRContext &ctx,
                                                 llvm::StringRef symName,
                                                 llvm::StringRef opName,
                                                 mlir::Type inputType,
                                                 mlir::Type outputType,
                                                 unsigned numInputs) {
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();
  ctx.allowUnregisteredDialects(true);

  llvm::StringRef placeholderOpName =
      numInputs == 1 ? llvm::StringRef("math.sqrt")
                     : llvm::StringRef("arith.addf");
  std::string moduleText = "module {\n  fabric.function_unit @";
  moduleText += symName.str();
  moduleText += "(";
  for (unsigned idx = 0; idx < numInputs; ++idx) {
    if (idx > 0)
      moduleText += ", ";
    moduleText += "%arg" + std::to_string(idx) + ": " + typeToText(inputType);
  }
  moduleText += ") -> (" + typeToText(outputType) +
                ") [latency = 0, interval = 1] {\n";
  moduleText += "    %0 = \"" + placeholderOpName.str() + "\"(";
  for (unsigned idx = 0; idx < numInputs; ++idx) {
    if (idx > 0)
      moduleText += ", ";
    moduleText += "%arg" + std::to_string(idx);
  }
  moduleText += ") : (";
  for (unsigned idx = 0; idx < numInputs; ++idx) {
    if (idx > 0)
      moduleText += ", ";
    moduleText += typeToText(inputType);
  }
  moduleText += ") -> " + typeToText(outputType) + "\n";
  moduleText += "    fabric.yield %0 : " + typeToText(outputType) + "\n";
  moduleText += "  }\n}\n";

  auto module = mlir::parseSourceString<mlir::ModuleOp>(moduleText, &ctx);
  if (!module) {
    std::cerr << "FAIL: unable to parse inline function unit module\n";
    return {};
  }

  loom::fabric::FunctionUnitOp fuOp;
  module->walk([&](loom::fabric::FunctionUnitOp op) { fuOp = op; });
  if (!fuOp) {
    std::cerr << "FAIL: parsed module does not contain fabric.function_unit\n";
    return {};
  }

  auto &body = fuOp.getBody().front();
  auto *yieldOp = body.getTerminator();
  auto *placeholder = &body.front();
  if (!placeholder) {
    std::cerr << "FAIL: parsed inline FU is missing placeholder body op\n";
    return {};
  }

  mlir::OpBuilder builder(&ctx);
  builder.setInsertionPoint(placeholder);
  mlir::OperationState bodyOpState(builder.getUnknownLoc(), opName.str());
  for (unsigned idx = 0; idx < numInputs; ++idx)
    bodyOpState.addOperands(body.getArgument(idx));
  bodyOpState.addTypes({outputType});
  auto *bodyOp = builder.insert(mlir::Operation::create(bodyOpState));
  yieldOp->erase();
  placeholder->erase();
  builder.setInsertionPointToEnd(&body);
  mlir::OperationState yieldState(builder.getUnknownLoc(), "fabric.yield");
  yieldState.addOperands(bodyOp->getResult(0));
  builder.insert(mlir::Operation::create(yieldState));

  BuiltFunctionUnit built;
  built.module = std::move(module);
  built.fu = fuOp;
  return built;
}

static bool expect(bool cond, const std::string &message) {
  if (cond)
    return true;
  std::cerr << "FAIL: " << message << "\n";
  return false;
}

static bool containsPath(const std::vector<std::string> &paths,
                         llvm::StringRef needle) {
  for (const auto &path : paths) {
    if (path == needle)
      return true;
  }
  return false;
}

static bool testRegistryCoverage() {
  loom::svgen::SVModuleRegistry registry;
  bool ok = true;

  ok &= expect(loom::svgen::SVModuleRegistry::isKnownOp("math.rsqrt"),
               "math.rsqrt should be known");
  ok &= expect(loom::svgen::SVModuleRegistry::isKnownOp("arith.minimumf"),
               "arith.minimumf should be known");
  ok &= expect(registry.requireArithOp("math.rsqrt", ""),
               "math.rsqrt should register");
  ok &= expect(registry.requireArithOp("arith.minimumf", ""),
               "arith.minimumf should register");
  ok &= expect(loom::svgen::SVModuleRegistry::getSVModuleName("math.rsqrt") ==
                   "fu_op_rsqrt",
               "math.rsqrt should map to fu_op_rsqrt");
  ok &= expect(
      loom::svgen::SVModuleRegistry::getSVFilePath("arith.minimumf") ==
          "arith/fu_op_minimumf.sv",
      "arith.minimumf should map to fu_op_minimumf.sv");

  const auto required = registry.getRequiredFiles();
  ok &= expect(containsPath(required, "math/fu_op_rsqrt.sv"),
               "math.rsqrt should add its RTL file");
  ok &= expect(containsPath(required, "arith/fu_op_minimumf.sv"),
               "arith.minimumf should add its RTL file");

  if (ok)
    std::cout << "PASS: testRegistryCoverage\n";
  return ok;
}

static bool testRsqrtSVAndSim() {
  mlir::DialectRegistry registry;
  registry.insert<loom::fabric::FabricDialect>();

  mlir::MLIRContext ctx(registry);
  ctx.allowUnregisteredDialects(true);

  auto built = buildInlineFunctionUnit(ctx, "fu_rsqrt", "math.rsqrt",
                                       mlir::Float64Type::get(&ctx),
                                       mlir::Float64Type::get(&ctx), 1);
  auto fu = built.fu;
  bool ok = true;

  ok &= expect(loom::svgen::validateFUTimingConstraints(fu),
               "math.rsqrt FU should satisfy timing validation");

  loom::svgen::SVModuleRegistry svRegistry;
  std::string svText;
  llvm::raw_string_ostream os(svText);
  std::string modName = loom::svgen::generateFUBody(fu, os, svRegistry, "");
  os.flush();
  ok &= expect(!modName.empty(), "math.rsqrt SV generation should succeed");
  ok &= expect(svText.find("fu_op_rsqrt") != std::string::npos,
               "math.rsqrt SV should instantiate fu_op_rsqrt");

  StaticModuleDesc desc;
  desc.kind = StaticModuleKind::FunctionUnit;
  desc.name = "fu_rsqrt";
  desc.hwNodeId = 101;
  desc.inputPorts = {1};
  desc.outputPorts = {2};
  desc.intAttrs.push_back({"latency", 0});
  desc.intAttrs.push_back({"interval", 1});
  desc.stringArrayAttrs.push_back({"ops", {"math.rsqrt"}});

  StaticMappedModel model;
  model.mutablePorts().push_back(
      makePort(1, desc.hwNodeId, StaticPortDirection::Input, 64));
  model.mutablePorts().push_back(
      makePort(2, desc.hwNodeId, StaticPortDirection::Output, 64));

  auto simModule = loom::sim::createFunctionUnitModule(desc, model, false);
  ok &= expect(static_cast<bool>(simModule),
               "math.rsqrt simulator module should be supported");
  if (!simModule)
    return false;

  SimChannel input;
  SimChannel output;
  input.valid = true;
  input.data = toBits(16.0);
  input.generation = loom::sim::composeTokenGeneration(desc.hwNodeId, 1);
  output.ready = true;
  simModule->inputs = {&input};
  simModule->outputs = {&output};
  simModule->reset();
  simModule->evaluate();

  ok &= expect(output.valid, "math.rsqrt output should be valid");
  ok &= expect(output.data == toBits(0.25), "math.rsqrt output mismatch");

  if (ok)
    std::cout << "PASS: testRsqrtSVAndSim\n";
  return ok;
}

static bool testMinimumfSVAndSim() {
  mlir::DialectRegistry registry;
  registry.insert<loom::fabric::FabricDialect>();

  mlir::MLIRContext ctx(registry);
  ctx.allowUnregisteredDialects(true);

  auto built = buildInlineFunctionUnit(ctx, "fu_minimumf", "arith.minimumf",
                                       mlir::Float32Type::get(&ctx),
                                       mlir::Float32Type::get(&ctx), 2);
  auto fu = built.fu;
  bool ok = true;

  ok &= expect(loom::svgen::validateFUTimingConstraints(fu),
               "arith.minimumf FU should satisfy timing validation");

  loom::svgen::SVModuleRegistry svRegistry;
  std::string svText;
  llvm::raw_string_ostream os(svText);
  std::string modName = loom::svgen::generateFUBody(fu, os, svRegistry, "");
  os.flush();
  ok &= expect(!modName.empty(), "arith.minimumf SV generation should succeed");
  ok &= expect(svText.find("fu_op_minimumf") != std::string::npos,
               "arith.minimumf SV should instantiate fu_op_minimumf");

  StaticModuleDesc desc;
  desc.kind = StaticModuleKind::FunctionUnit;
  desc.name = "fu_minimumf";
  desc.hwNodeId = 102;
  desc.inputPorts = {1, 2};
  desc.outputPorts = {3};
  desc.intAttrs.push_back({"latency", 0});
  desc.intAttrs.push_back({"interval", 1});
  desc.stringArrayAttrs.push_back({"ops", {"arith.minimumf"}});

  StaticMappedModel model;
  model.mutablePorts().push_back(
      makePort(1, desc.hwNodeId, StaticPortDirection::Input, 32));
  model.mutablePorts().push_back(
      makePort(2, desc.hwNodeId, StaticPortDirection::Input, 32));
  model.mutablePorts().push_back(
      makePort(3, desc.hwNodeId, StaticPortDirection::Output, 32));

  auto simModule = loom::sim::createFunctionUnitModule(desc, model, false);
  ok &= expect(static_cast<bool>(simModule),
               "arith.minimumf simulator module should be supported");
  if (!simModule)
    return false;

  SimChannel lhs;
  SimChannel rhs;
  SimChannel output;
  lhs.valid = true;
  rhs.valid = true;
  lhs.data = toBits(3.0f);
  rhs.data = toBits(1.5f);
  lhs.generation = loom::sim::composeTokenGeneration(desc.hwNodeId, 1);
  rhs.generation = loom::sim::composeTokenGeneration(desc.hwNodeId, 2);
  output.ready = true;
  simModule->inputs = {&lhs, &rhs};
  simModule->outputs = {&output};
  simModule->reset();
  simModule->evaluate();

  ok &= expect(output.valid, "arith.minimumf output should be valid");
  ok &= expect(output.data == toBits(1.5f), "arith.minimumf output mismatch");

  if (ok)
    std::cout << "PASS: testMinimumfSVAndSim\n";
  return ok;
}

} // namespace

int main() {
  if (!testRegistryCoverage())
    return 1;
  if (!testRsqrtSVAndSim())
    return 1;
  if (!testMinimumfSVAndSim())
    return 1;
  return 0;
}

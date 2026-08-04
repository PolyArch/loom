#include "Hardware/RTL/CirctConformance.h"
#include "Hardware/RTL/CommonSkeleton.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace loom::hardware::rtl {
llvm::Expected<std::string> emitCirctConformanceSystemVerilog() {
  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  mlir::OpBuilder builder(&context);
  const mlir::Location location = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> top = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(top->getBody());

  const mlir::Type bit = builder.getI1Type();
  const mlir::Type byte = builder.getIntegerType(8);
  const mlir::Type address = builder.getIntegerType(16);
  const mlir::Type clock = circt::seq::ClockType::get(&context);
  llvm::SmallVector<circt::hw::PortInfo, 24> inputPorts;
  llvm::SmallVector<circt::hw::PortInfo, 16> outputPorts;
  auto addInput = [&](llvm::StringRef name, mlir::Type type) {
    inputPorts.push_back(
        circt::hw::PortInfo{{builder.getStringAttr(name), type,
                             circt::hw::ModulePort::Direction::Input}});
  };
  auto addOutput = [&](llvm::StringRef name, mlir::Type type) {
    outputPorts.push_back(
        circt::hw::PortInfo{{builder.getStringAttr(name), type,
                             circt::hw::ModulePort::Direction::Output}});
  };

  addInput("clock", clock);
  addInput("async_reset", bit);
  addInput("sync_reset", bit);
  addInput("config_subtract", bit);
  addInput("lhs_valid", bit);
  addInput("lhs_data", byte);
  addInput("rhs_valid", bit);
  addInput("rhs_data", byte);
  addInput("result_ready", bit);
  addOutput("lhs_ready", bit);
  addOutput("rhs_ready", bit);
  addOutput("result_valid", bit);
  addOutput("result_data", byte);

  addInput("memory_command_valid", bit);
  addInput("memory_command_address", address);
  addInput("memory_command_write", bit);
  addInput("memory_command_write_data", byte);
  addOutput("memory_command_ready", bit);
  addOutput("memory_request_valid", bit);
  addOutput("memory_request_address", address);
  addOutput("memory_request_write", bit);
  addOutput("memory_request_write_data", byte);
  addInput("memory_request_ready", bit);
  addInput("memory_response_valid", bit);
  addInput("memory_response_data", byte);
  addOutput("memory_response_ready", bit);
  addOutput("memory_result_valid", bit);
  addOutput("memory_result_data", byte);
  addInput("memory_result_ready", bit);

  const circt::hw::ModulePortInfo portInfo(inputPorts, outputPorts);
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("loom_circt_conformance"),
      portInfo,
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        auto input = [&](llvm::StringRef name) {
          return accessor.getInput(name);
        };
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        circt::Backedge nextData = backedges.get(byte);
        circt::Backedge nextValid = backedges.get(bit);
        mlir::Value zeroData = circt::hw::ConstantOp::create(
            bodyBuilder, location, llvm::APInt(8, 0));
        mlir::Value zeroBit = circt::hw::ConstantOp::create(
            bodyBuilder, location, llvm::APInt(1, 0));

        mlir::Value dataRegister = circt::seq::FirRegOp::create(
            bodyBuilder, location, nextData, input("clock"),
            bodyBuilder.getStringAttr("result_data_reg"), input("async_reset"),
            zeroData, circt::hw::InnerSymAttr{}, true);
        mlir::Value validRegister = circt::seq::CompRegOp::create(
            bodyBuilder, location, nextValid, input("clock"),
            input("sync_reset"), zeroBit, "result_valid_reg");

        mlir::Value canAccept = circt::comb::OrOp::create(
            bodyBuilder, location,
            circt::comb::createOrFoldNot(bodyBuilder, location, validRegister),
            input("result_ready"));
        mlir::Value lhsReady = circt::comb::AndOp::create(
            bodyBuilder, location, canAccept, input("rhs_valid"));
        mlir::Value rhsReady = circt::comb::AndOp::create(
            bodyBuilder, location, canAccept, input("lhs_valid"));
        mlir::Value pairValid = circt::comb::AndOp::create(
            bodyBuilder, location, input("lhs_valid"), input("rhs_valid"));
        mlir::Value accept = circt::comb::AndOp::create(bodyBuilder, location,
                                                        pairValid, canAccept);
        mlir::Value sum = circt::comb::AddOp::create(
            bodyBuilder, location, input("lhs_data"), input("rhs_data"));
        mlir::Value difference = circt::comb::SubOp::create(
            bodyBuilder, location, input("lhs_data"), input("rhs_data"));
        mlir::Value configuredResult = circt::comb::MuxOp::create(
            bodyBuilder, location, input("config_subtract"), difference, sum);
        mlir::Value retainValid = circt::comb::AndOp::create(
            bodyBuilder, location, validRegister,
            circt::comb::createOrFoldNot(bodyBuilder, location,
                                         input("result_ready")));
        nextValid.setValue(circt::comb::OrOp::create(bodyBuilder, location,
                                                     accept, retainValid));
        nextData.setValue(circt::comb::MuxOp::create(
            bodyBuilder, location, accept, configuredResult, dataRegister));

        accessor.setOutput("lhs_ready", lhsReady);
        accessor.setOutput("rhs_ready", rhsReady);
        accessor.setOutput("result_valid", validRegister);
        accessor.setOutput("result_data", dataRegister);

        accessor.setOutput("memory_command_ready",
                           input("memory_request_ready"));
        accessor.setOutput("memory_request_valid",
                           input("memory_command_valid"));
        accessor.setOutput("memory_request_address",
                           input("memory_command_address"));
        accessor.setOutput("memory_request_write",
                           input("memory_command_write"));
        accessor.setOutput("memory_request_write_data",
                           input("memory_command_write_data"));
        accessor.setOutput("memory_response_ready",
                           input("memory_result_ready"));
        accessor.setOutput("memory_result_valid",
                           input("memory_response_valid"));
        accessor.setOutput("memory_result_data", input("memory_response_data"));
      });

  return lowerAndExportSpecializedSystemVerilog(*top);
}

} // namespace loom::hardware::rtl

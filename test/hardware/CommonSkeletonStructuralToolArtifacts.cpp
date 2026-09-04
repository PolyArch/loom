#include "CommonSkeletonStructuralToolArtifacts.h"

#include <fstream>

namespace loom::hardware::test {

llvm::Error writeBoundaryStructuralToolArtifacts(
    const std::filesystem::path &root, llvm::StringRef systemVerilog) {
  std::filesystem::create_directories(root);
  std::ofstream(root / "loom_module.sv") << systemVerilog.str();
  std::ofstream testbench(root / "testbench.sv");
  testbench << R"sv(
module testbench;
  logic        clock;
  logic        reset;
  logic [31:0] input_0_data;
  logic        input_0_valid;
  logic [3:0]  input_1_data;
  logic [4:0]  input_1_tag;
  logic        input_1_valid;
  logic        output_0_ready;
  logic        output_1_ready;
  logic        input_0_ready;
  logic        input_1_ready;
  logic [15:0] output_0_data;
  logic        output_0_valid;
  logic [2:0]  output_1_tag;
  logic        output_1_valid;
  integer      control;

)sv";
  testbench << portableAxiLiteSignalDeclarations();
  testbench << R"sv(

  loom_module dut(.*);

  initial begin
    clock = 0;
    reset = 0;
)sv";
  testbench << portableAxiLiteInitialization();
  testbench << R"sv(
    for (control = 0; control < 16; control = control + 1) begin
      input_0_data = 32'hcafe0000 ^ control;
      input_0_valid = control[3];
      input_1_data = control[3:0];
      input_1_tag = 5'h18 ^ control[4:0];
      input_1_valid = control[2];
      output_0_ready = control[1];
      output_1_ready = control[0];
      #1;
      if (input_0_ready !== output_0_ready ||
          input_1_ready !== output_1_ready ||
          output_0_data !== input_0_data[15:0] ||
          output_0_valid !== input_0_valid ||
          output_1_tag !== input_1_tag[2:0] ||
          output_1_valid !== input_1_valid)
        $fatal(1, "Module boundary passthrough changed transport semantics");
    end
    $finish;
  end
endmodule
)sv";
  std::ofstream(root / "common_skeleton.ys") << R"ys(
read_verilog -sv loom_module.sv
hierarchy -check -top loom_module
check -assert
select -assert-none loom_module/t:$*latch* loom_module/t:$_*LATCH* loom_module/t:$mem*
synth -top loom_module
check -assert
select -assert-none loom_module/t:$*latch* loom_module/t:$_*LATCH* loom_module/t:$mem*
)ys";
  return llvm::Error::success();
}

llvm::Error writeSpatialHierarchyToolArtifacts(
    const std::filesystem::path &root, llvm::StringRef systemVerilog,
    llvm::ArrayRef<PortableConfigurationImage> inactiveConfigurations) {
  std::filesystem::create_directories(root);
  std::ofstream(root / "spatial_hierarchy_module.sv") << systemVerilog.str();
  std::ofstream testbench(root / "spatial_hierarchy_testbench.sv");
  testbench << R"sv(
module spatial_hierarchy_testbench;
  logic clock;
  logic reset;
)sv";
  testbench << portableAxiLiteSignalDeclarations() << "\n";
  testbench << R"sv(  loom_module dut(
    .clock(clock),
    .reset(reset),
    .cfg_awaddr(cfg_awaddr),
    .cfg_awvalid(cfg_awvalid),
    .cfg_awready(cfg_awready),
    .cfg_wdata(cfg_wdata),
    .cfg_wstrb(cfg_wstrb),
    .cfg_wvalid(cfg_wvalid),
    .cfg_wready(cfg_wready),
    .cfg_bresp(cfg_bresp),
    .cfg_bvalid(cfg_bvalid),
    .cfg_bready(cfg_bready),
    .cfg_araddr(cfg_araddr),
    .cfg_arvalid(cfg_arvalid),
    .cfg_arready(cfg_arready),
    .cfg_rdata(cfg_rdata),
    .cfg_rresp(cfg_rresp),
    .cfg_rvalid(cfg_rvalid),
    .cfg_rready(cfg_rready)
  );

  always #5 clock = ~clock;
)sv";
  testbench << portableAxiLiteDriverTasks();
  testbench << portableCycleWatchdog();
  testbench << R"sv(

  initial begin
    clock = 0;
    reset = 1;
)sv";
  testbench << portableAxiLiteInitialization();
  testbench << R"sv(
    repeat (2) @(posedge clock);
    @(negedge clock);
    reset = 0;
)sv";
  if (!inactiveConfigurations.empty()) {
    const PortableConfigurationTarget &probe =
        inactiveConfigurations.front().first;
    testbench << "    cfg_write_together(32'hfffffff0, 32'hdeadbeef, "
                 "4'hf, 2'b11);\n"
              << "    cfg_read(32'd" << probe.commitAddress
              << ", cfg_readback, cfg_read_response);\n"
              << "    if (cfg_read_response !== 2'b10)\n"
                 "      $fatal(1, \"configuration commit read did not return "
                 "SLVERR\");\n"
              << "    cfg_read(32'd" << probe.baseAddress + 1
              << ", cfg_readback, cfg_read_response);\n"
              << "    if (cfg_read_response !== 2'b11)\n"
                 "      $fatal(1, \"misaligned configuration read did not "
                 "return DECERR\");\n";
  }
  for (const auto &[target, image] : inactiveConfigurations) {
    auto program = portableAxiLiteProgramAndVerify(target, image);
    if (!program)
      return program.takeError();
    testbench << *program;
  }
  testbench << R"sv(    cfg_read(32'hfffffff0, cfg_readback, cfg_read_response);
    if (cfg_read_response !== 2'b11)
      $fatal(1, "unmapped configuration read did not return DECERR");
    $finish;
  end
endmodule
)sv";
  std::ofstream(root / "spatial_hierarchy.ys") << R"ys(
read_verilog -sv spatial_hierarchy_module.sv
hierarchy -check -top loom_module
check -assert
proc
synth -top loom_module
check -assert
select -assert-none loom_module/t:$dlatch loom_module/t:$_DLATCH_*
)ys";
  return llvm::Error::success();
}

llvm::Error writeRepeatedSpatialCoreToolArtifacts(
    const std::filesystem::path &root, llvm::StringRef systemVerilog,
    const PortableConfigurationTarget &target,
    llvm::ArrayRef<std::uint8_t> activeImage) {
  std::filesystem::create_directories(root);
  std::ofstream(root / "repeated_spatial_core_module.sv")
      << systemVerilog.str();
  std::ofstream testbench(root / "repeated_spatial_core_testbench.sv");
  testbench << R"sv(
module repeated_spatial_core_testbench;
  logic clock;
  logic reset;
)sv";
  testbench << portableAxiLiteSignalDeclarations();
  testbench << R"sv(
  logic cfg_awready_0, cfg_awready_1;
  logic cfg_wready_0, cfg_wready_1;
  logic [1:0] cfg_bresp_0, cfg_bresp_1;
  logic cfg_bvalid_0, cfg_bvalid_1;
  logic cfg_arready_0, cfg_arready_1;
  logic [31:0] cfg_rdata_0, cfg_rdata_1;
  logic [1:0] cfg_rresp_0, cfg_rresp_1;
  logic cfg_rvalid_0, cfg_rvalid_1;

  loom_module core_0(
    .clock(clock), .reset(reset),
    .cfg_awaddr(cfg_awaddr), .cfg_awvalid(cfg_awvalid),
    .cfg_awready(cfg_awready_0), .cfg_wdata(cfg_wdata),
    .cfg_wstrb(cfg_wstrb), .cfg_wvalid(cfg_wvalid),
    .cfg_wready(cfg_wready_0), .cfg_bresp(cfg_bresp_0),
    .cfg_bvalid(cfg_bvalid_0), .cfg_bready(cfg_bready),
    .cfg_araddr(cfg_araddr), .cfg_arvalid(cfg_arvalid),
    .cfg_arready(cfg_arready_0), .cfg_rdata(cfg_rdata_0),
    .cfg_rresp(cfg_rresp_0), .cfg_rvalid(cfg_rvalid_0),
    .cfg_rready(cfg_rready));
  loom_module core_1(
    .clock(clock), .reset(reset),
    .cfg_awaddr(cfg_awaddr), .cfg_awvalid(cfg_awvalid),
    .cfg_awready(cfg_awready_1), .cfg_wdata(cfg_wdata),
    .cfg_wstrb(cfg_wstrb), .cfg_wvalid(cfg_wvalid),
    .cfg_wready(cfg_wready_1), .cfg_bresp(cfg_bresp_1),
    .cfg_bvalid(cfg_bvalid_1), .cfg_bready(cfg_bready),
    .cfg_araddr(cfg_araddr), .cfg_arvalid(cfg_arvalid),
    .cfg_arready(cfg_arready_1), .cfg_rdata(cfg_rdata_1),
    .cfg_rresp(cfg_rresp_1), .cfg_rvalid(cfg_rvalid_1),
    .cfg_rready(cfg_rready));

  always_comb begin
    cfg_awready = cfg_awready_0 & cfg_awready_1;
    cfg_wready = cfg_wready_0 & cfg_wready_1;
    cfg_bvalid = cfg_bvalid_0 & cfg_bvalid_1;
    cfg_bresp = cfg_bresp_0 | cfg_bresp_1;
    cfg_arready = cfg_arready_0 & cfg_arready_1;
    cfg_rvalid = cfg_rvalid_0 & cfg_rvalid_1;
    cfg_rresp = cfg_rresp_0 | cfg_rresp_1;
    cfg_rdata = (cfg_rdata_0 === cfg_rdata_1) ? cfg_rdata_0 : 32'hx;
  end

  always #5 clock = ~clock;
)sv";
  testbench << portableAxiLiteDriverTasks();
  testbench << portableCycleWatchdog();
  testbench << R"sv(

  initial begin
    clock = 0;
    reset = 1;
)sv";
  testbench << portableAxiLiteInitialization();
  testbench << R"sv(
    repeat (2) @(posedge clock);
    @(negedge clock);
    reset = 0;
)sv";
  auto program = portableAxiLiteProgramAndVerify(target, activeImage);
  if (!program)
    return program.takeError();
  testbench << *program;
  testbench << R"sv(    $finish;
  end
endmodule
)sv";
  return llvm::Error::success();
}

} // namespace loom::hardware::test

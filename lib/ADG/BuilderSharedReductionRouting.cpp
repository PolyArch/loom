#include "BuilderInternal.h"

using namespace loom::adg;
using namespace loom::adg::detail;

ModuleBuilder loom::adg::buildSharedReductionAdg() {
  ModuleBuilder module("shared_reduction_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("i64a", "!fabric.bits<64>")
      .addInput("i64b", "!fabric.bits<64>")
      .addInput("i64c", "!fabric.bits<64>")
      .addInput("i32a", "!fabric.bits<32>")
      .addInput("i32b", "!fabric.bits<32>")
      .addInput("i32c", "!fabric.bits<32>")
      .addInput("i32d", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>");

  addSharedReductionComputeResources(module);

  auto addSingleResultBits32Switch =
      [&](llvm::StringRef result,
          std::initializer_list<llvm::StringRef> inputs) {
        std::vector<std::string> inputNames;
        inputNames.reserve(inputs.size());
        for (llvm::StringRef input : inputs)
          inputNames.push_back(input.str());
        std::vector<std::string> resultNames = {result.str()};
        addUniformSwitch(module, resultNames, inputNames, "!fabric.bits<32>");
      };
  auto addSingleResultBits64Switch =
      [&](llvm::StringRef result,
          std::initializer_list<llvm::StringRef> inputs) {
        std::vector<std::string> inputNames;
        inputNames.reserve(inputs.size());
        for (llvm::StringRef input : inputs)
          inputNames.push_back(input.str());
        std::vector<std::string> resultNames = {result.str()};
        addUniformSwitch(module, resultNames, inputNames, "!fabric.bits<64>");
      };
  addSingleResultBits32Switch(
      "logic_mask_lhs",
      {"i32a", "data0", "data1", "bit_carry", "addr_unscaled", "logic_shifted",
       "int_xor", "aux_xor", "cmpi_pred", "cmpi_pred_aux", "running"});
  addSingleResultBits32Switch(
      "logic_mask_rhs", {"i32b", "i32c", "reduction_scale", "fp_invariant",
                         "bit_invariant", "bit_invariant_aux0",
                         "bit_invariant_aux1", "cmpi_pred", "cmpi_pred_aux"});
  const std::initializer_list<llvm::StringRef> auxMaskLhsSources = {
      "carried_scan", "bit_carry", "state_carry", "selected", "addr_shifted",
      "running",      "idx",       "data0",       "data1"};
  const std::initializer_list<llvm::StringRef> auxMaskRhsSources = {
      "i32a",
      "i32b",
      "i32c",
      "i32d",
      "reduction_scale",
      "fp_invariant",
      "bit_invariant",
      "bit_invariant_aux0",
      "bit_invariant_aux1",
      "addr_shift_const",
      "addr_aux_const",
      "addr_bias_const"};
  const std::initializer_list<llvm::StringRef> auxXorLhsSources = {
      "selected",     "addr_shifted", "addr_unscaled", "logic_shifted",
      "carried_scan", "bit_carry",    "state_carry",   "logic_masked",
      "addr_masked",  "int_xor",      "aux_masked"};
  const std::initializer_list<llvm::StringRef> auxXorRhsSources = {
      "carried_scan",
      "bit_carry",
      "state_carry",
      "i32a",
      "i32b",
      "data0",
      "data1",
      "i32c",
      "i32d",
      "reduction_scale",
      "fp_invariant",
      "bit_invariant",
      "bit_invariant_aux0",
      "bit_invariant_aux1"};
  addSingleResultBits32Switch("aux_mask_lhs", auxMaskLhsSources);
  addSingleResultBits32Switch("aux_mask_rhs", auxMaskRhsSources);
  addSingleResultBits32Switch("aux_xor_lhs", auxXorLhsSources);
  addSingleResultBits32Switch("aux_xor_rhs", auxXorRhsSources);
  addSingleResultBits32Switch("int_add_lhs", {"i32a",
                                              "data1",
                                              "data0",
                                              "carried_scan",
                                              "running",
                                              "squared_data",
                                              "bit_carry",
                                              "reduction_scale",
                                              "int_product",
                                              "int_product_aux",
                                              "fp_invariant",
                                              "bit_invariant",
                                              "bit_invariant_aux0",
                                              "bit_invariant_aux1",
                                              "aux_invariant0",
                                              "aux_invariant1",
                                              "aux_invariant2",
                                              "cast0_result",
                                              "cast1_result",
                                              "cast2_result",
                                              "cast3_result",
                                              "int_extui",
                                              "wide_truncated",
                                              "wide_truncated_aux"});
  addSingleResultBits32Switch("int_add_rhs", {"i32b",
                                              "data0",
                                              "data1",
                                              "fp_invariant",
                                              "idx",
                                              "reduction_scale",
                                              "bit_invariant",
                                              "bit_invariant_aux0",
                                              "bit_invariant_aux1",
                                              "int_rem",
                                              "aux_idx",
                                              "aux_active_idx",
                                              "cast0_result",
                                              "cast1_result",
                                              "cast2_result",
                                              "cast3_result",
                                              "int_extui",
                                              "int_product",
                                              "int_product_aux",
                                              "squared_data"});
  addSingleResultBits32Switch(
      "int_mul_lhs",
      {"i32a", "int_xor", "data0", "data1", "int_div0", "int_div1", "aux_idx",
       "aux_active_idx", "bit_invariant", "bit_invariant_aux0",
       "bit_invariant_aux1", "cast0_result", "cast1_result", "cast2_result",
       "cast3_result", "int_sum", "running"});
  addSingleResultBits32Switch(
      "int_mul_rhs",
      {"i32b", "data0", "data1", "reduction_scale", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "aux_invariant0",
       "aux_invariant1", "aux_invariant2", "fp_invariant", "cast0_result",
       "cast1_result", "cast2_result", "cast3_result"});
  addSingleResultBits32Switch("int_mul_aux_lhs",
                              {"i32a", "int_xor", "data0", "data1", "int_div0",
                               "int_div1", "aux_idx", "aux_active_idx"});
  addSingleResultBits32Switch(
      "int_mul_aux_rhs",
      {"i32b", "data0", "data1", "reduction_scale", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "aux_invariant0",
       "aux_invariant1", "aux_invariant2", "fp_invariant", "aux_active_idx1"});
  addSingleResultBits32Switch(
      "int_div0_lhs",
      {"int_sum", "addr_sum", "aux_idx", "aux_active_idx", "i32b", "i32c"});
  addSingleResultBits32Switch("int_div0_rhs",
                              {"i32c", "reduction_scale", "fp_invariant",
                               "bit_invariant", "bit_invariant_aux0",
                               "bit_invariant_aux1", "aux_invariant0",
                               "aux_invariant1", "aux_invariant2"});
  addSingleResultBits32Switch(
      "int_div1_lhs",
      {"int_sum", "addr_sum", "aux_idx", "aux_active_idx", "i32b", "i32c"});
  addSingleResultBits32Switch(
      "int_div1_rhs",
      {"i32c", "reduction_scale", "fp_invariant", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "aux_invariant0",
       "aux_invariant1", "aux_invariant2", "aux_active_idx"});
  addSingleResultBits32Switch("int_rem_lhs",
                              {"aux_idx", "aux_active_idx", "i32a", "i32b"});
  addSingleResultBits32Switch(
      "int_rem_rhs", {"reduction_scale", "bit_invariant", "bit_invariant_aux0",
                      "bit_invariant_aux1", "aux_invariant0", "aux_invariant1",
                      "aux_invariant2", "i32d", "aux_active_idx"});
  addSingleResultBits32Switch("uint_rem_lhs",
                              {"int_product", "aux_idx", "aux_active_idx",
                               "i32b", "addr_shifted", "running"});
  addSingleResultBits32Switch(
      "uint_rem_rhs", {"i32c", "reduction_scale", "bit_invariant",
                       "bit_invariant_aux0", "bit_invariant_aux1",
                       "aux_invariant0", "aux_invariant1", "aux_invariant2"});
  addSingleResultBits32Switch(
      "int_or_lhs",
      {"i32a", "logic_masked", "data0", "data1", "addr_shifted", "selected"});
  addSingleResultBits32Switch("int_or_rhs",
                              {"i32b", "logic_masked", "data0", "data1"});
  addSingleResultBits32Switch(
      "int_xor_lhs",
      {"i32a", "rotated", "logic_shifted", "addr_unscaled", "logic_masked",
       "data0", "packed_sat", "selected", "addr_masked", "aux_masked",
       "cmpf_pred", "cmpi_pred", "cmpi_pred_aux"});
  addSingleResultBits32Switch(
      "int_xor_rhs",
      {"i32b", "data1", "data0", "logic_masked", "reduction_scale",
       "fp_invariant", "bit_invariant", "bit_invariant_aux0",
       "bit_invariant_aux1", "carried_scan", "bit_carry", "state_carry",
       "addr_masked", "selected", "aux_masked"});
  addSingleResultBits32Switch("packed_sat_lhs",
                              {"i32a", "reduction_scale", "fp_invariant",
                               "bit_invariant", "bit_invariant_aux0",
                               "bit_invariant_aux1", "cast0_result",
                               "cast1_result", "cast2_result", "cast3_result"});
  addSingleResultBits32Switch("packed_sat_rhs",
                              {"logic_masked", "addr_masked", "data0", "data1",
                               "i32b", "reduction_scale", "fp_invariant",
                               "bit_invariant", "bit_invariant_aux0",
                               "bit_invariant_aux1", "cast0_result",
                               "cast1_result", "cast2_result", "cast3_result"});
  addSingleResultBits32Switch(
      "minmax_lhs", {"i32a", "i32b", "data0", "data1", "idx", "running",
                     "int_sum", "addr_sum", "addr_masked", "logic_masked",
                     "carried_scan", "bit_carry", "state_carry", "cast0_result",
                     "cast1_result", "cast2_result", "cast3_result"});
  addSingleResultBits32Switch(
      "minmax_rhs",
      {"i32b", "i32c", "data0", "data1", "idx", "running", "int_sum",
       "addr_sum", "addr_shift_const", "addr_aux_const", "addr_bias_const",
       "reduction_scale", "fp_invariant", "bit_invariant", "bit_invariant_aux0",
       "bit_invariant_aux1"});
  addSingleResultBits32Switch(
      "rotate_lhs",
      {"i32a", "data1", "data0", "logic_masked", "int_sum", "int_product"});
  addSingleResultBits32Switch(
      "rotate_rhs",
      {"i32b", "data1", "data0", "logic_masked", "int_sum", "int_product"});
  addSingleResultBits32Switch(
      "rotate_amount",
      {"i32c", "data0", "reduction_scale", "addr_shift_const"});
  addSingleResultBits32Switch(
      "cmp_lhs", {"i32a", "logic_masked", "data0", "data1", "bit_carry",
                  "running", "addr_sum", "addr_masked", "aux_masked"});
  addSingleResultBits32Switch("cmp_rhs", {"i32b", "i32c", "reduction_scale",
                                          "data1", "data0", "fp_invariant",
                                          "bit_invariant", "bit_invariant_aux0",
                                          "bit_invariant_aux1", "aux_masked"});
  addSingleResultBits32Switch(
      "int_extui_input", {"i32a", "cmpi_pred", "cmpi_pred_aux", "cmpf_pred",
                          "cmpi64_pred_aux_narrow", "logic_masked",
                          "addr_masked", "int_xor", "aux_xor"});
  addSingleResultBits32Switch(
      "select_pred",
      {"i32a", "logic_masked", "addr_masked", "cmpi_pred", "cmpi_pred_aux",
       "cmpi64_pred_aux_narrow", "cmpf_pred", "aux_masked"});
  addSingleResultBits32Switch(
      "select_true",
      {"i32b", "idx", "data1", "rotated", "data0", "int_sum", "addr_sum",
       "addr_shifted", "reduction_scale", "bit_invariant", "bit_invariant_aux0",
       "bit_invariant_aux1", "cast0_result", "cast1_result", "cast2_result",
       "cast3_result", "aux_xor", "carried_scan"});
  addSingleResultBits32Switch("select_false",
                              {"i32c", "rotated", "data0", "data1",
                               "carried_scan", "bit_carry", "addr_shift_const",
                               "addr_aux_const", "addr_bias_const", "aux_xor",
                               "running", "addr_shifted"});
  addSingleResultBits32Switch(
      "gate_cond",
      {"aux_rwc", "logic_masked", "addr_masked", "cmpi_pred", "cmpi_pred_aux",
       "cmpi64_pred_aux_narrow", "cmpf_pred", "fp_gate", "i32a"});
  addSingleResultBits32Switch("gate_value",
                              {"aux_idx", "idx", "running", "addr_sum",
                               "int_sum", "squared_data", "carried_scan",
                               "cast0_result", "cast1_result", "cast2_result",
                               "cast3_result", "bit_invariant"});
  addSingleResultBits32Switch("gate_value1",
                              {"aux_idx", "idx", "running", "addr_sum",
                               "int_sum", "squared_data", "carried_scan",
                               "cast0_result", "cast1_result", "cast2_result",
                               "cast3_result", "bit_invariant_aux0"});
  addSingleResultBits32Switch(
      "demux_sel", {"logic_masked", "addr_masked", "cmpi_pred", "cmpi_pred_aux",
                    "cmpi64_pred_aux_narrow", "cmpf_pred", "fp_gate", "i32a"});
  addSingleResultBits32Switch("demux_value",
                              {"carried_scan", "bit_carry", "state_carry",
                               "fp_invariant", "reduction_scale", "running"});
  addSingleResultBits32Switch(
      "demux_then_value",
      {"mac_result", "mac_result1", "mac_result2", "mac_result3", "fp_running",
       "fp_running_aux", "scaled_reduction", "data0", "data1", "int_sum",
       "addr_sum", "int_product", "int_product_aux", "selected"});
  addSingleResultBits32Switch(
      "mux_sel", {"logic_masked", "addr_masked", "cmpi_pred", "cmpi_pred_aux",
                  "cmpi64_pred_aux_narrow", "cmpf_pred", "fp_gate", "i32a"});
  addSingleResultBits32Switch("mux_false",
                              {"control_demux_false", "carried_scan",
                               "bit_carry", "state_carry", "fp_invariant"});
  addSingleResultBits32Switch(
      "mux_true", {"compute_demux_true", "mac_result", "mac_result1",
                   "mac_result2", "mac_result3", "fp_running", "fp_running_aux",
                   "scaled_reduction", "data0", "data1"});
  addSingleResultBits32Switch("control_token_demux_sel",
                              {"logic_masked", "addr_masked", "cmpi_pred",
                               "cmpi_pred_aux", "cmpi64_pred_aux_narrow",
                               "cmpf_pred", "fp_gate", "i32a"});
  addFifo(module, "control_token_demux_false_token",
          "control_token_demux_false", "!fabric.bits<32>", "!fabric.bits<0>", 1,
          true, true);
  addFifo(module, "control_token_demux_true_token", "control_token_demux_true",
          "!fabric.bits<32>", "!fabric.bits<0>", 1, true, true);
  addFifo(module, "control_token_muxed_token", "control_token_muxed",
          "!fabric.bits<32>", "!fabric.bits<0>", 1, true, true);
  addSingleResultBits32Switch("control_token_mux_sel",
                              {"logic_masked", "addr_masked", "cmpi_pred",
                               "cmpi_pred_aux", "cmpi64_pred_aux_narrow",
                               "cmpf_pred", "fp_gate", "i32a"});
  addUniformSwitch(module, {"control_token_mux_false"},
                   {"control_token_demux_false_token", "store_done0", "ctrl"},
                   "!fabric.bits<0>");
  addUniformSwitch(module, {"control_token_mux_true"},
                   {"store_done0", "control_token_demux_true_token", "ctrl"},
                   "!fabric.bits<0>");
  addSingleResultBits32Switch("load1_addr", {"idx",
                                             "i32b",
                                             "addr_unscaled",
                                             "cast0_result",
                                             "cast1_result",
                                             "cast2_result",
                                             "cast3_result",
                                             "running",
                                             "addr_sum",
                                             "squared_data",
                                             "int_sum",
                                             "carried_scan",
                                             "aux_idx",
                                             "aux_active_idx",
                                             "selected",
                                             "logic_masked",
                                             "addr_masked",
                                             "int_extui",
                                             "addr_shift_const",
                                             "addr_aux_const",
                                             "addr_bias_const",
                                             "addr_extra_const0",
                                             "addr_extra_const1",
                                             "wide_index_cast0_narrow",
                                             "wide_index_cast1_narrow"});
  addSingleResultBits32Switch(
      "cast0_input", {"i32a", "data0", "data1", "logic_masked", "packed_sat",
                      "idx", "running", "int_sum", "addr_sum", "uint_rem"});
  addSingleResultBits32Switch("cast1_input",
                              {"i32a", "data0", "data1", "logic_masked",
                               "packed_sat", "idx", "running", "int_sum",
                               "addr_sum", "uint_rem", "cast0_result"});
  addSingleResultBits32Switch(
      "cast2_input",
      {"i32a", "data0", "data1", "logic_masked", "packed_sat", "idx", "running",
       "int_sum", "addr_sum", "uint_rem", "cast0_result", "cast1_result"});
  addSingleResultBits32Switch(
      "cast3_input", {"i32a", "data0", "data1", "logic_masked", "packed_sat",
                      "idx", "running", "int_sum", "addr_sum", "uint_rem",
                      "cast0_result", "cast1_result", "cast2_result"});
  addSingleResultBits32Switch("wide_zext0_input",
                              {"data0", "data1", "i32a", "cast0_result",
                               "cast1_result", "unsigned_minmax",
                               "unsigned_min", "signed_min", "signed_max"});
  addSingleResultBits32Switch(
      "wide_zext1_input",
      {"data1", "data0", "i32b", "cast0_result", "cast1_result"});
  addSingleResultBits64Switch("wide_mul_lhs",
                              {"wide_zext1", "wide_zext0", "i64a", "i64b"});
  addSingleResultBits64Switch("wide_mul_rhs",
                              {"wide_zext0", "wide_zext1", "i64a", "i64c"});
  addSingleResultBits64Switch(
      "wide_div_lhs", {"wide_product", "wide_zext0", "wide_zext1", "i64a"});
  addSingleResultBits64Switch(
      "wide_div_rhs", {"i64a", "i64b", "i64c", "wide_zext0", "wide_zext1"});
  addSingleResultBits64Switch(
      "wide_rem_lhs", {"wide_product", "wide_zext0", "wide_zext1", "i64a"});
  addSingleResultBits64Switch(
      "wide_rem_rhs", {"i64a", "i64b", "i64c", "wide_zext0", "wide_zext1"});
  addSingleResultBits64Switch("wide_add_lhs",
                              {"i64a", "i64b", "i64c", "wide_shifted",
                               "wide_zext0", "wide_zext1", "wide_product",
                               "wide_signed_quotient", "wide_remainder"});
  addSingleResultBits64Switch("wide_add_rhs",
                              {"i64a", "i64b", "i64c", "wide_shifted",
                               "wide_zext0", "wide_zext1", "wide_product",
                               "wide_signed_quotient", "wide_remainder"});
  addSingleResultBits64Switch(
      "wide_add_aux_lhs",
      {"i64a", "i64b", "i64c", "wide_shifted", "wide_sum", "wide_zext0",
       "wide_zext1", "wide_product", "wide_signed_quotient", "wide_remainder"});
  addSingleResultBits64Switch(
      "wide_add_aux_rhs",
      {"i64a", "i64b", "i64c", "wide_shifted", "wide_sum", "wide_zext0",
       "wide_zext1", "wide_product", "wide_signed_quotient", "wide_remainder"});
  addSingleResultBits64Switch(
      "wide_shift_lhs",
      {"i64a", "i64b", "i64c", "wide_sum", "wide_sum_aux", "wide_zext0",
       "wide_zext1", "wide_product", "wide_signed_quotient", "wide_remainder"});
  addSingleResultBits64Switch(
      "wide_shift_rhs", {"i64a", "i64b", "i64c", "wide_zext0", "wide_zext1"});
  addSingleResultBits64Switch("wide_trunc_input",
                              {"wide_remainder", "wide_product", "wide_zext0",
                               "wide_zext1", "wide_pred_extui",
                               "wide_signed_quotient", "wide_shifted",
                               "wide_sum", "wide_sum_aux"});
  addSingleResultBits64Switch("wide_trunc_aux_input",
                              {"wide_sum", "wide_sum_aux", "wide_shifted",
                               "wide_remainder", "wide_product",
                               "wide_signed_quotient", "wide_zext0",
                               "wide_zext1", "wide_pred_extui"});
  addSingleResultBits64Switch("wide_index_cast0_input",
                              {"i64a", "i64b", "i64c", "wide_zext0",
                               "wide_zext1", "wide_product", "wide_sum",
                               "wide_sum_aux", "wide_shifted",
                               "wide_signed_quotient", "wide_remainder"});
  addSingleResultBits64Switch("wide_index_cast1_input",
                              {"i64a", "i64b", "i64c", "wide_zext0",
                               "wide_zext1", "wide_product", "wide_sum",
                               "wide_sum_aux", "wide_shifted",
                               "wide_signed_quotient", "wide_remainder"});
  addSingleResultBits64Switch(
      "cmp64_lhs",
      {"i64a", "i64b", "i64c", "wide_zext0", "wide_zext1", "wide_product",
       "wide_signed_quotient", "wide_remainder", "wide_shifted"});
  addSingleResultBits64Switch(
      "cmp64_rhs",
      {"i64a", "i64b", "i64c", "wide_zext0", "wide_zext1", "wide_product",
       "wide_signed_quotient", "wide_remainder", "wide_shifted"});
  addSingleResultBits32Switch("fp_negated_input",
                              {"data0", "data1", "data2", "data3", "data4",
                               "data5", "fp_running", "fp_running_aux",
                               "fp_diff", "fp_diff_aux", "scaled_reduction"});
  addSingleResultBits32Switch("load2_addr", {"i32c",
                                             "cast0_result",
                                             "cast1_result",
                                             "cast2_result",
                                             "cast3_result",
                                             "idx",
                                             "addr_sum",
                                             "running",
                                             "squared_data",
                                             "int_sum",
                                             "aux_idx",
                                             "aux_active_idx",
                                             "data0",
                                             "data1",
                                             "int_extui",
                                             "addr_shift_const",
                                             "addr_aux_const",
                                             "addr_bias_const",
                                             "addr_extra_const0",
                                             "addr_extra_const1",
                                             "wide_index_cast0_narrow",
                                             "wide_index_cast1_narrow"});
  addSingleResultBits32Switch("store0_value", {"scan_store_value",
                                               "fp_running",
                                               "fp_running_aux",
                                               "running",
                                               "mac_result",
                                               "mac_result1",
                                               "mac_result2",
                                               "mac_result3",
                                               "data0",
                                               "data1",
                                               "data2",
                                               "data3",
                                               "data4",
                                               "data5",
                                               "selected",
                                               "rotated",
                                               "addr_masked",
                                               "logic_masked",
                                               "int_or",
                                               "int_xor",
                                               "packed_sat",
                                               "cast0_result",
                                               "cast1_result",
                                               "cast2_result",
                                               "cast3_result",
                                               "abs_data",
                                               "scaled_reduction",
                                               "scaled_reduction_aux",
                                               "int_product",
                                               "reduction_scale",
                                               "int_sum",
                                               "addr_sum",
                                               "fp_diff",
                                               "fp_diff_aux",
                                               "compute_demux_false",
                                               "compute_demux_true",
                                               "wide_truncated",
                                               "fp_negated",
                                               "signed_min",
                                               "signed_max"});
  addSingleResultBits32Switch(
      "store1_value",
      {"i32d", "data0", "data1", "data2", "data3", "data4", "data5", "selected",
       "scaled_reduction", "scaled_reduction_aux", "mac_result", "mac_result1",
       "mac_result2", "mac_result3", "signed_min", "signed_max"});
  addUniformSwitch(module, {"vector_sync_mid"},
                   {"done0", "done1", "store_done0",
                    "control_token_demux_false_token",
                    "control_token_muxed_token"},
                   "!fabric.bits<0>");
  addUniformSwitch(module, {"sync_head"},
                   {"done0", "store_done0",
                    "control_token_demux_false_token"},
                   "!fabric.bits<0>");
  addUniformSwitch(module, {"sync_tail"}, {"store_done0", "done2"},
                   "!fabric.bits<0>");
  addUniformSwitch(module, {"sync_extra"},
                   {"store_done1", "done3", "store_done0"}, "!fabric.bits<0>");
  addUniformSwitch(module, {"sync_lane5"}, {"done5", "store_done0"},
                   "!fabric.bits<0>");
  addUniformSwitch(module, {"sync_lane6"}, {"done1", "done4", "store_done0"},
                   "!fabric.bits<0>");
  addUniformSwitch(module, {"sync_lane7"},
                   {"done2", "done5", "control_token_muxed_token"},
                   "!fabric.bits<0>");
  const std::initializer_list<llvm::StringRef> typedSyncControls = {
      "ctrl",       "done0",       "done1",       "done2",
      "done3",      "done4",       "done5",       "store_done0",
      "store_done1", "vector_sync_done", "sync_done",
      "control_token_demux_false_token",
      "control_token_demux_true_token", "control_token_muxed_token"};
  addUniformSwitch(module, {"typed_sync_i1_control"}, typedSyncControls,
                   "!fabric.bits<0>");
  addUniformSwitch(module, {"typed_sync_i8_control"}, typedSyncControls,
                   "!fabric.bits<0>");
  addUniformSwitch(module, {"typed_sync_i32_control"}, typedSyncControls,
                   "!fabric.bits<0>");
  addUniformSwitch(module, {"typed_sync_i64_control"}, typedSyncControls,
                   "!fabric.bits<0>");
  const std::initializer_list<llvm::StringRef> typedSyncValues32 = {
      "i32a",          "i32b",          "i32c",
      "i32d",          "idx",           "aux_idx",
      "running",       "carried_scan",  "bit_carry",
      "state_carry",   "data0",         "data1",
      "data2",         "data3",         "data4",
      "data5",         "int_sum",       "addr_sum",
      "int_product",   "int_product_aux", "int_div0",
      "int_div1",      "int_rem",       "uint_rem",
      "int_or",        "int_xor",       "aux_xor",
      "logic_shifted", "addr_shifted",  "logic_masked",
      "addr_masked",   "aux_masked",    "selected",
      "rotated",       "packed_sat",    "leading_zero_count",
      "cast0_result",  "cast1_result",  "cast2_result",
      "cast3_result",  "int_extui",     "fp_running",
      "fp_running_aux", "fp_diff",      "fp_diff_aux",
      "fp_negated",    "scaled_reduction", "scaled_reduction_aux",
      "control_demux_false", "control_demux_true", "compute_demux_false",
      "compute_demux_true", "cmpi_pred",
      "cmpi_pred_aux", "cmpf_pred"};
  addUniformSwitch(module, {"typed_sync_i1_value"}, typedSyncValues32,
                   "!fabric.bits<32>");
  addUniformSwitch(module, {"typed_sync_i8_value"}, typedSyncValues32,
                   "!fabric.bits<32>");
  addUniformSwitch(module, {"typed_sync_i32_value"}, typedSyncValues32,
                   "!fabric.bits<32>");
  addUniformSwitch(
      module, {"typed_sync_i64_value"},
      {"i64a", "i64b", "i64c", "wide_zext0", "wide_zext1",
       "wide_product", "wide_signed_quotient", "wide_remainder",
       "wide_sum", "wide_sum_aux", "wide_shifted", "cmpi64_pred"},
      "!fabric.bits<64>");
  addSingleResultBits32Switch("addr_add_lhs",
                              {"idx", "i32a", "i32b", "i32c", "squared_data",
                               "int_product", "running", "reduction_scale",
                               "int_product_aux", "data0", "data1"});
  addSingleResultBits32Switch("addr_add_rhs",
                              {"fp_invariant", "reduction_scale", "i32a",
                               "i32b", "idx", "int_rem", "aux_idx",
                               "aux_active_idx", "carried_scan", "int_product",
                               "int_product_aux", "squared_data", "running"});
  addSingleResultBits32Switch(
      "addr_mask_lhs",
      {"addr_sum", "idx", "data0", "data1", "logic_masked", "carried_scan",
       "bit_carry", "state_carry", "selected", "aux_masked", "aux_xor"});
  addSingleResultBits32Switch("addr_mask_rhs",
                              {"reduction_scale", "fp_invariant", "i32b",
                               "i32c", "int_xor", "packed_sat", "logic_masked",
                               "bit_invariant", "bit_invariant_aux0",
                               "bit_invariant_aux1", "aux_masked", "aux_xor"});
  addSingleResultBits32Switch(
      "addr_unscale_lhs", {"i32a", "addr_shifted", "bit_carry", "data0",
                           "squared_data", "int_product", "int_product_aux"});
  addSingleResultBits32Switch("addr_unscale_rhs",
                              {"i32b", "addr_shifted", "addr_shift_const",
                               "reduction_scale",
                               "fp_invariant", "bit_invariant",
                               "bit_invariant_aux0", "bit_invariant_aux1"});
  addSingleResultBits32Switch("logic_shift_lhs",
                              {"i32a", "data0", "data1", "carried_scan",
                               "bit_carry", "state_carry", "running",
                               "addr_unscaled", "addr_shifted", "logic_masked",
                               "addr_masked", "int_xor", "aux_xor"});
  addSingleResultBits32Switch(
      "logic_shift_rhs",
      {"i32b", "addr_shifted", "reduction_scale", "addr_shift_const",
       "addr_aux_const", "addr_bias_const", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "aux_invariant0",
       "aux_invariant1", "aux_invariant2", "wide_truncated",
       "wide_truncated_aux"});
  addSingleResultBits32Switch(
      "addr_shift_lhs", {"i32a", "carried_scan", "idx", "bit_carry",
                         "state_carry", "selected", "aux_masked", "aux_xor"});
  addSingleResultBits32Switch(
      "addr_shift_rhs",
      {"i32b", "reduction_scale", "bit_invariant", "bit_invariant_aux0",
       "bit_invariant_aux1", "addr_shift_const", "addr_aux_const",
       "addr_bias_const", "fp_invariant", "aux_masked", "aux_xor",
       "aux_invariant0", "aux_invariant1", "aux_invariant2", "wide_truncated",
       "wide_truncated_aux"});
  addSingleResultBits32Switch("load0_addr", {"idx",
                                             "addr_masked",
                                             "addr_shifted",
                                             "addr_unscaled",
                                             "carried_scan",
                                             "bit_carry",
                                             "state_carry",
                                             "squared_data",
                                             "running",
                                             "addr_sum",
                                             "int_product",
                                             "int_sum",
                                             "aux_idx",
                                             "aux_active_idx",
                                             "cast0_result",
                                             "cast1_result",
                                             "cast2_result",
                                             "cast3_result",
                                             "selected",
                                             "addr_shift_const",
                                             "addr_aux_const",
                                             "addr_bias_const",
                                             "int_extui",
                                             "wide_index_cast0_narrow",
                                             "wide_index_cast1_narrow"});
  addSingleResultBits32Switch(
      "load3_addr",
      {"i32d", "carried_scan", "idx", "squared_data", "running", "addr_sum",
       "int_sum", "aux_idx", "aux_active_idx", "int_extui",
       "wide_index_cast0_narrow", "wide_index_cast1_narrow"});
  addSingleResultBits32Switch(
      "load4_addr",
      {"idx", "squared_data", "running", "addr_sum", "int_product", "int_sum",
       "addr_unscaled", "addr_shifted", "aux_idx", "aux_active_idx",
       "int_extui", "wide_index_cast0_narrow", "wide_index_cast1_narrow"});
  addSingleResultBits32Switch(
      "load5_addr",
      {"idx", "squared_data", "running", "addr_sum", "int_product", "int_sum",
       "addr_unscaled", "addr_shifted", "aux_idx", "aux_active_idx",
       "int_extui", "wide_index_cast0_narrow", "wide_index_cast1_narrow"});
  addSingleResultBits32Switch(
      "store0_addr",
      {"idx", "addr_unscaled", "carried_scan", "addr_shift_const",
       "state_carry", "addr_aux_const", "addr_bias_const", "addr_extra_const0",
       "addr_extra_const1", "int_sum", "addr_sum", "aux_idx", "running",
       "aux_active_idx", "control_demux_false", "control_demux_true",
       "int_extui", "wide_index_cast0_narrow", "wide_index_cast1_narrow"});
  addSingleResultBits32Switch(
      "store1_addr",
      {"i32c", "idx", "addr_unscaled", "carried_scan", "addr_shift_const",
       "addr_aux_const", "addr_bias_const", "addr_extra_const0",
       "addr_extra_const1", "int_sum", "addr_sum", "aux_idx", "running",
       "aux_active_idx", "int_extui", "wide_index_cast0_narrow",
       "wide_index_cast1_narrow"});
  addSingleResultBits32Switch(
      "aux_stream_lb",
      {"addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch(
      "aux_stream_ub", {"int_product", "int_product_aux", "squared_data"});
  addSingleResultBits32Switch(
      "aux_stream_step",
      {"addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch("aux_invariant_cond", {"aux_rwc", "fp_gate"});
  addSingleResultBits32Switch(
      "aux_invariant0_value",
      {"i32a", "i32b", "i32c", "i32d", "reduction_scale", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "fp_invariant",
       "addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch(
      "aux_invariant1_value",
      {"i32a", "i32b", "i32c", "i32d", "reduction_scale", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "fp_invariant",
       "aux_invariant0", "addr_shift_const", "addr_aux_const",
       "addr_bias_const"});
  const std::vector<std::string> storeControlInputs = {
      "ctrl",
      "done0",
      "done1",
      "done2",
      "done3",
      "done4",
      "done5",
      "control_token_demux_false_token",
      "control_token_demux_true_token"};
  addUniformSwitch(module, {"store0_ctrl"}, storeControlInputs,
                   "!fabric.bits<0>");
  addUniformSwitch(module, {"store1_ctrl"}, storeControlInputs,
                   "!fabric.bits<0>");
  const std::vector<std::string> reductionLoadOperands = {
      "load0_addr", "ctrl", "load1_addr", "ctrl", "load2_addr", "ctrl",
      "load3_addr", "ctrl", "load4_addr", "ctrl", "load5_addr", "ctrl"};
  const std::vector<std::string> reductionStoreOperands = {
      "store0_addr", "store0_value", "store0_ctrl",
      "store1_addr", "store1_value", "store1_ctrl"};
  appendBodyOp(
      module,
      bodyOpWithResultLine(
          {BodyResultSpec{"data0", "!fabric.bits<32>"},
           BodyResultSpec{"done0", "!fabric.bits<0>"},
           BodyResultSpec{"data1", "!fabric.bits<32>"},
           BodyResultSpec{"done1", "!fabric.bits<0>"},
           BodyResultSpec{"data2", "!fabric.bits<32>"},
           BodyResultSpec{"done2", "!fabric.bits<0>"},
           BodyResultSpec{"data3", "!fabric.bits<32>"},
           BodyResultSpec{"done3", "!fabric.bits<0>"},
           BodyResultSpec{"data4", "!fabric.bits<32>"},
           BodyResultSpec{"done4", "!fabric.bits<0>"},
           BodyResultSpec{"data5", "!fabric.bits<32>"},
           BodyResultSpec{"done5", "!fabric.bits<0>"},
           BodyResultSpec{"store_done0", "!fabric.bits<0>"},
           BodyResultSpec{"store_done1", "!fabric.bits<0>"}},
          {directHeadAndListLine("fabric.mem [spatial] mgr(", "mgr", ") load(",
                                 reductionLoadOperands, ")"),
           directOperandListLine("                              store(",
                                 reductionStoreOperands, ")"),
           exactBodyLine(
               "      [{load_group_size = 6 : i32, store_group_size = 2 : "
               "i32, data_width = 32 : i32, dispatch_eligibility = "
               "{operation_port_requests = [[0 : i32], [0 : i32], "
               "[0 : i32], [0 : i32], [0 : i32], [0 : i32], [0 : i32], "
               "[0 : i32]], subordinate_requests = []}}]"),
           exactBodyLine(
               "      : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, "
               "!fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, "
               "!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, "
               "!fabric.bits<0>, !fabric.bits<32>, !fabric.bits<0>, "
               "!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, "
               "!fabric.bits<32>, !fabric.bits<0>, !fabric.bits<32>, "
               "!fabric.bits<32>, !fabric.bits<0>)")},
          "      -> "));
  addSingleResultBits32Switch("mul_lhs_input", {"data0",
                                                "data1",
                                                "data2",
                                                "idx",
                                                "data4",
                                                "int_div0",
                                                "int_div1",
                                                "aux_idx",
                                                "aux_active_idx",
                                                "bit_invariant",
                                                "bit_invariant_aux0",
                                                "bit_invariant_aux1",
                                                "cast0_result",
                                                "cast1_result",
                                                "cast2_result",
                                                "cast3_result",
                                                "aux_invariant0",
                                                "aux_invariant1",
                                                "aux_invariant2",
                                                "int_sum",
                                                "running"});
  addSingleResultBits32Switch(
      "mul_rhs_input",
      {"data0", "data1", "data2", "data4", "reduction_scale", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "aux_invariant0",
       "aux_invariant1", "aux_invariant2", "fp_invariant", "cast0_result",
       "cast1_result", "cast2_result", "cast3_result"});
  addSingleResultBits32Switch("reduction_input",
                              {"data0", "abs_data", "squared_data"});
  addSingleResultBits32Switch(
      "stream_sum_lhs",
      {"reduction_input", "carried_scan", "bit_carry", "state_carry",
       "int_product", "int_product_aux", "bit_invariant", "bit_invariant_aux0",
       "bit_invariant_aux1", "cast0_result", "cast1_result", "cast2_result",
       "cast3_result", "int_extui"});
  addSingleResultBits32Switch(
      "stream_sum_rhs",
      {"carried_scan", "fp_invariant", "reduction_scale", "bit_invariant_aux1",
       "int_rem", "aux_idx", "aux_invariant0", "aux_invariant1",
       "aux_invariant2", "bit_invariant", "bit_invariant_aux0", "cast0_result",
       "cast1_result", "cast2_result", "cast3_result", "addr_shifted",
       "int_extui"});
  addSingleResultBits32Switch(
      "scan_init",
      {"i32a", "addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch(
      "scan_scale",
      {"i32b", "addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch("fp_lhs",
                              {"carried_scan", "data0", "data2", "data4",
                               "reduction_scale", "mac_result1"});
  addSingleResultBits32Switch(
      "fp_rhs", {"data0", "data1", "data3", "data5", "reduction_scale"});
  addSingleResultBits32Switch("fp_lhs_aux",
                              {"bit_carry", "state_carry", "carried_scan",
                               "data0", "data1", "data2", "data4",
                               "reduction_scale", "mac_result1"});
  addSingleResultBits32Switch("fp_rhs_aux", {"data1", "data0", "data3", "data5",
                                             "reduction_scale", "fp_invariant",
                                             "bit_invariant"});
  addSingleResultBits32Switch("fp_diff_lhs", {"i32a", "data0"});
  addSingleResultBits32Switch("fp_diff_rhs",
                              {"i32b", "fp_invariant", "data1", "fp_div"});
  addSingleResultBits32Switch("fp_diff_aux_lhs", {"data1", "data0", "i32a"});
  addSingleResultBits32Switch("fp_diff_aux_rhs",
                              {"bit_invariant", "fp_invariant",
                               "bit_invariant_aux0", "bit_invariant_aux1",
                               "aux_invariant0", "aux_invariant1",
                               "aux_invariant2", "i32b", "data1", "fp_div"});
  addSingleResultBits32Switch("fp_div_lhs", {"data1", "data0"});
  addSingleResultBits32Switch("fp_div_rhs",
                              {"data2", "fp_invariant", "reduction_scale"});
  addSingleResultBits32Switch(
      "fp_invariant_value",
      {"i32b", "addr_shift_const", "addr_aux_const", "addr_bias_const"});
  addSingleResultBits32Switch("bit_invariant_value",
                              {"i32d", "reduction_scale"});
  addSingleResultBits32Switch("bit_invariant_aux0_value",
                              {"i32c", "fp_invariant"});
  addSingleResultBits32Switch("bit_invariant_aux1_value",
                              {"i32b", "reduction_scale", "addr_shift_const",
                               "addr_aux_const", "addr_bias_const"});
  const std::initializer_list<llvm::StringRef> scaledReductionLhsInputs = {
      "carried_scan",
      "fp_running",
      "fp_running_aux",
      "data1",
      "data3",
      "data5",
      "data0",
      "bit_invariant",
      "bit_invariant_aux0",
      "bit_invariant_aux1",
      "aux_invariant0",
      "aux_invariant1",
      "aux_invariant2",
      "fp_negated",
      "reduction_scale"};
  const std::initializer_list<llvm::StringRef> scaledReductionRhsInputs = {
      "reduction_scale", "data4",          "data5",      "data1",
      "data3",           "state_carry",    "bit_carry",  "aux_invariant0",
      "aux_invariant1",  "aux_invariant2", "fp_negated", "data0"};
  addSingleResultBits32Switch("scaled_reduction_lhs", scaledReductionLhsInputs);
  addSingleResultBits32Switch("scaled_reduction_rhs", scaledReductionRhsInputs);
  addSingleResultBits32Switch("scaled_reduction_aux_lhs",
                              scaledReductionLhsInputs);
  addSingleResultBits32Switch("scaled_reduction_aux_rhs",
                              scaledReductionRhsInputs);
  addSingleResultBits32Switch(
      "mac_lhs",
      {"i32a", "data0", "data2", "data4", "fp_diff", "fp_diff_aux",
       "scaled_reduction", "fp_invariant", "bit_invariant",
       "bit_invariant_aux0", "data1", "bit_invariant_aux1", "reduction_scale"});
  addSingleResultBits32Switch("mac_rhs", {"i32b", "data1", "data2", "data3",
                                          "data5", "fp_diff", "fp_diff_aux",
                                          "data0", "bit_carry", "state_carry"});
  addSingleResultBits32Switch("mac_acc",
                              {"i32c", "carried_scan", "bit_carry",
                               "scaled_reduction", "state_carry", "data0"});
  addSingleResultBits32Switch(
      "mac1_lhs", {"i32a", "data2", "data4", "data0", "fp_diff", "fp_diff_aux",
                   "fp_invariant", "bit_invariant", "bit_invariant_aux0",
                   "bit_invariant_aux1", "reduction_scale"});
  addSingleResultBits32Switch(
      "mac1_rhs", {"i32b", "data3", "data5", "data1", "fp_diff", "fp_diff_aux",
                   "bit_carry", "state_carry", "carried_scan"});
  addSingleResultBits32Switch("mac1_acc",
                              {"i32c", "mac_result", "scaled_reduction",
                               "carried_scan", "bit_carry", "state_carry"});
  addSingleResultBits32Switch(
      "mac2_lhs",
      {"i32a", "data0", "data2", "data4", "fp_invariant", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "reduction_scale"});
  addSingleResultBits32Switch("mac2_rhs",
                              {"i32b", "data0", "data1", "data3", "data5",
                               "bit_carry", "state_carry", "carried_scan"});
  addSingleResultBits32Switch("mac2_acc",
                              {"mac_result1", "mac_result", "scaled_reduction",
                               "bit_carry", "state_carry"});
  addSingleResultBits32Switch(
      "mac3_lhs",
      {"i32a", "data0", "data2", "data4", "fp_invariant", "bit_invariant",
       "bit_invariant_aux0", "bit_invariant_aux1", "reduction_scale"});
  addSingleResultBits32Switch("mac3_rhs",
                              {"i32b", "data0", "data1", "data3", "data5",
                               "bit_carry", "state_carry", "carried_scan",
                               "fp_running", "fp_running_aux"});
  addSingleResultBits32Switch(
      "mac3_acc", {"mac_result2", "mac_result1", "mac_result",
                   "scaled_reduction", "bit_carry", "state_carry", "data4"});
  addSingleResultBits32Switch("bit_carry_cond", {"i32a", "fp_gate"});
  addSingleResultBits32Switch(
      "bit_carry_init",
      {"i32b", "i32c", "addr_shift_const", "addr_bias_const"});
  addSingleResultBits32Switch(
      "bit_carry_next",
      {"i32c", "addr_unscaled", "mac_result", "mac_result1", "int_sum",
       "selected", "running", "mac_result2", "mac_result3", "data0",
       "state_carry", "aux_masked", "aux_xor", "fp_running_aux"});
  addSingleResultBits32Switch("state_carry_cond", {"fp_gate", "i32a"});
  addSingleResultBits32Switch("state_carry_init",
                              {"i32a", "i32b", "i32c", "i32d",
                               "addr_shift_const", "addr_aux_const",
                               "addr_bias_const", "data0", "data1"});
  addSingleResultBits32Switch("state_carry_next",
                              {"mac_result", "mac_result1", "mac_result2",
                               "mac_result3", "bit_carry", "carried_scan",
                               "int_sum", "data0", "running", "aux_masked",
                               "aux_xor", "fp_running_aux"});
  addSpatialSwitch(
      module, {"scan_feedback", "scan_store_value"},
      {"running",     "fp_running",      "mac_result",     "mac_result1",
       "mac_result2", "mac_result3",     "bit_carry",      "state_carry",
       "int_or",      "selected",        "int_sum",        "addr_sum",
       "int_product", "int_product_aux", "control_muxed",  "int_xor",
       "aux_masked",  "aux_xor",         "fp_running_aux", "uint_rem"},
      {"11111111111111111111", "00111100000000000000"});
  addUniformSwitch(module, {"sync_aux_done"},
                   {"store_done0", "done1", "done2", "done3", "done4", "done5",
                    "control_token_muxed_token"},
                   "!fabric.bits<0>");
  return module;
}

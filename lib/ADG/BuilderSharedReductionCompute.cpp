#include "BuilderInternal.h"

#include "llvm/ADT/STLExtras.h"

using namespace loom::adg;
using namespace loom::adg::detail;

void loom::adg::detail::addSharedReductionComputeResources(
    ModuleBuilder &module) {

  PeSpec streamPe;
  streamPe.inputs = {{"pa", "i64a", "!fabric.bits<64>", "!fabric.bits<32>"},
                     {"pb", "i64b", "!fabric.bits<64>", "!fabric.bits<32>"},
                     {"pc", "i64c", "!fabric.bits<64>", "!fabric.bits<32>"},
                     {"pd", "stream_sum_lhs", "!fabric.bits<32>", ""},
                     {"pe", "stream_sum_rhs", "!fabric.bits<32>", ""},
                     {"pi", "scan_init", "!fabric.bits<32>", ""},
                     {"pn", "scan_feedback", "!fabric.bits<32>", ""},
                     {"ps", "scan_scale", "!fabric.bits<32>", ""}};
  streamPe.resultNames = {"idx", "running", "carried_scan", "reduction_scale",
                          "fp_gate"};
  streamPe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>",
                          "!fabric.bits<32>", "!fabric.bits<32>",
                          "!fabric.bits<32>"};
  FuSpec streamFu;
  streamFu.inputs = {{"fa", "pa", "!fabric.bits<32>", ""},
                     {"fb", "pb", "!fabric.bits<32>", ""},
                     {"fc", "pc", "!fabric.bits<32>", ""},
                     {"sum_lhs", "pd", "!fabric.bits<32>", ""},
                     {"sum_rhs", "pe", "!fabric.bits<32>", ""},
                     {"init", "pi", "!fabric.bits<32>", ""},
                     {"next", "pn", "!fabric.bits<32>", ""},
                     {"scale", "ps", "!fabric.bits<32>", ""}};
  streamFu.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>",
                          "!fabric.bits<32>", "!fabric.bits<32>",
                          "!fabric.bits<32>"};
  streamFu.operations.push_back(
      FabricOpSpec{{"idx", "rwc"},
                   {"dataflow.stream"},
                   {"fa", "fb", "fc"},
                   {"!fabric.bits<32>", "!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>", "!fabric.bits<1>"},
                   {{"cont_cond", {"<", ">"}}, {"step_op", {"+="}}},
                   {{"cont_cond", "<"}, {"step_op", "+="}}});
  streamFu.operations.push_back(
      FabricOpSpec{{"carried"},
                   {"dataflow.carry"},
                   {"rwc", "init", "next"},
                   {"!fabric.bits<1>", "!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"},
                   {},
                   {}});
  streamFu.operations.push_back(
      FabricOpSpec{{"sum"},
                   {"arith.addi"},
                   {"sum_lhs", "sum_rhs"},
                   {"!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"},
                   {},
                   {}});
  streamFu.operations.push_back(
      FabricOpSpec{{"stable_scale"},
                   {"dataflow.invariant"},
                   {"rwc", "scale"},
                   {"!fabric.bits<1>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"},
                   {},
                   {}});
  streamFu.yieldValues = {"idx", "sum", "carried", "stable_scale", "rwc"};
  streamFu.yieldTypes = {"!fabric.bits<32>", "!fabric.bits<32>",
                         "!fabric.bits<32>", "!fabric.bits<32>",
                         "!fabric.bits<1>"};
  streamPe.fus.push_back(std::move(streamFu));
  module.addPe(std::move(streamPe));

  PeSpec auxStreamPe;
  auxStreamPe.inputs = {{"pa", "aux_stream_lb", "!fabric.bits<32>", ""},
                        {"pb", "aux_stream_ub", "!fabric.bits<32>", ""},
                        {"pc", "aux_stream_step", "!fabric.bits<32>", ""}};
  auxStreamPe.resultNames = {"aux_idx", "aux_rwc"};
  auxStreamPe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  FuSpec auxStreamFu;
  auxStreamFu.inputs = {{"fa", "pa", "!fabric.bits<32>", ""},
                        {"fb", "pb", "!fabric.bits<32>", ""},
                        {"fc", "pc", "!fabric.bits<32>", ""}};
  auxStreamFu.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  auxStreamFu.operations.push_back(
      FabricOpSpec{{"aux_op_idx", "aux_op_rwc"},
                   {"dataflow.stream"},
                   {"fa", "fb", "fc"},
                   {"!fabric.bits<32>", "!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>", "!fabric.bits<1>"},
                   {{"cont_cond", {"<", ">"}}, {"step_op", {"+="}}},
                   {{"cont_cond", "<"}, {"step_op", "+="}}});
  auxStreamFu.yieldValues = {"aux_op_idx", "aux_op_rwc"};
  auxStreamFu.yieldTypes = {"!fabric.bits<32>", "!fabric.bits<1>"};
  auxStreamPe.fus.push_back(std::move(auxStreamFu));
  module.addPe(std::move(auxStreamPe));

  PeSpec auxGatePe;
  auxGatePe.inputs = {{"pa", "gate_cond", "!fabric.bits<32>", ""},
                      {"pb", "gate_value", "!fabric.bits<32>", ""}};
  auxGatePe.resultNames = {"aux_gate_cond", "aux_active_idx"};
  auxGatePe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  auto makeGateFu = []() {
    FuSpec fu;
    fu.inputs = {{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                 {"value", "pb", "!fabric.bits<32>", ""}};
    fu.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
    fu.operations.push_back(
        FabricOpSpec{{"after_cond", "after_value"},
                     {"dataflow.gate"},
                     {"cond", "value"},
                     {"!fabric.bits<1>", "!fabric.bits<32>"},
                     {"!fabric.bits<1>", "!fabric.bits<32>"},
                     {},
                     {{"value_kind", "data"}}});
    fu.yieldValues = {"after_cond", "after_value"};
    fu.yieldTypes = {"!fabric.bits<1>", "!fabric.bits<32>"};
    return fu;
  };
  auxGatePe.fus.push_back(makeGateFu());
  module.addPe(std::move(auxGatePe));

  PeSpec auxGatePe1;
  auxGatePe1.inputs = {{"pa", "gate_cond", "!fabric.bits<32>", ""},
                       {"pb", "gate_value1", "!fabric.bits<32>", ""}};
  auxGatePe1.resultNames = {"aux_gate_cond1", "aux_active_idx1"};
  auxGatePe1.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  auxGatePe1.fus.push_back(makeGateFu());
  auxGatePe1.fus.push_back(makeGateFu());
  auxGatePe1.fus.push_back(makeGateFu());
  module.addPe(std::move(auxGatePe1));

  PeSpec absPe;
  absPe.inputs = {{"pa", "data0", "!fabric.bits<32>", ""}};
  absPe.resultNames = {"abs_data"};
  absPe.resultTypes = {"!fabric.bits<32>"};
  absPe.fus.push_back(FuSpec{{{"value", "pa", "!fabric.bits<32>", ""}},
                             {"!fabric.bits<32>"},
                             {FabricOpSpec{{"abs"},
                                           {"llvm.intr.abs"},
                                           {"value"},
                                           {"!fabric.bits<32>"},
                                           {"!fabric.bits<32>"},
                                           {},
                                           {}}},
                             {"abs"}});
  absPe.fus.push_back(FuSpec{{{"value", "pa", "!fabric.bits<32>", ""}},
                             {"!fabric.bits<32>"},
                             {FabricOpSpec{{"abs"},
                                           {"llvm.intr.fabs"},
                                           {"value"},
                                           {"!fabric.bits<32>"},
                                           {"!fabric.bits<32>"},
                                           {},
                                           {}}},
                             {"abs"}});
  module.addPe(std::move(absPe));

  PeSpec squaredPe;
  squaredPe.inputs = {{"pa", "mul_lhs_input", "!fabric.bits<32>", ""},
                      {"pb", "mul_rhs_input", "!fabric.bits<32>", ""}};
  squaredPe.resultNames = {"squared_data"};
  squaredPe.resultTypes = {"!fabric.bits<32>"};
  squaredPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"product"},
                           {"arith.muli"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"product"}});
  module.addPe(std::move(squaredPe));

  auto addFpBinaryPe = [&](std::string resultName, std::string lhsInput,
                           std::string rhsInput, llvm::StringRef valueName,
                           llvm::StringRef opName) {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(lhsInput), "!fabric.bits<32>", ""},
                 {"pb", std::move(rhsInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                {"rhs", "pb", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>"},
               {FabricOpSpec{{valueName.str()},
                             {opName.str()},
                             {"lhs", "rhs"},
                             {"!fabric.bits<32>", "!fabric.bits<32>"},
                             {"!fabric.bits<32>"},
                             {},
                             {}}},
               {valueName.str()}});
    module.addPe(std::move(pe));
  };

  addFpBinaryPe("fp_running", "fp_lhs", "fp_rhs", "sum", "arith.addf");
  addFpBinaryPe("fp_running_aux", "fp_lhs_aux", "fp_rhs_aux", "sum",
                "arith.addf");

  PeSpec fpInvariantPe;
  fpInvariantPe.inputs = {{"pa", "fp_gate", "!fabric.bits<32>", ""},
                          {"pb", "fp_invariant_value", "!fabric.bits<32>", ""}};
  fpInvariantPe.resultNames = {"fp_invariant"};
  fpInvariantPe.resultTypes = {"!fabric.bits<32>"};
  fpInvariantPe.fus.push_back(
      FuSpec{{{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
              {"value", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"stable"},
                           {"dataflow.invariant"},
                           {"cond", "value"},
                           {"!fabric.bits<1>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"stable"}});
  module.addPe(std::move(fpInvariantPe));

  auto addInvariantPe = [&](std::string resultName, std::string valueInput,
                            std::string condInput = "fp_gate") {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(condInput), "!fabric.bits<32>", ""},
                 {"pb", std::move(valueInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        FuSpec{{{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                {"value", "pb", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>"},
               {FabricOpSpec{{"stable"},
                             {"dataflow.invariant"},
                             {"cond", "value"},
                             {"!fabric.bits<1>", "!fabric.bits<32>"},
                             {"!fabric.bits<32>"},
                             {},
                             {}}},
               {"stable"}});
    module.addPe(std::move(pe));
  };
  addInvariantPe("bit_invariant", "bit_invariant_value", "aux_invariant_cond");
  addInvariantPe("bit_invariant_aux0", "bit_invariant_aux0_value",
                 "aux_invariant_cond");
  addInvariantPe("aux_invariant2", "bit_invariant_aux1_value",
                 "aux_invariant_cond");
  addInvariantPe("bit_invariant_aux1", "bit_invariant_aux1_value");
  auto addAuxInvariantPe = [&](std::string resultName, std::string valueInput) {
    PeSpec pe;
    pe.inputs = {{"pa", "aux_invariant_cond", "!fabric.bits<32>", ""},
                 {"pb", std::move(valueInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        FuSpec{{{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                {"value", "pb", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>"},
               {FabricOpSpec{{"stable"},
                             {"dataflow.invariant"},
                             {"cond", "value"},
                             {"!fabric.bits<1>", "!fabric.bits<32>"},
                             {"!fabric.bits<32>"},
                             {},
                             {}}},
               {"stable"}});
    module.addPe(std::move(pe));
  };
  addAuxInvariantPe("aux_invariant0", "aux_invariant0_value");
  addAuxInvariantPe("aux_invariant1", "aux_invariant1_value");

  addFpBinaryPe("fp_diff", "fp_diff_lhs", "fp_diff_rhs", "diff", "arith.subf");
  addFpBinaryPe("fp_diff_aux", "fp_diff_aux_lhs", "fp_diff_aux_rhs", "diff",
                "arith.subf");

  PeSpec scaledReductionPe;
  scaledReductionPe.inputs = {
      {"pa", "scaled_reduction_lhs", "!fabric.bits<32>", ""},
      {"pb", "scaled_reduction_rhs", "!fabric.bits<32>", ""}};
  scaledReductionPe.resultNames = {"scaled_reduction"};
  scaledReductionPe.resultTypes = {"!fabric.bits<32>"};
  scaledReductionPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"product"},
                           {"arith.mulf"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"product"}});
  module.addPe(std::move(scaledReductionPe));

  auto makeCarryFu = []() {
    return FuSpec{{{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                   {"init", "pb", "!fabric.bits<32>", ""},
                   {"next", "pc", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{"carried"},
                                {"dataflow.carry"},
                                {"cond", "init", "next"},
                                {"!fabric.bits<1>", "!fabric.bits<32>",
                                 "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {"carried"}};
  };
  PeSpec carryPe;
  carryPe.inputs = {{"pa", "bit_carry_cond", "!fabric.bits<32>", ""},
                    {"pb", "bit_carry_init", "!fabric.bits<32>", ""},
                    {"pc", "bit_carry_next", "!fabric.bits<32>", ""}};
  carryPe.resultNames = {"bit_carry"};
  carryPe.resultTypes = {"!fabric.bits<32>"};
  carryPe.fus.push_back(makeCarryFu());
  carryPe.fus.push_back(makeCarryFu());
  module.addPe(std::move(carryPe));

  auto makeBinary32Fu = [](std::string resultName,
                           std::vector<std::string> opList) {
    std::string yieldName = resultName;
    return FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                   {"rhs", "pb", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{std::move(resultName)},
                                std::move(opList),
                                {"lhs", "rhs"},
                                {"!fabric.bits<32>", "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {std::move(yieldName)}};
  };
  auto addBinary32Pe = [&](std::string peResultName, std::string lhsInput,
                           std::string rhsInput, std::string opResultName,
                           std::vector<std::string> opList) {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(lhsInput), "!fabric.bits<32>", ""},
                 {"pb", std::move(rhsInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(peResultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        makeBinary32Fu(std::move(opResultName), std::move(opList)));
    module.addPe(std::move(pe));
  };
  addBinary32Pe("int_sum", "int_add_lhs", "int_add_rhs", "sum",
                {"arith.addi", "arith.subi"});
  addBinary32Pe("int_product", "int_mul_lhs", "int_mul_rhs", "product",
                {"arith.muli"});
  addBinary32Pe("int_product_aux", "int_mul_aux_lhs", "int_mul_aux_rhs",
                "product", {"arith.muli"});
  addBinary32Pe("int_div0", "int_div0_lhs", "int_div0_rhs", "quotient",
                {"arith.divsi"});
  addBinary32Pe("int_div1", "int_div1_lhs", "int_div1_rhs", "quotient",
                {"arith.divsi"});
  addBinary32Pe("int_rem", "int_rem_lhs", "int_rem_rhs", "remainder",
                {"arith.remsi"});
  addBinary32Pe("uint_rem", "uint_rem_lhs", "uint_rem_rhs", "remainder",
                {"arith.divui", "arith.remui"});
  addBinary32Pe("fp_div", "fp_div_lhs", "fp_div_rhs", "quotient",
                {"arith.divf", "arith.remf"});
  auto addConfigurableConstPe = [&](std::string resultName) {
    PeSpec constPe;
    constPe.inputs = {{"pa", "ctrl", "!fabric.bits<0>", "!fabric.bits<32>"}};
    constPe.resultNames = {std::move(resultName)};
    constPe.resultTypes = {"!fabric.bits<32>"};
    constPe.fus.push_back(FuSpec{
        {{"ctrl_in", "pa", "!fabric.bits<32>", "!fabric.bits<0>"}},
        {"!fabric.bits<32>"},
        {FabricOpSpec{
            {"value"},
            {"dataflow.constant"},
            {"ctrl_in"},
            {"!fabric.bits<0>"},
            {"!fabric.bits<32>"},
            {{"const_hex_value", {"0x00000000", "0x00000001", "0x00000002"}}},
            {}}},
        {"value"}});
    module.addPe(std::move(constPe));
  };
  addConfigurableConstPe("addr_shift_const");
  addConfigurableConstPe("addr_aux_const");
  addConfigurableConstPe("addr_bias_const");
  addConfigurableConstPe("addr_extra_const0");
  addConfigurableConstPe("addr_extra_const1");
  addBinary32Pe("logic_shifted", "logic_shift_lhs", "logic_shift_rhs",
                "shifted", {"arith.shrsi", "arith.shrui"});

  PeSpec addrUnscalePe;
  addrUnscalePe.inputs = {{"pa", "addr_unscale_lhs", "!fabric.bits<32>", ""},
                          {"pb", "addr_unscale_rhs", "!fabric.bits<32>", ""}};
  addrUnscalePe.resultNames = {"addr_unscaled"};
  addrUnscalePe.resultTypes = {"!fabric.bits<32>"};
  addrUnscalePe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"shifted"},
                           {"arith.shrsi", "arith.shrui"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"shifted"}});
  module.addPe(std::move(addrUnscalePe));
  PeSpec addrShiftPe;
  addrShiftPe.inputs = {{"pa", "addr_shift_lhs", "!fabric.bits<32>", ""},
                        {"pb", "addr_shift_rhs", "!fabric.bits<32>", ""}};
  addrShiftPe.resultNames = {"addr_shifted"};
  addrShiftPe.resultTypes = {"!fabric.bits<32>"};
  addrShiftPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"shifted"},
                           {"arith.shli"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"shifted"}});
  module.addPe(std::move(addrShiftPe));
  PeSpec logicMaskPe;
  logicMaskPe.inputs = {{"pa", "logic_mask_lhs", "!fabric.bits<32>", ""},
                        {"pb", "logic_mask_rhs", "!fabric.bits<32>", ""}};
  logicMaskPe.resultNames = {"logic_masked"};
  logicMaskPe.resultTypes = {"!fabric.bits<32>"};
  logicMaskPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"masked"},
                           {"arith.andi"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"masked"}});
  module.addPe(std::move(logicMaskPe));
  addBinary32Pe("int_or", "int_or_lhs", "int_or_rhs", "combined",
                {"arith.ori"});
  addBinary32Pe("int_xor", "int_xor_lhs", "int_xor_rhs", "combined",
                {"arith.xori"});
  PeSpec packedSatPe;
  packedSatPe.inputs = {{"pa", "packed_sat_lhs", "!fabric.bits<32>", ""},
                        {"pb", "packed_sat_rhs", "!fabric.bits<32>", ""}};
  packedSatPe.resultNames = {"packed_sat"};
  packedSatPe.resultTypes = {"!fabric.bits<32>"};
  packedSatPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"packed"},
                           {"llvm.arm.qadd16", "llvm.arm.sadd16",
                            "llvm.arm.qsub16", "llvm.arm.qsub8"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"packed"}});
  module.addPe(std::move(packedSatPe));

  auto addFmulAddPe = [&](std::string resultName, std::string lhsInput,
                          std::string rhsInput, std::string accInput) {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(lhsInput), "!fabric.bits<32>", ""},
                 {"pb", std::move(rhsInput), "!fabric.bits<32>", ""},
                 {"pc", std::move(accInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                {"rhs", "pb", "!fabric.bits<32>", ""},
                {"acc", "pc", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>"},
               {FabricOpSpec{
                   {"mac"},
                   {"llvm.intr.fmuladd"},
                   {"lhs", "rhs", "acc"},
                   {"!fabric.bits<32>", "!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"},
                   {},
                   {}}},
               {"mac"}});
    module.addPe(std::move(pe));
  };
  addFmulAddPe("mac_result", "mac_lhs", "mac_rhs", "mac_acc");
  addFmulAddPe("mac_result1", "mac1_lhs", "mac1_rhs", "mac1_acc");

  PeSpec unsignedMinMaxPe;
  unsignedMinMaxPe.inputs = {{"pa", "minmax_lhs", "!fabric.bits<32>", ""},
                             {"pb", "minmax_rhs", "!fabric.bits<32>", ""}};
  unsignedMinMaxPe.resultNames = {"unsigned_minmax"};
  unsignedMinMaxPe.resultTypes = {"!fabric.bits<32>"};
  unsignedMinMaxPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"selected"},
                           {"llvm.intr.umax"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"selected"}});
  module.addPe(std::move(unsignedMinMaxPe));

  PeSpec unsignedMinPe;
  unsignedMinPe.inputs = {{"pa", "minmax_lhs", "!fabric.bits<32>", ""},
                          {"pb", "minmax_rhs", "!fabric.bits<32>", ""}};
  unsignedMinPe.resultNames = {"unsigned_min"};
  unsignedMinPe.resultTypes = {"!fabric.bits<32>"};
  unsignedMinPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"selected"},
                           {"llvm.intr.umin"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"selected"}});
  module.addPe(std::move(unsignedMinPe));

  PeSpec signedMinPe;
  signedMinPe.inputs = {{"pa", "minmax_lhs", "!fabric.bits<32>", ""},
                        {"pb", "minmax_rhs", "!fabric.bits<32>", ""}};
  signedMinPe.resultNames = {"signed_min"};
  signedMinPe.resultTypes = {"!fabric.bits<32>"};
  signedMinPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"selected"},
                           {"llvm.intr.smin"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"selected"}});
  module.addPe(std::move(signedMinPe));

  PeSpec signedMaxPe;
  signedMaxPe.inputs = {{"pa", "minmax_lhs", "!fabric.bits<32>", ""},
                        {"pb", "minmax_rhs", "!fabric.bits<32>", ""}};
  signedMaxPe.resultNames = {"signed_max"};
  signedMaxPe.resultTypes = {"!fabric.bits<32>"};
  signedMaxPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"selected"},
                           {"llvm.intr.smax"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"selected"}});
  module.addPe(std::move(signedMaxPe));

  auto makeUnary32YieldFu = [](std::string resultName, std::string opName) {
    std::string yieldName = resultName;
    return FuSpec{{{"value", "pa", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{std::move(resultName)},
                                {std::move(opName)},
                                {"value"},
                                {"!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {std::move(yieldName)}};
  };
  auto addUnary32YieldPe = [&](std::string resultName, std::string opName,
                               std::string inputName = "i32a") {
    std::string peResultName = resultName;
    PeSpec pe;
    pe.inputs = {{"pa", std::move(inputName), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(peResultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        makeUnary32YieldFu(std::move(resultName), std::move(opName)));
    module.addPe(std::move(pe));
  };

  PeSpec fshlPe;
  fshlPe.inputs = {{"pa", "rotate_lhs", "!fabric.bits<32>", ""},
                   {"pb", "rotate_rhs", "!fabric.bits<32>", ""},
                   {"pc", "rotate_amount", "!fabric.bits<32>", ""}};
  fshlPe.resultNames = {"rotated"};
  fshlPe.resultTypes = {"!fabric.bits<32>"};
  auto makeFshlFu = []() {
    return FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                   {"rhs", "pb", "!fabric.bits<32>", ""},
                   {"amount", "pc", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{"rotated_value"},
                                {"llvm.intr.fshl"},
                                {"lhs", "rhs", "amount"},
                                {"!fabric.bits<32>", "!fabric.bits<32>",
                                 "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {"rotated_value"}};
  };
  fshlPe.fus.push_back(makeFshlFu());
  fshlPe.fus.push_back(makeFshlFu());
  module.addPe(std::move(fshlPe));

  addUnary32YieldPe("abs", "llvm.intr.abs");
  addUnary32YieldPe("swapped", "llvm.intr.bswap");
  addUnary32YieldPe("leading_zero_count", "llvm.intr.ctlz");

  auto addCastBankPe = [&]() {
    constexpr unsigned kCastLanes = 4;
    const char *ports[] = {"pa", "pb", "pc", "pd"};
    PeSpec pe;
    FuSpec fu;
    for (unsigned i = 0; i < kCastLanes; ++i) {
      std::string index = std::to_string(i);
      std::string value = "value" + index;
      std::string converted = "converted" + index;
      pe.inputs.push_back(
          {ports[i], "cast" + index + "_input", "!fabric.bits<32>", ""});
      pe.resultNames.push_back("cast" + index + "_result");
      pe.resultTypes.push_back("!fabric.bits<32>");
      fu.inputs.push_back({value, ports[i], "!fabric.bits<32>", ""});
      fu.resultTypes.push_back("!fabric.bits<32>");
      fu.operations.push_back(
          FabricOpSpec{{converted},
                       {"llvm.trunc", "llvm.sext", "llvm.zext"},
                       {value},
                       {"!fabric.bits<32>"},
                       {"!fabric.bits<32>"},
                       {},
                       {}});
      fu.yieldValues.push_back(std::move(converted));
    }
    pe.fus.push_back(std::move(fu));
    module.addPe(std::move(pe));
  };
  addCastBankPe();
  addUnary32YieldPe("int_extui", "arith.extui", "int_extui_input");

  auto addWideExtensionPe = [&](std::string resultName, std::string inputName) {
    PeSpec pe;
    pe.inputs = {
        {"pa", std::move(inputName), "!fabric.bits<32>", "!fabric.bits<64>"}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<64>"};
    pe.fus.push_back(
        FuSpec{{{"value", "pa", "!fabric.bits<64>", "!fabric.bits<32>"}},
               {"!fabric.bits<64>"},
               {FabricOpSpec{{"wide"},
                             {"llvm.sext", "llvm.zext"},
                             {"value"},
                             {"!fabric.bits<32>"},
                             {"!fabric.bits<64>"},
                             {},
                             {}}},
               {"wide"}});
    module.addPe(std::move(pe));
  };
  addWideExtensionPe("wide_zext0", "wide_zext0_input");
  addWideExtensionPe("wide_zext1", "wide_zext1_input");

  auto addWideBinaryPe = [&](std::string peResultName, std::string lhsInput,
                             std::string rhsInput,
                             std::vector<std::string> opList) {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(lhsInput), "!fabric.bits<64>", ""},
                 {"pb", std::move(rhsInput), "!fabric.bits<64>", ""}};
    pe.resultNames = {std::move(peResultName)};
    pe.resultTypes = {"!fabric.bits<64>"};
    pe.fus.push_back(
        FuSpec{{{"lhs", "pa", "!fabric.bits<64>", ""},
                {"rhs", "pb", "!fabric.bits<64>", ""}},
               {"!fabric.bits<64>"},
               {FabricOpSpec{{"value"},
                             std::move(opList),
                             {"lhs", "rhs"},
                             {"!fabric.bits<64>", "!fabric.bits<64>"},
                             {"!fabric.bits<64>"},
                             {},
                             {}}},
               {"value"}});
    module.addPe(std::move(pe));
  };
  addWideBinaryPe("wide_product", "wide_mul_lhs", "wide_mul_rhs",
                  {"arith.muli"});
  addWideBinaryPe("wide_signed_quotient", "wide_div_lhs", "wide_div_rhs",
                  {"arith.divsi"});
  addWideBinaryPe("wide_remainder", "wide_rem_lhs", "wide_rem_rhs",
                  {"arith.divui", "arith.remui"});
  addWideBinaryPe("wide_sum", "wide_add_lhs", "wide_add_rhs",
                  {"arith.addi", "arith.subi"});
  addWideBinaryPe("wide_sum_aux", "wide_add_aux_lhs", "wide_add_aux_rhs",
                  {"arith.addi", "arith.subi"});
  addWideBinaryPe("wide_shifted", "wide_shift_lhs", "wide_shift_rhs",
                  {"arith.shli"});

  auto addWideTruncPe = [&](std::string resultName, std::string inputName) {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(inputName), "!fabric.bits<64>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<64>"};
    pe.fus.push_back(FuSpec{{{"value", "pa", "!fabric.bits<64>", ""}},
                            {"!fabric.bits<64>"},
                            {FabricOpSpec{{"narrow"},
                                          {"llvm.trunc"},
                                          {"value"},
                                          {"!fabric.bits<64>"},
                                          {"!fabric.bits<32>"},
                                          {},
                                          {}}},
                            {"narrow"},
                            {"!fabric.bits<32>"}});
    module.addPe(std::move(pe));
  };
  addWideTruncPe("wide_truncated_wide", "wide_trunc_input");
  addWideTruncPe("wide_truncated_aux_wide", "wide_trunc_aux_input");
  addWideNarrowingPe(module, "wide_index_cast0", "wide_index_cast0_input",
                     "arith.index_cast");
  addWideNarrowingPe(module, "wide_index_cast1", "wide_index_cast1_input",
                     "arith.index_cast");
  addFifo(module, "wide_truncated", "wide_truncated_wide", "!fabric.bits<64>",
          "!fabric.bits<32>", 1, true, true);
  addFifo(module, "wide_truncated_aux", "wide_truncated_aux_wide",
          "!fabric.bits<64>", "!fabric.bits<32>", 1, true, true);
  addFifo(module, "wide_index_cast0_narrow", "wide_index_cast0",
          "!fabric.bits<64>", "!fabric.bits<32>", 1, true, true);
  addFifo(module, "wide_index_cast1_narrow", "wide_index_cast1",
          "!fabric.bits<64>", "!fabric.bits<32>", 1, true, true);

  addUnary32YieldPe("fp", "llvm.uitofp");
  addUnary32YieldPe("fp_negated", "llvm.fneg", "fp_negated_input");

  auto addCmpPe = [&](std::string resultName, std::vector<std::string> opNames,
                      std::vector<std::string> predicates) {
    PeSpec pe;
    pe.inputs = {{"pa", "cmp_lhs", "!fabric.bits<32>", ""},
                 {"pb", "cmp_rhs", "!fabric.bits<32>", ""}};
    pe.resultNames = {resultName};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                {"rhs", "pb", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>"},
               {FabricOpSpec{{"pred"},
                             std::move(opNames),
                             {"lhs", "rhs"},
                             {"!fabric.bits<32>", "!fabric.bits<32>"},
                             {"!fabric.bits<1>"},
                             {{"predicate", std::move(predicates)}},
                             {}}},
               {"pred"},
               {"!fabric.bits<1>"}});
    module.addPe(std::move(pe));
  };
  addCmpPe("cmpf_pred", {"arith.cmpf"}, {"oeq", "ogt", "ugt", "ule", "olt"});
  std::vector<std::string> integerCmpPredicates = {
      "eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge"};
  addCmpPe("cmpi_pred", {"arith.cmpi", "llvm.icmp"}, integerCmpPredicates);
  addCmpPe("cmpi_pred_aux", {"arith.cmpi", "llvm.icmp"},
           std::move(integerCmpPredicates));

  auto addWideCmpPe = [&](std::string resultName, std::string resultType) {
    PeSpec pe;
    pe.inputs = {{"pa", "cmp64_lhs", "!fabric.bits<64>", ""},
                 {"pb", "cmp64_rhs", "!fabric.bits<64>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {resultType};
    pe.fus.push_back(
        FuSpec{{{"lhs", "pa", "!fabric.bits<64>", ""},
                {"rhs", "pb", "!fabric.bits<64>", ""}},
               {resultType},
               {FabricOpSpec{{"pred"},
                             {"arith.cmpi"},
                             {"lhs", "rhs"},
                             {"!fabric.bits<64>", "!fabric.bits<64>"},
                             {"!fabric.bits<1>"},
                             {{"predicate",
                               {"eq", "ne", "slt", "sle", "sgt", "sge", "ult",
                                "ule", "ugt", "uge"}}},
                             {}}},
               {"pred"},
               {"!fabric.bits<1>"}});
    module.addPe(std::move(pe));
  };
  addWideCmpPe("cmpi64_pred", "!fabric.bits<64>");
  addWideCmpPe("cmpi64_pred_aux", "!fabric.bits<64>");
  addFifo(module, "cmpi64_pred_aux_narrow", "cmpi64_pred_aux",
          "!fabric.bits<64>", "!fabric.bits<32>", 1, true, true);

  PeSpec widePredExtuiPe;
  widePredExtuiPe.inputs = {{"pa", "cmpi64_pred", "!fabric.bits<64>", ""}};
  widePredExtuiPe.resultNames = {"wide_pred_extui"};
  widePredExtuiPe.resultTypes = {"!fabric.bits<64>"};
  widePredExtuiPe.fus.push_back(
      FuSpec{{{"value", "pa", "!fabric.bits<64>", "!fabric.bits<1>"}},
             {"!fabric.bits<64>"},
             {FabricOpSpec{{"extended"},
                           {"arith.extui"},
                           {"value"},
                           {"!fabric.bits<1>"},
                           {"!fabric.bits<64>"},
                           {},
                           {}}},
             {"extended"}});
  module.addPe(std::move(widePredExtuiPe));

  PeSpec selectPe;
  selectPe.inputs = {{"pa", "select_pred", "!fabric.bits<32>", ""},
                     {"pb", "select_true", "!fabric.bits<32>", ""},
                     {"pc", "select_false", "!fabric.bits<32>", ""}};
  selectPe.resultNames = {"selected"};
  selectPe.resultTypes = {"!fabric.bits<32>"};
  auto makeSelectFu = [](llvm::StringRef opName) {
    return FuSpec{{{"sel", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                   {"when_true", "pb", "!fabric.bits<32>", ""},
                   {"when_false", "pc", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{"selected_value"},
                                {opName.str()},
                                {"sel", "when_true", "when_false"},
                                {"!fabric.bits<1>", "!fabric.bits<32>",
                                 "!fabric.bits<32>"},
                                {"!fabric.bits<32>"},
                                {},
                                {}}},
                  {"selected_value"}};
  };
  selectPe.fus.push_back(makeSelectFu("arith.select"));
  selectPe.fus.push_back(makeSelectFu("arith.select"));
  selectPe.fus.push_back(makeSelectFu("llvm.select"));
  module.addPe(std::move(selectPe));

  auto addDemuxPe = [&](llvm::StringRef valueInput, llvm::StringRef falseResult,
                        llvm::StringRef trueResult) {
    PeSpec demuxPe;
    demuxPe.inputs = {{"pa", "demux_sel", "!fabric.bits<32>", ""},
                      {"pb", valueInput.str(), "!fabric.bits<32>", ""}};
    demuxPe.resultNames = {falseResult.str(), trueResult.str()};
    demuxPe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
    demuxPe.fus.push_back(
        FuSpec{{{"sel", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
                {"value", "pb", "!fabric.bits<32>", ""}},
               {"!fabric.bits<32>", "!fabric.bits<32>"},
               {FabricOpSpec{{"false_lane", "true_lane"},
                             {"dataflow.demux"},
                             {"sel", "value"},
                             {"!fabric.bits<1>", "!fabric.bits<32>"},
                             {"!fabric.bits<32>", "!fabric.bits<32>"},
                             {},
                             {}}},
               {"false_lane", "true_lane"}});
    module.addPe(std::move(demuxPe));
  };
  addDemuxPe("demux_value", "control_demux_false", "control_demux_true");
  addDemuxPe("demux_then_value", "compute_demux_false", "compute_demux_true");

  PeSpec muxPe;
  muxPe.inputs = {{"pa", "mux_sel", "!fabric.bits<32>", ""},
                  {"pb", "mux_false", "!fabric.bits<32>", ""},
                  {"pc", "mux_true", "!fabric.bits<32>", ""}};
  muxPe.resultNames = {"control_muxed"};
  muxPe.resultTypes = {"!fabric.bits<32>"};
  muxPe.fus.push_back(FuSpec{
      {{"sel", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
       {"false_lane", "pb", "!fabric.bits<32>", ""},
       {"true_lane", "pc", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"selected_lane"},
                    {"dataflow.mux"},
                    {"sel", "false_lane", "true_lane"},
                    {"!fabric.bits<1>", "!fabric.bits<32>", "!fabric.bits<32>"},
                    {"!fabric.bits<32>"},
                    {},
                    {}}},
      {"selected_lane"}});
  module.addPe(std::move(muxPe));

  PeSpec controlDemuxPe;
  controlDemuxPe.inputs = {
      {"pa", "control_token_demux_sel", "!fabric.bits<32>", ""},
      {"pb", "ctrl", "!fabric.bits<0>", "!fabric.bits<32>"}};
  controlDemuxPe.resultNames = {"control_token_demux_false",
                                "control_token_demux_true"};
  controlDemuxPe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
  controlDemuxPe.fus.push_back(
      FuSpec{{{"sel", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
              {"value", "pb", "!fabric.bits<32>", "!fabric.bits<0>"}},
             {"!fabric.bits<32>", "!fabric.bits<32>"},
             {FabricOpSpec{{"false_lane", "true_lane"},
                           {"dataflow.demux"},
                           {"sel", "value"},
                           {"!fabric.bits<1>", "!fabric.bits<0>"},
                           {"!fabric.bits<0>", "!fabric.bits<0>"},
                           {},
                           {}}},
             {"false_lane", "true_lane"},
             {"!fabric.bits<0>", "!fabric.bits<0>"}});
  module.addPe(std::move(controlDemuxPe));

  PeSpec controlMuxPe;
  controlMuxPe.inputs = {
      {"pa", "control_token_mux_sel", "!fabric.bits<32>", ""},
      {"pb", "control_token_mux_false", "!fabric.bits<0>", "!fabric.bits<32>"},
      {"pc", "control_token_mux_true", "!fabric.bits<0>", "!fabric.bits<32>"}};
  controlMuxPe.resultNames = {"control_token_muxed"};
  controlMuxPe.resultTypes = {"!fabric.bits<32>"};
  controlMuxPe.fus.push_back(FuSpec{
      {{"sel", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
       {"false_lane", "pb", "!fabric.bits<32>", "!fabric.bits<0>"},
       {"true_lane", "pc", "!fabric.bits<32>", "!fabric.bits<0>"}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"selected_lane"},
                    {"dataflow.mux"},
                    {"sel", "false_lane", "true_lane"},
                    {"!fabric.bits<1>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {"!fabric.bits<0>"},
                    {},
                    {}}},
      {"selected_lane"},
      {"!fabric.bits<0>"}});
  module.addPe(std::move(controlMuxPe));

  PeSpec vectorSyncPe;
  vectorSyncPe.inputs = {{"pa", "sync_head", "!fabric.bits<0>", ""},
                         {"pb", "vector_sync_mid", "!fabric.bits<0>", ""},
                         {"pc", "sync_tail", "!fabric.bits<0>", ""},
                         {"pd", "sync_extra", "!fabric.bits<0>", ""},
                         {"pe", "done4", "!fabric.bits<0>", ""},
                         {"pf", "sync_lane5", "!fabric.bits<0>", ""},
                         {"pg", "store_done0", "!fabric.bits<0>", ""},
                         {"ph", "sync_lane6", "!fabric.bits<0>", ""},
                         {"pi", "sync_lane7", "!fabric.bits<0>", ""}};
  vectorSyncPe.resultTypes = {"!fabric.bits<0>"};
  vectorSyncPe.fus.push_back(FuSpec{
      {{"fa", "pa", "!fabric.bits<0>", ""},
       {"fb", "pb", "!fabric.bits<0>", ""},
       {"fc", "pc", "!fabric.bits<0>", ""},
       {"fd", "pd", "!fabric.bits<0>", ""},
       {"fe", "pe", "!fabric.bits<0>", ""},
       {"ff", "pf", "!fabric.bits<0>", ""},
       {"fg", "pg", "!fabric.bits<0>", ""},
       {"fh", "ph", "!fabric.bits<0>", ""},
       {"fi", "pi", "!fabric.bits<0>", ""}},
      {"!fabric.bits<0>"},
      {FabricOpSpec{{"sync_done0", "sync_done1", "sync_done2", "sync_done3",
                     "sync_done4", "sync_done5", "sync_done6", "sync_done7",
                     "sync_done8"},
                    {"dataflow.sync"},
                    {"fa", "fb", "fc", "fd", "fe", "ff", "fg", "fh", "fi"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>",
                     "!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>",
                     "!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>",
                     "!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>",
                     "!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {},
                    {{"bitmask", "111111111"}}}},
      {"sync_done0"}});
  module.addPe(std::move(vectorSyncPe));

  PeSpec syncPe;
  syncPe.inputs = {{"pc", "done0", "!fabric.bits<0>", ""},
                   {"pd", "sync_aux_done", "!fabric.bits<0>", ""}};
  syncPe.resultTypes = {"!fabric.bits<0>"};
  syncPe.fus.push_back(
      FuSpec{{{"fc", "pc", "!fabric.bits<0>", ""},
              {"fd", "pd", "!fabric.bits<0>", ""}},
             {"!fabric.bits<0>"},
             {FabricOpSpec{{"sync_done0", "sync_done1"},
                           {"dataflow.sync"},
                           {"fc", "fd"},
                           {"!fabric.bits<0>", "!fabric.bits<0>"},
                           {"!fabric.bits<0>", "!fabric.bits<0>"},
                           {},
                           {{"bitmask", "11"}}}},
             {"sync_done0"}});
  module.addPe(std::move(syncPe));

  PeSpec addrAddPe;
  addrAddPe.inputs = {{"pa", "addr_add_lhs", "!fabric.bits<32>", ""},
                      {"pb", "addr_add_rhs", "!fabric.bits<32>", ""}};
  addrAddPe.resultNames = {"addr_sum"};
  addrAddPe.resultTypes = {"!fabric.bits<32>"};
  addrAddPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"sum"},
                           {"arith.addi", "arith.subi"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"sum"}});
  module.addPe(std::move(addrAddPe));

  PeSpec addrMaskPe;
  addrMaskPe.inputs = {{"pa", "addr_mask_lhs", "!fabric.bits<32>", ""},
                       {"pb", "addr_mask_rhs", "!fabric.bits<32>", ""}};
  addrMaskPe.resultNames = {"addr_masked"};
  addrMaskPe.resultTypes = {"!fabric.bits<32>"};
  addrMaskPe.fus.push_back(
      FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
              {"rhs", "pb", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"masked"},
                           {"arith.andi"},
                           {"lhs", "rhs"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"masked"}});
  module.addPe(std::move(addrMaskPe));

  addBinary32Pe("aux_masked", "aux_mask_lhs", "aux_mask_rhs", "masked",
                {"arith.andi"});
  addBinary32Pe("aux_xor", "aux_xor_lhs", "aux_xor_rhs", "xor_value",
                {"arith.xori"});

  addFmulAddPe("mac_result2", "mac2_lhs", "mac2_rhs", "mac2_acc");
  addFmulAddPe("mac_result3", "mac3_lhs", "mac3_rhs", "mac3_acc");

  PeSpec stateCarryPe;
  stateCarryPe.inputs = {{"pa", "state_carry_cond", "!fabric.bits<32>", ""},
                         {"pb", "state_carry_init", "!fabric.bits<32>", ""},
                         {"pc", "state_carry_next", "!fabric.bits<32>", ""}};
  stateCarryPe.resultNames = {"state_carry"};
  stateCarryPe.resultTypes = {"!fabric.bits<32>"};
  stateCarryPe.fus.push_back(makeCarryFu());
  stateCarryPe.fus.push_back(makeCarryFu());
  module.addPe(std::move(stateCarryPe));

  addFpBinaryPe("scaled_reduction_aux", "scaled_reduction_aux_lhs",
                "scaled_reduction_aux_rhs", "product", "arith.mulf");
}

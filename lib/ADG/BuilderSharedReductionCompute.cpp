#include "BuilderInternal.h"

#include "Dataflow/IR/DataflowEnums.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

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
                   loopStreamCapability(dataflow::StreamStepKind::Add,
                                        {mlir::arith::CmpIPredicate::slt,
                                         mlir::arith::CmpIPredicate::sgt}),
                   {::dataflow::OperationSchemaId::DataflowStream},
                   {"fa", "fb", "fc"},
                   {"!fabric.bits<32>", "!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>", "!fabric.bits<1>"}});
  streamFu.operations.push_back(FabricOpSpec{
      {"carried"},
      builtinOpCapability(::fabric::ImplementationFamilyId::LoopCarry),
      {::dataflow::OperationSchemaId::DataflowCarry},
      {"rwc", "init", "next"},
      {"!fabric.bits<1>", "!fabric.bits<32>", "!fabric.bits<32>"},
      {"!fabric.bits<32>"}});
  streamFu.operations.push_back(
      FabricOpSpec{{"sum"},
                   builtinOpCapability(
                       ::fabric::ImplementationFamilyId::ScalarIntegerAddSub),
                   {::dataflow::OperationSchemaId::ArithAddI},
                   {"sum_lhs", "sum_rhs"},
                   {"!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"}});
  streamFu.operations.push_back(FabricOpSpec{
      {"stable_scale"},
      builtinOpCapability(::fabric::ImplementationFamilyId::LoopInvariant),
      {::dataflow::OperationSchemaId::DataflowInvariant},
      {"rwc", "scale"},
      {"!fabric.bits<1>", "!fabric.bits<32>"},
      {"!fabric.bits<32>"}});
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
                   loopStreamCapability(dataflow::StreamStepKind::Add,
                                        {mlir::arith::CmpIPredicate::slt,
                                         mlir::arith::CmpIPredicate::sgt}),
                   {::dataflow::OperationSchemaId::DataflowStream},
                   {"fa", "fb", "fc"},
                   {"!fabric.bits<32>", "!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>", "!fabric.bits<1>"}});
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
    fu.operations.push_back(FabricOpSpec{
        {"after_cond", "after_value"},
        builtinOpCapability(::fabric::ImplementationFamilyId::LoopGate),
        {::dataflow::OperationSchemaId::DataflowGate},
        {"cond", "value"},
        {"!fabric.bits<1>", "!fabric.bits<32>"},
        {"!fabric.bits<1>", "!fabric.bits<32>"}});
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
  // Integer absolute value is not a catalog resource: the initial catalog
  // lowers it through the ordinary compare, select, and subtract resources
  // this same catalog already declares rather than advertising an unproven
  // family. The floating-point form keeps its ScalarFloatSign resource.
  absPe.fus.push_back(FuSpec{
      {{"value", "pa", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"abs"},
                    builtinOpCapability(
                        ::fabric::ImplementationFamilyId::ScalarFloatSign),
                    {::dataflow::OperationSchemaId::MathAbsF},
                    {"value"},
                    {"!fabric.bits<32>"},
                    {"!fabric.bits<32>"}}},
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
             {FabricOpSpec{
                 {"product"},
                 builtinOpCapability(
                     ::fabric::ImplementationFamilyId::ScalarIntegerMultiply),
                 {::dataflow::OperationSchemaId::ArithMulI},
                 {"lhs", "rhs"},
                 {"!fabric.bits<32>", "!fabric.bits<32>"},
                 {"!fabric.bits<32>"}}},
             {"product"}});
  module.addPe(std::move(squaredPe));

  auto addFpBinaryPe = [&](std::string resultName, std::string lhsInput,
                           std::string rhsInput, llvm::StringRef valueName,
                           ::fabric::ImplementationFamilyId family,
                           ::dataflow::OperationSchemaId member) {
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
                             builtinOpCapability(family),
                             {member},
                             {"lhs", "rhs"},
                             {"!fabric.bits<32>", "!fabric.bits<32>"},
                             {"!fabric.bits<32>"}}},
               {valueName.str()}});
    module.addPe(std::move(pe));
  };

  addFpBinaryPe("fp_running", "fp_lhs", "fp_rhs", "sum",
                ::fabric::ImplementationFamilyId::ScalarFloatAddSub,
                ::dataflow::OperationSchemaId::ArithAddF);
  addFpBinaryPe("fp_running_aux", "fp_lhs_aux", "fp_rhs_aux", "sum",
                ::fabric::ImplementationFamilyId::ScalarFloatAddSub,
                ::dataflow::OperationSchemaId::ArithAddF);

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
                           builtinOpCapability(
                               ::fabric::ImplementationFamilyId::LoopInvariant),
                           {::dataflow::OperationSchemaId::DataflowInvariant},
                           {"cond", "value"},
                           {"!fabric.bits<1>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"}}},
             {"stable"}});
  module.addPe(std::move(fpInvariantPe));

  auto addInvariantPe = [&](std::string resultName, std::string valueInput,
                            std::string condInput = "fp_gate") {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(condInput), "!fabric.bits<32>", ""},
                 {"pb", std::move(valueInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(FuSpec{
        {{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
         {"value", "pb", "!fabric.bits<32>", ""}},
        {"!fabric.bits<32>"},
        {FabricOpSpec{{"stable"},
                      builtinOpCapability(
                          ::fabric::ImplementationFamilyId::LoopInvariant),
                      {::dataflow::OperationSchemaId::DataflowInvariant},
                      {"cond", "value"},
                      {"!fabric.bits<1>", "!fabric.bits<32>"},
                      {"!fabric.bits<32>"}}},
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
    pe.fus.push_back(FuSpec{
        {{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
         {"value", "pb", "!fabric.bits<32>", ""}},
        {"!fabric.bits<32>"},
        {FabricOpSpec{{"stable"},
                      builtinOpCapability(
                          ::fabric::ImplementationFamilyId::LoopInvariant),
                      {::dataflow::OperationSchemaId::DataflowInvariant},
                      {"cond", "value"},
                      {"!fabric.bits<1>", "!fabric.bits<32>"},
                      {"!fabric.bits<32>"}}},
        {"stable"}});
    module.addPe(std::move(pe));
  };
  addAuxInvariantPe("aux_invariant0", "aux_invariant0_value");
  addAuxInvariantPe("aux_invariant1", "aux_invariant1_value");

  addFpBinaryPe("fp_diff", "fp_diff_lhs", "fp_diff_rhs", "diff",
                ::fabric::ImplementationFamilyId::ScalarFloatAddSub,
                ::dataflow::OperationSchemaId::ArithSubF);
  addFpBinaryPe("fp_diff_aux", "fp_diff_aux_lhs", "fp_diff_aux_rhs", "diff",
                ::fabric::ImplementationFamilyId::ScalarFloatAddSub,
                ::dataflow::OperationSchemaId::ArithSubF);

  PeSpec scaledReductionPe;
  scaledReductionPe.inputs = {
      {"pa", "scaled_reduction_lhs", "!fabric.bits<32>", ""},
      {"pb", "scaled_reduction_rhs", "!fabric.bits<32>", ""}};
  scaledReductionPe.resultNames = {"scaled_reduction"};
  scaledReductionPe.resultTypes = {"!fabric.bits<32>"};
  scaledReductionPe.fus.push_back(FuSpec{
      {{"lhs", "pa", "!fabric.bits<32>", ""},
       {"rhs", "pb", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"product"},
                    builtinOpCapability(
                        ::fabric::ImplementationFamilyId::ScalarFloatMultiply),
                    {::dataflow::OperationSchemaId::ArithMulF},
                    {"lhs", "rhs"},
                    {"!fabric.bits<32>", "!fabric.bits<32>"},
                    {"!fabric.bits<32>"}}},
      {"product"}});
  module.addPe(std::move(scaledReductionPe));

  auto makeCarryFu = []() {
    return FuSpec{
        {{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
         {"init", "pb", "!fabric.bits<32>", ""},
         {"next", "pc", "!fabric.bits<32>", ""}},
        {"!fabric.bits<32>"},
        {FabricOpSpec{
            {"carried"},
            builtinOpCapability(::fabric::ImplementationFamilyId::LoopCarry),
            {::dataflow::OperationSchemaId::DataflowCarry},
            {"cond", "init", "next"},
            {"!fabric.bits<1>", "!fabric.bits<32>", "!fabric.bits<32>"},
            {"!fabric.bits<32>"}}},
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
                           ::fabric::ImplementationFamilyId family,
                           std::vector<::dataflow::OperationSchemaId> members) {
    std::string yieldName = resultName;
    return FuSpec{{{"lhs", "pa", "!fabric.bits<32>", ""},
                   {"rhs", "pb", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{std::move(resultName)},
                                builtinOpCapability(family),
                                std::move(members),
                                {"lhs", "rhs"},
                                {"!fabric.bits<32>", "!fabric.bits<32>"},
                                {"!fabric.bits<32>"}}},
                  {std::move(yieldName)}};
  };
  auto addBinary32Pe = [&](std::string peResultName, std::string lhsInput,
                           std::string rhsInput, std::string opResultName,
                           ::fabric::ImplementationFamilyId family,
                           std::vector<::dataflow::OperationSchemaId> members) {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(lhsInput), "!fabric.bits<32>", ""},
                 {"pb", std::move(rhsInput), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(peResultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(
        makeBinary32Fu(std::move(opResultName), family, std::move(members)));
    module.addPe(std::move(pe));
  };
  addBinary32Pe("int_sum", "int_add_lhs", "int_add_rhs", "sum",
                ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                {::dataflow::OperationSchemaId::ArithAddI,
                 ::dataflow::OperationSchemaId::ArithSubI});
  addBinary32Pe("int_product", "int_mul_lhs", "int_mul_rhs", "product",
                ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
                {::dataflow::OperationSchemaId::ArithMulI});
  addBinary32Pe("int_product_aux", "int_mul_aux_lhs", "int_mul_aux_rhs",
                "product",
                ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
                {::dataflow::OperationSchemaId::ArithMulI});
  recordUnsupportedCatalogResource(module,
                                   {::dataflow::OperationSchemaId::ArithDivSI});
  recordUnsupportedCatalogResource(module,
                                   {::dataflow::OperationSchemaId::ArithDivSI});
  recordUnsupportedCatalogResource(module,
                                   {::dataflow::OperationSchemaId::ArithRemSI});
  recordUnsupportedCatalogResource(module,
                                   {::dataflow::OperationSchemaId::ArithDivUI,
                                    ::dataflow::OperationSchemaId::ArithRemUI});
  recordUnsupportedCatalogResource(module,
                                   {::dataflow::OperationSchemaId::ArithDivF,
                                    ::dataflow::OperationSchemaId::ArithRemF});
  auto addConfigurableConstPe = [&](std::string resultName) {
    (void)resultName;
    recordUnsupportedCatalogResource(
        module, {::dataflow::OperationSchemaId::DataflowConstant});
  };
  addConfigurableConstPe("addr_shift_const");
  addConfigurableConstPe("addr_aux_const");
  addConfigurableConstPe("addr_bias_const");
  addConfigurableConstPe("addr_extra_const0");
  addConfigurableConstPe("addr_extra_const1");
  addBinary32Pe("logic_shifted", "logic_shift_lhs", "logic_shift_rhs",
                "shifted", ::fabric::ImplementationFamilyId::ScalarIntegerShift,
                {::dataflow::OperationSchemaId::ArithShRSI,
                 ::dataflow::OperationSchemaId::ArithShRUI});

  PeSpec addrUnscalePe;
  addrUnscalePe.inputs = {{"pa", "addr_unscale_lhs", "!fabric.bits<32>", ""},
                          {"pb", "addr_unscale_rhs", "!fabric.bits<32>", ""}};
  addrUnscalePe.resultNames = {"addr_unscaled"};
  addrUnscalePe.resultTypes = {"!fabric.bits<32>"};
  addrUnscalePe.fus.push_back(FuSpec{
      {{"lhs", "pa", "!fabric.bits<32>", ""},
       {"rhs", "pb", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"shifted"},
                    builtinOpCapability(
                        ::fabric::ImplementationFamilyId::ScalarIntegerShift),
                    {::dataflow::OperationSchemaId::ArithShRSI,
                     ::dataflow::OperationSchemaId::ArithShRUI},
                    {"lhs", "rhs"},
                    {"!fabric.bits<32>", "!fabric.bits<32>"},
                    {"!fabric.bits<32>"}}},
      {"shifted"}});
  module.addPe(std::move(addrUnscalePe));
  PeSpec addrShiftPe;
  addrShiftPe.inputs = {{"pa", "addr_shift_lhs", "!fabric.bits<32>", ""},
                        {"pb", "addr_shift_rhs", "!fabric.bits<32>", ""}};
  addrShiftPe.resultNames = {"addr_shifted"};
  addrShiftPe.resultTypes = {"!fabric.bits<32>"};
  addrShiftPe.fus.push_back(FuSpec{
      {{"lhs", "pa", "!fabric.bits<32>", ""},
       {"rhs", "pb", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"shifted"},
                    builtinOpCapability(
                        ::fabric::ImplementationFamilyId::ScalarIntegerShift),
                    {::dataflow::OperationSchemaId::ArithShLI},
                    {"lhs", "rhs"},
                    {"!fabric.bits<32>", "!fabric.bits<32>"},
                    {"!fabric.bits<32>"}}},
      {"shifted"}});
  module.addPe(std::move(addrShiftPe));
  PeSpec logicMaskPe;
  logicMaskPe.inputs = {{"pa", "logic_mask_lhs", "!fabric.bits<32>", ""},
                        {"pb", "logic_mask_rhs", "!fabric.bits<32>", ""}};
  logicMaskPe.resultNames = {"logic_masked"};
  logicMaskPe.resultTypes = {"!fabric.bits<32>"};
  logicMaskPe.fus.push_back(FuSpec{
      {{"lhs", "pa", "!fabric.bits<32>", ""},
       {"rhs", "pb", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"masked"},
                    builtinOpCapability(
                        ::fabric::ImplementationFamilyId::ScalarIntegerLogic),
                    {::dataflow::OperationSchemaId::ArithAndI},
                    {"lhs", "rhs"},
                    {"!fabric.bits<32>", "!fabric.bits<32>"},
                    {"!fabric.bits<32>"}}},
      {"masked"}});
  module.addPe(std::move(logicMaskPe));
  addBinary32Pe("int_or", "int_or_lhs", "int_or_rhs", "combined",
                ::fabric::ImplementationFamilyId::ScalarIntegerLogic,
                {::dataflow::OperationSchemaId::ArithOrI});
  addBinary32Pe("int_xor", "int_xor_lhs", "int_xor_rhs", "combined",
                ::fabric::ImplementationFamilyId::ScalarIntegerLogic,
                {::dataflow::OperationSchemaId::ArithXOrI});
  auto addFusedMultiplyAddPe = [&](std::string resultName, std::string lhsInput,
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
                   builtinOpCapability(
                       ::fabric::ImplementationFamilyId::ScalarFloatFma),
                   {::dataflow::OperationSchemaId::MathFma},
                   {"lhs", "rhs", "acc"},
                   {"!fabric.bits<32>", "!fabric.bits<32>", "!fabric.bits<32>"},
                   {"!fabric.bits<32>"}}},
               {"mac"}});
    module.addPe(std::move(pe));
  };
  addFusedMultiplyAddPe("mac_result", "mac_lhs", "mac_rhs", "mac_acc");
  addFusedMultiplyAddPe("mac_result1", "mac1_lhs", "mac1_rhs", "mac1_acc");

  PeSpec unsignedMinMaxPe;
  unsignedMinMaxPe.inputs = {{"pa", "minmax_lhs", "!fabric.bits<32>", ""},
                             {"pb", "minmax_rhs", "!fabric.bits<32>", ""}};
  unsignedMinMaxPe.resultNames = {"unsigned_minmax"};
  unsignedMinMaxPe.resultTypes = {"!fabric.bits<32>"};
  unsignedMinMaxPe.fus.push_back(FuSpec{
      {{"lhs", "pa", "!fabric.bits<32>", ""},
       {"rhs", "pb", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{
          {"selected"},
          builtinOpCapability(
              ::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax),
          {::dataflow::OperationSchemaId::ArithMaxUI},
          {"lhs", "rhs"},
          {"!fabric.bits<32>", "!fabric.bits<32>"},
          {"!fabric.bits<32>"}}},
      {"selected"}});
  module.addPe(std::move(unsignedMinMaxPe));

  PeSpec unsignedMinPe;
  unsignedMinPe.inputs = {{"pa", "minmax_lhs", "!fabric.bits<32>", ""},
                          {"pb", "minmax_rhs", "!fabric.bits<32>", ""}};
  unsignedMinPe.resultNames = {"unsigned_min"};
  unsignedMinPe.resultTypes = {"!fabric.bits<32>"};
  unsignedMinPe.fus.push_back(FuSpec{
      {{"lhs", "pa", "!fabric.bits<32>", ""},
       {"rhs", "pb", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{
          {"selected"},
          builtinOpCapability(
              ::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax),
          {::dataflow::OperationSchemaId::ArithMinUI},
          {"lhs", "rhs"},
          {"!fabric.bits<32>", "!fabric.bits<32>"},
          {"!fabric.bits<32>"}}},
      {"selected"}});
  module.addPe(std::move(unsignedMinPe));

  PeSpec signedMinPe;
  signedMinPe.inputs = {{"pa", "minmax_lhs", "!fabric.bits<32>", ""},
                        {"pb", "minmax_rhs", "!fabric.bits<32>", ""}};
  signedMinPe.resultNames = {"signed_min"};
  signedMinPe.resultTypes = {"!fabric.bits<32>"};
  signedMinPe.fus.push_back(FuSpec{
      {{"lhs", "pa", "!fabric.bits<32>", ""},
       {"rhs", "pb", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{
          {"selected"},
          builtinOpCapability(
              ::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax),
          {::dataflow::OperationSchemaId::ArithMinSI},
          {"lhs", "rhs"},
          {"!fabric.bits<32>", "!fabric.bits<32>"},
          {"!fabric.bits<32>"}}},
      {"selected"}});
  module.addPe(std::move(signedMinPe));

  PeSpec signedMaxPe;
  signedMaxPe.inputs = {{"pa", "minmax_lhs", "!fabric.bits<32>", ""},
                        {"pb", "minmax_rhs", "!fabric.bits<32>", ""}};
  signedMaxPe.resultNames = {"signed_max"};
  signedMaxPe.resultTypes = {"!fabric.bits<32>"};
  signedMaxPe.fus.push_back(FuSpec{
      {{"lhs", "pa", "!fabric.bits<32>", ""},
       {"rhs", "pb", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{
          {"selected"},
          builtinOpCapability(
              ::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax),
          {::dataflow::OperationSchemaId::ArithMaxSI},
          {"lhs", "rhs"},
          {"!fabric.bits<32>", "!fabric.bits<32>"},
          {"!fabric.bits<32>"}}},
      {"selected"}});
  module.addPe(std::move(signedMaxPe));

  auto makeUnary32YieldFu = [](std::string resultName,
                               ::fabric::ImplementationFamilyId family,
                               ::dataflow::OperationSchemaId member) {
    std::string yieldName = resultName;
    return FuSpec{{{"value", "pa", "!fabric.bits<32>", ""}},
                  {"!fabric.bits<32>"},
                  {FabricOpSpec{{std::move(resultName)},
                                builtinOpCapability(family),
                                {member},
                                {"value"},
                                {"!fabric.bits<32>"},
                                {"!fabric.bits<32>"}}},
                  {std::move(yieldName)}};
  };
  auto addUnary32YieldPe = [&](std::string resultName,
                               ::fabric::ImplementationFamilyId family,
                               ::dataflow::OperationSchemaId member,
                               std::string inputName = "i32a") {
    std::string peResultName = resultName;
    PeSpec pe;
    pe.inputs = {{"pa", std::move(inputName), "!fabric.bits<32>", ""}};
    pe.resultNames = {std::move(peResultName)};
    pe.resultTypes = {"!fabric.bits<32>"};
    pe.fus.push_back(makeUnary32YieldFu(std::move(resultName), family, member));
    module.addPe(std::move(pe));
  };

  // A registered schema without an implementation family is reported rather
  // than emitted.
  recordUnsupportedCatalogResource(module,
                                   {::dataflow::OperationSchemaId::LLVMFshl});

  recordUnsupportedCatalogResource(
      module, {::dataflow::OperationSchemaId::LLVMByteSwap,
               ::dataflow::OperationSchemaId::LLVMCountLeadingZeros});

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
                       builtinOpCapability(
                           ::fabric::ImplementationFamilyId::ScalarIntegerCast),
                       {::dataflow::OperationSchemaId::ArithTruncI,
                        ::dataflow::OperationSchemaId::ArithExtSI,
                        ::dataflow::OperationSchemaId::ArithExtUI},
                       {value},
                       {"!fabric.bits<32>"},
                       {"!fabric.bits<32>"}});
      fu.yieldValues.push_back(std::move(converted));
    }
    pe.fus.push_back(std::move(fu));
    module.addPe(std::move(pe));
  };
  addCastBankPe();
  addUnary32YieldPe(
      "int_extui", ::fabric::ImplementationFamilyId::ScalarIntegerCast,
      ::dataflow::OperationSchemaId::ArithExtUI, "int_extui_input");

  auto addWideExtensionPe = [&](std::string resultName, std::string inputName) {
    PeSpec pe;
    pe.inputs = {
        {"pa", std::move(inputName), "!fabric.bits<32>", "!fabric.bits<64>"}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<64>"};
    pe.fus.push_back(FuSpec{
        {{"value", "pa", "!fabric.bits<64>", "!fabric.bits<32>"}},
        {"!fabric.bits<64>"},
        {FabricOpSpec{{"wide"},
                      builtinOpCapability(
                          ::fabric::ImplementationFamilyId::ScalarIntegerCast),
                      {::dataflow::OperationSchemaId::ArithExtSI,
                       ::dataflow::OperationSchemaId::ArithExtUI},
                      {"value"},
                      {"!fabric.bits<32>"},
                      {"!fabric.bits<64>"}}},
        {"wide"}});
    module.addPe(std::move(pe));
  };
  addWideExtensionPe("wide_zext0", "wide_zext0_input");
  addWideExtensionPe("wide_zext1", "wide_zext1_input");

  auto addWideBinaryPe =
      [&](std::string peResultName, std::string lhsInput, std::string rhsInput,
          ::fabric::ImplementationFamilyId family,
          std::vector<::dataflow::OperationSchemaId> members) {
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
                                 builtinOpCapability(family),
                                 std::move(members),
                                 {"lhs", "rhs"},
                                 {"!fabric.bits<64>", "!fabric.bits<64>"},
                                 {"!fabric.bits<64>"}}},
                   {"value"}});
        module.addPe(std::move(pe));
      };
  addWideBinaryPe("wide_product", "wide_mul_lhs", "wide_mul_rhs",
                  ::fabric::ImplementationFamilyId::ScalarIntegerMultiply,
                  {::dataflow::OperationSchemaId::ArithMulI});
  recordUnsupportedCatalogResource(module,
                                   {::dataflow::OperationSchemaId::ArithDivSI});
  recordUnsupportedCatalogResource(module,
                                   {::dataflow::OperationSchemaId::ArithDivUI,
                                    ::dataflow::OperationSchemaId::ArithRemUI});
  addWideBinaryPe("wide_sum", "wide_add_lhs", "wide_add_rhs",
                  ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                  {::dataflow::OperationSchemaId::ArithAddI,
                   ::dataflow::OperationSchemaId::ArithSubI});
  addWideBinaryPe("wide_sum_aux", "wide_add_aux_lhs", "wide_add_aux_rhs",
                  ::fabric::ImplementationFamilyId::ScalarIntegerAddSub,
                  {::dataflow::OperationSchemaId::ArithAddI,
                   ::dataflow::OperationSchemaId::ArithSubI});
  addWideBinaryPe("wide_shifted", "wide_shift_lhs", "wide_shift_rhs",
                  ::fabric::ImplementationFamilyId::ScalarIntegerShift,
                  {::dataflow::OperationSchemaId::ArithShLI});

  auto addWideTruncPe = [&](std::string resultName, std::string inputName) {
    PeSpec pe;
    pe.inputs = {{"pa", std::move(inputName), "!fabric.bits<64>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {"!fabric.bits<64>"};
    pe.fus.push_back(FuSpec{
        {{"value", "pa", "!fabric.bits<64>", ""}},
        {"!fabric.bits<64>"},
        {FabricOpSpec{{"narrow"},
                      builtinOpCapability(
                          ::fabric::ImplementationFamilyId::ScalarIntegerCast),
                      {::dataflow::OperationSchemaId::ArithTruncI},
                      {"value"},
                      {"!fabric.bits<64>"},
                      {"!fabric.bits<32>"}}},
        {"narrow"},
        {"!fabric.bits<32>"}});
    module.addPe(std::move(pe));
  };
  addWideTruncPe("wide_truncated_wide", "wide_trunc_input");
  addWideTruncPe("wide_truncated_aux_wide", "wide_trunc_aux_input");
  addWideNarrowingPe(module, "wide_index_cast0", "wide_index_cast0_input",
                     ::fabric::ImplementationFamilyId::ScalarIntegerCast,
                     ::dataflow::OperationSchemaId::ArithIndexCast);
  addWideNarrowingPe(module, "wide_index_cast1", "wide_index_cast1_input",
                     ::fabric::ImplementationFamilyId::ScalarIntegerCast,
                     ::dataflow::OperationSchemaId::ArithIndexCast);
  module.addFifo(FifoSpec{"wide_truncated", "wide_truncated_wide",
                          "!fabric.bits<32>", 1, true, true});
  module.addFifo(FifoSpec{"wide_truncated_aux", "wide_truncated_aux_wide",
                          "!fabric.bits<32>", 1, true, true});
  module.addFifo(FifoSpec{"wide_index_cast0_narrow", "wide_index_cast0",
                          "!fabric.bits<32>", 1, true, true});
  module.addFifo(FifoSpec{"wide_index_cast1_narrow", "wide_index_cast1",
                          "!fabric.bits<32>", 1, true, true});

  addUnary32YieldPe("fp",
                    ::fabric::ImplementationFamilyId::ScalarIntegerToFloat,
                    ::dataflow::OperationSchemaId::ArithUIToFP);
  addUnary32YieldPe(
      "fp_negated", ::fabric::ImplementationFamilyId::ScalarFloatSign,
      ::dataflow::OperationSchemaId::ArithNegF, "fp_negated_input");

  auto addCmpPe = [&](std::string resultName,
                      ::fabric::ImplementationFamilyId family,
                      std::vector<::dataflow::OperationSchemaId> members) {
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
                             builtinOpCapability(family),
                             std::move(members),
                             {"lhs", "rhs"},
                             {"!fabric.bits<32>", "!fabric.bits<32>"},
                             {"!fabric.bits<1>"}}},
               {"pred"},
               {"!fabric.bits<1>"}});
    module.addPe(std::move(pe));
  };
  addCmpPe("cmpf_pred",
           ::fabric::ImplementationFamilyId::ScalarFloatCompareMinMax,
           {::dataflow::OperationSchemaId::ArithCmpF});
  addCmpPe("cmpi_pred",
           ::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax,
           {::dataflow::OperationSchemaId::ArithCmpI});
  addCmpPe("cmpi_pred_aux",
           ::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax,
           {::dataflow::OperationSchemaId::ArithCmpI});

  auto addWideCmpPe = [&](std::string resultName, std::string resultType) {
    PeSpec pe;
    pe.inputs = {{"pa", "cmp64_lhs", "!fabric.bits<64>", ""},
                 {"pb", "cmp64_rhs", "!fabric.bits<64>", ""}};
    pe.resultNames = {std::move(resultName)};
    pe.resultTypes = {resultType};
    pe.fus.push_back(FuSpec{
        {{"lhs", "pa", "!fabric.bits<64>", ""},
         {"rhs", "pb", "!fabric.bits<64>", ""}},
        {resultType},
        {FabricOpSpec{
            {"pred"},
            builtinOpCapability(
                ::fabric::ImplementationFamilyId::ScalarIntegerCompareMinMax),
            {::dataflow::OperationSchemaId::ArithCmpI},
            {"lhs", "rhs"},
            {"!fabric.bits<64>", "!fabric.bits<64>"},
            {"!fabric.bits<1>"}}},
        {"pred"},
        {"!fabric.bits<1>"}});
    module.addPe(std::move(pe));
  };
  addWideCmpPe("cmpi64_pred", "!fabric.bits<64>");
  addWideCmpPe("cmpi64_pred_aux", "!fabric.bits<64>");
  module.addFifo(FifoSpec{"cmpi64_pred_aux_narrow", "cmpi64_pred_aux",
                          "!fabric.bits<32>", 1, true, true});

  PeSpec widePredExtuiPe;
  widePredExtuiPe.inputs = {{"pa", "cmpi64_pred", "!fabric.bits<64>", ""}};
  widePredExtuiPe.resultNames = {"wide_pred_extui"};
  widePredExtuiPe.resultTypes = {"!fabric.bits<64>"};
  widePredExtuiPe.fus.push_back(FuSpec{
      {{"value", "pa", "!fabric.bits<64>", "!fabric.bits<1>"}},
      {"!fabric.bits<64>"},
      {FabricOpSpec{{"extended"},
                    builtinOpCapability(
                        ::fabric::ImplementationFamilyId::ScalarIntegerCast),
                    {::dataflow::OperationSchemaId::ArithExtUI},
                    {"value"},
                    {"!fabric.bits<1>"},
                    {"!fabric.bits<64>"}}},
      {"extended"}});
  module.addPe(std::move(widePredExtuiPe));

  PeSpec selectPe;
  selectPe.inputs = {{"pa", "select_pred", "!fabric.bits<32>", ""},
                     {"pb", "select_true", "!fabric.bits<32>", ""},
                     {"pc", "select_false", "!fabric.bits<32>", ""}};
  selectPe.resultNames = {"selected"};
  selectPe.resultTypes = {"!fabric.bits<32>"};
  auto makeSelectFu = [](::dataflow::OperationSchemaId member) {
    return FuSpec{
        {{"sel", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
         {"when_true", "pb", "!fabric.bits<32>", ""},
         {"when_false", "pc", "!fabric.bits<32>", ""}},
        {"!fabric.bits<32>"},
        {FabricOpSpec{
            {"selected_value"},
            builtinOpCapability(
                ::fabric::ImplementationFamilyId::ScalarValueSelect),
            {member},
            {"sel", "when_true", "when_false"},
            {"!fabric.bits<1>", "!fabric.bits<32>", "!fabric.bits<32>"},
            {"!fabric.bits<32>"}}},
        {"selected_value"}};
  };
  selectPe.fus.push_back(
      makeSelectFu(::dataflow::OperationSchemaId::ArithSelect));
  module.addPe(std::move(selectPe));

  auto addDemuxPe = [&](llvm::StringRef valueInput, llvm::StringRef falseResult,
                        llvm::StringRef trueResult) {
    recordUnsupportedCatalogResource(
        module, {::dataflow::OperationSchemaId::DataflowDemux});
  };
  addDemuxPe("demux_value", "control_demux_false", "control_demux_true");
  addDemuxPe("demux_then_value", "compute_demux_false", "compute_demux_true");

  recordUnsupportedCatalogResource(
      module, {::dataflow::OperationSchemaId::DataflowMux});

  recordUnsupportedCatalogResource(
      module, {::dataflow::OperationSchemaId::DataflowDemux});

  recordUnsupportedCatalogResource(
      module, {::dataflow::OperationSchemaId::DataflowMux});

  recordUnsupportedCatalogResource(
      module, {::dataflow::OperationSchemaId::DataflowSync});

  recordUnsupportedCatalogResource(
      module, {::dataflow::OperationSchemaId::DataflowSync});

  auto addTypedSyncPe = [&](llvm::StringRef name, llvm::StringRef boundaryType,
                            llvm::StringRef semanticType) {
    std::string control = (name + "_control").str();
    std::string value = (name + "_value").str();
    std::string rawDone = (name + "_done_wide").str();
    std::string done = (name + "_done").str();
    std::string published = (name + "_published").str();
    PeSpec pe;
    pe.inputs = {{"pc", control, "!fabric.bits<0>", boundaryType.str()},
                 {"pv", value, boundaryType.str(), ""}};
    pe.resultNames = {rawDone, published};
    pe.resultTypes = {boundaryType.str(), boundaryType.str()};
    pe.fus.push_back(
        FuSpec{{{"control", "pc", boundaryType.str(), "!fabric.bits<0>"},
                {"value", "pv", boundaryType.str(),
                 boundaryType == semanticType ? "" : semanticType.str()}},
               {boundaryType.str(), boundaryType.str()},
               {},
               {"done", "published"},
               {"!fabric.bits<0>", semanticType.str()}});
    module.addPe(std::move(pe));
    module.addFifo(FifoSpec{done, rawDone, "!fabric.bits<0>", 1, true, true});
  };
  addTypedSyncPe("typed_sync_i1", "!fabric.bits<32>", "!fabric.bits<1>");
  addTypedSyncPe("typed_sync_i8", "!fabric.bits<32>", "!fabric.bits<8>");
  addTypedSyncPe("typed_sync_i32", "!fabric.bits<32>", "!fabric.bits<32>");
  addTypedSyncPe("typed_sync_i64", "!fabric.bits<64>", "!fabric.bits<64>");

  PeSpec addrAddPe;
  addrAddPe.inputs = {{"pa", "addr_add_lhs", "!fabric.bits<32>", ""},
                      {"pb", "addr_add_rhs", "!fabric.bits<32>", ""}};
  addrAddPe.resultNames = {"addr_sum"};
  addrAddPe.resultTypes = {"!fabric.bits<32>"};
  addrAddPe.fus.push_back(FuSpec{
      {{"lhs", "pa", "!fabric.bits<32>", ""},
       {"rhs", "pb", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"sum"},
                    builtinOpCapability(
                        ::fabric::ImplementationFamilyId::ScalarIntegerAddSub),
                    {::dataflow::OperationSchemaId::ArithAddI,
                     ::dataflow::OperationSchemaId::ArithSubI},
                    {"lhs", "rhs"},
                    {"!fabric.bits<32>", "!fabric.bits<32>"},
                    {"!fabric.bits<32>"}}},
      {"sum"}});
  module.addPe(std::move(addrAddPe));

  PeSpec addrMaskPe;
  addrMaskPe.inputs = {{"pa", "addr_mask_lhs", "!fabric.bits<32>", ""},
                       {"pb", "addr_mask_rhs", "!fabric.bits<32>", ""}};
  addrMaskPe.resultNames = {"addr_masked"};
  addrMaskPe.resultTypes = {"!fabric.bits<32>"};
  addrMaskPe.fus.push_back(FuSpec{
      {{"lhs", "pa", "!fabric.bits<32>", ""},
       {"rhs", "pb", "!fabric.bits<32>", ""}},
      {"!fabric.bits<32>"},
      {FabricOpSpec{{"masked"},
                    builtinOpCapability(
                        ::fabric::ImplementationFamilyId::ScalarIntegerLogic),
                    {::dataflow::OperationSchemaId::ArithAndI},
                    {"lhs", "rhs"},
                    {"!fabric.bits<32>", "!fabric.bits<32>"},
                    {"!fabric.bits<32>"}}},
      {"masked"}});
  module.addPe(std::move(addrMaskPe));

  addBinary32Pe("aux_masked", "aux_mask_lhs", "aux_mask_rhs", "masked",
                ::fabric::ImplementationFamilyId::ScalarIntegerLogic,
                {::dataflow::OperationSchemaId::ArithAndI});
  addBinary32Pe("aux_xor", "aux_xor_lhs", "aux_xor_rhs", "xor_value",
                ::fabric::ImplementationFamilyId::ScalarIntegerLogic,
                {::dataflow::OperationSchemaId::ArithXOrI});

  addFusedMultiplyAddPe("mac_result2", "mac2_lhs", "mac2_rhs", "mac2_acc");
  addFusedMultiplyAddPe("mac_result3", "mac3_lhs", "mac3_rhs", "mac3_acc");

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
                "scaled_reduction_aux_rhs", "product",
                ::fabric::ImplementationFamilyId::ScalarFloatMultiply,
                ::dataflow::OperationSchemaId::ArithMulF);
}

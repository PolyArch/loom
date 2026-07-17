#include "BuilderInternal.h"

#include "Dataflow/IR/DataflowEnums.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

using namespace loom::adg;
using namespace loom::adg::detail;

namespace {

struct SharedMemoryAdgConfig {
  llvm::StringRef moduleName;
  unsigned wideInputCount = 0;
  unsigned loadCount = 18;
  unsigned storeCount = 9;
  unsigned constantCount = 30;
  unsigned addCount = 12;
  unsigned cmpCount = 12;
  unsigned minCount = 10;
  unsigned maxCount = 10;
  unsigned unsignedMinCount = 4;
  unsigned unsignedMaxCount = 0;
  unsigned unsignedSaturatingSubCount = 0;
  unsigned selectCount = 8;
  unsigned wideSelectCount = 0;
  unsigned mulCount = 8;
  unsigned divCount = 0;
  unsigned unsignedDivCount = 0;
  unsigned muxCount = 0;
  unsigned controlMuxCount = 0;
  unsigned demuxCount = 0;
  unsigned wideDemuxCount = 0;
  unsigned controlDemuxCount = 0;
  unsigned logicCount = 8;
  unsigned shiftCount = 8;
  unsigned castCount = 8;
  unsigned trunciCount = 0;
  unsigned wideConstantCount = 0;
  unsigned wideAddCount = 0;
  unsigned wideShiftCount = 0;
  unsigned wideCmpCount = 0;
  unsigned wideCastCount = 4;
  unsigned wideTrunciCount = 0;
  unsigned wideIndexCastCount = 0;
  unsigned wideIndexCastUiCount = 0;
  unsigned wideSextCount = 0;
  unsigned wideMulCount = 0;
  unsigned wideUnsignedDivCount = 0;
  unsigned wideDivCount = 0;
  unsigned wideMuxCount = 0;
  unsigned wideRouteBridgeCount = 0;
  unsigned ctlzCount = 0;
  unsigned signedRemCount = 0;
  unsigned fshlCount = 0;
  unsigned armPkhbtCount = 0;
  unsigned armPkhtbCount = 0;
  unsigned armSadd16Count = 0;
  unsigned armSxtab16Count = 0;
  unsigned armSxtb16Count = 0;
  unsigned extuiCount = 4;
  unsigned fpAddCount = 4;
  unsigned fpMulCount = 4;
  unsigned fpDivCount = 0;
  unsigned fnegCount = 0;
  unsigned fabsCount = 0;
  unsigned sqrtCount = 0;
  unsigned expCount = 0;
  unsigned cosCount = 0;
  unsigned toFpCount = 0;
  unsigned fromFpCount = 0;
  unsigned signedToFpCount = 0;
  unsigned signedFromFpCount = 0;
  unsigned fmaCount = 6;
  unsigned fpCmpCount = 4;
  unsigned syncCount = 4;
  unsigned syncArity = 6;
  unsigned streamCount = 0;
  unsigned carryCount = 0;
  unsigned controlCarryCount = 0;
  unsigned gateCount = 0;
  unsigned wideGateCount = 0;
  unsigned invariantCount = 0;
  unsigned wideInvariantCount = 0;
  std::vector<llvm::StringRef> constantHexValues = {
      "0x00000000", "0x00000001", "0x00000002", "0x00000003",
      "0x00000004", "0x00000008", "0x00000010", "0xffffffff"};
  std::vector<llvm::StringRef> wideConstantHexValues = {
      "0x00000000", "0x00000001", "0x00000002",
      "0x00000003", "0x00000004", "0x00000008"};
};

ModuleBuilder buildSharedMemoryLikeAdg(const SharedMemoryAdgConfig &config) {
  ModuleBuilder module(config.moduleName.str());
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("i32a", "!fabric.bits<32>")
      .addInput("i32b", "!fabric.bits<32>")
      .addInput("i32c", "!fabric.bits<32>")
      .addInput("i32d", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>");

  std::vector<std::string> wideInputs;
  for (unsigned index = 0; index < config.wideInputCount; ++index) {
    std::string name =
        index < 4 ? ("i64" + std::string(1, static_cast<char>('a' + index)))
                  : numbered("i64", index);
    module.addInput(name, "!fabric.bits<64>");
    wideInputs.push_back(std::move(name));
  }

  std::vector<std::string> sources32 = {"i32a", "i32b", "i32c", "i32d"};
  std::vector<std::string> sinks32;
  std::vector<std::string> sources64 = wideInputs;
  std::vector<std::string> sinks64;
  std::vector<std::string> sources0 = {"ctrl"};
  std::vector<std::string> sinks0;

  auto addBinaryBank = [&](llvm::StringRef prefix, unsigned count,
                           llvm::ArrayRef<llvm::StringRef> opNames) {
    for (unsigned index = 0; index < count; ++index) {
      std::string result = numbered(prefix, index);
      std::string lhs = result + "_lhs";
      std::string rhs = result + "_rhs";
      addConfigurableBinaryPe(module, result, lhs, rhs, opNames);
      sources32.push_back(result);
      sinks32.push_back(lhs);
      sinks32.push_back(rhs);
    }
  };
  auto addUnaryBank = [&](llvm::StringRef prefix, unsigned count,
                          llvm::StringRef opName) {
    for (unsigned index = 0; index < count; ++index) {
      std::string result = numbered(prefix, index);
      std::string input = result + "_input";
      addUnaryPe(module, result, input, opName);
      sources32.push_back(result);
      sinks32.push_back(input);
    }
  };
  auto addWideExtensionBank = [&](llvm::StringRef prefix, unsigned count,
                                  llvm::StringRef opName) {
    for (unsigned index = 0; index < count; ++index) {
      std::string result = numbered(prefix, index);
      std::string input = result + "_input";
      addWideExtensionPe(module, result, input, opName);
      sources64.push_back(result);
      sinks32.push_back(input);
    }
  };
  auto addWideTruncBank = [&](llvm::StringRef prefix, unsigned count) {
    for (unsigned index = 0; index < count; ++index) {
      std::string result = numbered(prefix, index);
      std::string wideResult = result + "_wide";
      std::string input = result + "_input";
      addWideTruncPe(module, wideResult, input);
      addFifo(module, result, wideResult, "!fabric.bits<64>",
              "!fabric.bits<32>", 1, true, true);
      sources32.push_back(result);
      sinks64.push_back(input);
    }
  };
  auto addWideNarrowingBank = [&](llvm::StringRef prefix, unsigned count,
                                  llvm::StringRef opName) {
    for (unsigned index = 0; index < count; ++index) {
      std::string result = numbered(prefix, index);
      std::string wideResult = result + "_wide";
      std::string input = result + "_input";
      addWideNarrowingPe(module, wideResult, input, opName);
      addFifo(module, result, wideResult, "!fabric.bits<64>",
              "!fabric.bits<32>", 1, true, true);
      sources32.push_back(result);
      sinks64.push_back(input);
    }
  };
  auto addWideBinaryBank = [&](llvm::StringRef prefix, unsigned count,
                               llvm::ArrayRef<llvm::StringRef> opNames) {
    for (unsigned index = 0; index < count; ++index) {
      std::string result = numbered(prefix, index);
      std::string lhs = result + "_lhs";
      std::string rhs = result + "_rhs";
      addConfigurableWideBinaryPe(module, result, lhs, rhs, opNames);
      sources64.push_back(result);
      sinks64.push_back(lhs);
      sinks64.push_back(rhs);
    }
  };
  auto addTernaryBank = [&](llvm::StringRef prefix, unsigned count,
                            llvm::StringRef opName) {
    for (unsigned index = 0; index < count; ++index) {
      std::string result = numbered(prefix, index);
      std::string lhs = result + "_lhs";
      std::string rhs = result + "_rhs";
      std::string acc = result + "_acc";
      addTernaryPe(module, result, lhs, rhs, acc, opName);
      sources32.push_back(result);
      sinks32.push_back(lhs);
      sinks32.push_back(rhs);
      sinks32.push_back(acc);
    }
  };
  auto addStreamBank = [&](llvm::StringRef prefix, unsigned count) {
    for (unsigned index = 0; index < count; ++index) {
      std::string stem = numbered(prefix, index);
      std::string idx = stem + "_idx";
      std::string rwc = stem + "_rwc";
      std::string lb = stem + "_lb";
      std::string ub = stem + "_ub";
      std::string step = stem + "_step";
      PeSpec pe;
      pe.inputs = {{"pa", lb, "!fabric.bits<32>", ""},
                   {"pb", ub, "!fabric.bits<32>", ""},
                   {"pc", step, "!fabric.bits<32>", ""}};
      pe.resultNames = {idx, rwc};
      pe.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
      FuSpec fu;
      fu.inputs = {{"fa", "pa", "!fabric.bits<32>", ""},
                   {"fb", "pb", "!fabric.bits<32>", ""},
                   {"fc", "pc", "!fabric.bits<32>", ""}};
      fu.resultTypes = {"!fabric.bits<32>", "!fabric.bits<32>"};
      fu.operations.push_back(FabricOpSpec{
          {"idx", "rwc"},
          {"dataflow.stream"},
          {"fa", "fb", "fc"},
          {"!fabric.bits<32>", "!fabric.bits<32>", "!fabric.bits<32>"},
          {"!fabric.bits<32>", "!fabric.bits<1>"},
          {},
          {},
          StreamConfig{dataflow::StreamStepKind::Add,
                       {mlir::arith::CmpIPredicate::slt,
                        mlir::arith::CmpIPredicate::sgt}}});
      fu.yieldValues = {"idx", "rwc"};
      fu.yieldTypes = {"!fabric.bits<32>", "!fabric.bits<1>"};
      pe.fus.push_back(std::move(fu));
      module.addPe(std::move(pe));
      sources32.push_back(idx);
      sources32.push_back(rwc);
      sinks32.push_back(lb);
      sinks32.push_back(ub);
      sinks32.push_back(step);
    }
  };
  auto addCarryBank = [&](llvm::StringRef prefix, unsigned count,
                          llvm::StringRef dataType,
                          std::vector<std::string> &dataSources,
                          std::vector<std::string> &dataSinks) {
    for (unsigned index = 0; index < count; ++index) {
      std::string result = numbered(prefix, index);
      std::string cond = result + "_cond";
      std::string init = result + "_init";
      std::string next = result + "_next";
      bool control = dataType == "!fabric.bits<0>";
      llvm::StringRef peType = control ? "!fabric.bits<32>" : dataType;
      std::string rawResult = control ? result + "_wide" : result;
      PeSpec pe;
      pe.inputs = {{"pa", cond, "!fabric.bits<32>", ""},
                   {"pb", init, dataType.str(), control ? peType.str() : ""},
                   {"pc", next, dataType.str(), control ? peType.str() : ""}};
      pe.resultNames = {rawResult};
      pe.resultTypes = {peType.str()};
      pe.fus.push_back(FuSpec{
          {{"cond", "pa", "!fabric.bits<32>", "!fabric.bits<1>"},
           {"init", "pb", peType.str(), control ? dataType.str() : ""},
           {"next", "pc", peType.str(), control ? dataType.str() : ""}},
          {peType.str()},
          {FabricOpSpec{{"carried"},
                        {"dataflow.carry"},
                        {"cond", "init", "next"},
                        {"!fabric.bits<1>", dataType.str(), dataType.str()},
                        {dataType.str()},
                        {},
                        {}}},
          {"carried"},
          {dataType.str()}});
      module.addPe(std::move(pe));
      if (control)
        addFifo(module, result, rawResult, "!fabric.bits<32>",
                "!fabric.bits<0>", 1, true, true);
      dataSources.push_back(result);
      sinks32.push_back(cond);
      dataSinks.push_back(init);
      dataSinks.push_back(next);
    }
  };
  auto addGateBank = [&](llvm::StringRef prefix, unsigned count,
                         llvm::StringRef dataType,
                         std::vector<std::string> &dataSources,
                         std::vector<std::string> &dataSinks) {
    for (unsigned index = 0; index < count; ++index) {
      std::string stem = numbered(prefix, index);
      std::string condOut = stem + "_cond";
      bool wide = dataType == "!fabric.bits<64>";
      std::string rawCondOut = wide ? condOut + "_wide" : condOut;
      std::string valueOut = stem + "_value";
      std::string cond = stem + "_cond_in";
      std::string value = stem + "_value_in";
      PeSpec pe;
      pe.inputs = {{"pa", cond, "!fabric.bits<32>", wide ? dataType.str() : ""},
                   {"pb", value, dataType.str(), ""}};
      pe.resultNames = {rawCondOut, valueOut};
      pe.resultTypes = {dataType.str(), dataType.str()};
      pe.fus.push_back(
          FuSpec{{{"cond", "pa", dataType.str(), "!fabric.bits<1>"},
                  {"value", "pb", dataType.str(), ""}},
                 {dataType.str(), dataType.str()},
                 {FabricOpSpec{{"after_cond", "after_value"},
                               {"dataflow.gate"},
                               {"cond", "value"},
                               {"!fabric.bits<1>", dataType.str()},
                               {"!fabric.bits<1>", dataType.str()},
                               {},
                               {{"value_kind", "data"}}}},
                 {"after_cond", "after_value"},
                 {"!fabric.bits<1>", dataType.str()}});
      module.addPe(std::move(pe));
      if (wide)
        addFifo(module, condOut, rawCondOut, "!fabric.bits<64>",
                "!fabric.bits<32>", 1, true, true);
      sources32.push_back(condOut);
      dataSources.push_back(valueOut);
      sinks32.push_back(cond);
      dataSinks.push_back(value);
    }
  };
  auto addInvariantBank = [&](llvm::StringRef prefix, unsigned count,
                              llvm::StringRef dataType,
                              std::vector<std::string> &dataSources,
                              std::vector<std::string> &dataSinks) {
    for (unsigned index = 0; index < count; ++index) {
      std::string result = numbered(prefix, index);
      std::string cond = result + "_cond";
      std::string value = result + "_value";
      bool wide = dataType == "!fabric.bits<64>";
      PeSpec pe;
      pe.inputs = {{"pa", cond, "!fabric.bits<32>", wide ? dataType.str() : ""},
                   {"pb", value, dataType.str(), ""}};
      pe.resultNames = {result};
      pe.resultTypes = {dataType.str()};
      pe.fus.push_back(
          FuSpec{{{"cond", "pa", dataType.str(), "!fabric.bits<1>"},
                  {"value", "pb", dataType.str(), ""}},
                 {dataType.str()},
                 {FabricOpSpec{{"stable"},
                               {"dataflow.invariant"},
                               {"cond", "value"},
                               {"!fabric.bits<1>", dataType.str()},
                               {dataType.str()},
                               {},
                               {}}},
                 {"stable"}});
      module.addPe(std::move(pe));
      dataSources.push_back(result);
      sinks32.push_back(cond);
      dataSinks.push_back(value);
    }
  };

  for (unsigned index = 0; index < config.constantCount; ++index) {
    std::string result = numbered("const", index);
    std::string control = result + "_ctrl";
    addConfigurableConstantPe(module, result, control,
                              config.constantHexValues);
    sources32.push_back(result);
    sinks0.push_back(control);
  }
  for (unsigned index = 0; index < config.wideConstantCount; ++index) {
    std::string result = numbered("wide_const", index);
    std::string control = result + "_ctrl";
    addConfigurableWideConstantPe(module, result, control,
                                  config.wideConstantHexValues);
    sources64.push_back(result);
    sinks0.push_back(control);
  }

  addStreamBank("stream", config.streamCount);
  addCarryBank("carry", config.carryCount, "!fabric.bits<32>", sources32,
               sinks32);
  addCarryBank("control_carry", config.controlCarryCount, "!fabric.bits<0>",
               sources0, sinks0);
  addGateBank("gate", config.gateCount, "!fabric.bits<32>", sources32, sinks32);
  addGateBank("wide_gate", config.wideGateCount, "!fabric.bits<64>", sources64,
              sinks64);
  addInvariantBank("invariant", config.invariantCount, "!fabric.bits<32>",
                   sources32, sinks32);
  addInvariantBank("wide_invariant", config.wideInvariantCount,
                   "!fabric.bits<64>", sources64, sinks64);

  addBinaryBank("add", config.addCount, {"arith.addi", "arith.subi"});
  addBinaryBank("mul", config.mulCount, {"arith.muli"});
  addBinaryBank("div", config.divCount, {"arith.divsi"});
  addBinaryBank("rem", config.signedRemCount, {"arith.remsi"});
  addBinaryBank("udiv", config.unsignedDivCount,
                {"arith.divui", "arith.remui"});
  addBinaryBank("fp_add", config.fpAddCount, {"arith.addf", "arith.subf"});
  addBinaryBank("fp_mul", config.fpMulCount, {"arith.mulf"});
  addBinaryBank("fp_div", config.fpDivCount, {"arith.divf"});
  addUnaryBank("fneg", config.fnegCount, "llvm.fneg");
  addUnaryBank("fabs", config.fabsCount, "llvm.intr.fabs");
  addUnaryBank("sqrt", config.sqrtCount, "math.sqrt");
  addUnaryBank("exp", config.expCount, "math.exp");
  addUnaryBank("cos", config.cosCount, "math.cos");
  addUnaryBank("sitofp", config.signedToFpCount, "llvm.sitofp");
  addUnaryBank("uitofp", config.toFpCount, "llvm.uitofp");
  addUnaryBank("fptosi", config.signedFromFpCount, "llvm.fptosi");
  addUnaryBank("fptoui", config.fromFpCount, "llvm.fptoui");
  addTernaryBank("fma", config.fmaCount, "llvm.intr.fmuladd");
  addTernaryBank("fshl", config.fshlCount, "llvm.intr.fshl");
  addTernaryBank("arm_pkhbt", config.armPkhbtCount, "llvm.arm.pkhbt");
  addTernaryBank("arm_pkhtb", config.armPkhtbCount, "llvm.arm.pkhtb");
  addBinaryBank("arm_sadd16", config.armSadd16Count, {"llvm.arm.sadd16"});
  addBinaryBank("arm_sxtab16", config.armSxtab16Count, {"llvm.arm.sxtab16"});
  addUnaryBank("arm_sxtb16", config.armSxtb16Count, "llvm.arm.sxtb16");
  addBinaryBank("and", config.logicCount, {"arith.andi"});
  addBinaryBank("or", config.logicCount, {"arith.ori"});
  addBinaryBank("xor", config.logicCount, {"arith.xori"});
  addBinaryBank("shift", config.shiftCount,
                {"arith.shli", "arith.shrsi", "arith.shrui"});
  addWideBinaryBank("wide_add", config.wideAddCount,
                    {"arith.addi", "arith.subi"});
  addWideBinaryBank("wide_mul", config.wideMulCount, {"arith.muli"});
  addWideBinaryBank("wide_udiv", config.wideUnsignedDivCount,
                    {"arith.divui", "arith.remui"});
  addWideBinaryBank("wide_shift", config.wideShiftCount,
                    {"arith.shli", "arith.shrsi", "arith.shrui"});
  addBinaryBank("umin", config.unsignedMinCount, {"llvm.intr.umin"});
  addBinaryBank("umax", config.unsignedMaxCount, {"llvm.intr.umax"});
  addBinaryBank("usub_sat", config.unsignedSaturatingSubCount,
                {"llvm.intr.usub.sat"});
  addBinaryBank("smin", config.minCount, {"llvm.intr.smin"});
  addBinaryBank("smax", config.maxCount, {"llvm.intr.smax"});

  for (unsigned index = 0; index < config.cmpCount; ++index) {
    std::string result = numbered("cmp", index);
    std::string lhs = result + "_lhs";
    std::string rhs = result + "_rhs";
    addCmpPe(module, result, lhs, rhs);
    sources32.push_back(result);
    sinks32.push_back(lhs);
    sinks32.push_back(rhs);
  }
  for (unsigned index = 0; index < config.wideCmpCount; ++index) {
    std::string result = numbered("wide_cmp", index);
    std::string pred = result + "_pred";
    std::string lhs = result + "_lhs";
    std::string rhs = result + "_rhs";
    addWideCmpPe(module, result, lhs, rhs);
    addFifo(module, pred, result, "!fabric.bits<64>", "!fabric.bits<32>", 1,
            true, true);
    sources32.push_back(pred);
    sinks64.push_back(lhs);
    sinks64.push_back(rhs);
  }

  for (unsigned index = 0; index < config.fpCmpCount; ++index) {
    std::string result = numbered("fp_cmp", index);
    std::string lhs = result + "_lhs";
    std::string rhs = result + "_rhs";
    addFloatCmpPe(module, result, lhs, rhs);
    sources32.push_back(result);
    sinks32.push_back(lhs);
    sinks32.push_back(rhs);
  }

  for (unsigned index = 0; index < config.selectCount; ++index) {
    std::string result = numbered("select", index);
    std::string pred = result + "_pred";
    std::string trueValue = result + "_true";
    std::string falseValue = result + "_false";
    addSelectPe(module, result, pred, trueValue, falseValue);
    sources32.push_back(result);
    sinks32.push_back(pred);
    sinks32.push_back(trueValue);
    sinks32.push_back(falseValue);
  }
  for (unsigned index = 0; index < config.wideSelectCount; ++index) {
    std::string result = numbered("wide_select", index);
    std::string pred = result + "_pred";
    std::string trueValue = result + "_true";
    std::string falseValue = result + "_false";
    addWideSelectPe(module, result, pred, trueValue, falseValue);
    sources64.push_back(result);
    sinks64.push_back(pred);
    sinks64.push_back(trueValue);
    sinks64.push_back(falseValue);
  }
  for (unsigned index = 0; index < config.muxCount; ++index) {
    std::string result = numbered("mux", index);
    std::string pred = result + "_pred";
    std::string falseValue = result + "_false";
    std::string trueValue = result + "_true";
    addDataMuxPe(module, result, pred, falseValue, trueValue);
    sources32.push_back(result);
    sinks32.push_back(pred);
    sinks32.push_back(falseValue);
    sinks32.push_back(trueValue);
  }
  for (unsigned index = 0; index < config.controlMuxCount; ++index) {
    std::string result = numbered("control_mux", index);
    std::string rawResult = result + "_wide";
    std::string pred = result + "_pred";
    std::string falseValue = result + "_false";
    std::string trueValue = result + "_true";
    addControlMuxPe(module, rawResult, pred, falseValue, trueValue);
    addFifo(module, result, rawResult, "!fabric.bits<32>", "!fabric.bits<0>", 1,
            true, true);
    sources0.push_back(result);
    sinks32.push_back(pred);
    sinks0.push_back(falseValue);
    sinks0.push_back(trueValue);
  }
  for (unsigned index = 0; index < config.demuxCount; ++index) {
    std::string stem = numbered("demux", index);
    std::string falseResult = stem + "_false";
    std::string trueResult = stem + "_true";
    std::string pred = stem + "_pred";
    std::string value = stem + "_value";
    addDataDemuxPe(module, falseResult, trueResult, pred, value);
    sources32.push_back(falseResult);
    sources32.push_back(trueResult);
    sinks32.push_back(pred);
    sinks32.push_back(value);
  }
  for (unsigned index = 0; index < config.wideDemuxCount; ++index) {
    std::string stem = numbered("wide_demux", index);
    std::string falseResult = stem + "_false";
    std::string trueResult = stem + "_true";
    std::string pred = stem + "_pred";
    std::string value = stem + "_value";
    addWideDataDemuxPe(module, falseResult, trueResult, pred, value);
    sources64.push_back(falseResult);
    sources64.push_back(trueResult);
    sinks32.push_back(pred);
    sinks64.push_back(value);
  }
  for (unsigned index = 0; index < config.controlDemuxCount; ++index) {
    std::string stem = numbered("control_demux", index);
    std::string falseResult = stem + "_false";
    std::string trueResult = stem + "_true";
    std::string falseWide = falseResult + "_wide";
    std::string trueWide = trueResult + "_wide";
    std::string pred = stem + "_pred";
    std::string value = stem + "_value";
    addControlDemuxPe(module, falseWide, trueWide, pred, value);
    addFifo(module, falseResult, falseWide, "!fabric.bits<32>",
            "!fabric.bits<0>", 1, true, true);
    addFifo(module, trueResult, trueWide, "!fabric.bits<32>", "!fabric.bits<0>",
            1, true, true);
    sources0.push_back(falseResult);
    sources0.push_back(trueResult);
    sinks32.push_back(pred);
    sinks0.push_back(value);
  }
  for (unsigned index = 0; index < config.wideMuxCount; ++index) {
    std::string result = numbered("wide_mux", index);
    std::string pred = result + "_pred";
    std::string falseValue = result + "_false";
    std::string trueValue = result + "_true";
    addWideDataMuxPe(module, result, pred, falseValue, trueValue);
    sources64.push_back(result);
    sinks64.push_back(pred);
    sinks64.push_back(falseValue);
    sinks64.push_back(trueValue);
  }

  addUnaryBank("cast", config.castCount, "llvm.trunc");
  addUnaryBank("sext", config.castCount, "llvm.sext");
  addUnaryBank("zext", config.castCount, "llvm.zext");
  addUnaryBank("trunci", config.trunciCount, "arith.trunci");
  addUnaryBank("ctlz", config.ctlzCount, "llvm.intr.ctlz");
  addWideExtensionBank("wide_zext", config.wideCastCount, "llvm.zext");
  addWideExtensionBank("wide_sext", config.wideSextCount, "llvm.sext");
  addWideBinaryBank("wide_div", config.wideDivCount, {"arith.divsi"});
  addWideTruncBank("wide_trunc", config.wideCastCount);
  addWideNarrowingBank("wide_trunci", config.wideTrunciCount, "arith.trunci");
  addWideNarrowingBank("wide_index_cast", config.wideIndexCastCount,
                       "arith.index_cast");
  addWideNarrowingBank("wide_index_castui", config.wideIndexCastUiCount,
                       "arith.index_castui");
  addUnaryBank("extui", config.extuiCount, "arith.extui");

  for (unsigned index = 0; index < config.syncCount; ++index) {
    std::string prefix = numbered("sync", index);
    addControlSyncPe(module, prefix, config.syncArity);
    for (unsigned lane = 0; lane < config.syncArity; ++lane) {
      sinks0.push_back((prefix + llvm::Twine("_in") + llvm::Twine(lane)).str());
      sources0.push_back(
          (prefix + llvm::Twine("_done") + llvm::Twine(lane)).str());
    }
  }

  for (unsigned index = 0; index < config.loadCount; ++index) {
    sources32.push_back(numbered("data", index));
    sources0.push_back(numbered("done", index));
    sinks32.push_back(numbered("load_addr", index));
    sinks0.push_back(numbered("load_ctrl", index));
  }
  for (unsigned index = 0; index < config.storeCount; ++index) {
    sources0.push_back(numbered("store_done", index));
    sinks32.push_back(numbered("store_addr", index));
    sinks32.push_back(numbered("store_value", index));
    sinks0.push_back(numbered("store_ctrl", index));
  }

  llvm::SmallVector<std::string, 4> wideRouteBridgeInputs;
  llvm::SmallVector<std::string, 4> wideRouteBridgeResults;
  for (unsigned index = 0; index < config.wideRouteBridgeCount; ++index) {
    std::string stem = numbered("wide_route_bridge", index);
    std::string input = stem + "_input";
    wideRouteBridgeInputs.push_back(input);
    wideRouteBridgeResults.push_back(stem);
    sinks32.push_back(input);
  }

  addUniformSwitch(module, sinks32, sources32, "!fabric.bits<32>");
  for (auto [input, result] :
       llvm::zip(wideRouteBridgeInputs, wideRouteBridgeResults)) {
    addFifo(module, result, input, "!fabric.bits<32>", "!fabric.bits<64>", 1,
            true, true);
    sources64.push_back(result);
  }
  addUniformSwitch(module, sinks64, sources64, "!fabric.bits<64>");
  addUniformSwitch(module, sinks0, sources0, "!fabric.bits<0>");
  addMemoryReductionMem(module, config.loadCount, config.storeCount);
  return module;
}

} // namespace

ModuleBuilder loom::adg::buildSharedMemoryReductionAdg() {
  SharedMemoryAdgConfig config;
  config.moduleName = "shared_memory_reduction_adg";
  config.selectCount = 12;
  config.wideSelectCount = 2;
  config.unsignedDivCount = 2;
  config.muxCount = 4;
  config.controlMuxCount = 3;
  config.demuxCount = 5;
  config.wideDemuxCount = 2;
  config.controlDemuxCount = 6;
  config.trunciCount = 4;
  config.wideConstantCount = 8;
  config.wideAddCount = 2;
  config.wideShiftCount = 3;
  config.wideCmpCount = 1;
  config.wideIndexCastCount = 6;
  config.wideIndexCastUiCount = 2;
  config.wideSextCount = 4;
  config.wideMulCount = 2;
  config.wideUnsignedDivCount = 2;
  config.wideMuxCount = 1;
  config.wideRouteBridgeCount = 2;
  config.streamCount = 1;
  config.controlCarryCount = 3;
  config.gateCount = 2;
  config.wideGateCount = 2;
  config.invariantCount = 2;
  config.wideInvariantCount = 2;
  config.constantHexValues = {
      "0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004",
      "0x00000008", "0x00000010", "0x3f800000", "0x40000000", "0xbf800000",
      "0x0000001e", "0x0000003f", "0xffffffff"};
  config.wideConstantHexValues = {"0x00000000", "0x00000001", "0x00000002",
                                  "0x00000003", "0x00000004", "0x00000008",
                                  "0x0000001f", "0x40000000"};
  return buildSharedMemoryLikeAdg(config);
}

ModuleBuilder loom::adg::buildSharedQuantizedWindowAdg() {
  SharedMemoryAdgConfig config;
  config.moduleName = "shared_quantized_window_adg";
  config.constantCount = 40;
  config.addCount = 84;
  config.cmpCount = 84;
  config.selectCount = 76;
  config.wideSelectCount = 20;
  config.mulCount = 48;
  config.divCount = 4;
  config.signedRemCount = 2;
  config.muxCount = 4;
  config.logicCount = 40;
  config.unsignedMaxCount = 2;
  config.shiftCount = 96;
  config.castCount = 44;
  config.wideConstantCount = 2;
  config.wideAddCount = 48;
  config.wideShiftCount = 28;
  config.wideCmpCount = 4;
  config.wideCastCount = 52;
  config.wideSextCount = 32;
  config.wideMulCount = 48;
  config.wideDivCount = 20;
  config.wideRouteBridgeCount = 16;
  config.ctlzCount = 2;
  config.fshlCount = 2;
  config.armPkhbtCount = 2;
  config.armPkhtbCount = 1;
  config.armSadd16Count = 4;
  config.armSxtab16Count = 2;
  config.armSxtb16Count = 4;
  config.minCount = 11;
  config.maxCount = 11;
  config.streamCount = 4;
  config.carryCount = 8;
  config.invariantCount = 8;
  config.constantHexValues = {
      "0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004",
      "0x00000008", "0x0000000f", "0x00000010", "0x00000018", "0x0000001b",
      "0x0000001f", "0x000000ff", "0x0000ff00", "0x0000ffef", "0x0000ffff",
      "0x00ff0000", "0x30000000", "0x40000000", "0xffffffff", "0xffff0000"};
  config.wideConstantHexValues = {"0x0000001f", "0x40000000"};
  return buildSharedMemoryLikeAdg(config);
}

ModuleBuilder loom::adg::buildSharedSignalWindowAdg() {
  SharedMemoryAdgConfig config;
  config.moduleName = "shared_signal_window_adg";
  config.wideInputCount = 4;
  config.loadCount = 40;
  config.storeCount = 40;
  config.constantCount = 48;
  config.addCount = 32;
  config.cmpCount = 16;
  config.selectCount = 16;
  config.mulCount = 16;
  config.divCount = 4;
  config.muxCount = 2;
  config.logicCount = 16;
  config.shiftCount = 16;
  config.castCount = 16;
  config.wideConstantCount = 3;
  config.wideAddCount = 2;
  config.wideCmpCount = 2;
  config.wideTrunciCount = 2;
  config.wideIndexCastCount = 4;
  config.wideIndexCastUiCount = 2;
  config.wideRouteBridgeCount = 2;
  config.fpAddCount = 72;
  config.fpMulCount = 24;
  config.fpDivCount = 4;
  config.fnegCount = 4;
  config.fabsCount = 2;
  config.sqrtCount = 2;
  config.expCount = 4;
  config.cosCount = 4;
  config.signedToFpCount = 2;
  config.toFpCount = 4;
  config.signedFromFpCount = 2;
  config.fromFpCount = 2;
  config.unsignedSaturatingSubCount = 2;
  config.constantHexValues = {
      "0x00000000", "0x00000001", "0x00000002", "0x00000003", "0x00000004",
      "0x00000008", "0x00000010", "0xffffffff", "0x3f800000", "0x40000000",
      "0xbf800000", "0x322bcc77", "0x3727c5ac", "0x3e22f983", "0x44000000",
      "0xc4000000", "0x000001ff"};
  config.wideConstantHexValues = {"0x0000000000000000", "0x0000000000000001",
                                  "0x0000000000000002", "0x0000000000000003",
                                  "0x0000000000000004", "0x0000000000000008",
                                  "0x0000000000000010"};
  config.fmaCount = 8;
  config.fpCmpCount = 8;
  config.syncCount = 4;
  config.syncArity = 20;
  config.streamCount = 4;
  config.carryCount = 28;
  config.gateCount = 28;
  config.invariantCount = 12;
  return buildSharedMemoryLikeAdg(config);
}

ModuleBuilder loom::adg::buildSharedVectorAluAdg() {
  ModuleBuilder module("shared_vector_alu_adg");
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("idx0", "!fabric.bits<32>")
      .addInput("idx1", "!fabric.bits<32>")
      .addInput("store_idx", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>")
      .addInput("i32a", "!fabric.bits<32>")
      .addInput("i32b", "!fabric.bits<32>");

  PeSpec xorPe;
  xorPe.inputs = {{"lhs", "bin0", "!fabric.bits<32>", ""},
                  {"rhs", "bin1", "!fabric.bits<32>", ""}};
  xorPe.resultNames = {"xored"};
  xorPe.resultTypes = {"!fabric.bits<32>"};
  xorPe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"arith.xori"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(xorPe));

  PeSpec bswapPe;
  bswapPe.inputs = {{"value", "unary", "!fabric.bits<32>", ""}};
  bswapPe.resultNames = {"swapped"};
  bswapPe.resultTypes = {"!fabric.bits<32>"};
  bswapPe.fus.push_back(FuSpec{{{"input", "value", "!fabric.bits<32>", ""}},
                               {"!fabric.bits<32>"},
                               {FabricOpSpec{{"result"},
                                             {"llvm.intr.bswap"},
                                             {"input"},
                                             {"!fabric.bits<32>"},
                                             {"!fabric.bits<32>"},
                                             {},
                                             {}}},
                               {"result"}});
  module.addPe(std::move(bswapPe));

  PeSpec floatMulPe;
  floatMulPe.inputs = {{"lhs", "bin0", "!fabric.bits<32>", ""},
                       {"rhs", "bin1", "!fabric.bits<32>", ""}};
  floatMulPe.resultNames = {"product"};
  floatMulPe.resultTypes = {"!fabric.bits<32>"};
  floatMulPe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"arith.mulf"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(floatMulPe));

  PeSpec intMulPe;
  intMulPe.inputs = {{"lhs", "bin0", "!fabric.bits<32>", ""},
                     {"rhs", "i32b", "!fabric.bits<32>", ""}};
  intMulPe.resultNames = {"int_product"};
  intMulPe.resultTypes = {"!fabric.bits<32>"};
  intMulPe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"arith.muli"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(intMulPe));

  PeSpec intAddPe;
  intAddPe.inputs = {{"lhs", "int_product", "!fabric.bits<32>", ""},
                     {"rhs", "bin1", "!fabric.bits<32>", ""}};
  intAddPe.resultNames = {"int_sum"};
  intAddPe.resultTypes = {"!fabric.bits<32>"};
  intAddPe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"arith.addi"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(intAddPe));

  PeSpec qsub16Pe;
  qsub16Pe.inputs = {{"lhs", "bin0", "!fabric.bits<32>", ""},
                     {"rhs", "bin1", "!fabric.bits<32>", ""}};
  qsub16Pe.resultNames = {"qsub16"};
  qsub16Pe.resultTypes = {"!fabric.bits<32>"};
  qsub16Pe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"llvm.arm.qsub16"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(qsub16Pe));

  PeSpec syncPe;
  syncPe.inputs = {{"pa", "sync0", "!fabric.bits<0>", ""},
                   {"pb", "sync1", "!fabric.bits<0>", ""},
                   {"pc", "sync2", "!fabric.bits<0>", ""}};
  syncPe.resultTypes = {"!fabric.bits<0>"};
  syncPe.fus.push_back(FuSpec{
      {{"fa", "pa", "!fabric.bits<0>", ""},
       {"fb", "pb", "!fabric.bits<0>", ""},
       {"fc", "pc", "!fabric.bits<0>", ""}},
      {"!fabric.bits<0>"},
      {FabricOpSpec{{"sa", "sb", "sc"},
                    {"dataflow.sync"},
                    {"fa", "fb", "fc"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {},
                    {{"bitmask", "111"}}}},
      {"sa"}});
  module.addPe(std::move(syncPe));

  addTwoLoadOneStoreMem(module);
  addUniformSwitch(module,
                   {"load_ctrl0", "load_ctrl1", "store_ctrl", "sync0",
                    "sync1", "sync2"},
                   {"ctrl", "done0", "done1", "store_done"},
                   "!fabric.bits<0>");
  addSpatialSwitch(module, {"bin0", "bin1", "unary"},
                   {"data0", "data1", "i32a"}, {"111", "111", "111"});
  addUniformSwitch(module, {"store_value"},
                   {"xored", "swapped", "product", "int_product", "int_sum",
                    "qsub16", "i32b"},
                   "!fabric.bits<32>");
  return module;
}

ModuleBuilder loom::adg::buildSharedVectorMathAdg() {
  SharedMemoryAdgConfig config;
  config.moduleName = "shared_vector_math_adg";
  config.loadCount = 8;
  config.storeCount = 4;
  config.constantCount = 6;
  config.addCount = 2;
  config.cmpCount = 1;
  config.minCount = 0;
  config.maxCount = 0;
  config.unsignedMinCount = 0;
  config.unsignedMaxCount = 0;
  config.selectCount = 0;
  config.mulCount = 3;
  config.divCount = 0;
  config.unsignedDivCount = 0;
  config.muxCount = 0;
  config.demuxCount = 7;
  config.wideDemuxCount = 2;
  config.controlDemuxCount = 8;
  config.logicCount = 3;
  config.shiftCount = 4;
  config.castCount = 0;
  config.trunciCount = 0;
  config.wideConstantCount = 2;
  config.wideAddCount = 0;
  config.wideShiftCount = 0;
  config.wideCmpCount = 0;
  config.wideCastCount = 5;
  config.wideIndexCastCount = 4;
  config.wideIndexCastUiCount = 0;
  config.wideSextCount = 0;
  config.wideMulCount = 0;
  config.wideUnsignedDivCount = 0;
  config.wideDivCount = 0;
  config.wideMuxCount = 0;
  config.wideRouteBridgeCount = 0;
  config.ctlzCount = 0;
  config.extuiCount = 0;
  config.fpAddCount = 0;
  config.fpMulCount = 4;
  config.fnegCount = 4;
  config.fmaCount = 12;
  config.fpCmpCount = 0;
  config.syncCount = 16;
  config.syncArity = 16;
  config.streamCount = 1;
  config.controlCarryCount = 4;
  config.gateCount = 3;
  config.invariantCount = 3;
  return buildSharedMemoryLikeAdg(config);
}

ModuleBuilder loom::adg::buildSharedVectorMeshAdg() {
  ModuleBuilder module("shared_vector_mesh_adg");
  addVisualLayout(module, {{"mem", 0, 1},
                           {"west0", 1, 0},
                           {"west1", 1, 1},
                           {"west_unary", 1, 2},
                           {"xored", 2, 0},
                           {"swapped", 2, 2},
                           {"store_value", 3, 1},
                           {"sync", 2, 3}});
  module.addInput("mgr", "memref<?x!fabric.bits<32>>")
      .addInput("idx0", "!fabric.bits<32>")
      .addInput("idx1", "!fabric.bits<32>")
      .addInput("store_idx", "!fabric.bits<32>")
      .addInput("ctrl", "!fabric.bits<0>")
      .addInput("i32a", "!fabric.bits<32>");

  PeSpec xorPe;
  xorPe.inputs = {{"lhs", "mesh_lhs", "!fabric.bits<32>", ""},
                  {"rhs", "mesh_rhs", "!fabric.bits<32>", ""}};
  xorPe.resultNames = {"xored"};
  xorPe.resultTypes = {"!fabric.bits<32>"};
  xorPe.fus.push_back(
      FuSpec{{{"a", "lhs", "!fabric.bits<32>", ""},
              {"b", "rhs", "!fabric.bits<32>", ""}},
             {"!fabric.bits<32>"},
             {FabricOpSpec{{"value"},
                           {"arith.xori"},
                           {"a", "b"},
                           {"!fabric.bits<32>", "!fabric.bits<32>"},
                           {"!fabric.bits<32>"},
                           {},
                           {}}},
             {"value"}});
  module.addPe(std::move(xorPe));

  PeSpec bswapPe;
  bswapPe.inputs = {{"value", "mesh_unary", "!fabric.bits<32>", ""}};
  bswapPe.resultNames = {"swapped"};
  bswapPe.resultTypes = {"!fabric.bits<32>"};
  bswapPe.fus.push_back(FuSpec{{{"input", "value", "!fabric.bits<32>", ""}},
                               {"!fabric.bits<32>"},
                               {FabricOpSpec{{"result"},
                                             {"llvm.intr.bswap"},
                                             {"input"},
                                             {"!fabric.bits<32>"},
                                             {"!fabric.bits<32>"},
                                             {},
                                             {}}},
                               {"result"}});
  module.addPe(std::move(bswapPe));

  PeSpec syncPe;
  syncPe.inputs = {{"pa", "sync0", "!fabric.bits<0>", ""},
                   {"pb", "sync1", "!fabric.bits<0>", ""},
                   {"pc", "sync2", "!fabric.bits<0>", ""}};
  syncPe.resultTypes = {"!fabric.bits<0>"};
  syncPe.fus.push_back(FuSpec{
      {{"fa", "pa", "!fabric.bits<0>", ""},
       {"fb", "pb", "!fabric.bits<0>", ""},
       {"fc", "pc", "!fabric.bits<0>", ""}},
      {"!fabric.bits<0>"},
      {FabricOpSpec{{"sa", "sb", "sc"},
                    {"dataflow.sync"},
                    {"fa", "fb", "fc"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {"!fabric.bits<0>", "!fabric.bits<0>", "!fabric.bits<0>"},
                    {},
                    {{"bitmask", "111"}}}},
      {"sa"}});
  module.addPe(std::move(syncPe));

  addTwoLoadOneStoreMem(module);
  addUniformSwitch(module,
                   {"load_ctrl0", "load_ctrl1", "store_ctrl", "sync0",
                    "sync1", "sync2"},
                   {"ctrl", "done0", "done1", "store_done"},
                   "!fabric.bits<0>");
  addSpatialSwitch(module, {"west0", "west1", "west_unary"},
                   {"data0", "data1", "i32a"}, {"111", "111", "101"});
  addSpatialSwitch(module, {"mesh_lhs", "mesh_rhs_pre"}, {"west0", "west1"},
                   {"11", "11"});
  addSpatialSwitch(module, {"mesh_rhs", "mesh_unary"},
                   {"mesh_rhs_pre", "west0", "west_unary"}, {"111", "011"});
  addUniformSwitch(module, {"store_value"}, {"xored", "swapped", "i32a"},
                   "!fabric.bits<32>");
  return module;
}

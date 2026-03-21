#include "fcc/SVGen/SVModuleRegistry.h"

#include "llvm/ADT/StringRef.h"

using namespace fcc::svgen;

namespace {

/// Mapping entry: MLIR op name -> { dialect_subdir, sv_filename }.
struct OpMapping {
  llvm::StringRef mlirOp;
  llvm::StringRef category;
  llvm::StringRef filename;
};

// All known MLIR op -> SV file mappings.
// Sorted by dialect for readability.
static const OpMapping kOpMappings[] = {
    // arith dialect
    {"arith.addi", "arith", "fu_op_addi.sv"},
    {"arith.subi", "arith", "fu_op_subi.sv"},
    {"arith.andi", "arith", "fu_op_andi.sv"},
    {"arith.ori", "arith", "fu_op_ori.sv"},
    {"arith.xori", "arith", "fu_op_xori.sv"},
    {"arith.shli", "arith", "fu_op_shli.sv"},
    {"arith.shrsi", "arith", "fu_op_shrsi.sv"},
    {"arith.shrui", "arith", "fu_op_shrui.sv"},
    {"arith.cmpi", "arith", "fu_op_cmpi.sv"},
    {"arith.extsi", "arith", "fu_op_extsi.sv"},
    {"arith.extui", "arith", "fu_op_extui.sv"},
    {"arith.trunci", "arith", "fu_op_trunci.sv"},
    {"arith.select", "arith", "fu_op_select.sv"},
    {"arith.index_cast", "arith", "fu_op_index_cast.sv"},
    {"arith.index_castui", "arith", "fu_op_index_castui.sv"},
    {"arith.muli", "arith", "fu_op_muli.sv"},
    {"arith.divsi", "arith", "fu_op_divsi.sv"},
    {"arith.divui", "arith", "fu_op_divui.sv"},
    {"arith.remsi", "arith", "fu_op_remsi.sv"},
    {"arith.remui", "arith", "fu_op_remui.sv"},
    // arith FP
    {"arith.addf", "arith", "fu_op_addf.sv"},
    {"arith.subf", "arith", "fu_op_subf.sv"},
    {"arith.mulf", "arith", "fu_op_mulf.sv"},
    {"arith.divf", "arith", "fu_op_divf.sv"},
    {"arith.cmpf", "arith", "fu_op_cmpf.sv"},
    {"arith.negf", "arith", "fu_op_negf.sv"},
    {"arith.fptosi", "arith", "fu_op_fptosi.sv"},
    {"arith.fptoui", "arith", "fu_op_fptoui.sv"},
    {"arith.sitofp", "arith", "fu_op_sitofp.sv"},
    {"arith.uitofp", "arith", "fu_op_uitofp.sv"},
    // math dialect
    {"math.absf", "math", "fu_op_absf.sv"},
    {"math.fma", "math", "fu_op_fma.sv"},
    {"math.sqrt", "math", "fu_op_sqrt.sv"},
    // math Tier 3 transcendental
    {"math.cos", "math", "fu_op_cos.sv"},
    {"math.sin", "math", "fu_op_sin.sv"},
    {"math.exp", "math", "fu_op_exp.sv"},
    {"math.log2", "math", "fu_op_log2.sv"},
    // llvm dialect
    {"llvm.intr.bitreverse", "llvm", "fu_op_bitreverse.sv"},
    // dataflow dialect
    {"dataflow.stream", "dataflow", "fu_op_stream.sv"},
    {"dataflow.gate", "dataflow", "fu_op_gate.sv"},
    {"dataflow.carry", "dataflow", "fu_op_carry.sv"},
    {"dataflow.invariant", "dataflow", "fu_op_invariant.sv"},
    // handshake dialect
    {"handshake.cond_br", "handshake", "fu_op_cond_br.sv"},
    {"handshake.constant", "handshake", "fu_op_constant.sv"},
    {"handshake.join", "handshake", "fu_op_join.sv"},
    {"handshake.load", "handshake", "fu_op_load.sv"},
    {"handshake.store", "handshake", "fu_op_store.sv"},
    {"handshake.mux", "handshake", "fu_op_mux.sv"},
};

static const OpMapping *findMapping(llvm::StringRef mlirOpName) {
  for (const auto &m : kOpMappings) {
    if (m.mlirOp == mlirOpName)
      return &m;
  }
  return nullptr;
}

// Tier 3 transcendental FP ops that need --fp-ip-profile.
static bool isTier3Impl(llvm::StringRef mlirOpName) {
  return mlirOpName == "math.cos" || mlirOpName == "math.sin" ||
         mlirOpName == "math.exp" || mlirOpName == "math.log2";
}

} // namespace

void SVModuleRegistry::requireCommonInfrastructure() {
  if (commonRequired_)
    return;
  commonRequired_ = true;
  requiredFiles_.insert("common/fabric_pkg.sv");
  requiredFiles_.insert("common/fabric_handshake_if.sv");
  requiredFiles_.insert("common/fabric_cfg_if.sv");
  requiredFiles_.insert("common/fabric_fifo_mem.sv");
  requiredFiles_.insert("common/fabric_rr_arbiter.sv");
  requiredFiles_.insert("common/fabric_broadcast_tracker.sv");
}

void SVModuleRegistry::requireModule(llvm::StringRef category,
                                     llvm::StringRef filename) {
  requireCommonInfrastructure();
  std::string path = (category + "/" + filename).str();
  requiredFiles_.insert(path);
}

bool SVModuleRegistry::requireArithOp(llvm::StringRef mlirOpName,
                                      llvm::StringRef fpIpProfile) {
  if (isTier3Impl(mlirOpName) && fpIpProfile.empty())
    return false;

  const OpMapping *mapping = findMapping(mlirOpName);
  if (!mapping) {
    // Not a known op. Return false so the caller can report the error.
    // fabric.mux and fabric.yield are handled separately before this is called.
    return false;
  }

  requireModule(mapping->category, mapping->filename);
  return true;
}

std::vector<std::string> SVModuleRegistry::getRequiredFiles() const {
  // Return in compilation order: packages first, then common, then leaves.
  // Since we use a std::set, common/ sorts first, which is correct.
  std::vector<std::string> result;
  result.reserve(requiredFiles_.size());

  // Packages and common first (they sort lexicographically before others).
  for (const auto &f : requiredFiles_) {
    if (f.find("common/") == 0)
      result.push_back(f);
  }
  // Then all other files.
  for (const auto &f : requiredFiles_) {
    if (f.find("common/") != 0)
      result.push_back(f);
  }
  return result;
}

bool SVModuleRegistry::isKnownOp(llvm::StringRef mlirOpName) {
  return findMapping(mlirOpName) != nullptr;
}

std::string SVModuleRegistry::getSVModuleName(llvm::StringRef mlirOpName) {
  const OpMapping *mapping = findMapping(mlirOpName);
  if (!mapping)
    return "";
  // Strip .sv extension from filename.
  llvm::StringRef fn = mapping->filename;
  if (fn.ends_with(".sv"))
    fn = fn.drop_back(3);
  return fn.str();
}

std::string SVModuleRegistry::getSVFilePath(llvm::StringRef mlirOpName) {
  const OpMapping *mapping = findMapping(mlirOpName);
  if (!mapping)
    return "";
  return (mapping->category + "/" + mapping->filename).str();
}

bool SVModuleRegistry::isTier3TranscendentalOp(llvm::StringRef mlirOpName) {
  return isTier3Impl(mlirOpName);
}

//===-- ADGExportSV.cpp - SystemVerilog export implementation --*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Implements ADGBuilder::exportSV() and the internal generateSV() method.
// Generates a self-contained SystemVerilog design directory from the ADG.
//
//===----------------------------------------------------------------------===//

#include "loom/Hardware/ADG/ADGBuilderImpl.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <set>
#include <sstream>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Helper: get data width in bits for a Type
//===----------------------------------------------------------------------===//

static unsigned getDataWidthBits(const Type &t) {
  switch (t.getKind()) {
  case Type::I1:    return 1;
  case Type::I8:    return 8;
  case Type::I16:   return 16;
  case Type::I32:   return 32;
  case Type::I64:   return 64;
  case Type::IN:    return t.getWidth();
  case Type::BF16:  return 16;
  case Type::F16:   return 16;
  case Type::F32:   return 32;
  case Type::F64:   return 64;
  case Type::Index: return 64;
  case Type::None:  return 0;
  case Type::Tagged:
    return getDataWidthBits(t.getValueType());
  }
  return 32;
}

static unsigned getTagWidthBits(const Type &t) {
  if (!t.isTagged())
    return 0;
  return getDataWidthBits(t.getTagType());
}

//===----------------------------------------------------------------------===//
// Helper: write string to file
//===----------------------------------------------------------------------===//

static void writeFile(const std::string &path, const std::string &content) {
  std::error_code ec;
  llvm::raw_fd_ostream out(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: cannot write output file: " << path << "\n"
                 << ec.message() << "\n";
    std::exit(1);
  }
  out << content;
  out.flush();
}

//===----------------------------------------------------------------------===//
// Helper: copy a template file from LOOM_SV_TEMPLATE_DIR
//===----------------------------------------------------------------------===//

static void copyTemplateFile(const std::string &srcDir,
                             const std::string &relPath,
                             const std::string &dstDir) {
  llvm::SmallString<256> srcPath(srcDir);
  llvm::sys::path::append(srcPath, relPath);

  llvm::SmallString<256> dstPath(dstDir);
  llvm::sys::path::append(dstPath, llvm::sys::path::filename(relPath));

  std::error_code ec = llvm::sys::fs::copy_file(srcPath, dstPath);
  if (ec) {
    llvm::errs() << "error: cannot copy template file: " << srcPath << " -> "
                 << dstPath << "\n"
                 << ec.message() << "\n";
    std::exit(1);
  }
}

//===----------------------------------------------------------------------===//
// SV module kind name
//===----------------------------------------------------------------------===//

static const char *svModuleName(ModuleKind kind) {
  switch (kind) {
  case ModuleKind::PE:            return "fabric_pe";
  case ModuleKind::ConstantPE:    return "fabric_pe_constant";
  case ModuleKind::LoadPE:        return "fabric_pe_load";
  case ModuleKind::StorePE:       return "fabric_pe_store";
  case ModuleKind::Switch:        return "fabric_switch";
  case ModuleKind::TemporalPE:    return "fabric_temporal_pe";
  case ModuleKind::TemporalSwitch:return "fabric_temporal_sw";
  case ModuleKind::Memory:        return "fabric_memory";
  case ModuleKind::ExtMemory:     return "fabric_extmemory";
  case ModuleKind::AddTag:        return "fabric_add_tag";
  case ModuleKind::MapTag:        return "fabric_map_tag";
  case ModuleKind::DelTag:        return "fabric_del_tag";
  case ModuleKind::Fifo:          return "fabric_fifo";
  }
  return "fabric_unknown";
}

//===----------------------------------------------------------------------===//
// Helper: compute NUM_CONNECTED for a switch definition
//===----------------------------------------------------------------------===//

static unsigned getNumConnected(const SwitchDef &def) {
  if (def.connectivity.empty())
    return def.numIn * def.numOut;
  unsigned count = 0;
  for (const auto &row : def.connectivity)
    for (bool v : row)
      if (v) ++count;
  return count;
}

//===----------------------------------------------------------------------===//
// Generate switch instance parameters
//===----------------------------------------------------------------------===//

static std::string genSwitchParams(const SwitchDef &def) {
  unsigned dw = getDataWidthBits(def.portType);
  unsigned tw = getTagWidthBits(def.portType);
  // Control-token types (Type::none) have zero data+tag width.
  // Use DATA_WIDTH=1 as the minimum SV representation so the
  // hardware compiles; the data bits are unused for control tokens.
  if (dw + tw == 0)
    dw = 1;
  std::ostringstream os;
  os << "    .NUM_INPUTS(" << def.numIn << "),\n";
  os << "    .NUM_OUTPUTS(" << def.numOut << "),\n";
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << ")";

  // Connectivity matrix
  if (!def.connectivity.empty()) {
    unsigned total = def.numOut * def.numIn;
    os << ",\n    .CONNECTIVITY(" << total << "'b";
    // Emit MSB first (highest output, highest input)
    for (int o = static_cast<int>(def.numOut) - 1; o >= 0; --o) {
      for (int i = static_cast<int>(def.numIn) - 1; i >= 0; --i) {
        if (static_cast<unsigned>(o) < def.connectivity.size() &&
            static_cast<unsigned>(i) < def.connectivity[o].size())
          os << (def.connectivity[o][i] ? "1" : "0");
        else
          os << "1";
      }
    }
    os << ")";
  }

  return os.str();
}

//===----------------------------------------------------------------------===//
// Generate FIFO instance parameters
//===----------------------------------------------------------------------===//

static std::string genFifoParams(const FifoDef &def) {
  unsigned dw = getDataWidthBits(def.elementType);
  unsigned tw = getTagWidthBits(def.elementType);
  // Control-token types (Type::none) have zero data+tag width.
  // Use DATA_WIDTH=1 as the minimum SV representation so the
  // hardware compiles; the data bits are unused for control tokens.
  if (dw + tw == 0)
    dw = 1;
  std::ostringstream os;
  os << "    .DEPTH(" << def.depth << "),\n";
  os << "    .DATA_WIDTH(" << dw << "),\n";
  os << "    .TAG_WIDTH(" << tw << "),\n";
  os << "    .BYPASSABLE(" << (def.bypassable ? 1 : 0) << ")";
  return os.str();
}

//===----------------------------------------------------------------------===//
// ADGBuilder::Impl::generateSV
//===----------------------------------------------------------------------===//

static bool hasSVTemplate(ModuleKind kind) {
  return kind == ModuleKind::Switch || kind == ModuleKind::Fifo;
}

/// Returns true for module kinds that expose error_valid/error_code ports.
static bool hasErrorOutput(ModuleKind kind) {
  return kind == ModuleKind::Switch || kind == ModuleKind::TemporalSwitch ||
         kind == ModuleKind::TemporalPE;
}

//===----------------------------------------------------------------------===//
// Helper: validate a name as a legal SystemVerilog identifier
//===----------------------------------------------------------------------===//

static bool isSVKeyword(const std::string &name) {
  // IEEE 1800-2017 Table B.1 - all reserved keywords.
  static const std::set<std::string> keywords = {
      "accept_on",    "alias",         "always",        "always_comb",
      "always_ff",    "always_latch",  "and",           "assert",
      "assign",       "assume",        "automatic",     "before",
      "begin",        "bind",          "bins",          "binsof",
      "bit",          "break",         "buf",           "bufif0",
      "bufif1",       "byte",          "case",          "casex",
      "casez",        "cell",          "chandle",       "checker",
      "class",        "clocking",      "cmos",          "config",
      "const",        "constraint",    "context",       "continue",
      "cover",        "covergroup",    "coverpoint",    "cross",
      "deassign",     "default",       "defparam",      "design",
      "disable",      "dist",          "do",            "edge",
      "else",         "end",           "endcase",       "endchecker",
      "endclass",     "endclocking",   "endconfig",     "endfunction",
      "endgenerate",  "endgroup",      "endinterface",  "endmodule",
      "endpackage",   "endprimitive",  "endprogram",    "endproperty",
      "endsequence",  "endspecify",    "endtable",      "endtask",
      "enum",         "event",         "eventually",    "expect",
      "export",       "extends",       "extern",        "final",
      "first_match",  "for",           "force",         "foreach",
      "forever",      "fork",          "forkjoin",      "function",
      "generate",     "genvar",        "global",        "highz0",
      "highz1",       "if",            "iff",           "ifnone",
      "ignore_bins",  "illegal_bins",  "implements",    "implies",
      "import",       "incdir",        "include",       "initial",
      "inout",        "input",         "inside",        "instance",
      "int",          "integer",       "interconnect",  "interface",
      "intersect",    "join",          "join_any",      "join_none",
      "large",        "let",           "liblist",       "library",
      "local",        "localparam",    "logic",         "longint",
      "macromodule",  "matches",       "medium",        "modport",
      "module",       "nand",          "negedge",       "nettype",
      "new",          "nexttime",      "nmos",          "nor",
      "noshowcancelled","not",         "notif0",        "notif1",
      "null",         "or",            "output",        "package",
      "packed",       "parameter",     "pmos",          "posedge",
      "primitive",    "priority",      "program",       "property",
      "protected",    "pull0",         "pull1",         "pulldown",
      "pullup",       "pulsestyle_ondetect","pulsestyle_onevent","pure",
      "rand",         "randc",         "randcase",      "randsequence",
      "rcmos",        "real",          "realtime",      "ref",
      "reg",          "reject_on",     "release",       "repeat",
      "restrict",     "return",        "rnmos",         "rpmos",
      "rtran",        "rtranif0",      "rtranif1",      "s_always",
      "s_eventually", "s_nexttime",    "s_until",       "s_until_with",
      "scalared",     "sequence",      "shortint",      "shortreal",
      "showcancelled","signed",        "small",         "soft",
      "solve",        "specify",       "specparam",     "static",
      "string",       "strong",        "strong0",       "strong1",
      "struct",       "super",         "supply0",       "supply1",
      "sync_accept_on","sync_reject_on","table",        "tagged",
      "task",         "this",          "throughout",    "time",
      "timeprecision","timeunit",      "tran",          "tranif0",
      "tranif1",      "tri",           "tri0",          "tri1",
      "triand",       "trior",         "trireg",        "type",
      "typedef",      "union",         "unique",        "unique0",
      "unsigned",     "until",         "until_with",    "untyped",
      "use",          "uwire",         "var",           "vectored",
      "virtual",      "void",          "wait",          "wait_order",
      "wand",         "weak",          "weak0",         "weak1",
      "while",        "wildcard",      "wire",          "with",
      "within",       "wor",           "xnor",          "xor",
  };
  return keywords.count(name) != 0;
}

static bool isValidSVIdentifier(const std::string &name) {
  if (name.empty())
    return false;
  // Must start with letter or underscore
  if (!std::isalpha(static_cast<unsigned char>(name[0])) && name[0] != '_')
    return false;
  for (char c : name) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
      return false;
  }
  return !isSVKeyword(name);
}

void ADGBuilder::Impl::generateSV(const std::string &directory) const {
  // Reject unsupported module kinds before producing any output
  for (const auto &inst : instances) {
    if (!hasSVTemplate(inst.kind)) {
      llvm::errs() << "error: exportSV does not support module kind '"
                   << svModuleName(inst.kind) << "' (instance '" << inst.name
                   << "')\n";
      std::exit(1);
    }
  }

  // Validate module name (emitted as <moduleName>_top)
  if (!isValidSVIdentifier(moduleName)) {
    llvm::errs() << "error: exportSV: module name '" << moduleName
                 << "' is not a valid SystemVerilog identifier\n";
    std::exit(1);
  }

  // Reject memref ports (not supported in SV export)
  for (const auto &p : ports) {
    if (p.isMemref) {
      llvm::errs() << "error: exportSV does not support memref port '"
                   << p.name << "'\n";
      std::exit(1);
    }
  }

  // Determine once whether any instance produces error signals
  bool hasErrorPorts = false;
  for (const auto &inst : instances) {
    if (hasErrorOutput(inst.kind)) {
      hasErrorPorts = true;
      break;
    }
  }

  // Validate instance names: must be valid SV identifiers and unique
  {
    std::set<std::string> seenNames;
    for (const auto &inst : instances) {
      if (!isValidSVIdentifier(inst.name)) {
        llvm::errs() << "error: exportSV: instance name '" << inst.name
                     << "' is not a valid SystemVerilog identifier\n";
        std::exit(1);
      }
      if (!seenNames.insert(inst.name).second) {
        llvm::errs() << "error: exportSV: duplicate instance name '"
                     << inst.name << "'\n";
        std::exit(1);
      }
    }
  }

  // Validate port names: must be valid SV identifiers, unique, and not
  // collide with generated aggregated port names (error_valid/error_code).
  {
    std::set<std::string> seenPorts;
    if (hasErrorPorts)
      seenPorts.insert("error");
    for (const auto &p : ports) {
      if (!isValidSVIdentifier(p.name)) {
        llvm::errs() << "error: exportSV: port name '" << p.name
                     << "' is not a valid SystemVerilog identifier\n";
        std::exit(1);
      }
      if (!seenPorts.insert(p.name).second) {
        llvm::errs() << "error: exportSV: port name '" << p.name
                     << "' collides with another port or reserved name\n";
        std::exit(1);
      }
    }
  }

  // Collect instance-derived internal signal names (wires + config ports).
  static const char *streamSuffixes[] = {"_valid", "_ready", "_data"};
  std::set<std::string> instanceDerived;
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];
    unsigned numIn = getInstanceInputCount(i);
    unsigned numOut = getInstanceOutputCount(i);
    for (unsigned p = 0; p < numIn; ++p) {
      std::string base = inst.name + "_in" + std::to_string(p);
      for (const char *sfx : streamSuffixes)
        instanceDerived.insert(base + sfx);
    }
    for (unsigned p = 0; p < numOut; ++p) {
      std::string base = inst.name + "_out" + std::to_string(p);
      for (const char *sfx : streamSuffixes)
        instanceDerived.insert(base + sfx);
    }
    if (hasErrorOutput(inst.kind)) {
      instanceDerived.insert(inst.name + "_error_valid");
      instanceDerived.insert(inst.name + "_error_code");
    }
    if (inst.kind == ModuleKind::Switch) {
      instanceDerived.insert(inst.name + "_cfg_route_table");
    } else if (inst.kind == ModuleKind::Fifo) {
      const auto &def = fifoDefs[inst.defIdx];
      if (def.bypassable)
        instanceDerived.insert(inst.name + "_cfg_data");
    }
  }

  // Check port suffixed names against instance-derived signals
  for (const auto &p : ports) {
    for (const char *sfx : streamSuffixes) {
      if (instanceDerived.count(p.name + sfx)) {
        llvm::errs() << "error: exportSV: port '" << p.name
                     << "' generates signal '" << p.name << sfx
                     << "' that collides with an internal/config signal\n";
        std::exit(1);
      }
    }
  }

  // Check instance names against top-level identifiers (fixed ports,
  // port-derived signals, error aggregation ports, and instance-derived names).
  {
    std::set<std::string> topLevel({"clk", "rst_n"});
    if (hasErrorPorts) {
      topLevel.insert("error_valid");
      topLevel.insert("error_code");
    }
    for (const auto &p : ports) {
      for (const char *sfx : streamSuffixes)
        topLevel.insert(p.name + sfx);
    }
    topLevel.insert(instanceDerived.begin(), instanceDerived.end());
    for (const auto &inst : instances) {
      if (topLevel.count(inst.name)) {
        llvm::errs() << "error: exportSV: instance name '" << inst.name
                     << "' collides with a generated top-level identifier\n";
        std::exit(1);
      }
    }
  }

  // Create output directories
  llvm::SmallString<256> libDir(directory);
  llvm::sys::path::append(libDir, "lib");

  llvm::sys::fs::create_directories(directory);
  llvm::sys::fs::create_directories(libDir);

  // Copy template files from LOOM_SV_TEMPLATE_DIR
#ifdef LOOM_SV_TEMPLATE_DIR
  const std::string templateDir = LOOM_SV_TEMPLATE_DIR;
#else
  const std::string templateDir = "";
#endif

  if (!templateDir.empty()) {
    // Copy Common/ files
    copyTemplateFile(templateDir, "Common/fabric_common.svh", libDir.str().str());
    // Copy Fabric/ files
    copyTemplateFile(templateDir, "Fabric/fabric_fifo.sv", libDir.str().str());
    copyTemplateFile(templateDir, "Fabric/fabric_switch.sv", libDir.str().str());
  }

  // ---- Generate top-level module ----
  std::ostringstream top;
  top << "//===-- " << moduleName << "_top.sv - Generated top-level module ---===//\n";
  top << "//\n";
  top << "// Generated by Loom ADG Builder. Do not edit.\n";
  top << "//\n";
  top << "//===----------------------------------------------------------------------===//\n\n";

  // Module port declarations
  top << "module " << moduleName << "_top (\n";
  top << "    input  logic clk,\n";

  // Collect stream ports
  std::vector<const ModulePort *> streamInputs, streamOutputs;
  for (const auto &p : ports) {
    if (p.isInput)
      streamInputs.push_back(&p);
    else
      streamOutputs.push_back(&p);
  }

  // Collect instance config ports (switch route tables, bypassable FIFO cfg)
  struct InstCfgPort {
    std::string portName; // top-level port name
    unsigned width;
  };
  std::vector<InstCfgPort> instCfgPorts;
  for (const auto &inst : instances) {
    if (inst.kind == ModuleKind::Switch) {
      const auto &def = switchDefs[inst.defIdx];
      instCfgPorts.push_back({inst.name + "_cfg_route_table",
                              getNumConnected(def)});
    } else if (inst.kind == ModuleKind::Fifo) {
      const auto &def = fifoDefs[inst.defIdx];
      if (def.bypassable)
        instCfgPorts.push_back({inst.name + "_cfg_data", 1});
    }
  }
  bool hasCfgPorts = !instCfgPorts.empty();

  bool hasMorePorts = !streamInputs.empty() || !streamOutputs.empty() ||
                      hasCfgPorts || hasErrorPorts;
  top << "    input  logic rst_n" << (hasMorePorts ? "," : "") << "\n";

  for (size_t i = 0; i < streamInputs.size(); ++i) {
    const auto *p = streamInputs[i];
    unsigned w = getDataWidthBits(p->type) + getTagWidthBits(p->type);
    bool last = (i + 1 == streamInputs.size()) && streamOutputs.empty() &&
                !hasCfgPorts && !hasErrorPorts;
    top << "    input  logic " << p->name << "_valid,\n";
    top << "    output logic " << p->name << "_ready,\n";
    top << "    input  logic " << (w > 1 ? "[" + std::to_string(w-1) + ":0] " : "")
        << p->name << "_data" << (last ? "" : ",") << "\n";
  }

  for (size_t i = 0; i < streamOutputs.size(); ++i) {
    const auto *p = streamOutputs[i];
    unsigned w = getDataWidthBits(p->type) + getTagWidthBits(p->type);
    bool last = (i + 1 == streamOutputs.size()) && !hasCfgPorts &&
                !hasErrorPorts;
    top << "    output logic " << p->name << "_valid,\n";
    top << "    input  logic " << p->name << "_ready,\n";
    top << "    output logic " << (w > 1 ? "[" + std::to_string(w-1) + ":0] " : "")
        << p->name << "_data" << (last ? "" : ",") << "\n";
  }

  // Per-instance config input ports
  for (size_t i = 0; i < instCfgPorts.size(); ++i) {
    const auto &cp = instCfgPorts[i];
    bool last = (i + 1 == instCfgPorts.size()) && !hasErrorPorts;
    top << "    input  logic "
        << (cp.width > 1 ? "[" + std::to_string(cp.width - 1) + ":0] " : "")
        << cp.portName << (last ? "" : ",") << "\n";
  }

  if (hasErrorPorts) {
    top << "    output logic        error_valid,\n";
    top << "    output logic [15:0] error_code\n";
  }
  top << ");\n\n";

  // Wire declarations for internal connections
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];
    unsigned numIn = getInstanceInputCount(i);
    unsigned numOut = getInstanceOutputCount(i);

    for (unsigned p = 0; p < numIn; ++p) {
      Type pt = getInstanceInputType(i, p);
      unsigned w = getDataWidthBits(pt) + getTagWidthBits(pt);
      top << "  logic " << inst.name << "_in" << p << "_valid;\n";
      top << "  logic " << inst.name << "_in" << p << "_ready;\n";
      if (w > 1)
        top << "  logic [" << (w-1) << ":0] " << inst.name << "_in" << p << "_data;\n";
      else
        top << "  logic " << inst.name << "_in" << p << "_data;\n";
    }
    for (unsigned p = 0; p < numOut; ++p) {
      Type pt = getInstanceOutputType(i, p);
      unsigned w = getDataWidthBits(pt) + getTagWidthBits(pt);
      top << "  logic " << inst.name << "_out" << p << "_valid;\n";
      top << "  logic " << inst.name << "_out" << p << "_ready;\n";
      if (w > 1)
        top << "  logic [" << (w-1) << ":0] " << inst.name << "_out" << p << "_data;\n";
      else
        top << "  logic " << inst.name << "_out" << p << "_data;\n";
    }

    // Per-instance error signals for modules that support them
    if (hasErrorOutput(inst.kind)) {
      top << "  logic " << inst.name << "_error_valid;\n";
      top << "  logic [15:0] " << inst.name << "_error_code;\n";
    }
    top << "\n";
  }

  // Instantiate components
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];
    top << "  // Instance: " << inst.name << " (" << svModuleName(inst.kind) << ")\n";

    switch (inst.kind) {
    case ModuleKind::Switch: {
      const auto &def = switchDefs[inst.defIdx];
      top << "  fabric_switch #(\n" << genSwitchParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      // Port connections
      top << "    .in_valid({";
      for (int p = def.numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_ready({";
      for (int p = def.numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .in_data({";
      for (int p = def.numIn - 1; p >= 0; --p) {
        top << inst.name << "_in" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_valid({";
      for (int p = def.numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_valid";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_ready({";
      for (int p = def.numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_ready";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .out_data({";
      for (int p = def.numOut - 1; p >= 0; --p) {
        top << inst.name << "_out" << p << "_data";
        if (p > 0) top << ", ";
      }
      top << "}),\n";
      top << "    .cfg_route_table(" << inst.name << "_cfg_route_table),\n";
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::Fifo: {
      const auto &def = fifoDefs[inst.defIdx];
      top << "  fabric_fifo #(\n" << genFifoParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in_data(" << inst.name << "_in0_data),\n";
      top << "    .out_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out_data(" << inst.name << "_out0_data),\n";
      if (def.bypassable)
        top << "    .cfg_data(" << inst.name << "_cfg_data)\n";
      else
        top << "    .cfg_data('0)\n";
      top << "  );\n\n";
      break;
    }
    default:
      llvm_unreachable("unsupported module kind (should be caught earlier)");
    }
  }

  // Collect ready sources per output port for fanout aggregation.
  // Key: "<inst_name>_out<port>" or "%<module_port_name>" -> list of ready signals.
  std::map<std::string, std::vector<std::string>> readySources;

  // Wire connections: module inputs to instances
  for (const auto &conn : inputConns) {
    const auto &port = ports[conn.portIdx];
    const auto &inst = instances[conn.instIdx];
    std::string sinkReady = inst.name + "_in" + std::to_string(conn.dstPort) + "_ready";
    top << "  assign " << inst.name << "_in" << conn.dstPort
        << "_valid = " << port.name << "_valid;\n";
    top << "  assign " << inst.name << "_in" << conn.dstPort
        << "_data = " << port.name << "_data;\n";
    readySources["%" + port.name].push_back(sinkReady);
  }
  // Emit aggregated ready for module input ports; drive unconnected to 0
  for (const auto *p : streamInputs) {
    std::string key = "%" + p->name;
    auto it = readySources.find(key);
    if (it == readySources.end()) {
      top << "  assign " << p->name << "_ready = 1'b0;\n";
    } else if (it->second.size() == 1) {
      top << "  assign " << p->name << "_ready = " << it->second[0] << ";\n";
    } else {
      top << "  assign " << p->name << "_ready = " << it->second[0];
      for (size_t s = 1; s < it->second.size(); ++s)
        top << " & " << it->second[s];
      top << ";\n";
    }
  }
  readySources.clear();

  // Wire connections: instances to module outputs
  std::set<unsigned> assignedOutputPorts;
  for (const auto &conn : outputConns) {
    const auto &inst = instances[conn.instIdx];
    const auto &port = ports[conn.portIdx];
    if (!assignedOutputPorts.insert(conn.portIdx).second) {
      llvm::errs() << "error: exportSV: module output '" << port.name
                   << "' has multiple source connections\n";
      std::exit(1);
    }
    std::string srcKey = inst.name + "_out" + std::to_string(conn.srcPort);
    top << "  assign " << port.name << "_valid = " << srcKey << "_valid;\n";
    top << "  assign " << port.name << "_data = " << srcKey << "_data;\n";
    readySources[srcKey].push_back(port.name + "_ready");
  }

  // Wire connections: internal (instance to instance)
  for (const auto &conn : internalConns) {
    const auto &srcInst = instances[conn.srcInst];
    const auto &dstInst = instances[conn.dstInst];
    std::string srcKey = srcInst.name + "_out" + std::to_string(conn.srcPort);
    top << "  assign " << dstInst.name << "_in" << conn.dstPort
        << "_valid = " << srcKey << "_valid;\n";
    top << "  assign " << dstInst.name << "_in" << conn.dstPort
        << "_data = " << srcKey << "_data;\n";
    readySources[srcKey].push_back(
        dstInst.name + "_in" + std::to_string(conn.dstPort) + "_ready");
  }

  // Emit aggregated ready for instance output ports
  for (const auto &[srcKey, sources] : readySources) {
    if (sources.size() == 1) {
      top << "  assign " << srcKey << "_ready = " << sources[0] << ";\n";
    } else {
      top << "  assign " << srcKey << "_ready = " << sources[0];
      for (size_t s = 1; s < sources.size(); ++s)
        top << " & " << sources[s];
      top << ";\n";
    }
  }

  // Error aggregation: latch first error until reset
  if (hasErrorPorts) {
    top << "\n  // Error latch: captures first error, held until reset\n";
    top << "  always_ff @(posedge clk or negedge rst_n) begin\n";
    top << "    if (!rst_n) begin\n";
    top << "      error_valid <= 1'b0;\n";
    top << "      error_code  <= 16'd0;\n";
    top << "    end else if (!error_valid) begin\n";
    // Only capture when no error latched yet
    bool first = true;
    for (const auto &inst : instances) {
      if (hasErrorOutput(inst.kind)) {
        top << "      " << (first ? "if" : "else if") << " ("
            << inst.name << "_error_valid) begin\n";
        top << "        error_valid <= 1'b1;\n";
        top << "        error_code  <= " << inst.name << "_error_code;\n";
        top << "      end\n";
        first = false;
      }
    }
    top << "    end\n";
    top << "  end\n";
  }

  top << "\nendmodule\n";

  // Write top-level module
  llvm::SmallString<256> topPath(directory);
  llvm::sys::path::append(topPath, moduleName + "_top.sv");
  writeFile(topPath.str().str(), top.str());
}

//===----------------------------------------------------------------------===//
// ADGBuilder::exportSV
//===----------------------------------------------------------------------===//

void ADGBuilder::exportSV(const std::string &directory) {
  auto validation = validateADG();
  if (!validation.success) {
    llvm::errs() << "error: ADG validation failed with "
                 << validation.errors.size() << " error(s):\n";
    for (const auto &err : validation.errors) {
      llvm::errs() << "  [" << err.code << "] " << err.message;
      if (!err.location.empty())
        llvm::errs() << " (at " << err.location << ")";
      llvm::errs() << "\n";
    }
    std::exit(1);
  }

  impl_->generateSV(directory);
}

} // namespace adg
} // namespace loom

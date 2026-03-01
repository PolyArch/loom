//===-- ADGExportSVUtil.cpp - SV export utility functions --------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//
//
// Utility and parsing helpers shared by the SystemVerilog export pipeline.
//
//===----------------------------------------------------------------------===//

#include "ADGExportSVInternal.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Helper: get data width in bits for a Type
//===----------------------------------------------------------------------===//

unsigned getDataWidthBits(const Type &t) {
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
  case Type::Index: return loom::ADDR_BIT_WIDTH;
  case Type::None:  return 0;
  case Type::Bits:  return t.getWidth();
  case Type::Tagged:
    return getDataWidthBits(t.getValueType());
  }
  return 32;
}

unsigned getTagWidthBits(const Type &t) {
  if (!t.isTagged())
    return 0;
  return getDataWidthBits(t.getTagType());
}

//===----------------------------------------------------------------------===//
// Helper: write string to file
//===----------------------------------------------------------------------===//

void writeFile(const std::string &path, const std::string &content) {
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

void copyTemplateFile(const std::string &srcDir,
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
// Helper: read a file into a string
//===----------------------------------------------------------------------===//

std::string readFile(const std::string &path) {
  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr) {
    llvm::errs() << "error: cannot read file: " << path << "\n";
    std::exit(1);
  }
  return (*bufOrErr)->getBuffer().str();
}

//===----------------------------------------------------------------------===//
// PE helpers: operation latency and body generation
//===----------------------------------------------------------------------===//

/// Return the typical latency (in cycles) for a single operation name.
unsigned getOpLatency(const std::string &opName) {
  // Bitwise / extension / truncation: 0 cycles
  if (opName == "arith.andi" || opName == "arith.ori" ||
      opName == "arith.xori" || opName == "arith.shli" ||
      opName == "arith.shrsi" || opName == "arith.shrui" ||
      opName == "arith.trunci" || opName == "arith.extsi" ||
      opName == "arith.extui" || opName == "arith.index_cast" ||
      opName == "arith.index_castui" || opName == "llvm.intr.bitreverse")
    return 0;
  // Integer add/sub/cmp/select: 1 cycle
  if (opName == "arith.addi" || opName == "arith.subi" ||
      opName == "arith.cmpi" || opName == "arith.select")
    return 1;
  // Integer mul: 3 cycles
  if (opName == "arith.muli")
    return 3;
  // Integer div/rem: 6 cycles
  if (opName == "arith.divsi" || opName == "arith.divui" ||
      opName == "arith.remsi" || opName == "arith.remui")
    return 6;
  // FP add/sub/cmp/neg: 4 cycles
  if (opName == "arith.addf" || opName == "arith.subf" ||
      opName == "arith.cmpf" || opName == "arith.negf" ||
      opName == "math.absf")
    return 4;
  // FP mul/fma: 5 cycles
  if (opName == "arith.mulf" || opName == "math.fma")
    return 5;
  // FP conversion: 4 cycles
  if (opName == "arith.sitofp" || opName == "arith.uitofp" ||
      opName == "arith.fptosi" || opName == "arith.fptoui")
    return 4;
  // FP div/sqrt: 10 cycles
  if (opName == "arith.divf" || opName == "math.sqrt")
    return 10;
  // FP transcendental: 12 cycles
  if (opName == "math.sin" || opName == "math.cos" ||
      opName == "math.exp" || opName == "math.log2")
    return 12;
  // Default: 1 cycle
  return 1;
}

/// Convert a dialect.op name to the SV module name (e.g., "arith.addi" -> "arith_addi").
std::string opToSVModule(const std::string &opName) {
  // Normalize LLVM intrinsic names: strip the ".intr" segment
  // e.g., "llvm.intr.bitreverse" -> "llvm_bitreverse"
  if (opName.find("llvm.intr.") == 0) {
    std::string result = "llvm_" + opName.substr(10);
    for (char &c : result)
      if (c == '.')
        c = '_';
    return result;
  }
  std::string result = opName;
  for (char &c : result)
    if (c == '.')
      c = '_';
  return result;
}

/// Return true if the op is a width-conversion operation (uses IN_WIDTH/OUT_WIDTH).
bool isConversionOp(const std::string &opName) {
  return opName == "arith.extsi" || opName == "arith.extui" ||
         opName == "arith.trunci" || opName == "arith.sitofp" ||
         opName == "arith.uitofp" || opName == "arith.fptosi" ||
         opName == "arith.fptoui" || opName == "arith.index_cast" ||
         opName == "arith.index_castui";
}

/// Return true if the op is a compare operation (1-bit result).
bool isCompareOp(const std::string &opName) {
  return opName == "arith.cmpi" || opName == "arith.cmpf";
}

/// Map an MLIR arith.cmpi predicate name to its integer encoding.
/// PREDICATE encoding: 0=eq,1=ne,2=slt,3=sle,4=sgt,5=sge,6=ult,7=ule,8=ugt,9=uge
int cmpiPredicateToInt(const std::string &pred) {
  if (pred == "eq") return 0;
  if (pred == "ne") return 1;
  if (pred == "slt") return 2;
  if (pred == "sle") return 3;
  if (pred == "sgt") return 4;
  if (pred == "sge") return 5;
  if (pred == "ult") return 6;
  if (pred == "ule") return 7;
  if (pred == "ugt") return 8;
  if (pred == "uge") return 9;
  return 0; // default: eq
}

/// Map an MLIR arith.cmpf predicate name to its integer encoding.
/// PREDICATE encoding: 0=false,1=oeq,2=ogt,3=oge,4=olt,5=ole,6=one,7=ord,
///   8=ueq,9=ugt,10=uge,11=ult,12=ule,13=une,14=uno,15=true
int cmpfPredicateToInt(const std::string &pred) {
  if (pred == "false") return 0;
  if (pred == "oeq") return 1;
  if (pred == "ogt") return 2;
  if (pred == "oge") return 3;
  if (pred == "olt") return 4;
  if (pred == "ole") return 5;
  if (pred == "one") return 6;
  if (pred == "ord") return 7;
  if (pred == "ueq") return 8;
  if (pred == "ugt") return 9;
  if (pred == "uge") return 10;
  if (pred == "ult") return 11;
  if (pred == "ule") return 12;
  if (pred == "une") return 13;
  if (pred == "uno") return 14;
  if (pred == "true") return 15;
  return 0; // default: false
}

/// Resolve a compare predicate integer from the operation name and predicate
/// string. Returns 0 if the op is not a compare or predicate is empty.
int resolveComparePredicate(const std::string &opName,
                                   const std::string &pred) {
  if (pred.empty()) return 0;
  if (opName == "arith.cmpi") return cmpiPredicateToInt(pred);
  if (opName == "arith.cmpf") return cmpfPredicateToInt(pred);
  return 0;
}

MLIRStmt parseMLIRLine(const std::string &line) {
  MLIRStmt stmt;
  auto eqPos = line.find('=');
  if (eqPos == std::string::npos)
    return stmt;

  // Extract result name from LHS
  std::string lhs = line.substr(0, eqPos);
  auto pctPos = lhs.find('%');
  if (pctPos == std::string::npos)
    return stmt;
  auto endRes = lhs.find_first_of(" \t,", pctPos);
  stmt.result = lhs.substr(pctPos, endRes - pctPos);

  // Extract RHS: "dialect.op %a, %b : type"
  std::string rhs = line.substr(eqPos + 1);
  // Trim leading whitespace
  auto rhsStart = rhs.find_first_not_of(" \t");
  if (rhsStart == std::string::npos)
    return stmt;
  rhs = rhs.substr(rhsStart);

  // Extract op name (first token)
  auto opEnd = rhs.find_first_of(" \t");
  if (opEnd == std::string::npos) {
    stmt.opName = rhs;
    return stmt;
  }
  stmt.opName = rhs.substr(0, opEnd);

  // Extract operands between op name and ':'
  auto colonPos = rhs.find(':');
  std::string opSection = rhs.substr(opEnd, colonPos != std::string::npos
                                                 ? colonPos - opEnd
                                                 : std::string::npos);
  // Extract predicate keyword for compare ops (text before first '%').
  // e.g., in " sgt, %a, %b" the predicate is "sgt".
  auto firstPct = opSection.find('%');
  if (firstPct != std::string::npos && firstPct > 0) {
    std::string predSection = opSection.substr(0, firstPct);
    auto ps = predSection.find_first_not_of(" \t,");
    if (ps != std::string::npos) {
      auto pe = predSection.find_last_not_of(" \t,");
      stmt.predicate = predSection.substr(ps, pe - ps + 1);
    }
  }

  // Find all %name tokens
  size_t pos = 0;
  while ((pos = opSection.find('%', pos)) != std::string::npos) {
    auto end = opSection.find_first_of(" ,:\t\n)", pos);
    stmt.operands.push_back(opSection.substr(pos, end - pos));
    pos = (end != std::string::npos) ? end + 1 : opSection.size();
  }

  // Capture type annotation (text after ':')
  if (colonPos != std::string::npos) {
    std::string ta = rhs.substr(colonPos + 1);
    auto taStart = ta.find_first_not_of(" \t");
    if (taStart != std::string::npos) {
      auto taEnd = ta.find_last_not_of(" \t\n\r");
      stmt.typeAnnotation = ta.substr(taStart, taEnd - taStart + 1);
    }
  }

  return stmt;
}

/// Parse width from an MLIR integer type string like "i32", "i16", "index".
/// Returns 0 if not a recognized integer type.
unsigned parseMLIRTypeWidth(const std::string &typeStr) {
  auto s = typeStr.find_first_not_of(" \t");
  if (s == std::string::npos) return 0;
  std::string t = typeStr.substr(s);
  auto e = t.find_first_of(" \t,)");
  if (e != std::string::npos) t = t.substr(0, e);
  if (t == "index") return loom::ADDR_BIT_WIDTH;
  if (t.size() > 1 && (t[0] == 'i' || t[0] == 'f')) {
    unsigned w = 0;
    for (size_t i = 1; i < t.size(); ++i) {
      if (!std::isdigit(t[i])) return 0;
      w = w * 10 + (t[i] - '0');
    }
    return w;
  }
  return 0;
}

/// Parse conversion op type annotation: "i16 to i32" -> {16, 32}.
/// Returns {0, 0} if parsing fails.
std::pair<unsigned, unsigned>
parseConversionWidths(const std::string &typeAnnotation) {
  // Format: "i16 to i32" or "i32 to i16"
  auto toPos = typeAnnotation.find(" to ");
  if (toPos == std::string::npos) {
    // Try without spaces: "i16to i32" is unlikely but handle "i16 -> i32"
    toPos = typeAnnotation.find("->");
    if (toPos != std::string::npos) {
      unsigned inW = parseMLIRTypeWidth(typeAnnotation.substr(0, toPos));
      unsigned outW = parseMLIRTypeWidth(typeAnnotation.substr(toPos + 2));
      return {inW, outW};
    }
    return {0, 0};
  }
  unsigned inW = parseMLIRTypeWidth(typeAnnotation.substr(0, toPos));
  unsigned outW = parseMLIRTypeWidth(typeAnnotation.substr(toPos + 4));
  return {inW, outW};
}

/// Extract all operation names from a bodyMLIR string.
std::vector<std::string> extractBodyMLIROps(const std::string &bodyMLIR) {
  std::vector<std::string> ops;
  std::string body = bodyMLIR;
  auto bbPos = body.find("^bb0(");
  if (bbPos != std::string::npos) {
    auto closePos = body.find("):", bbPos);
    if (closePos != std::string::npos)
      body = body.substr(closePos + 2);
  }
  std::istringstream stream(body);
  std::string line;
  while (std::getline(stream, line)) {
    auto s = line.find_first_not_of(" \t");
    if (s == std::string::npos) continue;
    line = line.substr(s);
    if (line.find("fabric.yield") == 0) continue;
    MLIRStmt stmt = parseMLIRLine(line);
    if (!stmt.opName.empty())
      ops.push_back(stmt.opName);
  }
  return ops;
}

/// Compute the critical-path latency through a bodyMLIR DAG.
/// Uses a simple approach: sum of all operation latencies along the
/// longest dependency chain.
unsigned computeBodyMLIRLatency(const std::string &bodyMLIR) {
  std::string body = bodyMLIR;
  auto bbPos = body.find("^bb0(");
  if (bbPos != std::string::npos) {
    auto closePos = body.find("):", bbPos);
    if (closePos != std::string::npos)
      body = body.substr(closePos + 2);
  }

  // Parse all stmts and build a dependency graph
  std::vector<MLIRStmt> stmts;
  std::istringstream stream(body);
  std::string line;
  while (std::getline(stream, line)) {
    auto s = line.find_first_not_of(" \t");
    if (s == std::string::npos) continue;
    line = line.substr(s);
    if (line.find("fabric.yield") == 0) continue;
    MLIRStmt stmt = parseMLIRLine(line);
    if (!stmt.opName.empty())
      stmts.push_back(stmt);
  }

  // Map SSA result -> index in stmts
  std::map<std::string, unsigned> resultToIdx;
  for (unsigned i = 0; i < stmts.size(); ++i)
    resultToIdx[stmts[i].result] = i;

  // Compute longest path to each node
  std::vector<unsigned> longest(stmts.size(), 0);
  for (unsigned i = 0; i < stmts.size(); ++i) {
    unsigned maxPred = 0;
    for (const auto &op : stmts[i].operands) {
      auto it = resultToIdx.find(op);
      if (it != resultToIdx.end())
        maxPred = std::max(maxPred, longest[it->second]);
    }
    longest[i] = maxPred + getOpLatency(stmts[i].opName);
  }

  unsigned maxLatency = 0;
  for (unsigned v : longest)
    maxLatency = std::max(maxLatency, v);
  return maxLatency;
}

/// Compute the maximum intermediate type width in a bodyMLIR.
/// Scans block arg types, conversion source/destination widths, and
/// operation type annotations to find the widest type used internally.
unsigned computeBodyMLIRMaxWidth(const std::string &bodyMLIR) {
  unsigned maxW = 0;
  std::string body = bodyMLIR;

  // Parse block arg types from "^bb0(%a: i16, %b: i32):"
  auto bbPos = body.find("^bb0(");
  if (bbPos != std::string::npos) {
    auto closePos = body.find("):", bbPos);
    if (closePos != std::string::npos) {
      std::string argList = body.substr(bbPos + 5, closePos - (bbPos + 5));
      std::istringstream argStream(argList);
      std::string token;
      while (std::getline(argStream, token, ',')) {
        auto colon = token.find(':');
        if (colon != std::string::npos) {
          unsigned w = parseMLIRTypeWidth(token.substr(colon + 1));
          maxW = std::max(maxW, w);
        }
      }
      body = body.substr(closePos + 2);
    }
  }

  // Scan each statement's type annotation for widths
  std::istringstream stream(body);
  std::string line;
  while (std::getline(stream, line)) {
    auto s = line.find_first_not_of(" \t");
    if (s == std::string::npos) continue;
    line = line.substr(s);
    if (line.find("fabric.yield") == 0) continue;
    MLIRStmt stmt = parseMLIRLine(line);
    if (stmt.opName.empty() || stmt.typeAnnotation.empty()) continue;

    if (isConversionOp(stmt.opName)) {
      auto [inW, outW] = parseConversionWidths(stmt.typeAnnotation);
      maxW = std::max(maxW, inW);
      maxW = std::max(maxW, outW);
    } else {
      // Non-conversion ops: type annotation is the result type (e.g., "i32")
      unsigned w = parseMLIRTypeWidth(stmt.typeAnnotation);
      maxW = std::max(maxW, w);
    }
  }
  return maxW;
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

bool isValidSVIdentifier(const std::string &name) {
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

//===----------------------------------------------------------------------===//
// Helper: count compare ops in a PEDef
//===----------------------------------------------------------------------===//

unsigned countCmpOps(const PEDef &def) {
  if (!def.singleOp.empty())
    return isCompareOp(def.singleOp) ? 1 : 0;
  if (!def.bodyMLIR.empty()) {
    unsigned count = 0;
    for (const auto &op : extractBodyMLIROps(def.bodyMLIR))
      if (isCompareOp(op))
        ++count;
    return count;
  }
  return 0;
}

bool hasCmpiOp(const PEDef &def) {
  if (def.singleOp == "arith.cmpi")
    return true;
  if (!def.bodyMLIR.empty()) {
    for (const auto &op : extractBodyMLIROps(def.bodyMLIR))
      if (op == "arith.cmpi")
        return true;
  }
  return false;
}

std::vector<std::string> getCmpOps(const PEDef &def) {
  std::vector<std::string> result;
  if (!def.singleOp.empty()) {
    if (isCompareOp(def.singleOp))
      result.push_back(def.singleOp);
    return result;
  }
  if (!def.bodyMLIR.empty()) {
    for (const auto &op : extractBodyMLIROps(def.bodyMLIR))
      if (isCompareOp(op))
        result.push_back(op);
  }
  return result;
}

} // namespace adg
} // namespace loom

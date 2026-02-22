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

#include "ADGExportSVInternal.h"

#include "loom/Hardware/ADG/ADGBuilderImpl.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// ADGBuilder::Impl::generateSV
//===----------------------------------------------------------------------===//

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

  // Reject memref ports (not supported in SV export).
  // This catches non-private Memory and ExtMemory modules that expose memref
  // at the module boundary.  Private memories are fine (no memref ports).
  for (const auto &p : ports) {
    if (p.isMemref) {
      llvm::errs() << "error: exportSV does not support memref port '"
                   << p.name
                   << "'; Memory/ExtMemory with external memref cannot be "
                      "exported to SystemVerilog\n";
      std::exit(1);
    }
  }

  // Validate that LoadPE/StorePE DATA_WIDTH can carry the connected memory's
  // address range.  The SV templates encode addresses in DATA_WIDTH-sized lanes,
  // so narrow element types (e.g. i8 with memDepth > 256) would silently
  // truncate upper address bits.
  for (const auto &conn : internalConns) {
    const auto &srcInst = instances[conn.srcInst];
    const auto &dstInst = instances[conn.dstInst];
    unsigned addrWidth = 0;
    unsigned dataWidth = 0;
    std::string peName;
    if ((srcInst.kind == ModuleKind::LoadPE && conn.srcPort == 0) ||
        (srcInst.kind == ModuleKind::StorePE && conn.srcPort == 0)) {
      // LoadPE out0 or StorePE out0 connects to a memory address port
      if (dstInst.kind == ModuleKind::Memory) {
        const auto &mdef = memoryDefs[dstInst.defIdx];
        unsigned md = mdef.shape.isDynamic() ? 64 : mdef.shape.getSize();
        if (md == 0) md = 64;
        addrWidth = ceilLog2(md);
      } else if (dstInst.kind == ModuleKind::ExtMemory) {
        const auto &mdef = extMemoryDefs[dstInst.defIdx];
        unsigned md = mdef.shape.isDynamic() ? 64 : mdef.shape.getSize();
        if (md == 0) md = 64;
        addrWidth = ceilLog2(md);
      }
      if (addrWidth > 0) {
        if (srcInst.kind == ModuleKind::LoadPE) {
          dataWidth = getDataWidthBits(loadPEDefs[srcInst.defIdx].dataType);
          peName = srcInst.name;
        } else {
          dataWidth = getDataWidthBits(storePEDefs[srcInst.defIdx].dataType);
          peName = srcInst.name;
        }
        if (dataWidth == 0) dataWidth = 1;
        if (dataWidth < addrWidth) {
          llvm::errs() << "error: exportSV: " << peName
                       << " DATA_WIDTH (" << dataWidth
                       << ") is too narrow to address " << dstInst.name
                       << " (needs " << addrWidth
                       << " bits); use a wider dataType\n";
          std::exit(1);
        }
      }
    }
  }

  // Determine once whether any instance produces error signals
  bool hasErrorPorts = false;
  for (const auto &inst : instances) {
    if (hasErrorOutput(inst.kind)) {
      hasErrorPorts = true;
      break;
    }
    if (inst.kind == ModuleKind::PE && peHasErrorOutput(peDefs[inst.defIdx])) {
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
    } else if (inst.kind == ModuleKind::AddTag) {
      instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::MapTag) {
      instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::PE) {
      const auto &def = peDefs[inst.defIdx];
      unsigned tw = def.inputPorts.size() > 0
                        ? getTagWidthBits(def.inputPorts[0])
                        : 0;
      unsigned peCfgBits = (tw > 0) ? def.outputPorts.size() * tw : 0;
      bool peHasStream = (def.singleOp == "dataflow.stream");
      if (!peHasStream && !def.bodyMLIR.empty()) {
        auto bodyOps = extractBodyMLIROps(def.bodyMLIR);
        peHasStream = (bodyOps.size() == 1 && bodyOps[0] == "dataflow.stream");
      }
      if (peHasStream)
        peCfgBits += 5;
      peCfgBits += countCmpOps(def) * 4;
      if (peCfgBits > 0)
        instanceDerived.insert(inst.name + "_cfg_data");
      if (peHasErrorOutput(def)) {
        instanceDerived.insert(inst.name + "_error_valid");
        instanceDerived.insert(inst.name + "_error_code");
      }
    } else if (inst.kind == ModuleKind::ConstantPE) {
      instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::LoadPE) {
      const auto &def = loadPEDefs[inst.defIdx];
      if (def.hwType == HardwareType::TagOverwrite && def.tagWidth > 0)
        instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::StorePE) {
      const auto &def = storePEDefs[inst.defIdx];
      if (def.hwType == HardwareType::TagOverwrite && def.tagWidth > 0)
        instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::TemporalSwitch) {
      instanceDerived.insert(inst.name + "_cfg_data");
    } else if (inst.kind == ModuleKind::TemporalPE) {
      instanceDerived.insert(inst.name + "_cfg_data");
    }
    // Memory and ExtMemory have no config ports (CONFIG_WIDTH = 0)
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

  // Track which operation dialects are used (for copying operation SV files)
  std::set<std::string> usedDialects;

  if (!templateDir.empty()) {
    // Copy Common/ header (always needed)
    copyTemplateFile(templateDir, "Common/fabric_common.svh", libDir.str().str());

    // Collect which module kinds are present in instances
    std::set<ModuleKind> usedKinds;
    for (const auto &inst : instances)
      usedKinds.insert(inst.kind);

    // Copy only the Fabric/ template files actually needed by the design.
    // PE and TemporalPE are generated per-instance (not directly instantiated
    // from the template), so they do not need a template copy in the output.
    static const std::pair<ModuleKind, const char *> kindToTemplate[] = {
        {ModuleKind::Fifo, "Fabric/fabric_fifo.sv"},
        {ModuleKind::Switch, "Fabric/fabric_switch.sv"},
        {ModuleKind::AddTag, "Fabric/fabric_add_tag.sv"},
        {ModuleKind::DelTag, "Fabric/fabric_del_tag.sv"},
        {ModuleKind::MapTag, "Fabric/fabric_map_tag.sv"},
        {ModuleKind::ConstantPE, "Fabric/fabric_pe_constant.sv"},
        {ModuleKind::LoadPE, "Fabric/fabric_pe_load.sv"},
        {ModuleKind::StorePE, "Fabric/fabric_pe_store.sv"},
        {ModuleKind::TemporalSwitch, "Fabric/fabric_temporal_sw.sv"},
        {ModuleKind::Memory, "Fabric/fabric_memory.sv"},
        {ModuleKind::ExtMemory, "Fabric/fabric_extmemory.sv"},
    };
    for (const auto &[kind, tmpl] : kindToTemplate) {
      if (usedKinds.count(kind))
        copyTemplateFile(templateDir, tmpl, libDir.str().str());
    }

    // For PE instances: generate body-filled customized modules
    for (const auto &inst : instances) {
      if (inst.kind == ModuleKind::PE) {
        const auto &def = peDefs[inst.defIdx];
        std::string customized = fillPETemplate(templateDir, def);

        // Replace module name to make it unique per instance
        std::string origName = "module " + def.name + "_pe";
        std::string newName = "module " + inst.name + "_pe";
        auto pos = customized.find(origName);
        if (pos != std::string::npos)
          customized.replace(pos, origName.size(), newName);

        llvm::SmallString<256> pePath(libDir);
        llvm::sys::path::append(pePath, inst.name + "_pe.sv");
        writeFile(pePath.str().str(), customized);

        // Track the dialect for operation module copying
        if (!def.singleOp.empty()) {
          auto dotPos = def.singleOp.find('.');
          if (dotPos != std::string::npos)
            usedDialects.insert(def.singleOp.substr(0, dotPos));
        }
        if (!def.bodyMLIR.empty()) {
          for (const auto &op : extractBodyMLIROps(def.bodyMLIR)) {
            auto dotPos = op.find('.');
            if (dotPos != std::string::npos)
              usedDialects.insert(op.substr(0, dotPos));
          }
        }
      }
    }

    // For TemporalPE instances: generate per-FU fabric_pe modules and
    // body-filled customized temporal_pe modules
    for (const auto &inst : instances) {
      if (inst.kind == ModuleKind::TemporalPE) {
        const auto &def = temporalPEDefs[inst.defIdx];

        // Generate customized fabric_pe modules for each FU type
        for (unsigned f = 0; f < def.fuPEDefIndices.size(); ++f) {
          const auto &fuDef = peDefs[def.fuPEDefIndices[f]];
          std::string fuModName = inst.name + "_fu" + std::to_string(f) + "_pe";
          std::string fuCustomized = fillPETemplate(templateDir, fuDef);

          // Replace module name to match FU module name
          std::string origPeName = "module " + fuDef.name + "_pe";
          std::string newPeName = "module " + fuModName;
          auto pePos = fuCustomized.find(origPeName);
          if (pePos != std::string::npos)
            fuCustomized.replace(pePos, origPeName.size(), newPeName);

          llvm::SmallString<256> fuPath(libDir);
          llvm::sys::path::append(fuPath, fuModName + ".sv");
          writeFile(fuPath.str().str(), fuCustomized);
        }

        // Generate the temporal PE module with per-port widths
        std::string customized =
            genFullTemporalPESV(templateDir, def, peDefs, inst.name);

        llvm::SmallString<256> tpePath(libDir);
        llvm::sys::path::append(tpePath, inst.name + "_temporal_pe.sv");
        writeFile(tpePath.str().str(), customized);

        // Track dialects used by FU types (singleOp and bodyMLIR)
        for (unsigned idx : def.fuPEDefIndices) {
          const auto &fuDef = peDefs[idx];
          if (!fuDef.singleOp.empty()) {
            auto dotPos = fuDef.singleOp.find('.');
            if (dotPos != std::string::npos)
              usedDialects.insert(fuDef.singleOp.substr(0, dotPos));
          }
          if (!fuDef.bodyMLIR.empty()) {
            for (const auto &op : extractBodyMLIROps(fuDef.bodyMLIR)) {
              auto dotPos = op.find('.');
              if (dotPos != std::string::npos)
                usedDialects.insert(op.substr(0, dotPos));
            }
          }
        }
      }
    }

    // Copy operation module SV files for used dialects
    // Map dialect names to directory names
    static const std::map<std::string, std::string> dialectDirs = {
        {"arith", "Arith"},
        {"math", "Math"},
        {"llvm", "LLVM"},
        {"dataflow", "Dataflow"}};

    for (const auto &dialect : usedDialects) {
      auto it = dialectDirs.find(dialect);
      if (it == dialectDirs.end())
        continue;

      // Collect all operation names used by PEs and TemporalPE FU types
      std::set<std::string> opsForDialect;
      auto addOpsFromDef = [&](const PEDef &def) {
        if (!def.singleOp.empty()) {
          auto dotPos = def.singleOp.find('.');
          if (dotPos != std::string::npos && def.singleOp.substr(0, dotPos) == dialect)
            opsForDialect.insert(def.singleOp);
        }
        if (!def.bodyMLIR.empty()) {
          for (const auto &op : extractBodyMLIROps(def.bodyMLIR)) {
            auto dotPos = op.find('.');
            if (dotPos != std::string::npos && op.substr(0, dotPos) == dialect)
              opsForDialect.insert(op);
          }
        }
      };
      for (const auto &inst : instances) {
        if (inst.kind == ModuleKind::PE) {
          addOpsFromDef(peDefs[inst.defIdx]);
        } else if (inst.kind == ModuleKind::TemporalPE) {
          const auto &tpeDef = temporalPEDefs[inst.defIdx];
          for (unsigned idx : tpeDef.fuPEDefIndices)
            addOpsFromDef(peDefs[idx]);
        }
      }
      for (const auto &op : opsForDialect) {
        std::string svFile = opToSVModule(op) + ".sv";
        copyTemplateFile(templateDir, it->second + "/" + svFile,
                         libDir.str().str());
      }
    }
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
    } else if (inst.kind == ModuleKind::AddTag) {
      const auto &def = addTagDefs[inst.defIdx];
      unsigned tw = getDataWidthBits(def.tagType);
      instCfgPorts.push_back({inst.name + "_cfg_data", tw});
    } else if (inst.kind == ModuleKind::MapTag) {
      const auto &def = mapTagDefs[inst.defIdx];
      unsigned itw = getDataWidthBits(def.inputTagType);
      unsigned otw = getDataWidthBits(def.outputTagType);
      unsigned entryWidth = 1 + itw + otw;
      instCfgPorts.push_back(
          {inst.name + "_cfg_data", def.tableSize * entryWidth});
    } else if (inst.kind == ModuleKind::PE) {
      const auto &def = peDefs[inst.defIdx];
      unsigned tw = def.inputPorts.size() > 0
                        ? getTagWidthBits(def.inputPorts[0])
                        : 0;
      unsigned cfgBits = (tw > 0) ? def.outputPorts.size() * tw : 0;
      bool hasStreamCfg = (def.singleOp == "dataflow.stream");
      if (!hasStreamCfg && !def.bodyMLIR.empty()) {
        auto bodyOps = extractBodyMLIROps(def.bodyMLIR);
        hasStreamCfg =
            (bodyOps.size() == 1 && bodyOps[0] == "dataflow.stream");
      }
      if (hasStreamCfg)
        cfgBits += 5;
      cfgBits += countCmpOps(def) * 4;
      if (cfgBits > 0)
        instCfgPorts.push_back({inst.name + "_cfg_data", cfgBits});
    } else if (inst.kind == ModuleKind::ConstantPE) {
      const auto &def = constantPEDefs[inst.defIdx];
      unsigned dw = getDataWidthBits(def.outputType);
      unsigned tw = getTagWidthBits(def.outputType);
      unsigned cfgBits = (tw > 0) ? dw + tw : dw;
      instCfgPorts.push_back({inst.name + "_cfg_data", cfgBits});
    } else if (inst.kind == ModuleKind::LoadPE) {
      const auto &def = loadPEDefs[inst.defIdx];
      if (def.interface == InterfaceCategory::Tagged &&
          def.hwType == HardwareType::TagOverwrite && def.tagWidth > 0)
        instCfgPorts.push_back({inst.name + "_cfg_data", def.tagWidth});
    } else if (inst.kind == ModuleKind::StorePE) {
      const auto &def = storePEDefs[inst.defIdx];
      if (def.interface == InterfaceCategory::Tagged &&
          def.hwType == HardwareType::TagOverwrite && def.tagWidth > 0)
        instCfgPorts.push_back({inst.name + "_cfg_data", def.tagWidth});
    } else if (inst.kind == ModuleKind::TemporalSwitch) {
      const auto &def = temporalSwitchDefs[inst.defIdx];
      unsigned tw = getTagWidthBits(def.interfaceType);
      unsigned numConn = getNumConnected(def);
      unsigned entryWidth = 1 + tw + numConn;
      instCfgPorts.push_back(
          {inst.name + "_cfg_data", def.numRouteTable * entryWidth});
    } else if (inst.kind == ModuleKind::TemporalPE) {
      const auto &def = temporalPEDefs[inst.defIdx];
      unsigned tw = getTagWidthBits(def.interfaceType);
      unsigned numIn = 1, numOut = 1;
      if (!def.fuPEDefIndices.empty()) {
        numIn = peDefs[def.fuPEDefIndices[0]].inputPorts.size();
        numOut = peDefs[def.fuPEDefIndices[0]].outputPorts.size();
      }
      unsigned fuTypes = def.fuPEDefIndices.size();
      // log2Ceil helper
      auto log2Ceil = [](unsigned v) -> unsigned {
        if (v <= 1) return 0;
        unsigned bits = 0;
        v--;
        while (v > 0) { bits++; v >>= 1; }
        return bits;
      };
      unsigned regBits = (def.numRegisters > 0) ?
          (1 + log2Ceil(std::max(def.numRegisters, 2u))) : 0;
      unsigned fuSelBits = (fuTypes > 1) ? log2Ceil(fuTypes) : 0;
      unsigned resBits = regBits;
      unsigned resultWidth = resBits + tw;
      unsigned insnWidth = 1 + tw + fuSelBits + numIn * regBits + numOut * resultWidth;
      unsigned totalFuCmpBits = 0;
      for (unsigned idx : def.fuPEDefIndices)
        totalFuCmpBits += countCmpOps(peDefs[idx]) * 4;
      unsigned cfgBits = totalFuCmpBits + def.numInstructions * insnWidth;
      if (cfgBits > 0)
        instCfgPorts.push_back({inst.name + "_cfg_data", cfgBits});
    } else if (inst.kind == ModuleKind::Memory) {
      const auto &def = memoryDefs[inst.defIdx];
      unsigned tw = 0;
      if (def.ldCount > 1 || def.stCount > 1) {
        unsigned maxCount = std::max(def.ldCount, def.stCount);
        unsigned tagBits = 1;
        while ((1u << tagBits) < maxCount)
          ++tagBits;
        tw = tagBits;
      }
      unsigned cfgBits = def.numRegion * (1 + tw + ((tw > 0) ? tw + 1 : 0) + DEFAULT_ADDR_WIDTH);
      if (cfgBits > 0)
        instCfgPorts.push_back({inst.name + "_cfg_data", cfgBits});
    } else if (inst.kind == ModuleKind::ExtMemory) {
      const auto &def = extMemoryDefs[inst.defIdx];
      unsigned tw = 0;
      if (def.ldCount > 1 || def.stCount > 1) {
        unsigned maxCount = std::max(def.ldCount, def.stCount);
        unsigned tagBits = 1;
        while ((1u << tagBits) < maxCount)
          ++tagBits;
        tw = tagBits;
      }
      unsigned cfgBits = def.numRegion * (1 + tw + ((tw > 0) ? tw + 1 : 0) + DEFAULT_ADDR_WIDTH);
      if (cfgBits > 0)
        instCfgPorts.push_back({inst.name + "_cfg_data", cfgBits});
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

  // Helper: compute the actual SV port data width for an instance input port.
  // For most modules this equals getDataWidthBits(type) + getTagWidthBits(type).
  // ConstantPE is special: its SV in_data port is SAFE_PW = DATA_WIDTH + TAG_WIDTH
  // (matching the output payload width), not the semantic control-token width.
  auto getInputSVWidth = [&](unsigned instIdx, unsigned port) -> unsigned {
    const auto &inst = instances[instIdx];
    if (inst.kind == ModuleKind::ConstantPE && port == 0) {
      const auto &cdef = constantPEDefs[inst.defIdx];
      unsigned w = getDataWidthBits(cdef.outputType) + getTagWidthBits(cdef.outputType);
      return w > 0 ? w : 1;
    }
    Type pt = getInstanceInputType(instIdx, port);
    return getDataWidthBits(pt) + getTagWidthBits(pt);
  };

  // Pre-compute per-instance payload dimensions for modules with packed-array
  // Wire declarations for internal connections
  for (size_t i = 0; i < instances.size(); ++i) {
    const auto &inst = instances[i];
    unsigned numIn = getInstanceInputCount(i);
    unsigned numOut = getInstanceOutputCount(i);

    // Per-port semantic widths for all instances.
    for (unsigned p = 0; p < numIn; ++p) {
      unsigned w = getInputSVWidth(i, p);
      if (w == 0) w = 1;
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
      if (w == 0) w = 1;
      top << "  logic " << inst.name << "_out" << p << "_valid;\n";
      top << "  logic " << inst.name << "_out" << p << "_ready;\n";
      if (w > 1)
        top << "  logic [" << (w-1) << ":0] " << inst.name << "_out" << p << "_data;\n";
      else
        top << "  logic " << inst.name << "_out" << p << "_data;\n";
    }

    // Per-instance error signals for modules that support them
    if (hasErrorOutput(inst.kind) ||
        (inst.kind == ModuleKind::PE && peHasErrorOutput(peDefs[inst.defIdx]))) {
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
    case ModuleKind::AddTag: {
      const auto &def = addTagDefs[inst.defIdx];
      top << "  fabric_add_tag #(\n" << genAddTagParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in_data(" << inst.name << "_in0_data),\n";
      top << "    .out_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out_data(" << inst.name << "_out0_data),\n";
      top << "    .cfg_data(" << inst.name << "_cfg_data)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::DelTag: {
      const auto &def = delTagDefs[inst.defIdx];
      top << "  fabric_del_tag #(\n" << genDelTagParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in_data(" << inst.name << "_in0_data),\n";
      top << "    .out_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out_data(" << inst.name << "_out0_data)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::MapTag: {
      const auto &def = mapTagDefs[inst.defIdx];
      top << "  fabric_map_tag #(\n" << genMapTagParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in_data(" << inst.name << "_in0_data),\n";
      top << "    .out_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out_data(" << inst.name << "_out0_data),\n";
      top << "    .cfg_data(" << inst.name << "_cfg_data),\n";
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::PE: {
      const auto &def = peDefs[inst.defIdx];
      unsigned numIn = def.inputPorts.size();
      unsigned numOut = def.outputPorts.size();
      bool peHasErr = peHasErrorOutput(def);
      // Per-port instantiation (matches genFullPESV per-port interface)
      top << "  " << inst.name << "_pe "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      for (unsigned p = 0; p < numIn; ++p) {
        top << "    .in" << p << "_valid(" << inst.name << "_in" << p << "_valid),\n";
        top << "    .in" << p << "_ready(" << inst.name << "_in" << p << "_ready),\n";
        top << "    .in" << p << "_data(" << inst.name << "_in" << p << "_data),\n";
      }
      for (unsigned p = 0; p < numOut; ++p) {
        top << "    .out" << p << "_valid(" << inst.name << "_out" << p << "_valid),\n";
        top << "    .out" << p << "_ready(" << inst.name << "_out" << p << "_ready),\n";
        top << "    .out" << p << "_data(" << inst.name << "_out" << p << "_data),\n";
      }
      unsigned tw = numIn > 0 ? getTagWidthBits(def.inputPorts[0]) : 0;
      unsigned cfgBits = (tw > 0) ? numOut * tw : 0;
      bool hasStreamCfg = (def.singleOp == "dataflow.stream");
      if (!hasStreamCfg && !def.bodyMLIR.empty()) {
        auto bodyOps = extractBodyMLIROps(def.bodyMLIR);
        hasStreamCfg =
            (bodyOps.size() == 1 && bodyOps[0] == "dataflow.stream");
      }
      if (hasStreamCfg)
        cfgBits += 5;
      cfgBits += countCmpOps(def) * 4;
      if (cfgBits > 0)
        top << "    .cfg_data(" << inst.name << "_cfg_data)"
            << (peHasErr ? "," : "") << "\n";
      else
        top << "    .cfg_data('0)"
            << (peHasErr ? "," : "") << "\n";
      if (peHasErr) {
        top << "    .error_valid(" << inst.name << "_error_valid),\n";
        top << "    .error_code(" << inst.name << "_error_code)\n";
      }
      top << "  );\n\n";
      break;
    }
    case ModuleKind::ConstantPE: {
      const auto &def = constantPEDefs[inst.defIdx];
      top << "  fabric_pe_constant #(\n" << genConstantPEParams(def)
          << "\n  ) " << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in_data(" << inst.name << "_in0_data),\n";
      top << "    .out_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out_data(" << inst.name << "_out0_data),\n";
      top << "    .cfg_data(" << inst.name << "_cfg_data)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::LoadPE: {
      const auto &def = loadPEDefs[inst.defIdx];
      top << "  fabric_pe_load #(\n" << genLoadPEParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in0_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in0_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in0_data(" << inst.name << "_in0_data),\n";
      top << "    .in1_valid(" << inst.name << "_in1_valid),\n";
      top << "    .in1_ready(" << inst.name << "_in1_ready),\n";
      top << "    .in1_data(" << inst.name << "_in1_data),\n";
      top << "    .in2_valid(" << inst.name << "_in2_valid),\n";
      top << "    .in2_ready(" << inst.name << "_in2_ready),\n";
      top << "    .in2_data(" << inst.name << "_in2_data),\n";
      top << "    .out0_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out0_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out0_data(" << inst.name << "_out0_data),\n";
      top << "    .out1_valid(" << inst.name << "_out1_valid),\n";
      top << "    .out1_ready(" << inst.name << "_out1_ready),\n";
      top << "    .out1_data(" << inst.name << "_out1_data),\n";
      if (def.interface == InterfaceCategory::Tagged &&
          def.hwType == HardwareType::TagOverwrite && def.tagWidth > 0)
        top << "    .cfg_data(" << inst.name << "_cfg_data)\n";
      else
        top << "    .cfg_data('0)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::StorePE: {
      const auto &def = storePEDefs[inst.defIdx];
      top << "  fabric_pe_store #(\n" << genStorePEParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      top << "    .in0_valid(" << inst.name << "_in0_valid),\n";
      top << "    .in0_ready(" << inst.name << "_in0_ready),\n";
      top << "    .in0_data(" << inst.name << "_in0_data),\n";
      top << "    .in1_valid(" << inst.name << "_in1_valid),\n";
      top << "    .in1_ready(" << inst.name << "_in1_ready),\n";
      top << "    .in1_data(" << inst.name << "_in1_data),\n";
      top << "    .in2_valid(" << inst.name << "_in2_valid),\n";
      top << "    .in2_ready(" << inst.name << "_in2_ready),\n";
      top << "    .in2_data(" << inst.name << "_in2_data),\n";
      top << "    .out0_valid(" << inst.name << "_out0_valid),\n";
      top << "    .out0_ready(" << inst.name << "_out0_ready),\n";
      top << "    .out0_data(" << inst.name << "_out0_data),\n";
      top << "    .out1_valid(" << inst.name << "_out1_valid),\n";
      top << "    .out1_ready(" << inst.name << "_out1_ready),\n";
      top << "    .out1_data(" << inst.name << "_out1_data),\n";
      if (def.interface == InterfaceCategory::Tagged &&
          def.hwType == HardwareType::TagOverwrite && def.tagWidth > 0)
        top << "    .cfg_data(" << inst.name << "_cfg_data)\n";
      else
        top << "    .cfg_data('0)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::TemporalSwitch: {
      const auto &def = temporalSwitchDefs[inst.defIdx];
      top << "  fabric_temporal_sw #(\n" << genTemporalSwitchParams(def)
          << "\n  ) " << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
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
      top << "    .cfg_data(" << inst.name << "_cfg_data),\n";
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::TemporalPE: {
      const auto &def = temporalPEDefs[inst.defIdx];
      unsigned numIn = 1, numOut = 1;
      if (!def.fuPEDefIndices.empty()) {
        numIn = peDefs[def.fuPEDefIndices[0]].inputPorts.size();
        numOut = peDefs[def.fuPEDefIndices[0]].outputPorts.size();
      }
      // Per-port instantiation (no parameter block; localparams baked in)
      top << "  " << inst.name << "_temporal_pe "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      for (unsigned p = 0; p < numIn; ++p) {
        top << "    .in" << p << "_valid(" << inst.name << "_in" << p << "_valid),\n";
        top << "    .in" << p << "_ready(" << inst.name << "_in" << p << "_ready),\n";
        top << "    .in" << p << "_data(" << inst.name << "_in" << p << "_data),\n";
      }
      for (unsigned p = 0; p < numOut; ++p) {
        top << "    .out" << p << "_valid(" << inst.name << "_out" << p << "_valid),\n";
        top << "    .out" << p << "_ready(" << inst.name << "_out" << p << "_ready),\n";
        top << "    .out" << p << "_data(" << inst.name << "_out" << p << "_data),\n";
      }
      top << "    .cfg_data(" << inst.name << "_cfg_data),\n";
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::Memory: {
      const auto &def = memoryDefs[inst.defIdx];
      top << "  fabric_memory #(\n" << genMemoryParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      // Named port connections mapped from indexed wires
      {
        unsigned inPort = 0;
        unsigned outPort = 0;
        // ld_addr input
        if (def.ldCount > 0) {
          top << "    .ld_addr_valid(" << inst.name << "_in" << inPort << "_valid),\n";
          top << "    .ld_addr_ready(" << inst.name << "_in" << inPort << "_ready),\n";
          top << "    .ld_addr_data(" << inst.name << "_in" << inPort << "_data),\n";
          inPort++;
        } else {
          top << "    .ld_addr_valid(1'b0), .ld_addr_ready(), .ld_addr_data('0),\n";
        }
        // st_addr + st_data inputs
        if (def.stCount > 0) {
          top << "    .st_addr_valid(" << inst.name << "_in" << inPort << "_valid),\n";
          top << "    .st_addr_ready(" << inst.name << "_in" << inPort << "_ready),\n";
          top << "    .st_addr_data(" << inst.name << "_in" << inPort << "_data),\n";
          inPort++;
          top << "    .st_data_valid(" << inst.name << "_in" << inPort << "_valid),\n";
          top << "    .st_data_ready(" << inst.name << "_in" << inPort << "_ready),\n";
          top << "    .st_data_data(" << inst.name << "_in" << inPort << "_data),\n";
          inPort++;
        } else {
          top << "    .st_addr_valid(1'b0), .st_addr_ready(), .st_addr_data('0),\n";
          top << "    .st_data_valid(1'b0), .st_data_ready(), .st_data_data('0),\n";
        }
        // memref output (non-private)
        if (!def.isPrivate) {
          top << "    .memref_valid(" << inst.name << "_out" << outPort << "_valid),\n";
          top << "    .memref_ready(" << inst.name << "_out" << outPort << "_ready),\n";
          outPort++;
        } else {
          top << "    .memref_valid(), .memref_ready(1'b1),\n";
        }
        // ld_data + ld_done outputs
        if (def.ldCount > 0) {
          top << "    .ld_data_valid(" << inst.name << "_out" << outPort << "_valid),\n";
          top << "    .ld_data_ready(" << inst.name << "_out" << outPort << "_ready),\n";
          top << "    .ld_data_data(" << inst.name << "_out" << outPort << "_data),\n";
          outPort++;
          top << "    .ld_done_valid(" << inst.name << "_out" << outPort << "_valid),\n";
          top << "    .ld_done_ready(" << inst.name << "_out" << outPort << "_ready),\n";
          top << "    .ld_done_data(" << inst.name << "_out" << outPort << "_data),\n";
          outPort++;
        } else {
          top << "    .ld_data_valid(), .ld_data_ready(1'b1), .ld_data_data(),\n";
          top << "    .ld_done_valid(), .ld_done_ready(1'b1), .ld_done_data(),\n";
        }
        // st_done output
        if (def.stCount > 0) {
          top << "    .st_done_valid(" << inst.name << "_out" << outPort << "_valid),\n";
          top << "    .st_done_ready(" << inst.name << "_out" << outPort << "_ready),\n";
          top << "    .st_done_data(" << inst.name << "_out" << outPort << "_data),\n";
          outPort++;
        } else {
          top << "    .st_done_valid(), .st_done_ready(1'b1), .st_done_data(),\n";
        }
      }
      {
        unsigned tw = 0;
        if (def.ldCount > 1 || def.stCount > 1) {
          unsigned maxCount = std::max(def.ldCount, def.stCount);
          unsigned tagBits = 1;
          while ((1u << tagBits) < maxCount)
            ++tagBits;
          tw = tagBits;
        }
        unsigned cfgBits = def.numRegion * (1 + tw + ((tw > 0) ? tw + 1 : 0) + DEFAULT_ADDR_WIDTH);
        if (cfgBits > 0)
          top << "    .cfg_data(" << inst.name << "_cfg_data),\n";
        else
          top << "    .cfg_data('0),\n";
      }
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
      top << "  );\n\n";
      break;
    }
    case ModuleKind::ExtMemory: {
      const auto &def = extMemoryDefs[inst.defIdx];
      top << "  fabric_extmemory #(\n" << genExtMemoryParams(def) << "\n  ) "
          << inst.name << " (\n";
      top << "    .clk(clk), .rst_n(rst_n),\n";
      // Named port connections mapped from indexed wires
      {
        unsigned inPort = 0;
        unsigned outPort = 0;
        // memref_bind input (always port 0)
        top << "    .memref_bind_valid(" << inst.name << "_in" << inPort << "_valid),\n";
        top << "    .memref_bind_ready(" << inst.name << "_in" << inPort << "_ready),\n";
        top << "    .memref_bind_data(" << inst.name << "_in" << inPort << "_data),\n";
        inPort++;
        // ld_addr input
        if (def.ldCount > 0) {
          top << "    .ld_addr_valid(" << inst.name << "_in" << inPort << "_valid),\n";
          top << "    .ld_addr_ready(" << inst.name << "_in" << inPort << "_ready),\n";
          top << "    .ld_addr_data(" << inst.name << "_in" << inPort << "_data),\n";
          inPort++;
        } else {
          top << "    .ld_addr_valid(1'b0), .ld_addr_ready(), .ld_addr_data('0),\n";
        }
        // st_addr + st_data inputs
        if (def.stCount > 0) {
          top << "    .st_addr_valid(" << inst.name << "_in" << inPort << "_valid),\n";
          top << "    .st_addr_ready(" << inst.name << "_in" << inPort << "_ready),\n";
          top << "    .st_addr_data(" << inst.name << "_in" << inPort << "_data),\n";
          inPort++;
          top << "    .st_data_valid(" << inst.name << "_in" << inPort << "_valid),\n";
          top << "    .st_data_ready(" << inst.name << "_in" << inPort << "_ready),\n";
          top << "    .st_data_data(" << inst.name << "_in" << inPort << "_data),\n";
          inPort++;
        } else {
          top << "    .st_addr_valid(1'b0), .st_addr_ready(), .st_addr_data('0),\n";
          top << "    .st_data_valid(1'b0), .st_data_ready(), .st_data_data('0),\n";
        }
        // ld_data + ld_done outputs
        if (def.ldCount > 0) {
          top << "    .ld_data_valid(" << inst.name << "_out" << outPort << "_valid),\n";
          top << "    .ld_data_ready(" << inst.name << "_out" << outPort << "_ready),\n";
          top << "    .ld_data_data(" << inst.name << "_out" << outPort << "_data),\n";
          outPort++;
          top << "    .ld_done_valid(" << inst.name << "_out" << outPort << "_valid),\n";
          top << "    .ld_done_ready(" << inst.name << "_out" << outPort << "_ready),\n";
          top << "    .ld_done_data(" << inst.name << "_out" << outPort << "_data),\n";
          outPort++;
        } else {
          top << "    .ld_data_valid(), .ld_data_ready(1'b1), .ld_data_data(),\n";
          top << "    .ld_done_valid(), .ld_done_ready(1'b1), .ld_done_data(),\n";
        }
        // st_done output
        if (def.stCount > 0) {
          top << "    .st_done_valid(" << inst.name << "_out" << outPort << "_valid),\n";
          top << "    .st_done_ready(" << inst.name << "_out" << outPort << "_ready),\n";
          top << "    .st_done_data(" << inst.name << "_out" << outPort << "_data),\n";
          outPort++;
        } else {
          top << "    .st_done_valid(), .st_done_ready(1'b1), .st_done_data(),\n";
        }
      }
      {
        unsigned tw = 0;
        if (def.ldCount > 1 || def.stCount > 1) {
          unsigned maxCount = std::max(def.ldCount, def.stCount);
          unsigned tagBits = 1;
          while ((1u << tagBits) < maxCount)
            ++tagBits;
          tw = tagBits;
        }
        unsigned cfgBits = def.numRegion * (1 + tw + ((tw > 0) ? tw + 1 : 0) + DEFAULT_ADDR_WIDTH);
        if (cfgBits > 0)
          top << "    .cfg_data(" << inst.name << "_cfg_data),\n";
        else
          top << "    .cfg_data('0),\n";
      }
      top << "    .error_valid(" << inst.name << "_error_valid),\n";
      top << "    .error_code(" << inst.name << "_error_code)\n";
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

  // Width-adapting data assignment: repack tag+data when src and dst use
  // different data widths within the same tag structure. Needed when per-port
  // temporal PE widths differ from module-level interface widths.
  auto emitDataAssign = [&](const std::string &dst, unsigned dstDW,
                            unsigned dstTW, const std::string &src,
                            unsigned srcDW, unsigned srcTW) {
    unsigned srcTotal = srcDW + srcTW;
    unsigned dstTotal = dstDW + dstTW;
    if (srcTotal == dstTotal || dstTotal == 0 || srcTotal == 0) {
      top << "  assign " << dst << " = " << src << ";\n";
    } else if (srcTW == 0 || dstTW == 0) {
      if (srcTotal > dstTotal)
        top << "  assign " << dst << " = " << src << "["
            << (dstTotal - 1) << ":0];\n";
      else
        top << "  assign " << dst << " = {"
            << (dstTotal - srcTotal) << "'b0, " << src << "};\n";
    } else if (srcDW > dstDW) {
      top << "  assign " << dst << " = {" << src << "["
          << (srcDW + srcTW - 1) << ":" << srcDW << "], " << src << "["
          << (dstDW - 1) << ":0]};\n";
    } else {
      top << "  assign " << dst << " = {" << src << "["
          << (srcDW + srcTW - 1) << ":" << srcDW << "], "
          << (dstDW - srcDW) << "'b0, " << src << "["
          << (srcDW - 1) << ":0]};\n";
    }
  };

  // Wire connections: module inputs to instances
  for (const auto &conn : inputConns) {
    const auto &port = ports[conn.portIdx];
    const auto &inst = instances[conn.instIdx];
    std::string sinkReady = inst.name + "_in" + std::to_string(conn.dstPort) + "_ready";
    top << "  assign " << inst.name << "_in" << conn.dstPort
        << "_valid = " << port.name << "_valid;\n";
    unsigned srcDW = getDataWidthBits(port.type);
    unsigned srcTW = getTagWidthBits(port.type);
    unsigned srcTotal = srcDW + srcTW;
    if (srcTotal == 0) srcTotal = 1;
    // Compute the actual SV port width for the destination instance input.
    // For most modules this equals the semantic type width (data + tag bits).
    // ConstantPE is special: its input port 0 has a narrow semantic type
    // (e.g. i1 control token = 1 bit), but the SV module's in_data port
    // uses SAFE_PW = DATA_WIDTH + TAG_WIDTH (matching output payload width,
    // e.g. 36 bits for tagged f32 with 4-bit tag).  Without getInputSVWidth,
    // dstSVWidth would be computed from the semantic type (1 bit), causing a
    // width mismatch when connecting a wide module port to a narrow wire.
    unsigned dstSVWidth = getInputSVWidth(conn.instIdx, conn.dstPort);
    if (dstSVWidth == 0) dstSVWidth = 1;
    Type dstType = getInstanceInputType(conn.instIdx, conn.dstPort);
    unsigned dstDW = getDataWidthBits(dstType);
    unsigned dstTW = getTagWidthBits(dstType);
    if (srcTotal != dstSVWidth) {
      // Use tag-aware slicing when both sides carry tags
      if (srcTW > 0 && dstTW > 0) {
        emitDataAssign(inst.name + "_in" + std::to_string(conn.dstPort) + "_data",
                       dstDW, dstTW, port.name + "_data", srcDW, srcTW);
      } else {
        emitDataAssign(inst.name + "_in" + std::to_string(conn.dstPort) + "_data",
                       dstSVWidth, 0, port.name + "_data", srcTotal, 0);
      }
    } else {
      top << "  assign " << inst.name << "_in" << conn.dstPort
          << "_data = " << port.name << "_data;\n";
    }
    readySources["%" + port.name].push_back(sinkReady);
  }
  // Emit ready for module input ports; drive unconnected to 0.
  // Validation (CPL_FANOUT_MODULE_BOUNDARY) guarantees at most one sink.
  for (const auto *p : streamInputs) {
    std::string key = "%" + p->name;
    auto it = readySources.find(key);
    if (it == readySources.end()) {
      top << "  assign " << p->name << "_ready = 1'b0;\n";
    } else {
      top << "  assign " << p->name << "_ready = " << it->second[0] << ";\n";
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
    Type srcType = getInstanceOutputType(conn.instIdx, conn.srcPort);
    unsigned srcDW = getDataWidthBits(srcType);
    unsigned srcTW = getTagWidthBits(srcType);
    unsigned dstDW = getDataWidthBits(port.type);
    unsigned dstTW = getTagWidthBits(port.type);
    if ((srcDW + srcTW) != (dstDW + dstTW)) {
      emitDataAssign(port.name + "_data", dstDW, dstTW,
                     srcKey + "_data", srcDW, srcTW);
    } else {
      top << "  assign " << port.name << "_data = " << srcKey << "_data;\n";
    }
    readySources[srcKey].push_back(port.name + "_ready");
  }

  // Wire connections: internal (instance to instance)
  for (const auto &conn : internalConns) {
    const auto &srcInst = instances[conn.srcInst];
    const auto &dstInst = instances[conn.dstInst];
    std::string srcKey = srcInst.name + "_out" + std::to_string(conn.srcPort);
    top << "  assign " << dstInst.name << "_in" << conn.dstPort
        << "_valid = " << srcKey << "_valid;\n";
    Type srcType2 = getInstanceOutputType(conn.srcInst, conn.srcPort);
    unsigned sDW = getDataWidthBits(srcType2);
    unsigned sTW = getTagWidthBits(srcType2);
    unsigned sTotal = sDW + sTW;
    if (sTotal == 0) sTotal = 1;
    unsigned dTotal = getInputSVWidth(conn.dstInst, conn.dstPort);
    if (dTotal == 0) dTotal = 1;
    if (sTotal != dTotal) {
      Type dstType2 = getInstanceInputType(conn.dstInst, conn.dstPort);
      unsigned dDW = getDataWidthBits(dstType2);
      unsigned dTW = getTagWidthBits(dstType2);
      // Use semantic tag-aware widths when both sides have tag info
      if (sTW > 0 && dTW > 0) {
        emitDataAssign(dstInst.name + "_in" + std::to_string(conn.dstPort) + "_data",
                       dDW, dTW, srcKey + "_data", sDW, sTW);
      } else {
        emitDataAssign(dstInst.name + "_in" + std::to_string(conn.dstPort) + "_data",
                       dTotal, 0, srcKey + "_data", sTotal, 0);
      }
    } else {
      top << "  assign " << dstInst.name << "_in" << conn.dstPort
          << "_data = " << srcKey << "_data;\n";
    }
    readySources[srcKey].push_back(
        dstInst.name + "_in" + std::to_string(conn.dstPort) + "_ready");
  }

  // Emit ready for instance output ports.
  // Validation (CPL_FANOUT_MODULE_INNER) guarantees at most one sink.
  for (const auto &[srcKey, sources] : readySources) {
    if (sources.size() == 1) {
      top << "  assign " << srcKey << "_ready = " << sources[0] << ";\n";
    } else {
      top << "  assign " << srcKey << "_ready = " << sources[0] << ";\n";
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
      bool instHasErr = hasErrorOutput(inst.kind) ||
          (inst.kind == ModuleKind::PE && peHasErrorOutput(peDefs[inst.defIdx]));
      if (instHasErr) {
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

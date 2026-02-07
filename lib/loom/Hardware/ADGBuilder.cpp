//===-- ADGBuilder.cpp - ADG Builder core implementation ----------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "ADGBuilderImpl.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// ADGBuilder
//===----------------------------------------------------------------------===//

ADGBuilder::ADGBuilder(const std::string &moduleName)
    : impl_(std::make_unique<Impl>()) {
  impl_->moduleName = moduleName;
}

ADGBuilder::~ADGBuilder() = default;

//===----------------------------------------------------------------------===//
// Module creation methods
//===----------------------------------------------------------------------===//

PEBuilder ADGBuilder::newPE(const std::string &name) {
  unsigned id = impl_->peDefs.size();
  impl_->peDefs.push_back({});
  impl_->peDefs.back().name = name;
  return PEBuilder(this, id);
}

ConstantPEBuilder ADGBuilder::newConstantPE(const std::string &name) {
  unsigned id = impl_->constantPEDefs.size();
  impl_->constantPEDefs.push_back({});
  impl_->constantPEDefs.back().name = name;
  return ConstantPEBuilder(this, id);
}

LoadPEBuilder ADGBuilder::newLoadPE(const std::string &name) {
  unsigned id = impl_->loadPEDefs.size();
  impl_->loadPEDefs.push_back({});
  impl_->loadPEDefs.back().name = name;
  return LoadPEBuilder(this, id);
}

StorePEBuilder ADGBuilder::newStorePE(const std::string &name) {
  unsigned id = impl_->storePEDefs.size();
  impl_->storePEDefs.push_back({});
  impl_->storePEDefs.back().name = name;
  return StorePEBuilder(this, id);
}

SwitchBuilder ADGBuilder::newSwitch(const std::string &name) {
  unsigned id = impl_->switchDefs.size();
  impl_->switchDefs.push_back({});
  impl_->switchDefs.back().name = name;
  return SwitchBuilder(this, id);
}

TemporalPEBuilder ADGBuilder::newTemporalPE(const std::string &name) {
  unsigned id = impl_->temporalPEDefs.size();
  impl_->temporalPEDefs.push_back({});
  impl_->temporalPEDefs.back().name = name;
  return TemporalPEBuilder(this, id);
}

TemporalSwitchBuilder ADGBuilder::newTemporalSwitch(const std::string &name) {
  unsigned id = impl_->temporalSwitchDefs.size();
  impl_->temporalSwitchDefs.push_back({});
  impl_->temporalSwitchDefs.back().name = name;
  return TemporalSwitchBuilder(this, id);
}

MemoryBuilder ADGBuilder::newMemory(const std::string &name) {
  unsigned id = impl_->memoryDefs.size();
  impl_->memoryDefs.push_back({});
  impl_->memoryDefs.back().name = name;
  return MemoryBuilder(this, id);
}

ExtMemoryBuilder ADGBuilder::newExtMemory(const std::string &name) {
  unsigned id = impl_->extMemoryDefs.size();
  impl_->extMemoryDefs.push_back({});
  impl_->extMemoryDefs.back().name = name;
  return ExtMemoryBuilder(this, id);
}

AddTagBuilder ADGBuilder::newAddTag(const std::string &name) {
  unsigned id = impl_->addTagDefs.size();
  impl_->addTagDefs.push_back({});
  impl_->addTagDefs.back().name = name;
  return AddTagBuilder(this, id);
}

MapTagBuilder ADGBuilder::newMapTag(const std::string &name) {
  unsigned id = impl_->mapTagDefs.size();
  impl_->mapTagDefs.push_back({});
  impl_->mapTagDefs.back().name = name;
  return MapTagBuilder(this, id);
}

DelTagBuilder ADGBuilder::newDelTag(const std::string &name) {
  unsigned id = impl_->delTagDefs.size();
  impl_->delTagDefs.push_back({});
  impl_->delTagDefs.back().name = name;
  return DelTagBuilder(this, id);
}

//===----------------------------------------------------------------------===//
// Clone (instantiation)
//===----------------------------------------------------------------------===//

InstanceHandle ADGBuilder::clone(PEHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::PE, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(SwitchHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::Switch, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(TemporalPEHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::TemporalPE, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(TemporalSwitchHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back(
      {ModuleKind::TemporalSwitch, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(MemoryHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::Memory, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(ExtMemoryHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::ExtMemory, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(ConstantPEHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back(
      {ModuleKind::ConstantPE, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(LoadPEHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::LoadPE, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(StorePEHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::StorePE, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(AddTagHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::AddTag, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(MapTagHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::MapTag, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(DelTagHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::DelTag, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(ModuleHandle source,
                                 const std::string &instanceName) {
  static const ModuleKind kindMap[] = {
      ModuleKind::PE,            ModuleKind::Switch,
      ModuleKind::TemporalPE,    ModuleKind::TemporalSwitch,
      ModuleKind::Memory,        ModuleKind::ExtMemory,
      ModuleKind::ConstantPE,    ModuleKind::LoadPE,
      ModuleKind::StorePE,       ModuleKind::AddTag,
      ModuleKind::MapTag,        ModuleKind::DelTag,
  };
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({kindMap[source.kind], source.id, instanceName});
  return InstanceHandle{id};
}

//===----------------------------------------------------------------------===//
// Internal connections
//===----------------------------------------------------------------------===//

void ADGBuilder::connect(InstanceHandle src, InstanceHandle dst) {
  assert(src.id < impl_->instances.size() && "invalid source instance");
  assert(dst.id < impl_->instances.size() && "invalid destination instance");
  impl_->internalConns.push_back({src.id, 0, dst.id, 0});
}

void ADGBuilder::connectPorts(InstanceHandle src, int srcPort,
                              InstanceHandle dst, int dstPort) {
  assert(src.id < impl_->instances.size() && "invalid source instance");
  assert(dst.id < impl_->instances.size() && "invalid destination instance");
  assert(srcPort >= 0 &&
         (unsigned)srcPort < impl_->getInstanceOutputCount(src.id) &&
         "source port out of range");
  assert(dstPort >= 0 &&
         (unsigned)dstPort < impl_->getInstanceInputCount(dst.id) &&
         "destination port out of range");
  impl_->internalConns.push_back(
      {src.id, srcPort, dst.id, dstPort});
}

//===----------------------------------------------------------------------===//
// Module I/O
//===----------------------------------------------------------------------===//

PortHandle ADGBuilder::addModuleInput(const std::string &name, Type type) {
  unsigned id = impl_->ports.size();
  impl_->ports.push_back({name, type, false, MemrefType::dynamic1D(Type::i32()),
                           true});
  return PortHandle{id};
}

PortHandle ADGBuilder::addModuleInput(const std::string &name,
                                      MemrefType memrefType) {
  unsigned id = impl_->ports.size();
  impl_->ports.push_back(
      {name, Type::index(), true, memrefType, true});
  return PortHandle{id};
}

PortHandle ADGBuilder::addModuleOutput(const std::string &name, Type type) {
  unsigned id = impl_->ports.size();
  impl_->ports.push_back({name, type, false, MemrefType::dynamic1D(Type::i32()),
                           false});
  return PortHandle{id};
}

PortHandle ADGBuilder::addModuleOutput(const std::string &name,
                                       MemrefType memrefType) {
  unsigned id = impl_->ports.size();
  impl_->ports.push_back(
      {name, Type::index(), true, memrefType, false});
  return PortHandle{id};
}

void ADGBuilder::connectToModuleInput(PortHandle port, InstanceHandle dst,
                                      int dstPort) {
  assert(port.id < impl_->ports.size() && "invalid port handle");
  assert(dst.id < impl_->instances.size() && "invalid instance handle");
  impl_->inputConns.push_back({port.id, dst.id, dstPort});
}

void ADGBuilder::connectToModuleOutput(InstanceHandle src, int srcPort,
                                       PortHandle port) {
  assert(src.id < impl_->instances.size() && "invalid instance handle");
  assert(port.id < impl_->ports.size() && "invalid port handle");
  impl_->outputConns.push_back({src.id, srcPort, port.id});
}

//===----------------------------------------------------------------------===//
// Topology
//===----------------------------------------------------------------------===//

MeshResult ADGBuilder::buildMesh(int rows, int cols, PEHandle peTemplate,
                                 SwitchHandle swTemplate, Topology topology) {
  assert(rows > 0 && "rows must be positive");
  assert(cols > 0 && "cols must be positive");

  MeshResult result;
  result.peGrid.resize(rows, std::vector<InstanceHandle>(cols));

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::string peName =
          "pe_" + std::to_string(r) + "_" + std::to_string(c);
      result.peGrid[r][c] = clone(peTemplate, peName);
    }
  }

  // Note: switches are not instantiated as part of the mesh grid. In MLIR SSA
  // form, all connections must form a DAG. The user creates unidirectional PE
  // connections using the returned peGrid handles. The topology parameter and
  // swTemplate record the intended topology for documentation/metadata.

  return result;
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

ValidationResult ADGBuilder::validateADG() {
  ValidationResult result;

  auto addError = [&](const std::string &code, const std::string &msg,
                      const std::string &loc = "") {
    result.errors.push_back({code, msg, loc});
    result.success = false;
  };

  // Validate switch definitions.
  for (size_t i = 0; i < impl_->switchDefs.size(); ++i) {
    const auto &sw = impl_->switchDefs[i];
    std::string loc = "switch @" + sw.name;
    if (sw.numIn > 32 || sw.numOut > 32)
      addError("COMP_SWITCH_PORT_LIMIT",
               "switch has more than 32 inputs or outputs", loc);
    if (!sw.connectivity.empty()) {
      if (sw.connectivity.size() != sw.numOut)
        addError("COMP_SWITCH_TABLE_SHAPE",
                 "connectivity_table row count != num_outputs", loc);
      for (size_t r = 0; r < sw.connectivity.size(); ++r) {
        if (sw.connectivity[r].size() != sw.numIn)
          addError("COMP_SWITCH_TABLE_SHAPE",
                   "connectivity_table column count != num_inputs", loc);
        bool hasOne = false;
        for (bool v : sw.connectivity[r]) if (v) hasOne = true;
        if (!hasOne)
          addError("COMP_SWITCH_ROW_EMPTY",
                   "connectivity row " + std::to_string(r) + " has no 1",
                   loc);
      }
      for (unsigned c = 0; c < sw.numIn; ++c) {
        bool hasOne = false;
        for (unsigned r = 0; r < sw.connectivity.size(); ++r)
          if (c < sw.connectivity[r].size() && sw.connectivity[r][c])
            hasOne = true;
        if (!hasOne)
          addError("COMP_SWITCH_COL_EMPTY",
                   "connectivity column " + std::to_string(c) + " has no 1",
                   loc);
      }
    }
  }

  // Validate temporal switch definitions.
  for (size_t i = 0; i < impl_->temporalSwitchDefs.size(); ++i) {
    const auto &ts = impl_->temporalSwitchDefs[i];
    std::string loc = "temporal_sw @" + ts.name;
    if (ts.numIn > 32 || ts.numOut > 32)
      addError("COMP_TEMPORAL_SW_PORT_LIMIT",
               "temporal switch has more than 32 inputs or outputs", loc);
    if (ts.numRouteTable < 1)
      addError("COMP_TEMPORAL_SW_NUM_ROUTE_TABLE",
               "num_route_table must be >= 1", loc);
    if (!ts.connectivity.empty()) {
      if (ts.connectivity.size() != ts.numOut)
        addError("COMP_TEMPORAL_SW_TABLE_SHAPE",
                 "connectivity_table row count != num_outputs", loc);
      for (size_t r = 0; r < ts.connectivity.size(); ++r) {
        bool hasOne = false;
        for (bool v : ts.connectivity[r]) if (v) hasOne = true;
        if (!hasOne)
          addError("COMP_TEMPORAL_SW_ROW_EMPTY",
                   "connectivity row " + std::to_string(r) + " has no 1",
                   loc);
      }
      for (unsigned c = 0; c < ts.numIn; ++c) {
        bool hasOne = false;
        for (unsigned r = 0; r < ts.connectivity.size(); ++r)
          if (c < ts.connectivity[r].size() && ts.connectivity[r][c])
            hasOne = true;
        if (!hasOne)
          addError("COMP_TEMPORAL_SW_COL_EMPTY",
                   "connectivity column " + std::to_string(c) + " has no 1",
                   loc);
      }
    }
  }

  // Validate temporal PE definitions.
  for (size_t i = 0; i < impl_->temporalPEDefs.size(); ++i) {
    const auto &tp = impl_->temporalPEDefs[i];
    std::string loc = "temporal_pe @" + tp.name;
    if (tp.numInstructions < 1)
      addError("COMP_TEMPORAL_PE_NUM_INSTRUCTION",
               "num_instruction must be >= 1", loc);
    if (tp.numRegisters > 0 && tp.regFifoDepth == 0)
      addError("COMP_TEMPORAL_PE_REG_FIFO_DEPTH",
               "reg_fifo_depth must be > 0 when num_register > 0", loc);
    if (tp.numRegisters == 0 && tp.regFifoDepth > 0)
      addError("COMP_TEMPORAL_PE_REG_FIFO_DEPTH",
               "reg_fifo_depth must be 0 when num_register == 0", loc);
    if (tp.fuPEDefIndices.empty())
      addError("COMP_TEMPORAL_PE_EMPTY_BODY",
               "temporal PE has no FU definitions", loc);
    if (!tp.shareModeB && tp.shareBufferSize > 0)
      addError("COMP_TEMPORAL_PE_OPERAND_BUFFER_MODE_A_HAS_SIZE",
               "operand_buffer_size set without enable_share_operand_buffer",
               loc);
    if (tp.shareModeB && tp.shareBufferSize == 0)
      addError("COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_MISSING",
               "operand_buffer_size missing with share_operand_buffer", loc);
    if (tp.shareModeB && tp.shareBufferSize > 8192)
      addError("COMP_TEMPORAL_PE_OPERAND_BUFFER_SIZE_RANGE",
               "operand_buffer_size out of range [1, 8192]", loc);
  }

  // Validate memory definitions.
  for (size_t i = 0; i < impl_->memoryDefs.size(); ++i) {
    const auto &mem = impl_->memoryDefs[i];
    std::string loc = "memory @" + mem.name;
    if (mem.ldCount == 0 && mem.stCount == 0)
      addError("COMP_MEMORY_PORTS_EMPTY",
               "ldCount and stCount are both 0", loc);
    if (mem.stCount > 0 && mem.lsqDepth < 1)
      addError("COMP_MEMORY_LSQ_MIN",
               "lsqDepth must be >= 1 when stCount > 0", loc);
    if (mem.stCount == 0 && mem.lsqDepth > 0)
      addError("COMP_MEMORY_LSQ_WITHOUT_STORE",
               "lsqDepth must be 0 when stCount == 0", loc);
    if (mem.shape.isDynamic())
      addError("COMP_MEMORY_STATIC_REQUIRED",
               "fabric.memory requires static memref shape", loc);
  }

  // Validate external memory definitions.
  for (size_t i = 0; i < impl_->extMemoryDefs.size(); ++i) {
    const auto &em = impl_->extMemoryDefs[i];
    std::string loc = "extmemory @" + em.name;
    if (em.ldCount == 0 && em.stCount == 0)
      addError("COMP_MEMORY_PORTS_EMPTY",
               "ldCount and stCount are both 0", loc);
    if (em.stCount > 0 && em.lsqDepth < 1)
      addError("COMP_MEMORY_LSQ_MIN",
               "lsqDepth must be >= 1 when stCount > 0", loc);
    if (em.stCount == 0 && em.lsqDepth > 0)
      addError("COMP_MEMORY_LSQ_WITHOUT_STORE",
               "lsqDepth must be 0 when stCount == 0", loc);
  }

  // Validate map_tag definitions.
  for (size_t i = 0; i < impl_->mapTagDefs.size(); ++i) {
    const auto &mt = impl_->mapTagDefs[i];
    std::string loc = "map_tag @" + mt.name;
    if (mt.tableSize < 1 || mt.tableSize > 256)
      addError("COMP_MAP_TAG_TABLE_SIZE",
               "table_size out of range [1, 256]", loc);
  }

  // Validate tag width range for add_tag and del_tag.
  for (size_t i = 0; i < impl_->addTagDefs.size(); ++i) {
    const auto &at = impl_->addTagDefs[i];
    std::string loc = "add_tag @" + at.name;
    if (at.tagType.getKind() == Type::IN &&
        (at.tagType.getWidth() < 1 || at.tagType.getWidth() > 16))
      addError("COMP_TAG_WIDTH_RANGE",
               "tag width outside [1, 16]", loc);
  }

  // Validate PE definitions.
  for (size_t i = 0; i < impl_->peDefs.size(); ++i) {
    const auto &pe = impl_->peDefs[i];
    std::string loc = "pe @" + pe.name;
    if (pe.inputPorts.empty())
      addError("COMP_PE_EMPTY_BODY", "PE has no input ports", loc);
    if (pe.outputPorts.empty())
      addError("COMP_PE_EMPTY_BODY", "PE has no output ports", loc);
    if (pe.bodyMLIR.empty() && pe.singleOp.empty())
      addError("COMP_PE_EMPTY_BODY", "PE has no body or operation", loc);
    // Check for mixed interface (some tagged, some not).
    bool hasTagged = false, hasNative = false;
    for (const auto &t : pe.inputPorts)
      (t.isTagged() ? hasTagged : hasNative) = true;
    for (const auto &t : pe.outputPorts)
      (t.isTagged() ? hasTagged : hasNative) = true;
    if (hasTagged && hasNative)
      addError("COMP_PE_MIXED_INTERFACE",
               "PE has mixed native and tagged ports", loc);
  }

  // Check for empty module body.
  if (impl_->instances.empty())
    addError("COMP_MODULE_EMPTY_BODY",
             "module has no instances", "module @" + impl_->moduleName);

  // Note: cycle detection is intentionally omitted. ADG architectures use
  // bidirectional switch connections (PE<->SW, SW<->SW) which form valid
  // routing cycles. The MLIR verifier handles structural correctness.

  return result;
}

//===----------------------------------------------------------------------===//
// Export
//===----------------------------------------------------------------------===//

void ADGBuilder::exportMLIR(const std::string &path) {
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

  std::string mlirText = impl_->generateMLIR();

  mlir::MLIRContext context;
  context.getDiagEngine().registerHandler([](mlir::Diagnostic &diag) {
    diag.print(llvm::errs());
    llvm::errs() << "\n";
    return mlir::success();
  });

  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  context.getOrLoadDialect<loom::fabric::FabricDialect>();
  context.getOrLoadDialect<circt::handshake::HandshakeDialect>();

  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
  if (!module) {
    llvm::errs() << "error: failed to parse generated MLIR\n";
    llvm::errs() << "--- generated MLIR ---\n" << mlirText << "---\n";
    std::exit(1);
  }

  if (failed(mlir::verify(*module))) {
    llvm::errs() << "error: generated MLIR failed verification\n";
    llvm::errs() << "--- generated MLIR ---\n" << mlirText << "---\n";
    std::exit(1);
  }

  std::error_code ec;
  llvm::raw_fd_ostream output(path, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "error: cannot write output file: " << path << "\n";
    llvm::errs() << ec.message() << "\n";
    std::exit(1);
  }

  module->print(output);
  output.flush();
}

//===----------------------------------------------------------------------===//
// Instance port queries
//===----------------------------------------------------------------------===//

unsigned ADGBuilder::Impl::getInstanceInputCount(unsigned instIdx) const {
  const auto &inst = instances[instIdx];
  switch (inst.kind) {
  case ModuleKind::PE:
    return peDefs[inst.defIdx].inputPorts.size();
  case ModuleKind::ConstantPE:
    // Constant PE has 1 input (control token).
    return 1;
  case ModuleKind::LoadPE: {
    // addr, data_in (from memory), ctrl = 3 inputs
    return 3;
  }
  case ModuleKind::StorePE: {
    // addr, data, ctrl = 3 inputs
    return 3;
  }
  case ModuleKind::Switch:
    return switchDefs[inst.defIdx].numIn;
  case ModuleKind::TemporalPE: {
    // Temporal PE has same number of I/O as its FU interface.
    auto &tpe = temporalPEDefs[inst.defIdx];
    if (!tpe.fuPEDefIndices.empty())
      return peDefs[tpe.fuPEDefIndices[0]].inputPorts.size();
    return 1;
  }
  case ModuleKind::TemporalSwitch:
    return temporalSwitchDefs[inst.defIdx].numIn;
  case ModuleKind::Memory: {
    auto &mem = memoryDefs[inst.defIdx];
    // load addrs + store addrs + store data = ldCount + 2*stCount
    return mem.ldCount + 2 * mem.stCount;
  }
  case ModuleKind::ExtMemory: {
    auto &mem = extMemoryDefs[inst.defIdx];
    // memref + load addrs + store addrs + store data = 1 + ldCount + 2*stCount
    return 1 + mem.ldCount + 2 * mem.stCount;
  }
  case ModuleKind::AddTag:
    return 1; // value in
  case ModuleKind::MapTag:
    return 1; // tagged in
  case ModuleKind::DelTag:
    return 1; // tagged in
  }
  return 0;
}

unsigned ADGBuilder::Impl::getInstanceOutputCount(unsigned instIdx) const {
  const auto &inst = instances[instIdx];
  switch (inst.kind) {
  case ModuleKind::PE:
    return peDefs[inst.defIdx].outputPorts.size();
  case ModuleKind::ConstantPE:
    return 1; // constant value
  case ModuleKind::LoadPE:
    return 2; // data_out, addr_out
  case ModuleKind::StorePE:
    return 2; // addr_out, done
  case ModuleKind::Switch:
    return switchDefs[inst.defIdx].numOut;
  case ModuleKind::TemporalPE: {
    auto &tpe = temporalPEDefs[inst.defIdx];
    if (!tpe.fuPEDefIndices.empty())
      return peDefs[tpe.fuPEDefIndices[0]].outputPorts.size();
    return 1;
  }
  case ModuleKind::TemporalSwitch:
    return temporalSwitchDefs[inst.defIdx].numOut;
  case ModuleKind::Memory: {
    auto &mem = memoryDefs[inst.defIdx];
    // Output: [memref?] [lddata * ldCount] [lddone] [stdone?]
    unsigned count = 0;
    if (!mem.isPrivate)
      count++; // memref output
    count += mem.ldCount; // load data outputs
    count++; // lddone (always present)
    if (mem.stCount > 0)
      count++; // stdone
    return count;
  }
  case ModuleKind::ExtMemory: {
    auto &mem = extMemoryDefs[inst.defIdx];
    // Output: [lddata * ldCount] [lddone] [stdone?]
    unsigned count = mem.ldCount + 1; // data + lddone
    if (mem.stCount > 0)
      count++; // stdone
    return count;
  }
  case ModuleKind::AddTag:
    return 1; // tagged out
  case ModuleKind::MapTag:
    return 1; // tagged out (remapped)
  case ModuleKind::DelTag:
    return 1; // native value out
  }
  return 0;
}

Type ADGBuilder::Impl::getInstanceInputType(unsigned instIdx, int port) const {
  const auto &inst = instances[instIdx];
  switch (inst.kind) {
  case ModuleKind::PE:
    return peDefs[inst.defIdx].inputPorts[port];
  case ModuleKind::Switch:
    return switchDefs[inst.defIdx].portType;
  case ModuleKind::TemporalPE:
    return temporalPEDefs[inst.defIdx].interfaceType;
  case ModuleKind::TemporalSwitch:
    return temporalSwitchDefs[inst.defIdx].interfaceType;
  case ModuleKind::AddTag:
    return addTagDefs[inst.defIdx].valueType;
  case ModuleKind::MapTag: {
    auto &def = mapTagDefs[inst.defIdx];
    return Type::tagged(def.valueType, def.inputTagType);
  }
  case ModuleKind::DelTag:
    return delTagDefs[inst.defIdx].inputType;
  case ModuleKind::ConstantPE: {
    auto &def = constantPEDefs[inst.defIdx];
    if (def.outputType.isTagged()) {
      Type tagType = def.outputType.getTagType();
      return Type::tagged(Type::none(), tagType);
    }
    return Type::none();
  }
  case ModuleKind::LoadPE: {
    auto &def = loadPEDefs[inst.defIdx];
    if (def.interface == InterfaceCategory::Tagged) {
      Type tagType = Type::iN(def.tagWidth);
      if (port == 0) return Type::tagged(Type::index(), tagType);
      if (port == 1) return Type::tagged(def.dataType, tagType);
      return Type::tagged(Type::none(), tagType);
    }
    if (port == 0) return Type::index();
    if (port == 1) return def.dataType;
    return Type::none();
  }
  case ModuleKind::StorePE: {
    auto &def = storePEDefs[inst.defIdx];
    if (def.interface == InterfaceCategory::Tagged) {
      Type tagType = Type::iN(def.tagWidth);
      if (port == 0) return Type::tagged(Type::index(), tagType);
      if (port == 1) return Type::tagged(def.dataType, tagType);
      return Type::tagged(Type::none(), tagType);
    }
    if (port == 0) return Type::index();
    if (port == 1) return def.dataType;
    return Type::none();
  }
  case ModuleKind::Memory: {
    auto &def = memoryDefs[inst.defIdx];
    Type elemType = def.shape.getElemType();
    bool isTaggedMem = def.ldCount > 1 || def.stCount > 1;
    unsigned tagWidth = 4;
    if (isTaggedMem) {
      unsigned maxCount = std::max(def.ldCount, def.stCount);
      tagWidth = 1;
      while ((1u << tagWidth) < maxCount) tagWidth++;
      if (tagWidth < 1) tagWidth = 1;
    }
    Type tagType = Type::iN(tagWidth);

    // Input layout: [ld_addr * ldCount, st_addr * stCount, st_data * stCount]
    unsigned idx = (unsigned)port;
    if (idx < def.ldCount) {
      return isTaggedMem ? Type::tagged(Type::index(), tagType) : Type::index();
    }
    idx -= def.ldCount;
    if (idx < def.stCount) {
      return isTaggedMem ? Type::tagged(Type::index(), tagType) : Type::index();
    }
    idx -= def.stCount;
    return isTaggedMem ? Type::tagged(elemType, tagType) : elemType;
  }
  case ModuleKind::ExtMemory: {
    auto &def = extMemoryDefs[inst.defIdx];
    Type elemType = def.shape.getElemType();
    // First input is memref (special)
    if (port == 0) return Type::index();
    unsigned adjPort = (unsigned)port - 1;

    bool isTaggedMem = def.ldCount > 1 || def.stCount > 1;
    unsigned tagWidth = 4;
    if (isTaggedMem) {
      unsigned maxCount = std::max(def.ldCount, def.stCount);
      tagWidth = 1;
      while ((1u << tagWidth) < maxCount) tagWidth++;
      if (tagWidth < 1) tagWidth = 1;
    }
    Type tagType = Type::iN(tagWidth);

    if (adjPort < def.ldCount) {
      return isTaggedMem ? Type::tagged(Type::index(), tagType) : Type::index();
    }
    adjPort -= def.ldCount;
    if (adjPort < def.stCount) {
      return isTaggedMem ? Type::tagged(Type::index(), tagType) : Type::index();
    }
    adjPort -= def.stCount;
    return isTaggedMem ? Type::tagged(elemType, tagType) : elemType;
  }
  }
  return Type::i32();
}

Type ADGBuilder::Impl::getInstanceOutputType(unsigned instIdx, int port) const {
  const auto &inst = instances[instIdx];
  switch (inst.kind) {
  case ModuleKind::PE:
    return peDefs[inst.defIdx].outputPorts[port];
  case ModuleKind::Switch:
    return switchDefs[inst.defIdx].portType;
  case ModuleKind::TemporalPE:
    return temporalPEDefs[inst.defIdx].interfaceType;
  case ModuleKind::TemporalSwitch:
    return temporalSwitchDefs[inst.defIdx].interfaceType;
  case ModuleKind::AddTag: {
    auto &def = addTagDefs[inst.defIdx];
    return Type::tagged(def.valueType, def.tagType);
  }
  case ModuleKind::MapTag: {
    auto &def = mapTagDefs[inst.defIdx];
    return Type::tagged(def.valueType, def.outputTagType);
  }
  case ModuleKind::DelTag: {
    auto &def = delTagDefs[inst.defIdx];
    return def.inputType.getValueType();
  }
  case ModuleKind::ConstantPE:
    return constantPEDefs[inst.defIdx].outputType;
  case ModuleKind::LoadPE: {
    auto &def = loadPEDefs[inst.defIdx];
    if (def.interface == InterfaceCategory::Tagged) {
      Type tagType = Type::iN(def.tagWidth);
      if (port == 0) return Type::tagged(def.dataType, tagType);
      return Type::tagged(Type::index(), tagType);
    }
    if (port == 0) return def.dataType;
    return Type::index();
  }
  case ModuleKind::StorePE: {
    auto &def = storePEDefs[inst.defIdx];
    if (def.interface == InterfaceCategory::Tagged) {
      Type tagType = Type::iN(def.tagWidth);
      if (port == 0) return Type::tagged(Type::index(), tagType);
      return Type::tagged(Type::none(), tagType);
    }
    if (port == 0) return Type::index();
    return Type::none();
  }
  case ModuleKind::Memory: {
    auto &def = memoryDefs[inst.defIdx];
    Type elemType = def.shape.getElemType();
    unsigned idx = 0;
    if (!def.isPrivate) { if (port == 0) return elemType; idx++; }
    if ((unsigned)port < idx + def.ldCount)
      return elemType;
    return Type::none();
  }
  case ModuleKind::ExtMemory: {
    auto &def = extMemoryDefs[inst.defIdx];
    Type elemType = def.shape.getElemType();
    if ((unsigned)port < def.ldCount)
      return elemType;
    return Type::none();
  }
  default:
    return Type::i32();
  }
}

} // namespace adg
} // namespace loom

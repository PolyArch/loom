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

//===----------------------------------------------------------------------===//
// Internal connections
//===----------------------------------------------------------------------===//

void ADGBuilder::connect(InstanceHandle src, InstanceHandle dst) {
  impl_->internalConns.push_back({src.id, 0, dst.id, 0});
}

void ADGBuilder::connectPorts(InstanceHandle src, int srcPort,
                              InstanceHandle dst, int dstPort) {
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
  impl_->inputConns.push_back({port.id, dst.id, dstPort});
}

void ADGBuilder::connectToModuleOutput(InstanceHandle src, int srcPort,
                                       PortHandle port) {
  impl_->outputConns.push_back({src.id, srcPort, port.id});
}

//===----------------------------------------------------------------------===//
// Topology
//===----------------------------------------------------------------------===//

MeshResult ADGBuilder::buildMesh(int rows, int cols, PEHandle peTemplate,
                                 SwitchHandle swTemplate, Topology topology) {
  MeshResult result;
  result.peGrid.resize(rows, std::vector<InstanceHandle>(cols));
  result.swGrid.resize(rows, std::vector<InstanceHandle>(cols));

  // Create PE and switch instances.
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::string peName =
          "pe_" + std::to_string(r) + "_" + std::to_string(c);
      result.peGrid[r][c] = clone(peTemplate, peName);

      std::string swName =
          "sw_" + std::to_string(r) + "_" + std::to_string(c);
      result.swGrid[r][c] = clone(swTemplate, swName);
    }
  }

  auto &swDef = impl_->switchDefs[swTemplate.id];
  bool wrapAround = (topology == Topology::Torus ||
                     topology == Topology::DiagonalTorus);

  // Connect PEs to their local switch (PE[r][c] <-> SW[r][c]).
  // PE output 0 -> SW input (from PE), SW output (to PE) -> PE input 0.
  // For a standard mesh, PE connects to SW via additional ports beyond NESW.
  // We connect PE[r][c] output 0 -> SW[r][c] input (numIn-1), etc.
  // Actually, for simplicity in a standard mesh:
  // Each PE connects bidirectionally to its local switch.
  // PE out 0 -> SW in (last port), SW out (last port) -> PE in 0.
  // But the switch port ordering is N=0, E=1, S=2, W=3 for inter-switch.
  // PE connections use port indices 4+ (or we connect PE to SW directionally).

  // Standard approach: PE[r][c] connects to SW[r][c].
  // Switch NESW ports connect to neighbor switches.
  // PE is attached to its local switch on extra ports.
  unsigned swIn = swDef.numIn;
  unsigned swOut = swDef.numOut;

  // Inter-switch connections: N=0, E=1, S=2, W=3.
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      auto sw = result.swGrid[r][c];

      // East connection: SW[r][c] out 1 -> SW[r][c+1] in 3 (West)
      if (c + 1 < cols) {
        auto swE = result.swGrid[r][c + 1];
        connectPorts(sw, 1, swE, 3); // East out -> West in
        connectPorts(swE, 3, sw, 1); // West out -> East in
      } else if (wrapAround && cols > 1) {
        auto swE = result.swGrid[r][0];
        connectPorts(sw, 1, swE, 3);
        connectPorts(swE, 3, sw, 1);
      }

      // South connection: SW[r][c] out 2 -> SW[r+1][c] in 0 (North)
      if (r + 1 < rows) {
        auto swS = result.swGrid[r + 1][c];
        connectPorts(sw, 2, swS, 0); // South out -> North in
        connectPorts(swS, 0, sw, 2); // North out -> South in
      } else if (wrapAround && rows > 1) {
        auto swS = result.swGrid[0][c];
        connectPorts(sw, 2, swS, 0);
        connectPorts(swS, 0, sw, 2);
      }
    }
  }

  // PE-to-switch connections: use port index 4 for PE attachment.
  // PE[r][c] output 0 -> SW[r][c] input 4; SW[r][c] output 4 -> PE[r][c] input 0
  if (swIn > 4 && swOut > 4) {
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        connectPorts(result.peGrid[r][c], 0, result.swGrid[r][c], 4);
        connectPorts(result.swGrid[r][c], 4, result.peGrid[r][c], 0);
      }
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

ValidationResult ADGBuilder::validateADG() {
  ValidationResult result;
  result.success = true;
  return result;
}

//===----------------------------------------------------------------------===//
// Export
//===----------------------------------------------------------------------===//

void ADGBuilder::exportMLIR(const std::string &path) {
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
  default:
    return Type::i32();
  }
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

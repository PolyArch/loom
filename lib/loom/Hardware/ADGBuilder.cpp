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
// Error reporting
//===----------------------------------------------------------------------===//

/// Report a builder API misuse error and exit.
static void builderError(const char *api, const std::string &msg) {
  llvm::errs() << "error: " << api << ": " << msg << "\n";
  std::exit(1);
}

//===----------------------------------------------------------------------===//
// Clone (instantiation)
//===----------------------------------------------------------------------===//

InstanceHandle ADGBuilder::clone(PEHandle source,
                                 const std::string &instanceName) {
  if (source.id >= impl_->peDefs.size())
    builderError("clone(PEHandle)", "invalid PE definition id " +
                 std::to_string(source.id));
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::PE, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(SwitchHandle source,
                                 const std::string &instanceName) {
  if (source.id >= impl_->switchDefs.size())
    builderError("clone(SwitchHandle)", "invalid switch definition id " +
                 std::to_string(source.id));
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::Switch, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(TemporalPEHandle source,
                                 const std::string &instanceName) {
  if (source.id >= impl_->temporalPEDefs.size())
    builderError("clone(TemporalPEHandle)",
                 "invalid temporal PE definition id " +
                 std::to_string(source.id));
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::TemporalPE, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(TemporalSwitchHandle source,
                                 const std::string &instanceName) {
  if (source.id >= impl_->temporalSwitchDefs.size())
    builderError("clone(TemporalSwitchHandle)",
                 "invalid temporal switch definition id " +
                 std::to_string(source.id));
  unsigned id = impl_->instances.size();
  impl_->instances.push_back(
      {ModuleKind::TemporalSwitch, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(MemoryHandle source,
                                 const std::string &instanceName) {
  if (source.id >= impl_->memoryDefs.size())
    builderError("clone(MemoryHandle)", "invalid memory definition id " +
                 std::to_string(source.id));
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::Memory, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(ExtMemoryHandle source,
                                 const std::string &instanceName) {
  if (source.id >= impl_->extMemoryDefs.size())
    builderError("clone(ExtMemoryHandle)",
                 "invalid external memory definition id " +
                 std::to_string(source.id));
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::ExtMemory, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(ConstantPEHandle source,
                                 const std::string &instanceName) {
  if (source.id >= impl_->constantPEDefs.size())
    builderError("clone(ConstantPEHandle)",
                 "invalid constant PE definition id " +
                 std::to_string(source.id));
  unsigned id = impl_->instances.size();
  impl_->instances.push_back(
      {ModuleKind::ConstantPE, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(LoadPEHandle source,
                                 const std::string &instanceName) {
  if (source.id >= impl_->loadPEDefs.size())
    builderError("clone(LoadPEHandle)", "invalid load PE definition id " +
                 std::to_string(source.id));
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::LoadPE, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(StorePEHandle source,
                                 const std::string &instanceName) {
  if (source.id >= impl_->storePEDefs.size())
    builderError("clone(StorePEHandle)", "invalid store PE definition id " +
                 std::to_string(source.id));
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({ModuleKind::StorePE, source.id, instanceName});
  return InstanceHandle{id};
}

InstanceHandle ADGBuilder::clone(ModuleHandle source,
                                 const std::string &instanceName) {
  // Tag-operation handles are not valid clone sources (they auto-instantiate).
  if (source.kind == ModuleHandle::AddTag ||
      source.kind == ModuleHandle::MapTag ||
      source.kind == ModuleHandle::DelTag)
    builderError("clone",
                 "tag-operation handles cannot be cloned; they auto-instantiate "
                 "via their builder's implicit conversion to InstanceHandle");

  // Validate definition ID bounds.
  static const struct { ModuleKind kind; const char *label; } kindMap[] = {
      {ModuleKind::PE, "PE"},
      {ModuleKind::Switch, "switch"},
      {ModuleKind::TemporalPE, "temporal PE"},
      {ModuleKind::TemporalSwitch, "temporal switch"},
      {ModuleKind::Memory, "memory"},
      {ModuleKind::ExtMemory, "external memory"},
      {ModuleKind::ConstantPE, "constant PE"},
      {ModuleKind::LoadPE, "load PE"},
      {ModuleKind::StorePE, "store PE"},
  };
  const unsigned defSizes[] = {
      (unsigned)impl_->peDefs.size(),
      (unsigned)impl_->switchDefs.size(),
      (unsigned)impl_->temporalPEDefs.size(),
      (unsigned)impl_->temporalSwitchDefs.size(),
      (unsigned)impl_->memoryDefs.size(),
      (unsigned)impl_->extMemoryDefs.size(),
      (unsigned)impl_->constantPEDefs.size(),
      (unsigned)impl_->loadPEDefs.size(),
      (unsigned)impl_->storePEDefs.size(),
  };
  unsigned kindIdx = (unsigned)source.kind;
  if (source.id >= defSizes[kindIdx])
    builderError("clone(ModuleHandle)",
                 std::string("invalid ") + kindMap[kindIdx].label +
                 " definition id " + std::to_string(source.id));
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({kindMap[kindIdx].kind, source.id, instanceName});
  return InstanceHandle{id};
}

//===----------------------------------------------------------------------===//
// Internal connections
//===----------------------------------------------------------------------===//

void ADGBuilder::connect(InstanceHandle src, InstanceHandle dst) {
  if (src.id >= impl_->instances.size())
    builderError("connect", "invalid source instance id " +
                 std::to_string(src.id));
  if (dst.id >= impl_->instances.size())
    builderError("connect", "invalid destination instance id " +
                 std::to_string(dst.id));
  impl_->internalConns.push_back({src.id, 0, dst.id, 0});
}

void ADGBuilder::connectPorts(InstanceHandle src, int srcPort,
                              InstanceHandle dst, int dstPort) {
  if (src.id >= impl_->instances.size())
    builderError("connectPorts", "invalid source instance id " +
                 std::to_string(src.id));
  if (dst.id >= impl_->instances.size())
    builderError("connectPorts", "invalid destination instance id " +
                 std::to_string(dst.id));
  if (srcPort < 0 || (unsigned)srcPort >= impl_->getInstanceOutputCount(src.id))
    builderError("connectPorts", "source port " + std::to_string(srcPort) +
                 " out of range [0, " +
                 std::to_string(impl_->getInstanceOutputCount(src.id)) +
                 ") for instance '" + impl_->instances[src.id].name + "'");
  if (dstPort < 0 || (unsigned)dstPort >= impl_->getInstanceInputCount(dst.id))
    builderError("connectPorts", "destination port " + std::to_string(dstPort) +
                 " out of range [0, " +
                 std::to_string(impl_->getInstanceInputCount(dst.id)) +
                 ") for instance '" + impl_->instances[dst.id].name + "'");
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
  if (port.id >= impl_->ports.size())
    builderError("connectToModuleInput", "invalid port handle id " +
                 std::to_string(port.id));
  if (!impl_->ports[port.id].isInput)
    builderError("connectToModuleInput", "port '" +
                 impl_->ports[port.id].name + "' is an output port, not input");
  if (dst.id >= impl_->instances.size())
    builderError("connectToModuleInput", "invalid instance handle id " +
                 std::to_string(dst.id));
  if (dstPort < 0 ||
      (unsigned)dstPort >= impl_->getInstanceInputCount(dst.id))
    builderError("connectToModuleInput", "destination port " +
                 std::to_string(dstPort) + " out of range [0, " +
                 std::to_string(impl_->getInstanceInputCount(dst.id)) +
                 ") for instance '" + impl_->instances[dst.id].name + "'");
  impl_->inputConns.push_back({port.id, dst.id, dstPort});
}

void ADGBuilder::connectToModuleOutput(InstanceHandle src, int srcPort,
                                       PortHandle port) {
  if (src.id >= impl_->instances.size())
    builderError("connectToModuleOutput", "invalid instance handle id " +
                 std::to_string(src.id));
  if (port.id >= impl_->ports.size())
    builderError("connectToModuleOutput", "invalid port handle id " +
                 std::to_string(port.id));
  if (impl_->ports[port.id].isInput)
    builderError("connectToModuleOutput", "port '" +
                 impl_->ports[port.id].name + "' is an input port, not output");
  if (srcPort < 0 ||
      (unsigned)srcPort >= impl_->getInstanceOutputCount(src.id))
    builderError("connectToModuleOutput", "source port " +
                 std::to_string(srcPort) + " out of range [0, " +
                 std::to_string(impl_->getInstanceOutputCount(src.id)) +
                 ") for instance '" + impl_->instances[src.id].name + "'");
  impl_->outputConns.push_back({src.id, srcPort, port.id});
}

//===----------------------------------------------------------------------===//
// Topology
//===----------------------------------------------------------------------===//

MeshResult ADGBuilder::buildMesh(int rows, int cols, PEHandle peTemplate,
                                 SwitchHandle swTemplate, Topology topology) {
  if (rows <= 0)
    builderError("buildMesh", "rows must be positive");
  if (cols <= 0)
    builderError("buildMesh", "cols must be positive");
  if (peTemplate.id >= impl_->peDefs.size())
    builderError("buildMesh", "invalid PE template handle id " +
                 std::to_string(peTemplate.id));
  if (swTemplate.id >= impl_->switchDefs.size())
    builderError("buildMesh", "invalid switch template handle id " +
                 std::to_string(swTemplate.id));

  // Validate switch port count. Need at least 5 ports for N/E/S/W + PE-local.
  auto &swDef = impl_->switchDefs[swTemplate.id];
  if (swDef.numIn < 5)
    builderError("buildMesh",
                 "switch needs >= 5 input ports for mesh topology");
  if (swDef.numOut < 5)
    builderError("buildMesh",
                 "switch needs >= 5 output ports for mesh topology");

  bool diagonal = (topology == Topology::DiagonalMesh ||
                   topology == Topology::DiagonalTorus);
  if (diagonal) {
    // Diagonal topologies require ports 5 (SE) and 6 (SW) on switches.
    if (swDef.numIn < 7)
      builderError("buildMesh",
                   "diagonal topology requires >= 7 input ports on switch "
                   "(N=0,E=1,S=2,W=3,PE=4,SE=5,SW=6); have " +
                   std::to_string(swDef.numIn));
    if (swDef.numOut < 7)
      builderError("buildMesh",
                   "diagonal topology requires >= 7 output ports on switch "
                   "(N=0,E=1,S=2,W=3,PE=4,SE=5,SW=6); have " +
                   std::to_string(swDef.numOut));
  }

  MeshResult result;
  result.peGrid.resize(rows, std::vector<InstanceHandle>(cols));
  result.swGrid.resize(rows, std::vector<InstanceHandle>(cols));

  // Create PE instances first (they appear first in topological order).
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::string peName =
          "pe_" + std::to_string(r) + "_" + std::to_string(c);
      result.peGrid[r][c] = clone(peTemplate, peName);
    }
  }

  // Create switch instances (they follow PEs in topological order).
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::string swName =
          "sw_" + std::to_string(r) + "_" + std::to_string(c);
      result.swGrid[r][c] = clone(swTemplate, swName);
    }
  }

  // Wire PE output 0 -> local switch port 4 (PE-local input).
  // Data flows from PEs into the switch network.
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      connectPorts(result.peGrid[r][c], 0, result.swGrid[r][c], 4);
    }
  }

  // Switch-to-switch connections.
  // Switch port ordering: N=0, E=1, S=2, W=3, PE-local=4+
  // Note: `diagonal` was already computed above for port validation.
  bool torus = (topology == Topology::Torus ||
                topology == Topology::DiagonalTorus);

  // Use a unique mesh ID to avoid name collisions when multiple buildMesh
  // calls are made.
  static unsigned meshCounter = 0;
  unsigned meshId = meshCounter++;
  Type swPortType = swDef.portType;
  std::string mPrefix = "m" + std::to_string(meshId) + "_";

  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      // East: SW[r][c] out 1 -> SW[r][c+1] in 3
      if (c + 1 < cols) {
        connectPorts(result.swGrid[r][c], 1, result.swGrid[r][c + 1], 3);
      }

      // South: SW[r][c] out 2 -> SW[r+1][c] in 0
      if (r + 1 < rows) {
        connectPorts(result.swGrid[r][c], 2, result.swGrid[r + 1][c], 0);
      }

      if (diagonal) {
        // SE diagonal: SW[r][c] out 5 -> SW[r+1][c+1] in 5
        if (r + 1 < rows && c + 1 < cols) {
          connectPorts(result.swGrid[r][c], 5,
                       result.swGrid[r + 1][c + 1], 5);
        }

        // SW diagonal: SW[r][c] out 6 -> SW[r+1][c-1] in 6
        if (r + 1 < rows && c - 1 >= 0) {
          connectPorts(result.swGrid[r][c], 6,
                       result.swGrid[r + 1][c - 1], 6);
        }
      }
    }
  }

  // Torus/DiagonalTorus wraparound: create module I/O pairs for wrap edges.
  // These cannot be internal connections due to MLIR SSA DAG constraint.
  // Each wraparound creates an output port (from the source switch) and a
  // corresponding input port (to the destination switch) with clear naming.
  if (torus) {
    // East-West wraparound: SW[r][cols-1] out 1 -> SW[r][0] in 3
    for (int r = 0; r < rows; ++r) {
      std::string wrapName = mPrefix + "wrap_ew_r" + std::to_string(r);
      auto wrapOut = addModuleOutput(wrapName + "_out", swPortType);
      connectToModuleOutput(result.swGrid[r][cols - 1], 1, wrapOut);
      auto wrapIn = addModuleInput(wrapName + "_in", swPortType);
      connectToModuleInput(wrapIn, result.swGrid[r][0], 3);
    }

    // North-South wraparound: SW[rows-1][c] out 2 -> SW[0][c] in 0
    for (int c = 0; c < cols; ++c) {
      std::string wrapName = mPrefix + "wrap_ns_c" + std::to_string(c);
      auto wrapOut = addModuleOutput(wrapName + "_out", swPortType);
      connectToModuleOutput(result.swGrid[rows - 1][c], 2, wrapOut);
      auto wrapIn = addModuleInput(wrapName + "_in", swPortType);
      connectToModuleInput(wrapIn, result.swGrid[0][c], 0);
    }

    if (diagonal) {
      // SE diagonal wraparound (rows): SW[rows-1][c] out 5 -> SW[0][c+1] in 5
      for (int c = 0; c + 1 < cols; ++c) {
        std::string wrapName = mPrefix + "wrap_se_r_c" + std::to_string(c);
        auto wrapOut = addModuleOutput(wrapName + "_out", swPortType);
        connectToModuleOutput(result.swGrid[rows - 1][c], 5, wrapOut);
        auto wrapIn = addModuleInput(wrapName + "_in", swPortType);
        connectToModuleInput(wrapIn, result.swGrid[0][c + 1], 5);
      }

      // SE diagonal wraparound (cols): SW[r][cols-1] out 5 -> SW[r+1][0] in 5
      for (int r = 0; r + 1 < rows; ++r) {
        std::string wrapName = mPrefix + "wrap_se_c_r" + std::to_string(r);
        auto wrapOut = addModuleOutput(wrapName + "_out", swPortType);
        connectToModuleOutput(result.swGrid[r][cols - 1], 5, wrapOut);
        auto wrapIn = addModuleInput(wrapName + "_in", swPortType);
        connectToModuleInput(wrapIn, result.swGrid[r + 1][0], 5);
      }

      // SE diagonal wraparound (corner): SW[rows-1][cols-1] out 5 -> SW[0][0] in 5
      {
        std::string wrapName = mPrefix + "wrap_se_corner";
        auto wrapOut = addModuleOutput(wrapName + "_out", swPortType);
        connectToModuleOutput(result.swGrid[rows - 1][cols - 1], 5, wrapOut);
        auto wrapIn = addModuleInput(wrapName + "_in", swPortType);
        connectToModuleInput(wrapIn, result.swGrid[0][0], 5);
      }

      // SW diagonal wraparound (rows): SW[rows-1][c] out 6 -> SW[0][c-1] in 6
      for (int c = 1; c < cols; ++c) {
        std::string wrapName = mPrefix + "wrap_sw_r_c" + std::to_string(c);
        auto wrapOut = addModuleOutput(wrapName + "_out", swPortType);
        connectToModuleOutput(result.swGrid[rows - 1][c], 6, wrapOut);
        auto wrapIn = addModuleInput(wrapName + "_in", swPortType);
        connectToModuleInput(wrapIn, result.swGrid[0][c - 1], 6);
      }

      // SW diagonal wraparound (cols): SW[r][0] out 6 -> SW[r+1][cols-1] in 6
      for (int r = 0; r + 1 < rows; ++r) {
        std::string wrapName = mPrefix + "wrap_sw_c_r" + std::to_string(r);
        auto wrapOut = addModuleOutput(wrapName + "_out", swPortType);
        connectToModuleOutput(result.swGrid[r][0], 6, wrapOut);
        auto wrapIn = addModuleInput(wrapName + "_in", swPortType);
        connectToModuleInput(wrapIn, result.swGrid[r + 1][cols - 1], 6);
      }

      // SW diagonal wraparound (corner): SW[rows-1][0] out 6 -> SW[0][cols-1] in 6
      {
        std::string wrapName = mPrefix + "wrap_sw_corner";
        auto wrapOut = addModuleOutput(wrapName + "_out", swPortType);
        connectToModuleOutput(result.swGrid[rows - 1][0], 6, wrapOut);
        auto wrapIn = addModuleInput(wrapName + "_in", swPortType);
        connectToModuleInput(wrapIn, result.swGrid[0][cols - 1], 6);
      }
    }
  }

  // Connect unconnected switch input ports to auto-generated module inputs.
  // This ensures the MLIR generator has valid operands for all switch ports.
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      unsigned swInst = result.swGrid[r][c].id;
      unsigned numIn = impl_->getInstanceInputCount(swInst);
      // Check which input ports are already connected.
      std::vector<bool> connected(numIn, false);
      for (const auto &conn : impl_->internalConns) {
        if (conn.dstInst == swInst)
          connected[conn.dstPort] = true;
      }
      for (const auto &conn : impl_->inputConns) {
        if (conn.instIdx == swInst)
          connected[conn.dstPort] = true;
      }
      // Create module inputs for unconnected input ports.
      for (unsigned p = 0; p < numIn; ++p) {
        if (!connected[p]) {
          std::string portName =
              mPrefix + "sw_" +
              std::to_string(r) + "_" + std::to_string(c) +
              "_in" + std::to_string(p);
          auto mPort = addModuleInput(portName, swPortType);
          connectToModuleInput(mPort, result.swGrid[r][c], p);
        }
      }

      // Auto-fill unconnected switch output ports with module outputs.
      unsigned numOut = impl_->getInstanceOutputCount(swInst);
      std::vector<bool> outUsed(numOut, false);
      for (const auto &conn : impl_->internalConns) {
        if (conn.srcInst == swInst)
          outUsed[conn.srcPort] = true;
      }
      for (const auto &conn : impl_->outputConns) {
        if (conn.instIdx == swInst)
          outUsed[conn.srcPort] = true;
      }
      for (unsigned p = 0; p < numOut; ++p) {
        if (!outUsed[p]) {
          std::string portName =
              mPrefix + "sw_" +
              std::to_string(r) + "_" + std::to_string(c) +
              "_out" + std::to_string(p);
          auto mPort = addModuleOutput(portName, swPortType);
          connectToModuleOutput(result.swGrid[r][c], p, mPort);
        }
      }
    }
  }

  return result;
}

// Validation is in ADGBuilderValidation.cpp.

//===----------------------------------------------------------------------===//
// Query
//===----------------------------------------------------------------------===//

std::vector<std::string> ADGBuilder::getModuleInputNames() const {
  std::vector<std::string> names;
  for (const auto &p : impl_->ports)
    if (p.isInput)
      names.push_back(p.name);
  return names;
}

std::vector<std::string> ADGBuilder::getModuleOutputNames() const {
  std::vector<std::string> names;
  for (const auto &p : impl_->ports)
    if (!p.isInput)
      names.push_back(p.name);
  return names;
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
    bool isTaggedMem = def.ldCount > 1 || def.stCount > 1;
    unsigned tagWidth = 4;
    if (isTaggedMem) {
      unsigned maxCount = std::max(def.ldCount, def.stCount);
      tagWidth = 1;
      while ((1u << tagWidth) < maxCount) tagWidth++;
      if (tagWidth < 1) tagWidth = 1;
    }
    Type tagType = Type::iN(tagWidth);

    unsigned idx = 0;
    // Non-private memory port 0 is memref -- return index as placeholder for
    // scalar type query (the PortType variant returns the actual memref).
    if (!def.isPrivate) { if (port == 0) return Type::index(); idx++; }
    // Output layout: [memref?] [lddata * ldCount] [lddone] [stdone?]
    if ((unsigned)port < idx + def.ldCount)
      return isTaggedMem ? Type::tagged(elemType, tagType) : elemType;
    // Remaining are done tokens.
    return isTaggedMem ? Type::tagged(Type::none(), tagType) : Type::none();
  }
  case ModuleKind::ExtMemory: {
    auto &def = extMemoryDefs[inst.defIdx];
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

    // Output layout: [lddata * ldCount] [lddone] [stdone?]
    if ((unsigned)port < def.ldCount)
      return isTaggedMem ? Type::tagged(elemType, tagType) : elemType;
    return isTaggedMem ? Type::tagged(Type::none(), tagType) : Type::none();
  }
  default:
    return Type::i32();
  }
}

PortType ADGBuilder::Impl::getInstanceOutputPortType(unsigned instIdx,
                                                     int port) const {
  const auto &inst = instances[instIdx];
  // Memory non-private port 0 is memref.
  if (inst.kind == ModuleKind::Memory) {
    auto &def = memoryDefs[inst.defIdx];
    if (!def.isPrivate && port == 0)
      return PortType::memref(def.shape);
  }
  return PortType::scalar(getInstanceOutputType(instIdx, port));
}

PortType ADGBuilder::Impl::getInstanceInputPortType(unsigned instIdx,
                                                    int port) const {
  const auto &inst = instances[instIdx];
  // ExtMemory port 0 is memref.
  if (inst.kind == ModuleKind::ExtMemory && port == 0) {
    auto &def = extMemoryDefs[inst.defIdx];
    return PortType::memref(def.shape);
  }
  return PortType::scalar(getInstanceInputType(instIdx, port));
}

} // namespace adg
} // namespace loom

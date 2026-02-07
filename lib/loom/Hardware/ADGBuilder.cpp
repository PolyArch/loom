//===-- ADGBuilder.cpp - ADG Builder implementation --------------*- C++ -*-===//
//
// Part of the Loom project.
//
//===----------------------------------------------------------------------===//

#include "loom/Hardware/adg.h"

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

#include <cassert>
#include <sstream>

namespace loom {
namespace adg {

//===----------------------------------------------------------------------===//
// Type
//===----------------------------------------------------------------------===//

std::string Type::toMLIR() const {
  switch (kind_) {
  case I1:    return "i1";
  case I8:    return "i8";
  case I16:   return "i16";
  case I32:   return "i32";
  case I64:   return "i64";
  case IN:    return "i" + std::to_string(width_);
  case BF16:  return "bf16";
  case F16:   return "f16";
  case F32:   return "f32";
  case F64:   return "f64";
  case Index: return "index";
  case None:  return "none";
  }
  return "i32"; // fallback
}

//===----------------------------------------------------------------------===//
// ADGBuilder::Impl
//===----------------------------------------------------------------------===//

struct ADGBuilder::Impl {
  std::string moduleName;

  struct PEDef {
    std::string name;
    int16_t latMin = 1, latTyp = 1, latMax = 1;
    int16_t intMin = 1, intTyp = 1, intMax = 1;
    std::vector<Type> inputPorts;
    std::vector<Type> outputPorts;
    std::string bodyMLIR;   // raw MLIR body from setBodyMLIR()
    std::string singleOp;   // dialect op name from addOp()
  };
  std::vector<PEDef> peDefs;

  struct InstanceDef {
    unsigned peDefIdx;
    std::string name;
  };
  std::vector<InstanceDef> instances;

  struct ModulePort {
    std::string name;
    Type type;
    bool isInput;
  };
  std::vector<ModulePort> ports;

  struct InputConn {
    unsigned portIdx;
    unsigned instIdx;
    int dstPort;
  };
  struct OutputConn {
    unsigned instIdx;
    int srcPort;
    unsigned portIdx;
  };
  std::vector<InputConn> inputConns;
  std::vector<OutputConn> outputConns;

  /// Generate the PE body MLIR text for a PEDef.
  std::string generatePEBody(const PEDef &pe) const;

  /// Generate full MLIR text from the internal state.
  std::string generateMLIR() const;
};

std::string ADGBuilder::Impl::generatePEBody(const PEDef &pe) const {
  if (!pe.bodyMLIR.empty())
    return pe.bodyMLIR;

  // Auto-generate body from singleOp.
  assert(!pe.singleOp.empty() && "PE must have either bodyMLIR or singleOp");
  assert(!pe.inputPorts.empty() && "PE must have input ports");
  assert(!pe.outputPorts.empty() && "PE must have output ports");

  std::ostringstream os;
  // Build the operation: %0 = <op> %arg0, %arg1 : <type>
  os << "  %0 = " << pe.singleOp;
  for (size_t i = 0; i < pe.inputPorts.size(); ++i) {
    os << (i == 0 ? " " : ", ");
    os << "%arg" << i;
  }
  os << " : " << pe.outputPorts[0].toMLIR() << "\n";
  os << "  fabric.yield %0 : " << pe.outputPorts[0].toMLIR() << "\n";
  return os.str();
}

std::string ADGBuilder::Impl::generateMLIR() const {
  std::ostringstream os;

  // Emit top-level named PE definitions.
  for (const auto &pe : peDefs) {
    os << "fabric.pe @" << pe.name << "(";
    for (size_t i = 0; i < pe.inputPorts.size(); ++i) {
      if (i > 0) os << ", ";
      os << "%arg" << i << ": " << pe.inputPorts[i].toMLIR();
    }
    os << ")\n";
    os << "    [latency = ["
       << pe.latMin << " : i16, "
       << pe.latTyp << " : i16, "
       << pe.latMax << " : i16]";
    os << ", interval = ["
       << pe.intMin << " : i16, "
       << pe.intTyp << " : i16, "
       << pe.intMax << " : i16]";
    os << "]\n";
    os << "    -> (";
    for (size_t i = 0; i < pe.outputPorts.size(); ++i) {
      if (i > 0) os << ", ";
      os << pe.outputPorts[i].toMLIR();
    }
    os << ") {\n";
    os << generatePEBody(pe);
    os << "}\n\n";
  }

  // Collect module input and output ports in order.
  std::vector<const ModulePort *> inputPorts, outputPorts;
  for (const auto &p : ports) {
    if (p.isInput)
      inputPorts.push_back(&p);
    else
      outputPorts.push_back(&p);
  }

  // Emit fabric.module.
  os << "fabric.module @" << moduleName << "(";
  for (size_t i = 0; i < inputPorts.size(); ++i) {
    if (i > 0) os << ", ";
    os << "%" << inputPorts[i]->name << ": " << inputPorts[i]->type.toMLIR();
  }
  os << ") -> (";
  for (size_t i = 0; i < outputPorts.size(); ++i) {
    if (i > 0) os << ", ";
    os << outputPorts[i]->type.toMLIR();
  }
  os << ") {\n";

  // Build a map: portIdx -> index among input ports (for block arg names).
  std::map<unsigned, unsigned> inputPortToArgIdx;
  for (size_t i = 0; i < inputPorts.size(); ++i) {
    // Find this port's index in the global ports list.
    for (unsigned j = 0; j < ports.size(); ++j) {
      if (&ports[j] == inputPorts[i]) {
        inputPortToArgIdx[j] = i;
        break;
      }
    }
  }

  // For each instance, resolve its input connections to SSA values and emit.
  // SSA counter for instance results.
  unsigned ssaCounter = 0;

  // Map: (instIdx, srcPort) -> SSA name
  std::map<std::pair<unsigned, int>, std::string> instResultSSA;

  for (size_t ii = 0; ii < instances.size(); ++ii) {
    const auto &inst = instances[ii];
    const auto &pe = peDefs[inst.peDefIdx];

    // Gather operand SSA names for this instance's inputs.
    std::vector<std::string> operands(pe.inputPorts.size());
    for (const auto &conn : inputConns) {
      if (conn.instIdx == ii) {
        // This connection feeds a module input port to inst's dstPort.
        auto it = inputPortToArgIdx.find(conn.portIdx);
        assert(it != inputPortToArgIdx.end());
        operands[conn.dstPort] = "%" + inputPorts[it->second]->name;
      }
    }

    // Emit fabric.instance.
    std::string resultName = "%" + std::to_string(ssaCounter);
    // Build result names for multi-output.
    std::vector<std::string> resultNames;
    for (size_t r = 0; r < pe.outputPorts.size(); ++r) {
      resultNames.push_back("%" + std::to_string(ssaCounter + r));
      instResultSSA[{ii, (int)r}] = resultNames.back();
    }
    ssaCounter += pe.outputPorts.size();

    os << "  ";
    for (size_t r = 0; r < resultNames.size(); ++r) {
      if (r > 0) os << ", ";
      os << resultNames[r];
    }
    os << " = fabric.instance @" << pe.name << "(";
    for (size_t o = 0; o < operands.size(); ++o) {
      if (o > 0) os << ", ";
      os << operands[o];
    }
    os << ")";

    // Add sym_name attribute if the instance has a name.
    if (!inst.name.empty()) {
      os << " {sym_name = \"" << inst.name << "\"}";
    }

    os << " : (";
    for (size_t p = 0; p < pe.inputPorts.size(); ++p) {
      if (p > 0) os << ", ";
      os << pe.inputPorts[p].toMLIR();
    }
    os << ") -> (";
    for (size_t p = 0; p < pe.outputPorts.size(); ++p) {
      if (p > 0) os << ", ";
      os << pe.outputPorts[p].toMLIR();
    }
    os << ")\n";
  }

  // Emit fabric.yield with output connections.
  os << "  fabric.yield";
  if (!outputPorts.empty()) {
    os << " ";
    std::vector<std::pair<std::string, std::string>> yieldArgs;
    for (size_t oi = 0; oi < outputPorts.size(); ++oi) {
      // Find the output connection for this output port.
      unsigned outPortIdx = 0;
      for (unsigned j = 0; j < ports.size(); ++j) {
        if (&ports[j] == outputPorts[oi]) {
          outPortIdx = j;
          break;
        }
      }
      for (const auto &conn : outputConns) {
        if (conn.portIdx == outPortIdx) {
          auto it = instResultSSA.find({conn.instIdx, conn.srcPort});
          assert(it != instResultSSA.end());
          yieldArgs.push_back({it->second, outputPorts[oi]->type.toMLIR()});
          break;
        }
      }
    }
    for (size_t i = 0; i < yieldArgs.size(); ++i) {
      if (i > 0) os << ", ";
      os << yieldArgs[i].first;
    }
    os << " : ";
    for (size_t i = 0; i < yieldArgs.size(); ++i) {
      if (i > 0) os << ", ";
      os << yieldArgs[i].second;
    }
  }
  os << "\n";
  os << "}\n";

  return os.str();
}

//===----------------------------------------------------------------------===//
// PEBuilder
//===----------------------------------------------------------------------===//

PEBuilder::PEBuilder(ADGBuilder *builder, unsigned peId)
    : builder_(builder), peId_(peId) {}

PEBuilder &PEBuilder::setLatency(int16_t min, int16_t typical, int16_t max) {
  auto &pe = builder_->impl_->peDefs[peId_];
  pe.latMin = min;
  pe.latTyp = typical;
  pe.latMax = max;
  return *this;
}

PEBuilder &PEBuilder::setInterval(int16_t min, int16_t typical, int16_t max) {
  auto &pe = builder_->impl_->peDefs[peId_];
  pe.intMin = min;
  pe.intTyp = typical;
  pe.intMax = max;
  return *this;
}

PEBuilder &PEBuilder::setInputPorts(std::vector<Type> types) {
  builder_->impl_->peDefs[peId_].inputPorts = std::move(types);
  return *this;
}

PEBuilder &PEBuilder::setOutputPorts(std::vector<Type> types) {
  builder_->impl_->peDefs[peId_].outputPorts = std::move(types);
  return *this;
}

PEBuilder &PEBuilder::addOp(const std::string &opName) {
  builder_->impl_->peDefs[peId_].singleOp = opName;
  return *this;
}

PEBuilder &PEBuilder::setBodyMLIR(const std::string &mlirString) {
  builder_->impl_->peDefs[peId_].bodyMLIR = mlirString;
  return *this;
}

PEBuilder::operator PEHandle() const { return PEHandle{peId_}; }

//===----------------------------------------------------------------------===//
// ADGBuilder
//===----------------------------------------------------------------------===//

ADGBuilder::ADGBuilder(const std::string &moduleName)
    : impl_(std::make_unique<Impl>()) {
  impl_->moduleName = moduleName;
}

ADGBuilder::~ADGBuilder() = default;

PEBuilder ADGBuilder::newPE(const std::string &name) {
  unsigned id = impl_->peDefs.size();
  impl_->peDefs.push_back({});
  impl_->peDefs.back().name = name;
  return PEBuilder(this, id);
}

InstanceHandle ADGBuilder::clone(PEHandle source,
                                 const std::string &instanceName) {
  unsigned id = impl_->instances.size();
  impl_->instances.push_back({source.id, instanceName});
  return InstanceHandle{id};
}

PortHandle ADGBuilder::addModuleInput(const std::string &name, Type type) {
  unsigned id = impl_->ports.size();
  impl_->ports.push_back({name, type, true});
  return PortHandle{id};
}

PortHandle ADGBuilder::addModuleOutput(const std::string &name, Type type) {
  unsigned id = impl_->ports.size();
  impl_->ports.push_back({name, type, false});
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

void ADGBuilder::validateADG() {
  // Placeholder for future validation logic.
}

void ADGBuilder::exportMLIR(const std::string &path) {
  std::string mlirText = impl_->generateMLIR();

  // Parse and verify through MLIR infrastructure.
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

  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
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

  // Write to file.
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

} // namespace adg
} // namespace loom

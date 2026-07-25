#ifndef LOOM_ADG_BUILDER_H
#define LOOM_ADG_BUILDER_H

#include "Fabric/IR/FabricEnums.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace dataflow {
enum class StreamStepKind : std::uint32_t;
}

namespace mlir::arith {
enum class CmpIPredicate : std::uint64_t;
}

namespace loom {
namespace adg {

struct ModuleBuilderInternals;

enum class Schedule { Spatial, Temporal };

struct PortBinding {
  std::string localName;
  std::string sourceName;
  std::string type;
  std::string castType;
};

struct StreamConfig {
  StreamConfig(dataflow::StreamStepKind stepKind,
               std::vector<mlir::arith::CmpIPredicate> predicates,
               std::optional<mlir::arith::CmpIPredicate> selectedPredicate =
                   std::nullopt)
      : stepKind(stepKind), predicates(std::move(predicates)),
        selectedPredicate(selectedPredicate) {}

  dataflow::StreamStepKind stepKind;
  std::vector<mlir::arith::CmpIPredicate> predicates;
  std::optional<mlir::arith::CmpIPredicate> selectedPredicate = std::nullopt;
};

struct FabricOpSpec {
  std::vector<std::string> results;
  std::vector<std::string> opList;
  std::vector<std::string> operands;
  std::vector<std::string> operandTypes;
  std::vector<std::string> resultTypes;
  std::map<std::string, std::vector<std::string>> hwParams;
  std::map<std::string, std::string> swConfigs;
  std::optional<StreamConfig> streamConfig = std::nullopt;
};

struct FuSpec {
  std::vector<PortBinding> inputs;
  std::vector<std::string> resultTypes;
  std::vector<FabricOpSpec> operations;
  std::vector<std::string> yieldValues;
  std::vector<std::string> yieldTypes = {};
};

struct TemporalPeConfig {
  unsigned tagWidth = 0;
  unsigned numInstruction = 0;
  std::string fuConfigMode;
  ::fabric::OperandBufferMode operandBufferMode =
      ::fabric::OperandBufferMode::PerInstruction;
  // Entries per mode-derived allocation unit. Zero means the recipe supplied
  // nothing and is rejected before Fabric emission; no mode has a default.
  unsigned operandBufferSize = 0;
  unsigned numRegFifo = 0;
  unsigned regFifoDepth = 0;
  unsigned regFifoPorts = 0;
};

struct PeSpec {
  Schedule schedule = Schedule::Spatial;
  std::vector<PortBinding> inputs;
  std::vector<std::string> resultNames;
  std::vector<std::string> resultTypes;
  std::vector<FuSpec> fus;
  TemporalPeConfig temporal;
};

struct SwitchSpec {
  Schedule schedule = Schedule::Spatial;
  std::vector<std::string> inputs;
  std::vector<std::string> resultTypes;
  std::vector<std::string> connectivityTable;
  unsigned temporalRouteTableSize = 0;
};

struct FifoSpec {
  FifoSpec(std::string resultName, std::string sourceName,
           std::string resultType, unsigned maxDepth, bool bypassable,
           std::optional<bool> bypassed = std::nullopt)
      : resultName(std::move(resultName)), sourceName(std::move(sourceName)),
        resultType(std::move(resultType)), maxDepth(maxDepth),
        bypassable(bypassable), bypassed(bypassed) {}

  std::string resultName;
  std::string sourceName;
  std::string resultType;
  unsigned maxDepth;
  bool bypassable;
  std::optional<bool> bypassed = std::nullopt;
};

struct BoundaryInput {
  BoundaryInput(std::string sourceName,
                std::optional<std::string> destinationType = std::nullopt)
      : sourceName(std::move(sourceName)),
        destinationType(std::move(destinationType)) {}

  std::string sourceName;
  std::optional<std::string> destinationType = std::nullopt;
};

struct BoundarySpec {
  BoundarySpec(::fabric::BoundaryDirection direction,
               std::vector<BoundaryInput> inputs,
               std::vector<std::string> resultNames,
               std::vector<std::string> resultTypes)
      : direction(direction), inputs(std::move(inputs)),
        resultNames(std::move(resultNames)),
        resultTypes(std::move(resultTypes)) {}

  ::fabric::BoundaryDirection direction;
  std::vector<BoundaryInput> inputs;
  std::vector<std::string> resultNames;
  std::vector<std::string> resultTypes;
};

struct MemLoadPort {
  std::string address;
  std::string control;
};

struct MemStorePort {
  std::string address;
  std::string data;
  std::string control;
};

struct MemSubordinateOutput {
  std::string name;
  std::string type;
};

struct MemDispatchEligibility {
  std::vector<std::vector<unsigned>> operationPortRequests;
  std::vector<std::vector<unsigned>> subordinateRequests;
};

struct MemSpec {
  MemSpec(Schedule schedule, std::vector<std::string> managerInputs,
          std::vector<MemSubordinateOutput> subordinateOutputs,
          MemDispatchEligibility dispatchEligibility)
      : schedule(schedule), managerInputs(std::move(managerInputs)),
        subordinateOutputs(std::move(subordinateOutputs)),
        dispatchEligibility(std::move(dispatchEligibility)) {}

  Schedule schedule;
  std::vector<std::string> managerInputs;
  std::vector<MemSubordinateOutput> subordinateOutputs;
  MemDispatchEligibility dispatchEligibility;
  std::vector<MemLoadPort> loads;
  std::vector<MemStorePort> stores;
  unsigned dataWidth = 0;
  unsigned temporalTagWidth = 0;
  unsigned temporalOperationTableSize = 0;
};

class ModuleBuilder {
public:
  explicit ModuleBuilder(std::string name);

  ModuleBuilder &addInput(std::string name, std::string type);
  ModuleBuilder &addOutput(std::string sourceName);
  ModuleBuilder &addPe(PeSpec pe);
  ModuleBuilder &addSwitch(SwitchSpec sw);
  ModuleBuilder &addFifo(FifoSpec fifo);
  ModuleBuilder &addBoundary(BoundarySpec boundary);
  ModuleBuilder &addMem(MemSpec mem);
  ModuleBuilder &addAttribute(std::string name, std::string value);

  llvm::Error print(llvm::raw_ostream &os) const;

private:
  friend struct ModuleBuilderInternals;

  struct BodyLineSpec {
    std::vector<std::string> fragments;
    std::vector<std::string> operands;
    bool moduleScope = true;
  };

  struct BodyResultSpec {
    std::string name;
    std::string type;
  };

  struct BodyOpSpec {
    std::vector<BodyResultSpec> results;
    std::vector<BodyLineSpec> lines;
  };

  struct Input {
    std::string name;
    std::string type;
  };

  struct Output {
    std::size_t useId;
  };

  struct Attribute {
    std::string name;
    std::string value;
  };

  struct DirectUse {
    std::string sourceName;
  };

  struct PeEntry {
    PeSpec spec;
    std::vector<std::size_t> useIds;
  };

  struct SwitchEntry {
    SwitchSpec spec;
    std::vector<std::size_t> useIds;
  };

  struct MemEntry {
    MemSpec spec;
    std::vector<std::size_t> useIds;
  };

  struct BodyOpEntry {
    BodyOpSpec spec;
    std::vector<std::vector<std::size_t>> lineUseIds;
  };

  struct FifoEntry {
    FifoSpec spec;
    std::size_t useId;
  };

  struct BoundaryEntry {
    BoundarySpec spec;
    std::vector<std::size_t> useIds;
  };

  using BodyEntry = std::variant<BodyOpEntry, FifoEntry, BoundaryEntry>;

  std::size_t registerDirectUse(std::string sourceName);
  ModuleBuilder &addBodyOp(BodyOpSpec op);

  std::string name;
  std::vector<Input> inputs;
  std::vector<Output> outputs;
  std::vector<Attribute> attributes;
  std::vector<DirectUse> directUses;
  std::vector<PeEntry> pes;
  std::vector<SwitchEntry> switches;
  std::vector<MemEntry> mems;
  std::vector<BodyEntry> bodyEntries;
};

struct SystemNodeSpec {
  std::string name;
  std::string kind;
  std::vector<std::string> ports;
  std::string spatialModule;
  std::string scalar;
  std::string function;
  std::optional<std::uint64_t> bytes;
  std::map<std::string, std::uint64_t> params;
};

struct SystemLinkSpec {
  std::string srcNode;
  std::string srcPort;
  std::string srcChannel;
  std::string dstNode;
  std::string dstPort;
  std::string dstChannel;
};

class SystemBuilder {
public:
  explicit SystemBuilder(std::string name, std::string memoryModel);

  SystemBuilder &addHostCore(std::string name, std::string scalar,
                             std::vector<std::string> ports);
  SystemBuilder &addSpatialAccelerator(std::string name,
                                       std::string spatialModule,
                                       std::string scalar,
                                       std::vector<std::string> ports);
  SystemBuilder &addFixedAccelerator(std::string name, std::string function,
                                     std::vector<std::string> ports);
  SystemBuilder &addCache(std::string name, std::uint64_t lineBytes,
                          std::uint64_t capacityBytes,
                          std::vector<std::string> ports);
  SystemBuilder &addDmaEngine(std::string name, std::uint64_t queueDepth,
                              std::vector<std::string> ports);
  SystemBuilder &addMemory(std::string name, std::uint64_t bytes,
                           std::vector<std::string> ports);
  SystemBuilder &connect(std::string srcNode, std::string srcPort,
                         std::string srcChannel, std::string dstNode,
                         std::string dstPort, std::string dstChannel);

  llvm::Error print(llvm::raw_ostream &os) const;

private:
  std::string name;
  std::string memoryModel;
  std::vector<SystemNodeSpec> nodes;
  std::vector<SystemLinkSpec> links;
};

ModuleBuilder buildMinimalSpatialAdg();
ModuleBuilder buildMinimalTemporalAdg();
ModuleBuilder buildSharedReductionAdg();
ModuleBuilder buildSharedMemoryReductionAdg();
ModuleBuilder buildSharedQuantizedWindowAdg();
ModuleBuilder buildSharedSignalWindowAdg();
ModuleBuilder buildSharedVectorAluAdg();
ModuleBuilder buildSharedVectorMathAdg();
ModuleBuilder buildSharedVectorMeshAdg();
ModuleBuilder buildFullSpatialCoreAdg();
SystemBuilder buildHeterogeneousSocAdg();

llvm::Error writeMinimalSpatialAdg(llvm::raw_ostream &os);
llvm::Error writeMinimalTemporalAdg(llvm::raw_ostream &os);
llvm::Error writeSharedReductionAdg(llvm::raw_ostream &os);
llvm::Error writeSharedMemoryReductionAdg(llvm::raw_ostream &os);
llvm::Error writeSharedQuantizedWindowAdg(llvm::raw_ostream &os);
llvm::Error writeSharedSignalWindowAdg(llvm::raw_ostream &os);
llvm::Error writeSharedVectorAluAdg(llvm::raw_ostream &os);
llvm::Error writeSharedVectorMathAdg(llvm::raw_ostream &os);
llvm::Error writeSharedVectorMeshAdg(llvm::raw_ostream &os);
llvm::Error writeFullSpatialCoreAdg(llvm::raw_ostream &os);
llvm::Error writeHeterogeneousSocAdg(llvm::raw_ostream &os);
llvm::Error writeSpatialTopologyMatrixAdg(llvm::raw_ostream &os,
                                          llvm::StringRef family);
llvm::Error writeSystemTopologyMatrixAdg(llvm::raw_ostream &os,
                                         llvm::StringRef family);

} // namespace adg
} // namespace loom

#endif // LOOM_ADG_BUILDER_H

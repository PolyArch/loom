#ifndef LOOM_ADG_BUILDER_H
#define LOOM_ADG_BUILDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace loom {
namespace adg {

enum class Schedule { Spatial, Temporal };

struct PortBinding {
  std::string localName;
  std::string sourceName;
  std::string type;
  std::string castType;
};

struct FabricOpSpec {
  std::vector<std::string> results;
  std::vector<std::string> opList;
  std::vector<std::string> operands;
  std::vector<std::string> operandTypes;
  std::vector<std::string> resultTypes;
  std::map<std::string, std::vector<std::string>> hwParams;
  std::map<std::string, std::string> swConfigs;
};

struct FuSpec {
  std::vector<PortBinding> inputs;
  std::vector<std::string> resultTypes;
  std::vector<FabricOpSpec> operations;
  std::vector<std::string> yieldValues;
};

struct TemporalPeConfig {
  unsigned tagWidth = 0;
  unsigned numInstruction = 0;
  std::string fuConfigMode;
  std::string operandBufferMode;
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

struct MemLoadPort {
  std::string address;
  std::string control;
};

struct MemStorePort {
  std::string address;
  std::string data;
  std::string control;
};

struct MemSpec {
  Schedule schedule = Schedule::Spatial;
  std::string manager;
  std::vector<MemLoadPort> loads;
  std::vector<MemStorePort> stores;
  unsigned temporalTagWidth = 0;
  unsigned temporalAddrTableSize = 0;
};

class ModuleBuilder {
public:
  explicit ModuleBuilder(std::string name);

  ModuleBuilder &addInput(std::string name, std::string type);
  ModuleBuilder &addPe(PeSpec pe);
  ModuleBuilder &addSwitch(SwitchSpec sw);
  ModuleBuilder &addMem(MemSpec mem);
  ModuleBuilder &addExactBodyLine(std::string line);

  llvm::Error print(llvm::raw_ostream &os) const;

private:
  struct Input {
    std::string name;
    std::string type;
  };

  std::string name;
  std::vector<Input> inputs;
  std::vector<PeSpec> pes;
  std::vector<SwitchSpec> switches;
  std::vector<MemSpec> mems;
  std::vector<std::string> exactBodyLines;
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
ModuleBuilder buildFullSpatialCoreAdg();
SystemBuilder buildHeterogeneousSocAdg();

llvm::Error writeMinimalSpatialAdg(llvm::raw_ostream &os);
llvm::Error writeMinimalTemporalAdg(llvm::raw_ostream &os);
llvm::Error writeSharedReductionAdg(llvm::raw_ostream &os);
llvm::Error writeFullSpatialCoreAdg(llvm::raw_ostream &os);
llvm::Error writeHeterogeneousSocAdg(llvm::raw_ostream &os);

} // namespace adg
} // namespace loom

#endif // LOOM_ADG_BUILDER_H

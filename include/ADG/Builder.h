#ifndef LOOM_ADG_BUILDER_H
#define LOOM_ADG_BUILDER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
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

struct PeSpec {
  Schedule schedule = Schedule::Spatial;
  std::vector<PortBinding> inputs;
  std::vector<std::string> resultTypes;
  std::vector<FuSpec> fus;
};

struct MemLoadPort {
  std::string address;
  std::string control;
};

struct MemSpec {
  Schedule schedule = Schedule::Spatial;
  std::string manager;
  std::vector<MemLoadPort> loads;
  unsigned storePorts = 0;
};

class ModuleBuilder {
public:
  explicit ModuleBuilder(std::string name);

  ModuleBuilder &addInput(std::string name, std::string type);
  ModuleBuilder &addPe(PeSpec pe);
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
  std::vector<MemSpec> mems;
  std::vector<std::string> exactBodyLines;
};

ModuleBuilder buildSharedReductionAdg();

llvm::Error writeSharedReductionAdg(llvm::raw_ostream &os);

} // namespace adg
} // namespace loom

#endif // LOOM_ADG_BUILDER_H

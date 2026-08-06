#include "EDA/Adapters/OpenSource/Yosys.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"

#include <cctype>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::open_source {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "yosys_structure_invalid: " + message);
}

bool isPortableIdentifier(llvm::StringRef value) {
  if (value.empty())
    return false;
  const auto isFirst = [](unsigned char character) {
    return std::isalpha(character) || character == '_';
  };
  const auto isRest = [](unsigned char character) {
    return std::isalnum(character) || character == '_' || character == '$';
  };
  if (!isFirst(static_cast<unsigned char>(value.front())))
    return false;
  return llvm::all_of(value.drop_front(), [&](char character) {
    return isRest(static_cast<unsigned char>(character));
  });
}

bool isNonzeroAttribute(const llvm::json::Value &value) {
  if (std::optional<std::int64_t> integer = value.getAsInteger())
    return *integer != 0;
  if (std::optional<bool> boolean = value.getAsBoolean())
    return *boolean;
  if (std::optional<llvm::StringRef> text = value.getAsString())
    return text->contains('1');
  return false;
}

llvm::Expected<const llvm::json::Object &>
requireObject(const llvm::json::Object &object, llvm::StringRef field,
              const llvm::Twine &context) {
  const llvm::json::Object *value = object.getObject(field);
  if (!value)
    return invalid(context + " requires object field '" + field + "'");
  return *value;
}

llvm::Expected<YosysPortGeometry::Direction>
parseDirection(const llvm::json::Value &value, const llvm::Twine &context) {
  const std::optional<llvm::StringRef> direction = value.getAsString();
  if (!direction)
    return invalid(context + " direction must be a string");
  if (*direction == "input")
    return YosysPortGeometry::Direction::Input;
  if (*direction == "output")
    return YosysPortGeometry::Direction::Output;
  if (*direction == "inout")
    return YosysPortGeometry::Direction::Inout;
  return invalid(context + " has an unknown direction");
}

llvm::Expected<std::vector<YosysSignalBit>>
parseBits(const llvm::json::Array &bits, const llvm::Twine &context) {
  std::vector<YosysSignalBit> result;
  result.reserve(bits.size());
  for (const llvm::json::Value &value : bits) {
    if (std::optional<std::int64_t> bit = value.getAsInteger()) {
      if (*bit < 0)
        return invalid(context + " contains a negative signal bit");
      result.push_back(
          YosysSignalBit{static_cast<std::uint64_t>(*bit)});
      continue;
    }
    const std::optional<llvm::StringRef> constant = value.getAsString();
    if (!constant || constant->size() != 1 ||
        (*constant != "0" && *constant != "1" && *constant != "x" &&
         *constant != "z"))
      return invalid(context + " contains an invalid signal bit");
    result.push_back(YosysSignalBit{constant->front()});
  }
  return result;
}

llvm::Expected<YosysPortGeometry>
parsePortGeometry(const llvm::json::Object &port, const llvm::Twine &context) {
  auto direction = parseDirection(*port.get("direction"), context);
  if (!direction)
    return direction.takeError();
  const llvm::json::Array *bits = port.getArray("bits");
  if (!bits || bits->empty())
    return invalid(context + " requires nonempty signal bits");
  auto signalBits = parseBits(*bits, context);
  if (!signalBits)
    return signalBits.takeError();

  YosysPortGeometry geometry{*direction, std::move(*signalBits), 0, false,
                             false};
  if (const llvm::json::Value *offset = port.get("offset")) {
    const std::optional<std::int64_t> value = offset->getAsInteger();
    if (!value)
      return invalid(context + " offset is not an integer");
    geometry.offset = *value;
  }
  for (const auto &[field, destination] :
       {std::pair<llvm::StringRef, bool *>("upto", &geometry.upto),
        std::pair<llvm::StringRef, bool *>("signed", &geometry.isSigned)}) {
    if (const llvm::json::Value *value = port.get(field)) {
      if (std::optional<std::int64_t> integer = value->getAsInteger())
        *destination = *integer != 0;
      else if (std::optional<bool> boolean = value->getAsBoolean())
        *destination = *boolean;
      else
        return invalid(context + " flag '" + field + "' has an invalid value");
    }
  }
  return geometry;
}

} // namespace

llvm::Expected<std::string> renderYosysSynthesisDriver(llvm::StringRef topModule) {
  if (!isPortableIdentifier(topModule))
    return invalid("top module is not a portable HDL identifier");
  std::string driver;
  driver += "read_verilog -sv inputs/design.sv\n";
  driver += "hierarchy -check -top " + topModule.str() + "\n";
  driver += "proc\n";
  driver += "opt\n";
  driver += "check -assert -nolatches\n";
  driver += "write_json outputs/rtl-structure.json\n";
  driver += "synth -top " + topModule.str() + "\n";
  driver += "dfflibmap -liberty inputs/library.lib\n";
  driver += "abc -liberty inputs/library.lib\n";
  driver += "read_liberty -lib inputs/library.lib\n";
  driver += "clean\n";
  driver += "check -assert -nolatches\n";
  driver += "write_verilog -noattr -nodec -simple-lhs outputs/netlist.v\n";
  driver += "design -reset\n";
  driver += "read_liberty -lib inputs/library.lib\n";
  driver += "read_verilog outputs/netlist.v\n";
  driver += "hierarchy -check -top " + topModule.str() + "\n";
  driver += "proc\n";
  driver += "opt\n";
  driver += "check -assert -nolatches\n";
  driver += "write_json outputs/netlist-structure.json\n";
  return driver;
}

llvm::Expected<YosysStructureFacts>
parseYosysStructureFacts(llvm::StringRef contents) {
  auto parsed = llvm::json::parse(contents);
  if (!parsed)
    return invalid("structural JSON is malformed: " +
                   llvm::toString(parsed.takeError()));
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("structural JSON root is not an object");
  auto modules = requireObject(*root, "modules", "structural JSON root");
  if (!modules)
    return modules.takeError();

  YosysStructureFacts structure;
  for (const auto &[name, value] : *modules) {
    const llvm::json::Object *module = value.getAsObject();
    if (!module)
      return invalid("structural JSON module is not an object");
    YosysModuleFacts facts;
    if (const llvm::json::Object *attributes =
            module->getObject("attributes"))
      for (llvm::StringRef attributeName :
           {llvm::StringRef("blackbox"), llvm::StringRef("whitebox")})
        if (const llvm::json::Value *attribute =
                attributes->get(attributeName))
          facts.declaredBox =
              facts.declaredBox || isNonzeroAttribute(*attribute);
    if (const llvm::json::Object *processes = module->getObject("processes"))
      facts.hasProcesses = !processes->empty();
    if (const llvm::json::Object *memories = module->getObject("memories"))
      facts.hasMemories = !memories->empty();
    if (const llvm::json::Object *ports = module->getObject("ports"))
      for (const auto &[portName, portValue] : *ports) {
        const llvm::json::Object *port = portValue.getAsObject();
        if (!port)
          return invalid("module port is not an object");
        auto geometry = parsePortGeometry(
            *port, llvm::Twine("module '") + llvm::StringRef(name) + "' port");
        if (!geometry)
          return geometry.takeError();
        facts.ports.emplace(portName.str(), *geometry);
      }
    if (const llvm::json::Object *cells = module->getObject("cells"))
      for (const auto &[cellName, cellValue] : *cells) {
        const llvm::json::Object *cell = cellValue.getAsObject();
        if (!cell)
          return invalid("module cell is not an object");
        YosysCellFacts cellFacts;
        const std::optional<llvm::StringRef> type = cell->getString("type");
        if (!type)
          return invalid("module cell type must be a string");
        cellFacts.type = type->str();
        auto directions = requireObject(*cell, "port_directions", "module cell");
        if (!directions)
          return directions.takeError();
        for (const auto &[portName, directionValue] : *directions) {
          auto direction = parseDirection(directionValue, "module cell port");
          if (!direction)
            return direction.takeError();
          cellFacts.portDirections.emplace(portName.str(), *direction);
        }
        auto connections = requireObject(*cell, "connections", "module cell");
        if (!connections)
          return connections.takeError();
        for (const auto &[portName, connectionValue] : *connections) {
          const llvm::json::Array *bits = connectionValue.getAsArray();
          if (!bits)
            return invalid("module cell connection requires a bit array");
          auto signalBits =
              parseBits(*bits, llvm::Twine("module cell '") +
                                   llvm::StringRef(cellName) + "' connection");
          if (!signalBits)
            return signalBits.takeError();
          cellFacts.connections.emplace(portName.str(), std::move(*signalBits));
        }
        facts.cells.emplace(cellName.str(), std::move(cellFacts));
      }
    if (const llvm::json::Object *netnames = module->getObject("netnames"))
      for (const auto &[netName, netValue] : *netnames) {
        const llvm::json::Object *net = netValue.getAsObject();
        if (!net || !net->getArray("bits"))
          return invalid("module netname has no signal bits");
        facts.netNames.push_back(netName.str());
      }
    structure.modules.emplace(name.str(), std::move(facts));
  }
  return structure;
}

llvm::Error
validateYosysSynthesizedStructure(const YosysStructureFacts &structure,
                                  llvm::StringRef topModule) {
  const auto topEntry = structure.modules.find(topModule.str());
  if (topEntry == structure.modules.end())
    return invalid("structural JSON does not contain the exact top module");
  const YosysModuleFacts &top = topEntry->second;
  if (top.declaredBox)
    return invalid("structural JSON contains an unexpected blackbox top");
  for (const auto &[name, module] : structure.modules) {
    if (name == topModule)
      continue;
    if (!module.declaredBox)
      return invalid("structural JSON contains an unexpected functional "
                     "module");
  }
  if (top.hasProcesses)
    return invalid("structural JSON contains residual processes");
  if (top.hasMemories)
    return invalid("structural JSON contains residual memories");

  std::map<std::uint64_t, unsigned> drivers;
  const auto recordDrivers =
      [&drivers](const std::vector<YosysSignalBit> &bits) {
        for (const YosysSignalBit &bit : bits)
          if (const std::uint64_t *net =
                  std::get_if<std::uint64_t>(&bit.value))
            ++drivers[*net];
      };
  // Input and inout top port bits drive their nets.
  for (const auto &[name, port] : top.ports)
    if (port.direction != YosysPortGeometry::Direction::Output)
      recordDrivers(port.bits);

  for (const auto &[name, cell] : top.cells) {
    if (llvm::StringRef(cell.type).starts_with("$"))
      return invalid("structural JSON contains an unmapped generic cell '" +
                     cell.type + "'");
    const auto definition = structure.modules.find(cell.type);
    if (definition == structure.modules.end())
      return invalid("structural JSON contains an undeclared cell type '" +
                     cell.type + "'");
    for (const auto &[portName, connection] : cell.connections) {
      if (definition->second.ports.find(portName) ==
          definition->second.ports.end())
        return invalid("structural JSON cell '" + cell.type +
                       "' has no declared port '" + portName + "'");
      const auto direction = cell.portDirections.find(portName);
      if (direction == cell.portDirections.end())
        return invalid("structural JSON cell '" + cell.type +
                       "' connection '" + portName + "' has no direction");
      if (direction->second != YosysPortGeometry::Direction::Input)
        recordDrivers(connection);
    }
  }

  // Every required top output bit needs exactly one defined driver.
  for (const auto &[name, port] : top.ports) {
    if (port.direction == YosysPortGeometry::Direction::Input)
      continue;
    for (const YosysSignalBit &bit : port.bits) {
      if (const char *constant = std::get_if<char>(&bit.value)) {
        if (*constant != '0' && *constant != '1')
          return invalid("required top output '" + name +
                         "' has a non-defined constant");
        continue;
      }
      const std::uint64_t net = std::get<std::uint64_t>(bit.value);
      const auto found = drivers.find(net);
      if (found == drivers.end())
        return invalid("required top output '" + name + "' is undriven");
      if (found->second != 1)
        return invalid("required top output '" + name +
                       "' has multiple drivers");
    }
  }
  return llvm::Error::success();
}

llvm::Error
compareYosysTopPortGeometry(const YosysStructureFacts &preSynthesis,
                            const YosysStructureFacts &postSynthesis,
                            llvm::StringRef topModule) {
  const auto pre = preSynthesis.modules.find(topModule.str());
  const auto post = postSynthesis.modules.find(topModule.str());
  if (pre == preSynthesis.modules.end() ||
      post == postSynthesis.modules.end())
    return invalid("structural JSON does not contain the exact top module");
  if (pre->second.ports != post->second.ports)
    return invalid("synthesized netlist changed the exact top port geometry");
  return llvm::Error::success();
}

} // namespace loom::eda::open_source

#include "EDA/Adapters/OpenSource/Yosys.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <array>
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
  // Explicit ASCII ranges keep the grammar host-locale independent.
  const auto isFirst = [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') || character == '_';
  };
  const auto isRest = [&](char character) {
    return isFirst(character) || (character >= '0' && character <= '9') ||
           character == '$';
  };
  if (value.empty() || !isFirst(value.front()))
    return false;
  return llvm::all_of(value.drop_front(), isRest);
}

enum class YosysTokenPolicy { Quoted, AbcCompatible };

llvm::Expected<std::string> encodeYosysToken(llvm::StringRef value,
                                             YosysTokenPolicy policy) {
  if (value.empty() || value.contains('\0') || value.contains('\n') ||
      value.contains('\r') || value.contains('"') || value.contains('\\'))
    return invalid(
        "Yosys token is empty or cannot be represented consistently");
  const bool bare = llvm::all_of(value, [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') ||
           (character >= '0' && character <= '9') || character == '_' ||
           character == '-' || character == '+' || character == '.' ||
           character == '/';
  });
  if (bare)
    return value.str();
  if (policy == YosysTokenPolicy::AbcCompatible)
    return invalid("Liberty path requires quoting that ABC cannot preserve");
  std::string encoded = "\"";
  for (char character : value)
    encoded.push_back(character);
  encoded.push_back('"');
  return encoded;
}

/// An attribute is a boolean fact only in Yosys's admitted scalar encodings;
/// any other present value fails closed.
llvm::Expected<bool> nonzeroAttribute(const llvm::json::Value &value) {
  if (std::optional<std::int64_t> integer = value.getAsInteger())
    return *integer != 0;
  if (std::optional<bool> boolean = value.getAsBoolean())
    return *boolean;
  if (std::optional<llvm::StringRef> text = value.getAsString())
    return text->contains('1');
  return invalid("attribute value is not an admitted scalar encoding");
}

llvm::Expected<const llvm::json::Object &>
requireObject(const llvm::json::Object &object, llvm::StringRef field,
              const llvm::Twine &context) {
  const llvm::json::Object *value = object.getObject(field);
  if (!value)
    return invalid(context + " requires object field '" + field + "'");
  return *value;
}

/// An optional container is legal only when absent or an object; a present
/// but wrong-typed container never reads as absent.
llvm::Expected<const llvm::json::Object *>
optionalObject(const llvm::json::Object &object, llvm::StringRef field,
               const llvm::Twine &context) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return nullptr;
  const llvm::json::Object *child = value->getAsObject();
  if (!child)
    return invalid(context + " field '" + field + "' must be an object");
  return child;
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
      result.push_back(YosysSignalBit{static_cast<std::uint64_t>(*bit)});
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
  const llvm::json::Value *directionValue = port.get("direction");
  if (!directionValue)
    return invalid(context + " requires a direction");
  auto direction = parseDirection(*directionValue, context);
  if (!direction)
    return direction.takeError();
  const llvm::json::Value *bitsValue = port.get("bits");
  if (!bitsValue)
    return invalid(context + " requires signal bits");
  const llvm::json::Array *bits = bitsValue->getAsArray();
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

llvm::Expected<std::string>
renderYosysSynthesisDriver(llvm::StringRef topModule) {
  const std::array<std::string, 1> sources{"inputs/design.sv"};
  return renderYosysSynthesisDriver(topModule, sources, "inputs/library.lib");
}

namespace {

llvm::Expected<std::string> renderSynthesisDriver(
    llvm::StringRef topModule, llvm::ArrayRef<std::string> rtlSources,
    llvm::StringRef standardCellLiberty, const YosysMappedChildren *children) {
  if (!isPortableIdentifier(topModule))
    return invalid("top module is not a portable HDL identifier");
  if (rtlSources.empty())
    return invalid("RTL source closure is empty");
  std::vector<std::string> encodedSources;
  encodedSources.reserve(rtlSources.size());
  for (const std::string &source : rtlSources) {
    auto encoded = encodeYosysToken(source, YosysTokenPolicy::Quoted);
    if (!encoded)
      return encoded.takeError();
    encodedSources.push_back(std::move(*encoded));
  }
  auto encodedLiberty =
      encodeYosysToken(standardCellLiberty, YosysTokenPolicy::AbcCompatible);
  if (!encodedLiberty)
    return encodedLiberty.takeError();
  std::string mappedLibraries, preserveInstances;
  if (children) {
    if (!std::is_sorted(children->netlistPaths.begin(),
                        children->netlistPaths.end()) ||
        std::adjacent_find(children->netlistPaths.begin(),
                           children->netlistPaths.end()) !=
            children->netlistPaths.end() ||
        !std::is_sorted(children->directModuleNames.begin(),
                        children->directModuleNames.end()) ||
        std::adjacent_find(children->directModuleNames.begin(),
                           children->directModuleNames.end()) !=
            children->directModuleNames.end() ||
        children->netlistPaths.empty() != children->directModuleNames.empty())
      return invalid(
          "mapped child paths and direct definitions are not canonical");
    for (const auto &path : children->netlistPaths) {
      auto encoded = encodeYosysToken(path, YosysTokenPolicy::Quoted);
      if (!encoded)
        return encoded.takeError();
      mappedLibraries += "read_verilog -lib -nowb " + *encoded + "\n";
    }
    for (const auto &name : children->directModuleNames) {
      if (!isPortableIdentifier(name) || name == topModule)
        return invalid(
            "mapped child definition is not distinct from the exact top");
      preserveInstances += "setattr -set keep 1 t:" + name + "\n";
    }
  }
  std::string driver = mappedLibraries;
  for (const std::string &source : encodedSources)
    driver += "read_verilog -sv " + source + "\n";
  driver += "hierarchy -check -top " + topModule.str() + "\n";
  driver += preserveInstances;
  driver += "proc\n";
  driver += "opt\n";
  driver += "check -assert -nolatches\n";
  driver += "write_json " + yosysRtlStructureOutputPath.str() + "\n";
  // The SAT-based sharing pass of the default synth script is intractable on
  // SpatialCore RTL (hours and tens of gigabytes without reaching technology
  // mapping); resource sharing is a quality-of-results heuristic, not part
  // of the netlist contract, so the flow synthesizes without it.
  driver += std::string(children ? "synth -noshare -top "
                                 : "synth -flatten -noshare -top ") +
            topModule.str() + "\n";
  driver += "dfflibmap -liberty " + *encodedLiberty + "\n";
  driver += "abc -liberty " + *encodedLiberty + "\n";
  driver += "read_liberty -lib " + *encodedLiberty + "\n";
  driver += "clean\n";
  driver += "check -assert -nolatches\n";
  driver += "write_verilog -noattr -nodec -simple-lhs " +
            yosysNetlistOutputPath.str() + "\n";
  driver += "design -reset\n";
  driver += "read_liberty -lib " + *encodedLiberty + "\n";
  driver += mappedLibraries;
  driver += "read_verilog " + yosysNetlistOutputPath.str() + "\n";
  driver += "hierarchy -check -top " + topModule.str() + "\n";
  driver += preserveInstances;
  driver += "proc\n";
  driver += "opt\n";
  driver += "check -assert -nolatches\n";
  driver += "write_json " + yosysNetlistStructureOutputPath.str() + "\n";
  return driver;
}

} // namespace

llvm::Expected<std::string>
renderYosysSynthesisDriver(llvm::StringRef topModule,
                           llvm::ArrayRef<std::string> rtlSources,
                           llvm::StringRef standardCellLiberty) {
  return renderSynthesisDriver(topModule, rtlSources, standardCellLiberty,
                               nullptr);
}

llvm::Expected<std::string> renderYosysBlockSynthesisDriver(
    llvm::StringRef topModule, llvm::ArrayRef<std::string> rtlSources,
    llvm::StringRef standardCellLiberty, const YosysMappedChildren &children) {
  return renderSynthesisDriver(topModule, rtlSources, standardCellLiberty,
                               &children);
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
    const std::string context =
        (llvm::Twine("module '") + llvm::StringRef(name) + "'").str();
    YosysModuleFacts facts;
    auto attributes = optionalObject(*module, "attributes", context);
    if (!attributes)
      return attributes.takeError();
    if (*attributes)
      for (llvm::StringRef attributeName :
           {llvm::StringRef("blackbox"), llvm::StringRef("whitebox")})
        if (const llvm::json::Value *attribute =
                (*attributes)->get(attributeName)) {
          auto nonzero = nonzeroAttribute(*attribute);
          if (!nonzero)
            return nonzero.takeError();
          facts.declaredBox = facts.declaredBox || *nonzero;
        }
    auto processes = optionalObject(*module, "processes", context);
    if (!processes)
      return processes.takeError();
    facts.hasProcesses = *processes && !(*processes)->empty();
    auto memories = optionalObject(*module, "memories", context);
    if (!memories)
      return memories.takeError();
    facts.hasMemories = *memories && !(*memories)->empty();
    auto ports = optionalObject(*module, "ports", context);
    if (!ports)
      return ports.takeError();
    if (*ports)
      for (const auto &[portName, portValue] : **ports) {
        const llvm::json::Object *port = portValue.getAsObject();
        if (!port)
          return invalid(context + " port is not an object");
        auto geometry = parsePortGeometry(*port, context + " port");
        if (!geometry)
          return geometry.takeError();
        facts.ports.emplace(portName.str(), std::move(*geometry));
      }
    auto cells = optionalObject(*module, "cells", context);
    if (!cells)
      return cells.takeError();
    if (*cells)
      for (const auto &[cellName, cellValue] : **cells) {
        const llvm::json::Object *cell = cellValue.getAsObject();
        if (!cell)
          return invalid(context + " cell is not an object");
        YosysCellFacts cellFacts;
        const std::optional<llvm::StringRef> type = cell->getString("type");
        if (!type)
          return invalid(context + " cell type must be a string");
        cellFacts.type = type->str();
        auto directions =
            requireObject(*cell, "port_directions", context + " cell");
        if (!directions)
          return directions.takeError();
        for (const auto &[portName, directionValue] : *directions) {
          auto direction =
              parseDirection(directionValue, context + " cell port");
          if (!direction)
            return direction.takeError();
          cellFacts.portDirections.emplace(portName.str(), *direction);
        }
        auto connections =
            requireObject(*cell, "connections", context + " cell");
        if (!connections)
          return connections.takeError();
        for (const auto &[portName, connectionValue] : *connections) {
          const llvm::json::Array *bits = connectionValue.getAsArray();
          if (!bits)
            return invalid(context + " cell connection requires a bit array");
          auto signalBits =
              parseBits(*bits, context + llvm::Twine(" cell '") +
                                   llvm::StringRef(cellName) + "' connection");
          if (!signalBits)
            return signalBits.takeError();
          cellFacts.connections.emplace(portName.str(), std::move(*signalBits));
        }
        facts.cells.emplace(cellName.str(), std::move(cellFacts));
      }
    // Netnames carry no fact the validator consumes; only their shape is
    // checked so a malformed document still fails closed.
    auto netnames = optionalObject(*module, "netnames", context);
    if (!netnames)
      return netnames.takeError();
    if (*netnames)
      for (const auto &[netName, netValue] : **netnames) {
        const llvm::json::Object *net = netValue.getAsObject();
        if (!net || !net->getArray("bits"))
          return invalid(context + " netname has no signal bits");
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
    // A blackbox or whitebox attribute never hides executable structure.
    if (module.hasProcesses || module.hasMemories || !module.cells.empty())
      return invalid("structural JSON box module '" + name +
                     "' hides a structural body");
  }
  if (top.hasProcesses)
    return invalid("structural JSON contains residual processes");
  if (top.hasMemories)
    return invalid("structural JSON contains residual memories");

  std::map<std::uint64_t, unsigned> drivers;
  const auto recordDrivers =
      [&drivers](const std::vector<YosysSignalBit> &bits) {
        for (const YosysSignalBit &bit : bits)
          if (const std::uint64_t *net = std::get_if<std::uint64_t>(&bit.value))
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
    if (cell.type == topModule)
      return invalid("structural JSON cell instantiates the functional top");
    const auto definition = structure.modules.find(cell.type);
    if (definition == structure.modules.end())
      return invalid("structural JSON contains an undeclared cell type '" +
                     cell.type + "'");
    for (const auto &[portName, connection] : cell.connections) {
      if (connection.empty())
        return invalid("structural JSON cell '" + cell.type + "' connection '" +
                       portName + "' is empty");
      const auto definitionPort = definition->second.ports.find(portName);
      if (definitionPort == definition->second.ports.end())
        return invalid("structural JSON cell '" + cell.type +
                       "' has no declared port '" + portName + "'");
      const auto direction = cell.portDirections.find(portName);
      if (direction == cell.portDirections.end())
        return invalid("structural JSON cell '" + cell.type + "' connection '" +
                       portName + "' has no direction");
      if (direction->second != definitionPort->second.direction)
        return invalid("structural JSON cell '" + cell.type + "' connection '" +
                       portName + "' direction does not match its definition");
      if (connection.size() != definitionPort->second.bits.size())
        return invalid("structural JSON cell '" + cell.type + "' connection '" +
                       portName + "' width does not match its definition");
      if (direction->second != YosysPortGeometry::Direction::Input)
        recordDrivers(connection);
    }
  }

  // The exact-one-defined-driver obligation applies only to exact Output
  // ports: this counter does not model tri-state ownership and never proves
  // an inout against itself. Input and inout bits stay driver sources.
  for (const auto &[name, port] : top.ports) {
    if (port.direction != YosysPortGeometry::Direction::Output)
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
  if (pre == preSynthesis.modules.end() || post == postSynthesis.modules.end())
    return invalid("structural JSON does not contain the exact top module");
  if (pre->second.ports.size() != post->second.ports.size())
    return invalid("synthesized netlist changed the exact top port geometry");
  auto prePort = pre->second.ports.begin();
  auto postPort = post->second.ports.begin();
  for (; prePort != pre->second.ports.end(); ++prePort, ++postPort)
    if (prePort->first != postPort->first ||
        prePort->second.direction != postPort->second.direction ||
        prePort->second.bits.size() != postPort->second.bits.size())
      return invalid("synthesized netlist changed the exact top port geometry");
  return llvm::Error::success();
}

} // namespace loom::eda::open_source

#include "Hardware/RTL/RtlModuleGraph.h"

#include "circt/Conversion/ExportVerilog.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Support/InstanceGraphInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace loom::hardware::rtl {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "rtl_module_graph_invalid: " + message);
}

std::string printType(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

std::string printAttribute(mlir::Attribute attribute) {
  if (!attribute)
    return {};
  std::string text;
  llvm::raw_string_ostream stream(text);
  attribute.print(stream);
  return text;
}

RtlModulePortDirection portDirection(circt::hw::ModulePort::Direction value) {
  switch (value) {
  case circt::hw::ModulePort::Input:
    return RtlModulePortDirection::Input;
  case circt::hw::ModulePort::Output:
    return RtlModulePortDirection::Output;
  case circt::hw::ModulePort::InOut:
    return RtlModulePortDirection::Inout;
  }
  llvm_unreachable("unknown HW module port direction");
}

bool canonicalName(llvm::StringRef name) {
  if (name.empty() ||
      !(std::isalpha(static_cast<unsigned char>(name.front())) ||
        name.front() == '_'))
    return false;
  return llvm::all_of(name.drop_front(), [](char character) {
    const unsigned char value = static_cast<unsigned char>(character);
    return std::isalnum(value) || character == '_' || character == '$';
  });
}

struct DefinitionDraft final {
  RtlModuleProjection projection;
  std::map<std::string, std::uint64_t> dependencies;
};

llvm::Expected<std::vector<RtlModulePortProjection>>
projectPorts(circt::hw::HWModuleLike module) {
  std::vector<RtlModulePortProjection> result;
  result.reserve(module.getNumPorts());
  for (const circt::hw::PortInfo &port : module.getPortList()) {
    if (port.getName().empty())
      return invalid("module port has no exact name");
    llvm::SmallVector<mlir::NamedAttribute> semanticAttributes;
    if (port.attrs)
      for (mlir::NamedAttribute attribute : port.attrs)
        if (attribute.getName().strref() != "hw.verilogName")
          semanticAttributes.push_back(attribute);
    const mlir::DictionaryAttr attributes =
        mlir::DictionaryAttr::get(module.getContext(), semanticAttributes);
    result.push_back(RtlModulePortProjection{
        port.getName().str(), printType(port.type), printAttribute(attributes),
        portDirection(port.dir)});
  }
  return result;
}

llvm::Expected<DefinitionDraft> projectDefinition(mlir::Operation *operation) {
  auto module = llvm::dyn_cast<circt::hw::HWModuleLike>(operation);
  if (!module)
    return invalid("definition is not an HW module");
  auto symbol = mlir::SymbolTable::getSymbolName(operation);
  if (!symbol || symbol.getValue().empty())
    return invalid("HW module has no exact IR symbol");
  const llvm::StringRef emitted = circt::hw::getVerilogModuleName(operation);
  if (!canonicalName(emitted))
    return invalid("HW module has a noncanonical emitted name");
  auto ports = projectPorts(module);
  if (!ports)
    return ports.takeError();
  RtlModuleDefinitionKind kind;
  mlir::ArrayAttr parameters;
  if (auto concrete = llvm::dyn_cast<circt::hw::HWModuleOp>(operation)) {
    kind = RtlModuleDefinitionKind::Concrete;
    parameters = concrete.getParametersAttr();
  } else if (auto external =
                 llvm::dyn_cast<circt::hw::HWModuleExternOp>(operation)) {
    kind = RtlModuleDefinitionKind::External;
    parameters = external.getParametersAttr();
  } else {
    return invalid("generated HW module survived specialization");
  }
  return DefinitionDraft{RtlModuleProjection{symbol.getValue().str(),
                                             emitted.str(),
                                             kind,
                                             false,
                                             std::move(*ports),
                                             printAttribute(parameters),
                                             {},
                                             std::nullopt},
                         {}};
}

bool sameStructure(const RtlModuleGraphProjection &lhs,
                   const RtlModuleGraphProjection &rhs) {
  if (lhs.topModule != rhs.topModule ||
      lhs.modules.size() != rhs.modules.size())
    return false;
  for (std::size_t index = 0; index != lhs.modules.size(); ++index) {
    const RtlModuleProjection &a = lhs.modules[index];
    const RtlModuleProjection &b = rhs.modules[index];
    if (a.irSymbol != b.irSymbol || a.emittedName != b.emittedName ||
        a.kind != b.kind || a.reachable != b.reachable || a.ports != b.ports ||
        a.parameters != b.parameters || a.dependencies != b.dependencies)
      return false;
  }
  return true;
}

llvm::Error markReachableAndCheckDag(RtlModuleGraphProjection &graph) {
  enum class Color : std::uint8_t { Unvisited, Active, Complete };
  std::vector<Color> colors(graph.modules.size(), Color::Unvisited);
  const auto visit = [&](auto &&self, std::size_t module,
                         bool reachable) -> llvm::Error {
    if (module >= graph.modules.size())
      return invalid("module dependency ordinal is out of range");
    if (colors[module] == Color::Active)
      return invalid("HW module instance graph contains a cycle");
    if (reachable)
      graph.modules[module].reachable = true;
    if (colors[module] == Color::Complete)
      return llvm::Error::success();
    colors[module] = Color::Active;
    for (const RtlModuleDependency &dependency :
         graph.modules[module].dependencies) {
      if (dependency.multiplicity == 0)
        return invalid("module dependency has zero multiplicity");
      if (llvm::Error error = self(self, dependency.targetModule, reachable))
        return error;
    }
    colors[module] = Color::Complete;
    return llvm::Error::success();
  };
  if (llvm::Error error = visit(visit, graph.topModule, true))
    return error;
  for (std::size_t module = 0; module != graph.modules.size(); ++module)
    if (colors[module] == Color::Unvisited)
      if (llvm::Error error = visit(visit, module, false))
        return error;
  return llvm::Error::success();
}

std::string outputFileName(const RtlModuleProjection &module) {
  return module.emittedName + ".sv";
}

std::string outputMarker(llvm::StringRef filename) {
  return ("\n// ----- 8< ----- FILE \"" + filename + "\" ----- 8< -----\n\n")
      .str();
}

class FramedEmissionOstream final : public llvm::raw_ostream {
public:
  static llvm::Expected<std::unique_ptr<FramedEmissionOstream>>
  create(llvm::raw_ostream &output, const RtlModuleGraphProjection &graph) {
    auto source = BlobDigestBuilder::create();
    if (!source)
      return source.takeError();
    auto preamble = BlobDigestBuilder::create();
    if (!preamble)
      return preamble.takeError();
    auto stream =
        std::unique_ptr<FramedEmissionOstream>(new FramedEmissionOstream(
            output, graph, std::move(*source), std::move(*preamble)));
    return stream;
  }

  llvm::Expected<RtlModuleGraphProjection>
  finish(RtlModuleGraphProjection graph) {
    flush();
    if (!error_.empty())
      return invalid(error_);
    if (!pending_.empty()) {
      if (llvm::StringRef(pending_).starts_with(genericMarkerPrefix))
        return invalid("truncated CIRCT output-file framing");
      confirmed_.append(pending_);
      pending_.clear();
    }
    flushConfirmed();
    if (!error_.empty())
      return invalid(error_);
    finishActive(position_);
    if (!error_.empty())
      return invalid(error_);

    for (std::size_t index = 0; index != graph.modules.size(); ++index) {
      const bool concrete =
          graph.modules[index].kind == RtlModuleDefinitionKind::Concrete;
      if (concrete != states_[index].observed)
        return invalid(concrete ? "concrete module was not emitted"
                                : "external module unexpectedly emitted");
      if (concrete)
        graph.modules[index].emission = std::move(states_[index].range);
    }
    auto preambleDigest = preambleDigest_.finish();
    if (!preambleDigest)
      return preambleDigest.takeError();
    graph.preamble.emplace(0, preambleByteCount_, *preambleDigest);
    auto sourceDigest = sourceDigest_.finish();
    if (!sourceDigest)
      return sourceDigest.takeError();
    graph.sourceDigest = *sourceDigest;
    graph.sourceByteCount = position_;
    graph.framingByteCount = framingByteCount_;

    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    ranges.reserve(graph.modules.size());
    for (const RtlModuleProjection &module : graph.modules)
      if (module.emission) {
        if (module.emission->offset > graph.sourceByteCount ||
            module.emission->byteCount >
                graph.sourceByteCount - module.emission->offset)
          return invalid("module emission range is outside the RTL source");
        ranges.emplace_back(module.emission->offset,
                            module.emission->offset +
                                module.emission->byteCount);
      }
    llvm::sort(ranges);
    for (std::size_t index = 1; index != ranges.size(); ++index)
      if (ranges[index - 1].second > ranges[index].first)
        return invalid("module emission ranges overlap");

    std::uint64_t accounted = preambleByteCount_ + framingByteCount_;
    for (const RtlModuleProjection &module : graph.modules)
      if (module.emission) {
        if (accounted > std::numeric_limits<std::uint64_t>::max() -
                            module.emission->byteCount)
          return invalid("emitted byte accounting overflow");
        accounted += module.emission->byteCount;
      }
    if (accounted != graph.sourceByteCount)
      return invalid("module ranges do not cover the framed RTL source");
    return graph;
  }

private:
  struct State final {
    bool observed = false;
    std::uint64_t offset = 0;
    std::uint64_t byteCount = 0;
    std::optional<BlobDigestBuilder> digest;
    std::optional<RtlModuleEmissionRange> range;
  };

  FramedEmissionOstream(llvm::raw_ostream &output,
                        const RtlModuleGraphProjection &graph,
                        BlobDigestBuilder sourceDigest,
                        BlobDigestBuilder preambleDigest)
      : output_(output), sourceDigest_(std::move(sourceDigest)),
        preambleDigest_(std::move(preambleDigest)),
        states_(graph.modules.size()) {
    SetUnbuffered();
    for (std::size_t index = 0; index != graph.modules.size(); ++index) {
      if (graph.modules[index].kind != RtlModuleDefinitionKind::Concrete)
        continue;
      std::string marker = outputMarker(outputFileName(graph.modules[index]));
      if (!markers_.emplace(std::move(marker), index).second)
        error_ = "two concrete modules have the same output framing";
    }
  }

  void write_impl(const char *data, std::size_t size) override {
    output_.write(data, size);
    if (llvm::Error error = sourceDigest_.update(llvm::ArrayRef<std::uint8_t>(
            reinterpret_cast<const std::uint8_t *>(data), size))) {
      if (error_.empty())
        error_ = llvm::toString(std::move(error));
      else
        llvm::consumeError(std::move(error));
    }
    if (!error_.empty()) {
      position_ += size;
      return;
    }
    for (char character : llvm::StringRef(data, size)) {
      ++position_;
      pending_.push_back(character);
      processPending();
    }
    flushConfirmed();
  }

  std::uint64_t current_pos() const override { return position_; }

  void processPending() {
    while (!pending_.empty() && error_.empty()) {
      if (llvm::StringRef(genericMarkerPrefix).starts_with(pending_))
        return;
      if (!llvm::StringRef(pending_).starts_with(genericMarkerPrefix)) {
        confirmed_.push_back(pending_.front());
        pending_.erase(pending_.begin());
        continue;
      }
      auto candidate = llvm::find_if(markers_, [&](const auto &entry) {
        return llvm::StringRef(entry.first).starts_with(pending_);
      });
      if (candidate != markers_.end()) {
        if (candidate->first.size() == pending_.size()) {
          flushConfirmed();
          const std::uint64_t markerStart = position_ - pending_.size();
          finishActive(markerStart);
          State &state = states_[candidate->second];
          if (state.observed) {
            error_ = "CIRCT emitted one module output more than once";
            return;
          }
          auto digest = BlobDigestBuilder::create();
          if (!digest) {
            error_ = llvm::toString(digest.takeError());
            return;
          }
          state.observed = true;
          state.offset = position_;
          state.digest.emplace(std::move(*digest));
          activeModule_ = candidate->second;
          framingByteCount_ += pending_.size();
          pending_.clear();
        }
        return;
      }
      error_ = "CIRCT emitted an unknown output-file frame";
    }
  }

  void flushConfirmed() {
    if (confirmed_.empty() || !error_.empty())
      return;
    llvm::ArrayRef<std::uint8_t> bytes(
        reinterpret_cast<const std::uint8_t *>(confirmed_.data()),
        confirmed_.size());
    BlobDigestBuilder *digest = &preambleDigest_;
    std::uint64_t *count = &preambleByteCount_;
    if (activeModule_) {
      State &state = states_[*activeModule_];
      digest = &*state.digest;
      count = &state.byteCount;
    }
    if (llvm::Error error = digest->update(bytes)) {
      error_ = llvm::toString(std::move(error));
      return;
    }
    *count += confirmed_.size();
    confirmed_.clear();
  }

  void finishActive(std::uint64_t endOffset) {
    if (!activeModule_ || !error_.empty())
      return;
    State &state = states_[*activeModule_];
    if (!state.digest || endOffset < state.offset ||
        endOffset - state.offset != state.byteCount) {
      error_ = "module output range accounting is inconsistent";
      return;
    }
    auto digest = state.digest->finish();
    if (!digest) {
      error_ = llvm::toString(digest.takeError());
      return;
    }
    state.range.emplace(state.offset, state.byteCount, *digest);
    state.digest.reset();
    activeModule_.reset();
  }

  static constexpr llvm::StringLiteral genericMarkerPrefix =
      "\n// ----- 8< ----- FILE \"";

  llvm::raw_ostream &output_;
  BlobDigestBuilder sourceDigest_;
  BlobDigestBuilder preambleDigest_;
  std::map<std::string, std::size_t> markers_;
  std::vector<State> states_;
  std::optional<std::size_t> activeModule_;
  std::string pending_;
  std::string confirmed_;
  std::string error_;
  std::uint64_t position_ = 0;
  std::uint64_t preambleByteCount_ = 0;
  std::uint64_t framingByteCount_ = 0;
};

llvm::Error prepareOutputFiles(mlir::ModuleOp module,
                               const RtlModuleGraphProjection &graph) {
  for (const RtlModuleProjection &definition : graph.modules) {
    if (definition.kind != RtlModuleDefinitionKind::Concrete)
      continue;
    auto operation =
        module.lookupSymbol<circt::hw::HWModuleOp>(definition.irSymbol);
    if (!operation)
      return invalid("concrete module disappeared before emission");
    if (operation->hasAttr("output_file"))
      return invalid("module already has an independently owned output file");
    operation->setAttr(
        "output_file",
        circt::hw::OutputFileAttr::getFromFilename(
            module.getContext(), outputFileName(definition), true, false));
  }
  return llvm::Error::success();
}

} // namespace

llvm::Expected<RtlModuleGraphProjection>
projectRtlModuleGraph(mlir::ModuleOp module, llvm::StringRef exactTopModule) {
  if (exactTopModule.empty())
    return invalid("exact top module is empty");
  if (!module.getOps<circt::hw::HWModuleGeneratedOp>().empty())
    return invalid("generated HW module survived specialization");
  if (!module.getOps<circt::sv::SVVerbatimModuleOp>().empty())
    return invalid("verbatim module is outside the HW symbol graph");

  std::map<std::string, DefinitionDraft> definitions;
  for (mlir::Operation &operation : *module.getBody()) {
    if (!llvm::isa<circt::hw::HWModuleOp, circt::hw::HWModuleExternOp>(
            operation))
      continue;
    auto definition = projectDefinition(&operation);
    if (!definition)
      return definition.takeError();
    if (!definitions
             .emplace(definition->projection.irSymbol, std::move(*definition))
             .second)
      return invalid("HW module IR symbol is duplicated");
  }
  if (definitions.empty())
    return invalid("HW module catalog is empty");
  if (!definitions.count(exactTopModule.str()))
    return invalid("exact top module is absent");

  circt::hw::InstanceGraph instanceGraph(module);
  for (auto &[symbol, definition] : definitions) {
    if (definition.projection.kind != RtlModuleDefinitionKind::Concrete)
      continue;
    auto *node = instanceGraph.lookupOrNull(
        mlir::StringAttr::get(module.getContext(), symbol));
    if (!node)
      return invalid("concrete module is absent from the HW instance graph");
    for (circt::igraph::InstanceRecord *record : *node) {
      auto instance = record->getInstance();
      if (!instance || instance.getReferencedModuleNames().size() != 1)
        return invalid("multi-target instance is outside the exact DAG");
      auto *target = record->getTarget();
      if (!target || !target->getModule())
        return invalid("instance target has no HW module definition");
      const std::string targetSymbol =
          target->getModule().getModuleName().str();
      if (!definitions.count(targetSymbol))
        return invalid("instance target is outside the HW module catalog");
      std::uint64_t &multiplicity = definition.dependencies[targetSymbol];
      if (multiplicity == std::numeric_limits<std::uint64_t>::max())
        return invalid("module dependency multiplicity overflow");
      ++multiplicity;
    }
  }

  std::vector<DefinitionDraft *> ordered;
  ordered.reserve(definitions.size());
  for (auto &[symbol, definition] : definitions)
    ordered.push_back(&definition);
  llvm::sort(
      ordered, [](const DefinitionDraft *lhs, const DefinitionDraft *rhs) {
        return std::tie(lhs->projection.emittedName, lhs->projection.irSymbol) <
               std::tie(rhs->projection.emittedName, rhs->projection.irSymbol);
      });
  for (std::size_t index = 1; index != ordered.size(); ++index)
    if (ordered[index - 1]->projection.emittedName ==
        ordered[index]->projection.emittedName)
      return invalid("two HW modules have the same emitted name");

  std::map<std::string, std::size_t> ordinals;
  RtlModuleGraphProjection graph;
  graph.modules.reserve(ordered.size());
  for (DefinitionDraft *definition : ordered) {
    const std::size_t ordinal = graph.modules.size();
    ordinals.emplace(definition->projection.irSymbol, ordinal);
    graph.modules.push_back(definition->projection);
  }
  for (DefinitionDraft *definition : ordered) {
    RtlModuleProjection &projection =
        graph.modules[ordinals.at(definition->projection.irSymbol)];
    for (const auto &[target, multiplicity] : definition->dependencies)
      projection.dependencies.push_back(
          RtlModuleDependency{ordinals.at(target), multiplicity});
    llvm::sort(projection.dependencies,
               [](RtlModuleDependency lhs, RtlModuleDependency rhs) {
                 return lhs.targetModule < rhs.targetModule;
               });
  }
  graph.topModule = ordinals.at(exactTopModule.str());
  if (llvm::Error error = markReachableAndCheckDag(graph))
    return std::move(error);
  return graph;
}

llvm::Expected<RtlModuleGraphSourceBinding>
bindRtlModuleGraphSource(const RtlModuleGraphProjection &graph,
                         llvm::StringRef source) {
  const auto digestOf = [](llvm::StringRef bytes) {
    return computeBlobDigest(llvm::ArrayRef<std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
  };
  if (!graph.sourceDigest || graph.topModule >= graph.modules.size())
    return invalid("module graph is not bound to emitted source");
  if (source.size() != graph.sourceByteCount ||
      digestOf(source) != *graph.sourceDigest)
    return invalid("RTL payload does not match the module graph source");

  const auto verifyRange =
      [&](const RtlModuleEmissionRange &range,
          llvm::StringRef owner) -> llvm::Expected<llvm::StringRef> {
    if (range.offset > source.size() ||
        range.byteCount > source.size() - range.offset)
      return invalid(owner + " source range is outside the RTL payload");
    const llvm::StringRef bytes =
        source.substr(static_cast<std::size_t>(range.offset),
                      static_cast<std::size_t>(range.byteCount));
    if (digestOf(bytes) != range.digest)
      return invalid(owner + " source range digest is inconsistent");
    return bytes;
  };
  const auto saturatingAdd = [](std::uint64_t lhs, std::uint64_t rhs) {
    return lhs > std::numeric_limits<std::uint64_t>::max() - rhs
               ? std::numeric_limits<std::uint64_t>::max()
               : lhs + rhs;
  };

  std::uint64_t accounted = graph.framingByteCount;
  llvm::StringRef preamble;
  if (graph.preamble) {
    if (graph.preamble->offset != 0)
      return invalid("RTL preamble does not start at byte zero");
    auto bytes = verifyRange(*graph.preamble, "RTL preamble");
    if (!bytes)
      return bytes.takeError();
    preamble = *bytes;
    accounted = saturatingAdd(accounted, graph.preamble->byteCount);
  }
  RtlModuleGraphSourceBinding binding{source, preamble, {}};
  binding.moduleBytes_.resize(graph.modules.size());
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  if (!preamble.empty())
    ranges.emplace_back(0, preamble.size());
  for (std::size_t ordinal = 0; ordinal != graph.modules.size(); ++ordinal) {
    const RtlModuleProjection &module = graph.modules[ordinal];
    if (module.kind == RtlModuleDefinitionKind::External) {
      if (module.emission)
        return invalid("external module unexpectedly owns source bytes");
      continue;
    }
    if (!module.emission)
      return invalid("concrete module has no emitted source range");
    auto bytes = verifyRange(*module.emission, module.emittedName);
    if (!bytes)
      return bytes.takeError();
    binding.moduleBytes_[ordinal] = *bytes;
    accounted = saturatingAdd(accounted, module.emission->byteCount);
    ranges.emplace_back(
        module.emission->offset,
        saturatingAdd(module.emission->offset, module.emission->byteCount));
  }
  if (accounted != graph.sourceByteCount)
    return invalid("module ranges do not cover the exact RTL payload");
  llvm::sort(ranges);
  for (std::size_t index = 1; index < ranges.size(); ++index)
    if (ranges[index - 1].second > ranges[index].first)
      return invalid("module emission ranges overlap");
  return binding;
}

llvm::Expected<RtlModuleGraphProjection>
exportFramedRtlModuleGraph(mlir::ModuleOp module,
                           const RtlModuleGraphProjection &before,
                           llvm::raw_ostream &output) {
  if (before.topModule >= before.modules.size())
    return invalid("pre-export graph has an invalid top module");
  if (llvm::Error error = prepareOutputFiles(module, before))
    return std::move(error);
  auto tracked = FramedEmissionOstream::create(output, before);
  if (!tracked)
    return tracked.takeError();
  if (mlir::failed(circt::exportVerilog(module, **tracked)))
    return invalid("ExportVerilog rejected the framed module graph");
  auto emitted = (*tracked)->finish(before);
  if (!emitted)
    return emitted.takeError();
  auto after =
      projectRtlModuleGraph(module, before.modules[before.topModule].irSymbol);
  if (!after)
    return after.takeError();
  if (!sameStructure(before, *after))
    return invalid("HW module graph changed during Verilog emission");
  return emitted;
}

} // namespace loom::hardware::rtl

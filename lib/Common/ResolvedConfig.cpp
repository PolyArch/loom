#include "Common/ResolvedConfig.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <set>
#include <string>

using llvm::StringRef;

namespace {

llvm::Error makeErr(const llvm::Twine &msg) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 msg.str().c_str());
}

llvm::Error diagnostic(llvm::StringRef code, llvm::StringRef key,
                       llvm::StringRef detail = "") {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << code;
  if (!key.empty())
    os << ": " << key;
  if (!detail.empty())
    os << ": " << detail;
  return makeErr(os.str());
}

StringRef stripQuotes(StringRef value) {
  value = value.trim();
  if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') &&
      value.front() == value.back())
    return value.drop_front().drop_back();
  return value;
}

bool isQuotedScalar(llvm::yaml::Node *node) {
  auto *scalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(node);
  if (!scalar)
    return false;
  const char *begin = scalar->getSourceRange().Start.getPointer();
  const char *end = scalar->getSourceRange().End.getPointer();
  if (!begin || !end)
    return false;
  while (begin < end && std::isspace(static_cast<unsigned char>(*begin)))
    ++begin;
  return begin < end && (*begin == '"' || *begin == '\'');
}

template <unsigned N>
StringRef scalarValue(llvm::yaml::Node *node, llvm::SmallString<N> &storage) {
  auto *scalar = llvm::dyn_cast_or_null<llvm::yaml::ScalarNode>(node);
  if (!scalar)
    return {};
  return stripQuotes(scalar->getValue(storage));
}

llvm::Expected<std::string> requireScalarString(llvm::yaml::Node *node,
                                                llvm::StringRef key) {
  llvm::SmallString<64> storage;
  StringRef value = scalarValue(node, storage);
  if (value.empty())
    return diagnostic("config_type_mismatch", key, "expected scalar string");
  return value.str();
}

llvm::Expected<unsigned> requireUnsigned(llvm::yaml::Node *node,
                                         llvm::StringRef key,
                                         bool allowZero = false) {
  if (isQuotedScalar(node))
    return diagnostic("config_type_mismatch", key,
                      "expected unsigned integer, got string");
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  std::uint64_t value = 0;
  if (StringRef(*valueOrErr).getAsInteger(10, value) ||
      (!allowZero && value == 0))
    return diagnostic("config_type_mismatch", key,
                      "expected positive unsigned integer");
  if (value > std::numeric_limits<unsigned>::max())
    return diagnostic("config_range_violation", key,
                      "unsigned integer exceeds supported range");
  return static_cast<unsigned>(value);
}

llvm::Expected<std::uint64_t> requireUInt64(llvm::yaml::Node *node,
                                            llvm::StringRef key) {
  if (isQuotedScalar(node))
    return diagnostic("config_type_mismatch", key,
                      "expected unsigned integer, got string");
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  std::uint64_t value = 0;
  if (StringRef(*valueOrErr).getAsInteger(10, value))
    return diagnostic("config_type_mismatch", key, "expected unsigned integer");
  return value;
}

llvm::Expected<double> requireDouble(llvm::yaml::Node *node,
                                     llvm::StringRef key) {
  if (isQuotedScalar(node))
    return diagnostic("config_type_mismatch", key,
                      "expected number, got string");
  auto valueOrErr = requireScalarString(node, key);
  if (!valueOrErr)
    return valueOrErr.takeError();
  double value = 0.0;
  if (StringRef(*valueOrErr).getAsDouble(value))
    return diagnostic("config_type_mismatch", key, "expected number");
  return value;
}

struct FabricTechMapAlgorithmName {
  loom::FabricTechMapAlgorithm algorithm;
  llvm::StringLiteral name;
};

constexpr FabricTechMapAlgorithmName fabricTechMapAlgorithmNames[] = {
    {loom::FabricTechMapAlgorithm::Greedy, "greedy"},
    {loom::FabricTechMapAlgorithm::List, "list"},
    {loom::FabricTechMapAlgorithm::Beam, "beam"},
    {loom::FabricTechMapAlgorithm::SimulatedAnnealing, "sa"},
    {loom::FabricTechMapAlgorithm::ILP, "ilp"},
};

llvm::Expected<loom::FabricTechMapAlgorithm>
parseFabricTechMapAlgorithm(StringRef value, StringRef canonicalKey) {
  for (const FabricTechMapAlgorithmName &entry : fabricTechMapAlgorithmNames)
    if (entry.name == value)
      return entry.algorithm;
  return diagnostic("config_unknown_policy", canonicalKey, value);
}

StringRef fabricTechMapAlgorithmName(loom::FabricTechMapAlgorithm algorithm) {
  for (const FabricTechMapAlgorithmName &entry : fabricTechMapAlgorithmNames)
    if (entry.algorithm == algorithm)
      return entry.name;
  llvm_unreachable("invalid resolved Fabric TechMapping algorithm");
}

bool isKnownObjective(StringRef value) {
  return value == "minimize_runtime" || value == "maximize_throughput" ||
         value == "maximize_performance_per_watt" ||
         value == "maximize_performance_per_area" || value == "minimize_area" ||
         value == "minimize_dynamic_power" ||
         value == "minimize_leakage_power" ||
         value == "minimize_unsupported_scope_diagnostics" ||
         value == "minimize_energy" || value == "minimize_power";
}

bool isKnownRankingPolicy(StringRef value) {
  return value == "weighted_sum" || value == "lexicographic" ||
         value == "pareto";
}

struct ConfigPatch {
  std::optional<std::string> configId;
  std::optional<unsigned> addrBits;
  std::optional<unsigned> indexWidth;
  std::optional<unsigned> memBusWidth;
  std::optional<double> alpha;
  std::optional<double> beta;
  std::optional<double> gamma;
  std::optional<loom::FabricTechMapAlgorithm> algorithm;
  std::optional<unsigned> beamWidth;
  std::optional<unsigned> saSteps;
  std::optional<std::uint64_t> saSeed;
  std::optional<unsigned> threads;
  std::optional<std::string> rankingPolicy;
  std::optional<std::vector<loom::ResolvedDseObjective>> objectives;
  std::set<std::string> touchedKeys;
};

llvm::Error touch(ConfigPatch &patch, llvm::StringRef canonicalKey) {
  patch.touchedKeys.insert(canonicalKey.str());
  return llvm::Error::success();
}

llvm::Error checkDuplicateKey(llvm::StringSet<> &seen, llvm::StringRef prefix,
                              llvm::StringRef key) {
  if (key.empty())
    return diagnostic("config_type_mismatch", prefix,
                      "mapping key must be a scalar");
  if (!seen.insert(key).second) {
    std::string fullKey =
        prefix.empty() ? key.str() : (prefix + "." + key).str();
    return diagnostic("config_duplicate_key", fullKey);
  }
  return llvm::Error::success();
}

llvm::Error mergeSiblingPatch(ConfigPatch &dst, const ConfigPatch &src) {
  for (const std::string &key : src.touchedKeys)
    if (dst.touchedKeys.count(key) != 0)
      return diagnostic("config_conflicting_sources", key);

  if (src.configId)
    dst.configId = src.configId;
  if (src.addrBits)
    dst.addrBits = src.addrBits;
  if (src.indexWidth)
    dst.indexWidth = src.indexWidth;
  if (src.memBusWidth)
    dst.memBusWidth = src.memBusWidth;
  if (src.alpha)
    dst.alpha = src.alpha;
  if (src.beta)
    dst.beta = src.beta;
  if (src.gamma)
    dst.gamma = src.gamma;
  if (src.algorithm)
    dst.algorithm = src.algorithm;
  if (src.beamWidth)
    dst.beamWidth = src.beamWidth;
  if (src.saSteps)
    dst.saSteps = src.saSteps;
  if (src.saSeed)
    dst.saSeed = src.saSeed;
  if (src.threads)
    dst.threads = src.threads;
  if (src.rankingPolicy)
    dst.rankingPolicy = src.rankingPolicy;
  if (src.objectives)
    dst.objectives = src.objectives;
  dst.touchedKeys.insert(src.touchedKeys.begin(), src.touchedKeys.end());
  return llvm::Error::success();
}

void applyPatch(loom::ResolvedConfig &config, const ConfigPatch &patch) {
  if (patch.configId)
    config.configId = *patch.configId;
  if (patch.addrBits)
    config.global.addrBits = *patch.addrBits;
  if (patch.indexWidth)
    config.global.indexWidth = *patch.indexWidth;
  if (patch.memBusWidth)
    config.global.memBusWidth = *patch.memBusWidth;
  if (patch.alpha)
    config.fabricTechMap.alpha = *patch.alpha;
  if (patch.beta)
    config.fabricTechMap.beta = *patch.beta;
  if (patch.gamma)
    config.fabricTechMap.gamma = *patch.gamma;
  if (patch.algorithm)
    config.fabricTechMap.algorithm = *patch.algorithm;
  if (patch.beamWidth)
    config.fabricTechMap.beamWidth = *patch.beamWidth;
  if (patch.saSteps)
    config.fabricTechMap.saSteps = *patch.saSteps;
  if (patch.saSeed)
    config.fabricTechMap.saSeed = *patch.saSeed;
  if (patch.threads)
    config.fabricTechMap.threads = *patch.threads;
  if (patch.rankingPolicy)
    config.dse.rankingPolicy = *patch.rankingPolicy;
  if (patch.objectives)
    config.dse.objectives = *patch.objectives;
}

llvm::Error parseGlobal(ConfigPatch &patch, llvm::yaml::MappingNode &map) {
  llvm::StringSet<> seen;
  for (auto &kv : map) {
    llvm::SmallString<64> keyStorage;
    StringRef key = scalarValue(kv.getKey(), keyStorage);
    if (llvm::Error err = checkDuplicateKey(seen, "global", key))
      return err;
    std::string canonicalKey = ("global." + key).str();
    if (key == "addr_bits") {
      auto valueOrErr = requireUnsigned(kv.getValue(), canonicalKey);
      if (!valueOrErr)
        return valueOrErr.takeError();
      patch.addrBits = *valueOrErr;
    } else if (key == "index_width") {
      auto valueOrErr = requireUnsigned(kv.getValue(), canonicalKey);
      if (!valueOrErr)
        return valueOrErr.takeError();
      patch.indexWidth = *valueOrErr;
    } else if (key == "mem_bus_width") {
      auto valueOrErr = requireUnsigned(kv.getValue(), canonicalKey);
      if (!valueOrErr)
        return valueOrErr.takeError();
      patch.memBusWidth = *valueOrErr;
    } else {
      return diagnostic("config_unknown_key", canonicalKey);
    }
    if (llvm::Error err = touch(patch, canonicalKey))
      return err;
  }
  return llvm::Error::success();
}

llvm::Error parseFabricTechMap(ConfigPatch &patch,
                               llvm::yaml::MappingNode &map) {
  llvm::StringSet<> seen;
  for (auto &kv : map) {
    llvm::SmallString<64> keyStorage;
    StringRef key = scalarValue(kv.getKey(), keyStorage);
    if (llvm::Error err = checkDuplicateKey(seen, "fabric_techmap", key))
      return err;
    std::string canonicalKey = ("fabric_techmap." + key).str();
    if (key == "alpha") {
      auto valueOrErr = requireDouble(kv.getValue(), canonicalKey);
      if (!valueOrErr)
        return valueOrErr.takeError();
      if (*valueOrErr < 0.0)
        return diagnostic("config_range_violation", canonicalKey,
                          "weight must be non-negative");
      patch.alpha = *valueOrErr;
    } else if (key == "beta") {
      auto valueOrErr = requireDouble(kv.getValue(), canonicalKey);
      if (!valueOrErr)
        return valueOrErr.takeError();
      if (*valueOrErr < 0.0)
        return diagnostic("config_range_violation", canonicalKey,
                          "weight must be non-negative");
      patch.beta = *valueOrErr;
    } else if (key == "gamma") {
      auto valueOrErr = requireDouble(kv.getValue(), canonicalKey);
      if (!valueOrErr)
        return valueOrErr.takeError();
      if (*valueOrErr < 0.0)
        return diagnostic("config_range_violation", canonicalKey,
                          "weight must be non-negative");
      patch.gamma = *valueOrErr;
    } else if (key == "algorithm") {
      auto valueOrErr = requireScalarString(kv.getValue(), canonicalKey);
      if (!valueOrErr)
        return valueOrErr.takeError();
      auto algorithmOrErr =
          parseFabricTechMapAlgorithm(*valueOrErr, canonicalKey);
      if (!algorithmOrErr)
        return algorithmOrErr.takeError();
      patch.algorithm = *algorithmOrErr;
    } else if (key == "beam_width") {
      auto valueOrErr = requireUnsigned(kv.getValue(), canonicalKey);
      if (!valueOrErr)
        return valueOrErr.takeError();
      patch.beamWidth = *valueOrErr;
    } else if (key == "sa_steps") {
      auto valueOrErr =
          requireUnsigned(kv.getValue(), canonicalKey, /*allowZero=*/true);
      if (!valueOrErr)
        return valueOrErr.takeError();
      patch.saSteps = *valueOrErr;
    } else if (key == "sa_seed") {
      auto valueOrErr = requireUInt64(kv.getValue(), canonicalKey);
      if (!valueOrErr)
        return valueOrErr.takeError();
      patch.saSeed = *valueOrErr;
    } else if (key == "threads") {
      auto valueOrErr =
          requireUnsigned(kv.getValue(), canonicalKey, /*allowZero=*/true);
      if (!valueOrErr)
        return valueOrErr.takeError();
      patch.threads = *valueOrErr;
    } else {
      return diagnostic("config_unknown_key", canonicalKey);
    }
    if (llvm::Error err = touch(patch, canonicalKey))
      return err;
  }
  return llvm::Error::success();
}

llvm::Error
parseDseObjectives(ConfigPatch &patch, llvm::yaml::Node *node,
                   llvm::StringRef canonicalKey = "dse.objectives") {
  auto *sequence = llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(node);
  if (!sequence)
    return diagnostic("config_type_mismatch", canonicalKey,
                      "expected objective array");

  std::vector<loom::ResolvedDseObjective> objectives;
  for (auto &entryNode : *sequence) {
    auto *entry = llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(&entryNode);
    if (!entry)
      return diagnostic("config_type_mismatch", canonicalKey,
                        "objective entry must be a mapping");

    std::optional<std::string> objectiveId;
    std::optional<double> weight;
    llvm::StringSet<> seen;
    for (auto &kv : *entry) {
      llvm::SmallString<64> keyStorage;
      StringRef key = scalarValue(kv.getKey(), keyStorage);
      if (llvm::Error err = checkDuplicateKey(seen, canonicalKey, key))
        return err;
      std::string fieldKey = (canonicalKey + "." + key).str();
      if (key == "objective_id") {
        auto valueOrErr = requireScalarString(kv.getValue(), fieldKey);
        if (!valueOrErr)
          return valueOrErr.takeError();
        if (!isKnownObjective(*valueOrErr))
          return diagnostic("config_unknown_objective", fieldKey, *valueOrErr);
        objectiveId = *valueOrErr;
      } else if (key == "weight") {
        auto valueOrErr = requireDouble(kv.getValue(), fieldKey);
        if (!valueOrErr)
          return valueOrErr.takeError();
        if (*valueOrErr < 0.0)
          return diagnostic("config_range_violation", fieldKey,
                            "weight must be non-negative");
        weight = *valueOrErr;
      } else {
        return diagnostic("config_unknown_key", fieldKey);
      }
    }
    if (!objectiveId)
      return diagnostic("config_missing_required_profile", canonicalKey,
                        "objective_id");
    objectives.push_back(
        loom::ResolvedDseObjective{*objectiveId, weight.value_or(1.0)});
  }
  patch.objectives = std::move(objectives);
  return touch(patch, canonicalKey);
}

llvm::Error parseDse(ConfigPatch &patch, llvm::yaml::MappingNode &map) {
  llvm::StringSet<> seen;
  for (auto &kv : map) {
    llvm::SmallString<64> keyStorage;
    StringRef key = scalarValue(kv.getKey(), keyStorage);
    if (llvm::Error err = checkDuplicateKey(seen, "dse", key))
      return err;
    std::string canonicalKey = ("dse." + key).str();
    if (key == "ranking_policy") {
      auto valueOrErr = requireScalarString(kv.getValue(), canonicalKey);
      if (!valueOrErr)
        return valueOrErr.takeError();
      if (!isKnownRankingPolicy(*valueOrErr))
        return diagnostic("config_unknown_policy", canonicalKey, *valueOrErr);
      patch.rankingPolicy = *valueOrErr;
      if (llvm::Error err = touch(patch, canonicalKey))
        return err;
    } else if (key == "objectives") {
      if (llvm::Error err = parseDseObjectives(patch, kv.getValue()))
        return err;
    } else {
      return diagnostic("config_unknown_key", canonicalKey);
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::vector<std::string>> parseIncludes(llvm::yaml::Node *node,
                                                       llvm::StringRef key) {
  std::vector<std::string> includes;
  if (auto scalar = requireScalarString(node, key)) {
    includes.push_back(*scalar);
    return includes;
  } else {
    llvm::consumeError(scalar.takeError());
  }
  auto *sequence = llvm::dyn_cast_or_null<llvm::yaml::SequenceNode>(node);
  if (!sequence)
    return diagnostic("config_type_mismatch", key,
                      "include must be a scalar or array of scalars");
  for (auto &entry : *sequence) {
    auto valueOrErr = requireScalarString(&entry, key);
    if (!valueOrErr)
      return valueOrErr.takeError();
    includes.push_back(*valueOrErr);
  }
  return includes;
}

llvm::Expected<ConfigPatch>
parseConfigPatchFromMapping(llvm::yaml::MappingNode &topMap,
                            llvm::StringRef sourceName, llvm::StringRef baseDir,
                            std::set<std::string> &activeFiles);

llvm::Expected<ConfigPatch>
parseConfigFilePatch(llvm::StringRef path, std::set<std::string> &activeFiles) {
  if (activeFiles.count(path.str()) != 0)
    return diagnostic("config_parse_failed", path, "cyclic include");
  activeFiles.insert(path.str());

  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (std::error_code ec = bufferOrErr.getError()) {
    activeFiles.erase(path.str());
    return makeErr("config_parse_failed: " + path + ": " + ec.message());
  }
  llvm::SourceMgr sourceMgr;
  llvm::yaml::Stream stream((*bufferOrErr)->getBuffer(), sourceMgr);
  auto it = stream.begin();
  if (it == stream.end()) {
    activeFiles.erase(path.str());
    return ConfigPatch();
  }
  llvm::yaml::Node *root = it->getRoot();
  if (!root) {
    activeFiles.erase(path.str());
    return ConfigPatch();
  }
  auto *topMap = llvm::dyn_cast<llvm::yaml::MappingNode>(root);
  if (!topMap) {
    activeFiles.erase(path.str());
    return diagnostic("config_type_mismatch", path, "top-level mapping");
  }
  llvm::SmallString<256> base(path);
  llvm::sys::path::remove_filename(base);
  auto patchOrErr =
      parseConfigPatchFromMapping(*topMap, path, base.str(), activeFiles);
  activeFiles.erase(path.str());
  if (!patchOrErr)
    return patchOrErr.takeError();
  ++it;
  if (it != stream.end())
    return diagnostic("config_parse_failed", path,
                      "multiple YAML documents are not supported");
  if (stream.failed())
    return diagnostic("config_parse_failed", path);
  return *patchOrErr;
}

llvm::Expected<ConfigPatch>
parseConfigPatchFromMapping(llvm::yaml::MappingNode &topMap,
                            llvm::StringRef sourceName, llvm::StringRef baseDir,
                            std::set<std::string> &activeFiles) {
  ConfigPatch included;
  ConfigPatch local;
  llvm::StringSet<> seen;
  for (auto &kv : topMap) {
    llvm::SmallString<64> keyStorage;
    StringRef key = scalarValue(kv.getKey(), keyStorage);
    if (llvm::Error err = checkDuplicateKey(seen, "", key))
      return err;
    if (key == "include") {
      auto includesOrErr = parseIncludes(kv.getValue(), "include");
      if (!includesOrErr)
        return includesOrErr.takeError();
      for (const std::string &include : *includesOrErr) {
        llvm::SmallString<256> includePath(include);
        if (!llvm::sys::path::is_absolute(includePath)) {
          includePath = baseDir;
          llvm::sys::path::append(includePath, include);
        }
        auto includePatchOrErr =
            parseConfigFilePatch(includePath.str(), activeFiles);
        if (!includePatchOrErr)
          return includePatchOrErr.takeError();
        if (llvm::Error err = mergeSiblingPatch(included, *includePatchOrErr))
          return err;
      }
      continue;
    }
    if (key == "config_id") {
      auto valueOrErr = requireScalarString(kv.getValue(), "config_id");
      if (!valueOrErr)
        return valueOrErr.takeError();
      local.configId = *valueOrErr;
      if (llvm::Error err = touch(local, "config_id"))
        return err;
    } else if (key == "global") {
      auto *map =
          llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(kv.getValue());
      if (!map)
        return diagnostic("config_type_mismatch", "global", "expected mapping");
      if (llvm::Error err = parseGlobal(local, *map))
        return err;
    } else if (key == "fabric_techmap") {
      auto *map =
          llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(kv.getValue());
      if (!map)
        return diagnostic("config_type_mismatch", "fabric_techmap",
                          "expected mapping");
      if (llvm::Error err = parseFabricTechMap(local, *map))
        return err;
    } else if (key == "dse") {
      auto *map =
          llvm::dyn_cast_or_null<llvm::yaml::MappingNode>(kv.getValue());
      if (!map)
        return diagnostic("config_type_mismatch", "dse", "expected mapping");
      if (llvm::Error err = parseDse(local, *map))
        return err;
    } else {
      return diagnostic("config_unknown_key", key);
    }
  }

  ConfigPatch merged = included;
  for (const std::string &key : local.touchedKeys)
    merged.touchedKeys.insert(key);
  if (local.configId)
    merged.configId = local.configId;
  if (local.addrBits)
    merged.addrBits = local.addrBits;
  if (local.indexWidth)
    merged.indexWidth = local.indexWidth;
  if (local.memBusWidth)
    merged.memBusWidth = local.memBusWidth;
  if (local.alpha)
    merged.alpha = local.alpha;
  if (local.beta)
    merged.beta = local.beta;
  if (local.gamma)
    merged.gamma = local.gamma;
  if (local.algorithm)
    merged.algorithm = local.algorithm;
  if (local.beamWidth)
    merged.beamWidth = local.beamWidth;
  if (local.saSteps)
    merged.saSteps = local.saSteps;
  if (local.saSeed)
    merged.saSeed = local.saSeed;
  if (local.threads)
    merged.threads = local.threads;
  if (local.rankingPolicy)
    merged.rankingPolicy = local.rankingPolicy;
  if (local.objectives)
    merged.objectives = local.objectives;
  (void)sourceName;
  return merged;
}

std::string hashString(llvm::StringRef text) {
  llvm::SHA256 hasher;
  hasher.update(text);
  auto digest = hasher.final();
  return llvm::toHex(llvm::ArrayRef<std::uint8_t>(digest.data(), digest.size()),
                     true);
}

llvm::json::Array objectivesJson(const loom::ResolvedConfig &config) {
  llvm::json::Array objectives;
  for (const loom::ResolvedDseObjective &objective : config.dse.objectives) {
    objectives.push_back(llvm::json::Object{
        {"objective_id", objective.objectiveId},
        {"weight", objective.weight},
    });
  }
  return objectives;
}

llvm::json::Object
resolvedConfigJsonObject(const loom::ResolvedConfig &config) {
  return llvm::json::Object{
      {"config_id", config.configId},
      {"global",
       llvm::json::Object{
           {"addr_bits", static_cast<int64_t>(config.global.addrBits)},
           {"index_width", static_cast<int64_t>(config.global.indexWidth)},
           {"mem_bus_width", static_cast<int64_t>(config.global.memBusWidth)},
       }},
      {"fabric_techmap",
       llvm::json::Object{
           {"algorithm",
            fabricTechMapAlgorithmName(config.fabricTechMap.algorithm)},
           {"alpha", config.fabricTechMap.alpha},
           {"beta", config.fabricTechMap.beta},
           {"gamma", config.fabricTechMap.gamma},
           {"beam_width", static_cast<int64_t>(config.fabricTechMap.beamWidth)},
           {"sa_steps", static_cast<int64_t>(config.fabricTechMap.saSteps)},
           {"sa_seed", static_cast<int64_t>(config.fabricTechMap.saSeed)},
           {"threads", static_cast<int64_t>(config.fabricTechMap.threads)},
       }},
      {"dse",
       llvm::json::Object{
           {"ranking_policy", config.dse.rankingPolicy},
           {"objectives", objectivesJson(config)},
       }},
  };
}

} // namespace

loom::ResolvedConfig loom::defaultResolvedConfig() {
  ResolvedConfig config;
  config.dse.objectives.push_back(
      ResolvedDseObjective{"minimize_runtime", 1.0});
  return config;
}

llvm::Expected<loom::ResolvedConfig>
loom::parseResolvedConfig(llvm::StringRef body, llvm::StringRef sourceName) {
  llvm::SourceMgr sourceMgr;
  llvm::yaml::Stream stream(body, sourceMgr);
  auto it = stream.begin();
  if (it == stream.end())
    return defaultResolvedConfig();
  llvm::yaml::Node *root = it->getRoot();
  if (!root)
    return defaultResolvedConfig();
  auto *topMap = llvm::dyn_cast<llvm::yaml::MappingNode>(root);
  if (!topMap)
    return diagnostic("config_type_mismatch", sourceName, "top-level mapping");

  std::set<std::string> activeFiles;
  ConfigPatch patch;
  auto patchOrErr =
      parseConfigPatchFromMapping(*topMap, sourceName, "", activeFiles);
  if (!patchOrErr)
    return patchOrErr.takeError();
  patch = *patchOrErr;
  ++it;
  if (it != stream.end())
    return diagnostic("config_parse_failed", sourceName,
                      "multiple YAML documents are not supported");
  if (stream.failed())
    return diagnostic("config_parse_failed", sourceName);

  ResolvedConfig config = defaultResolvedConfig();
  applyPatch(config, patch);
  return config;
}

llvm::Expected<loom::ResolvedConfig>
loom::loadResolvedConfig(llvm::StringRef path) {
  std::set<std::string> activeFiles;
  auto patchOrErr = parseConfigFilePatch(path, activeFiles);
  if (!patchOrErr)
    return patchOrErr.takeError();

  ResolvedConfig config = defaultResolvedConfig();
  applyPatch(config, *patchOrErr);
  return config;
}

std::string
loom::canonicalResolvedConfigJson(const loom::ResolvedConfig &config) {
  return llvm::formatv("{0:2}",
                       llvm::json::Value(resolvedConfigJsonObject(config)))
      .str();
}

std::string
loom::resolvedConfigFingerprint(const loom::ResolvedConfig &config) {
  return hashString(canonicalResolvedConfigJson(config));
}

std::string
loom::canonicalComponentConfigViewJson(const loom::ResolvedConfig &config,
                                       llvm::StringRef viewId) {
  llvm::json::Object root{
      {"config_id", config.configId},
      {"view_id", viewId.str()},
      {"resolved_config_fingerprint", resolvedConfigFingerprint(config)},
  };
  return llvm::formatv("{0:2}", llvm::json::Value(std::move(root))).str();
}

std::string loom::componentConfigFingerprint(const loom::ResolvedConfig &config,
                                             llvm::StringRef viewId) {
  return hashString(canonicalComponentConfigViewJson(config, viewId));
}

bool loom::isResolvedConfigFingerprint(llvm::StringRef fingerprint) {
  if (fingerprint.size() != 64)
    return false;
  return std::all_of(fingerprint.begin(), fingerprint.end(), [](char c) {
    return std::isdigit(static_cast<unsigned char>(c)) ||
           (c >= 'a' && c <= 'f');
  });
}

#include "Common/Config.h"

#include "Common/ResolvedConfig.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdlib>
#include <string>

using ::llvm::StringRef;

namespace {

// ---------- Shared helpers -------------------------------------------------

::llvm::Error makeErr(const ::llvm::Twine &msg) {
  return ::llvm::createStringError(::llvm::inconvertibleErrorCode(),
                                   msg.str().c_str());
}

bool stringIsKnownAlgorithm(StringRef s) {
  return s == "greedy" || s == "list" || s == "beam" || s == "sa" || s == "ilp";
}

::llvm::Error validate(const ::loom::TechMapConfig &cfg) {
  if (!stringIsKnownAlgorithm(cfg.algorithm))
    return makeErr(
        "techmap.algorithm must be one of greedy|list|beam|sa|ilp, got '" +
        cfg.algorithm + "'");
  if (cfg.beamWidth == 0)
    return makeErr("techmap.beam_width must be >= 1");
  if (!(cfg.alpha >= 0.0 && cfg.beta >= 0.0 && cfg.gamma >= 0.0))
    return makeErr("techmap.{alpha,beta,gamma} must all be >= 0");
  return ::llvm::Error::success();
}

// ---------- TOML mini-parser ----------------------------------------------
//
// Only the subset we need:
//   [techmap]
//   key = value
// where value is a string ("..." or '...'), unsigned int, double, or bool.
// Comments start with '#'. Whitespace is trimmed. Keys outside [techmap]
// are silently ignored (forward-compat with future sections).

StringRef trim(StringRef s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
    s = s.drop_front();
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    s = s.drop_back();
  return s;
}

StringRef stripComment(StringRef s) {
  // Strip trailing `# ...` comments (we don't support `#` inside strings; the
  // strings we care about never contain `#`, so this is fine).
  size_t hash = s.find('#');
  if (hash != StringRef::npos)
    s = s.substr(0, hash);
  return trim(s);
}

::llvm::Expected<double> parseTomlDouble(StringRef v) {
  double out = 0.0;
  if (v.getAsDouble(out))
    return makeErr("expected number, got '" + v + "'");
  return out;
}

::llvm::Expected<uint64_t> parseTomlUInt(StringRef v) {
  uint64_t out = 0;
  if (v.getAsInteger(10, out))
    return makeErr("expected unsigned integer, got '" + v + "'");
  return out;
}

::llvm::Expected<std::string> parseTomlString(StringRef v) {
  if (v.size() < 2)
    return makeErr("expected quoted string, got '" + v + "'");
  char q = v.front();
  if ((q != '"' && q != '\'') || v.back() != q)
    return makeErr("expected quoted string, got '" + v + "'");
  return v.drop_front().drop_back().str();
}

::llvm::Error applyTomlKV(::loom::TechMapConfig &cfg, StringRef key,
                          StringRef value) {
  value = trim(value);
  if (key == "alpha") {
    auto v = parseTomlDouble(value);
    if (!v)
      return v.takeError();
    cfg.alpha = *v;
  } else if (key == "beta") {
    auto v = parseTomlDouble(value);
    if (!v)
      return v.takeError();
    cfg.beta = *v;
  } else if (key == "gamma") {
    auto v = parseTomlDouble(value);
    if (!v)
      return v.takeError();
    cfg.gamma = *v;
  } else if (key == "algorithm") {
    auto v = parseTomlString(value);
    if (!v)
      return v.takeError();
    cfg.algorithm = std::move(*v);
  } else if (key == "beam_width") {
    auto v = parseTomlUInt(value);
    if (!v)
      return v.takeError();
    cfg.beamWidth = static_cast<unsigned>(*v);
  } else if (key == "sa_steps") {
    auto v = parseTomlUInt(value);
    if (!v)
      return v.takeError();
    cfg.saSteps = static_cast<unsigned>(*v);
  } else if (key == "sa_seed") {
    auto v = parseTomlUInt(value);
    if (!v)
      return v.takeError();
    cfg.saSeed = *v;
  } else if (key == "threads") {
    auto v = parseTomlUInt(value);
    if (!v)
      return v.takeError();
    cfg.threads = static_cast<unsigned>(*v);
  } else {
    // Unknown key: ignore for forward compatibility. Bool value still
    // validated for syntax.
    if (value == "true" || value == "false")
      return ::llvm::Error::success();
  }
  return ::llvm::Error::success();
}

} // namespace

namespace loom {

TechMapConfig::TechMapConfig() {
  ResolvedConfig resolved = defaultResolvedConfig();
  alpha = resolved.fabricTechMap.alpha;
  beta = resolved.fabricTechMap.beta;
  gamma = resolved.fabricTechMap.gamma;
  algorithm = resolved.fabricTechMap.algorithm;
  beamWidth = resolved.fabricTechMap.beamWidth;
  saSteps = resolved.fabricTechMap.saSteps;
  saSeed = resolved.fabricTechMap.saSeed;
  threads = resolved.fabricTechMap.threads;
}

::llvm::Expected<TechMapConfig> parseTechMapConfigTOML(StringRef body) {
  TechMapConfig cfg;
  StringRef section;
  size_t lineNo = 0;
  while (!body.empty()) {
    ++lineNo;
    auto split = body.split('\n');
    StringRef line = stripComment(split.first);
    body = split.second;
    if (line.empty())
      continue;

    if (line.front() == '[') {
      if (line.back() != ']')
        return makeErr("toml line " + ::llvm::Twine(lineNo) +
                       ": malformed section header");
      section = trim(line.drop_front().drop_back());
      continue;
    }

    auto eq = line.find('=');
    if (eq == StringRef::npos)
      return makeErr("toml line " + ::llvm::Twine(lineNo) +
                     ": expected `key = value`");
    StringRef key = trim(line.substr(0, eq));
    StringRef value = trim(line.substr(eq + 1));
    if (section != "techmap")
      continue;
    if (auto e = applyTomlKV(cfg, key, value))
      return std::move(e);
  }
  if (auto e = validate(cfg))
    return std::move(e);
  return cfg;
}

// ---------- YAML loader ----------------------------------------------------

::llvm::Expected<TechMapConfig> parseTechMapConfigYAML(StringRef body) {
  // We hand-walk a tiny YAML subset rather than registering MappingTraits, so
  // that unknown keys can be tolerated with a precise location.
  TechMapConfig cfg;
  ::llvm::SourceMgr sm;
  ::llvm::yaml::Stream stream(body, sm);

  auto it = stream.begin();
  if (it == stream.end())
    return cfg;
  ::llvm::yaml::Node *root = it->getRoot();
  if (!root)
    return cfg;
  auto *topMap = ::llvm::dyn_cast<::llvm::yaml::MappingNode>(root);
  if (!topMap)
    return makeErr("yaml: top-level must be a mapping");

  ::llvm::yaml::MappingNode *techmap = nullptr;
  for (auto &kv : *topMap) {
    auto *keyNode = ::llvm::dyn_cast<::llvm::yaml::ScalarNode>(kv.getKey());
    if (!keyNode)
      continue;
    ::llvm::SmallString<16> kbuf;
    StringRef keyName = keyNode->getValue(kbuf);
    if (keyName == "techmap") {
      techmap = ::llvm::dyn_cast<::llvm::yaml::MappingNode>(kv.getValue());
      break;
    }
  }
  if (!techmap) {
    if (auto e = validate(cfg))
      return std::move(e);
    return cfg;
  }

  for (auto &kv : *techmap) {
    auto *keyNode = ::llvm::dyn_cast<::llvm::yaml::ScalarNode>(kv.getKey());
    auto *valNode = ::llvm::dyn_cast<::llvm::yaml::ScalarNode>(kv.getValue());
    if (!keyNode || !valNode)
      continue;
    ::llvm::SmallString<16> kbuf, vbuf;
    StringRef key = keyNode->getValue(kbuf);
    StringRef val = valNode->getValue(vbuf);
    val = trim(val);
    if (val.size() >= 2 && (val.front() == '"' || val.front() == '\'') &&
        val.front() == val.back())
      val = val.drop_front().drop_back();

    if (key == "alpha") {
      double v;
      if (val.getAsDouble(v))
        return makeErr("yaml: alpha not a number: '" + val + "'");
      cfg.alpha = v;
    } else if (key == "beta") {
      double v;
      if (val.getAsDouble(v))
        return makeErr("yaml: beta not a number: '" + val + "'");
      cfg.beta = v;
    } else if (key == "gamma") {
      double v;
      if (val.getAsDouble(v))
        return makeErr("yaml: gamma not a number: '" + val + "'");
      cfg.gamma = v;
    } else if (key == "algorithm") {
      cfg.algorithm = val.str();
    } else if (key == "beam_width") {
      uint64_t v;
      if (val.getAsInteger(10, v))
        return makeErr("yaml: beam_width not an integer: '" + val + "'");
      cfg.beamWidth = static_cast<unsigned>(v);
    } else if (key == "sa_steps") {
      uint64_t v;
      if (val.getAsInteger(10, v))
        return makeErr("yaml: sa_steps not an integer: '" + val + "'");
      cfg.saSteps = static_cast<unsigned>(v);
    } else if (key == "sa_seed") {
      uint64_t v;
      if (val.getAsInteger(10, v))
        return makeErr("yaml: sa_seed not an integer: '" + val + "'");
      cfg.saSeed = v;
    } else if (key == "threads") {
      uint64_t v;
      if (val.getAsInteger(10, v))
        return makeErr("yaml: threads not an integer: '" + val + "'");
      cfg.threads = static_cast<unsigned>(v);
    }
    // Unknown keys silently ignored.
  }

  if (auto e = validate(cfg))
    return std::move(e);
  return cfg;
}

::llvm::Expected<TechMapConfig> loadTechMapConfig(StringRef path) {
  auto bufOrErr = ::llvm::MemoryBuffer::getFile(path);
  if (auto ec = bufOrErr.getError())
    return makeErr("cannot open config '" + path + "': " + ec.message());
  StringRef body = bufOrErr.get()->getBuffer();

  StringRef ext = ::llvm::sys::path::extension(path);
  if (ext == ".yaml" || ext == ".yml")
    return parseTechMapConfigYAML(body);
  if (ext == ".toml")
    return parseTechMapConfigTOML(body);
  return makeErr("unrecognized config extension '" + ext +
                 "' (expected .yaml/.yml/.toml)");
}

} // namespace loom

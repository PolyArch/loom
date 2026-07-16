#include "Common/SynthConfig.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdlib>
#include <set>
#include <string>

using ::llvm::StringRef;

namespace {

// ---------- Shared helpers -------------------------------------------------

::llvm::Error makeErr(const ::llvm::Twine &msg) {
  return ::llvm::createStringError(::llvm::inconvertibleErrorCode(),
                                   msg.str().c_str());
}

StringRef trim(StringRef s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
    s = s.drop_front();
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    s = s.drop_back();
  return s;
}

StringRef stripComment(StringRef s) {
  // Strip trailing `# ...` comments. The values we care about never contain a
  // literal '#', so naive splitting on the first '#' is fine.
  size_t hash = s.find('#');
  if (hash != StringRef::npos)
    s = s.substr(0, hash);
  return trim(s);
}

StringRef stripQuotes(StringRef v) {
  v = trim(v);
  if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') &&
      v.front() == v.back())
    v = v.drop_front().drop_back();
  return v;
}

bool stringIsKnownStrategy(StringRef s) { return s == "anchor"; }

::llvm::Expected<bool> parseBool(StringRef v, StringRef ctx) {
  v = stripQuotes(v);
  if (v == "true" || v == "True" || v == "TRUE")
    return true;
  if (v == "false" || v == "False" || v == "FALSE")
    return false;
  return makeErr(ctx + ": expected bool, got '" + v + "'");
}

::llvm::Expected<double> parseDouble(StringRef v, StringRef ctx) {
  v = stripQuotes(v);
  double out = 0.0;
  if (v.getAsDouble(out))
    return makeErr(ctx + ": expected number, got '" + v + "'");
  return out;
}

// Parse an unsigned integer; the literal `auto` is mapped to 0 (for the
// `workers` field). The `acceptAuto` flag controls whether `auto` is a
// valid spelling or an error.
::llvm::Expected<uint64_t> parseUInt(StringRef v, StringRef ctx,
                                     bool acceptAuto) {
  v = stripQuotes(v);
  if (acceptAuto && v == "auto")
    return uint64_t{0};
  uint64_t out = 0;
  if (v.getAsInteger(10, out))
    return makeErr(ctx + ": expected unsigned integer" +
                   (acceptAuto ? StringRef(" or 'auto'") : StringRef("")) +
                   ", got '" + v + "'");
  return out;
}

::llvm::Error validate(const ::loom::SynthConfig &cfg) {
  if (!stringIsKnownStrategy(cfg.strategy))
    return makeErr("synth.strategy must be 'anchor', got '" + cfg.strategy +
                   "'");
  if (!(cfg.costMuxPenalty >= 0.0 && cfg.costDemuxPenalty >= 0.0 &&
        cfg.costCarryPenalty >= 0.0))
    return makeErr("synth.cost.{mux,demux,carry}_penalty must all be >= 0");
  return ::llvm::Error::success();
}

// ---------- YAML helpers ---------------------------------------------------

// Get the string spelling of a scalar node, with surrounding quotes preserved
// as YAMLParser returns them. The caller is expected to invoke `stripQuotes`
// on the result if it cares.
template <unsigned N>
StringRef scalarValue(::llvm::yaml::Node *n, ::llvm::SmallString<N> &buf) {
  auto *s = ::llvm::dyn_cast_or_null<::llvm::yaml::ScalarNode>(n);
  if (!s)
    return {};
  return s->getValue(buf);
}

::llvm::Error makeYamlSchemaErr(::llvm::SourceMgr &sm, ::llvm::yaml::Node *node,
                                const ::llvm::Twine &message) {
  if (node) {
    ::llvm::SMLoc loc = node->getSourceRange().Start;
    if (loc.isValid()) {
      auto [line, column] = sm.getLineAndColumn(loc);
      return makeErr("yaml line " + ::llvm::Twine(line) + " column " +
                     ::llvm::Twine(column) + ": " + message);
    }
  }
  return makeErr("yaml: " + message);
}

// Apply a single `key: value` at the top of the synth map. Handles both
// scalar leaves (e.g. `strategy: ...`) and nested maps (e.g. `parallelism:`).
::llvm::Error applySynthYAMLEntry(::loom::SynthConfig &cfg, StringRef key,
                                  ::llvm::yaml::Node *keyNode,
                                  ::llvm::yaml::Node *value,
                                  ::llvm::SourceMgr &sm);

// Apply a single nested mapping (e.g. the `parallelism:` block) by dispatching
// on `parent.child` keys.
::llvm::Error applySynthYAMLNested(::loom::SynthConfig &cfg, StringRef parent,
                                   ::llvm::yaml::MappingNode *map,
                                   ::llvm::SourceMgr &sm) {
  ::llvm::StringSet<> seenKeys;
  for (auto &kv : *map) {
    ::llvm::SmallString<32> kbuf;
    StringRef child = scalarValue(kv.getKey(), kbuf);
    if (child.empty())
      return makeYamlSchemaErr(sm, kv.getKey(),
                               "mapping keys must be scalar strings");
    if (!seenKeys.insert(child).second)
      return makeYamlSchemaErr(sm, kv.getKey(),
                               ::llvm::Twine("duplicate key 'synth.") + parent +
                                   "." + child + "'");
    auto *valNode = kv.getValue();

    if (parent == "parallelism") {
      ::llvm::SmallString<32> vbuf;
      StringRef val = scalarValue(valNode, vbuf);
      if (child == "cross_group") {
        auto v = parseBool(val, "synth.parallelism.cross_group");
        if (!v)
          return v.takeError();
        cfg.parallelismCrossGroup = *v;
      } else if (child == "workers") {
        auto v =
            parseUInt(val, "synth.parallelism.workers", /*acceptAuto=*/true);
        if (!v)
          return v.takeError();
        cfg.parallelismWorkers = static_cast<unsigned>(*v);
      } else {
        return makeYamlSchemaErr(sm, kv.getKey(),
                                 ::llvm::Twine("unknown key 'synth.") + parent +
                                     "." + child + "'");
      }
      continue;
    }

    if (parent == "coverage_verifier") {
      ::llvm::SmallString<32> vbuf;
      StringRef val = scalarValue(valNode, vbuf);
      if (child == "parallel_match") {
        auto v = parseBool(val, "synth.coverage_verifier.parallel_match");
        if (!v)
          return v.takeError();
        cfg.coverageVerifierParallelMatch = *v;
      } else {
        return makeYamlSchemaErr(sm, kv.getKey(),
                                 ::llvm::Twine("unknown key 'synth.") + parent +
                                     "." + child + "'");
      }
      continue;
    }

    if (parent == "cost") {
      ::llvm::SmallString<32> vbuf;
      StringRef val = scalarValue(valNode, vbuf);
      if (child == "mux_penalty") {
        auto v = parseDouble(val, "synth.cost.mux_penalty");
        if (!v)
          return v.takeError();
        cfg.costMuxPenalty = *v;
      } else if (child == "demux_penalty") {
        auto v = parseDouble(val, "synth.cost.demux_penalty");
        if (!v)
          return v.takeError();
        cfg.costDemuxPenalty = *v;
      } else if (child == "carry_penalty") {
        auto v = parseDouble(val, "synth.cost.carry_penalty");
        if (!v)
          return v.takeError();
        cfg.costCarryPenalty = *v;
      } else {
        return makeYamlSchemaErr(sm, kv.getKey(),
                                 ::llvm::Twine("unknown key 'synth.") + parent +
                                     "." + child + "'");
      }
      continue;
    }

    if (parent == "anchor") {
      ::llvm::SmallString<32> vbuf;
      StringRef val = scalarValue(valNode, vbuf);
      if (child == "allow_intra_position_mux") {
        auto v = parseBool(val, "synth.anchor.allow_intra_position_mux");
        if (!v)
          return v.takeError();
        cfg.anchorAllowIntraPositionMux = *v;
      } else {
        return makeYamlSchemaErr(sm, kv.getKey(),
                                 ::llvm::Twine("unknown key 'synth.") + parent +
                                     "." + child + "'");
      }
      continue;
    }
  }
  return ::llvm::Error::success();
}

bool isKnownYamlSection(StringRef key) {
  return key == "parallelism" || key == "coverage_verifier" || key == "cost" ||
         key == "anchor";
}

::llvm::Error applySynthYAMLEntry(::loom::SynthConfig &cfg, StringRef key,
                                  ::llvm::yaml::Node *keyNode,
                                  ::llvm::yaml::Node *value,
                                  ::llvm::SourceMgr &sm) {
  // Scalar leaves at the top of the synth map.
  if (key == "strategy") {
    ::llvm::SmallString<32> vbuf;
    StringRef val = scalarValue(value, vbuf);
    cfg.strategy = stripQuotes(val).str();
    return ::llvm::Error::success();
  }
  if (isKnownYamlSection(key)) {
    auto *map = ::llvm::dyn_cast_or_null<::llvm::yaml::MappingNode>(value);
    if (!map)
      return makeYamlSchemaErr(sm, keyNode,
                               ::llvm::Twine("section 'synth.") + key +
                                   "' must be a mapping");
    return applySynthYAMLNested(cfg, key, map, sm);
  }

  if (::llvm::isa_and_nonnull<::llvm::yaml::MappingNode>(value))
    return makeYamlSchemaErr(
        sm, keyNode, ::llvm::Twine("unknown section 'synth.") + key + "'");
  return makeYamlSchemaErr(sm, keyNode,
                           ::llvm::Twine("unknown key 'synth.") + key + "'");
}

// ---------- TOML helpers ---------------------------------------------------
//
// Subset of TOML the loader recognizes:
//   [synth]
//   strategy = "anchor"
//
//   [synth.parallelism]
//   cross_group = true
//   workers = "auto"   # or 0
//
//   [synth.cost]
//   mux_penalty = 1.5
//   ...
//
::llvm::Error applyTomlKV(::loom::SynthConfig &cfg, StringRef section,
                          StringRef key, StringRef value, size_t lineNo) {
  value = trim(value);

  auto setBool = [&](bool &target, StringRef ctx) -> ::llvm::Error {
    auto v = parseBool(value, ctx);
    if (!v)
      return v.takeError();
    target = *v;
    return ::llvm::Error::success();
  };
  auto setDouble = [&](double &target, StringRef ctx) -> ::llvm::Error {
    auto v = parseDouble(value, ctx);
    if (!v)
      return v.takeError();
    target = *v;
    return ::llvm::Error::success();
  };
  auto setUInt32 = [&](unsigned &target, StringRef ctx,
                       bool acceptAuto) -> ::llvm::Error {
    auto v = parseUInt(value, ctx, acceptAuto);
    if (!v)
      return v.takeError();
    target = static_cast<unsigned>(*v);
    return ::llvm::Error::success();
  };
  if (section == "synth") {
    if (key == "strategy")
      cfg.strategy = stripQuotes(value).str();
    else
      return makeErr("toml line " + ::llvm::Twine(lineNo) +
                     ": unknown key 'synth." + key + "'");
    return ::llvm::Error::success();
  }
  if (section == "synth.parallelism") {
    if (key == "cross_group")
      return setBool(cfg.parallelismCrossGroup,
                     "synth.parallelism.cross_group");
    if (key == "workers")
      return setUInt32(cfg.parallelismWorkers, "synth.parallelism.workers",
                       /*acceptAuto=*/true);
    return makeErr("toml line " + ::llvm::Twine(lineNo) +
                   ": unknown key 'synth.parallelism." + key + "'");
  }
  if (section == "synth.coverage_verifier") {
    if (key == "parallel_match")
      return setBool(cfg.coverageVerifierParallelMatch,
                     "synth.coverage_verifier.parallel_match");
    return makeErr("toml line " + ::llvm::Twine(lineNo) +
                   ": unknown key 'synth.coverage_verifier." + key + "'");
  }
  if (section == "synth.cost") {
    if (key == "mux_penalty")
      return setDouble(cfg.costMuxPenalty, "synth.cost.mux_penalty");
    if (key == "demux_penalty")
      return setDouble(cfg.costDemuxPenalty, "synth.cost.demux_penalty");
    if (key == "carry_penalty")
      return setDouble(cfg.costCarryPenalty, "synth.cost.carry_penalty");
    return makeErr("toml line " + ::llvm::Twine(lineNo) +
                   ": unknown key 'synth.cost." + key + "'");
  }
  if (section == "synth.anchor") {
    if (key == "allow_intra_position_mux")
      return setBool(cfg.anchorAllowIntraPositionMux,
                     "synth.anchor.allow_intra_position_mux");
    return makeErr("toml line " + ::llvm::Twine(lineNo) +
                   ": unknown key 'synth.anchor." + key + "'");
  }
  std::string path = section.empty() ? key.str() : (section + "." + key).str();
  return makeErr("toml line " + ::llvm::Twine(lineNo) + ": unknown key '" +
                 path + "'");
}

bool isKnownTomlSection(StringRef section) {
  return section == "synth" || section == "synth.parallelism" ||
         section == "synth.coverage_verifier" || section == "synth.cost" ||
         section == "synth.anchor";
}

} // namespace

namespace loom {

::llvm::Expected<SynthConfig> parseSynthConfigYAML(StringRef body) {
  // Hand-walk a tiny YAML subset rather than registering MappingTraits so
  // schema and duplicate diagnostics retain precise source locations.
  SynthConfig cfg;
  ::llvm::SourceMgr sm;

  // Capture YAMLParser diagnostics (line/col, message) into a local string
  // instead of letting them leak to stderr. Installed before the Stream is
  // constructed so the very first tokenizer error is captured.
  std::string capturedDiag;
  auto diagSink = [](const ::llvm::SMDiagnostic &diag, void *ctx) {
    auto *sink = static_cast<std::string *>(ctx);
    if (!sink->empty())
      sink->push_back('\n');
    ::llvm::raw_string_ostream os(*sink);
    diag.print(/*ProgName=*/nullptr, os, /*ShowColors=*/false,
               /*ShowKindLabel=*/false);
    os.flush();
    // Trim trailing whitespace so the captured diagnostic composes cleanly
    // into the error string.
    while (!sink->empty() &&
           std::isspace(static_cast<unsigned char>(sink->back())))
      sink->pop_back();
  };
  sm.setDiagHandler(diagSink, &capturedDiag);

  ::llvm::yaml::Stream stream(body, sm);

  auto buildParseErr = [&]() -> ::llvm::Error {
    if (!capturedDiag.empty())
      return makeErr("yaml: " + ::llvm::Twine(capturedDiag));
    return makeErr("yaml: parse error in body");
  };

  auto it = stream.begin();
  if (it == stream.end()) {
    if (stream.failed())
      return buildParseErr();
    return cfg;
  }
  ::llvm::yaml::Node *root = it->getRoot();
  // Check the parser state immediately after the first getRoot() call so a
  // malformed body cannot fall through with a non-null but garbage `root`.
  if (stream.failed())
    return buildParseErr();
  if (!root)
    return cfg;
  auto *topMap = ::llvm::dyn_cast<::llvm::yaml::MappingNode>(root);
  if (!topMap)
    return makeErr("yaml: top-level must be a mapping");

  auto topIt = topMap->begin();
  if (topIt == topMap->end())
    return makeErr("yaml: top-level mapping must contain a 'synth:' key");
  auto *synthKeyNode =
      ::llvm::dyn_cast<::llvm::yaml::ScalarNode>(topIt->getKey());
  if (!synthKeyNode)
    return makeYamlSchemaErr(sm, topIt->getKey(),
                             "top-level keys must be scalar strings");
  ::llvm::SmallString<16> synthKeyBuffer;
  StringRef synthKey = synthKeyNode->getValue(synthKeyBuffer);
  if (synthKey != "synth") {
    if (::llvm::isa_and_nonnull<::llvm::yaml::MappingNode>(topIt->getValue()))
      return makeYamlSchemaErr(sm, synthKeyNode,
                               ::llvm::Twine("unknown section '") + synthKey +
                                   "'");
    return makeYamlSchemaErr(sm, synthKeyNode,
                             ::llvm::Twine("unknown key '") + synthKey + "'");
  }
  auto *synth = ::llvm::dyn_cast<::llvm::yaml::MappingNode>(topIt->getValue());
  if (!synth)
    return makeYamlSchemaErr(sm, synthKeyNode,
                             "section 'synth' must be a mapping");

  ::llvm::StringSet<> seenSynthKeys;
  for (auto &kv : *synth) {
    auto *keyNode = ::llvm::dyn_cast<::llvm::yaml::ScalarNode>(kv.getKey());
    if (!keyNode)
      return makeYamlSchemaErr(sm, kv.getKey(),
                               "mapping keys must be scalar strings");
    ::llvm::SmallString<32> kbuf;
    StringRef key = keyNode->getValue(kbuf);
    if (!seenSynthKeys.insert(key).second) {
      const char *kind = isKnownYamlSection(key) ? "section" : "key";
      return makeYamlSchemaErr(sm, kv.getKey(),
                               ::llvm::Twine("duplicate ") + kind + " 'synth." +
                                   key + "'");
    }
    if (auto e = applySynthYAMLEntry(cfg, key, kv.getKey(), kv.getValue(), sm))
      return std::move(e);
  }
  if (stream.failed())
    return buildParseErr();

  ++topIt;
  if (topIt != topMap->end()) {
    auto *keyNode = ::llvm::dyn_cast<::llvm::yaml::ScalarNode>(topIt->getKey());
    if (!keyNode)
      return makeYamlSchemaErr(sm, topIt->getKey(),
                               "top-level keys must be scalar strings");
    ::llvm::SmallString<16> keyBuffer;
    StringRef key = keyNode->getValue(keyBuffer);
    if (key == "synth")
      return makeYamlSchemaErr(sm, topIt->getKey(),
                               "duplicate section 'synth'");
    if (::llvm::isa_and_nonnull<::llvm::yaml::MappingNode>(topIt->getValue()))
      return makeYamlSchemaErr(sm, topIt->getKey(),
                               ::llvm::Twine("unknown section '") + key + "'");
    return makeYamlSchemaErr(sm, topIt->getKey(),
                             ::llvm::Twine("unknown key '") + key + "'");
  }

  if (auto e = validate(cfg))
    return std::move(e);
  return cfg;
}

::llvm::Expected<SynthConfig> parseSynthConfigTOML(StringRef body) {
  SynthConfig cfg;
  StringRef section;
  std::set<std::string> seenSections;
  std::set<std::string> seenKeys;
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
      if (!isKnownTomlSection(section))
        return makeErr("toml line " + ::llvm::Twine(lineNo) +
                       ": unknown section '" + section + "'");
      if (!seenSections.insert(section.str()).second)
        return makeErr("toml line " + ::llvm::Twine(lineNo) +
                       ": duplicate section '" + section + "'");
      continue;
    }

    auto eq = line.find('=');
    if (eq == StringRef::npos)
      return makeErr("toml line " + ::llvm::Twine(lineNo) +
                     ": expected `key = value`");
    StringRef key = trim(line.substr(0, eq));
    StringRef value = trim(line.substr(eq + 1));
    std::string fullKey =
        section.empty() ? key.str() : (section + "." + key).str();
    if (!seenKeys.insert(fullKey).second)
      return makeErr("toml line " + ::llvm::Twine(lineNo) +
                     ": duplicate key '" + fullKey + "'");
    if (auto e = applyTomlKV(cfg, section, key, value, lineNo))
      return std::move(e);
  }
  if (auto e = validate(cfg))
    return std::move(e);
  return cfg;
}

::llvm::Expected<SynthConfig> loadSynthConfig(StringRef path) {
  auto bufOrErr = ::llvm::MemoryBuffer::getFile(path);
  if (auto ec = bufOrErr.getError())
    return makeErr("cannot open config '" + path + "': " + ec.message());
  StringRef body = bufOrErr.get()->getBuffer();

  StringRef ext = ::llvm::sys::path::extension(path);
  if (ext == ".yaml" || ext == ".yml")
    return parseSynthConfigYAML(body);
  if (ext == ".toml")
    return parseSynthConfigTOML(body);
  return makeErr("unrecognized config extension '" + ext +
                 "' (expected .yaml/.yml/.toml)");
}

} // namespace loom

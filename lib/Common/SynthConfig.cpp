#include "Common/SynthConfig.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"
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

bool stringIsKnownStrategy(StringRef s) {
  return s == "anchor" || s == "mcs" || s == "incremental" ||
         s == "incremental_random";
}

bool stringIsKnownOrderHeuristic(StringRef s) {
  return s == "largest_first" || s == "smallest_first" ||
         s == "random_seeded";
}

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
    return makeErr(
        "synth.strategy must be one of "
        "anchor|mcs|incremental|incremental_random, got '" +
        cfg.strategy + "'");
  for (const auto &s : cfg.fallbackChain)
    if (!stringIsKnownStrategy(s))
      return makeErr(
          "synth.fallback_chain entry must be one of "
          "anchor|mcs|incremental|incremental_random, got '" +
          s + "'");
  if (!stringIsKnownOrderHeuristic(cfg.incrementalInputOrderHeuristic))
    return makeErr("synth.incremental.input_order_heuristic must be one of "
                   "largest_first|smallest_first|random_seeded, got '" +
                   cfg.incrementalInputOrderHeuristic + "'");
  if (!stringIsKnownOrderHeuristic(
          cfg.incrementalRandomInputOrderHeuristic))
    return makeErr(
        "synth.incremental_random.input_order_heuristic must be one of "
        "largest_first|smallest_first|random_seeded, got '" +
        cfg.incrementalRandomInputOrderHeuristic + "'");
  if (!(cfg.costMuxPenalty >= 0.0 && cfg.costDemuxPenalty >= 0.0 &&
        cfg.costCarryPenalty >= 0.0))
    return makeErr("synth.cost.{mux,demux,carry}_penalty must all be >= 0");
  if (cfg.incrementalRandomRestarts == 0)
    return makeErr("synth.incremental_random.restarts must be >= 1");
  if (cfg.mcsBranchWorkers == 0)
    return makeErr("synth.mcs.branch_workers must be >= 1");
  if (cfg.mcsCandidateCap == 0)
    return makeErr("synth.mcs.candidate_cap must be >= 1");
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

// Apply a single `key: value` at the top of the synth map. Handles both
// scalar leaves (e.g. `strategy: ...`) and nested maps (e.g. `parallelism:`).
::llvm::Error applySynthYAMLEntry(::loom::SynthConfig &cfg, StringRef key,
                                  ::llvm::yaml::Node *value);

// Apply a single nested mapping (e.g. the `parallelism:` block) by dispatching
// on `parent.child` keys.
::llvm::Error applySynthYAMLNested(::loom::SynthConfig &cfg,
                                   StringRef parent,
                                   ::llvm::yaml::MappingNode *map) {
  for (auto &kv : *map) {
    ::llvm::SmallString<32> kbuf;
    StringRef child = scalarValue(kv.getKey(), kbuf);
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
        auto v = parseUInt(val, "synth.parallelism.workers", /*acceptAuto=*/true);
        if (!v)
          return v.takeError();
        cfg.parallelismWorkers = static_cast<unsigned>(*v);
      }
      continue;
    }

    if (parent == "coverage_verifier") {
      ::llvm::SmallString<32> vbuf;
      StringRef val = scalarValue(valNode, vbuf);
      if (child == "enabled") {
        auto v = parseBool(val, "synth.coverage_verifier.enabled");
        if (!v)
          return v.takeError();
        cfg.coverageVerifierEnabled = *v;
      } else if (child == "parallel_match") {
        auto v = parseBool(val, "synth.coverage_verifier.parallel_match");
        if (!v)
          return v.takeError();
        cfg.coverageVerifierParallelMatch = *v;
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
      }
      continue;
    }

    if (parent == "incremental") {
      ::llvm::SmallString<32> vbuf;
      StringRef val = scalarValue(valNode, vbuf);
      if (child == "input_order_heuristic") {
        cfg.incrementalInputOrderHeuristic = stripQuotes(val).str();
      } else if (child == "coverage_verify_each_attempt") {
        auto v = parseBool(
            val, "synth.incremental.coverage_verify_each_attempt");
        if (!v)
          return v.takeError();
        cfg.incrementalCoverageVerifyEachAttempt = *v;
      }
      continue;
    }

    if (parent == "incremental_random") {
      ::llvm::SmallString<32> vbuf;
      StringRef val = scalarValue(valNode, vbuf);
      if (child == "restarts") {
        auto v = parseUInt(val, "synth.incremental_random.restarts",
                           /*acceptAuto=*/false);
        if (!v)
          return v.takeError();
        cfg.incrementalRandomRestarts = static_cast<unsigned>(*v);
      } else if (child == "seed") {
        auto v = parseUInt(val, "synth.incremental_random.seed",
                           /*acceptAuto=*/false);
        if (!v)
          return v.takeError();
        cfg.incrementalRandomSeed = *v;
      } else if (child == "input_order_heuristic") {
        cfg.incrementalRandomInputOrderHeuristic = stripQuotes(val).str();
      }
      continue;
    }

    if (parent == "mcs") {
      ::llvm::SmallString<32> vbuf;
      StringRef val = scalarValue(valNode, vbuf);
      if (child == "timeout_sec") {
        auto v = parseUInt(val, "synth.mcs.timeout_sec", /*acceptAuto=*/false);
        if (!v)
          return v.takeError();
        cfg.mcsTimeoutSec = static_cast<unsigned>(*v);
      } else if (child == "branch_workers") {
        auto v = parseUInt(val, "synth.mcs.branch_workers",
                           /*acceptAuto=*/false);
        if (!v)
          return v.takeError();
        cfg.mcsBranchWorkers = static_cast<unsigned>(*v);
      } else if (child == "candidate_cap") {
        auto v = parseUInt(val, "synth.mcs.candidate_cap",
                           /*acceptAuto=*/false);
        if (!v)
          return v.takeError();
        cfg.mcsCandidateCap = static_cast<unsigned>(*v);
      }
      continue;
    }
    // Unknown nested section: silently ignored for forward compatibility.
  }
  return ::llvm::Error::success();
}

::llvm::Error applySynthYAMLEntry(::loom::SynthConfig &cfg, StringRef key,
                                  ::llvm::yaml::Node *value) {
  // Scalar leaves at the top of the synth map.
  if (key == "strategy") {
    ::llvm::SmallString<32> vbuf;
    StringRef val = scalarValue(value, vbuf);
    cfg.strategy = stripQuotes(val).str();
    return ::llvm::Error::success();
  }
  if (key == "scc_full_unroll") {
    ::llvm::SmallString<32> vbuf;
    StringRef val = scalarValue(value, vbuf);
    auto v = parseBool(val, "synth.scc_full_unroll");
    if (!v)
      return v.takeError();
    cfg.sccFullUnroll = *v;
    return ::llvm::Error::success();
  }
  if (key == "subgraph_share_recurse") {
    ::llvm::SmallString<32> vbuf;
    StringRef val = scalarValue(value, vbuf);
    auto v = parseBool(val, "synth.subgraph_share_recurse");
    if (!v)
      return v.takeError();
    cfg.subgraphShareRecurse = *v;
    return ::llvm::Error::success();
  }

  // fallback_chain is a flow/block sequence of strings.
  if (key == "fallback_chain") {
    auto *seq = ::llvm::dyn_cast_or_null<::llvm::yaml::SequenceNode>(value);
    if (!seq) {
      // Allow an empty/flow scalar like `[]` to be the empty list. If the
      // YAML parser surfaced it as a scalar rather than a sequence, accept
      // the empty case and reject anything else.
      ::llvm::SmallString<8> vbuf;
      StringRef val = trim(scalarValue(value, vbuf));
      if (val.empty() || val == "[]")
        return ::llvm::Error::success();
      return makeErr(
          "synth.fallback_chain must be a YAML list, got '" + val + "'");
    }
    cfg.fallbackChain.clear();
    for (auto &item : *seq) {
      ::llvm::SmallString<32> ibuf;
      StringRef name = scalarValue(&item, ibuf);
      cfg.fallbackChain.push_back(stripQuotes(name).str());
    }
    return ::llvm::Error::success();
  }

  // Nested maps.
  if (auto *m = ::llvm::dyn_cast_or_null<::llvm::yaml::MappingNode>(value))
    return applySynthYAMLNested(cfg, key, m);

  // Unknown scalar key: forward-compat, silently ignore.
  return ::llvm::Error::success();
}

// ---------- TOML helpers ---------------------------------------------------
//
// Subset of TOML the loader recognizes:
//   [synth]
//   strategy = "anchor"
//   scc_full_unroll = false
//   fallback_chain = ["anchor", "mcs"]
//
//   [synth.parallelism]
//   cross_group = true
//   workers = "auto"   # or 0
//
//   [synth.cost]
//   mux_penalty = 1.5
//   ...
//
// Sections outside the `synth.*` family are silently ignored for
// forward-compat, matching the TechMapConfig loader's behavior.

::llvm::Expected<std::vector<std::string>>
parseTomlStringArray(StringRef v, StringRef ctx) {
  v = trim(v);
  if (v.size() < 2 || v.front() != '[' || v.back() != ']')
    return makeErr(ctx + ": expected '[...]' string list, got '" + v + "'");
  StringRef body = trim(v.drop_front().drop_back());
  std::vector<std::string> out;
  if (body.empty())
    return out;
  // Split on commas. We do not support strings containing literal commas.
  while (!body.empty()) {
    auto split = body.split(',');
    StringRef item = trim(split.first);
    body = trim(split.second);
    if (item.empty())
      continue;
    if (item.size() < 2 || (item.front() != '"' && item.front() != '\'') ||
        item.front() != item.back())
      return makeErr(ctx + ": list entry must be quoted, got '" + item + "'");
    out.push_back(item.drop_front().drop_back().str());
  }
  return out;
}

::llvm::Error applyTomlKV(::loom::SynthConfig &cfg, StringRef section,
                          StringRef key, StringRef value) {
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
  auto setUInt64 = [&](uint64_t &target, StringRef ctx) -> ::llvm::Error {
    auto v = parseUInt(value, ctx, /*acceptAuto=*/false);
    if (!v)
      return v.takeError();
    target = *v;
    return ::llvm::Error::success();
  };

  if (section == "synth") {
    if (key == "strategy") {
      cfg.strategy = stripQuotes(value).str();
    } else if (key == "scc_full_unroll") {
      return setBool(cfg.sccFullUnroll, "synth.scc_full_unroll");
    } else if (key == "subgraph_share_recurse") {
      return setBool(cfg.subgraphShareRecurse, "synth.subgraph_share_recurse");
    } else if (key == "fallback_chain") {
      auto v = parseTomlStringArray(value, "synth.fallback_chain");
      if (!v)
        return v.takeError();
      cfg.fallbackChain = std::move(*v);
    }
    return ::llvm::Error::success();
  }
  if (section == "synth.parallelism") {
    if (key == "cross_group")
      return setBool(cfg.parallelismCrossGroup,
                     "synth.parallelism.cross_group");
    if (key == "workers")
      return setUInt32(cfg.parallelismWorkers, "synth.parallelism.workers",
                       /*acceptAuto=*/true);
    return ::llvm::Error::success();
  }
  if (section == "synth.coverage_verifier") {
    if (key == "enabled")
      return setBool(cfg.coverageVerifierEnabled,
                     "synth.coverage_verifier.enabled");
    if (key == "parallel_match")
      return setBool(cfg.coverageVerifierParallelMatch,
                     "synth.coverage_verifier.parallel_match");
    return ::llvm::Error::success();
  }
  if (section == "synth.cost") {
    if (key == "mux_penalty")
      return setDouble(cfg.costMuxPenalty, "synth.cost.mux_penalty");
    if (key == "demux_penalty")
      return setDouble(cfg.costDemuxPenalty, "synth.cost.demux_penalty");
    if (key == "carry_penalty")
      return setDouble(cfg.costCarryPenalty, "synth.cost.carry_penalty");
    return ::llvm::Error::success();
  }
  if (section == "synth.anchor") {
    if (key == "allow_intra_position_mux")
      return setBool(cfg.anchorAllowIntraPositionMux,
                     "synth.anchor.allow_intra_position_mux");
    return ::llvm::Error::success();
  }
  if (section == "synth.incremental") {
    if (key == "input_order_heuristic") {
      cfg.incrementalInputOrderHeuristic = stripQuotes(value).str();
    } else if (key == "coverage_verify_each_attempt") {
      return setBool(cfg.incrementalCoverageVerifyEachAttempt,
                     "synth.incremental.coverage_verify_each_attempt");
    }
    return ::llvm::Error::success();
  }
  if (section == "synth.incremental_random") {
    if (key == "restarts") {
      return setUInt32(cfg.incrementalRandomRestarts,
                       "synth.incremental_random.restarts",
                       /*acceptAuto=*/false);
    }
    if (key == "seed")
      return setUInt64(cfg.incrementalRandomSeed,
                       "synth.incremental_random.seed");
    if (key == "input_order_heuristic") {
      cfg.incrementalRandomInputOrderHeuristic = stripQuotes(value).str();
    }
    return ::llvm::Error::success();
  }
  if (section == "synth.mcs") {
    if (key == "timeout_sec")
      return setUInt32(cfg.mcsTimeoutSec, "synth.mcs.timeout_sec",
                       /*acceptAuto=*/false);
    if (key == "branch_workers")
      return setUInt32(cfg.mcsBranchWorkers, "synth.mcs.branch_workers",
                       /*acceptAuto=*/false);
    if (key == "candidate_cap")
      return setUInt32(cfg.mcsCandidateCap, "synth.mcs.candidate_cap",
                       /*acceptAuto=*/false);
    return ::llvm::Error::success();
  }
  // Unknown section: silently ignored.
  return ::llvm::Error::success();
}

} // namespace

namespace loom {

::llvm::Expected<SynthConfig> parseSynthConfigYAML(StringRef body) {
  // Hand-walk a tiny YAML subset rather than registering MappingTraits, so
  // that unknown keys are tolerated with a precise location.
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

  ::llvm::yaml::MappingNode *synth = nullptr;
  for (auto &kv : *topMap) {
    auto *keyNode = ::llvm::dyn_cast<::llvm::yaml::ScalarNode>(kv.getKey());
    if (!keyNode)
      continue;
    ::llvm::SmallString<16> kbuf;
    StringRef keyName = keyNode->getValue(kbuf);
    if (keyName == "synth") {
      synth = ::llvm::dyn_cast<::llvm::yaml::MappingNode>(kv.getValue());
      break;
    }
  }
  if (stream.failed())
    return buildParseErr();
  if (!synth)
    return makeErr("yaml: top-level mapping must contain a 'synth:' key");

  for (auto &kv : *synth) {
    auto *keyNode = ::llvm::dyn_cast<::llvm::yaml::ScalarNode>(kv.getKey());
    if (!keyNode)
      continue;
    ::llvm::SmallString<32> kbuf;
    StringRef key = keyNode->getValue(kbuf);
    if (auto e = applySynthYAMLEntry(cfg, key, kv.getValue()))
      return std::move(e);
  }
  if (stream.failed())
    return buildParseErr();

  if (auto e = validate(cfg))
    return std::move(e);
  return cfg;
}

::llvm::Expected<SynthConfig> parseSynthConfigTOML(StringRef body) {
  SynthConfig cfg;
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
    if (auto e = applyTomlKV(cfg, section, key, value))
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

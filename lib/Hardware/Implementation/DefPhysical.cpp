#include "Hardware/Implementation/DefPhysical.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::hardware {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "def_physical_invalid: " + message);
}

bool keyword(llvm::StringRef token, llvm::StringRef expected) {
  return token.equals_insensitive(expected);
}

bool punctuation(char character) {
  return character == '(' || character == ')' || character == ';' ||
         character == '+' || character == '-';
}

llvm::Expected<std::vector<std::string>> lex(llvm::StringRef contents) {
  if (contents.empty() || contents.contains('\0') || contents.contains('\r'))
    return invalid("DEF must be nonempty LF text without NUL");
  std::vector<std::string> tokens;
  for (std::size_t offset = 0; offset < contents.size();) {
    const char character = contents[offset];
    if (character == '#') {
      while (offset < contents.size() && contents[offset] != '\n')
        ++offset;
      continue;
    }
    if (llvm::isSpace(static_cast<unsigned char>(character))) {
      ++offset;
      continue;
    }
    if (punctuation(character)) {
      tokens.emplace_back(1, character);
      ++offset;
      continue;
    }
    if (character == '"') {
      const std::size_t begin = offset++;
      bool escaped = false;
      while (offset < contents.size()) {
        const char next = contents[offset++];
        if (next == '"' && !escaped)
          break;
        escaped = next == '\\' && !escaped;
        if (next != '\\')
          escaped = false;
      }
      if (contents[offset - 1] != '"')
        return invalid("quoted DEF token is unterminated");
      tokens.push_back(contents.slice(begin, offset).str());
      continue;
    }
    const std::size_t begin = offset;
    while (offset < contents.size() &&
           !llvm::isSpace(static_cast<unsigned char>(contents[offset])) &&
           contents[offset] != '#' && !punctuation(contents[offset]))
      ++offset;
    if (begin == offset)
      return invalid("DEF contains an unsupported token");
    tokens.push_back(contents.slice(begin, offset).str());
  }
  return tokens;
}

llvm::Expected<std::uint64_t> parseCount(llvm::StringRef token,
                                         llvm::StringRef section) {
  std::uint64_t value = 0;
  const auto result = std::from_chars(token.begin(), token.end(), value, 10);
  if (result.ec != std::errc() || result.ptr != token.end())
    return invalid(section + " count is not an unsigned integer");
  return value;
}

std::optional<DefSupplyUse> parseUse(llvm::StringRef token) {
  if (keyword(token, "POWER"))
    return DefSupplyUse::Power;
  if (keyword(token, "GROUND"))
    return DefSupplyUse::Ground;
  return std::nullopt;
}

struct ParsedSections final {
  std::optional<std::string> designName;
  std::vector<DefSpecialNet> specialNets;
  std::vector<DefTopLevelPin> pins;
  bool sawEndDesign = false;
  bool sawRoute = false;
};

llvm::Expected<std::vector<std::vector<std::string>>>
parseEntries(const std::vector<std::string> &tokens, std::size_t &offset,
             llvm::StringRef section) {
  if (offset >= tokens.size())
    return invalid(section + " count is absent");
  auto count = parseCount(tokens[offset++], section);
  if (!count)
    return count.takeError();
  if (offset >= tokens.size() || tokens[offset++] != ";")
    return invalid(section + " header lacks its terminator");

  std::vector<std::vector<std::string>> entries;
  while (offset < tokens.size()) {
    if (keyword(tokens[offset], "END")) {
      if (offset + 1 >= tokens.size() || !keyword(tokens[offset + 1], section))
        return invalid(section + " has a foreign END marker");
      offset += 2;
      break;
    }
    if (tokens[offset++] != "-")
      return invalid(section + " entry must begin with '-'");
    std::vector<std::string> entry;
    while (offset < tokens.size() && tokens[offset] != ";")
      entry.push_back(tokens[offset++]);
    if (offset >= tokens.size() || entry.empty())
      return invalid(section + " entry is unterminated or unnamed");
    ++offset;
    entries.push_back(std::move(entry));
  }
  if (entries.size() != *count)
    return invalid(section + " entry count does not match its header");
  return entries;
}

std::optional<std::string> clauseValue(const std::vector<std::string> &entry,
                                       llvm::StringRef name) {
  for (std::size_t index = 1; index + 2 < entry.size(); ++index)
    if (entry[index] == "+" && keyword(entry[index + 1], name))
      return entry[index + 2];
  return std::nullopt;
}

bool hasClause(const std::vector<std::string> &entry, llvm::StringRef name) {
  for (std::size_t index = 1; index + 1 < entry.size(); ++index)
    if (entry[index] == "+" && keyword(entry[index + 1], name))
      return true;
  return false;
}

llvm::Expected<ParsedSections>
parseSections(const std::vector<std::string> &tokens) {
  ParsedSections parsed;
  std::set<std::string> specialNames;
  std::set<std::string> pinNames;
  for (std::size_t offset = 0; offset < tokens.size();) {
    if (keyword(tokens[offset], "DESIGN")) {
      if (parsed.designName || offset + 2 >= tokens.size() ||
          tokens[offset + 2] != ";")
        return invalid("DESIGN statement is missing, duplicate, or malformed");
      parsed.designName = tokens[offset + 1];
      offset += 3;
      continue;
    }
    if (keyword(tokens[offset], "PINS")) {
      ++offset;
      auto entries = parseEntries(tokens, offset, "PINS");
      if (!entries)
        return entries.takeError();
      for (const auto &entry : *entries) {
        if (!pinNames.insert(entry.front()).second)
          return invalid("PINS contains a duplicate pin name");
        const auto net = clauseValue(entry, "NET");
        const auto use = clauseValue(entry, "USE");
        parsed.pins.push_back(
            {entry.front(), net.value_or(""),
             use ? parseUse(*use) : std::nullopt,
             hasClause(entry, "PLACED") || hasClause(entry, "FIXED"),
             hasClause(entry, "LAYER")});
      }
      continue;
    }
    if (keyword(tokens[offset], "SPECIALNETS")) {
      ++offset;
      auto entries = parseEntries(tokens, offset, "SPECIALNETS");
      if (!entries)
        return entries.takeError();
      for (const auto &entry : *entries) {
        if (!specialNames.insert(entry.front()).second)
          return invalid("SPECIALNETS contains a duplicate net name");
        const auto use = clauseValue(entry, "USE");
        const bool routed =
            hasClause(entry, "ROUTED") || hasClause(entry, "FIXED");
        parsed.sawRoute |= routed;
        parsed.specialNets.push_back(
            {entry.front(), use ? parseUse(*use) : std::nullopt, routed});
      }
      continue;
    }
    if (keyword(tokens[offset], "NETS")) {
      ++offset;
      auto entries = parseEntries(tokens, offset, "NETS");
      if (!entries)
        return entries.takeError();
      for (const auto &entry : *entries)
        parsed.sawRoute |=
            hasClause(entry, "ROUTED") || hasClause(entry, "FIXED");
      continue;
    }
    if (keyword(tokens[offset], "END") && offset + 1 < tokens.size() &&
        keyword(tokens[offset + 1], "DESIGN")) {
      parsed.sawEndDesign = true;
      offset += 2;
      continue;
    }
    ++offset;
  }
  return parsed;
}

} // namespace

llvm::Expected<DefPhysicalDesign>
parseDefPhysicalDesign(llvm::StringRef contents, llvm::StringRef expectedTop,
                       RepresentationPhysicalStage stage) {
  auto tokens = lex(contents);
  if (!tokens)
    return tokens.takeError();
  auto parsed = parseSections(*tokens);
  if (!parsed)
    return parsed.takeError();
  if (!parsed->designName || *parsed->designName != expectedTop)
    return invalid("DESIGN does not name the exact representation top");
  if (!parsed->sawEndDesign)
    return invalid("END DESIGN is absent");
  if ((stage == RepresentationPhysicalStage::Routed ||
       stage == RepresentationPhysicalStage::Extracted) &&
      !parsed->sawRoute)
    return invalid("routed physical state has no routed DEF network");
  return DefPhysicalDesign{std::move(*parsed->designName),
                           std::move(parsed->specialNets),
                           std::move(parsed->pins)};
}

std::optional<DefSingleSupplyNetwork>
deriveDefSingleSupplyNetwork(const DefPhysicalDesign &design) {
  const DefSpecialNet *power = nullptr;
  const DefSpecialNet *ground = nullptr;
  for (const DefSpecialNet &net : design.specialNets) {
    if (!net.use)
      continue;
    const DefSpecialNet **slot =
        *net.use == DefSupplyUse::Power ? &power : &ground;
    if (*slot || !net.routed)
      return std::nullopt;
    *slot = &net;
  }
  if (!power || !ground)
    return std::nullopt;

  bool powerPin = false;
  bool groundPin = false;
  for (const DefTopLevelPin &pin : design.topLevelPins) {
    if (!pin.use)
      continue;
    if (!pin.placedOrFixed || !pin.hasLayerGeometry || pin.net.empty())
      return std::nullopt;
    if (*pin.use == DefSupplyUse::Power) {
      if (pin.net != power->name)
        return std::nullopt;
      powerPin = true;
    } else {
      if (pin.net != ground->name)
        return std::nullopt;
      groundPin = true;
    }
  }
  if (!powerPin || !groundPin)
    return std::nullopt;
  return DefSingleSupplyNetwork{power->name, ground->name};
}

} // namespace loom::hardware

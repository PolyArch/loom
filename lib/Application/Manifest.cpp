#include "Application/Manifest.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace loom::application {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      "application_manifest_invalid: " + message);
}

llvm::Error
rejectUnknownFields(const llvm::json::Object &object, llvm::StringRef context,
                    std::initializer_list<llvm::StringRef> allowed) {
  for (const auto &field : object) {
    const llvm::StringRef key(field.first);
    if (!llvm::is_contained(allowed, key))
      return invalid(context + " has unknown field '" + key + "'");
  }
  return llvm::Error::success();
}

llvm::Expected<llvm::StringRef> requireString(const llvm::json::Object &object,
                                              llvm::StringRef field,
                                              llvm::StringRef context) {
  std::optional<llvm::StringRef> value = object.getString(field);
  if (!value)
    return invalid(context + " requires string field '" + field + "'");
  return *value;
}

llvm::Expected<const llvm::json::Object *>
requireObject(const llvm::json::Object &object, llvm::StringRef field,
              llvm::StringRef context) {
  const llvm::json::Object *value = object.getObject(field);
  if (!value)
    return invalid(context + " requires object field '" + field + "'");
  return value;
}

llvm::Expected<const llvm::json::Array *>
requireArray(const llvm::json::Object &object, llvm::StringRef field,
             llvm::StringRef context) {
  const llvm::json::Array *value = object.getArray(field);
  if (!value)
    return invalid(context + " requires array field '" + field + "'");
  return value;
}

bool validLogicalName(llvm::StringRef value) {
  if (value.empty())
    return false;
  auto isLowerOrDigit = [](unsigned char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= '0' && character <= '9');
  };
  if (!isLowerOrDigit(value.front()))
    return false;
  return llvm::all_of(value.drop_front(), [&](unsigned char character) {
    return isLowerOrDigit(character) || character == '-' || character == '_' ||
           character == '.';
  });
}

llvm::Error validateLogicalName(llvm::StringRef value,
                                llvm::StringRef context) {
  if (!validLogicalName(value))
    return invalid(context + " must be a lowercase stable logical name");
  return llvm::Error::success();
}

llvm::Error validateRelativePath(llvm::StringRef value,
                                 llvm::StringRef context) {
  if (value.empty() || value.starts_with('/') || value.ends_with('/') ||
      value.contains("//") || value.contains('\\'))
    return invalid(context + " must be a normalized relative path");
  for (unsigned char character : value.bytes())
    if (character < 0x21 || character > 0x7e)
      return invalid(context + " must contain visible ASCII path bytes");
  llvm::SmallVector<llvm::StringRef, 8> components;
  value.split(components, '/');
  if (llvm::any_of(components, [](llvm::StringRef component) {
        return component.empty() || component == "." || component == "..";
      }))
    return invalid(context + " contains a non-canonical path component");
  return llvm::Error::success();
}

llvm::Error validateOption(llvm::StringRef value, llvm::StringRef context) {
  if (value.empty())
    return invalid(context + " cannot be empty");
  for (unsigned char character : value.bytes())
    if (character < 0x20 || character > 0x7e)
      return invalid(context + " must contain printable ASCII bytes");
  return llvm::Error::success();
}

template <typename Range, typename Projection>
llvm::Error requireStrictOrder(const Range &values, Projection projection,
                               llvm::StringRef context) {
  for (std::size_t index = 1; index < values.size(); ++index)
    if (!(projection(values[index - 1]) < projection(values[index])))
      return invalid(context + " must be strictly ordered and unique");
  return llvm::Error::success();
}

llvm::Expected<SourceKind> parseSourceKind(llvm::StringRef value) {
  if (value == "gitlink")
    return SourceKind::Gitlink;
  if (value == "repository")
    return SourceKind::Repository;
  return invalid("source kind must be 'gitlink' or 'repository'");
}

llvm::Expected<LanguageMode> parseLanguageMode(llvm::StringRef value) {
  if (value == "c")
    return LanguageMode::C;
  if (value == "c++")
    return LanguageMode::Cxx;
  return invalid("language mode must be 'c' or 'c++'");
}

llvm::Expected<OracleKind> parseOracleKind(llvm::StringRef value) {
  if (value == "exact")
    return OracleKind::Exact;
  if (value == "typed_invariant")
    return OracleKind::TypedInvariant;
  return invalid("oracle kind must be 'exact' or 'typed_invariant'");
}

llvm::Expected<ExecutionSelection>
parseExecutionSelection(llvm::StringRef value) {
  if (value == "smoke")
    return ExecutionSelection::Smoke;
  if (value == "validation")
    return ExecutionSelection::Validation;
  if (value == "scale_eda")
    return ExecutionSelection::ScaleEda;
  return invalid(
      "execution selection must be 'smoke', 'validation', or 'scale_eda'");
}

llvm::Expected<std::vector<std::string>>
parseStringArray(const llvm::json::Array &array, const std::string &context,
                 bool pathValues, bool requireCanonicalOrder) {
  std::vector<std::string> result;
  result.reserve(array.size());
  for (const llvm::json::Value &element : array) {
    std::optional<llvm::StringRef> value = element.getAsString();
    if (!value)
      return invalid(context + " entries must be strings");
    if (pathValues) {
      if (llvm::Error error = validateRelativePath(*value, context + " entry"))
        return std::move(error);
    } else if (llvm::Error error = validateOption(*value, context + " entry")) {
      return std::move(error);
    }
    result.push_back(value->str());
  }
  if (requireCanonicalOrder)
    if (llvm::Error error = requireStrictOrder(
            result,
            [](const std::string &value) -> const std::string & {
              return value;
            },
            context))
      return std::move(error);
  return result;
}

bool hasSourceExtension(llvm::StringRef path, LanguageMode language) {
  if (path.ends_with(".c"))
    return true;
  if (language == LanguageMode::C)
    return false;
  return path.ends_with(".cc") || path.ends_with(".cpp") ||
         path.ends_with(".cxx") || path.ends_with(".c++") ||
         path.ends_with(".C");
}

llvm::Expected<SourceSelection> parseSource(const llvm::json::Object &object,
                                            const std::string &context) {
  if (llvm::Error error =
          rejectUnknownFields(object, context, {"kind", "root"}))
    return std::move(error);
  auto kindText = requireString(object, "kind", context);
  if (!kindText)
    return kindText.takeError();
  auto root = requireString(object, "root", context);
  if (!root)
    return root.takeError();
  auto kind = parseSourceKind(*kindText);
  if (!kind)
    return kind.takeError();
  if (llvm::Error error = validateRelativePath(*root, context + " root"))
    return std::move(error);
  return SourceSelection{*kind, root->str()};
}

llvm::Expected<BuildSelection> parseBuild(const llvm::json::Object &object,
                                          const std::string &context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context,
          {"entry", "language", "sources", "compiler_options", "link_options"}))
    return std::move(error);
  auto entry = requireString(object, "entry", context);
  if (!entry)
    return entry.takeError();
  auto languageText = requireString(object, "language", context);
  if (!languageText)
    return languageText.takeError();
  auto sourcesArray = requireArray(object, "sources", context);
  if (!sourcesArray)
    return sourcesArray.takeError();
  auto compilerArray = requireArray(object, "compiler_options", context);
  if (!compilerArray)
    return compilerArray.takeError();
  auto linkArray = requireArray(object, "link_options", context);
  if (!linkArray)
    return linkArray.takeError();
  if (llvm::Error error = validateRelativePath(*entry, context + " entry"))
    return std::move(error);
  auto language = parseLanguageMode(*languageText);
  if (!language)
    return language.takeError();
  auto sources =
      parseStringArray(**sourcesArray, context + " sources", true, true);
  if (!sources)
    return sources.takeError();
  auto compilerOptions = parseStringArray(
      **compilerArray, context + " compiler_options", false, false);
  if (!compilerOptions)
    return compilerOptions.takeError();
  auto linkOptions =
      parseStringArray(**linkArray, context + " link_options", false, false);
  if (!linkOptions)
    return linkOptions.takeError();
  if (sources->empty())
    return invalid(context + " requires a nonempty exact source selection");
  if (!std::binary_search(sources->begin(), sources->end(), entry->str()))
    return invalid(context + " entry is not in the exact source selection");
  for (const std::string &source : *sources)
    if (!hasSourceExtension(source, *language))
      return invalid(context + " source '" + source +
                     "' is outside the selected C/C++ language mode");
  return BuildSelection{entry->str(), *language, std::move(*sources),
                        std::move(*compilerOptions), std::move(*linkOptions)};
}

llvm::Expected<CachedInput> parseCachedInput(const llvm::json::Object &object,
                                             const std::string &context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context, {"logical_name", "path", "sha256"}))
    return std::move(error);
  auto logicalName = requireString(object, "logical_name", context);
  if (!logicalName)
    return logicalName.takeError();
  auto path = requireString(object, "path", context);
  if (!path)
    return path.takeError();
  auto digestText = requireString(object, "sha256", context);
  if (!digestText)
    return digestText.takeError();
  if (llvm::Error error =
          validateLogicalName(*logicalName, context + " logical_name"))
    return std::move(error);
  if (llvm::Error error = validateRelativePath(*path, context + " path"))
    return std::move(error);
  auto digest = parseBlobDigestHex(*digestText);
  if (!digest)
    return invalid(
        context + " has invalid sha256: " + llvm::toString(digest.takeError()));
  return CachedInput{logicalName->str(), path->str(), std::move(*digest)};
}

llvm::Expected<OracleSelection> parseOracle(const llvm::json::Object &object,
                                            const std::string &context) {
  if (llvm::Error error =
          rejectUnknownFields(object, context, {"kind", "entry"}))
    return std::move(error);
  auto kindText = requireString(object, "kind", context);
  if (!kindText)
    return kindText.takeError();
  auto entry = requireString(object, "entry", context);
  if (!entry)
    return entry.takeError();
  auto kind = parseOracleKind(*kindText);
  if (!kind)
    return kind.takeError();
  if (llvm::Error error = validateRelativePath(*entry, context + " entry"))
    return std::move(error);
  return OracleSelection{*kind, entry->str()};
}

llvm::Expected<WorkloadInputSelection>
parseWorkloadInput(const llvm::json::Object &object,
                   const std::string &context) {
  if (llvm::Error error = rejectUnknownFields(
          object, context,
          {"name", "workload", "runtime_input", "cached_inputs", "oracle"}))
    return std::move(error);
  auto name = requireString(object, "name", context);
  if (!name)
    return name.takeError();
  auto workload = requireString(object, "workload", context);
  if (!workload)
    return workload.takeError();
  auto runtimeInput = requireString(object, "runtime_input", context);
  if (!runtimeInput)
    return runtimeInput.takeError();
  auto cachedArray = requireArray(object, "cached_inputs", context);
  if (!cachedArray)
    return cachedArray.takeError();
  auto oracleObject = requireObject(object, "oracle", context);
  if (!oracleObject)
    return oracleObject.takeError();
  if (llvm::Error error = validateLogicalName(*name, context + " name"))
    return std::move(error);
  if (llvm::Error error = validateLogicalName(*workload, context + " workload"))
    return std::move(error);
  if (llvm::Error error =
          validateLogicalName(*runtimeInput, context + " runtime_input"))
    return std::move(error);
  auto cached =
      parseStringArray(**cachedArray, context + " cached_inputs", false, true);
  if (!cached)
    return cached.takeError();
  for (const std::string &logicalName : *cached)
    if (llvm::Error error = validateLogicalName(
            logicalName, context + " cached input reference"))
      return std::move(error);
  auto oracle = parseOracle(**oracleObject, context + " oracle");
  if (!oracle)
    return oracle.takeError();
  return WorkloadInputSelection{name->str(), workload->str(),
                                runtimeInput->str(), std::move(*cached),
                                std::move(*oracle)};
}

llvm::Expected<ApplicationDefinition>
parseApplication(const llvm::json::Object &object, std::size_t ordinal) {
  const std::string context = "application[" + std::to_string(ordinal) + "]";
  if (llvm::Error error =
          rejectUnknownFields(object, context,
                              {"identity", "source", "build", "cached_inputs",
                               "inputs", "selections"}))
    return std::move(error);
  auto identity = requireString(object, "identity", context);
  if (!identity)
    return identity.takeError();
  auto sourceObject = requireObject(object, "source", context);
  if (!sourceObject)
    return sourceObject.takeError();
  auto buildObject = requireObject(object, "build", context);
  if (!buildObject)
    return buildObject.takeError();
  auto cachedArray = requireArray(object, "cached_inputs", context);
  if (!cachedArray)
    return cachedArray.takeError();
  auto inputsArray = requireArray(object, "inputs", context);
  if (!inputsArray)
    return inputsArray.takeError();
  auto selectionsArray = requireArray(object, "selections", context);
  if (!selectionsArray)
    return selectionsArray.takeError();
  if (llvm::Error error = validateLogicalName(*identity, context + " identity"))
    return std::move(error);
  auto source = parseSource(**sourceObject, context + " source");
  if (!source)
    return source.takeError();
  auto build = parseBuild(**buildObject, context + " build");
  if (!build)
    return build.takeError();

  std::vector<CachedInput> cachedInputs;
  cachedInputs.reserve((*cachedArray)->size());
  for (std::size_t index = 0; index != (*cachedArray)->size(); ++index) {
    const llvm::json::Object *entry = (**cachedArray)[index].getAsObject();
    if (!entry)
      return invalid(context + " cached_inputs entries must be objects");
    auto parsed = parseCachedInput(*entry, context + " cached_inputs[" +
                                               std::to_string(index) + "]");
    if (!parsed)
      return parsed.takeError();
    cachedInputs.push_back(std::move(*parsed));
  }
  if (llvm::Error error = requireStrictOrder(
          cachedInputs,
          [](const CachedInput &input) -> const std::string & {
            return input.logicalName;
          },
          context + " cached_inputs"))
    return std::move(error);
  std::set<std::string> cachePaths;
  for (const CachedInput &input : cachedInputs)
    if (!cachePaths.insert(input.path).second)
      return invalid(context + " cached_inputs reuse one cache path");

  std::vector<WorkloadInputSelection> inputs;
  inputs.reserve((*inputsArray)->size());
  std::set<std::string> referencedCacheInputs;
  for (std::size_t index = 0; index != (*inputsArray)->size(); ++index) {
    const llvm::json::Object *entry = (**inputsArray)[index].getAsObject();
    if (!entry)
      return invalid(context + " inputs entries must be objects");
    auto parsed = parseWorkloadInput(*entry, context + " inputs[" +
                                                 std::to_string(index) + "]");
    if (!parsed)
      return parsed.takeError();
    for (const std::string &reference : parsed->cachedInputs) {
      const auto found =
          llvm::lower_bound(cachedInputs, reference,
                            [](const CachedInput &input, llvm::StringRef name) {
                              return input.logicalName < name;
                            });
      if (found == cachedInputs.end() || found->logicalName != reference)
        return invalid(context + " input references unknown cached input '" +
                       reference + "'");
      referencedCacheInputs.insert(reference);
    }
    inputs.push_back(std::move(*parsed));
  }
  if (inputs.empty())
    return invalid(context + " requires a named workload/input selection");
  if (llvm::Error error = requireStrictOrder(
          inputs,
          [](const WorkloadInputSelection &input) -> const std::string & {
            return input.name;
          },
          context + " inputs"))
    return std::move(error);
  if (referencedCacheInputs.size() != cachedInputs.size())
    return invalid(context + " declares an unused cached input");

  std::vector<ExecutionSelection> selections;
  selections.reserve((*selectionsArray)->size());
  for (const llvm::json::Value &value : **selectionsArray) {
    std::optional<llvm::StringRef> spelling = value.getAsString();
    if (!spelling)
      return invalid(context + " selections entries must be strings");
    auto parsed = parseExecutionSelection(*spelling);
    if (!parsed)
      return parsed.takeError();
    selections.push_back(*parsed);
  }
  if (selections.empty())
    return invalid(context + " requires an execution selection");
  if (llvm::Error error = requireStrictOrder(
          selections,
          [](ExecutionSelection selection) {
            return static_cast<std::uint8_t>(selection);
          },
          context + " selections"))
    return std::move(error);

  return ApplicationDefinition{identity->str(),   std::move(*source),
                               std::move(*build), std::move(cachedInputs),
                               std::move(inputs), std::move(selections)};
}

} // namespace

llvm::StringRef toString(SourceKind kind) {
  switch (kind) {
  case SourceKind::Gitlink:
    return "gitlink";
  case SourceKind::Repository:
    return "repository";
  }
  llvm_unreachable("unknown SourceKind");
}

llvm::StringRef toString(LanguageMode mode) {
  switch (mode) {
  case LanguageMode::C:
    return "c";
  case LanguageMode::Cxx:
    return "c++";
  }
  llvm_unreachable("unknown LanguageMode");
}

llvm::StringRef toString(OracleKind kind) {
  switch (kind) {
  case OracleKind::Exact:
    return "exact";
  case OracleKind::TypedInvariant:
    return "typed_invariant";
  }
  llvm_unreachable("unknown OracleKind");
}

llvm::StringRef toString(ExecutionSelection selection) {
  switch (selection) {
  case ExecutionSelection::Smoke:
    return "smoke";
  case ExecutionSelection::Validation:
    return "validation";
  case ExecutionSelection::ScaleEda:
    return "scale_eda";
  }
  llvm_unreachable("unknown ExecutionSelection");
}

llvm::Expected<ApplicationManifest>
parseApplicationManifest(llvm::StringRef jsonText) {
  auto parsed = llvm::json::parse(jsonText);
  if (!parsed)
    return invalid("cannot parse JSON: " + llvm::toString(parsed.takeError()));
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return invalid("root must be an object");
  if (llvm::Error error = rejectUnknownFields(
          *root, "manifest", {"schema", "version", "applications"}))
    return std::move(error);
  auto schema = requireString(*root, "schema", "manifest");
  if (!schema)
    return schema.takeError();
  auto version = requireString(*root, "version", "manifest");
  if (!version)
    return version.takeError();
  auto applicationsArray = requireArray(*root, "applications", "manifest");
  if (!applicationsArray)
    return applicationsArray.takeError();
  if (*schema != ApplicationManifest::schemaIdentity ||
      *version != ApplicationManifest::schemaVersion)
    return invalid("unsupported schema or version");
  if ((*applicationsArray)->empty())
    return invalid("applications inventory is empty");

  std::vector<ApplicationDefinition> applications;
  applications.reserve((*applicationsArray)->size());
  for (std::size_t index = 0; index != (*applicationsArray)->size(); ++index) {
    const llvm::json::Object *object =
        (**applicationsArray)[index].getAsObject();
    if (!object)
      return invalid("applications entries must be objects");
    auto application = parseApplication(*object, index);
    if (!application)
      return application.takeError();
    applications.push_back(std::move(*application));
  }
  if (llvm::Error error = requireStrictOrder(
          applications,
          [](const ApplicationDefinition &application) -> const std::string & {
            return application.identity;
          },
          "applications"))
    return std::move(error);
  return ApplicationManifest(std::move(applications));
}

llvm::Expected<ApplicationManifest>
loadApplicationManifest(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path, false, false);
  if (!buffer)
    return llvm::createStringError(
        buffer.getError(), "cannot read application manifest '%s': %s",
        path.str().c_str(), buffer.getError().message().c_str());
  return parseApplicationManifest((*buffer)->getBuffer());
}

std::vector<std::string>
selectApplicationIdentities(const ApplicationManifest &manifest,
                            ExecutionSelection selection) {
  std::vector<std::string> identities;
  for (const ApplicationDefinition &application : manifest.applications())
    if (llvm::is_contained(application.selections, selection))
      identities.push_back(application.identity);
  return identities;
}

} // namespace loom::application

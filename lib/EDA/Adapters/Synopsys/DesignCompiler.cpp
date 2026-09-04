#include "EDA/Adapters/Synopsys/DesignCompiler.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace loom::eda::synopsys {
namespace {

constexpr SynopsysImplementationState acceptedStates[]{
    {hardware::RepresentationRootVariant::Rtl, std::nullopt}};
constexpr llvm::StringLiteral providerInputs[]{"target_library"};
constexpr llvm::StringLiteral declaredOutputs[]{
    designCompilerGateNetlistOutputPath};

const SynopsysInvocationDescriptor descriptor{
    &external_tool::designCompilerProvider(),
    "loom.eda.synopsys.design_compiler.gate_netlist@1",
    SynopsysOperation::LogicSynthesis,
    acceptedStates,
    true,
    true,
    true,
    providerInputs,
    declaredOutputs,
};

const external_tool::ResolvedExternalFile *
findExternal(const SynopsysBundleInputs &inputs, llvm::StringRef slot) {
  const auto found =
      llvm::find_if(inputs.frozen.externalFiles, [&](const auto &file) {
        return file.providerInputSlot == slot;
      });
  return found == inputs.frozen.externalFiles.end() ? nullptr : &*found;
}

bool containsTopModule(llvm::StringRef text, llvm::StringRef top) {
  const auto isIdentifierContinuation = [](char character) {
    return (character >= 'A' && character <= 'Z') ||
           (character >= 'a' && character <= 'z') ||
           (character >= '0' && character <= '9') || character == '_' ||
           character == '$';
  };
  std::size_t offset = 0;
  while (true) {
    const std::size_t found = text.find("module", offset);
    if (found == llvm::StringRef::npos)
      return false;
    const bool leftBoundary =
        found == 0 || !isIdentifierContinuation(text[found - 1]);
    llvm::StringRef rest = text.drop_front(found + 6).ltrim();
    if (leftBoundary && rest.starts_with(top)) {
      const llvm::StringRef suffix = rest.drop_front(top.size());
      if (suffix.empty() || suffix.front() == '(' || suffix.front() == '#' ||
          suffix.front() == ';' || suffix.front() == ' ' ||
          suffix.front() == '\t' || suffix.front() == '\n')
        return true;
    }
    offset = found + 6;
  }
}

} // namespace

const SynopsysInvocationDescriptor &designCompilerDescriptor() {
  return descriptor;
}

namespace {

llvm::Expected<std::string>
renderDriver(llvm::StringRef top, llvm::ArrayRef<std::string> rtlSources,
             llvm::ArrayRef<std::string> generationConstraints,
             llvm::StringRef targetLibrary, DesignCompilerHierarchy hierarchy,
             const DesignCompilerMappedChildren *children) {
  if (!isPortableHdlIdentifier(top))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "top is not a portable HDL identifier");
  if (rtlSources.empty())
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "RTL source inventory is empty");
  for (llvm::StringRef source : rtlSources)
    if (llvm::Error error = validateBundleInputPath(
            descriptor.implementationSemanticIdentity, source))
      return std::move(error);
  if ((generationConstraints.empty() &&
       hierarchy == DesignCompilerHierarchy::Optimize) ||
      !llvm::is_sorted(generationConstraints) ||
      std::adjacent_find(generationConstraints.begin(),
                         generationConstraints.end()) !=
          generationConstraints.end())
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingSemanticInput,
        descriptor.implementationSemanticIdentity,
        "generation constraint inventory is empty or not canonical");
  for (llvm::StringRef constraint : generationConstraints)
    if (llvm::Error error = validateBundleInputPath(
            descriptor.implementationSemanticIdentity, constraint))
      return std::move(error);
  std::string mappedReads;
  std::string childNames;
  if (children) {
    if (children->netlistPaths.empty() || children->definitionNames.empty() ||
        !llvm::is_sorted(children->netlistPaths) ||
        !llvm::is_sorted(children->definitionNames) ||
        std::adjacent_find(children->netlistPaths.begin(),
                           children->netlistPaths.end()) !=
            children->netlistPaths.end() ||
        std::adjacent_find(children->definitionNames.begin(),
                           children->definitionNames.end()) !=
            children->definitionNames.end())
      return makeSynopsysAdapterError(
          SynopsysAdapterFailureKind::MissingSemanticInput,
          descriptor.implementationSemanticIdentity,
          "compiled child inventory is empty or noncanonical");
    for (const auto &path : children->netlistPaths) {
      if (llvm::Error error = validateBundleInputPath(
              descriptor.implementationSemanticIdentity, path))
        return std::move(error);
      auto word =
          renderTclWord(descriptor.implementationSemanticIdentity, path);
      if (!word)
        return word.takeError();
      mappedReads += "read_verilog " + *word + "\n";
    }
    for (const auto &name : children->definitionNames) {
      if (!isPortableHdlIdentifier(name) || name == top)
        return makeSynopsysAdapterError(
            SynopsysAdapterFailureKind::MissingSemanticInput,
            descriptor.implementationSemanticIdentity,
            "compiled child definition is invalid or shadows the parent");
      auto word =
          renderTclWord(descriptor.implementationSemanticIdentity, name);
      if (!word)
        return word.takeError();
      childNames += (childNames.empty() ? "" : " ") + *word;
    }
  }
  auto topWord = renderTclWord(descriptor.implementationSemanticIdentity, top);
  if (!topWord)
    return topWord.takeError();
  auto libraryWord =
      renderTclWord(descriptor.implementationSemanticIdentity, targetLibrary);
  if (!libraryWord)
    return libraryWord.takeError();

  std::string sourceList;
  for (llvm::StringRef source : rtlSources) {
    auto word =
        renderTclWord(descriptor.implementationSemanticIdentity, source);
    if (!word)
      return word.takeError();
    sourceList += (sourceList.empty() ? "" : " ") + *word;
  }
  std::string commands =
      "set loom_target_library [list " + *libraryWord +
      "]\n"
      "set_app_var target_library $loom_target_library\n"
      "set_app_var link_library [concat {*} $loom_target_library]\n" +
      mappedReads + "analyze -format sverilog [list " + sourceList +
      "]\n"
      "elaborate " +
      *topWord +
      "\n"
      "current_design " +
      *topWord +
      "\n"
      "link\n";
  for (llvm::StringRef constraint : generationConstraints) {
    auto word =
        renderTclWord(descriptor.implementationSemanticIdentity, constraint);
    if (!word)
      return word.takeError();
    commands += "read_sdc " + *word + "\n";
  }
  if (children)
    commands +=
        "set loom_child_design_names [list " + childNames +
        "]\n"
        "set loom_child_designs [get_designs $loom_child_design_names]\n"
        "if {[sizeof_collection $loom_child_designs] != " +
        std::to_string(children->definitionNames.size()) +
        "} {error {Compiled child definition inventory changed}}\n"
        "set_dont_touch $loom_child_designs true\n"
        "foreach_in_collection loom_child_cell [get_cells -hierarchical *] {\n"
        "  if {[lsearch -exact $loom_child_design_names "
        "[get_attribute $loom_child_cell ref_name]] >= 0} {\n"
        "    set_dont_touch $loom_child_cell true\n"
        "  }\n"
        "}\n";
  switch (hierarchy) {
  case DesignCompilerHierarchy::Optimize:
    commands += "compile_ultra\ncheck_design\n";
    break;
  case DesignCompilerHierarchy::PreserveDefinitions:
    commands += "set_ungroup [get_designs *] false\n"
                "compile_ultra -no_autoungroup -no_boundary_optimization\n"
                "if {![check_design]} {error {Block design check failed}}\n"
                "change_names -rules verilog -hierarchy\n";
    break;
  }
  const std::string publication =
      children ? "write -format verilog -output {" +
                     designCompilerGateNetlistOutputPath.str() +
                "} [get_object_name [remove_from_collection [get_designs *] "
                "$loom_child_designs]]\n"
               : "write -format verilog -hierarchy -output {" +
                     designCompilerGateNetlistOutputPath.str() + "}\n";
  return renderSynopsysTclBatch(commands, publication);
}

} // namespace

llvm::Expected<std::string> renderDesignCompilerDriver(
    llvm::StringRef top, llvm::ArrayRef<std::string> rtlSources,
    llvm::ArrayRef<std::string> generationConstraints,
    llvm::StringRef targetLibrary, DesignCompilerHierarchy hierarchy) {
  return renderDriver(top, rtlSources, generationConstraints, targetLibrary,
                      hierarchy, nullptr);
}

llvm::Expected<std::string> renderDesignCompilerParentDriver(
    llvm::StringRef top, llvm::StringRef parentRtlSource,
    const DesignCompilerMappedChildren &children,
    llvm::ArrayRef<std::string> generationConstraints,
    llvm::StringRef targetLibrary) {
  return renderDriver(top, {parentRtlSource.str()}, generationConstraints,
                      targetLibrary,
                      DesignCompilerHierarchy::PreserveDefinitions, &children);
}

llvm::Expected<DesignCompilerGateNetlist>
parseDesignCompilerGateNetlist(llvm::StringRef contents, llvm::StringRef top) {
  if (!isPortableHdlIdentifier(top))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::ParserFailure,
        descriptor.implementationSemanticIdentity,
        "expected top is not a portable HDL identifier");
  if (contents.empty() || contents.contains('\0') || contents.contains('\r'))
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::ParserFailure,
        descriptor.implementationSemanticIdentity,
        "gate netlist is empty or violates the LF text contract");
  if (!containsTopModule(contents, top))
    return makeSynopsysAdapterError(SynopsysAdapterFailureKind::ParserFailure,
                                    descriptor.implementationSemanticIdentity,
                                    "gate netlist does not define exact top '" +
                                        top + "'");
  return DesignCompilerGateNetlist{contents.str()};
}

llvm::Expected<external_tool::ExternalToolInvocationBundleSpec>
makeDesignCompilerBundleSpec(
    const SynopsysBundleInputs &inputs, llvm::StringRef top,
    llvm::ArrayRef<std::string> rtlSources,
    llvm::ArrayRef<std::string> generationConstraints) {
  std::vector<std::string> requiredInputs(rtlSources.begin(), rtlSources.end());
  requiredInputs.insert(requiredInputs.end(), generationConstraints.begin(),
                        generationConstraints.end());
  if (llvm::Error error =
          validateSynopsysSemanticInputs(descriptor, inputs, requiredInputs))
    return std::move(error);
  const external_tool::ResolvedExternalFile *library =
      findExternal(inputs, "target_library");
  if (!library)
    return makeSynopsysAdapterError(
        SynopsysAdapterFailureKind::MissingProviderInput,
        descriptor.implementationSemanticIdentity, "target_library is absent");
  auto driver = renderDesignCompilerDriver(
      top, rtlSources, generationConstraints, library->absolutePath);
  if (!driver)
    return driver.takeError();
  std::vector<std::vector<std::string>> commands{
      {inputs.frozen.tool.executable, "-f", "drivers/design-compiler.tcl"}};
  std::vector<external_tool::MaterializedBundleFile> drivers{
      {"drivers/design-compiler.tcl", std::move(*driver), std::nullopt, false}};
  return makeSynopsysInvocationBundleSpec(
      descriptor, inputs, std::move(commands), std::move(drivers));
}

llvm::Expected<DesignCompilerGateNetlist> importDesignCompilerGateNetlist(
    const external_tool::PreparedExternalToolInvocation &prepared,
    const SynopsysBundleInputs &inputs, llvm::StringRef top) {
  auto imported = importSynopsysInvocation(descriptor, prepared, inputs);
  if (!imported)
    return imported.takeError();
  auto contents = readSynopsysDeclaredOutput(
      descriptor, *imported, descriptor.declaredOutputs.front());
  if (!contents)
    return contents.takeError();
  return parseDesignCompilerGateNetlist(*contents, top);
}

} // namespace loom::eda::synopsys

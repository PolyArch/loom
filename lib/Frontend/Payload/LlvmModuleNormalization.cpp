#include "Frontend/Payload/LlvmModuleNormalization.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitstreamReader.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace loom {
namespace {

llvm::Error rejected(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

llvm::Expected<std::string>
readModuleDataLayoutRecord(llvm::BitstreamCursor &stream) {
  if (llvm::Error error = stream.EnterSubBlock(llvm::bitc::MODULE_BLOCK_ID))
    return std::move(error);

  llvm::SmallVector<std::uint64_t, 64> record;
  std::string dataLayout;
  bool sawDataLayout = false;
  while (true) {
    llvm::Expected<llvm::BitstreamEntry> entry =
        stream.advanceSkippingSubblocks();
    if (!entry)
      return entry.takeError();
    switch (entry->Kind) {
    case llvm::BitstreamEntry::SubBlock:
    case llvm::BitstreamEntry::Error:
      return rejected("malformed module block");
    case llvm::BitstreamEntry::EndBlock:
      return dataLayout;
    case llvm::BitstreamEntry::Record:
      break;
    }

    llvm::Expected<unsigned> code = stream.readRecord(entry->ID, record);
    if (!code)
      return code.takeError();
    if (*code == llvm::bitc::MODULE_CODE_DATALAYOUT) {
      if (sawDataLayout)
        return rejected("duplicate data layout record");
      sawDataLayout = true;
      dataLayout.clear();
      dataLayout.reserve(record.size());
      for (std::uint64_t byte : record) {
        if (byte > 0xff)
          return rejected("invalid data layout record");
        dataLayout.push_back(static_cast<char>(byte));
      }
    }
    record.clear();
  }
}

llvm::Expected<std::string> readRawDataLayout(llvm::MemoryBufferRef buffer) {
  llvm::Expected<std::vector<llvm::BitcodeModule>> modules =
      llvm::getBitcodeModuleList(buffer);
  if (!modules)
    return modules.takeError();
  if (modules->size() != 1)
    return rejected("expected exactly one bitcode module");

  const llvm::StringRef moduleBuffer = modules->front().getBuffer();
  llvm::BitstreamCursor stream(llvm::ArrayRef<std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(moduleBuffer.data()),
      moduleBuffer.size()));
  while (true) {
    llvm::Expected<llvm::BitstreamEntry> entry = stream.advance();
    if (!entry)
      return entry.takeError();
    switch (entry->Kind) {
    case llvm::BitstreamEntry::EndBlock:
    case llvm::BitstreamEntry::Error:
      return rejected("malformed bitcode block");
    case llvm::BitstreamEntry::SubBlock:
      if (entry->ID == llvm::bitc::MODULE_BLOCK_ID)
        return readModuleDataLayoutRecord(stream);
      if (llvm::Error error = stream.SkipBlock())
        return std::move(error);
      break;
    case llvm::BitstreamEntry::Record: {
      llvm::Expected<unsigned> skipped = stream.skipRecord(entry->ID);
      if (!skipped)
        return skipped.takeError();
      break;
    }
    }
  }
}

using NamedValue = std::pair<llvm::Value *, std::string>;

void rebuildValueSymbolTable(llvm::ArrayRef<NamedValue> values) {
  for (const NamedValue &entry : values)
    entry.first->setName("");
  for (const NamedValue &entry : values)
    entry.first->setName(entry.second);
}

void canonicalizeValueSymbolTables(llvm::Module &module) {
  std::vector<NamedValue> moduleValues;
  moduleValues.reserve(module.global_size() + module.size() +
                       module.alias_size() + module.ifunc_size());
  for (llvm::GlobalVariable &global : module.globals())
    if (global.hasName())
      moduleValues.emplace_back(&global, global.getName().str());
  for (llvm::Function &function : module)
    if (function.hasName())
      moduleValues.emplace_back(&function, function.getName().str());
  for (llvm::GlobalAlias &alias : module.aliases())
    if (alias.hasName())
      moduleValues.emplace_back(&alias, alias.getName().str());
  for (llvm::GlobalIFunc &ifunc : module.ifuncs())
    if (ifunc.hasName())
      moduleValues.emplace_back(&ifunc, ifunc.getName().str());
  rebuildValueSymbolTable(moduleValues);

  for (llvm::Function &function : module) {
    std::vector<NamedValue> localValues;
    for (llvm::Argument &argument : function.args())
      if (argument.hasName())
        localValues.emplace_back(&argument, argument.getName().str());
    for (llvm::BasicBlock &block : function) {
      if (block.hasName())
        localValues.emplace_back(&block, block.getName().str());
      for (llvm::Instruction &instruction : block)
        if (instruction.hasName())
          localValues.emplace_back(&instruction, instruction.getName().str());
    }
    rebuildValueSymbolTable(localValues);
  }
}

} // namespace

llvm::Expected<std::unique_ptr<llvm::Module>>
parseCompleteLlvmModule(llvm::ArrayRef<std::uint8_t> bitcode,
                        llvm::LLVMContext &context) {
  const llvm::MemoryBufferRef buffer(
      llvm::StringRef(reinterpret_cast<const char *>(bitcode.data()),
                      bitcode.size()),
      "loom.relocatable_accelerator_payload");

  // BitcodeReader upgrades layout strings before its callback. Read the owner
  // record first, then override that upgrade with the exact validated spelling.
  llvm::Expected<std::string> rawDataLayout = readRawDataLayout(buffer);
  if (!rawDataLayout)
    return rejected("llvm_module_unparsable: " +
                    llvm::toString(rawDataLayout.takeError()));
  const std::string sourceDataLayout = std::move(*rawDataLayout);
  if (!sourceDataLayout.empty()) {
    llvm::Expected<llvm::DataLayout> parsedLayout =
        llvm::DataLayout::parse(sourceDataLayout);
    if (!parsedLayout)
      return rejected("data_layout_invalid: " +
                      llvm::toString(parsedLayout.takeError()));
  }

  llvm::ParserCallbacks callbacks(
      [&sourceDataLayout](llvm::StringRef,
                          llvm::StringRef) -> std::optional<std::string> {
        return sourceDataLayout;
      });
  llvm::Expected<std::unique_ptr<llvm::Module>> parsed =
      llvm::parseBitcodeFile(buffer, context, std::move(callbacks));
  if (!parsed)
    return rejected("llvm_module_unparsable: " +
                    llvm::toString(parsed.takeError()));
  // parseBitcodeFile already reads every function body. Materializing again
  // states the complete-module requirement explicitly and surfaces any deferred
  // failure before the module is treated as whole.
  if (llvm::Error error = (*parsed)->materializeAll())
    return rejected("llvm_module_materialization_failed: " +
                    llvm::toString(std::move(error)));

  std::string verifierReport;
  llvm::raw_string_ostream verifierStream(verifierReport);
  if (llvm::verifyModule(**parsed, &verifierStream))
    return rejected("llvm_module_invalid: " + verifierReport);
  return std::move(*parsed);
}

llvm::Expected<NormalizedLlvmModule>
normalizeLlvmModule(llvm::ArrayRef<std::uint8_t> sourceBitcode) {
  llvm::LLVMContext context;
  llvm::Expected<std::unique_ptr<llvm::Module>> parsed =
      parseCompleteLlvmModule(sourceBitcode, context);
  if (!parsed)
    return parsed.takeError();
  llvm::Module &module = **parsed;

  const std::string sourceTriple = module.getTargetTriple().str();
  if (sourceTriple.empty())
    return rejected("target_triple_absent: the module declares no target "
                    "triple");

  // Triple::normalize is the pinned canonical printer for target triples: it
  // accepts equivalent spellings and returns exactly one canonical form.
  const std::string canonicalTargetTriple =
      llvm::Triple::normalize(sourceTriple);
  const llvm::Triple triple(canonicalTargetTriple);
  if (triple.getArch() == llvm::Triple::UnknownArch)
    return rejected("target_triple_unsupported: the pinned LLVM provider does "
                    "not recognize target triple '" +
                    canonicalTargetTriple + "'");

  const std::string sourceDataLayout = module.getDataLayoutStr();
  if (sourceDataLayout.empty())
    return rejected("data_layout_absent: the module declares no data layout");

  module.setTargetTriple(triple);
  // LLVM's value symbol tables are hash maps. Rebuild them from structural IR
  // order so their bitcode record order does not depend on parser insertion
  // history; every exact name and all semantic IR ordering remain unchanged.
  canonicalizeValueSymbolTables(module);

  llvm::SmallVector<char, 0> written;
  llvm::raw_svector_ostream stream(written);
  llvm::WriteBitcodeToFile(module, stream, /*ShouldPreserveUseListOrder=*/false,
                           /*Index=*/nullptr, /*GenerateHash=*/false);

  NormalizedLlvmModule normalized;
  normalized.canonicalTargetTriple = canonicalTargetTriple;
  normalized.dataLayout = sourceDataLayout;
  normalized.bitcode.assign(written.begin(), written.end());
  normalized.bitcodeDigest = llvm::SHA256::hash(normalized.bitcode);
  return normalized;
}

} // namespace loom

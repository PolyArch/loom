#include "Frontend/Payload/RelocatableAcceleratorPayload.h"
#include "Common/Artifact.h"
#include "Common/ComponentViewDigest.h"
#include "Common/ResolvedConfig.h"
#include "Frontend/Payload/AbiCompatibilityKey.h"
#include "Frontend/Payload/FrontendConfigView.h"
#include "Frontend/Payload/LlvmModuleNormalization.h"
#include "RelocatablePayloadRootCodec.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace loom;

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << "\n";
  std::exit(1);
}

void require(const char *test, bool condition, const std::string &message) {
  if (!condition)
    fail(test, message);
}

template <typename T>
T takeExpected(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
std::string rejectionMessage(const char *test, llvm::Expected<T> value) {
  if (value)
    fail(test, "a value that must fail closed was accepted");
  return llvm::toString(value.takeError());
}

void requireRejection(const char *test, const std::string &message,
                      llvm::StringRef marker, const std::string &what) {
  if (!llvm::StringRef(message).contains(marker))
    fail(test, what + ": " + message);
}

constexpr llvm::StringRef canonicalTargetTriple = "x86_64-unknown-linux-gnu";
constexpr llvm::StringRef anchorDataLayout =
    "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-"
    "n8:16:32:64-S128";
constexpr llvm::StringRef riscv64Lp64eDataLayout =
    "e-m:e-p:64:64-i64:64-i128:128-n32:64-S64";

constexpr llvm::StringRef riscv64Lp64eAssembly = R"(
target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S64"
target triple = "riscv64-unknown-elf"

define i32 @sum(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
)";

constexpr llvm::StringRef anchorAssembly = R"(
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @loom_payload_anchor(i32 %value) {
entry:
  ret i32 %value
}
)";

std::vector<std::uint8_t> buildSourceBitcode(const char *test,
                                             llvm::StringRef assembly,
                                             bool generateHash = false) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(assembly, diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, "test module assembly did not parse: " + message);
  }

  llvm::SmallVector<char, 0> buffer;
  llvm::raw_svector_ostream stream(buffer);
  llvm::WriteBitcodeToFile(*module, stream,
                           /*ShouldPreserveUseListOrder=*/false,
                           /*Index=*/nullptr, generateHash);
  return std::vector<std::uint8_t>(buffer.begin(), buffer.end());
}

std::string printBitcodeAsAssembly(const char *test,
                                   llvm::ArrayRef<std::uint8_t> bitcode) {
  llvm::LLVMContext context;
  const llvm::MemoryBufferRef buffer(
      llvm::StringRef(reinterpret_cast<const char *>(bitcode.data()),
                      bitcode.size()),
      "loom.payload.anchor");
  llvm::Expected<std::unique_ptr<llvm::Module>> module =
      llvm::parseBitcodeFile(buffer, context);
  if (!module)
    fail(test, llvm::toString(module.takeError()));

  std::string printed;
  llvm::raw_string_ostream stream(printed);
  (*module)->print(stream, nullptr);
  return printed;
}

std::vector<std::uint8_t> toVector(llvm::ArrayRef<std::uint8_t> bytes) {
  return std::vector<std::uint8_t>(bytes.begin(), bytes.end());
}

std::vector<std::uint8_t> flipped(llvm::ArrayRef<std::uint8_t> bytes) {
  std::vector<std::uint8_t> changed = toVector(bytes);
  changed.back() ^= 0x01;
  return changed;
}

using RootField =
    llvm::ArrayRef<std::uint8_t> detail::RelocatablePayloadRoot::*;

std::vector<std::uint8_t> rootWithField(const char *test,
                                        llvm::ArrayRef<std::uint8_t> canonical,
                                        RootField field,
                                        llvm::ArrayRef<std::uint8_t> value) {
  detail::RelocatablePayloadRoot root =
      takeExpected(test, detail::decodeRelocatablePayloadRoot(canonical));
  root.*field = value;
  return detail::encodeRelocatablePayloadRoot(root);
}

ResolvedFrontendConfigView anchorView() {
  return projectResolvedFrontendConfigView(defaultResolvedConfig());
}

RelocatableAcceleratorPayload anchorPayload(const char *test) {
  return takeExpected(
      test, RelocatableAcceleratorPayload::create(
                buildSourceBitcode(test, anchorAssembly), anchorView()));
}

void normalizationPreservesRiscv64Lp64eDataLayoutSpelling() {
  const NormalizedLlvmModule normalized = takeExpected(
      __func__,
      normalizeLlvmModule(buildSourceBitcode(__func__, riscv64Lp64eAssembly)));
  require(__func__,
          normalized.canonicalTargetTriple == "riscv64-unknown-unknown-elf",
          "unexpected canonical target triple: " +
              normalized.canonicalTargetTriple);
  require(__func__, normalized.dataLayout == riscv64Lp64eDataLayout,
          "normalization replaced the module-owned data layout: " +
              normalized.dataLayout);
  require(__func__,
          llvm::StringRef(printBitcodeAsAssembly(__func__, normalized.bitcode))
              .contains("target datalayout = \"" +
                        riscv64Lp64eDataLayout.str() + "\""),
          "normalized bitcode did not preserve the exact data layout spelling");
}

void normalizationIsIdempotentForNamedRiscvValues() {
  const NormalizedLlvmModule first = takeExpected(
      __func__,
      normalizeLlvmModule(buildSourceBitcode(__func__, riscv64Lp64eAssembly)));
  const NormalizedLlvmModule second =
      takeExpected(__func__, normalizeLlvmModule(first.bitcode));
  require(__func__, second.bitcode == first.bitcode,
          "rewriting a normalized module changed its canonical bytes");
}

void payloadRoundTripsThroughCanonicalBytes() {
  const RelocatableAcceleratorPayload payload = anchorPayload(__func__);
  require(__func__,
          llvm::StringRef(payload.llvmProvider().repositoryIdentity) ==
              buildSelectedLlvmProvider().repositoryIdentity,
          "the payload provider is not the build-selected provider");
  require(__func__, payload.targetTriple() == canonicalTargetTriple,
          "unexpected payload target triple: " + payload.targetTriple().str());
  require(__func__, payload.dataLayout() == anchorDataLayout,
          "unexpected payload data layout: " + payload.dataLayout().str());

  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();
  const RelocatableAcceleratorPayload decoded =
      takeExpected(__func__, decodeRelocatableAcceleratorPayload(
                                 RelocatableAcceleratorPayload::artifactSchema,
                                 canonical.bytes()));
  require(__func__, decoded.identity() == payload.identity(),
          "the decoded payload identity changed");
  require(__func__, decoded.normalizedBitcode() == payload.normalizedBitcode(),
          "the decoded normalized module bytes changed");
  require(__func__,
          decoded.abiCompatibilityKey() == payload.abiCompatibilityKey(),
          "the decoded ABI compatibility key changed");
  require(__func__,
          decoded.canonicalSemanticBytes().bytes() == canonical.bytes(),
          "re-encoding the decoded payload changed the canonical bytes");
  require(__func__, anchorPayload(__func__).identity() == payload.identity(),
          "payload creation was not deterministic");

  AbiCompatibilityKeyInputs inputs;
  inputs.repositoryIdentity = payload.llvmProvider().repositoryIdentity;
  inputs.fullCommitIdentity = payload.llvmProvider().fullCommitIdentity;
  inputs.canonicalTargetTriple = payload.targetTriple();
  inputs.viewSchemaDescriptorBytes =
      payload.frontendConfigView().schemaDescriptorBytes();
  inputs.viewCanonicalBytes = payload.frontendConfigView().canonicalViewBytes();
  require(__func__,
          payload.abiCompatibilityKey() == computeAbiCompatibilityKey(inputs),
          "the stored ABI compatibility key is not the production key");
}

void normalizationFailsClosed() {
  constexpr llvm::StringRef missingDataLayout = R"(
target triple = "x86_64-unknown-linux-gnu"

define void @missing_layout() {
entry:
  ret void
}
)";
  requireRejection(
      __func__,
      rejectionMessage(__func__, normalizeLlvmModule(buildSourceBitcode(
                                     __func__, missingDataLayout))),
      "data_layout_absent", "unexpected absent data layout rejection");

  const std::vector<std::uint8_t> garbage(64, 0x5a);
  requireRejection(__func__,
                   rejectionMessage(__func__, normalizeLlvmModule(garbage)),
                   "llvm_module_unparsable", "unexpected garbage rejection");
}

void unsupportedSchemaAndMalformedEncodingAreRejected() {
  const RelocatableAcceleratorPayload payload = anchorPayload(__func__);
  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();

  constexpr ArtifactSchemaDescriptor newerMajor{
      "loom.relocatable_accelerator_payload", SchemaVersion{2, 0}};
  requireRejection(
      __func__,
      rejectionMessage(__func__, decodeRelocatableAcceleratorPayload(
                                     newerMajor, canonical.bytes())),
      "relocatable_payload_schema_unsupported", "unexpected schema rejection");

  const std::vector<std::uint8_t> truncated =
      toVector(canonical.bytes().drop_back(1));
  requireRejection(
      __func__,
      rejectionMessage(
          __func__,
          decodeRelocatableAcceleratorPayload(
              RelocatableAcceleratorPayload::artifactSchema, truncated)),
      "relocatable_payload_encoding", "unexpected truncation rejection");
}

void staleRootProjectionsAreRejected() {
  const RelocatableAcceleratorPayload payload = anchorPayload(__func__);
  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();
  using Root = detail::RelocatablePayloadRoot;

  struct Case {
    const char *what;
    RootField field;
    std::vector<std::uint8_t> value;
    llvm::StringRef marker;
  };
  const Case cases[] = {
      {"frontend view digest", &Root::viewDigest,
       flipped(payload.frontendConfigView().digest().bytes()),
       "component_view_digest_mismatch"},
      {"normalized bitcode digest", &Root::normalizedBitcodeDigest,
       flipped(payload.normalizedBitcodeDigest()),
       "normalized_bitcode_digest_mismatch"},
      {"ABI compatibility key", &Root::abiCompatibilityKey,
       flipped(payload.abiCompatibilityKey().bytes()),
       "abi_compatibility_key_mismatch"},
  };

  for (const Case &testCase : cases)
    requireRejection(
        __func__,
        rejectionMessage(__func__,
                         decodeRelocatableAcceleratorPayload(
                             RelocatableAcceleratorPayload::artifactSchema,
                             rootWithField(__func__, canonical.bytes(),
                                           testCase.field, testCase.value))),
        testCase.marker,
        std::string("unexpected rejection for a stale ") + testCase.what);
}

void noncanonicalModuleBytesAreRejected() {
  const RelocatableAcceleratorPayload payload = anchorPayload(__func__);
  const CanonicalSemanticBytes canonical = payload.canonicalSemanticBytes();
  const std::vector<std::uint8_t> noncanonical =
      buildSourceBitcode(__func__, anchorAssembly, /*generateHash=*/true);
  require(__func__, noncanonical != toVector(payload.normalizedBitcode()),
          "the noncanonical fixture matches the production writer output");

  requireRejection(
      __func__,
      rejectionMessage(
          __func__,
          decodeRelocatableAcceleratorPayload(
              RelocatableAcceleratorPayload::artifactSchema,
              rootWithField(__func__, canonical.bytes(),
                            &detail::RelocatablePayloadRoot::normalizedBitcode,
                            noncanonical))),
      "normalized_bitcode_not_canonical",
      "unexpected noncanonical module rejection");
}

} // namespace

int main() {
  normalizationPreservesRiscv64Lp64eDataLayoutSpelling();
  normalizationIsIdempotentForNamedRiscvValues();
  payloadRoundTripsThroughCanonicalBytes();
  normalizationFailsClosed();
  unsupportedSchemaAndMalformedEncodingAreRejected();
  staleRootProjectionsAreRejected();
  noncanonicalModuleBytesAreRejected();
  return 0;
}

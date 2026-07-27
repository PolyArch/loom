#include "Frontend/Compilation/StaticGlobalMemory.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

namespace {

[[noreturn]] void fail(const char *test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename T> T take(const char *test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

std::unique_ptr<llvm::Module> parseModule(const char *test,
                                          llvm::LLVMContext &context) {
  constexpr llvm::StringLiteral source = R"llvm(
target datalayout = "e-m:e-p:64:64-i64:64-n32:64-S128"
target triple = "riscv64-unknown-unknown-elf"

@table = private constant [4 x i32]
    [i32 0, i32 287454020, i32 -1, i32 1432778632], align 16
@counter = internal global i32 7, align 4
@external = external global i8
)llvm";
  llvm::SMDiagnostic diagnostic;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(source, "<static-memory>");
  auto module = llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
  if (!module) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    diagnostic.print(test, stream);
    fail(test, stream.str());
  }
  return module;
}

void projectsDefinedAndExternalGlobals() {
  const char *test = "projectsDefinedAndExternalGlobals";
  llvm::LLVMContext context;
  auto module = parseModule(test, context);
  loom::frontend::StaticGlobalMemoryCatalog catalog =
      take(test, loom::frontend::projectStaticGlobalMemory(*module));

  if (catalog.dataLayout != module->getDataLayoutStr())
    fail(test, "catalog did not retain the exact module DataLayout");
  if (catalog.globals.size() != 3)
    fail(test, "catalog is not total over addressable globals");

  const auto *table = catalog.lookup("table");
  if (!table ||
      table->provision != loom::frontend::StaticGlobalProvision::Image ||
      table->permissions != loom::frontend::StaticMemoryPermissions::ReadOnly ||
      table->sizeBytes != 16 || table->alignmentBytes != 16)
    fail(test, "constant table projection lost its storage contract");
  const std::uint8_t expected[] = {0x00, 0x00, 0x00, 0x00, 0x44, 0x33,
                                   0x22, 0x11, 0xff, 0xff, 0xff, 0xff,
                                   0x88, 0x77, 0x66, 0x55};
  if (!std::equal(table->bytes.begin(), table->bytes.end(), expected,
                  expected + sizeof(expected)))
    fail(test, "constant table bytes do not follow the exact DataLayout");

  const auto *counter = catalog.lookup("counter");
  if (!counter ||
      counter->provision != loom::frontend::StaticGlobalProvision::Image ||
      counter->permissions !=
          loom::frontend::StaticMemoryPermissions::ReadWrite ||
      counter->bytes != std::vector<std::uint8_t>({7, 0, 0, 0}))
    fail(test, "writable initializer was not projected exactly");

  const auto *external = catalog.lookup("external");
  if (!external ||
      external->provision !=
          loom::frontend::StaticGlobalProvision::ExternalRuntime ||
      !external->bytes.empty())
    fail(test, "external storage was mistaken for a local static image");
}

} // namespace

int main() {
  projectsDefinedAndExternalGlobals();
  llvm::outs() << "static global memory projection anchor passed\n";
  return EXIT_SUCCESS;
}

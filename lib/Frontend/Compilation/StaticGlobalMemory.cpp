#include "Frontend/Compilation/StaticGlobalMemory.h"

#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <system_error>

namespace loom::frontend {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::createStringError(
      std::make_error_code(std::errc::invalid_argument),
      llvm::Twine("static_global_memory_invalid: ") + message);
}

llvm::Expected<std::uint64_t> fixedSize(llvm::TypeSize size,
                                        llvm::StringRef context) {
  if (size.isScalable())
    return invalid(llvm::Twine(context) + " has scalable storage size");
  return size.getFixedValue();
}

bool writeScalarBits(const llvm::APInt &value, std::uint64_t byteCount,
                     bool littleEndian, std::uint64_t offset,
                     llvm::MutableArrayRef<std::uint8_t> output) {
  if (byteCount > std::numeric_limits<unsigned>::max() / 8 ||
      offset > output.size() || byteCount > output.size() - offset)
    return false;
  llvm::APInt bits = value.zextOrTrunc(static_cast<unsigned>(byteCount * 8));
  for (std::uint64_t byte = 0; byte < byteCount; ++byte) {
    const std::uint64_t sourceByte = littleEndian ? byte : byteCount - byte - 1;
    output[offset + byte] = static_cast<std::uint8_t>(
        bits.extractBitsAsZExtValue(8, sourceByte * 8));
  }
  return true;
}

bool writeConstant(const llvm::Constant &constant, const llvm::DataLayout &dl,
                   std::uint64_t offset,
                   llvm::MutableArrayRef<std::uint8_t> output);

bool writeSequential(const llvm::Constant &constant, const llvm::DataLayout &dl,
                     std::uint64_t offset,
                     llvm::MutableArrayRef<std::uint8_t> output) {
  llvm::Type *type = constant.getType();
  llvm::Type *elementType = nullptr;
  std::uint64_t elementCount = 0;
  std::uint64_t stride = 0;

  if (auto *array = llvm::dyn_cast<llvm::ArrayType>(type)) {
    elementType = array->getElementType();
    elementCount = array->getNumElements();
    llvm::TypeSize allocSize = dl.getTypeAllocSize(elementType);
    if (allocSize.isScalable())
      return false;
    stride = allocSize.getFixedValue();
  } else if (auto *vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
    elementType = vector->getElementType();
    elementCount = vector->getNumElements();
    llvm::TypeSize storeSize = dl.getTypeStoreSize(elementType);
    llvm::TypeSize vectorStoreSize = dl.getTypeStoreSize(type);
    if (storeSize.isScalable() || vectorStoreSize.isScalable())
      return false;
    stride = storeSize.getFixedValue();
    if (elementCount != 0 &&
        (stride > std::numeric_limits<std::uint64_t>::max() / elementCount ||
         stride * elementCount != vectorStoreSize.getFixedValue()))
      return false;
  } else {
    return false;
  }

  for (std::uint64_t index = 0; index < elementCount; ++index) {
    if (offset > output.size() || stride > output.size() - offset ||
        (stride != 0 && index > (output.size() - offset) / stride))
      return false;
    llvm::Constant *element = constant.getAggregateElement(index);
    if (!element ||
        !writeConstant(*element, dl, offset + index * stride, output))
      return false;
  }
  return true;
}

bool writeConstant(const llvm::Constant &constant, const llvm::DataLayout &dl,
                   std::uint64_t offset,
                   llvm::MutableArrayRef<std::uint8_t> output) {
  llvm::TypeSize allocSize = dl.getTypeAllocSize(constant.getType());
  if (allocSize.isScalable() || offset > output.size() ||
      allocSize.getFixedValue() > output.size() - offset)
    return false;

  if (llvm::isa<llvm::ConstantAggregateZero>(constant))
    return true;
  if (llvm::isa<llvm::ConstantPointerNull>(constant))
    return false;
  if (llvm::isa<llvm::UndefValue, llvm::PoisonValue>(constant))
    return false;

  if (const auto *integer = llvm::dyn_cast<llvm::ConstantInt>(&constant)) {
    llvm::TypeSize storeSize = dl.getTypeStoreSize(integer->getType());
    return !storeSize.isScalable() &&
           writeScalarBits(integer->getValue(), storeSize.getFixedValue(),
                           dl.isLittleEndian(), offset, output);
  }
  if (const auto *floating = llvm::dyn_cast<llvm::ConstantFP>(&constant)) {
    llvm::TypeSize storeSize = dl.getTypeStoreSize(floating->getType());
    return !storeSize.isScalable() &&
           writeScalarBits(floating->getValueAPF().bitcastToAPInt(),
                           storeSize.getFixedValue(), dl.isLittleEndian(),
                           offset, output);
  }

  if (llvm::isa<llvm::ConstantDataSequential, llvm::ConstantArray,
                llvm::ConstantVector>(constant))
    return writeSequential(constant, dl, offset, output);

  if (const auto *structure = llvm::dyn_cast<llvm::ConstantStruct>(&constant)) {
    const llvm::StructLayout *layout =
        dl.getStructLayout(llvm::cast<llvm::StructType>(structure->getType()));
    for (unsigned index = 0; index < structure->getNumOperands(); ++index) {
      auto *field = llvm::cast<llvm::Constant>(structure->getOperand(index));
      if (!writeConstant(*field, dl, offset + layout->getElementOffset(index),
                         output))
        return false;
    }
    return true;
  }

  if (const auto *expression = llvm::dyn_cast<llvm::ConstantExpr>(&constant)) {
    llvm::Constant *folded = llvm::ConstantFoldConstant(expression, dl);
    return folded && folded != expression &&
           writeConstant(*folded, dl, offset, output);
  }
  return false;
}

} // namespace

const StaticGlobalMemory *
StaticGlobalMemoryCatalog::lookup(llvm::StringRef symbol) const {
  auto found =
      std::lower_bound(globals.begin(), globals.end(), symbol,
                       [](const StaticGlobalMemory &global,
                          llvm::StringRef key) { return global.symbol < key; });
  return found != globals.end() && found->symbol == symbol ? &*found : nullptr;
}

llvm::Expected<StaticGlobalMemoryCatalog>
projectStaticGlobalMemory(const llvm::Module &module) {
  if (module.getDataLayoutStr().empty() && !module.global_empty())
    return invalid("linked LLVM module has no DataLayout");

  StaticGlobalMemoryCatalog catalog;
  catalog.dataLayout = module.getDataLayoutStr();
  if (module.global_empty())
    return catalog;
  const llvm::DataLayout &dl = module.getDataLayout();
  catalog.globals.reserve(module.global_size());

  for (const llvm::GlobalVariable &global : module.globals()) {
    if (!global.hasName())
      return invalid("addressable LLVM global has no symbol");
    auto size =
        fixedSize(dl.getTypeAllocSize(global.getValueType()), "LLVM global");
    if (!size)
      return size.takeError();
    if (*size > std::numeric_limits<std::size_t>::max())
      return invalid("LLVM global exceeds host addressable size");

    StaticGlobalMemory projected;
    projected.symbol = global.getName().str();
    projected.sizeBytes = *size;
    projected.alignmentBytes =
        global.getAlign() ? global.getAlign()->value()
                          : dl.getABITypeAlign(global.getValueType()).value();
    projected.permissions = global.isConstant()
                                ? StaticMemoryPermissions::ReadOnly
                                : StaticMemoryPermissions::ReadWrite;

    const bool mayHaveImage =
        global.hasInitializer() && !global.isThreadLocal() &&
        !global.isExternallyInitialized() && global.getAddressSpace() == 0;
    if (mayHaveImage && projected.sizeBytes != 0) {
      projected.bytes.assign(projected.sizeBytes, 0);
      if (writeConstant(*global.getInitializer(), dl, 0, projected.bytes)) {
        projected.provision = StaticGlobalProvision::Image;
      } else {
        projected.bytes.clear();
      }
    }
    catalog.globals.push_back(std::move(projected));
  }

  llvm::sort(catalog.globals,
             [](const StaticGlobalMemory &lhs, const StaticGlobalMemory &rhs) {
               return lhs.symbol < rhs.symbol;
             });
  for (std::size_t index = 1; index < catalog.globals.size(); ++index)
    if (catalog.globals[index - 1].symbol == catalog.globals[index].symbol)
      return invalid("duplicate LLVM global symbol");
  return catalog;
}

} // namespace loom::frontend

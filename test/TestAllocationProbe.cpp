#include "TestAllocationProbe.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <cstdlib>
#include <new>

namespace {

thread_local bool probeEnabled = false;
thread_local std::size_t allocationCount = 0;

void recordAllocation() noexcept {
  if (probeEnabled) {
    ++allocationCount;
  }
}

} // namespace

void loom::test::startAllocationProbe() {
  allocationCount = 0;
  probeEnabled = true;
}

std::size_t loom::test::stopAllocationProbe() {
  probeEnabled = false;
  return allocationCount;
}

bool loom::test::allocationProbeIsCalibrated() {
  startAllocationProbe();
  void *storage = std::malloc(8);
  if (!storage) {
    stopAllocationProbe();
    return false;
  }
  void *grown = std::realloc(storage, 32);
  if (!grown) {
    std::free(storage);
    stopAllocationProbe();
    return false;
  }
  storage = grown;
  llvm::SmallVector<std::uint64_t, 0> values;
  values.push_back(1);
  static_cast<volatile unsigned char *>(storage)[0] =
      static_cast<unsigned char>(values.front());
  std::free(storage);
  const std::size_t observed = stopAllocationProbe();
#if defined(__linux__)
  return observed >= 3;
#else
  return observed >= 1;
#endif
}

#if defined(__linux__)
extern "C" {

void *__real_malloc(std::size_t size);
void *__real_calloc(std::size_t count, std::size_t size);
void *__real_realloc(void *storage, std::size_t size);
void *__real_aligned_alloc(std::size_t alignment, std::size_t size);
int __real_posix_memalign(void **storage, std::size_t alignment,
                          std::size_t size);

void *__wrap_malloc(std::size_t size) {
  recordAllocation();
  return __real_malloc(size);
}

void *__wrap_calloc(std::size_t count, std::size_t size) {
  recordAllocation();
  return __real_calloc(count, size);
}

void *__wrap_realloc(void *storage, std::size_t size) {
  recordAllocation();
  return __real_realloc(storage, size);
}

void *__wrap_aligned_alloc(std::size_t alignment, std::size_t size) {
  recordAllocation();
  return __real_aligned_alloc(alignment, size);
}

int __wrap_posix_memalign(void **storage, std::size_t alignment,
                          std::size_t size) {
  recordAllocation();
  return __real_posix_memalign(storage, alignment, size);
}

} // extern "C"
#endif

void *operator new(std::size_t size) {
  recordAllocation();
  if (void *storage = std::malloc(size == 0 ? 1 : size))
    return storage;
  throw std::bad_alloc();
}

void *operator new[](std::size_t size) { return ::operator new(size); }

void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

void *operator new[](std::size_t size, const std::nothrow_t &tag) noexcept {
  return ::operator new(size, tag);
}

void operator delete(void *storage) noexcept { std::free(storage); }
void operator delete[](void *storage) noexcept { ::operator delete(storage); }
void operator delete(void *storage, std::size_t) noexcept {
  ::operator delete(storage);
}
void operator delete[](void *storage, std::size_t) noexcept {
  ::operator delete(storage);
}
void operator delete(void *storage, const std::nothrow_t &) noexcept {
  ::operator delete(storage);
}
void operator delete[](void *storage, const std::nothrow_t &) noexcept {
  ::operator delete(storage);
}

void *operator new(std::size_t size, std::align_val_t alignment) {
  recordAllocation();
  const std::size_t nonzeroSize = size == 0 ? 1 : size;
  if (static_cast<std::size_t>(alignment) <= alignof(std::max_align_t)) {
    if (void *storage = std::malloc(nonzeroSize))
      return storage;
    throw std::bad_alloc();
  }
  void *storage = nullptr;
  if (posix_memalign(&storage, static_cast<std::size_t>(alignment),
                     nonzeroSize) == 0)
    return storage;
  throw std::bad_alloc();
}

void *operator new[](std::size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}

void *operator new(std::size_t size, std::align_val_t alignment,
                   const std::nothrow_t &) noexcept {
  try {
    return ::operator new(size, alignment);
  } catch (...) {
    return nullptr;
  }
}

void *operator new[](std::size_t size, std::align_val_t alignment,
                     const std::nothrow_t &tag) noexcept {
  return ::operator new(size, alignment, tag);
}

void operator delete(void *storage, std::align_val_t) noexcept {
  std::free(storage);
}
void operator delete[](void *storage, std::align_val_t alignment) noexcept {
  ::operator delete(storage, alignment);
}
void operator delete(void *storage, std::size_t,
                     std::align_val_t alignment) noexcept {
  ::operator delete(storage, alignment);
}
void operator delete[](void *storage, std::size_t,
                       std::align_val_t alignment) noexcept {
  ::operator delete(storage, alignment);
}
void operator delete(void *storage, std::align_val_t alignment,
                     const std::nothrow_t &) noexcept {
  ::operator delete(storage, alignment);
}
void operator delete[](void *storage, std::align_val_t alignment,
                       const std::nothrow_t &) noexcept {
  ::operator delete(storage, alignment);
}

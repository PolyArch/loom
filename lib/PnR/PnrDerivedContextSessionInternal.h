#ifndef LOOM_LIB_PNR_PNRDERIVEDCONTEXTSESSIONINTERNAL_H
#define LOOM_LIB_PNR_PNRDERIVEDCONTEXTSESSIONINTERNAL_H

#include "PnR/PnrDerivedContext.h"

#include "llvm/Support/Error.h"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

namespace loom::pnr::detail {

inline constexpr std::uint64_t pnrDerivedContextCacheAlgorithmVersion = 1;

enum class PnrDerivedContextDomain : std::uint8_t {
  FabricStatic,
  FabricTiming,
  SystemStatic,
  SystemActive,
};

struct PnrDerivedContextSessionKey final {
  PnrDerivedContextDomain domain = PnrDerivedContextDomain::FabricStatic;
  std::array<std::uint8_t, 32> digest{};
  std::uint64_t algorithmVersion = pnrDerivedContextCacheAlgorithmVersion;
};

struct PnrDerivedContextSessionEntry final {
  std::shared_ptr<const void> context;
  std::uint64_t constructionNanoseconds = 0;
  std::uint64_t retainedBytes = 0;
  std::uint64_t deterministicWork = 0;
};

class PnrDerivedContextSessionState final {
public:
  struct Lookup final {
    std::shared_ptr<const PnrDerivedContextSessionEntry> entry;
    bool reservedConstruction = false;
  };

  explicit PnrDerivedContextSessionState(std::size_t entryLimit);

  llvm::Expected<Lookup> lookupOrReserve(PnrDerivedContextDomain domain,
                                         llvm::ArrayRef<std::uint8_t> digest);
  std::shared_ptr<const PnrDerivedContextSessionEntry>
  complete(PnrDerivedContextDomain domain, llvm::ArrayRef<std::uint8_t> digest,
           std::shared_ptr<const void> context,
           std::uint64_t constructionNanoseconds,
           std::uint64_t retainedBytes, std::uint64_t deterministicWork);
  void abandon(PnrDerivedContextDomain domain,
               llvm::ArrayRef<std::uint8_t> digest,
               std::uint64_t constructionNanoseconds);
  void recordRevalidation();
  PnrDerivedContextSessionStatistics statistics() const;

private:
  static void add(std::uint64_t &destination, std::uint64_t value);

  struct KeyLess final {
    bool operator()(const PnrDerivedContextSessionKey &lhs,
                    const PnrDerivedContextSessionKey &rhs) const;
  };

  const std::size_t entryLimit_ = 0;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::map<PnrDerivedContextSessionKey,
           std::shared_ptr<const PnrDerivedContextSessionEntry>, KeyLess>
      entries_;
  std::map<PnrDerivedContextSessionKey, std::thread::id, KeyLess> constructing_;
  PnrDerivedContextSessionStatistics statistics_;
};

std::shared_ptr<PnrDerivedContextSessionState>
currentPnrDerivedContextSession();

template <typename T>
std::shared_ptr<const T> contextFromEntry(
    const std::shared_ptr<const PnrDerivedContextSessionEntry> &entry) {
  return std::shared_ptr<const T>(entry->context,
                                  static_cast<const T *>(entry->context.get()));
}

} // namespace loom::pnr::detail

#endif // LOOM_LIB_PNR_PNRDERIVEDCONTEXTSESSIONINTERNAL_H

#include "InitializerChoiceOrder.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

using namespace loom::pnr;
using namespace loom::pnr::detail;

namespace {

PnrIndex selectRemainingChoice(llvm::MutableArrayRef<PnrIndex> fenwick,
                               std::uint64_t selectedRank) {
  const PnrIndex count = static_cast<PnrIndex>(fenwick.size());
  PnrIndex step = 1;
  while (step <= count / 2)
    step *= 2;

  PnrIndex prefix = 0;
  for (; step != 0; step /= 2) {
    const PnrIndex next = prefix + step;
    if (next > count)
      continue;
    const PnrIndex subtreeCount = fenwick[next - 1];
    if (subtreeCount <= selectedRank) {
      selectedRank -= subtreeCount;
      prefix = next;
    }
  }
  assert(prefix < count && "selected rank exceeds initializer domain");

  for (PnrIndex index = prefix + 1; index <= count;
       index += index & (0 - index)) {
    assert(fenwick[index - 1] != 0 &&
           "initializer choice was selected more than once");
    --fenwick[index - 1];
  }
  return prefix;
}

} // namespace

llvm::Error loom::pnr::detail::buildInitializerChoiceOrder(
    llvm::ArrayRef<PnrIndex> canonicalChoices,
    DeterministicPnrRandomStream *diversificationStream,
    llvm::MutableArrayRef<PnrIndex> choiceOrder,
    llvm::MutableArrayRef<PnrIndex> fenwickScratch) {
  assert(choiceOrder.size() >= canonicalChoices.size());
  assert(fenwickScratch.size() >= canonicalChoices.size());
  choiceOrder = choiceOrder.take_front(canonicalChoices.size());
  fenwickScratch = fenwickScratch.take_front(canonicalChoices.size());

  if (!diversificationStream) {
    std::copy(canonicalChoices.begin(), canonicalChoices.end(),
              choiceOrder.begin());
    return llvm::Error::success();
  }

  const PnrIndex count = static_cast<PnrIndex>(canonicalChoices.size());
  for (PnrIndex local = 1; local <= count; ++local)
    fenwickScratch[local - 1] = local & (0 - local);
  for (PnrIndex position = 0; position < count; ++position) {
    auto rank = diversificationStream->nextBounded(count - position);
    if (!rank)
      return rank.takeError();
    const PnrIndex selected = selectRemainingChoice(fenwickScratch, *rank);
    choiceOrder[position] = canonicalChoices[selected];
  }
  return llvm::Error::success();
}

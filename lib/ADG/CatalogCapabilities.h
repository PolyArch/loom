#ifndef LOOM_ADG_CATALOGCAPABILITIES_H
#define LOOM_ADG_CATALOGCAPABILITIES_H

#include "Fabric/IR/ImplementationFamily.h"

namespace loom::adg::detail {

inline ::fabric::PointerFormatRelation catalogPointerFormats() {
  return ::fabric::PointerFormatRelation::get(
      {{0, 32, 32, ::loom::PointerLayoutKind::StableIntegral},
       {0, 64, 64, ::loom::PointerLayoutKind::StableIntegral}});
}

} // namespace loom::adg::detail

#endif // LOOM_ADG_CATALOGCAPABILITIES_H

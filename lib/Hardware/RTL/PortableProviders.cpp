#include "Hardware/RTL/PortableProviders.h"

#include "Hardware/RTL/Providers/FixedVectorIntegerAddSub.h"
#include "Hardware/RTL/Providers/FixedVectorIntegerCompareMinMax.h"
#include "Hardware/RTL/Providers/FixedVectorIntegerMultiply.h"
#include "Hardware/RTL/Providers/FixedVectorPackUnpack.h"
#include "Hardware/RTL/Providers/FixedVectorShuffle.h"
#include "Hardware/RTL/Providers/FixedVectorSliceAlignMerge.h"
#include "Hardware/RTL/Providers/FixedVectorValueSelect.h"
#include "Hardware/RTL/Providers/FloatSign.h"
#include "Hardware/RTL/Providers/IntegerCountZeros.h"
#include "Hardware/RTL/Providers/IntegerLogic.h"
#include "Hardware/RTL/Providers/IntegerSaturatingAddSub.h"
#include "Hardware/RTL/Providers/IntegerShift.h"
#include "Hardware/RTL/Providers/LoopCarry.h"
#include "Hardware/RTL/Providers/LoopGate.h"
#include "Hardware/RTL/Providers/LoopInvariant.h"
#include "Hardware/RTL/Providers/ScalarBitReinterpret.h"
#include "Hardware/RTL/Providers/ScalarFloatFma.h"
#include "Hardware/RTL/Providers/ScalarIntegerAddSub.h"
#include "Hardware/RTL/Providers/ScalarIntegerCast.h"
#include "Hardware/RTL/Providers/ScalarIntegerCompareMinMax.h"
#include "Hardware/RTL/Providers/ScalarIntegerMultiply.h"
#include "Hardware/RTL/Providers/ScalarSignedIntegerDivRem.h"
#include "Hardware/RTL/Providers/ScalarUnsignedIntegerDivRem.h"
#include "Hardware/RTL/Providers/ScalarValueSelect.h"

#include <utility>

namespace loom::hardware::rtl {

llvm::Error
registerPortableOperationProviders(FabricOperationProviderRegistry &registry) {
  FabricOperationProviderRegistry candidate = registry;
  if (llvm::Error error =
          registerPortableScalarIntegerAddSubProvider(candidate))
    return error;
  if (llvm::Error error = registerPortableIntegerLogicProviders(candidate))
    return error;
  if (llvm::Error error = registerPortableIntegerShiftProviders(candidate))
    return error;
  if (llvm::Error error =
          registerPortableScalarIntegerCompareMinMaxProvider(candidate))
    return error;
  if (llvm::Error error = registerPortableScalarValueSelectProvider(candidate))
    return error;
  if (llvm::Error error = registerPortableScalarIntegerCastProvider(candidate))
    return error;
  if (llvm::Error error =
          registerPortableScalarBitReinterpretProvider(candidate))
    return error;
  if (llvm::Error error = registerPortableFloatSignProviders(candidate))
    return error;
  if (llvm::Error error =
          registerPortableScalarIntegerMultiplyProvider(candidate))
    return error;
  if (llvm::Error error = registerPortableScalarFloatFmaProvider(candidate))
    return error;
  if (llvm::Error error = registerPortableLoopCarryProvider(candidate))
    return error;
  if (llvm::Error error = registerPortableLoopInvariantProvider(candidate))
    return error;
  if (llvm::Error error = registerPortableLoopGateProvider(candidate))
    return error;
  if (llvm::Error error =
          registerPortableFixedVectorIntegerAddSubProvider(candidate))
    return error;
  if (llvm::Error error =
          registerPortableFixedVectorIntegerCompareMinMaxProvider(candidate))
    return error;
  if (llvm::Error error =
          registerPortableFixedVectorValueSelectProvider(candidate))
    return error;
  if (llvm::Error error =
          registerPortableFixedVectorIntegerMultiplyProvider(candidate))
    return error;
  if (llvm::Error error = registerPortableFixedVectorPackProvider(candidate))
    return error;
  if (llvm::Error error = registerPortableFixedVectorUnpackProvider(candidate))
    return error;
  if (llvm::Error error =
          registerPortableScalarSignedIntegerDivRemProvider(candidate))
    return error;
  if (llvm::Error error =
          registerPortableScalarUnsignedIntegerDivRemProvider(candidate))
    return error;
  if (llvm::Error error =
          registerPortableIntegerSaturatingAddSubProviders(candidate))
    return error;
  if (llvm::Error error = registerPortableIntegerCountZerosProviders(candidate))
    return error;
  if (llvm::Error error =
          registerPortableFixedVectorSliceAlignMergeProvider(candidate))
    return error;
  if (llvm::Error error = registerPortableFixedVectorShuffleProvider(candidate))
    return error;
  registry = std::move(candidate);
  return llvm::Error::success();
}

} // namespace loom::hardware::rtl

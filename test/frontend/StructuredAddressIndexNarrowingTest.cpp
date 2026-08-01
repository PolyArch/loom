#include "StructuredAddressIndexNarrowing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace {

[[noreturn]] void fail(const llvm::Twine &message) {
  llvm::errs() << "structuredAddressIndexNarrowing: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void normalizesAsymmetricPointerInduction() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n32:64-S128"
} {
  llvm.func @kernel(%base: !llvm.ptr) {
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c4_i64 = arith.constant 4 : i64
    %result:3 = scf.while (%remaining = %c4_i32, %cursor = %base)
        : (i32, !llvm.ptr) -> (i32, !llvm.ptr, i32) {
      %value = llvm.load %cursor : !llvm.ptr -> i32
      %next_cursor = llvm.getelementptr inbounds %cursor[%c4_i64]
          : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %next_remaining = arith.addi %remaining, %c-1_i32 : i32
      %more = arith.cmpi ne, %next_remaining, %c0_i32 : i32
      scf.condition(%more) %next_remaining, %next_cursor, %value
          : i32, !llvm.ptr, i32
    } do {
    ^bb0(%remaining: i32, %cursor: !llvm.ptr, %last: i32):
      scf.yield %remaining, %cursor : i32, !llvm.ptr
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the asymmetric pointer-induction fixture");
  auto function = module->lookupSymbol<mlir::LLVM::LLVMFuncOp>("kernel");
  if (!function)
    fail("fixture omitted kernel");

  auto normalized = loom::frontend::detail::materializeAddressIndexContract(
      *module, function.getOperation(), 64,
      [](mlir::Block *, mlir::Block *) { return llvm::Error::success(); });
  if (!normalized)
    fail(llvm::toString(normalized.takeError()));

  mlir::scf::WhileOp loop;
  function.walk([&](mlir::scf::WhileOp candidate) { loop = candidate; });
  if (!loop || loop.getInits().size() != 2 || loop.getNumResults() != 3)
    fail("pointer induction did not preserve the asymmetric loop contract");
  auto initOffset =
      llvm::dyn_cast<mlir::IntegerType>(loop.getInits()[1].getType());
  auto resultOffset =
      llvm::dyn_cast<mlir::IntegerType>(loop.getResult(1).getType());
  if (!initOffset || initOffset.getWidth() != 64 || !resultOffset ||
      resultOffset.getWidth() != 64)
    fail("pointer induction did not materialize the selected offset width");
}

void normalizesInvariantDynamicByteStride() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n32:64-S128"
} {
  llvm.func @complex_stride(%input: !llvm.ptr, %output: !llvm.ptr,
                            %rows: i16) {
    %c0_i16 = arith.constant 0 : i16
    %c-1_i16 = arith.constant -1 : i16
    %c3_i64 = arith.constant 3 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %row_count = arith.extui %rows : i16 to i64
    %output_stride = arith.shli %row_count, %c3_i64
        overflow<nsw, nuw> : i64
    %result:3 = scf.while (%input_cursor = %input, %remaining = %rows,
                           %output_cursor = %output)
        : (!llvm.ptr, i16, !llvm.ptr) -> (!llvm.ptr, i16, !llvm.ptr) {
      %imaginary_input = llvm.getelementptr inbounds %input_cursor[%c4_i64]
          : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %real = llvm.load %input_cursor : !llvm.ptr -> f32
      %imaginary = llvm.load %imaginary_input : !llvm.ptr -> f32
      llvm.store %real, %output_cursor : f32, !llvm.ptr
      %imaginary_output = llvm.getelementptr inbounds %output_cursor[%c4_i64]
          : (!llvm.ptr, i64) -> !llvm.ptr, i8
      llvm.store %imaginary, %imaginary_output : f32, !llvm.ptr
      %next_input = llvm.getelementptr inbounds %input_cursor[%c8_i64]
          : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %next_output = llvm.getelementptr inbounds %output_cursor[%output_stride]
          : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %next_remaining = arith.addi %remaining, %c-1_i16 : i16
      %more = arith.cmpi ne, %next_remaining, %c0_i16 : i16
      scf.condition(%more) %next_input, %next_remaining, %next_output
          : !llvm.ptr, i16, !llvm.ptr
    } do {
    ^bb0(%input_cursor: !llvm.ptr, %remaining: i16,
         %output_cursor: !llvm.ptr):
      scf.yield %input_cursor, %remaining, %output_cursor
          : !llvm.ptr, i16, !llvm.ptr
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the dynamic byte-stride fixture");
  auto function =
      module->lookupSymbol<mlir::LLVM::LLVMFuncOp>("complex_stride");
  if (!function)
    fail("dynamic byte-stride fixture omitted complex_stride");

  auto normalized = loom::frontend::detail::materializeAddressIndexContract(
      *module, function.getOperation(), 64,
      [](mlir::Block *, mlir::Block *) { return llvm::Error::success(); });
  if (!normalized)
    fail(llvm::toString(normalized.takeError()));

  mlir::scf::WhileOp loop;
  function.walk([&](mlir::scf::WhileOp candidate) { loop = candidate; });
  if (!loop)
    fail("dynamic byte-stride normalization removed the counted loop");
  for (mlir::Value init : loop.getInits())
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(init.getType()))
      fail("dynamic byte stride retained raw pointer induction");
  bool sawExactElementProjection = false;
  function.walk([&](mlir::arith::ShRSIOp shift) {
    auto amount = shift.getRhs().getDefiningOp<mlir::arith::ConstantOp>();
    auto integer = amount ? llvm::dyn_cast<mlir::IntegerAttr>(amount.getValue())
                          : mlir::IntegerAttr{};
    sawExactElementProjection |=
        shift.getIsExact() && integer && integer.getValue() == 2;
  });
  if (!sawExactElementProjection)
    fail("dynamic byte stride lost its exact element projection");
}

void preservesNarrowSourceRangeThroughStrideWidening() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n32:64-S128"
} {
  llvm.func @widened_complex_stride(%input: !llvm.ptr, %rows: i16) {
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %rows_i32 = arith.extui %rows : i16 to i32
    %stride_i32 = arith.shli %rows_i32, %c1_i32
        overflow<nsw, nuw> : i32
    %stride_i64 = arith.extui %stride_i32 nneg : i32 to i64
    %result:2 = scf.while (%cursor = %input, %remaining = %rows_i32)
        : (!llvm.ptr, i32) -> (!llvm.ptr, i32) {
      %value = llvm.load %cursor : !llvm.ptr -> f32
      %next_cursor = llvm.getelementptr inbounds %cursor[%stride_i64]
          : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
      %next_remaining = arith.addi %remaining, %c-1_i32 : i32
      %more = arith.cmpi ne, %next_remaining, %c0_i32 : i32
      scf.condition(%more) %next_cursor, %next_remaining : !llvm.ptr, i32
    } do {
    ^bb0(%cursor: !llvm.ptr, %remaining: i32):
      scf.yield %cursor, %remaining : !llvm.ptr, i32
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the widened stride-range fixture");
  auto function =
      module->lookupSymbol<mlir::LLVM::LLVMFuncOp>("widened_complex_stride");
  if (!function)
    fail("widened stride-range fixture omitted its function");

  auto normalized = loom::frontend::detail::materializeAddressIndexContract(
      *module, function.getOperation(), 64,
      [](mlir::Block *, mlir::Block *) { return llvm::Error::success(); });
  if (!normalized)
    fail(llvm::toString(normalized.takeError()));

  mlir::scf::WhileOp loop;
  function.walk([&](mlir::scf::WhileOp candidate) { loop = candidate; });
  if (!loop)
    fail("widened stride-range normalization removed the counted loop");
  for (mlir::Value init : loop.getInits())
    if (llvm::isa<mlir::LLVM::LLVMPointerType>(init.getType()))
      fail("widened stride-range retained raw pointer induction");
}

void normalizesWideCarrierWithNarrowTripCount() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n32:64-S128"
} {
  llvm.func @matrix_chunks(%base: !llvm.ptr, %rows: i16, %columns: i16) {
    %c0_i64 = arith.constant 0 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c32_i64 = arith.constant 32 : i64
    %rows_i64 = arith.extui %rows : i16 to i64
    %columns_i64 = arith.extui %columns : i16 to i64
    %elements = arith.muli %rows_i64, %columns_i64
        overflow<nsw, nuw> : i64
    %chunks = arith.shrui %elements, %c2_i64 : i64
    %empty = arith.cmpi eq, %chunks, %c0_i64 : i64
    scf.if %empty {
    } else {
      %result:2 = scf.while (%cursor = %base, %remaining = %chunks)
          : (!llvm.ptr, i64) -> (!llvm.ptr, i64) {
        %value = llvm.load %cursor : !llvm.ptr -> f64
        %next_cursor = llvm.getelementptr inbounds %cursor[%c32_i64]
            : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %next_remaining = arith.addi %remaining, %c-1_i64
            overflow<nsw> : i64
        %more = arith.cmpi ne, %next_remaining, %c0_i64 : i64
        scf.condition(%more) %next_cursor, %next_remaining : !llvm.ptr, i64
      } do {
      ^bb0(%cursor: !llvm.ptr, %remaining: i64):
        scf.yield %cursor, %remaining : !llvm.ptr, i64
      }
    }
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the narrow trip-count fixture");
  auto function = module->lookupSymbol<mlir::LLVM::LLVMFuncOp>("matrix_chunks");
  if (!function)
    fail("narrow trip-count fixture omitted its function");

  auto normalized = loom::frontend::detail::materializeAddressIndexContract(
      *module, function.getOperation(), 64,
      [](mlir::Block *, mlir::Block *) { return llvm::Error::success(); });
  if (!normalized)
    fail(llvm::toString(normalized.takeError()));

  function.walk([&](mlir::scf::WhileOp loop) {
    for (mlir::Value init : loop.getInits())
      if (llvm::isa<mlir::LLVM::LLVMPointerType>(init.getType()))
        fail("narrow trip-count proof retained raw pointer induction");
  });
}

void normalizesConditionallyExecutedPointerInduction() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::DLTIDialect,
                  mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module attributes {
  llvm.data_layout = "e-m:e-p:64:64-i64:64-n32:64-S128"
} {
  llvm.func @conditional_induction(%base: !llvm.ptr, %count: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c4_i64 = arith.constant 4 : i64
    %skip = arith.cmpi eq, %count, %c0_i32 : i32
    %selected = scf.if %skip -> (!llvm.ptr) {
      scf.yield %base : !llvm.ptr
    } else {
      %result:2 = scf.while (%cursor = %base, %remaining = %count)
          : (!llvm.ptr, i32) -> (!llvm.ptr, i32) {
        %value = llvm.load %cursor : !llvm.ptr -> i32
        %next_cursor = llvm.getelementptr inbounds %cursor[%c4_i64]
            : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %next_remaining = arith.addi %remaining, %c-1_i32 : i32
        %more = arith.cmpi ne, %next_remaining, %c0_i32 : i32
        scf.condition(%more) %next_cursor, %next_remaining
            : !llvm.ptr, i32
      } do {
      ^bb0(%cursor: !llvm.ptr, %remaining: i32):
        scf.yield %cursor, %remaining : !llvm.ptr, i32
      }
      scf.yield %result#0 : !llvm.ptr
    }
    %tail = llvm.load %selected : !llvm.ptr -> i32
    %same = scf.if %skip -> (!llvm.ptr) {
      scf.yield %base : !llvm.ptr
    } else {
      scf.yield %base : !llvm.ptr
    }
    %same_value = llvm.load %same : !llvm.ptr -> i32
    llvm.return
  }
}
)mlir",
                                                        &context);
  if (!module)
    fail("cannot parse the conditional pointer-induction fixture");
  auto function =
      module->lookupSymbol<mlir::LLVM::LLVMFuncOp>("conditional_induction");
  if (!function)
    fail("conditional pointer-induction fixture omitted its function");

  auto normalized = loom::frontend::detail::materializeAddressIndexContract(
      *module, function.getOperation(), 64,
      [](mlir::Block *, mlir::Block *) { return llvm::Error::success(); });
  if (!normalized)
    fail(llvm::toString(normalized.takeError()));

  function.walk([&](mlir::scf::WhileOp loop) {
    for (mlir::Value init : loop.getInits())
      if (llvm::isa<mlir::LLVM::LLVMPointerType>(init.getType()))
        fail("conditional normalization retained raw pointer induction");
  });
  function.walk([&](mlir::scf::IfOp select) {
    for (mlir::Type type : select.getResultTypes())
      if (llvm::isa<mlir::LLVM::LLVMPointerType>(type))
        fail("conditional normalization retained a selected capability");
  });
}

} // namespace

int main() {
  normalizesAsymmetricPointerInduction();
  normalizesInvariantDynamicByteStride();
  preservesNarrowSourceRangeThroughStrideWidening();
  normalizesWideCarrierWithNarrowTripCount();
  normalizesConditionallyExecutedPointerInduction();
  llvm::outs() << "structured address index narrowing anchor passed\n";
  return EXIT_SUCCESS;
}

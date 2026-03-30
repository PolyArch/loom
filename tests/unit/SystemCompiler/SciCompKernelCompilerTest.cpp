// Regression test for the function-specific SciComp KernelCompiler path.

#include "loom/SystemCompiler/KernelCompiler.h"

#include "loom/Dialect/Dataflow/DataflowDialect.h"
#include "loom/Dialect/Fabric/FabricDialect.h"

#include "circt/Dialect/Handshake/HandshakeDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace mlir;
using namespace loom::tapestry;

static void registerDialects(MLIRContext &ctx) {
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<cf::ControlFlowDialect>();
  ctx.getOrLoadDialect<DLTIDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<LLVM::LLVMDialect>();
  ctx.getOrLoadDialect<math::MathDialect>();
  ctx.getOrLoadDialect<memref::MemRefDialect>();
  ctx.getOrLoadDialect<scf::SCFDialect>();
  ctx.getOrLoadDialect<ub::UBDialect>();
  ctx.getOrLoadDialect<loom::dataflow::DataflowDialect>();
  ctx.getOrLoadDialect<loom::fabric::FabricDialect>();
  ctx.getOrLoadDialect<circt::handshake::HandshakeDialect>();
}

static std::filesystem::path writeStencilSource() {
  const auto path = std::filesystem::temp_directory_path() /
                    "loom-scicomp-kernel-compiler.c";
  std::ofstream os(path);
  os << R"c(
void stencil_5pt(const float *in, float *out, int rows, int cols,
                 int halo_w, float factor) {
  if (rows <= 0 || cols <= 0 || halo_w <= 0)
    return;

  int total_rows = rows + 2 * halo_w;
  int total_cols = cols + 2 * halo_w;

  for (int i = 0; i < total_rows * total_cols; ++i)
    out[i] = in[i];

  for (int row = halo_w; row < rows + halo_w; ++row) {
    for (int col = halo_w; col < cols + halo_w; ++col) {
      float center = in[row * total_cols + col];
      float sum = in[(row - 1) * total_cols + col] +
                  in[(row + 1) * total_cols + col] +
                  in[row * total_cols + (col - 1)] +
                  in[row * total_cols + (col + 1)] -
                  4.0f * center;
      out[row * total_cols + col] = sum * factor;
    }
  }
}

void stencil_9pt(const float *in, float *out, int rows, int cols, int halo_w,
                 float factor) {
  if (rows <= 0 || cols <= 0 || halo_w <= 0)
    return;

  int total_rows = rows + 2 * halo_w;
  int total_cols = cols + 2 * halo_w;

  for (int i = 0; i < total_rows * total_cols; ++i)
    out[i] = in[i];

  for (int row = halo_w; row < rows + halo_w; ++row) {
    for (int col = halo_w; col < cols + halo_w; ++col) {
      float center = in[row * total_cols + col];
      float orth = in[(row - 1) * total_cols + col] +
                   in[(row + 1) * total_cols + col] +
                   in[row * total_cols + (col - 1)] +
                   in[row * total_cols + (col + 1)];
      float diag = in[(row - 1) * total_cols + (col - 1)] +
                   in[(row - 1) * total_cols + (col + 1)] +
                   in[(row + 1) * total_cols + (col - 1)] +
                   in[(row + 1) * total_cols + (col + 1)];
      out[row * total_cols + col] = (orth + 0.5f * diag - 6.0f * center) *
                                    factor;
    }
  }
}

void stencil_5pt_unroll2(const float *in, float *out, int rows, int cols,
                         int halo_w, float factor) {
  if (rows <= 0 || cols <= 0 || halo_w <= 0)
    return;

  int total_rows = rows + 2 * halo_w;
  int total_cols = cols + 2 * halo_w;

  for (int i = 0; i < total_rows * total_cols; ++i)
    out[i] = in[i];

  for (int row = halo_w; row < rows + halo_w; ++row) {
    int col = halo_w;
    for (; col + 1 < cols + halo_w; col += 2) {
      for (int lane = 0; lane < 2; ++lane) {
        int c = col + lane;
        float center = in[row * total_cols + c];
        float sum = in[(row - 1) * total_cols + c] +
                    in[(row + 1) * total_cols + c] +
                    in[row * total_cols + (c - 1)] +
                    in[row * total_cols + (c + 1)] -
                    4.0f * center;
        out[row * total_cols + c] = sum * factor;
      }
    }
    for (; col < cols + halo_w; ++col) {
      float center = in[row * total_cols + col];
      float sum = in[(row - 1) * total_cols + col] +
                  in[(row + 1) * total_cols + col] +
                  in[row * total_cols + (col - 1)] +
                  in[row * total_cols + (col + 1)] -
                  4.0f * center;
      out[row * total_cols + col] = sum * factor;
    }
  }
}
)c";
  os.close();
  return path;
}

static bool checkKernel(MLIRContext &ctx, KernelCompiler &compiler,
                        const std::string &name) {
  bool sawError = false;
  std::string diagText;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error)
      sawError = true;
    diagText += diag.str();
    diagText.push_back('\n');
    return success();
  });

  KernelCompileResult result = compiler.compile(name);
  if (!result.success || !result.dfgModule) {
    std::cerr << "FAIL: compile failed for " << name << ": "
              << result.diagnostics << "\n";
    return false;
  }
  if (failed(verify(*result.dfgModule))) {
    std::cerr << "FAIL: lowered module does not verify for " << name << "\n";
    return false;
  }

  std::string printed;
  llvm::raw_string_ostream os(printed);
  result.dfgModule->print(os);
  os.flush();
  if (printed.find("handshake.func") == std::string::npos) {
    std::cerr << "FAIL: handshake.func missing for " << name << "\n";
    return false;
  }
  if (sawError) {
    std::cerr << "FAIL: emitted MLIR error diagnostics for " << name << "\n"
              << diagText;
    return false;
  }

  std::cout << "PASS: " << name << "\n";
  return true;
}

static bool testSciCompKernelCompiler() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect>();
  registry.insert<cf::ControlFlowDialect>();
  registry.insert<DLTIDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<LLVM::LLVMDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<ub::UBDialect>();
  registry.insert<loom::dataflow::DataflowDialect>();
  registry.insert<loom::fabric::FabricDialect>();
  registry.insert<circt::handshake::HandshakeDialect>();

  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  registerDialects(ctx);

  KernelCompiler compiler(ctx);
  std::filesystem::path sourcePath = writeStencilSource();
  if (!compiler.loadSource(sourcePath.string())) {
    std::cerr << "FAIL: KernelCompiler.loadSource failed\n";
    return false;
  }

  if (!checkKernel(ctx, compiler, "stencil_5pt"))
    return false;
  if (!checkKernel(ctx, compiler, "stencil_9pt"))
    return false;
  if (!checkKernel(ctx, compiler, "stencil_5pt_unroll2"))
    return false;

  std::cout << "PASS: testSciCompKernelCompiler\n";
  return true;
}

int main() {
  if (!testSciCompKernelCompiler())
    return 1;
  return 0;
}

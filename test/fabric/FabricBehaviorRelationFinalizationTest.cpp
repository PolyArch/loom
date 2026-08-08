#include "Common/ArtifactFinalizer.h"
#include "Dataflow/IR/DataflowDialect.h"
#include "Fabric/Artifact/FabricArtifact.h"
#include "Fabric/IR/FabricDialect.h"
#include "Fabric/IR/FabricOps.h"
#include "Fabric/IR/OperationResourceContract.h"
#include "Fabric/IR/ResourceContractRecord.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef test, const std::string &message) {
  llvm::errs() << test << ": " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void require(llvm::StringRef test, bool condition, llvm::StringRef message) {
  if (!condition)
    fail(test, message.str());
}

template <typename T> T take(llvm::StringRef test, llvm::Expected<T> value) {
  if (!value)
    fail(test, llvm::toString(value.takeError()));
  return std::move(*value);
}

template <typename T>
void expectError(llvm::StringRef test, llvm::Expected<T> value,
                 llvm::StringRef expected) {
  if (value)
    fail(test, "accepted an invalid Fabric behavior relation");
  const std::string message = llvm::toString(value.takeError());
  require(test, llvm::StringRef(message).contains(expected), message);
}

std::unique_ptr<mlir::MLIRContext> makeContext() {
  mlir::DialectRegistry registry;
  registry.insert<::dataflow::DataflowDialect, ::fabric::FabricDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  auto context = std::make_unique<mlir::MLIRContext>(
      registry, mlir::MLIRContext::Threading::DISABLED);
  context->loadAllAvailableDialects();
  return context;
}

void materializeOperationContracts(llvm::StringRef test, mlir::ModuleOp source,
                                   mlir::MLIRContext &context) {
  const std::vector<std::uint8_t> contract =
      take(test, ::fabric::encodeResourceContractRecord(
                     ::fabric::oneCycleElasticOperationResourceContract()));
  const std::vector<std::int8_t> signedContract(contract.begin(),
                                                contract.end());
  source.walk([&](::fabric::OpOp operation) {
    operation->setAttr(::fabric::kResourceContractRecordAttrName,
                       mlir::DenseI8ArrayAttr::get(&context, signedContract));
  });
}

::fabric::ModuleOp uniqueFabricModule(llvm::StringRef test,
                                      mlir::ModuleOp source) {
  ::fabric::ModuleOp root;
  source.walk([&](::fabric::ModuleOp candidate) {
    require(test, !root, "fixture contains multiple Fabric roots");
    root = candidate;
  });
  require(test, static_cast<bool>(root), "Fabric fixture has no root");
  return root;
}

void physicalBehaviorCollapseRemovesSemanticField(
    const std::filesystem::path &rootPath) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(rootPath);
  loom::ArtifactStore store(rootPath.string());
  std::unique_ptr<mlir::MLIRContext> context = makeContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @narrow_signed_compare(
          %a: !fabric.bits<8>, %b: !fabric.bits<8>)
          -> !fabric.bits<8> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<8>, %pb = %b : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<8>, %fb = %pb : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value = fabric.op [@arith.cmpi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerCompareMinMax>,
               hw_params = {
                 integer_widths = [8 : i32, 32 : i32],
                 predicates = ["slt"]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value : !fabric.bits<8>
          }
        }
        fabric.yield %pe : !fabric.bits<8>
      }
    }
  )mlir",
                                                        context.get());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");

  materializeOperationContracts(test, *source, *context);
  ::fabric::ModuleOp root = uniqueFabricModule(test, *source);

  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(root, store));
  require(test, finalized.view().fuTemplates().size() == 1,
          "narrow compare did not publish one FU template");
  const auto capabilities = finalized.view().resolvedFabricOpCapabilities(
      finalized.view().fuTemplates().front());
  require(test, capabilities.size() == 1,
          "narrow compare did not publish one operation capability");
  require(test, capabilities.front().configurationFieldSchema.empty(),
          "unreachable wide behavior created a semantic field");
}

void physicallyUnreachableBehaviorFailsFinalization(
    const std::filesystem::path &rootPath) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(rootPath);
  loom::ArtifactStore store(rootPath.string());
  std::unique_ptr<mlir::MLIRContext> context = makeContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @unreachable_add(
          %a: !fabric.bits<8>, %b: !fabric.bits<8>)
          -> !fabric.bits<8> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<8>, %pb = %b : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<8>, %fb = %pb : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value = fabric.op [@arith.addi] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<ScalarIntegerAddSub>,
               hw_params = {integer_widths = [32 : i32]}}
              : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value : !fabric.bits<8>
          }
        }
        fabric.yield %pe : !fabric.bits<8>
      }
    }
  )mlir",
                                                        context.get());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  materializeOperationContracts(test, *source, *context);

  expectError(test,
              loom::fabric::finalizeFabricRoot(
                  uniqueFabricModule(test, *source), store),
              "physically reachable behavior");
}

void physicallyUnreachableDirectBehaviorFailsFinalization(
    const std::filesystem::path &rootPath) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(rootPath);
  loom::ArtifactStore store(rootPath.string());
  std::unique_ptr<mlir::MLIRContext> context = makeContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @unreachable_shuffle(%a: !fabric.bits<8>)
          -> !fabric.bits<8> {
        %pe = fabric.pe [spatial] (%pa = %a : !fabric.bits<8>)
            -> !fabric.bits<8> {
          %fu = fabric.fu (%fa = %pa : !fabric.bits<8>)
              -> !fabric.bits<8> {
            %value = fabric.op [@vector.shuffle] (%fa)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorShuffle>,
               hw_params = {
                 integer_element_widths = [8 : i32],
                 float_element_formats = [],
                 max_operand_payload_bits = 8 : i32,
                 max_result_payload_bits = 8 : i32,
                 max_block_payload_bits = 8 : i32,
                 max_source_blocks = 2 : i32,
                 max_result_blocks = 1 : i32}}
              : (!fabric.bits<8>) -> !fabric.bits<8>
            fabric.yield %value : !fabric.bits<8>
          }
        }
        fabric.yield %pe : !fabric.bits<8>
      }
    }
  )mlir",
                                                        context.get());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  materializeOperationContracts(test, *source, *context);

  expectError(test,
              loom::fabric::finalizeFabricRoot(
                  uniqueFabricModule(test, *source), store),
              "physically reachable behavior");
}

void zeroBitSlicePublishesNoSemanticField(
    const std::filesystem::path &rootPath) {
  const llvm::StringRef test = __func__;
  std::filesystem::create_directories(rootPath);
  loom::ArtifactStore store(rootPath.string());
  std::unique_ptr<mlir::MLIRContext> context = makeContext();
  auto source = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      fabric.module @singleton_slice(
          %a: !fabric.bits<1>, %b: !fabric.bits<1>)
          -> !fabric.bits<1> {
        %pe = fabric.pe [spatial]
            (%pa = %a : !fabric.bits<1>, %pb = %b : !fabric.bits<1>)
            -> !fabric.bits<1> {
          %fu = fabric.fu
              (%fa = %pa : !fabric.bits<1>, %fb = %pb : !fabric.bits<1>)
              -> !fabric.bits<1> {
            %value = fabric.op [@vector.extract] (%fa, %fb)
              {implementation_family =
                 #fabric.implementation_family<FixedVectorSliceAlignMerge>,
               hw_params = {
                 integer_element_widths = [1 : i32],
                 float_element_formats = [],
                 max_container_payload_bits = 1 : i32,
                 max_slice_payload_bits = 1 : i32,
                 max_dynamic_position_rank = 0 : i32,
                 resolved_index_widths = []}}
              : (!fabric.bits<1>, !fabric.bits<1>) -> !fabric.bits<1>
            fabric.yield %value : !fabric.bits<1>
          }
        }
        fabric.yield %pe : !fabric.bits<1>
      }
    }
  )mlir",
                                                        context.get());
  require(test, static_cast<bool>(source), "could not parse Fabric fixture");
  materializeOperationContracts(test, *source, *context);
  loom::fabric::FinalizedFabricRoot finalized =
      take(test, loom::fabric::finalizeFabricRoot(
                     uniqueFabricModule(test, *source), store));
  const auto capabilities = finalized.view().resolvedFabricOpCapabilities(
      finalized.view().fuTemplates().front());
  require(test,
          capabilities.size() == 1 &&
              capabilities.front().configurationFieldSchema.empty(),
          "zero-bit singleton slice published a semantic field");
}

void everyIntegerSchemaMustRemainReachable(
    const std::filesystem::path &rootPath) {
  const llvm::StringRef test = __func__;
  const auto check = [&](llvm::StringRef name, llvm::StringRef family,
                         llvm::StringRef params) {
    const std::filesystem::path directory = rootPath / name.str();
    std::filesystem::create_directories(directory);
    loom::ArtifactStore store(directory.string());
    std::unique_ptr<mlir::MLIRContext> context = makeContext();
    const std::string text =
        ("module { fabric.module @" + name +
         "(%a: !fabric.bits<8>, %b: !fabric.bits<8>) -> !fabric.bits<8> { "
         "%pe = fabric.pe [spatial] (%pa = %a : !fabric.bits<8>, "
         "%pb = %b : !fabric.bits<8>) -> !fabric.bits<8> { "
         "%fu = fabric.fu (%fa = %pa : !fabric.bits<8>, "
         "%fb = %pb : !fabric.bits<8>) -> !fabric.bits<8> { "
         "%predicate = fabric.op [@arith.cmpi, @arith.minsi] (%fa, %fb) "
         "{implementation_family = #fabric.implementation_family<" +
         family + ">, hw_params = " + params +
         "} : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<1> "
         "%value = fabric.op [@arith.extui] (%predicate) "
         "{implementation_family = "
         "#fabric.implementation_family<ScalarIntegerCast>, "
         "hw_params = {width_pairs = [[1 : i32, 8 : i32]], "
         "resolved_index_widths = []}} "
         ": (!fabric.bits<1>) -> !fabric.bits<8> "
         "fabric.yield %value : !fabric.bits<8> } } "
         "fabric.yield %pe : !fabric.bits<8> } }")
            .str();
    auto source = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    require(test, static_cast<bool>(source), "could not parse Fabric fixture");
    materializeOperationContracts(test, *source, *context);
    expectError(test,
                loom::fabric::finalizeFabricRoot(
                    uniqueFabricModule(test, *source), store),
                "enabled schema");
  };

  check("scalar_integer", "ScalarIntegerCompareMinMax",
        "{integer_widths = [8 : i32], predicates = [\"slt\"]}");
  check("vector_integer", "FixedVectorIntegerCompareMinMax",
        "{element_widths = [8 : i32], max_payload_bits = 8 : i32, "
        "predicates = [\"slt\"]}");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("main", "expected one artifact directory argument");
  physicalBehaviorCollapseRemovesSemanticField(argv[1]);
  physicallyUnreachableBehaviorFailsFinalization(
      std::filesystem::path(argv[1]) / "unreachable");
  physicallyUnreachableDirectBehaviorFailsFinalization(
      std::filesystem::path(argv[1]) / "unreachable_direct");
  zeroBitSlicePublishesNoSemanticField(std::filesystem::path(argv[1]) /
                                       "zero_bit_slice");
  everyIntegerSchemaMustRemainReachable(std::filesystem::path(argv[1]) /
                                        "integer_schema");
  return EXIT_SUCCESS;
}

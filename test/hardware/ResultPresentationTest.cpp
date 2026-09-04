#include "Fabric/IR/ResultPresentation.h"
#include "Arbitration.h"
#include "Hardware/RTL/CommonSkeleton.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(llvm::StringRef message) {
  llvm::errs() << "result presentation: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

void checkPeriodicFairness() {
  // Independent context services expose different periodic request sets.
  // Ordinary valid-request round robin can phase-lock and starve a context.
  // Every eligible position must be presented within the sum of its owning
  // service periods, regardless of initial dispatch/presentation alignment.
  for (unsigned sizesCode = 0; sizesCode != 27; ++sizesCode) {
    unsigned code = sizesCode;
    std::vector<fabric::ResultPresentationRequest> requests(3);
    unsigned positionCount = 0;
    unsigned bound = 0;
    unsigned phaseCount = 1;
    for (auto &request : requests) {
      unsigned count = code % 3 + 1;
      code /= 3;
      request.laneDestinations = {llvm::APInt(3, 1)};
      request.evaluations.assign(count, {true, false});
      positionCount += count;
      bound += count * count;
      phaseCount *= count;
    }
    for (unsigned phases = 0; phases != phaseCount; ++phases)
      for (unsigned initial = 0; initial != positionCount; ++initial) {
        unsigned cursor = initial;
        llvm::SmallBitVector seen(positionCount);
        for (unsigned cycle = 0; cycle != bound; ++cycle) {
          unsigned phaseCode = phases;
          unsigned position = 0;
          std::vector<unsigned> current;
          for (auto &request : requests) {
            unsigned count = request.evaluations.size();
            unsigned context = (cycle + phaseCode % count) % count;
            phaseCode /= count;
            for (unsigned c = 0; c != count; ++c)
              request.evaluations[c].evaluated = c == context;
            current.push_back(position + context);
            position += count;
          }
          auto result = fabric::selectResultPresentation(requests, cursor);
          for (unsigned requester = 0; requester != requests.size();
               ++requester)
            if (result.selected[requester])
              seen.set(current[requester]);
          cursor = result.nextCursor;
        }
        if (seen.count() != positionCount)
          fail("periodic context evaluation starved a presentation position");
      }
  }
}

void checkFocusAndMultiplicity() {
  std::vector<fabric::ResultPresentationRequest> requests(3);
  requests[0] = {{llvm::APInt(3, 1), llvm::APInt(3, 1)},
                 {{false, false}, {true, false}}};
  requests[1] = {{llvm::APInt(3, 1)}, {{true, true}}};
  requests[2] = {{llvm::APInt(3, 2)}, {{true, true}}};
  auto result = fabric::selectResultPresentation(requests, 0);
  if (result.selected[0] || !result.selected[1] || !result.selected[2] ||
      result.nextCursor != 0)
    fail("duplicate lanes or opportunistic offers changed the waiting focus");
  requests[0].laneDestinations = {llvm::APInt(3, 0)};
  requests[0].evaluations[1].evaluated = true;
  result = fabric::selectResultPresentation(requests, 0);
  if (result.nextCursor != 2)
    fail(
        "an eligible empty evaluation did not advance past ineligible context");
  for (auto &request : requests)
    for (auto &evaluation : request.evaluations)
      evaluation.eligible = false;
  result = fabric::selectResultPresentation(requests, 2);
  if (result.selected.any() || result.nextCursor != 2)
    fail("an empty eligible domain invented a presentation position");
}

std::string buildRtl() {
  using namespace loom::hardware::rtl;
  mlir::MLIRContext context;
  context.loadDialect<circt::comb::CombDialect, circt::hw::HWDialect,
                      circt::seq::SeqDialect, circt::sv::SVDialect>();
  mlir::OpBuilder builder(&context);
  const auto location = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(location);
  builder.setInsertionPointToStart(module.getBody());
  llvm::SmallVector<circt::hw::PortInfo> inputs;
  llvm::SmallVector<circt::hw::PortInfo> outputs;
  for (llvm::StringRef name : {"clock", "reset"})
    inputs.push_back(
        {{builder.getStringAttr(name),
          name == "clock" ? mlir::Type(circt::seq::ClockType::get(&context))
                          : mlir::Type(builder.getI1Type()),
          circt::hw::ModulePort::Direction::Input}});
  for (unsigned requester = 0; requester != 3; ++requester)
    inputs.push_back(
        {{builder.getStringAttr("claims_" + std::to_string(requester)),
          builder.getIntegerType(6), circt::hw::ModulePort::Direction::Input}});
  for (llvm::StringRef name : {"eligible", "evaluated"})
    inputs.push_back({{builder.getStringAttr(name), builder.getIntegerType(4),
                       circt::hw::ModulePort::Direction::Input}});
  outputs.push_back(
      {{builder.getStringAttr("selected"), builder.getIntegerType(3),
        circt::hw::ModulePort::Direction::Output}});
  circt::hw::HWModuleOp::create(
      builder, location, builder.getStringAttr("result_presentation"),
      circt::hw::ModulePortInfo(inputs, outputs),
      [&](mlir::OpBuilder &bodyBuilder,
          circt::hw::HWModulePortAccessor &accessor) {
        circt::BackedgeBuilder backedges(bodyBuilder, location);
        std::vector<llvm::SmallVector<mlir::Value>> claims(3), eligible(3),
            evaluated(3);
        for (unsigned requester = 0; requester != 3; ++requester) {
          auto packed =
              accessor.getInput("claims_" + std::to_string(requester));
          for (unsigned lane = 0; lane != 2; ++lane)
            claims[requester].push_back(circt::comb::ExtractOp::create(
                bodyBuilder, location, packed, lane * 3, 3));
        }
        unsigned position = 0;
        for (unsigned requester = 0; requester != 3; ++requester)
          for (unsigned context = 0; context != (requester == 0 ? 2 : 1);
               ++context, ++position) {
            eligible[requester].push_back(circt::comb::ExtractOp::create(
                bodyBuilder, location, accessor.getInput("eligible"), position,
                1));
            evaluated[requester].push_back(circt::comb::ExtractOp::create(
                bodyBuilder, location, accessor.getInput("evaluated"), position,
                1));
          }
        auto priority = hierarchy::makeResultPresentationPriority(
            bodyBuilder, location, backedges, eligible, evaluated,
            accessor.getInput("clock"), accessor.getInput("reset"),
            "presentation_cursor", hierarchy::ClockResetPlan{});
        auto selected = hierarchy::selectResultPresentation(
            bodyBuilder, location, claims, priority);
        accessor.setOutput(
            "selected", hierarchy::packBits(bodyBuilder, location, selected));
      });
  auto rtl = lowerAndExportSpecializedSystemVerilog(module);
  if (!rtl)
    fail(llvm::toString(rtl.takeError()));
  return std::move(*rtl);
}

void writeTestbench(std::ostream &stream, std::ostream &vectors) {
  stream << R"(module result_presentation_testbench;
  reg clock = 0;
  reg reset = 1;
  reg [5:0] claims_0 = 0, claims_1 = 0, claims_2 = 0;
  reg [3:0] eligible = 0, evaluated = 0;
  wire [2:0] selected;
  result_presentation dut(.*);
  task tick(input [5:0] a, b, c, input [3:0] e, v,
            input [2:0] expected);
    claims_0 = a; claims_1 = b; claims_2 = c;
    eligible = e; evaluated = v;
    #1;
    if (selected !== expected)
      $fatal(1, "atomic presentation disagrees with Fabric reference");
    clock = 1; #1; clock = 0; #1;
  endtask
  initial begin
    #1; clock = 1; #1; clock = 0; #1; reset = 0;
)";
  unsigned cursor = 0;
  unsigned cycle = 0;
  const auto tick = [&](unsigned a, unsigned b, unsigned c, unsigned eligible,
                        unsigned evaluated) {
    std::vector<fabric::ResultPresentationRequest> requests(3);
    unsigned position = 0;
    unsigned values[] = {a, b, c};
    for (unsigned requester = 0; requester != 3; ++requester) {
      requests[requester].laneDestinations = {
          llvm::APInt(3, values[requester] & 7),
          llvm::APInt(3, (values[requester] >> 3) & 7)};
      for (unsigned context = 0; context != (requester == 0 ? 2 : 1);
           ++context, ++position)
        requests[requester].evaluations.push_back(
            {bool(eligible & (1 << position)),
             bool(evaluated & (1 << position))});
    }
    auto result = fabric::selectResultPresentation(requests, cursor);
    unsigned expected = 0;
    for (unsigned requester = 0; requester != 3; ++requester)
      expected |= unsigned(result.selected[requester]) << requester;
    vectors << std::hex << a << ' ' << b << ' ' << c << ' ' << eligible << ' '
            << evaluated << ' ' << expected << '\n';
    cursor = result.nextCursor;
    ++cycle;
  };
  // Every three-destination shape, both distinct lanes and duplicate lane
  // claims. Context A alternates while its peers are continuously exposed.
  for (unsigned shape = 0; shape != 512; ++shape)
    for (unsigned duplicate = 0; duplicate != 2; ++duplicate) {
      unsigned claims[3];
      for (unsigned requester = 0; requester != 3; ++requester) {
        unsigned mask = (shape >> (3 * requester)) & 7;
        unsigned lane = mask & -mask;
        claims[requester] =
            duplicate ? mask | (lane << 3) : (mask ^ lane) | (lane << 3);
      }
      for (unsigned pass = 0; pass != 4; ++pass)
        tick(claims[0], claims[1], claims[2], 15, 12 | (1 << (cycle % 2)));
    }
  // Ineligible positions, evaluation without an offer, and opportunistic
  // offers while the focused context has not yet been evaluated.
  for (unsigned eligible = 0; eligible != 16; ++eligible)
    for (unsigned evaluated = 0; evaluated != 16; ++evaluated) {
      tick(0, 3, 4, eligible, evaluated);
      tick(1, 3, 4, eligible, evaluated);
    }
  stream << R"(    begin
      integer vectors, fields;
      reg [5:0] a, b, c;
      reg [3:0] e, v;
      reg [2:0] expected;
      vectors = $fopen("result_presentation_cases.txt", "r");
      if (vectors == 0) $fatal(1, "cannot open presentation cases");
      while (!$feof(vectors)) begin
        fields = $fscanf(vectors, "%h %h %h %h %h %h\n",
                         a, b, c, e, v, expected);
        if (fields != 6) $fatal(1, "invalid presentation case");
        tick(a, b, c, e, v, expected);
      end
      $fclose(vectors);
    end
    $finish;
  end
endmodule
)";
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2)
    fail("expected an artifact directory");
  checkPeriodicFairness();
  checkFocusAndMultiplicity();
  const std::filesystem::path root(argv[1]);
  std::filesystem::create_directories(root);
  std::ofstream(root / "result_presentation.sv") << buildRtl();
  std::ofstream testbench(root / "result_presentation_testbench.sv");
  std::ofstream vectors(root / "result_presentation_cases.txt");
  writeTestbench(testbench, vectors);
}

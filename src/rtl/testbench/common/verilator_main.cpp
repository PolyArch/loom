// Minimal C++ main harness for Verilator simulation.
// Instantiates the top-level module, advances Verilated time each tick,
// and runs until the testbench signals completion via $finish.
//
// The SV testbench generates its own clock internally (tb_clk_rst_gen),
// so the C++ harness only needs to call eval() and advance the
// simulation time so that delay-based constructs (#N) work correctly.

#include <cstdlib>
#include <memory>

#include "verilated.h"

// Verilator generates the top module header based on --top-module.
// The build system names it Vtb_module_wrapper by default.
#include "Vtb_module_wrapper.h"

// Verilator 5+ requires a VerilatedContext; for older versions the
// static Verilated:: API is used instead.  We support both.

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    auto top = std::make_unique<Vtb_module_wrapper>();

    // Main simulation loop.
    // Each iteration: evaluate combinational logic, then advance
    // Verilated time by 1 time-unit so that SV delay statements
    // (#(HALF_PERIOD) in tb_clk_rst_gen) can make progress.
    while (!Verilated::gotFinish()) {
        top->eval();
        Verilated::timeInc(1);
    }

    top->final();
    return 0;
}

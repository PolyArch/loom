// Minimal C++ main harness for Verilator simulation.
// Instantiates the top-level module, drives the clock, and runs until
// the testbench signals completion via $finish.

#include <cstdlib>
#include <memory>

#include "verilated.h"

// Verilator generates the top module header based on --top-module.
// The build system names it Vtb_module_wrapper by default.
#include "Vtb_module_wrapper.h"

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    Verilated::traceEverOn(true);

    auto top = std::make_unique<Vtb_module_wrapper>();

    while (!Verilated::gotFinish()) {
        top->eval();
    }

    top->final();
    return 0;
}
